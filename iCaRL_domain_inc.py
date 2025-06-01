import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import numpy as np
from sklearn.metrics import accuracy_score
import random
import argparse

# Argument parser
parser = argparse.ArgumentParser(description="Continual Learning with Prototypes")

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser.add_argument("--num_tasks", type=int, default=20, help="Number of continual learning tasks")
parser.add_argument("--classes_per_task", type=int, default=5, help="Number of classes per task")
parser.add_argument("--train_features_path", type=str, required=True, help="Path to train features file (.npy)")
parser.add_argument("--train_labels_path", type=str, required=True, help="Path to train labels file (.npy)")
parser.add_argument("--test_features_path", type=str, required=True, help="Path to test features file (.npy)")
parser.add_argument("--test_labels_path", type=str, required=True, help="Path to test labels file (.npy)")

args = parser.parse_args()


# Parameters
num_tasks = args.num_tasks
classes_per_task = args.classes_per_task
batch_size = 64
num_epochs = 5
learning_rate = 0.0001
exemplar_size_per_class = 31  # Fixed exemplar size per class
temperature = 2.0  # Temperature for distillation loss

train_features_path = args.train_features_path
train_labels_path = args.train_labels_path
test_features_path = args.test_features_path
test_labels_path = args.test_labels_path

# Custom Dataset for pre-extracted features
class FeatureDataset(torch.utils.data.Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features).float()
        self.labels = torch.tensor(labels).long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


train_features_np = np.load(train_features_path)
train_labels_np = np.load(train_labels_path)
test_features_np = np.load(test_features_path)
test_labels_np = np.load(test_labels_path)

train_feature_dataset = FeatureDataset(train_features_np, train_labels_np)
test_feature_dataset = FeatureDataset(test_features_np, test_labels_np)

# Function to split dataset into tasks
def split_dataset(dataset, num_tasks):
    class_indices = [[] for _ in range(num_tasks * classes_per_task)]
    for idx, (_, label) in enumerate(dataset):
        class_indices[label].append(idx)
    tasks = []
    for task_id in range(num_tasks):
        task_classes = list(range(task_id * classes_per_task, (task_id + 1) * classes_per_task))
        task_indices = []
        for cls in task_classes:
            task_indices.extend(class_indices[cls])
        tasks.append(Subset(dataset, task_indices))
    return tasks

# Split train and test datasets into tasks
train_tasks = split_dataset(train_feature_dataset, num_tasks)
test_tasks = split_dataset(test_feature_dataset, num_tasks)

# Define the Linear Model
class LinearModel(nn.Module):
    def __init__(self, input_dim=1024, output_dim=512):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

model = LinearModel(input_dim=1024, output_dim=512).to(device)

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

class iCaRL:
    def __init__(self, model, exemplar_size_per_class=31):
        self.model = model
        self.exemplar_size_per_class = exemplar_size_per_class  # Fixed samples per class
        self.exemplar_memory = {}  # Dictionary to store exemplars for each class
        self.seen_classes = []  # List of classes seen so far
        self.previous_logits = None  # Initialize to None for distillation loss

    def classify(self, features):
        features = F.normalize(features, dim=1)

        # Ensure exemplar means are computed correctly
        exemplar_means = torch.stack([
            F.normalize(torch.mean(torch.stack(self.exemplar_memory[c]), dim=0), dim=0)
            for c in self.seen_classes
        ]).to(device)

        # Cosine similarity-based classification
        similarities = torch.mm(features, exemplar_means.T)
        preds = torch.argmax(similarities, dim=1)
        return torch.tensor(self.seen_classes, device=device)[preds]

    def update_exemplar_memory(self, features, labels):
        class_ids = torch.unique(labels).tolist()
        for class_id in class_ids:
            if class_id not in self.seen_classes:
                self.seen_classes.append(class_id)

            # Extract features for the current class
            indices = (labels == class_id).nonzero(as_tuple=True)[0]
            class_features = features[indices]

            # Handle case when fewer samples than exemplar size exist
            if class_features.size(0) <= self.exemplar_size_per_class:
                self.exemplar_memory[class_id] = [f for f in class_features]
                continue

            # Compute class mean and select exemplars
            class_mean = torch.mean(class_features, dim=0)
            exemplars = []
            for _ in range(self.exemplar_size_per_class):
                distances = torch.norm(class_features - class_mean, dim=1)
                min_idx = torch.argmin(distances)

                exemplars.append(class_features[min_idx])
                class_features = torch.cat((class_features[:min_idx], class_features[min_idx + 1:]), dim=0)

            self.exemplar_memory[class_id] = exemplars

    def distillation_loss(self, current_logits, temperature):
        if self.previous_logits is None:
            # No previous logits available; return zero loss
            return torch.tensor(0.0, device=device)
        
        previous_logits = self.previous_logits[:, :len(self.seen_classes)]  

        num_exemplars = previous_logits.size(0)
        batch_size = current_logits.size(0)
        repeated_old_logits = previous_logits.repeat(batch_size // num_exemplars + 1, 1)[:batch_size] 
        
        # Compute softmax distributions
        old_soft = F.softmax(repeated_old_logits / temperature, dim=1)
        current_soft = F.log_softmax(current_logits / temperature, dim=1)

        # Compute distillation loss
        return F.kl_div(current_soft, old_soft, reduction='batchmean') * (temperature ** 2)


# Save logits for distillation after training
def save_logits(model, icarl_agent):
    if not icarl_agent.exemplar_memory:
        return None

    all_logits = []
    for exemplars in icarl_agent.exemplar_memory.values():
        if len(exemplars) > 0:
            exemplars_tensor = torch.stack(exemplars).to(device)
            logits = model(exemplars_tensor).detach()
            all_logits.append(logits)

    if all_logits:
        return torch.cat(all_logits)
    return None  

# Initialize iCaRL
icarl_agent = iCaRL(model, exemplar_size_per_class=exemplar_size_per_class)


def main():
    task_accuracies = {t: [] for t in range(num_tasks)}

    for task_id in range(num_tasks):
        print(f"\n=== Training on Task {task_id + 1}/{num_tasks} ===")

        # Data loaders for the current task
        train_loader = DataLoader(train_tasks[task_id], batch_size=batch_size, shuffle=True, num_workers=2)
        test_loader = DataLoader(test_tasks[task_id], batch_size=batch_size, shuffle=False, num_workers=2)

        current_classes = list(range(task_id * classes_per_task, (task_id + 1) * classes_per_task))
        for cls in current_classes:
            if cls not in icarl_agent.seen_classes:
                icarl_agent.seen_classes.append(cls)

        # Training for the current task
        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0.0

            for batch_features, batch_labels in train_loader:
                batch_features = batch_features.to(device)
                batch_labels = batch_labels.to(device)

                # Forward pass
                logits = model(batch_features)

                # Binary cross-entropy loss for all seen classes
                binary_targets = torch.zeros(batch_labels.size(0), len(icarl_agent.seen_classes)).to(device)
                for i, label in enumerate(batch_labels):
                    binary_targets[i, icarl_agent.seen_classes.index(label.item() % 10)] = 1

                classification_loss = F.binary_cross_entropy_with_logits(logits[:, :len(icarl_agent.seen_classes)], binary_targets)

                # Distillation loss
                dist_loss = icarl_agent.distillation_loss(logits[:, :len(icarl_agent.seen_classes)], temperature=2.0)

                # Total loss
                loss = classification_loss + dist_loss

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            avg_epoch_loss = epoch_loss / len(train_loader)
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_epoch_loss:.4f}")

        # Save logits for distillation
        new_logits = save_logits(model, icarl_agent)
        if new_logits is not None:
            icarl_agent.previous_logits = new_logits

        # Update exemplar memory
        with torch.no_grad():
            for batch_features, batch_labels in train_loader:
                icarl_agent.update_exemplar_memory(batch_features, batch_labels)

        # Evaluate on all tasks seen so far
        for eval_task_id in range(task_id + 1):
            eval_loader = DataLoader(test_tasks[eval_task_id], batch_size=batch_size, shuffle=False, num_workers=2)
            all_predictions = []
            all_labels = []

            with torch.no_grad():
                for eval_features, eval_labels in eval_loader:
                    eval_features = eval_features.to(device)
                    preds = icarl_agent.classify(eval_features)

                    preds_mod = preds.cpu().numpy() % classes_per_task
                    labels_mod = eval_labels.numpy() % classes_per_task

                    all_predictions.extend(preds_mod)
                    all_labels.extend(labels_mod)

            accuracy = accuracy_score(all_labels, all_predictions)
            task_accuracies[eval_task_id].append(accuracy * 100)
            print(f"  Test Accuracy on Task {eval_task_id + 1} (mod-10): {accuracy * 100:.2f}%")

        # Average accuracy
        avg_acc = np.mean([task_accuracies[t][-1] for t in range(task_id + 1)])
        print(f"Average Accuracy on Tasks [1..{task_id+1}]: {avg_acc:.2f}%")


if __name__ == "__main__":
    main()
