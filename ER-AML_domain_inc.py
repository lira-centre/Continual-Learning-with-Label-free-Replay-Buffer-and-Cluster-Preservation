import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
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

# Parse arguments
args = parser.parse_args()


# Parameters
num_tasks = args.num_tasks
classes_per_task = args.classes_per_task
batch_size = 64
num_epochs = 5
learning_rate = 0.0001
M = 31  # Experience replay buffer size per class
temperature = 0.07

train_features_path = args.train_features_path
train_labels_path = args.train_labels_path
test_features_path = args.test_features_path
test_labels_path = args.test_labels_path

# Custom Dataset for pre-extracted features
class FeatureDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features).float()
        self.labels = torch.tensor(labels).long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# Load pre-saved features and labels
train_features_np = np.load(train_features_path)
train_labels_np = np.load(train_labels_path)
test_features_np = np.load(test_features_path)
test_labels_np = np.load(test_labels_path)

# Create FeatureDataset instances
train_feature_dataset = FeatureDataset(train_features_np, train_labels_np)
test_feature_dataset = FeatureDataset(test_features_np, test_labels_np)

# Function to split dataset into tasks
def split_dataset(dataset, num_tasks, classes_per_task):
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
train_tasks = split_dataset(train_feature_dataset, num_tasks, classes_per_task)
test_tasks = split_dataset(test_feature_dataset, num_tasks, classes_per_task)

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

class Buffer(nn.Module):
    def __init__(self, args, input_size=None):
        super().__init__()
        self.args = args

        if input_size is None:
            input_size = args.input_size

        buffer_size = args.mem_size

        self.bx = torch.FloatTensor(buffer_size, *input_size).fill_(0).to(device)
        self.by = torch.LongTensor(buffer_size).fill_(0).to(device)
        self.bt = torch.LongTensor(buffer_size).fill_(0).to(device)
        self.current_index = 0
        self.n_seen_so_far = 0

    def add_reservoir(self, x, y, t):
        n_elem = x.size(0)
        place_left = max(0, self.bx.size(0) - self.current_index)

        if place_left > 0:
            offset = min(place_left, n_elem)
            self.bx[self.current_index: self.current_index + offset].data.copy_(x[:offset])
            self.by[self.current_index: self.current_index + offset].data.copy_(y[:offset])
            self.bt[self.current_index: self.current_index + offset].fill_(t)
            self.current_index += offset
            self.n_seen_so_far += offset

        if n_elem > place_left:
            indices = torch.randint(0, self.n_seen_so_far, (n_elem - place_left,))
            valid_indices = indices < self.bx.size(0)
            indices = indices[valid_indices]
            self.bx[indices] = x[place_left:][valid_indices]
            self.by[indices] = y[place_left:][valid_indices]
            self.bt[indices] = t

    def sample(self, batch_size):
        indices = torch.randperm(self.current_index)[:batch_size]
        return self.bx[indices], self.by[indices], self.bt[indices]

    def fetch_pos_neg_samples(self, label, task, idx, data=None, task_free=True):
        task = torch.tensor([task] * label.size(0), device=device).view(-1, 1) 
        
        if self.current_index == 0:
            empty_tensor = torch.zeros((label.size(0), self.bx.size(1)), device=device)
            return empty_tensor, empty_tensor, None

        buffer_task = self.bt[:self.current_index].view(1, -1)  
        buffer_label = self.by[:self.current_index].view(1, -1)  
        
        # Compare labels and tasks
        same_label = label.view(-1, 1) == buffer_label 
        same_task = task.view(-1, 1) == buffer_task  
        
        # Identify valid positive and negative indices
        valid_pos = same_label & ~same_task  
        valid_neg = ~same_label  
        
        if valid_pos.sum().item() == 0:
            pos_idx = torch.randint(0, self.current_index, (label.size(0),), device=device)
        else:
            pos_idx = torch.multinomial(valid_pos.float(), 1, replacement=True).squeeze(1)  
        
        if valid_neg.sum().item() == 0:
            neg_idx = torch.randint(0, self.current_index, (label.size(0),), device=device)
        else:
            neg_idx = torch.multinomial(valid_neg.float(), 1, replacement=True).squeeze(1) 
        
        return self.bx[pos_idx], self.bx[neg_idx], None





# Implementation of ER_AML
class ER_AML(nn.Module):
    def __init__(self, model, buffer, args):
        super(ER_AML, self).__init__()
        self.model = model
        self.buffer = buffer
        self.args = args
        self.temperature = args.supcon_temperature

    def observe(self, inc_x, inc_y, inc_t, inc_idx, rehearse=False):
        # Forward pass for incoming data
        logits_inc = self.model(inc_x)
        loss = F.cross_entropy(logits_inc, inc_y)

        if rehearse:
            # Fetch positive and negative pairs from the buffer
            pos_x, neg_x, _ = self.buffer.fetch_pos_neg_samples(inc_y, inc_t, inc_idx)

            # Normalize embeddings
            hidden = F.normalize(self.model(inc_x), dim=1)
            pos_hidden = F.normalize(self.model(pos_x), dim=1)
            neg_hidden = F.normalize(self.model(neg_x), dim=1)

            # Contrastive loss: using positive and negative pairs
            supcon_loss = self.compute_supcon_loss(hidden, pos_hidden, neg_hidden, inc_y)

            loss += supcon_loss

        return loss, 0.0 

    def compute_supcon_loss(self, hidden, pos_hidden, neg_hidden, labels):
        labels = labels.contiguous().view(-1, 1)  
        device = hidden.device

        # Combine positive and negative samples
        combined_hidden = torch.cat([hidden, pos_hidden, neg_hidden], dim=0)  
        combined_labels = torch.cat([labels, labels, labels], dim=0)  

        # Normalize embeddings
        combined_hidden = F.normalize(combined_hidden, dim=1)

        # Compute similarity matrix
        similarity_matrix = torch.matmul(combined_hidden, combined_hidden.T) / self.temperature 

        # Mask out self-similarities
        logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
        logits = similarity_matrix - logits_max.detach()

        # Generate mask for positive pairs
        mask = torch.eq(combined_labels, combined_labels.T).float().to(device)  
        mask.fill_diagonal_(0)  

        # Compute log probabilities
        exp_logits = torch.exp(logits) * (1 - torch.eye(logits.size(0), device=device))
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        # Compute mean log-likelihood for positive pairs
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-12)

        # Loss is the negative mean of the log-likelihood
        loss = -mean_log_prob_pos.mean()
        return loss


    def predict(self, x):
        return self.model(x)

# Initialize replay buffer
replay_buffer = Buffer(args=argparse.Namespace(mem_size=M, input_size=(1024,), n_classes=num_tasks * classes_per_task)).to(device)

# Initialize ER_AML method
agent = ER_AML(model, replay_buffer, argparse.Namespace(supcon_temperature=temperature, buffer_batch_size=batch_size))


def main():
    task_accuracies = {t: [] for t in range(num_tasks)}

    for task_id in range(num_tasks):
        print(f"\n=== Training on Task {task_id + 1}/{num_tasks} ===")

        # Data loaders for the current task
        train_loader = DataLoader(train_tasks[task_id], batch_size=batch_size, shuffle=True, num_workers=2)
        test_loader = DataLoader(test_tasks[task_id], batch_size=batch_size, shuffle=False, num_workers=2)

        # Training for the current task
        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0.0

            for batch_features, batch_labels in train_loader:
                batch_features = batch_features.to(device)
                batch_labels = batch_labels.to(device)
                idx = torch.arange(batch_features.size(0)).to(device)

                # Observe and compute loss
                loss, replay_loss = agent.observe(batch_features, batch_labels, task_id, idx, rehearse=True)

                # Backpropagation and optimization
                optimizer.zero_grad()
                (loss + replay_loss).backward()
                optimizer.step()

                epoch_loss += loss.item() + replay_loss

            avg_epoch_loss = epoch_loss / len(train_loader)
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_epoch_loss:.4f}")

        # -------------------------------
        # Evaluate after current task
        # -------------------------------
        # Evaluate on all tasks seen so far (i.e., tasks 0..task_id)
        for eval_task_id in range(task_id + 1):
            eval_loader = DataLoader(test_tasks[eval_task_id], batch_size=batch_size, shuffle=False, num_workers=2)
            all_predictions = []
            all_labels = []

            with torch.no_grad():
                for eval_features, eval_labels in eval_loader:
                    eval_features = eval_features.to(device)
                    eval_labels = eval_labels.to(device)
                    logits = agent.predict(eval_features)
                    predictions = torch.argmax(logits, dim=1)

                    preds_mod = predictions.cpu().numpy() % classes_per_task
                    labels_mod = eval_labels.cpu().numpy() % classes_per_task

                    all_predictions.extend(preds_mod)
                    all_labels.extend(labels_mod)

            accuracy = accuracy_score(all_labels, all_predictions)
            task_accuracies[eval_task_id].append(accuracy * 100)
            print(f"  Test Accuracy on Task {eval_task_id + 1} (mod-10): {accuracy * 100:.2f}%")

        # print the average accuracy over tasks 1..(task_id+1)
        avg_acc = np.mean([task_accuracies[t][-1] for t in range(task_id + 1)])
        print(f"Average Accuracy on Tasks [1..{task_id+1}]: {avg_acc:.2f}%")



if __name__ == "__main__":
    main()