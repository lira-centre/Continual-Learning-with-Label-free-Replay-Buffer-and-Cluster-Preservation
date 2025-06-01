import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
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

# Add arguments
parser.add_argument("--num_classes", type=int, default=5, help="Number of classes")
parser.add_argument("--train_features_path", type=str, required=True, help="Path to train features file (.npy)")
parser.add_argument("--train_labels_path", type=str, required=True, help="Path to train labels file (.npy)")
parser.add_argument("--test_features_path", type=str, required=True, help="Path to test features file (.npy)")
parser.add_argument("--test_labels_path", type=str, required=True, help="Path to test labels file (.npy)")

# Parse arguments
args = parser.parse_args()

# Parameters
batch_size = 64
num_epochs = 5
learning_rate = 0.0001
temperature = 0.07  # Temperature for Supervised Contrastive Loss
num_classes = args.num_classes 

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

# Define the Linear Model
class LinearModel(nn.Module):
    def __init__(self, input_dim=1024, output_dim=512):  # Embedding dimension set to 512
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

model = LinearModel(input_dim=1024, output_dim=512).to(device)

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Supervised Contrastive Loss function
class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(SupervisedContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device
        labels = labels.contiguous().view(-1, 1)
        features = F.normalize(features, dim=1)

        mask = torch.eq(labels, labels.T).float().to(device)

        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature
        )

        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(features.shape[0]).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-12)

        loss = -mean_log_prob_pos
        loss = loss.mean()
        return loss

contrastive_loss_fn = SupervisedContrastiveLoss(temperature=temperature)

# Function to calculate prototypes with K-means
def calculate_prototypes_kmeans(model, data_loader, num_classes):
    model.eval()
    all_embeddings = []
    all_labels = []
    with torch.no_grad():
        for features, labels in data_loader:
            features, labels = features.to(device), labels.to(device)
            embeddings = model(features)
            all_embeddings.append(embeddings)
            all_labels.append(labels)
    
    all_embeddings = torch.cat(all_embeddings)
    all_labels = torch.cat(all_labels)

    # Apply K-means clustering
    kmeans = KMeans(n_clusters=num_classes, random_state=42)
    cluster_labels = kmeans.fit_predict(all_embeddings.cpu().numpy())

    prototypes = []
    prototype_labels = []
    
    for cluster_id in range(num_classes):
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        cluster_embeddings = all_embeddings[cluster_indices]
        
        # Find the prototype closest to the cluster center
        center = torch.tensor(kmeans.cluster_centers_[cluster_id], device=device)
        distances = torch.norm(cluster_embeddings - center, dim=1)
        prototype_index = cluster_indices[torch.argmin(distances).item()]
        
        prototypes.append(all_embeddings[prototype_index])
        prototype_labels.append(all_labels[prototype_index].item())
    
    return prototypes, prototype_labels

# Nearest-prototype classification
def nearest_prototype_classification(model, prototypes, prototype_labels, data_loader):
    model.eval()
    predictions = []
    true_labels = []
    with torch.no_grad():
        for features, labels in data_loader:
            features = features.to(device)
            embeddings = model(features)
            for embedding in embeddings:
                distances = [torch.norm(embedding - prototype) for prototype in prototypes]
                predicted_label = prototype_labels[torch.argmin(torch.tensor(distances))]
                predictions.append(predicted_label)
            true_labels.extend(labels.cpu().numpy())
    return accuracy_score(true_labels, predictions)

# Main training loop for offline learning
def main():
    # Data loader for training on all classes
    train_loader = DataLoader(train_feature_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_feature_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Training on all 100 classes at once
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        for batch_features, batch_labels in train_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)

            # Forward pass through the linear model
            transformed_features = model(batch_features)
            loss = contrastive_loss_fn(transformed_features, batch_labels)

            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_epoch_loss:.4f}")

    # After training, calculate prototypes for all classes
    prototypes, prototype_labels = calculate_prototypes_kmeans(model, train_loader, num_classes)

    # Evaluate on test set using nearest-prototype classification
    accuracy = nearest_prototype_classification(model, prototypes, prototype_labels, test_loader)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()
