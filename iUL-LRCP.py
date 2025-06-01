import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
from sklearn.metrics import accuracy_score
import random
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
import argparse

# Argument parser
parser = argparse.ArgumentParser(description="Continual Learning with Prototypes")

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Add arguments
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

# Loss function hyperparameters
lambda_preserve = 0.5
lambda_push = 2
num_selected_dims = 5  # Number of dimensions to select
threshold_correlation = 0.3  # Threshold for correlation in dimension selection
temperature = 0.07  # Temperature parameter for Supervised Contrastive Loss

sigma_bands=[1, 2, 3]

train_features_path = args.train_features_path
train_labels_path = args.train_labels_path
test_features_path = args.test_features_path
test_labels_path = args.test_labels_path

# Custom Dataset for pre-extracted features
class FeatureDataset(Dataset):
    def __init__(self, features, labels):
        """
        Args:
            features (np.ndarray): Numpy array of features.
            labels (np.ndarray): Numpy array of labels.
        """
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
    class_indices = [[] for _ in range(num_tasks * classes_per_task)]  # Assuming 100 classes as in CIFAR-100
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

# Memory for prototypes and cluster parameters
prototypes_input = {}  # {class_label: prototype_feature_vector}
prototypes_latent = {}
support_prototypes_input = {}  # {class_label: [support_feature_vectors]}
support_prototypes_latent = {}
cluster_means = {}  # {class_label: mean_feature_vector}
cluster_stds = {}  # {class_label: std_feature_vector}

# Function to compute class means and standard deviations
def compute_class_stats(features, labels, num_classes, task_id):
    kmeans = KMeans(n_clusters=num_classes, random_state=42)
    labels = labels.cpu().numpy()
    features_np = features.cpu().detach().numpy()
    
    # Apply K-means clustering
    kmeans.fit(features_np)

    class_means = {}
    class_stds = {}

    for i in range(task_id * num_classes, (task_id + 1) * num_classes):
        cluster_features= features[torch.tensor(kmeans.labels_ == i - (task_id * num_classes))]
        if cluster_features.size(0) == 0:
            continue
        std = torch.std(cluster_features, dim=0)
        # To avoid division by zero in std
        std[std == 0] = 1e-6
        class_means[i] = torch.tensor(kmeans.cluster_centers_[i - (task_id * num_classes)]).to(device)
        class_stds[i] = std
    return class_means, class_stds, kmeans.labels_

def select_prototypes(features, features_transformed, labels, cluster_labels, class_means, prototype_labels, task_id, num_classes, model):
    """
    Select prototypes in both the original input space (1024D) and the transformed space (512D).
    :param features: 1024D input features
    :param labels: Corresponding class labels
    :param class_means: Mean feature vectors for each class
    :param model: The linear model to transform 1024D to 512D
    :return: Two dictionaries: {class_label: prototype_input_1024D}, {class_label: prototype_latent_512D}
    """
    prototypes_input = {}
    prototypes_latent = {}
    unique_labels = torch.unique(labels)
    
    for index, cluster_center in class_means.items():
        cluster_features = features[cluster_labels == index - (task_id * num_classes)]
        cluster_features_transformed = features_transformed[cluster_labels == index - (task_id * num_classes)]
        cluster_actual_labels = labels[cluster_labels == index - (task_id * num_classes)]
        distances = torch.norm(cluster_features_transformed - cluster_center, dim=1)
        idx = torch.argmin(distances)
        # Save the prototype in input space (1024D)
        proto_input = cluster_features[idx].detach().clone()
        prototype_label = cluster_actual_labels[idx].detach().clone()
        prototypes_input[unique_labels[index - (task_id * num_classes)].item()] = proto_input

        # Transform the prototype to latent space (512D)
        proto_latent = model(proto_input.unsqueeze(0)).squeeze(0)
        prototypes_latent[unique_labels[index - (task_id * num_classes)].item()] = proto_latent
        prototype_labels.append(prototype_label.item())

    return prototypes_input, prototypes_latent, prototype_labels


# Function to select dimensions unsupervisedly
def select_dimensions_unsupervised(features, features_transformed, num_selected_dims, threshold_correlation=0.9):
    """
    Select dimensions based on maximum variance and minimum correlation (unsupervised).
    :param features: Tensor of shape [num_samples, feature_dim]
    :param num_selected_dims: Number of dimensions to select
    :param threshold_correlation: Threshold for correlation to consider dimensions redundant
    :return: List of selected dimension indices
    """
    features_np = features_transformed.cpu().numpy()
    # Compute variance of each dimension
    variances = np.var(features_np, axis=0)
    # Sort dimensions by variance in descending order
    sorted_dims = np.argsort(-variances)
    selected_dims = []
    # Compute correlation matrix
    corr_matrix = np.corrcoef(features_np, rowvar=False)
    for dim in sorted_dims:
        if len(selected_dims) == 0:
            selected_dims.append(dim)
        else:
            # Compute maximum absolute correlation with selected dimensions
            max_corr = max([abs(corr_matrix[dim, d]) for d in selected_dims])
            if max_corr < threshold_correlation:
                selected_dims.append(dim)
        if len(selected_dims) >= num_selected_dims:
            break
    return selected_dims

def select_dimensions_unsupervised_per_cluster(features, features_transformed, labels, cluster_labels, num_selected_dims, num_classes, threshold_correlation=0.9):
    """
    Select dimensions based on maximum variance and minimum correlation (unsupervised),
    applied separately to each cluster.
    :param features: Tensor of shape [num_samples, feature_dim]
    :param labels: Tensor of shape [num_samples]
    :param num_selected_dims: Number of dimensions to select
    :param threshold_correlation: Threshold for correlation to consider dimensions redundant
    :return: Dictionary {class_label: [selected dimension indices]}
    """
    selected_dims_per_class = {}
    for i in range(num_classes):
        cluster_features = features[cluster_labels == i]
        cluster_features_transformed = features_transformed[cluster_labels == i]
        if cluster_features.size(0) == 0:
            selected_dims_per_class[i] = []
            continue
        selected_dims = select_dimensions_unsupervised(cluster_features, cluster_features_transformed, num_selected_dims, threshold_correlation)
        selected_dims_per_class[i] = selected_dims
    return selected_dims_per_class


def select_support_prototypes_multi_sigma(features_transformed, features, labels, cluster_labels, class_means, class_stds, selected_dims_per_class, sigma_bands, num_classes, current_task_id, model):
    """
    Select support prototypes along multiple sigma bands for each selected dimension.

    Args:
        features (torch.Tensor): Latent representations of shape [num_samples, feature_dim].
        inputs (torch.Tensor): Input representations corresponding to features, of shape [num_samples, ...].
        labels (torch.Tensor): Labels of shape [num_samples].
        class_means (dict): {class_label: mean_feature_vector}.
        class_stds (dict): {class_label: std_feature_vector}.
        selected_dims_per_class (dict): {class_label: [selected dimension indices]}.
        sigma_bands (list): List of sigma bands to use (e.g., [1, 2, 3]).

    Returns:
        support_prototypes_input: {class_label: [input_support_prototypes]}
        support_prototypes_latent: {class_label: [latent_support_prototypes]}
    """
    support_prototypes_input = {}
    support_prototypes_latent = {}

    for i in range(current_task_id * num_classes, (current_task_id + 1) * num_classes):
        cluster_features_transformed = features_transformed[cluster_labels == i - (current_task_id * num_classes)]
        cluster_features = features[cluster_labels == i - (current_task_id * num_classes)]
        mean = class_means[i]
        std = class_stds[i]
        selected_dims = selected_dims_per_class.get(i - (current_task_id * num_classes), [])
        support_inputs = []
        support_latents = []

        for dim in selected_dims:
            for k in sigma_bands:
                for direction in [-1, 1]:
                    # Compute the target point in the latent space
                    target_latent = mean.clone()
                    target_latent[dim] += direction * k * std[dim]

                    # Find the nearest sample to this target point
                    distances = torch.norm(cluster_features_transformed - target_latent, dim=1)
                    idx = torch.argmin(distances)
                    support_inputs.append(cluster_features[idx])
                    support_latents.append(cluster_features_transformed[idx])

        support_prototypes_input[i] = support_inputs
        support_prototypes_latent[i] = support_latents

    return support_prototypes_input, support_prototypes_latent


def compute_mmd(x, y, kernel='rbf', sigma=1.0):
    """
    Compute the MMD loss between two sets of samples using the specified kernel.

    Args:
        x (torch.Tensor): Samples from distribution P. Shape: (n_samples_x, feature_dim)
        y (torch.Tensor): Samples from distribution Q. Shape: (n_samples_y, feature_dim)
        kernel (str): Kernel type ('rbf' or 'linear')
        sigma (float): Bandwidth parameter for the RBF kernel

    Returns:
        mmd_loss (torch.Tensor): Scalar tensor representing the MMD loss
    """
    device = x.device

    # Compute kernels
    if kernel == 'linear':
        xx = torch.mm(x, x.t())  # (n_x, n_x)
        yy = torch.mm(y, y.t())  # (n_y, n_y)
        xy = torch.mm(x, y.t())  # (n_x, n_y)
    elif kernel == 'rbf':
        xx = rbf_kernel(x, x, sigma)  # (n_x, n_x)
        yy = rbf_kernel(y, y, sigma)  # (n_y, n_y)
        xy = rbf_kernel(x, y, sigma)  # (n_x, n_y)
    else:
        raise ValueError(f"Unsupported kernel type: {kernel}")

    mmd = xx.mean() - 2 * xy.mean() + yy.mean()
    return mmd


def rbf_kernel(x, y, sigma):
    """
    Compute the RBF kernel between x and y.

    Args:
        x (torch.Tensor): Shape: (n_samples_x, feature_dim)
        y (torch.Tensor): Shape: (n_samples_y, feature_dim)
        sigma (float): Bandwidth parameter

    Returns:
        torch.Tensor: Kernel matrix of shape (n_samples_x, n_samples_y)
    """
    x_norm = (x ** 2).sum(dim=1).view(-1, 1)
    y_norm = (y ** 2).sum(dim=1).view(1, -1)
    dist = x_norm + y_norm - 2.0 * torch.mm(x, y.t())
    kernel = torch.exp(-dist / (2 * sigma ** 2))
    return kernel


def cluster_preservation_loss_mmd(model, support_prototypes_input, support_prototypes_latent, main_prototypes_input, main_prototypes_latent, sigma=1.0):
    """
    Compute the Cluster Preservation Loss using MMD between distributions.

    Args:
        model (nn.Module): The current model.
        support_prototypes_input (dict): {class_label: [input_support_prototypes]}.
        support_prototypes_latent (dict): {class_label: [latent_support_prototypes]}.
        main_prototypes_input (dict): {class_label: input_main_prototype}.
        main_prototypes_latent (dict): {class_label: latent_main_prototype}.
        sigma (float): Bandwidth parameter for the RBF kernel.

    Returns:
        torch.Tensor: Scalar tensor representing the total MMD loss.
    """
    device = next(model.parameters()).device
    total_mmd_loss = torch.tensor(0.0, device=device)

    for cls in support_prototypes_input:
        # Collect old latent representations
        z_old = torch.stack(support_prototypes_latent[cls]).to(device)  # Shape: (num_support, feature_dim)
        z_old_main = main_prototypes_latent[cls].unsqueeze(0).to(device)  # Shape: (1, feature_dim)
        z_old_all = torch.cat([z_old_main, z_old], dim=0)  # Shape: (num_samples, feature_dim)

        # Collect input representations and compute new latent representations
        x_support = torch.stack(support_prototypes_input[cls]).to(device)
        x_main = main_prototypes_input[cls].unsqueeze(0).to(device)
        x_all = torch.cat([x_main, x_support], dim=0)  # Shape: (num_samples, ...)

        # Compute new latent representations
        z_new_all = model(x_all)  # Shape: (num_samples, feature_dim)

        # Compute MMD loss between z_old_all and z_new_all
        mmd_loss = compute_mmd(z_old_all, z_new_all, kernel='rbf', sigma=sigma)

        # Accumulate total loss
        total_mmd_loss += mmd_loss

    return total_mmd_loss

#'''
def contrastive_push_away_loss(new_features, prototypes_input, prototypes_latent, previous_means, previous_stds, model, epsilon=1e-6):
    """
    Compute the Push Away Loss with temperature scaling based on sigma values for each cluster.
    
    Args:
        new_features (torch.Tensor): Feature representations of the new samples.
            Shape: (batch_size, feature_dim)
        prototypes_input (dict): Dictionary mapping class labels to feature vectors (prototypes).
        previous_means (dict): Dictionary mapping class labels to feature vectors (cluster means) of previous classes.
        previous_stds (dict): Dictionary mapping class labels to standard deviation vectors of previous classes.
        model (torch.nn.Module): Model to compute new representations of prototypes.
        temperature (float): Temperature scaling factor.
        epsilon (float): Small constant to prevent division by zero.
    
    Returns:
        torch.Tensor: Scalar tensor representing the loss.
    """
    device = new_features.device
    if not previous_means:
        # No previous classes, return zero loss
        return torch.tensor(0.0, device=device)
    
    # Normalize new features
    new_features = F.normalize(new_features, dim=1)  # Shape: (batch_size, feature_dim)
    
    # Prepare lists to collect per-cluster contributions
    loss_terms = []
    
    for cls in prototypes_input:
        prototype_input = prototypes_input[cls].unsqueeze(0).to(device)  # Shape: (1, feature_dim)
        # Compute new representation of the prototype in latent space
        #prototype_new_latent = model(prototype_input).T
        prototype_latent = prototypes_latent[cls].unsqueeze(0).T.to(device)

        # Retrieve cluster-specific standard deviation and compute sigma
        std = previous_stds[cls].to(device)  # Shape: (feature_dim,)
        sigma = std + epsilon  # Ensure sigma is positive

        # Normalize the prototype representation
        prototype_latent = F.normalize(prototype_latent, dim=0)  # Shape: (feature_dim,)

        # Compute cosine similarity between new features and prototype of previous class
        sim = torch.matmul(new_features, prototype_latent)  # Shape: (batch_size,)

        # Scale similarity by sigma-adjusted temperature
        scaled_sim = sim / ((1 - sigma) *7 )  # Shape: (batch_size,)

        # Directly minimize the scaled similarity
        loss = scaled_sim  # Shape: (batch_size,)

        # Accumulate loss terms
        loss_terms.append(loss)
    
    # Stack and sum loss terms over clusters
    loss_terms = torch.stack(loss_terms, dim=1)
    
    # Compute mean over samples and clusters
    total_loss = torch.mean(loss_terms)
    
    return total_loss
#'''


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

        # For numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # Mask out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(features.shape[0]).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # Compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        # Compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-12)

        # Loss
        loss = -mean_log_prob_pos
        loss = loss.mean()
        return loss

contrastive_loss_fn = SupervisedContrastiveLoss(temperature=temperature)

# Function for nearest prototype classification
def nearest_prototype_classification(features, prototypes, prototype_labels):
    """
    Classify each feature by the nearest prototype.
    :param features: Tensor of shape [num_samples, feature_dim]
    :param prototypes: Dictionary {class_label: prototype_feature_vector}
    :return: List of predicted class labels
    """
    predictions = []
    prototype_vectors = torch.stack([prototypes[cls] for cls in list(prototypes.keys())]).to(device)  # [num_classes, feature_dim]

    with torch.no_grad():
        for feature in features:
            # Compute distances to all prototypes
            distances = torch.norm(prototype_vectors - feature, dim=1)  # [num_classes]
            min_distance, min_idx = torch.min(distances, dim=0)
            predicted_label = prototype_labels[min_idx.item()]
            predictions.append(predicted_label)
    return predictions

def main():
    prototype_labels = []
    for task_id in range(num_tasks):
        print(f"\n=== Training on Task {task_id + 1}/{num_tasks} ===")

        # Data loaders for the current task
        train_loader = DataLoader(train_tasks[task_id], batch_size=batch_size, shuffle=True, num_workers=2)
        test_loader = DataLoader(test_tasks[task_id], batch_size=batch_size, shuffle=False, num_workers=2)

        # Determine current and previous classes
        current_classes = list(range(task_id * classes_per_task, (task_id + 1) * classes_per_task))
        if task_id == 0:
            previous_classes = []
        else:
            previous_classes = list(prototypes_latent.keys())
        
        # Training for the current task
        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0.0

            # Number of clusters (equal to number of classes in current task)
            n_clusters = classes_per_task

            # Initialize MiniBatchKMeans once per epoch or task
            mb_kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=256)

            for batch_features, batch_labels in train_loader:
                batch_features = batch_features.to(device)
                batch_labels = batch_labels.to(device)

                # Forward pass through the linear model
                transformed_features = model(batch_features)

                # Convert batch_features to numpy array for clustering (on CPU)
                batch_features_np = batch_features.detach().cpu().numpy()

                # Update MiniBatchKMeans with the current batch features
                mb_kmeans.partial_fit(batch_features_np)
                
                # Predict cluster labels using batch_features
                pseudo_labels = mb_kmeans.predict(batch_features_np)
                pseudo_labels = torch.tensor(pseudo_labels, dtype=torch.long, device=device)
         
                # Task-Specific Loss (Supervised Contrastive Loss)
                task_loss = contrastive_loss_fn(transformed_features, pseudo_labels)

                # Cluster Preservation Loss
                preserve_loss = 0.0

                # Cluster Preservation Loss using the new function
                if previous_classes:
                    preserve_loss = cluster_preservation_loss_mmd(model, support_prototypes_input, support_prototypes_latent, prototypes_input, prototypes_latent)
                else:
                    preserve_loss = torch.tensor(0.0, device=device)

                # Push Away Previous Clusters Loss
                if previous_classes:
                    prev_means = {cls: cluster_means[cls] for cls in previous_classes}
                    prev_stds = {cls: cluster_stds[cls] for cls in previous_classes}
                    push_loss = contrastive_push_away_loss(transformed_features, prototypes_input, prototypes_latent, prev_means, prev_stds, model, epsilon=1e-6)
                else:
                    push_loss = 0.0

                # Total loss
                total_loss = task_loss + lambda_preserve * preserve_loss + lambda_push * push_loss

                # Backpropagation and optimization
                optimizer.zero_grad()
                total_loss.backward(retain_graph=True)
                optimizer.step()

                epoch_loss += total_loss.item()

            avg_epoch_loss = epoch_loss / len(train_loader)
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_epoch_loss:.4f}")

        # After training, collect transformed features and labels for the current task
        model.eval()
        all_features = []
        all_labels = []
        all_features_transformed = []

        with torch.no_grad():
            for batch_features, batch_labels in train_loader:
                batch_features = batch_features.to(device)
                batch_labels = batch_labels.to(device)
                transformed = model(batch_features)

                all_features_transformed.append(transformed)
                all_features.append(batch_features)

                all_labels.append(batch_labels)

        all_features = torch.cat(all_features, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        all_features_transformed = torch.cat(all_features_transformed, dim=0)

        # Compute class means and stds for the current task
        class_means, class_stds, cluster_labels = compute_class_stats(all_features_transformed, all_labels, classes_per_task, task_id)

        # Select prototypes for the current task (both 1024D and 512D)
        task_prototypes_input, task_prototypes_latent, prototype_labels = select_prototypes(all_features, all_features_transformed, all_labels, cluster_labels, class_means, prototype_labels, task_id, classes_per_task, model)
        #'''
        # Ensure prototypes are detached
        for cls in task_prototypes_input:
            task_prototypes_input[cls] = task_prototypes_input[cls].detach()
        # Ensure prototypes are detached
        for cls in task_prototypes_latent:
            task_prototypes_latent[cls] = task_prototypes_latent[cls].detach()
        #'''
        
        # Select dimensions unsupervisedly for the current task
        selected_dims_per_class = select_dimensions_unsupervised_per_cluster(
            all_features, all_features_transformed, all_labels, cluster_labels, num_selected_dims, classes_per_task, threshold_correlation)
        
        # Select support prototypes for the current task (both 1024D and 512D)
        task_support_prototypes_input, task_support_prototypes_latent = select_support_prototypes_multi_sigma(
            all_features_transformed, all_features, all_labels, cluster_labels, class_means, class_stds, selected_dims_per_class, sigma_bands, classes_per_task, task_id, model)
        #'''
        # Ensure prototypes are detached
        for cls in task_support_prototypes_input:
            for j in range(len(task_support_prototypes_input[cls])):
                task_support_prototypes_input[cls][j] = task_support_prototypes_input[cls][j].detach()
            # Ensure prototypes are detached
        for cls in task_support_prototypes_latent:
            for j in range(len(task_support_prototypes_latent[cls])):
                task_support_prototypes_latent[cls][j] = task_support_prototypes_latent[cls][j].detach()

        #'''
        #'''
        # Update cluster parameters and save prototypes
        prototypes_input.update(task_prototypes_input)
        prototypes_latent.update(task_prototypes_latent)
        support_prototypes_input.update(task_support_prototypes_input)
        support_prototypes_latent.update(task_support_prototypes_latent)
        cluster_means.update(class_means)
        cluster_stds.update(class_stds)

        
        #'''
        # Evaluation on all seen tasks
        model.eval()
        all_test_features = []
        all_test_labels = []

        # Collect transformed test features for all seen tasks
        for eval_task_id in range(task_id + 1):
            eval_test_loader = DataLoader(test_tasks[eval_task_id], batch_size=batch_size, shuffle=False, num_workers=2)
            with torch.no_grad():
                for batch_features, batch_labels in eval_test_loader:
                    batch_features = batch_features.to(device)
                    batch_labels = batch_labels.to(device)
                    transformed = model(batch_features)
                    all_test_features.append(transformed)
                    all_test_labels.append(batch_labels)

        all_test_features = torch.cat(all_test_features, dim=0)
        all_test_labels = torch.cat(all_test_labels, dim=0)

        # Nearest prototype classification
        predictions = nearest_prototype_classification(all_test_features, prototypes_latent, prototype_labels)

        # Calculate accuracy
        accuracy = accuracy_score(all_test_labels.cpu(), predictions)
        print(f"Test Accuracy on Tasks 1 to {task_id + 1}: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()

