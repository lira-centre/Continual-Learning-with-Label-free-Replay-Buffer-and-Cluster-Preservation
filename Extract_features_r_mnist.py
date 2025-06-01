import torch
from torchvision import datasets, transforms
import numpy as np
import os
from sklearn.model_selection import train_test_split

def extract_features(dataset, model, device):
    """
    Extract features and labels from the dataset using the given model.
    """
    model.eval()
    features_list = []
    labels_list = []

    with torch.no_grad():
        for i in range(len(dataset)):
            image, label = dataset[i]
            image = image.unsqueeze(0).to(device)  # Add batch dimension and move to device
            features = model(image)  # Extract features using the model's forward pass
            features_list.append(features.cpu().numpy())
            labels_list.append(np.array(label))
            print(f"Processing sample {i + 1}/{len(dataset)}")

    features = np.concatenate(features_list, axis=0)
    labels = np.array(labels_list)
    return features, labels

def save_features_and_labels(dataset_name, features, labels, output_path):
    """
    Save extracted features and labels into .npy files.
    """
    os.makedirs(output_path, exist_ok=True)
    np.save(os.path.join(output_path, f'{dataset_name}_features.npy'), features)
    np.save(os.path.join(output_path, f'{dataset_name}_labels.npy'), labels)

def main(data_root, output_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data transformations
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),  
        transforms.Resize((224, 224)),             
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2762]),
    ])

    # Load MNIST dataset and randomly select 10,000 images (1,000 per class)
    full_dataset = datasets.MNIST(root=data_root, train=True, download=True, transform=transform)
    selected_indices = []
    for i in range(10):
        indices = np.where(np.array(full_dataset.targets) == i)[0]
        selected_indices.extend(np.random.choice(indices, 1000, replace=False))

    selected_dataset = torch.utils.data.Subset(full_dataset, selected_indices)

    # Load pre-trained model
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
    model = model.to(device)

    # Initialize containers for all features and labels
    all_features = []
    all_labels = []

    # Process original dataset
    print("Processing original MNIST dataset...")
    original_features, original_labels = extract_features(selected_dataset, model, device)
    all_features.append(original_features)
    all_labels.append(original_labels)

    # Process rotated datasets (60, 120, ..., 300 degrees)
    for i, angle in enumerate(range(60, 301, 60)):
        print(f"Processing rotated MNIST dataset at {angle} degrees...")
        rotation_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((224, 224)),
            transforms.RandomRotation((angle, angle)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2762]),
        ])

        rotated_dataset = datasets.MNIST(
            root=data_root,
            train=True,
            download=True,
            transform=rotation_transform
        )
        rotated_dataset = torch.utils.data.Subset(rotated_dataset, selected_indices)

        rotated_features, rotated_labels = extract_features(rotated_dataset, model, device)

        # Adjust labels for rotation
        rotated_labels += (i + 1) * 10
        all_features.append(rotated_features)
        all_labels.append(rotated_labels)

    # Concatenate all features and labels
    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # Split data into train (90%) and test (10%) sets
    train_features, test_features, train_labels, test_labels = train_test_split(
        all_features, all_labels, test_size=0.1, random_state=42, stratify=all_labels
    )

    # Save features and labels for train and test sets
    save_features_and_labels('r_mnist_train', train_features, train_labels, output_path)
    save_features_and_labels('r_mnist_test', test_features, test_labels, output_path)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True, help='Path to MNIST data')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save extracted features')
    args = parser.parse_args()

    main(args.data_root, args.output_path)
