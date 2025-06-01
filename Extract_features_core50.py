import os
import numpy as np
import torch
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split

def extract_features(dataset, model, device, domain_index):
    """
    Extract features and labels from the dataset using the given model.
    """
    model.eval()
    features_list = []
    labels_list = []

    with torch.no_grad():
        for i, (image, label) in enumerate(dataset):
    
            image = image.unsqueeze(0).to(device)  # Add batch dimension and move to device
            features = model(image)  # Extract features using the model's forward pass
            features_list.append(features.cpu().numpy().flatten())
            labels_list.append(label + (domain_index * 50))
            print(f"Processed image {i + 1}/{len(dataset)}")

    features = np.array(features_list)
    labels = np.array(labels_list)
    return features, labels

def save_features_and_labels(dataset_name, features, labels, output_path):
    """
    Save extracted features and labels into .npy files.
    """
    os.makedirs(output_path, exist_ok=True)
    np.save(os.path.join(output_path, f'{dataset_name}_features.npy'), features)
    np.save(os.path.join(output_path, f'{dataset_name}_labels.npy'), labels)

def process_pacs(data_root, output_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load pre-trained model
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
    model = model.to(device)

    # Initialize containers for all features and labels
    all_features = []
    all_labels = []

    # Class label mapping
    domain_to_label_offset = {
        "s1": 0,
        "s2": 50,
        "s3": 100,
        "s4": 150,
        "s5": 200,
        "s6": 250,
        "s7": 300,
        "s8": 350,
        "s9": 400,
        "s10": 450,
        "s11": 500,
    }

    domain_index = 0

    # Process each domain
    for domain, label_offset in domain_to_label_offset.items():
        print(f"Processing domain: {domain}")
        domain_path = os.path.join(data_root, domain)

        # Load dataset
        domain_dataset = datasets.ImageFolder(root=domain_path, transform=transform)

        # Update labels for the current domain
        domain_dataset.targets = [label + label_offset + 1 for label in domain_dataset.targets]

        # Extract features
        features, labels = extract_features(domain_dataset, model, device, domain_index)

        domain_index += 1

        # Append features and labels to the main list
        all_features.append(features)
        all_labels.append(labels)

    # Concatenate all features and labels
    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # Split into train (90%) and test (10%)
    train_features, test_features, train_labels, test_labels = train_test_split(
        all_features, all_labels, test_size=0.1, random_state=42, stratify=all_labels
    )

    # Save train and test features and labels
    save_features_and_labels("core50_train", train_features, train_labels, output_path)
    save_features_and_labels("core50_test", test_features, test_labels, output_path)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True, help='Path to PACS dataset root')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save extracted features and labels')
    args = parser.parse_args()

    process_pacs(args.data_root, args.output_path)
