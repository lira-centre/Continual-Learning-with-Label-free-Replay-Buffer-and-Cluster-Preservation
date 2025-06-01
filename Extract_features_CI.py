import argparse
import torch
import numpy as np
import os
import pickle
from tqdm import tqdm
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder

def preprocess_data(images):
    images = images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).astype('float32') / 255.0
    return images

class ImageNet32Dataset(Dataset):
    def __init__(self, images, labels, transform):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = (self.images[idx] * 255).astype('uint8')
        img = Image.fromarray(img, mode='RGB')
        img = self.transform(img)
        label = self.labels[idx]
        return img, label

def load_tinyimagenet(data_path, split):
    if split == "train":
        data_dir = os.path.join(data_path, "train")
        classes = sorted(os.listdir(data_dir))
        dataset = []
        for class_id in classes:
            class_dir = os.path.join(data_dir, class_id, "images")
            for img_file in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_file)
                dataset.append((img_path, class_id))
    elif split == "val":
        val_dir = os.path.join(data_path, "val")
        annotations = os.path.join(val_dir, "val_annotations.txt")
        with open(annotations, "r") as f:
            lines = f.readlines()
        dataset = [(os.path.join(val_dir, "images", line.split("\t")[0]), line.split("\t")[1]) for line in lines]
    else:
        raise ValueError("Invalid split. Must be 'train' or 'val'.")
    return dataset

def extract_features(dataset, model, device, transform, class_to_idx=None):
    model.eval()
    features_list = []
    labels_list = []
    with torch.no_grad():
        for i, sample in tqdm(enumerate(dataset), total=len(dataset)):
            if isinstance(sample, tuple):
                image, label = sample
                if isinstance(image, str):
                    image = Image.open(image).convert("RGB")
                if class_to_idx:
                    label = class_to_idx[label]
            else:
                image, label = sample

            if isinstance(image, torch.Tensor):
                image = image.to(device).unsqueeze(0)  # CIFAR-100 case
            else:
                image = transform(image).unsqueeze(0).to(device)

            features = model(image)
            features_list.append(features.cpu().numpy())
            labels_list.append(np.array(label))

    features = np.concatenate(features_list, axis=0)
    labels = np.array(labels_list)
    return features, labels


def extract_and_save_features(batch_path, dataset_name, model, device, transform, output_path, is_test=False, batch_size=64):
    with open(batch_path, 'rb') as file:
        batch = pickle.load(file)
    images = preprocess_data(batch['data'])
    labels = np.array(batch['labels'])
    dataset = ImageNet32Dataset(images, labels, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    model.eval()
    features_list = []
    labels_list = []
    with torch.no_grad():
        for batch_images, batch_labels in tqdm(dataloader, desc=f"Processing {os.path.basename(batch_path)}"):
            batch_images = batch_images.to(device)
            features = model(batch_images)
            features_list.append(features.cpu().numpy())
            labels_list.append(batch_labels.numpy())
    features = np.concatenate(features_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    features_path = os.path.join(output_path, f'{dataset_name}_features.npy')
    labels_path = os.path.join(output_path, f'{dataset_name}_labels.npy')
    if os.path.exists(features_path) and not is_test:
        existing_features = np.load(features_path)
        existing_labels = np.load(labels_path)
        features = np.concatenate([existing_features, features], axis=0)
        labels = np.concatenate([existing_labels, labels], axis=0)
    np.save(features_path, features)
    np.save(labels_path, labels)

def save_features_and_labels(dataset_name, dataset, model, device, transform, output_path, class_to_idx=None):
    features, labels = extract_features(dataset, model, device, transform, class_to_idx)
    os.makedirs(output_path, exist_ok=True)
    np.save(os.path.join(output_path, f'{dataset_name}_features.npy'), features)
    np.save(os.path.join(output_path, f'{dataset_name}_labels.npy'), labels)

def main(dataset, data_path, output_path, batch=None, process_val=False, batch_size=64, test_size=0.2, random_seed=42):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14').to(device)

    if dataset == "cifar100":
        train_dataset = datasets.CIFAR100(root=data_path, train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR100(root=data_path, train=False, download=True, transform=transform)
        save_features_and_labels('cifar100_train', train_dataset, model, device, transform, output_path)
        save_features_and_labels('cifar100_test', test_dataset, model, device, transform, output_path)

    elif dataset == "tinyimagenet":
        train_dataset = load_tinyimagenet(data_path, "train")
        class_to_idx = {class_id: idx for idx, class_id in enumerate(sorted({label for _, label in train_dataset}))}
        val_dataset = load_tinyimagenet(data_path, "val")
        save_features_and_labels('tinyimagenet_train', train_dataset, model, device, transform, output_path, class_to_idx)
        save_features_and_labels('tinyimagenet_val', val_dataset, model, device, transform, output_path, class_to_idx)

    elif dataset == "imagenet32":
        if process_val:
            val_path = os.path.join(data_path, 'val_data')
            extract_and_save_features(val_path, 'imagenet32_test', model, device, transform, output_path, is_test=True, batch_size=batch_size)
        elif batch:
            batch_path = os.path.join(data_path, f'train_data_batch_{batch}')
            extract_and_save_features(batch_path, 'imagenet32_train', model, device, transform, output_path, batch_size=batch_size)
        else:
            print("Specify a batch number (1-10) or set process_val=True to process validation data.")

    elif dataset == "caltech256":
        full_dataset = ImageFolder(root=data_path, transform=transform)
        test_size = int(test_size * len(full_dataset))
        train_size = len(full_dataset) - test_size
        trainval_dataset, test_dataset = random_split(full_dataset, [train_size, test_size], generator=torch.Generator().manual_seed(random_seed))
        save_features_and_labels('caltech256_trainval', trainval_dataset, model, device, transform, output_path)
        save_features_and_labels('caltech256_test', test_dataset, model, device, transform, output_path)

    else:
        raise ValueError("Invalid dataset name. Choose from 'cifar100', 'tinyimagenet', 'imagenet32', 'caltech256'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract features from different datasets using DINOv2")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name: 'cifar100', 'tinyimagenet', 'imagenet32', 'caltech256'")
    parser.add_argument("--data_path", type=str, required=True, help="Path to dataset directory")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save extracted features")
    parser.add_argument("--batch", type=int, help="For ImageNet32, specify a batch number (1-10)")
    parser.add_argument("--process_val", action="store_true", help="For ImageNet32, process validation data")
    args = parser.parse_args()
    main(args.dataset, args.data_path, args.output_path, args.batch, args.process_val)
