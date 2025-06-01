# Continual-Learning-with-Label-free-Replay-Buffer-and-Cluster-Preservation

# **Introduction**

This repository provides code for class-incremental and domain-incremental continual learning experiments of proposed iSL-LRCP and iUL-LRCP methods in the [paper](https://arxiv.org/abs/2504.07240). 

The README is structured as follows:
- **Feature Extraction**: Instructions on extracting DINOv2 ViT-L/14 features for both class-incremental and domain-incremental datasets.
- **Running Experiments**: Guidelines on running continual learning experiments in both class-incremental and domain-incremental settings.

To reproduce results, firstly, DINOv2 features must be extracted and saved as .npy files for the according datasets, then experiments can be conducted on .npy files. This two-stage approach was faster and easier to run than single-stage runs.

# **Feature Extraction**

## **Extracting DINOv2 Features for Class-Incremental Datasets**

### **Overview**
`Extract_features_CI.py` is a feature extraction script designed to extract **DINOv2 ViT-L/14 features** for class-incremental learning datasets. It supports multiple datasets and saves extracted features and labels in `.npy` format, which can be used in continual learning experiments.

### **Usage**
Run the script with the following command:

```bash
python Extract_features_CI.py --dataset {dataset_name} --data_path ./data --output_path ./features
```

### **Supported Datasets**
The script supports the following datasets, specified via the `--dataset` argument:

- **CIFAR-100** (`cifar100`)
- **Caltech256** (`caltech256`)
- **Tiny ImageNet** (`tinyimagenet`)
- **ImageNet32** (`imagenet32`)

### **Dataset-Specific Requirements**
Before running the script, ensure that:
- The dataset is **downloaded** and located in the specified `--data_path` directory.
- The `--output_path` directory exists or is writable, as extracted `.npy` files will be stored there.

### **ImageNet32 Special Parameters**
For **ImageNet32**, you must specify **either**:
- `--process_val` → To extract validation features.
- `--batch N` → To extract training features from batch **N** (`1-5`).

### **Examples**

Extracting features from **batch 3** of ImageNet32 training data:

```bash
python Extract_features_CI.py --dataset imagenet32 --data_path ./data \
    --output_path ./features --batch 3
```

Extracting **validation set** features from ImageNet32:

```bash
python Extract_features_CI.py \
    --dataset imagenet32 --data_path ./data \ 
    --output_path ./features --process_val
```

## **Extracting DINOv2 Features for Domain-Incremental Datasets**

### **CORe50 Dataset**
Run the following command to extract features for the **CORe50** dataset:

```bash
python Extract_features_core50.py --data_root ./data  --output_path ./features
```

### **R-MNIST Dataset**
Run the following command to extract features for the **R-MNIST** dataset:

```bash
python Extract_features_r_mnist.py --data_root ./data --output_path ./features
```

# **Running Experiments**

## **Class-Incremental Setting**

### **Proposed and Baseline Methods**
To run experiments in the **class-incremental setting**, use one of the following commands:

```bash
python {method}.py \
    --num_tasks num_tasks --classes_per_task classes_per_task \
    --train_features_path ./{dataset_name}_train_features.npy \
    --train_labels_path ./{dataset_name}_train_labels.npy \
    --test_features_path ./{dataset_name}_test_features.npy \
    --test_labels_path ./{dataset_name}_test_labels.npy
```

The `{method}` placeholder should be replaced with one of the following methods:
- `iSL-LRCP` (Supervised variant of our proposed method)
- `iUL-LRCP` (Unsupervised variant of our proposed method)
- `ER-AML` (Baseline method)
- `iCaRL` (Baseline method)


### **Offline Learning**
To run an offline learning experiment, use the following command:

```bash
python offline_lft.py \
    --num_classes=num_classes \
    --train_features_path ./{dataset_name}_train_features.npy \
    --train_labels_path ./{dataset_name}_train_labels.npy \
    --test_features_path ./{dataset_name}_test_features.npy \
    --test_labels_path ./{dataset_name}_test_labels.npy
```


## **Domain-Incremental Setting**

### **Proposed and Baseline Methods**
To run experiments in the **domain-incremental setting**, use one of the following commands:

```bash
python {method}.py \
    --num_tasks num_tasks --classes_per_task classes_per_task \
    --train_features_path ./{dataset_name}_train_features.npy \
    --train_labels_path ./{dataset_name}_train_labels.npy \
    --test_features_path ./{dataset_name}_test_features.npy \
    --test_labels_path ./{dataset_name}_test_labels.npy
```

The `{method}` placeholder should be replaced with one of the following methods:
- `iSL-LRCP_domain_inc` (Supervised variant of our proposed method)
- `ER-AML_domain_inc` (Baseline method)
- `iCaRL_domain_inc` (Baseline method)

### **Offline Learning**
To run an offline learning experiment, use the following command:

```bash
python offline_lft_domain_inc.py \
    --num_domains num_domains --classes_per_task classes_per_task \
    --train_features_path ./{dataset_name}_train_features.npy \
    --train_labels_path ./{dataset_name}_train_labels.npy \
    --test_features_path ./{dataset_name}_test_features.npy \
    --test_labels_path ./{dataset_name}_test_labels.npy
```
