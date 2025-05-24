import torch
import numpy as np
from torchvision import datasets, transforms
import os
from torch.utils.data import DataLoader, Dataset, Subset
import random

class DatasetSplit(Dataset):
    """
    An abstract Dataset class for splitting a dataset among clients
    """
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = idxs

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

def get_dataset(dataset_name, data_dir='./data'):
    """
    Get dataset and split into train and test
    """
    if dataset_name == 'mnist':
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=train_transform)
        test_dataset = datasets.MNIST(root=data_dir, train=False, download=True, transform=test_transform)

    elif dataset_name == 'fmnist':
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))
        ])
        train_dataset = datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=train_transform)
        test_dataset = datasets.FashionMNIST(root=data_dir, train=False, download=True, transform=test_transform)

    elif dataset_name == 'cifar10':
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)
        test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_transform)
    
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")

    return train_dataset, test_dataset

def iid_partition(dataset, num_clients):
    """
    Homogeneous partitioning: data is uniformly distributed across clients
    """
    num_items = len(dataset)
    items_per_client = num_items // num_clients
    dict_clients = {}
    all_idxs = list(range(num_items))
    random.shuffle(all_idxs)
    
    for i in range(num_clients):
        dict_clients[i] = all_idxs[i * items_per_client:(i + 1) * items_per_client]
    
    return dict_clients

def label_partition(dataset, num_clients, num_classes_per_client):
    """
    Label Quantity Partitioning: Each client gets data from exactly k classes
    """
    labels = np.array([dataset[i][1] for i in range(len(dataset))])
    all_classes = sorted(np.unique(labels))
    num_classes = len(all_classes)
    
    dict_clients = {i: [] for i in range(num_clients)}
    
    # Assign k classes to each client
    class_assignment = {}
    for i in range(num_clients):
        # Randomly select k classes for this client
        selected_classes = np.random.choice(all_classes, size=num_classes_per_client, replace=False)
        class_assignment[i] = selected_classes
    
    # Create a mapping of class to all indices with that class
    class_to_indices = {}
    for c in all_classes:
        class_to_indices[c] = np.where(labels == c)[0]
    
    # Assign data to clients according to their class assignments
    for client_idx, classes in class_assignment.items():
        client_data = []
        for c in classes:
            client_data.extend(class_to_indices[c])
        dict_clients[client_idx] = client_data
    
    return dict_clients

def dirichlet_partition(dataset, num_clients, beta=0.5):
    """
    Non-IID data partition based on Dirichlet distribution
    """
    labels = np.array([dataset[i][1] for i in range(len(dataset))])
    all_classes = sorted(np.unique(labels))
    num_classes = len(all_classes)
    
    # Calculate class distribution for each client using Dirichlet
    class_distributions = np.random.dirichlet([beta] * num_clients, num_classes)
    
    # Create a mapping of class to all indices with that class
    class_to_indices = {}
    for c in all_classes:
        class_to_indices[c] = np.where(labels == c)[0]
    
    # Initialize client data
    dict_clients = {i: [] for i in range(num_clients)}
    
    # Distribute indices according to the Dirichlet distribution
    for c, distribution in enumerate(class_distributions):
        class_indices = class_to_indices[all_classes[c]]
        np.random.shuffle(class_indices)
        
        # Calculate number of samples per client for this class
        num_samples_per_client = (distribution * len(class_indices)).astype(int)
        diff = len(class_indices) - sum(num_samples_per_client)
        
        # Adjust for rounding errors
        if diff > 0:
            for i in range(diff):
                num_samples_per_client[i] += 1
        elif diff < 0:
            for i in range(-diff):
                num_samples_per_client[i] -= 1
        
        # Distribute samples
        idx = 0
        for client_idx, num_samples in enumerate(num_samples_per_client):
            dict_clients[client_idx].extend(class_indices[idx:idx+num_samples])
            idx += num_samples
    
    return dict_clients

def create_dataloaders(dataset, partition_indices, batch_size=64):
    """
    Create DataLoaders for each client
    """
    client_loaders = {}
    
    for client_idx, indices in partition_indices.items():
        client_dataset = DatasetSplit(dataset, indices)
        client_loaders[client_idx] = DataLoader(
            client_dataset, 
            batch_size=batch_size, 
            shuffle=True
        )
    
    return client_loaders