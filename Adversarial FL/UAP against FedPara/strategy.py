import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Subset, random_split
import numpy as np
import os
from utils import NUM_CLASSES # Assuming utils.py is in the same directory

def load_cifar10_data(data_path="./cifar_data", batch_size=64, data_fraction_overall=1.0):
    """Loads CIFAR-10 train and test sets."""
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Normalizes to [-1, 1]
    ])

    full_trainset = datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform)
    full_testset = datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform)

    if data_fraction_overall < 1.0:
        num_train_samples = int(data_fraction_overall * len(full_trainset))
        num_test_samples = int(data_fraction_overall * len(full_testset))
        
        train_indices = np.random.choice(len(full_trainset), num_train_samples, replace=False)
        test_indices = np.random.choice(len(full_testset), num_test_samples, replace=False)
        
        trainset = Subset(full_trainset, train_indices)
        testset = Subset(full_testset, test_indices)
    else:
        trainset = full_trainset
        testset = full_testset
    
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    print(f"CIFAR-10 data loaded: {len(trainset)} train samples, {len(testset)} test samples.")
    return trainset, test_loader


def partition_data_homogeneous(trainset, num_clients, batch_size):
    """Distributes data homogeneously (IID) among clients."""
    if num_clients == 0: return []
    if len(trainset) < num_clients :
        print(f"Warning: Not enough data for {num_clients} clients. Some clients might get no data.")
        # Fallback: give all data to first client if only one, or distribute as much as possible
        if num_clients == 1:
            return [DataLoader(trainset, batch_size=batch_size, shuffle=True)]
        else: # Distribute what's available, remaining clients get empty loaders
            samples_per_client = 1 
            actual_num_clients_with_data = len(trainset)
            lengths = [samples_per_client] * actual_num_clients_with_data
            client_datasets = random_split(trainset, lengths)
            client_loaders = [DataLoader(ds, batch_size=batch_size, shuffle=True) for ds in client_datasets]
            for _ in range(num_clients - actual_num_clients_with_data):
                client_loaders.append(DataLoader(Subset(trainset,[]),batch_size=batch_size,shuffle=True))
            return client_loaders


    samples_per_client = len(trainset) // num_clients
    lengths = [samples_per_client] * num_clients
    remainder = len(trainset) % num_clients
    for i in range(remainder):
        lengths[i] += 1
    
    client_datasets = random_split(trainset, lengths)
    client_loaders = [DataLoader(ds, batch_size=batch_size, shuffle=True) for ds in client_datasets]
    print(f"Data partitioned homogeneously for {num_clients} clients. Sizes: {[len(dl.dataset) for dl in client_loaders]}")
    return client_loaders


def partition_data_label_quantity(trainset, num_clients, num_classes_per_client, batch_size, total_num_classes=NUM_CLASSES):
    """Partitions data so each client gets data from exactly `num_classes_per_client` classes."""
    if num_clients == 0: return []
    
    # Get targets from the base dataset if trainset is a Subset
    if isinstance(trainset, Subset):
        targets = np.array(trainset.dataset.targets)[trainset.indices]
        original_indices = np.array(trainset.indices)
    else:
        targets = np.array(trainset.targets)
        original_indices = np.arange(len(trainset))

    # Map class to indices
    class_indices = {i: original_indices[targets == i] for i in range(total_num_classes)}
    for c in class_indices: np.random.shuffle(class_indices[c]) # Shuffle indices within each class

    client_data_indices = [[] for _ in range(num_clients)]
    
    # Ensure enough classes are available
    if total_num_classes < num_classes_per_client * num_clients / np.ceil(num_clients * num_classes_per_client / total_num_classes): # Approx check
         print(f"Warning: Not enough distinct classes for all clients with num_classes_per_client={num_classes_per_client}. Distribution might be skewed or some clients get fewer classes.")

    # Distribute classes to clients cyclically or randomly to try and balance
    available_classes = list(range(total_num_classes))
    np.random.shuffle(available_classes)
    
    class_ptr = 0
    for client_id in range(num_clients):
        assigned_classes_for_client = []
        for _ in range(num_classes_per_client):
            if not available_classes: # Should not happen if total_num_classes is large enough
                # If run out of globally available distinct classes, allow re-picking (less ideal)
                 # This part needs more robust handling for edge cases where class diversity is low
                pick_from = list(range(total_num_classes)) 
                np.random.shuffle(pick_from)
                assigned_classes_for_client.append(pick_from[0])
                continue

            assigned_classes_for_client.append(available_classes[class_ptr % len(available_classes)])
            class_ptr +=1
        
        # Distribute samples for assigned classes
        for cls in assigned_classes_for_client:
            # Simple split: give a chunk of this class's data
            # This could be improved for more balanced distribution of samples per class
            num_samples_this_class = len(class_indices[cls])
            if num_samples_this_class == 0: continue

            # Simplified: give up to total/num_clients samples of this class if available
            # More robust: divide samples of a class among clients that get this class
            # For now, split roughly
            
            # Estimate how many clients might get this class to divide samples
            # This is a heuristic. A better way would be to pre-assign classes then divide samples.
            clients_getting_this_class_approx = max(1, np.ceil(num_clients * num_classes_per_client / total_num_classes))
            
            samples_to_take = num_samples_this_class // int(clients_getting_this_class_approx)
            samples_to_take = max(1, samples_to_take) # take at least one if available

            # Pop samples from class_indices[cls]
            taken_samples_count = 0
            while class_indices[cls].size > 0 and taken_samples_count < samples_to_take:
                client_data_indices[client_id].append(class_indices[cls][-1])
                class_indices[cls] = class_indices[cls][:-1] # "Pop" by slicing
                taken_samples_count += 1
    
    client_loaders = []
    for indices in client_data_indices:
        if indices: # only create loader if there are samples
            # Use trainset.dataset to access the original full dataset
            client_ds = Subset(trainset.dataset if isinstance(trainset, Subset) else trainset, list(set(indices))) 
            client_loaders.append(DataLoader(client_ds, batch_size=batch_size, shuffle=True))
        else: # Add empty dataloader
            client_loaders.append(DataLoader(Subset(trainset.dataset if isinstance(trainset, Subset) else trainset, []), batch_size=batch_size, shuffle=True))

    print(f"Data partitioned by Label Quantity (#C={num_classes_per_client}) for {num_clients} clients. Sizes: {[len(dl.dataset) for dl in client_loaders]}")
    # For verification:
    # for i, loader in enumerate(client_loaders):
    #     labels_this_client = []
    #     if len(loader.dataset) > 0:
    #         for _, lab in loader.dataset: labels_this_client.append(lab)
    #     print(f"Client {i}, num_samples: {len(loader.dataset)}, unique_labels: {np.unique(labels_this_client)}")
    return client_loaders


def partition_data_dirichlet(trainset, num_clients, beta, batch_size, total_num_classes=NUM_CLASSES):
    """Partitions data using Dirichlet distribution for label skew Non-IID."""
    if num_clients == 0: return []

    if isinstance(trainset, Subset):
        targets = np.array(trainset.dataset.targets)[trainset.indices]
        original_indices_all = np.array(trainset.indices)
    else:
        targets = np.array(trainset.targets)
        original_indices_all = np.arange(len(trainset))
    
    min_size = 0
    min_require_size = 10 # Each client should have at least 10 samples, can be adjusted
    
    client_data_indices = [[] for _ in range(num_clients)]
    
    # Ensure there's enough data for the number of clients and classes
    if len(original_indices_all) < num_clients * min_require_size:
        print(f"Warning: Not enough data for Dirichlet partitioning with min_require_size={min_require_size}. len(data)={len(original_indices_all)}, num_clients={num_clients}")
        # Fallback to simpler distribution or error
        return partition_data_homogeneous(trainset, num_clients, batch_size) # Fallback

    for k in range(total_num_classes):
        idx_k = original_indices_all[targets == k]
        np.random.shuffle(idx_k)
        proportions = np.random.dirichlet(np.repeat(beta, num_clients))
        
        # Balance proportions if some are too small leading to no samples
        proportions = np.array([p * (len(idx_j) < min_require_size) for p, idx_j in zip(proportions, client_data_indices)]) # Heuristic to give more to clients with less data
        proportions = proportions / proportions.sum() # Normalize again
        
        proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
        
        split_indices = np.split(idx_k, proportions)
        for i in range(num_clients):
            client_data_indices[i].extend(split_indices[i].tolist())

    client_loaders = []
    for indices in client_data_indices:
        if indices:
            # Use trainset.dataset to access the original full dataset
            client_ds = Subset(trainset.dataset if isinstance(trainset, Subset) else trainset, list(set(indices)))
            client_loaders.append(DataLoader(client_ds, batch_size=batch_size, shuffle=True))
        else:
            client_loaders.append(DataLoader(Subset(trainset.dataset if isinstance(trainset, Subset) else trainset, []), batch_size=batch_size, shuffle=True))
            
    print(f"Data partitioned by Dirichlet (beta={beta}) for {num_clients} clients. Sizes: {[len(dl.dataset) for dl in client_loaders]}")
    return client_loaders