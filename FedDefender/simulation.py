import torch
import argparse
import os
import sys
import json
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from client import Client
from server import Server
from strategy import FedParaIRLSStrategy
from utils import set_seed, save_results, compare_partitions
import dataset as dataset_module
import models as models_module

def run_experiment(config):
    """
    Run a single federated learning experiment
    """
    # Set seed for reproducibility
    set_seed(config['seed'])
    
    # Set device
    #device = torch.device('cuda' if torch.cuda.is_available() and config['use_cuda'] else 'cpu')
    if not torch.cuda.is_available():
       raise RuntimeError("CUDA n'est pas disponible. Vérifiez votre GPU ou vos drivers.")

    device = torch.device('cuda')  # Forcer GPU
    print("Utilisation forcée du GPU (CUDA)")

    print(f"Using device: {device}")
    
    # Load dataset
    print(f"Loading {config['dataset']} dataset...")
    train_dataset, test_dataset = dataset_module.get_dataset(config['dataset'], config['data_dir'])
    
    # Create test loader
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=config['batch_size'],
        shuffle=False
    )
    
    # Partition dataset
    print(f"Partitioning data using {config['partition']} strategy...")
    if config['partition'] == 'iid':
        partition_indices = dataset_module.iid_partition(train_dataset, config['num_clients'])
    elif config['partition'] == 'label':
        partition_indices = dataset_module.label_partition(
            train_dataset, 
            config['num_clients'], 
            config['num_classes_per_client']
        )
    elif config['partition'] == 'dirichlet':
        partition_indices = dataset_module.dirichlet_partition(
            train_dataset, 
            config['num_clients'], 
            config['beta']
        )
    else:
        raise ValueError(f"Partition strategy {config['partition']} not supported")
    
    # Create data loaders for clients
    client_loaders = dataset_module.create_dataloaders(
        train_dataset, 
        partition_indices, 
        config['batch_size']
    )
    
    # Initialize global model
    print("Initializing global model...")
    global_model = models_module.get_model(config['dataset'], device)
    
    # Determine malicious clients if attacks enabled
    malicious_ids = set()
    if config['attack']:
        num_mal = max(1, int(config['attack_fraction'] * config['num_clients']))
        ids = np.random.choice(range(config['num_clients']), num_mal, replace=False)
        malicious_ids = set(int(i) for i in ids)
        print(f"Malicious clients: {malicious_ids}")

    
    # Initialize clients
    print(f"Setting up {config['num_clients']} clients...")
    clients = []
    for client_id in range(config['num_clients']):
        local_model = models_module.get_model(config['dataset'], device)
        is_mal = config['attack'] and (client_id in malicious_ids)
        clients.append(Client(
            client_id=client_id,
            local_model=local_model,
            dataloader=client_loaders[client_id],
            local_epochs=config['local_epochs'],
            device=device,
            is_malicious=is_mal,
            attack_type=config['attack_type']
        ))
    
    # Initialize server
    server = Server(
        global_model=global_model,
        client_list=clients,
        test_loader=test_loader,
        device=device
    )
    
    # Initialize federated learning strategy
    print("Initializing federated learning strategy...")
    

    strategy = FedParaIRLSStrategy(
    server, clients,
    num_rounds=config['num_rounds'],
    client_sampling_rate=config['client_sampling_rate'],
    fedpara_rank=8, fedpara_steps=800, fedpara_lr=1e-2,
    irls_lambda=2.0, irls_thresh=0.1
)

    
    # Start federated learning
    print(f"Starting federated learning with {config['partition']} partitioning...")
    history, final_model = strategy.train()
    
    # Save results
    if config['save_results']:
        save_results(history, final_model, config, config['save_dir'])
    
    return history, final_model


def run_all_experiments():
    from utils import compare_partitions
    datasets = ['cifar10','mnist', 'fmnist']
    
    for dataset in datasets:
        partition_configs = [
            {'name': 'iid', 'label': 'IID', 'params': {}},
            {'name': 'label', 'label': 'Label (C=1)', 'params': {'num_classes_per_client': 1}},
            {'name': 'label', 'label': 'Label (C=2)', 'params': {'num_classes_per_client': 2}},
            {'name': 'label', 'label': 'Label (C=3)', 'params': {'num_classes_per_client': 3}},
            {'name': 'dirichlet', 'label': 'Dirichlet (β=0.5)', 'params': {'beta': 0.5}},
        ]

        histories = []
        labels = []

        for part in partition_configs:
            config = {
                'dataset': dataset,
                'partition': part['name'],
                'num_clients': 10,
                'num_rounds': 10,
                'local_epochs': 5,
                'batch_size': 64,
                'client_sampling_rate': 1.0,
                'seed': 42,
                'use_cuda': torch.cuda.is_available(),
                'data_dir': './data',
                'save_results': True,
                'save_dir': f'./results/{dataset}',
                'attack': True,
                'attack_type': 'label_flipping',
                'attack_fraction': 0.1
            }
            config.update(part['params'])

            print(f"\n Running {dataset.upper()} with {part['label']}")
            history, _ = run_experiment(config)
            histories.append(history)
            labels.append(part['label'])

        # Courbes comparatives pour ce dataset
        compare_partitions(
            histories,
            labels,
            dataset_name=dataset,
            save_path=f"./results/{dataset}_partition_comparison.png"
        )


def main():
    parser = argparse.ArgumentParser(description='Federated Learning Simulation')
    
    # Dataset options
    parser.add_argument('--dataset', type=str, default='mnist', choices=['cifar10','mnist', 'fmnist'],
                        help='Dataset to use')
    
    # Partition options
    parser.add_argument('--partition', type=str, default='iid', choices=['iid', 'label', 'dirichlet'],
                        help='Data partition strategy')
    parser.add_argument('--num_classes_per_client', type=int, default=1,
                        help='Number of classes per client for label partition')
    parser.add_argument('--beta', type=float, default=0.5,
                        help='Beta parameter for Dirichlet partition')
    
    # Training options
    parser.add_argument('--num_clients', type=int, default=10,
                        help='Number of clients')
    parser.add_argument('--num_rounds', type=int, default=20,
                        help='Number of communication rounds')
    parser.add_argument('--local_epochs', type=int, default=10,
                        help='Number of local epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--client_sampling_rate', type=float, default=1.0,
                        help='Fraction of clients to sample each round')
    
    # Attack options
    parser.add_argument('--attack', action='store_true',
                        help='Enable malicious clients')
    parser.add_argument('--attack_type', type=str, default='label_flipping',
                        choices=['label_flipping', 'sign_flipping', 'gaussian_noise'],
                        help='Type of attack')
    parser.add_argument('--attack_fraction', type=float, default=0.3,
                        help='Fraction of malicious clients (0-1)')
    
    # Misc options
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--use_cuda', action='store_true',
                        help='Use CUDA if available')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Data directory')
    parser.add_argument('--save_results', action='store_true',
                        help='Save results')
    parser.add_argument('--save_dir', type=str, default='./results',
                        help='Directory to save results')
    parser.add_argument('--run_all', action='store_true',
                        help='Run all experiments')
    
    args = parser.parse_args()
    
    if args.run_all:
      run_all_experiments()

    else:
        config = vars(args)
        run_experiment(config)





if __name__ == "__main__":
    main()
