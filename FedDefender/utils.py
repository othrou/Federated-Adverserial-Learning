import matplotlib.pyplot as plt
import numpy as np
import torch
import random
import os
import json
import pickle

def set_seed(seed):
    """
    Set seed for reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def plot_metrics(history, title="Federated Learning Performance", save_path=None):
    """
    Plot training metrics
    
    Args:
        history: Dict with 'round', 'accuracy', and 'loss' lists
        title: Plot title
        save_path: Path to save the plot (if None, plot is displayed)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    
    # Accuracy plot
    ax1.plot(history['round'], history['accuracy'], 'b-', marker='o')
    ax1.set_title('Test Accuracy vs. Communication Rounds')
    ax1.set_xlabel('Communication Round')
    ax1.set_ylabel('Test Accuracy (%)')
    ax1.grid(True)
    
    # Loss plot
    ax2.plot(history['round'], history['loss'], 'r-', marker='o')
    ax2.set_title('Test Loss vs. Communication Rounds')
    ax2.set_xlabel('Communication Round')
    ax2.set_ylabel('Test Loss')
    ax2.grid(True)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    
    plt.close()
import os
import json
import torch
import time

def save_results(history, model_or_state, config, save_dir='./results'):
    """
    Save training results and model
    
    Args:
        history: Training history (dict)
        model_or_state: either an nn.Module or a state_dict (OrderedDict)
        config: Configuration dictionary
        save_dir: Directory to save results
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Génère un nom de fichier unique
    timestamp = int(time.time())
    base = f"{config['dataset']}_{config['partition']}_{timestamp}"
    
    # 1) Sauvegarde de l'historique
    with open(f"{save_dir}/{base}_history.json", 'w') as f:
        json.dump(history, f)
    
    # 2) Sauvegarde du modèle ou du state_dict
    if hasattr(model_or_state, 'state_dict'):
        # un objet nn.Module
        state = model_or_state.state_dict()
    else:
        # on suppose que c'est déjà un state_dict
        state = model_or_state
    torch.save(state, f"{save_dir}/{base}_model.pth")
    
    # 3) Sauvegarde de la config
    with open(f"{save_dir}/{base}_config.json", 'w') as f:
        json.dump(config, f)
    print(f"Saved results to {save_dir}/{base}_history.json")
    # Generate and save plot
import matplotlib.pyplot as plt

def plot_metrics(history, save_path, title):
    # --- 1) Normalisation du format history ---
    # Si c'est une liste de dicts, on reconstruit un dict de listes
    if isinstance(history, list) and len(history) > 0 and isinstance(history[0], dict):
        # clés attendues : 'round', 'accuracy', 'loss'
        metrics = { k: [h[k] for h in history] for k in history[0].keys() }
    elif isinstance(history, dict):
        metrics = history
    else:
        raise ValueError(f"format inattendu pour history : {type(history)}")

    rounds   = metrics.get('round',   [])
    accuracy = metrics.get('accuracy',[])
    loss     = metrics.get('loss',    [])

    # --- 2) Tracé ---
    fig, ax1 = plt.subplots(figsize=(8,5))
    ax1.plot(rounds, accuracy, 'b-', marker='o', label='Accuracy')
    ax1.set_xlabel('Round')
    ax1.set_ylabel('Accuracy (%)', color='b')
    ax1.tick_params(axis='y', labelcolor='b')

    ax2 = ax1.twinx()
    ax2.plot(rounds, loss, 'r--', marker='x', label='Loss')
    ax2.set_ylabel('Loss', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    plt.title(title)
    fig.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Saved plot → {save_path}")


def compare_partitions(histories, partition_names, dataset_name, save_path=None):
    """
    Compare different partition strategies
    
    Args:
        histories: List of history dictionaries
        partition_names: List of partition strategy names
        dataset_name: Name of the dataset
        save_path: Path to save the comparison plot
    """
    plt.figure(figsize=(12, 8))
    
    for i, (history, name) in enumerate(zip(histories, partition_names)):
        plt.plot(history['round'], history['accuracy'], marker='o', label=name)
    
    plt.title(f'Test Accuracy vs. Communication Rounds - {dataset_name.upper()}')
    plt.xlabel('Communication Round')
    plt.ylabel('Test Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    
    plt.close()