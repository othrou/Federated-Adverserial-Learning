import torch
import copy
import numpy as np
import os

from utils import (
    CIFAR10CNN, CIFAR10FedParaCNN, UAPAttack, NoAttack,
    NUM_CLASSES, CLASSES, get_model_parameters_size
)
from strategy import (
    load_cifar10_data, partition_data_homogeneous,
    partition_data_label_quantity, partition_data_dirichlet
)
from client import Client
from server import Server
from visualizations import (
    plot_accuracy_vs_rounds, plot_class_accuracy_comparison,
    plot_communication_savings, calculate_target_class_survival_rate
)

# --- Experiment Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Federated Learning Parameters
NUM_ROUNDS = 20
NUM_CLIENTS = 10
CLIENT_LOCAL_EPOCHS = 2
LEARNING_RATE = 0.01
BATCH_SIZE = 64
DATA_FRACTION_OVERALL = 1.0 # Use all available CIFAR-10 data for splitting

# Model Configuration
# MODEL_TYPE = "standard" # "standard" or "fedpara"
MODEL_TYPE = "fedpara" 
FEDPARA_RANK = 8       # Example rank for FedPara model

# Attack Configuration
NUM_MALICIOUS_CLIENTS = 3  # Example: first 3 clients are malicious
UAP_TARGET_CLASS_TO_PERTURB = 0 # e.g., 'plane'. Malicious clients will try to make model misclassify this class.

# Data Partitioning Strategies to Test
PARTITIONING_STRATEGIES = {
    "Homogeneous": {"fn": partition_data_homogeneous, "args": {}},
    "LabelQuantity_C1": {"fn": partition_data_label_quantity, "args": {"num_classes_per_client": 1}},
    "LabelQuantity_C2": {"fn": partition_data_label_quantity, "args": {"num_classes_per_client": 2}},
    "LabelQuantity_C3": {"fn": partition_data_label_quantity, "args": {"num_classes_per_client": 3}},
    "Dirichlet_Beta0.5": {"fn": partition_data_dirichlet, "args": {"beta": 0.5}},
}

SAVED_MODELS_PATH = "./saved_fl_models"
if not os.path.exists(SAVED_MODELS_PATH):
    os.makedirs(SAVED_MODELS_PATH)

def get_model_architecture_and_args():
    if MODEL_TYPE == "fedpara":
        model_fn = CIFAR10FedParaCNN
        model_args = {"num_classes": NUM_CLASSES, "fedpara_rank": FEDPARA_RANK}
    else: # standard
        model_fn = CIFAR10CNN
        model_args = {"num_classes": NUM_CLASSES}
    return model_fn, model_args

def run_federated_simulation(partition_name, partition_fn, partition_args):
    print(f"\n\n{'='*30} Starting Simulation for: {partition_name} {'='*30}")

    # 1. Load Data
    full_trainset, test_loader = load_cifar10_data(batch_size=BATCH_SIZE, data_fraction_overall=DATA_FRACTION_OVERALL)

    # 2. Partition Data
    client_dataloaders = partition_fn(full_trainset, NUM_CLIENTS, batch_size=BATCH_SIZE, **partition_args)
    
    # Check if any dataloader is completely empty and warn/handle
    if any(len(loader.dataset) == 0 for loader in client_dataloaders):
        print("Warning: One or more clients have no data after partitioning.")
        # Decide if simulation should proceed or skip if critical clients are empty.
        # For now, we'll let it proceed, Client class handles empty dataloaders.

    # 3. Initialize Server
    model_architecture_fn, model_initial_args = get_model_architecture_and_args()
    server = Server(
        global_model_architecture_fn=model_architecture_fn,
        model_args=model_initial_args,
        test_loader=test_loader,
        device=DEVICE
    )
    initial_global_model_state_for_attack_training = server.get_global_model_state()
    # Create a temporary model instance for attack training reference
    temp_model_for_attack_ref = model_architecture_fn(**model_initial_args)
    temp_model_for_attack_ref.load_state_dict(initial_global_model_state_for_attack_training)


    # 4. Initialize Clients
    clients = []
    for i in range(NUM_CLIENTS):
        attack_to_assign = NoAttack()
        if i < NUM_MALICIOUS_CLIENTS:
            print(f"Client {i} will be malicious, targeting class {UAP_TARGET_CLASS_TO_PERTURB}.")
            attack_to_assign = UAPAttack(target_label_to_perturb=UAP_TARGET_CLASS_TO_PERTURB)
        
        client = Client(
            client_id=i,
            local_data_loader=client_dataloaders[i] if i < len(client_dataloaders) else [], # Handle if fewer loaders than clients
            local_epochs=CLIENT_LOCAL_EPOCHS,
            learning_rate=LEARNING_RATE,
            device=DEVICE,
            attack_obj=attack_to_assign
        )
        # Train UAP attack generator if client is malicious (one-time before FL rounds)
        if client.is_malicious and client.attack_obj.requires_training:
             client.train_attack_if_needed(temp_model_for_attack_ref)
        clients.append(client)
    
    del temp_model_for_attack_ref
    if DEVICE.type == 'cuda': torch.cuda.empty_cache()

    # 5. Federated Learning Rounds
    server.evaluate_global_model(round_num=0) # Initial evaluation

    for r in range(1, NUM_ROUNDS + 1):
        print(f"\n--- Round {r}/{NUM_ROUNDS} ({partition_name}) ---")
        
        # Server sends global model to all clients (or selected ones if sampling)
        current_global_state = server.get_global_model_state()
        client_updates = []

        # For this setup, all 10 clients participate
        selected_clients_this_round = clients 

        for client in selected_clients_this_round:
            client.set_model(current_global_state, model_architecture_fn, model_initial_args) # Give client current global model
            updated_state = client.local_train()
            if updated_state: # Client might return None if it couldn't train
                client_updates.append(updated_state)
            
            if DEVICE.type == 'cuda': torch.cuda.empty_cache()
        
        # Server aggregates updates
        server.aggregate_updates(client_updates)
        
        # Server evaluates new global model
        server.evaluate_global_model(round_num=r)
        
        # Optional: Save model periodically
        if r % 10 == 0 or r == NUM_ROUNDS:
            server.save_final_model(SAVED_MODELS_PATH, f"global_model_{partition_name}_round_{r}.pth")

    print(f"\nFederated Training Finished for {partition_name}.")
    return server.get_round_log()


if __name__ == "__main__":
    all_experiment_logs = []
    all_experiment_titles = []

    # For class accuracy comparison and survival rate
    initial_class_acc_map = {} 
    final_class_acc_map = {}

    for name, config in PARTITIONING_STRATEGIES.items():
        training_log = run_federated_simulation(name, config["fn"], config["args"])
        all_experiment_logs.append(training_log)
        all_experiment_titles.append(f"{name} (Attack: {NUM_MALICIOUS_CLIENTS} UAP clients on class {UAP_TARGET_CLASS_TO_PERTURB})")
        
        if training_log:
            initial_class_acc_map[name] = training_log[0]['class_acc'] # Round 0
            final_class_acc_map[name] = training_log[-1]['class_acc']  # Last round

    # Plot overall accuracy comparison
    plot_accuracy_vs_rounds(all_experiment_logs, all_experiment_titles, 
                            overall_title=f"FL Accuracy ({MODEL_TYPE.capitalize()} Model, {NUM_MALICIOUS_CLIENTS} Malicious)")

    # Plot class accuracy for each partitioning strategy
    for name in PARTITIONING_STRATEGIES.keys():
        if name in initial_class_acc_map and name in final_class_acc_map:
            plot_class_accuracy_comparison(initial_class_acc_map[name], final_class_acc_map[name], name)
            
            # Calculate and print survival rate for the UAP target class
            survival = calculate_target_class_survival_rate(
                initial_class_acc_map[name],
                final_class_acc_map[name],
                UAP_TARGET_CLASS_TO_PERTURB
            )
            print(f"Survival rate for target class {CLASSES[UAP_TARGET_CLASS_TO_PERTURB]} under {name}: {survival:.2f}%")


    # Plot communication savings (FedPara vs Standard)
    standard_cnn = CIFAR10CNN(num_classes=NUM_CLASSES)
    fedpara_cnn = CIFAR10FedParaCNN(num_classes=NUM_CLASSES, fedpara_rank=FEDPARA_RANK)
    
    size_standard = get_model_parameters_size(standard_cnn)
    size_fedpara = get_model_parameters_size(fedpara_cnn)
    
    print(f"Standard CNN Size: {size_standard:.2f} MB")
    print(f"FedPara CNN (rank {FEDPARA_RANK}) Size: {size_fedpara:.2f} MB")
    plot_communication_savings(size_standard, size_fedpara)

    print("\nAll simulations finished.")