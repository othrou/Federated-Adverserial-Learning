import torch
import copy
import numpy as np
from utils import test_model_accuracy, NoDefense, save_model_state # Assuming utils.py

class Server:
    def __init__(self, global_model_architecture_fn, model_args, test_loader, device, defense_obj=None):
        """
        global_model_architecture_fn: A function that returns a new model instance (e.g., CIFAR10FedParaCNN)
        model_args: A dictionary of arguments for the model_architecture_fn
        """
        self.model_architecture_fn = global_model_architecture_fn
        self.model_args = model_args
        self.global_model = global_model_architecture_fn(**model_args).to(device)
        self.test_loader = test_loader
        self.device = device
        self.defense = defense_obj if defense_obj else NoDefense()
        self.round_log = []
        self.current_round = 0

    def get_global_model_state(self):
        return copy.deepcopy(self.global_model.state_dict())

    def aggregate_updates(self, client_updated_states):
        if not client_updated_states:
            print("Warning: No client updates to aggregate. Global model unchanged.")
            return

        # Apply defense mechanism
        # The defense mechanism might filter or modify updates
        processed_updates = self.defense.run(self.get_global_model_state(), client_updated_states)

        if not processed_updates:
            print("Warning: Defense resulted in no updates for aggregation. Global model unchanged.")
            return

        # Standard FedAvg aggregation
        global_state_dict = self.global_model.state_dict()
        avg_state_dict = copy.deepcopy(global_state_dict)
        
        # Zero out numerically relevant parts of avg_state_dict for summation
        for key in avg_state_dict:
            if avg_state_dict[key].dtype.is_floating_point:
                avg_state_dict[key].zero_()
            # For non-floating point (e.g. num_batches_tracked in BatchNorm), keep first client's or handle appropriately
            # Simple FedAvg often just averages weights, biases. Buffers are sometimes tricky.
            # For FedPara, X1,Y1,X2,Y2 and bias are parameters.
        
        num_contrib_clients = 0
        for client_state in processed_updates:
            if client_state is None: continue # Skip if a client update was nullified
            for key in global_state_dict:
                if key in client_state and global_state_dict[key].dtype.is_floating_point:
                    avg_state_dict[key].data += client_state[key].data.to(self.device) # Accumulate on device
                elif key in client_state and not global_state_dict[key].dtype.is_floating_point and num_contrib_clients == 0:
                    # For non-float params like num_batches_tracked, take from first client
                    avg_state_dict[key].data.copy_(client_state[key].data)
            num_contrib_clients +=1
        
        if num_contrib_clients > 0:
            for key in avg_state_dict:
                if avg_state_dict[key].dtype.is_floating_point:
                    avg_state_dict[key].data /= num_contrib_clients
            self.global_model.load_state_dict(avg_state_dict)
            print(f"Global model aggregated from {num_contrib_clients} client updates.")
        else:
            print("No valid client updates to aggregate after defense. Global model unchanged.")


    def evaluate_global_model(self, round_num):
        total_acc, class_acc = test_model_accuracy(self.global_model, self.test_loader, self.device)
        self.round_log.append({'round': round_num, 'global_acc': total_acc, 'class_acc': class_acc.tolist()})
        print(f"Round {round_num} - Global Model Accuracy: {total_acc:.2f}%")
        # print(f"Round {round_num} - Class Accuracies: {[f'{c:.2f}%' for c in class_acc]}")
        return total_acc, class_acc

    def save_final_model(self, path, filename="final_global_model.pth"):
        save_model_state(self.global_model, path, filename)

    def get_round_log(self):
        return self.round_log