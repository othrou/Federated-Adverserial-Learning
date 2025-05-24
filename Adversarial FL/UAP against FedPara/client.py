import torch
import torch.optim as optim
import torch.nn as nn
import copy
from utils import generic_train_client_model, NoAttack, UAPAttack # Assuming utils.py

class Client:
    def __init__(self, client_id, local_data_loader, local_epochs, learning_rate, device, attack_obj=None):
        self.client_id = client_id
        self.local_data_loader = local_data_loader if local_data_loader else []
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        self.device = device
        self.model = None # Will be set by the server
        self.attack_obj = attack_obj if attack_obj else NoAttack()
        self.is_malicious = not isinstance(self.attack_obj, NoAttack)

        if self.is_malicious and self.attack_obj.requires_training:
            print(f"Client {self.client_id} is malicious and its attack requires training.")
            # Attack training needs a model context. This will be handled in simulation.py
            # after the initial global model is created and before FL starts.

    def set_model(self, global_model_state_dict, model_architecture_fn, model_args):
        """
        Sets or updates the client's local model from the global model state.
        model_architecture_fn: A function that returns a new model instance (e.g., CIFAR10FedParaCNN)
        model_args: A dictionary of arguments for the model_architecture_fn
        """
        # Create a new model instance of the correct architecture
        self.model = model_architecture_fn(**model_args).to(self.device)
        self.model.load_state_dict(copy.deepcopy(global_model_state_dict))

    def train_attack_if_needed(self, initial_global_model_for_attack_train):
        """
        Trains the attack generator if the client is malicious and the attack requires training.
        This should be called once before federated training starts if UAP is used.
        """
        if self.is_malicious and self.attack_obj.requires_training and not self.attack_obj.trained:
            print(f"Client {self.client_id}: Training its attack '{str(self.attack_obj)}'...")
            if not self.local_data_loader or len(self.local_data_loader.dataset) == 0:
                print(f"Client {self.client_id}: Cannot train attack, local data loader is empty.")
                self.attack_obj.trained = True # Mark as trained to avoid re-attempts
                return

            # Use a copy of the initial global model for attack training
            # This model state should ideally be what the attacker expects to attack
            model_for_attack = copy.deepcopy(initial_global_model_for_attack_train)
            model_for_attack.to(self.device).eval()
            
            self.attack_obj.train(model_for_attack, self.local_data_loader, self.device)
            del model_for_attack
            if self.device.type == 'cuda': torch.cuda.empty_cache()
            print(f"Client {self.client_id}: Attack training complete.")


    def local_train(self):
        """Performs local training and returns the updated model state_dict."""
        if not self.model:
            print(f"Client {self.client_id}: Model not set. Skipping training.")
            return None
        if not self.local_data_loader or len(self.local_data_loader.dataset) == 0:
            print(f"Client {self.client_id}: No local data. Skipping training.")
            # Return the current model state without training
            return copy.deepcopy(self.model.state_dict())

        self.model.train()
        optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9)
        criterion = nn.CrossEntropyLoss()

        print(f"  Client {self.client_id} ({'Malicious' if self.is_malicious else 'Benign'}) starting local training for {self.local_epochs} epochs...")
        
        generic_train_client_model(
            model=self.model,
            num_epochs=self.local_epochs,
            trainloader=self.local_data_loader,
            optimizer=optimizer,
            criterion=criterion,
            client_attack_obj=self.attack_obj,
            device=self.device,
            verbose_epoch=True # Can be made a parameter
        )
        print(f"  Client {self.client_id} local training finished.")
        return copy.deepcopy(self.model.state_dict())