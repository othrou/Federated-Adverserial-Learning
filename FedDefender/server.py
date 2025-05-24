import torch
import copy
import numpy as np

class Server:
    def __init__(self, global_model, client_list, test_loader, device='cpu'):
        """
        Initialize the Federated Learning server
        """
        self.global_model = global_model
        self.client_list = client_list
        self.test_loader = test_loader
        self.device = device
        self.round = 0
        self.criterion = torch.nn.CrossEntropyLoss()
        self.metrics = {'round': [], 'accuracy': [], 'loss': []}
    
    def aggregate(self, client_models, client_stats):
        """
        Aggregate updated models from clients (FedAvg)
        """
        total_samples = sum(stats['train_samples'] for stats in client_stats)
        global_dict = copy.deepcopy(self.global_model.state_dict())
        for k in global_dict.keys():
            global_dict[k] = torch.zeros_like(global_dict[k])
        
        for client_idx, client_dict in enumerate(client_models):
            weight = client_stats[client_idx]['train_samples'] / total_samples
            for k in global_dict.keys():
                global_dict[k] += client_dict[k] * weight
        
        return global_dict

    def evaluate(self):
        """
        Evaluate the global model on the test dataset
        """
        self.global_model.eval()
        test_loss = 0
        correct = 0
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.global_model(data)
                test_loss += self.criterion(output, target).item() * data.size(0)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        
        test_loss /= len(self.test_loader.dataset)
        accuracy = 100. * correct / len(self.test_loader.dataset)
        
        print(f"\nGlobal Model - Test set: Avg. loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%\n")
        
        self.metrics['round'].append(self.round)
        self.metrics['accuracy'].append(accuracy)
        self.metrics['loss'].append(test_loss)
        
        return accuracy, test_loss
    
    def train_round(self, selected_clients=None):
        """
        Perform one round of federated learning
        """
        self.round += 1
        print(f"\n--- Round {self.round} ---")
        
        if selected_clients is None:
            selected_clients = list(range(len(self.client_list)))
        
        global_model_dict = copy.deepcopy(self.global_model.state_dict())
        client_models = []
        client_stats = []
        
        for idx in selected_clients:
            client = self.client_list[idx]
            local_model_dict, stats = client.local_update(global_model_dict)
            client_models.append(local_model_dict)
            client_stats.append(stats)
        
        updated_global_dict = self.aggregate(client_models, client_stats)
        self.global_model.load_state_dict(updated_global_dict)
        self.evaluate()
        
        return self.global_model

    def set_weights(self, new_weights):
        """
        Set new weights for the global model (used by defense strategies).
        
        Args:
            new_weights: A state_dict representing the new global model weights
        """
        self.global_model.load_state_dict(new_weights)
        self.global_model.to(self.device)
