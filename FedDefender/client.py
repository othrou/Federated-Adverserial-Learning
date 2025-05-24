import torch
import torch.nn as nn
import torch.optim as optim
import copy

class Client:
    def __init__(self, client_id, local_model, dataloader,
                 local_epochs=10, device='cpu',
                 is_malicious=False, attack_type='label_flipping'):
        """
        Initialize a Federated Learning client
        Args:
            client_id: Unique identifier for the client
            local_model: PyTorch model for local training
            dataloader: DataLoader with client's local data
            local_epochs: Number of epochs for local training
            device: Device to run computations on ('cpu' or 'cuda')
            is_malicious: Whether this client performs an attack
            attack_type: Type of attack ('label_flipping', 'sign_flipping', 'gaussian_noise')
        """
        self.client_id = client_id
        self.local_model = local_model.to(device)
        self.dataloader = dataloader
        self.local_epochs = local_epochs
        self.device = device
        self.is_malicious = is_malicious
        self.attack_type = attack_type
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.local_model.parameters(), lr=0.01, momentum=0.9)

    def local_update(self, global_model=None):
        """
        Update local model from global model and perform local training
        Returns updated model state_dict and stats
        """
        # Load global model weights
        if global_model is not None:
            self.local_model.load_state_dict(copy.deepcopy(global_model))

        # Local training
        self.local_model.train()
        avg_loss = None
        for epoch in range(self.local_epochs):
            epoch_loss = 0.0
            for data, target in self.dataloader:
                data, target = data.to(self.device), target.to(self.device)

                # Label flipping attack at data level
                if self.is_malicious and self.attack_type == 'label_flipping':
                    target = (9 - target)

                self.optimizer.zero_grad()
                output = self.local_model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
            avg_loss = epoch_loss / len(self.dataloader)
            print(f"Client {self.client_id}, Local Epoch {epoch+1}/{self.local_epochs}, Loss: {avg_loss:.6f}")

        # After training, prepare the state_dict for return
        new_state = copy.deepcopy(self.local_model.state_dict())

        # Sign flipping attack on the update delta
        if self.is_malicious and self.attack_type == 'sign_flipping':
            flipped = {}
            for k in new_state.keys():
                delta = new_state[k] - global_model[k]
                flipped[k] = global_model[k] - delta
            new_state = flipped

        # Gaussian noise attack on weights
        if self.is_malicious and self.attack_type == 'gaussian_noise':
            for k in new_state.keys():
                noise = torch.randn_like(new_state[k]) * 0.1
                new_state[k] = new_state[k] + noise

        stats = {
            'client_id': self.client_id,
            'train_samples': len(self.dataloader.dataset),
            'loss': avg_loss
        }
        return new_state, stats

    def evaluate(self, test_loader):
        """
        Evaluate the local model on test data
        """
        self.local_model.eval()
        test_loss = 0.0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.local_model(data)
                test_loss += self.criterion(output, target).item() * data.size(0)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(test_loader.dataset)
        accuracy = 100.0 * correct / len(test_loader.dataset)
        return accuracy, test_loss
