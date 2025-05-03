import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

class Client:
    def __init__(self, model, train_data, batch_size=32, malicious=False):
        self.model = model
        self.train_data = train_data
        self.malicious = malicious
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()
        self.batch_size = batch_size

    def train(self, num_epochs=1):
        self.model.train()
        train_loader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)
        
        for epoch in range(num_epochs):
            for data, target in train_loader:
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()

    def get_update(self):
        # Return model weights update (difference from global model)
        return self.model.state_dict()

    def poison_update(self):
        # If malicious, add noise to the model's weights
        if self.malicious:
            for param in self.model.parameters():
                param.data += torch.randn_like(param.data) * 0.1  # Add noise for poisoning
        return self.get_update()
