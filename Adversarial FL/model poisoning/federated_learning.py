import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from client import Client
from server import Server
from attack import Attack
from torch.utils.data import DataLoader, random_split, Subset


# Define the model architecture (simple CNN)
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64*7*7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load the Fashion-MNIST dataset
def load_data():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    # Load the training data
    train_data = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)

    # Split the dataset into 5 parts, one for each client
    num_clients = 5
    client_data = []
    client_size = len(train_data) // num_clients

    for i in range(num_clients):
        subset_indices = list(range(i * client_size, (i + 1) * client_size))
        client_data.append(Subset(train_data, subset_indices))

    return client_data

# Running the federated learning with model poisoning
def run_federated_learning():
    # Initialize global model and server
    global_model = CNNModel()
    server = Server(global_model)

    # Load data and create clients
    client_data = load_data()
    clients = [Client(global_model, client_data[i], malicious=(i == 4)) for i in range(5)]  # Malicious client is the last one

    client_weights = [1 for _ in range(5)]  # Equal weight for each client
    updates = []

    # Federated learning rounds
    for round in range(10):
        updates = []
        for client in clients:
            if client.malicious:
                updates.append(client.poison_update())  # Poisoned update for malicious client
            else:
                updates.append(client.get_update())  # Normal update from benign clients

        # Aggregate the updates on the server
        server.aggregate_updates(updates, client_weights)

        # Optionally, print the model's state to track poisoning effects
        print(f"Round {round + 1}: Global model updated")

run_federated_learning()
