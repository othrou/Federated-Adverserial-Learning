from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import numpy as np
import torchvision.datasets as datasets
from torch.utils.data import random_split, Subset
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import copy
import os

# --- Global Settings ---
torch.manual_seed(42)
np.random.seed(42)
CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
NUM_CLASSES = len(CLASSES)

# --- Helper functions for state_dict comparison ---
def compare_state_dicts(sd1, sd2):
    if sd1.keys() != sd2.keys():
        return False
    for key in sd1:
        if key not in sd2:
            return False
        val1, val2 = sd1[key], sd2[key]
        if isinstance(val1, torch.Tensor) and isinstance(val2, torch.Tensor):
            if not torch.equal(val1.cpu(), val2.cpu()):
                return False
        elif type(val1) != type(val2) or val1 != val2:
            return False
    return True

def find_state_dict_index_in_list(list_of_state_dicts, target_state_dict):
    for i, current_s_dict in enumerate(list_of_state_dicts):
        if compare_state_dicts(current_s_dict, target_state_dict):
            return i
    raise ValueError("State dictionary not found in the list.")

# --- Attacks ---
class Attack:
    def __init__(self):
        self.requires_training = False

    def train(self, target_model, dataloader, device):
        return self

    def apply(self, inputs, labels, model, criterion, device):
        return self.run(inputs, labels, device)

    def run(self, inputs, labels, device):
        raise NotImplementedError("Each attack must implement 'run' or 'apply'")

    def __repr__(self):
        return f"(attack={self.__class__.__name__})"

class NoAttack(Attack):
    def __init__(self):
        super(NoAttack, self).__init__()

    def run(self, inputs, labels, device):
        return inputs, labels

class UAP(nn.Module):
    def __init__(self, num_channels=3, img_height=32, img_width=32, initial_magnitude=0.1):
        super(UAP, self).__init__()
        self.perturbation = nn.Parameter(torch.zeros(1, num_channels, img_height, img_width), requires_grad=True)
        nn.init.uniform_(self.perturbation, -initial_magnitude, initial_magnitude)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(0)
        return torch.clamp(x + self.perturbation, -1.0, 1.0) # CIFAR-10 images are normalized to [-1, 1]

def uap_train_routine(dataloader, uap_generator, target_model, device, epochs=5, lr=0.01, target_class_for_uap_objective=0, fooling_target=True):
    print(f"Starting UAP training for {epochs} epochs. Target class for UAP objective: {target_class_for_uap_objective}, Fooling target: {fooling_target}")
    optimizer = optim.Adam(uap_generator.parameters(), lr=lr)
    target_model.eval()
    uap_generator.train()
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        epoch_loss_sum = 0.0
        batches = 0
        if not dataloader or len(dataloader) == 0:
            print(f"UAP Train Epoch {epoch+1}/{epochs}: Dataloader is empty. Skipping.")
            continue
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            perturbed_inputs = uap_generator(inputs)
            outputs = target_model(perturbed_inputs)

            if fooling_target:
                target_labels = torch.full_like(labels, target_class_for_uap_objective, device=device)
                loss = criterion(outputs, target_labels)
            else: # Misclassification (untargeted)
                loss = -criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            epoch_loss_sum += loss.item()
            batches += 1
        if batches > 0:
            print(f"UAP Train Epoch {epoch+1}/{epochs}, Avg Loss: {epoch_loss_sum/batches:.4f}")
        else:
            print(f"UAP Train Epoch {epoch+1}/{epochs}, No data processed.")
    uap_generator.eval()
    print("UAP training finished.")
    return uap_generator # Return the trained generator

class UAPAttack(Attack):
    def __init__(self, target_label_to_perturb, img_height=32, img_width=32, num_channels=3):
        super(UAPAttack, self).__init__()
        self.target_label_to_perturb = target_label_to_perturb
        self.img_height = img_height
        self.img_width = img_width
        self.num_channels = num_channels
        self.generator = UAP(num_channels, img_height, img_width)
        self.trained = False
        self.requires_training = True

    def train(self, target_model, dataloader, device):
        if not dataloader or len(dataloader.dataset) == 0:
            print(f"Warning: Dataloader for UAP training (class {self.target_label_to_perturb}) is empty. Skipping training.")
            self.trained = True
            return self

        self.generator.to(device)
        target_model.to(device).eval()
        
        print(f"Training UAP generator to make model misclassify images of true class {self.target_label_to_perturb}...")

        # Filter dataloader for the target class to perturb
        original_dataset = dataloader.dataset
        indices_of_target_class = [i for i, (_, label) in enumerate(original_dataset) if label == self.target_label_to_perturb]

        if not indices_of_target_class:
            print(f"Warning: No images of class {self.target_label_to_perturb} found for UAP training. Skipping.")
            self.trained = True
            return self

        uap_training_data_subset = Subset(original_dataset, indices_of_target_class)
        # Use a smaller batch size for UAP training if the subset is small
        uap_batch_size = min(dataloader.batch_size if dataloader.batch_size else 32, len(uap_training_data_subset))
        if uap_batch_size == 0:
            print(f"Warning: UAP training subset for class {self.target_label_to_perturb} is empty. Skipping UAP training.")
            self.trained = True
            return self

        uap_train_loader = DataLoader(uap_training_data_subset, batch_size=uap_batch_size, shuffle=True)

        # Define UAP objective: misclassify to any other class (e.g., target class 0 if original is not 0, else 1)
        uap_objective_target_class = (self.target_label_to_perturb + 1) % NUM_CLASSES # Simple choice

        self.generator = uap_train_routine(
            uap_train_loader, self.generator, target_model, device,
            epochs=5, lr=0.005, target_class_for_uap_objective=uap_objective_target_class, fooling_target=True
        )
        self.trained = True
        print(f"UAP generator training finished for perturbing class {self.target_label_to_perturb}.")
        return self

    def run(self, inputs, labels, device):
        if not self.trained:
            print("Warning: UAPAttack used without prior training. Perturbation will be random/zero.")
        
        self.generator.to(device)
        perturbed_inputs = inputs.clone()
        for k in range(inputs.size(0)):
            if labels[k] == self.target_label_to_perturb:
                single_input = inputs[k].unsqueeze(0).to(device)
                perturbed_single = self.generator(single_input).squeeze(0)
                perturbed_inputs[k] = perturbed_single.detach()
        return perturbed_inputs, labels

    def __repr__(self):
        return f"(attack=UAPAttack, target_label_to_perturb={self.target_label_to_perturb})"

# --- FedPara Layer ---
class FedParaLinear(nn.Module):
    def __init__(self, in_features, out_features, rank_r, use_bias=True, non_linearity_fn_str=None):
        super(FedParaLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank_r = rank_r
        self.use_bias = use_bias

        if non_linearity_fn_str == "tanh":
            self.non_linearity_fn = torch.tanh
        elif non_linearity_fn_str == "relu":
            self.non_linearity_fn = torch.relu
        else: # None or other strings mean no non-linearity on W1, W2
            self.non_linearity_fn = None

        self.X1 = nn.Parameter(torch.Tensor(out_features, rank_r))
        self.Y1 = nn.Parameter(torch.Tensor(in_features, rank_r))
        self.X2 = nn.Parameter(torch.Tensor(out_features, rank_r))
        self.Y2 = nn.Parameter(torch.Tensor(in_features, rank_r))

        if use_bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.X1, a=np.sqrt(5))
        nn.init.kaiming_uniform_(self.Y1, a=np.sqrt(5))
        nn.init.kaiming_uniform_(self.X2, a=np.sqrt(5))
        nn.init.kaiming_uniform_(self.Y2, a=np.sqrt(5))
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def get_weight(self):
        W1 = torch.matmul(self.X1, self.Y1.t())
        W2 = torch.matmul(self.X2, self.Y2.t())
        if self.non_linearity_fn:
            W1 = self.non_linearity_fn(W1)
            W2 = self.non_linearity_fn(W2)
        return W1 * W2 # Hadamard product

    def forward(self, x):
        weight = self.get_weight()
        return F.linear(x, weight, self.bias)

    def __repr__(self):
        return f'FedParaLinear(in_features={self.in_features}, out_features={self.out_features}, rank_r={self.rank_r}, bias={self.bias is not None})'

# --- Models for CIFAR-10 ---
class CIFAR10CNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(CIFAR10CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=2)
        self.fc1 = nn.Linear(32 * 8 * 8, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class CIFAR10FedParaCNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, fedpara_rank=10):
        super(CIFAR10FedParaCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, padding=2) # Non-FedPara conv
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=2) # Non-FedPara conv

        # FedPara for FC layers as in the paper's spirit (or convert conv to FedPara based on Prop 3 if needed)
        # For simplicity, applying to FC layers first.
        # Ranks can be tuned. The paper suggests r1=r2=R and R^2 >= min(m,n) for full rank capability [cite: 73, 90]
        # For FC1: m=120, n=32*8*8=2048. R^2 >= 120 => R>=11.
        # For FC2: m=84, n=120. R^2 >= 84 => R>=10.
        # For FC3: m=num_classes, n=84. R^2 >= num_classes => R>=4 for num_classes=10.
        # Using user-provided fedpara_rank as a base.
        rank_fc1 = fedpara_rank
        rank_fc2 = max(5, fedpara_rank // 2 if fedpara_rank // 2 > 0 else 5) # ensure rank > 0
        rank_fc3 = max(3, fedpara_rank // 3 if fedpara_rank // 3 > 0 else 3) # ensure rank > 0

        self.fc1 = FedParaLinear(32 * 8 * 8, 120, rank_r=rank_fc1, non_linearity_fn_str="tanh")
        self.fc2 = FedParaLinear(120, 84, rank_r=rank_fc2, non_linearity_fn_str="tanh")
        self.fc3 = FedParaLinear(84, num_classes, rank_r=rank_fc3, non_linearity_fn_str=None) # No non-lin on last layer's weights

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x)) # Activation on output of linear layer
        x = F.relu(self.fc2(x)) # Activation on output of linear layer
        x = self.fc3(x)
        return x

# --- Training and Testing Utilities ---
def generic_train_client_model(model, num_epochs, trainloader, optimizer, criterion, client_attack_obj, device, verbose_epoch=False):
    train_losses = []
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        num_batches = 0
        if not trainloader or len(trainloader) == 0:
            if verbose_epoch: print(f"      Client Epoch {epoch+1}/{num_epochs}, Loss: NaN (empty dataloader)")
            train_losses.append(float('nan'))
            continue

        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Apply attack if any
            inputs_after_attack, labels_after_attack = client_attack_obj.apply(inputs, labels, model, criterion, device)

            optimizer.zero_grad()
            outputs = model(inputs_after_attack)
            loss = criterion(outputs, labels_after_attack) # Use labels after attack if they changed
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            num_batches += 1
        
        if num_batches > 0:
            epoch_loss = running_loss / num_batches
            train_losses.append(epoch_loss)
            if verbose_epoch:
                print(f"      Client Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")
        else:
            train_losses.append(float('nan'))
            if verbose_epoch:
                print(f"      Client Epoch {epoch+1}/{num_epochs}, Loss: NaN (no data processed)")
    return train_losses

def test_model_accuracy(model, testloader, device, num_classes=NUM_CLASSES):
    model.eval()
    correct = 0
    total = 0
    class_correct = list(0. for _ in range(num_classes))
    class_total = list(0. for _ in range(num_classes))

    if not testloader or len(testloader.dataset) == 0:
        return 0.0, np.array([0.] * num_classes)

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # For class accuracy
            c = (predicted == labels).squeeze()
            for i in range(labels.size(0)):
                label = labels[i].item()
                if label < num_classes: # handles if a batch has labels outside expected range
                    class_correct[label] += c[i].item() if c.numel() > 1 else c.item() # handle batch size 1
                    class_total[label] += 1
    
    total_acc = 100 * correct / total if total > 0 else 0.0
    
    class_accuracies = np.zeros(num_classes)
    for i in range(num_classes):
        if class_total[i] > 0:
            class_accuracies[i] = 100 * class_correct[i] / class_total[i]
        else:
            class_accuracies[i] = 0.0
            
    return total_acc, class_accuracies

def save_model_state(model, path, filename="model.pth"):
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(model.state_dict(), os.path.join(path, filename))
    print(f"Model state saved to {os.path.join(path, filename)}")

# --- Defense (Placeholder) ---
class NoDefense:
    def run(self, global_model_state, client_updated_states):
        # No modification to client updates
        return client_updated_states

    def __str__(self):
        return "NoDefense"

def get_model_parameters_size(model):
    """Calculates model size in MB."""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    return param_size / (1024**2)