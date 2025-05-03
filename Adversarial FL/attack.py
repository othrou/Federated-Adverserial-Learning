import torch

class Attack:
    def __init__(self, model, targeted_class, malicious_weight_factor=5):
        self.model = model
        self.targeted_class = targeted_class
        self.malicious_weight_factor = malicious_weight_factor

    def poison(self):
        # Poison the model by adjusting weights to target a specific misclassification
        for param in self.model.parameters():
            param.data += torch.randn_like(param.data) * self.malicious_weight_factor
        return self.model.state_dict()
