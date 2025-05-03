import torch
import torch.nn as nn

class Server:
    def __init__(self, model):
        self.model = model

    def aggregate_updates(self, updates, client_weights):
        # Aggregate updates using weighted average (simple federated averaging)
        aggregated_update = {}
        for key in self.model.state_dict().keys():
            aggregated_update[key] = torch.zeros_like(self.model.state_dict()[key])

        total_weight = sum(client_weights)
        for i, update in enumerate(updates):
            for key in aggregated_update.keys():
                aggregated_update[key] += client_weights[i] * update[key]

        # Update global model weights
        for key in self.model.state_dict().keys():
            self.model.state_dict()[key].data = aggregated_update[key] / total_weight
