import torch
from defence_utils import IRLS_aggregation_split_restricted
from sklearn.ensemble import IsolationForest
import numpy as np
import matplotlib.pyplot as plt

# FedPara Compression Utility
def fedpara_trainable_compress(delta_w, rank=8, C=None, steps=1000, lr=1e-2):
    if delta_w.dim() != 2:
        raise ValueError("Only supports 2D tensors")
    m, n = delta_w.shape
    device = delta_w.device

    if C is None:
        C = torch.ones(m, n, device=device)

    A = torch.randn(m, rank, device=device).mul_(0.01).requires_grad_()
    B = torch.randn(n, rank, device=device).mul_(0.01).requires_grad_()

    optimizer = torch.optim.Adam([A, B], lr=lr)

    for step in range(steps):
        optimizer.zero_grad()
        approx = (A @ B.T) * C
        loss = torch.nn.functional.mse_loss(approx, delta_w)
        loss.backward()
        optimizer.step()

    W_approx = (A @ B.T) * C
    return W_approx, A.detach(), B.detach()

# Helper functions
def vectorize_state_dict(state_dict, device=None):
    vecs = []
    for v in state_dict.values():
        t = v.flatten()
        if device:
            t = t.to(device)
        vecs.append(t)
    return torch.cat(vecs)

def unvectorize_state_dict(vec, template_state_dict, device=None):
    new_dict, pointer = {}, 0
    for k, v in template_state_dict.items():
        numel = v.numel()
        chunk = vec[pointer:pointer + numel]
        if device:
            chunk = chunk.to(device)
        new_dict[k] = chunk.view(v.size())
        pointer += numel
    return new_dict

class FedParaIRLSStrategy:
    def __init__(self, server, clients,
                 num_rounds, client_sampling_rate,
                 fedpara_rank=8, fedpara_steps=500, fedpara_lr=1e-2,
                 irls_lambda=2.0, irls_thresh=0.1):
        self.server = server
        self.clients = clients
        self.num_rounds = num_rounds
        self.client_sampling_rate = client_sampling_rate
        self.rank = fedpara_rank
        self.steps = fedpara_steps
        self.lr = fedpara_lr
        self.irls_lambda = irls_lambda
        self.irls_thresh = irls_thresh

    def detect_anomalies(self, local_vecs, contamination=0.1):
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        iso_forest.fit(local_vecs)
        predictions = iso_forest.predict(local_vecs)
        suspect_clients = np.where(predictions == -1)[0]
        print(f"Clients suspects détectés : {suspect_clients}")
        return predictions, suspect_clients
 

    def train(self):
        rounds, accuracies, losses = [], [], []
        device = self.server.device

        for rnd in range(1, self.num_rounds + 1):
            print(f"\n--- Round {rnd}/{self.num_rounds} ---")

            global_dict = self.server.global_model.state_dict()
            global_vec = vectorize_state_dict(global_dict, device)

            local_dicts = []
            for client in self.clients:
                local_state, _ = client.local_update(global_dict)
                local_dicts.append(local_state)

            local_vecs = torch.stack([
                vectorize_state_dict(ld, device) for ld in local_dicts
            ]).cpu().numpy()

            predictions, suspect_clients = self.detect_anomalies(local_vecs)
            

            valid_indices = np.where(predictions == 1)[0]
            valid_local_dicts = [local_dicts[i] for i in valid_indices]

            valid_local_vecs = torch.stack([
                vectorize_state_dict(ld, device) for ld in valid_local_dicts
            ])

            delta_matrix = valid_local_vecs - global_vec.unsqueeze(0)

            W_approx, A, B = fedpara_trainable_compress(
                delta_matrix, rank=self.rank, steps=self.steps, lr=self.lr
            )

            compressed_dicts = []
            for i in range(W_approx.size(0)):
                vec = global_vec + W_approx[i]
                compressed_dicts.append(
                    unvectorize_state_dict(vec, global_dict, device)
                )

            compressed_map = {i: compressed_dicts[i] for i in range(len(compressed_dicts))}

            global_weights, _ = IRLS_aggregation_split_restricted(
                compressed_map, LAMBDA=self.irls_lambda, thresh=self.irls_thresh
            )

            self.server.set_weights(global_weights)
            acc, loss = self.server.evaluate()

            rounds.append(rnd)
            accuracies.append(acc)
            losses.append(loss)

        history = {'round': rounds, 'accuracy': accuracies, 'loss': losses}
        final_model = self.server.global_model.state_dict()
        return history, final_model
