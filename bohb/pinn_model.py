
###model.py
import torch
import torch.nn as nn
import numpy as np

class PINNNet(nn.Module):
    def __init__(self, layers, emb_dim, activation="tanh"):
        super(PINNNet, self).__init__()

        self.net = nn.ModuleList()

        for i in range(len(layers)-1):
            self.net.append(nn.Linear(layers[i], layers[i+1]))
            nn.init.xavier_normal_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

        # ✅ activation selection
        if activation == "tanh":
            self.act = torch.tanh
        elif activation == "sin":
            self.act = torch.sin
        elif activation == "softplus":
            self.act = nn.Softplus()
        else:
            raise ValueError("Unknown activation")

    def forward(self, x, t, emb):
        X = torch.cat([x, t, emb], dim=1)
        y = X

        for layer in self.net[:-1]:
            y = self.act(layer(y))   # ✅ changes here

        y = self.net[-1](y)
        return torch.sigmoid(y)

def pde_residual(model, x, t, emb, D, alpha, beta):
    """PDE residual exactly apke code jaisa"""
    x, t = x.clone().requires_grad_(True), t.clone().requires_grad_(True)
    u = model(x, t, emb)
    u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
    f = alpha * u * (1.0 - u) - beta * u
    return u_t - D * u_xx - f

print("✅ pinn_model.py created successfully!")