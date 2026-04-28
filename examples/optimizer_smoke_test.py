"""Minimal CPU smoke test for the paper-facing optimizers."""

import torch

from adam_imbalance import AdamLDN, AdamS, AdamSLDN


def run_optimizer(optimizer_cls):
    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 8),
        torch.nn.ReLU(),
        torch.nn.Linear(8, 2),
    )
    optimizer = optimizer_cls(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    x = torch.randn(16, 4)
    y = torch.randint(0, 2, (16,))

    losses = []
    for _ in range(3):
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    return losses


if __name__ == "__main__":
    for optimizer_cls in (AdamLDN, AdamS, AdamSLDN):
        print(optimizer_cls.__name__, run_optimizer(optimizer_cls))
