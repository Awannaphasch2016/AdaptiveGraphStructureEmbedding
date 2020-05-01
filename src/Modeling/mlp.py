import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 2),
        )

    def forward(self, x):
        x = self.layers(x)
        return x


if __name__ == '__main__':
    model = MLP()
    model.train()
    pred = model(torch.ones(16))

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    model.train()
    optimizer.zero_grad()
    F.nll_loss(pred.view(2, 1).type(torch.float),
               torch.zeros(2).type(torch.long))
    optimizer.step()

