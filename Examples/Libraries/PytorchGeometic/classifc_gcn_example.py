import os.path as osp
import argparse

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, ChebConv  # noqa

parser = argparse.ArgumentParser()
parser.add_argument('--use_gdc', action='store_true',
                    help='Use GDC preprocessing.')
args = parser.parse_args()

dataset = 'Cora'
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', '..', 'Data', 'External')
dataset = Planetoid(path, dataset, T.NormalizeFeatures())
data = dataset[0]

if args.use_gdc:
    gdc = T.GDC(self_loop_weight=1, normalization_in='sym',
                normalization_out='col',
                diffusion_kwargs=dict(method='ppr', alpha=0.05),
                sparsification_kwargs=dict(method='topk', k=128,
                                           dim=0), exact=True)
    data = gdc(data)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(dataset.num_features, 16, cached=True,
                             normalize=not args.use_gdc)
        self.conv2 = GCNConv(16, dataset.num_classes, cached=True,
                             normalize=not args.use_gdc)
        # self.conv1 = ChebConv(data.num_features, 16, K=2)
        # self.conv2 = ChebConv(16, data.num_features, K=2)

        self.reg_params = self.conv1.parameters()
        self.non_reg_params = self.conv2.parameters()

    def forward(self):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return F.log_softmax(x, dim=1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, data = Net().to(device), data.to(device)
optimizer = torch.optim.Adam([
    dict(params=model.reg_params, weight_decay=5e-4),
    dict(params=model.non_reg_params, weight_decay=0)
], lr=0.01)


def train():
    model.train()
    optimizer.zero_grad()
    logits = model()
    x = F.nll_loss(model()[data.train_mask], data.y[data.train_mask])
    x.backward()
    optimizer.step()
    return logits, x


@torch.no_grad()
def test():
    model.eval()
    logits, accs = model(), []
    x = F.nll_loss(logits[data.test_mask], data.y[data.test_mask])
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs, x


best_val_acc = test_acc = 0
loss_dict = {}
for epoch in range(1, 5000):
    logits, train_loss = train()
    loss_dict.setdefault('train_loss',[]).append(train_loss)
    (train_acc, val_acc, tmp_test_acc), test_loss = test()
    loss_dict.setdefault('test_loss',[]).append(test_loss)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        test_acc = tmp_test_acc
    log = 'Epoch: {:03d}, Train: {:.4f}, Train loss: {:.4f},Val: {:.4f}, Test: {:.4f}, Test_loss: {:.4f}'
    print(log.format(epoch, train_acc, train_loss,best_val_acc, test_acc, test_loss))

import matplotlib.pyplot as plt
plt.plot(loss_dict['train_loss'], label='train_loss')
plt.plot(loss_dict['test_loss'], label='test_loss')
plt.legend()
plt.show()
