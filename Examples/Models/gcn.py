import argparse
import os.path as osp

# import dgl
# from dgl.nn.pytorch import GraphConv

import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv  # noqa

from Examples.Libraries.PytorchGeometic.InMemoryDataset_class import *


# f = r'C:\Users\Anak\PycharmProjects\AdaptiveGraphStructureEmbedding\Data\Preprocessed\Cora\cora_percent=0_noise=0.adjlist'
# f = r'C:\Users\Anak\PycharmProjects\AdaptiveGraphStructureEmbedding\Data\Preprocessed\Cora\cora_added_edges_same_class_nodes_percent=0.001_noise=0.adjlist'
# f = r'C:\Users\Anak\PycharmProjects\AdaptiveGraphStructureEmbedding\Data\Preprocessed\Cora\cora_added_edges_same_class_nodes_percent=0.001_noise=0.0005.adjlist'
# f = r'C:\Users\Anak\PycharmProjects\AdaptiveGraphStructureEmbedding\Data\Preprocessed\Cora\cora_added_edges_same_class_nodes_percent=0.001_noise=0.001.adjlist'
# f = r'C:\Users\Anak\PycharmProjects\AdaptiveGraphStructureEmbedding\Data\Preprocessed\Cora\cora_added_edges_same_class_nodes_percent=0.001_noise=0.002.adjlist'
# f = r'C:\Users\Anak\PycharmProjects\AdaptiveGraphStructureEmbedding\Data\Preprocessed\Cora\cora_added_edges_same_class_nodes_percent=0.001_noise=0.003.adjlist'
# f = r'C:\Users\Anak\PycharmProjects\AdaptiveGraphStructureEmbedding\Data\Preprocessed\Cora\cora_added_edges_same_class_nodes_percent=0.005_noise=0.adjlist'
# f = r'C:\Users\Anak\PycharmProjects\AdaptiveGraphStructureEmbedding\Data\Preprocessed\Cora\cora_added_edges_same_class_nodes_percent=0.001_noise=0.005.adjlist'
# f = r'C:\Users\Anak\PycharmProjects\AdaptiveGraphStructureEmbedding\Data\Preprocessed\Cora\cora_added_edges_same_class_nodes_percent=0.08_noise=0.08.adjlist'
# f = r'C:\Users\Anak\PycharmProjects\AdaptiveGraphStructureEmbedding\Data\Preprocessed\Cora\cora_added_edges_same_class_nodes_percent=0.2_noise=0.adjlist'
# f = r'C:\Users\Anak\PycharmProjects\AdaptiveGraphStructureEmbedding\Data\Preprocessed\Cora\cora_added_edges_same_class_nodes_percent=0.2_noise=0.2.adjlist'
# f = r'C:\Users\Anak\PycharmProjects\AdaptiveGraphStructureEmbedding\Data\Preprocessed\Cora\cora_added_edges_same_class_nodes_percent=0.5_noise=0.adjlist'
# f = r'C:\Users\Anak\PycharmProjects\AdaptiveGraphStructureEmbedding\Data\Preprocessed\Cora\cora_added_edges_same_class_nodes_percent=0.5_noise=0.5.adjlist'
# f = r'C:\Users\Anak\PycharmProjects\AdaptiveGraphStructureEmbedding\Data\Preprocessed\Cora\cora_added_edges_same_class_nodes_percent=0.8_noise=0.adjlist'
# f = r'C:\Users\Anak\PycharmProjects\AdaptiveGraphStructureEmbedding\Data\Preprocessed\Cora\cora_added_edges_same_class_nodes_percent=0.8_noise=0.8.adjlist'
# f = r'C:\Users\Anak\PycharmProjects\AdaptiveGraphStructureEmbedding\Data\Preprocessed\Cora\cora_added_edges_same_class_nodes_percent=1_noise=0.adjlist'


def convert_data(f):
    if f.split('.')[-1] == 'adjlist':
        # pass
        # elif f.split('.')[-1] == 'embeddings':
        # TODO here>> figure out why file that is pass into process is changed to byte file
        G = process(f)
        import torch
        edge_index = []
        x = []
        for i, j in G.items():
            for n in j:
                edge_index.append([i, n])
            x.append(i)
        # TODO currently it cannot yet accept node features.
        data.edge_index = torch.tensor(edge_index).transpose(1, 0)
    else:
        raise ValueError('no')



class Net(torch.nn.Module):
    def __init__(self, data, dataset):
        super(Net, self).__init__()
        self.conv1 = GCNConv(dataset.num_features, 16, cached=True,
                             normalize=not args.use_gdc)
        self.conv2 = GCNConv(16, dataset.num_classes, cached=True,
                             normalize=not args.use_gdc)
        # self.conv1 = ChebConv(data.num_features, 16, K=2)
        # self.conv2 = ChebConv(16, data.num_features, K=2)

        self.reg_params = self.conv1.parameters()
        self.non_reg_params = self.conv2.parameters()
        self.embedded_x = None

    def forward(self):
        # TODO preprocess .adjlist so that it output the following: x, edge_index, edge_weight.
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr

        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        self.embedded_x = x
        x = F.log_softmax(x, dim=1)
        return x
        # return F.log_softmax(x, dim=1)


def train(model, optimizer):
    model.train()
    optimizer.zero_grad()
    F.nll_loss(model()[data.train_mask], data.y[data.train_mask]).backward()
    optimizer.step()
    return model.embedded_x


@torch.no_grad()
def test(model):
    model.eval()
    logits, accs = model(), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs


def run_gcn(model, optimizer):
    best_val_acc = test_acc = 0
    performance_history = {}
    for epoch in range(1, 201):
        embedded_x = train(model, optimizer)
        train_acc, val_acc, tmp_test_acc = test(model)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            test_acc = tmp_test_acc
        log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
        performance_history.setdefault('train_acc', f'{train_acc:0.4f}')
        performance_history.setdefault('val_acc', f'{best_val_acc:0.4f}')
        performance_history.setdefault('test_acc', f'{test_acc:0.4f}')
        if epoch / 50 == 0:
            print(f'{epoch}')
            # print(log.format(epoch, train_acc, best_val_acc, test_acc))
    return epoch, train_acc, best_val_acc, test_acc


def init_gcn():
    dataset = 'Cora'
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', '..', 'Data',
                    'External')
    dataset = Planetoid(path, dataset, T.NormalizeFeatures())
    data = dataset[0]

    # if args.use_gdc:
    #     gdc = T.GDC(self_loop_weight=1, normalization_in='sym',
    #                 normalization_out='col',
    #                 diffusion_kwargs=dict(method='ppr', alpha=0.05),
    #                 sparsification_kwargs=dict(method='topk', k=128,
    #                                            dim=0), exact=True)
    #     data = gdc(data)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, data = Net(data, dataset).to(device), data.to(device)
    optimizer = torch.optim.Adam([
        dict(params=model.reg_params, weight_decay=5e-4),
        dict(params=model.non_reg_params, weight_decay=0)
    ], lr=0.01)
    return data, dataset, model, optimizer


# save2file(percentage, noise_percentage, preprocessing_strategy='add_edges', dataset='Cora')
# def save2file(percentage, noise_percentage, preprocessing_strategy , dataset):
#
#     suffix = ''
#     if preprocessing_strategy == 'add_edges':
#         suffix += '_added_edges_same_class_nodes'
#     if preprocessing_strategy == 'add_nodes':
#         suffix += '_add_same_class_to_new_nodes'
#
#     suffix += f'_percent={percentage}_noise={noise_percentage}'
#     import os
#     cur_dir = os.getcwd()
#     dataset_path = osp.join(osp.dirname(osp.dirname(cur_dir)), '..', 'Data',
#                             'Preprocessed', 'Cora')
#     # tmp = f'../../Data/karate_club_{saved_folder}.adjlist'
#     file_name = f'{dataset}{suffix}.adjlist'
#
#     file_path = osp.join(dataset_path, file_name)
#     tmp = file_path
#
#     if not os.path.exists(osp.dirname(tmp)):
#         os.makedirs(osp.dirname(tmp))
#
#     adjlist = []
#     # TODO assume that nodes in order
#     ## figure out if nodes are in order of row index?
#     ## > save to file using  pandas.
#     ##      >> convert from torch to pandas
#     ##      >> if nodes are in order save include index to represent nodes
#
#     # for ind, (node, emb_feat) in enumerate(embedded_x):
#     #     non_zero_weight_edges = [node]
#     #
#     #     for j in list(emb_feat.keys()):
#     #         non_zero_weight_edges.append(j)
#     #     adjlist.append(non_zero_weight_edges)
#     #
#     # assert len(adjlist) == len(G.nodes()), ""
#     #
#     # print(f"writing to {tmp}...")
#     #
#     # with open(tmp, 'w') as f:
#     #
#     #     txt = ""
#     #     for ind, i in enumerate(adjlist):
#     #         v = [f'{i[0]}']
#     #         for j in i[1:]:
#     #             # v += f'{j}'
#     #             v.append(f'{j}')
#     #         # TODO VALIDATE
#     #         txt += ' '.join(v) + '\n'
#     #         # txt += "\n"
#     #     f.write(txt)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--use_gdc', action='store_true',
                        help='Use GDC preprocessing.')
    args = parser.parse_args()

    # f = [r'C:\Users\Anak\PycharmProjects\AdaptiveGraphStructureEmbedding\Data\Preprocessed\Cora\cora_percent=0_noise=0.adjlist',
    #      r'C:\Users\Anak\PycharmProjects\AdaptiveGraphStructureEmbedding\Data\Preprocessed\Cora\cora_added_edges_same_class_nodes_percent=0.001_noise=0.0005.adjlist',
    #      r'C:\Users\Anak\PycharmProjects\AdaptiveGraphStructureEmbedding\Data\Preprocessed\Cora\cora_added_edges_same_class_nodes_percent=0.001_noise=0.001.adjlist',
    #      r'C:\Users\Anak\PycharmProjects\AdaptiveGraphStructureEmbedding\Data\Preprocessed\Cora\cora_added_edges_same_class_nodes_percent=0.001_noise=0.005.adjlist']
    # f = [r'C:\Users\Anak\PycharmProjects\AdaptiveGraphStructureEmbedding\Data\Preprocessed\Cora\cora_percent=0_noise=0.adjlist']
    # f = [r'C:\Users\Anak\PycharmProjects\AdaptiveGraphStructureEmbedding\Data\Preprocessed\Cora\cora_added_edges_same_class_nodes_percent=0.001_noise=0.0005.adjlist']
    f = [
        r'C:\Users\Anak\PycharmProjects\AdaptiveGraphStructureEmbedding\Data\Preprocessed\Cora\cora_added_edges_same_class_nodes_percent=0.001_noise=0.001.adjlist']
    # f = [r'C:\Users\Anak\PycharmProjects\AdaptiveGraphStructureEmbedding\Data\Preprocessed\Cora\cora_added_edges_same_class_nodes_percent=0.001_noise=0.0015.adjlist']
    # f = [r'C:\Users\Anak\PycharmProjects\AdaptiveGraphStructureEmbedding\Data\Preprocessed\Cora\cora_added_edges_same_class_nodes_percent=0.001_noise=0.002.adjlist']
    # f = [r'C:\Users\Anak\PycharmProjects\AdaptiveGraphStructureEmbedding\Data\Preprocessed\Cora\cora_added_edges_same_class_nodes_percent=0.005_noise=0.adjlist']
    # f = [r'C:\Users\Anak\PycharmProjects\AdaptiveGraphStructureEmbedding\Data\Preprocessed\Cora\cora_added_edges_same_class_nodes_percent=0.01_noise=0.adjlist']
    # f = [r'C:\Users\Anak\PycharmProjects\AdaptiveGraphStructureEmbedding\Data\Preprocessed\Cora\cora_added_edges_same_class_nodes_percent=0.001_noise=0.adjlist']

    for i in f:
        data, dataset, model, optimizer = init_gcn()

        convert_data(i)
        for i in range(100):
            log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
            epoch, train_acc, best_val_acc, test_acc = run_gcn(model, optimizer)
            if i <= 10:
                print(log.format(epoch, train_acc, best_val_acc, test_acc))
            else:
                exit()
