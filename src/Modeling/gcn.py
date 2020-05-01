import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0,parent_dir)

import argparse
import os.path as osp

import dgl
from dgl.nn.pytorch import GraphConv
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv  # noqa

from Examples.Libraries.PytorchGeometic.InMemoryDataset_class import *
from arg_parser import args


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


def convert_data(gcn, f):
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
        gcn.data.edge_index = torch.tensor(edge_index).transpose(1, 0)
    else:
        raise ValueError('no')

def get_gen_labels_for_min_and_maj():
    pass

class Net(torch.nn.Module):
    def __init__(self, data):
        super(Net, self).__init__()
        self.data = data
        self.conv1 = GraphConv(data.num_features, 16)
        self.conv2 = GraphConv(16, 16)
        self.conv3 = GraphConv(16, data.num_classes)

        import torch.nn as nn
        self.discriminator = nn.Sequential(
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 2),
        )

    def forward(self, g, x , get_conv1_emb=False, external_input=None):
        '''

        :param g: DGL graph.
        :return:
        '''

        if get_conv1_emb:
            h = self.conv1(g, x) # here what is input to conv1?
            x_after_conv1 = h
            return x_after_conv1

        if external_input is not None:
            h = torch.relu(x)
            h = torch.nn.functional.dropout(h) # add  the followign argument training=True or False

            h = self.conv2(g, h)
            x_after_conv2 = h
            h = torch.relu(x)
            h = torch.nn.functional.dropout(h) # add  the followign argument training=True or False

            h = torch.cat((x, external_input), dim=0)
            h = self.discriminator(h)

            logits = F.log_softmax(h, dim=1)

            return x_after_conv2,  logits
        else:
            raise ValueError('error')

class GCN:
    def __init__(self,data):
        self.data = data
        # self.dataset = dataset
        self.model = None
        self.optimizer = None
        self.init_gcn()

    def get_dgl_graph(self):
        src = self.data.edge_index[0]
        dst = self.data.edge_index[1]
        import numpy as np
        u = np.concatenate([src, dst])
        v = np.concatenate([dst, src])
        return dgl.DGLGraph((u,v))

    def loss(self, x, y):
        return F.nll_loss(x,y)

    def loss_and_step(self, x, y):
        self.optimizer.zero_grad()
        loss = self.loss(x,y)
        loss.backward(retain_graph=True)
        self.optimizer.step()
        return loss

    def discriminator(self, model_emb, external_input = None):
        if external_input is not None:
            x_as_input_to_softmax = torch.cat((model_emb, external_input), dim=0)
        else:
            x_as_input_to_softmax = model_emb
        # TODO change log_softmax to discriminator here
        logits = F.log_softmax(x_as_input_to_softmax, dim=1)
        # self.optimizer.zero_grad()
        return logits

    def train(self):
        self.model.train()
        #=====================
        #==torch geometric
        #=====================
        
        # x_after_conv1, x_after_conv2 = self.model()

        #=====================
        #==dgl
        #=====================
        
        x_after_conv1, x_after_conv2 = self.model(self.get_dgl_graph())

        return x_after_conv1, x_after_conv2


    @torch.no_grad()
    # def test(self):
    def test(self, data, y ):
        self.model.eval()

        (emb_after_cov1, emb_after_cov2), accs = self.model(self.get_dgl_graph()), [] # gdl
        logits = F.log_softmax(emb_after_cov2, dim=1)

        for _, mask in data('train_mask', 'test_mask'):
            pred = logits[mask].max(1)[1]
            acc = pred.eq(y[mask]).sum().item() / mask.sum().item()
            accs.append(acc)

        return accs

    def run_gcn(self):
        best_val_acc = test_acc = 0
        performance_history = {}
        for epoch in range(1, 201):
            # embedded_x = self.train()
            x_after_conv1 , x_after_conv2 = self.train()
            logits = self.discriminator(x_after_conv2)
            self.loss_and_step(logits[self.data.train_mask], self.data.y[self.data.train_mask])

            train_acc, val_acc, tmp_test_acc = self.test()

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                test_acc = tmp_test_acc

            log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
            performance_history.setdefault('train_acc', f'{train_acc:0.4f}')
            performance_history.setdefault('val_acc', f'{best_val_acc:0.4f}')
            # performance_history.setdefault('test_acc', f'{test_acc:0.4f}')

            if epoch / 50 == 0:
                print(f'{epoch}')
                # print(log.format(epoch, train_acc, best_val_acc, test_acc))

        return epoch, train_acc, best_val_acc, test_acc


    def init_gcn(self):

        if args.use_gdc:
            gdc = T.GDC(self_loop_weight=1, normalization_in='sym',
                        normalization_out='col',
                        diffusion_kwargs=dict(method='ppr', alpha=0.05),
                        sparsification_kwargs=dict(method='topk', k=128,
                                                   dim=0), exact=True)
            self.data = gdc(self.data)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model, self.data = Net(self.data).to(device), self.data

        # #=====================
        # #==torch parameters
        # #=====================
        # self.optimizer = torch.optim.Adam([
        #     dict(params=self.model.reg_params, weight_decay=5e-4),
        #     dict(params=self.model.non_reg_params, weight_decay=0)
        # ], lr=0.01)

        #=====================
        #==dgl
        #=====================
        import itertools
        # what is type of x?
        self.optimizer = torch.optim.Adam(itertools.chain(self.model.parameters()), lr=0.01)
        # self.optimizer = torch.optim.Adam(itertools.chain(self.model.parameters(),self.data.x.parameters()), lr=0.01)


        # return data, dataset, self.model, optimizer

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

        dataset = 'Cora'
        path = osp.join(osp.dirname(osp.realpath(__file__)), '..', '..', 'Data',
                        'External')
        dataset = Planetoid(path, dataset, T.NormalizeFeatures())
        data = dataset[0]

        gcn = GCN(data, dataset )
        # data, dataset, model, optimizer = init_gcn()

        convert_data(gcn, i)
        for i in range(100):
            log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
            epoch, train_acc, best_val_acc, test_acc = gcn.run_gcn()
            if i <= 10:
                print(log.format(epoch, train_acc, best_val_acc, test_acc))
            else:
                exit()
