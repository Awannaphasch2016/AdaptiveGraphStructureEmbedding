
import os
import os.path as osp

import networkx as nx
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid

from src.Visualization import *
class CiteSeer:
    def __init__(self, add_edges=None):
        assert isinstance(add_edges, bool), ''

        data = dataset[0]
        citeseer_edges = data['edge_index'].numpy().T
        citeseer_nodes = data['x'].numpy()
        citeseer_y = data.y.numpy().T

        # TODO here>> check why there are not 3327 nodes
        nx_G = nx.Graph()
        nx_G.add_edges_from(citeseer_edges)

        nodes_attr = {}
        # TODO VALIDATE that nodes are named by its row_ind
        for ind, (node, label) in enumerate(zip(citeseer_nodes, citeseer_y)):
            nodes_attr[ind] = {'label': label}

        # TODO SKIP OPTIMIZE how to get nodes classes from networkx ?
        ## use set_node_attributes() but it seem so slow to use with node 1433 attribute of cora
        nx.set_node_attributes(nx_G, nodes_attr)
        self.G = nx_G
        self.label = 'label'
        self.add_edges = add_edges
        if self.add_edges:
            self.add_edges_betwen_same_class_node()
        # self.dataset_name = 'citeseer'

    def add_edges_betwen_same_class_node(self):

        nodes_class = {}
        for k, v in list(self.G.nodes(data=True)):
            # if v['club'] not in nodes_class:
            try:
                nodes_class.setdefault(v[self.label], []).append(k)
            except:
                print(k, v)
                exit()
        from itertools import combinations

        # Get all combinations of [1, 2, 3]
        # and length 2
        all_edges_of_same_class_nodes = []
        for k, i in nodes_class.items():
            for j in combinations(i, 2):
                all_edges_of_same_class_nodes.append(j)

        self.G.add_edges_from(all_edges_of_same_class_nodes)
        return self.G

    def save2file(self):

        dataset_path = osp.join(osp.dirname(osp.dirname(cur_dir)), 'Data',
                                'Preprocessed', 'CiteSeer')
        # tmp = f'../../Data/karate_club_{saved_folder}.adjlist'
        folder = ''
        if self.add_edges:
            folder += '_added_edges_same_class_nodes'

        file_name = f'citeseer{folder}.adjlist'

        file_path = osp.join(dataset_path, file_name)
        tmp = file_path

        # print(tmp)
        # exit()

        if not os.path.exists(osp.dirname(tmp)):
            os.makedirs(osp.dirname(tmp))

        adjlist = []
        for ind, (node, neighbor) in enumerate(self.G.adjacency()):
            non_zero_weight_edges = [node]
            for j in list(neighbor.keys()):
                non_zero_weight_edges.append(j)
            adjlist.append(non_zero_weight_edges)

        assert len(adjlist) == len(self.G.nodes()), ""

        with open(tmp, 'w') as f:
            print(f"writing to {tmp}...")
            txt = ""
            for ind, i in enumerate(adjlist):
                v = [f'{ind}']
                for j in i:
                    # v += f'{j}'
                    v.append(f'{j}')
                # TODO VALIDATE
                txt += ' '.join(v) + '\n'
                # txt += "\n"
            f.write(txt)

if __name__ == '__main__':
    raise ValueError('Please validate that the code below work as expected; use make_cora.py to refer for help')

    dataset_name = 'CiteSeer'
    cur_dir = os.getcwd()
    path = osp.join(osp.dirname(osp.dirname(cur_dir)), '..', 'Data', 'External')
    dataset = Planetoid(path, dataset_name, T.TargetIndegree())

    # =====================
    # ==Hyper parameters
    # =====================
    save_adjlist_to_file = False
    plot_graph = True

    # add_edges = False

    # TODO pass .adjlist file or .embedding file to draw_grpah
    # change ame of draw_graph_before_embedding to plot_graph
    file_to_be_plotted =  r'C:\Users\Anak\PycharmProjects\AdaptiveGraphStructureEmbedding\Data\Preprocessed\Cora\cora_added_edges_same_class_nodes.embeddings'
    # is_embedding = True
    # # is_embedding = False

    add_edges = True
    add_nodes = False

    # add_edges = False
    # add_nodes = True

    percentage = [
        0.1]  # add percentage of added edges wrt to max number of edges
    noise_percentage = [
        0]  # add noise as percentage of added edges wrt to noise percentage

    assert add_edges + add_nodes <= 1, 'both value cannot be True'

    if add_edges:
        strategy = 'add_edges'
    elif add_nodes:
        strategy = 'add_nodes'
    else:
        raise ValueError('correct preprocessed strategy must be selected')
    #=====================
    #==running main function
    #=====================
    for i in percentage:
        for j in noise_percentage:
            if dataset_name == 'CiteSeer':
                citeseer = CiteSeer(add_edges=add_edges, add_nodes=add_nodes,
                            percentage=i, noise_percentage=j)

                if save_adjlist_to_file:
                    citeseer.save2file()

                if add_nodes:
                    strategy = 'add_nodes'
                elif add_edges:
                    strategy = 'add_edges'
                else:
                    raise ValueError('please select graph preprocessing strategy to use')

                # TODO here>> pass in directory to draw_graph_before_embedding

                if plot_graph:
                    draw_graph_before_embedding(citeseer.G,
                                                file_to_be_plotted=file_to_be_plotted,
                                                label=citeseer.label,
                                                strategy=strategy,
                                                dataset='cora')

            else:
                raise ValueError('this dataset does not exist')

