import os
import os.path as osp

import networkx as nx
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid



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


class Cora:
    def __init__(self, add_edges=None, add_nodes=None, percentage=None,
                 noise_percentage=None):
        assert isinstance(add_edges, bool), ''
        assert isinstance(add_nodes, bool), ''
        assert percentage >= 0 and percentage <= 1, ''
        assert noise_percentage >= 0 and noise_percentage <= 1, ''

        data = dataset[0]
        cora_edges = data['edge_index'].numpy().T
        cora_nodes = data['x'].numpy()
        cora_y = data.y.numpy().T

        nx_G = nx.Graph()
        nx_G.add_edges_from(cora_edges)

        nodes_attr = {}
        # TODO VALIDATE that nodes are named by its row_ind
        for ind, (node, label) in enumerate(zip(cora_nodes, cora_y)):
            nodes_attr[ind] = {'label': label}

        # TODO SKIP OPTIMIZE how to get nodes classes from networkx ?
        ## use set_node_attributes() but it seem so slow to use with node 1433 attribute of cora
        nx.set_node_attributes(nx_G, nodes_attr)
        self.G = nx_G
        self.label = 'label'
        self.add_edges = add_edges
        self.add_nodes = add_nodes
        self.percentage = percentage
        self.noise_percentage = noise_percentage

        if self.add_edges:
            self.add_edges_betwen_same_class_node()
        elif self.add_nodes:
            self.add_same_class_to_new_nodes()

        # self.dataset = 'cora'

    def add_same_class_to_new_nodes(self):
        assert 0 == min(self.G.nodes()), ''

        nodes_class = {}
        for k, v in list(self.G.nodes(data=True)):
            # if v['club'] not in nodes_class:
            try:

                nodes_class.setdefault(v[self.label], []).append(k)
            except:
                print(k, v)
                exit()

        # TODO select percent of edges to be added to new node
        import random
        all_added_edges = []
        for i,v in nodes_class.items():
            n = len(v)
            num_selected_nodes = int(n * self.percentage)
            random.shuffle(v)
            all_added_edges.extend(v)
            nodes_class[i] = v[:num_selected_nodes]

        total_num_edges =  len(all_added_edges)
        selected_edges_to_new_nodes_of_the_same_class = []
        new_node_representation_of_class = {}
        new_node_count = 0
        for k, i in nodes_class.items():
            if k not in new_node_representation_of_class:
                new_node_count += 1
                node_id = max(list(self.G.nodes())) + new_node_count
                new_node_representation_of_class.setdefault(k, node_id)
                for node in i:
                    selected_edges_to_new_nodes_of_the_same_class.append(
                        (new_node_representation_of_class[k], node))

        self.G.add_nodes_from(list(new_node_representation_of_class.values()))
        self.G.add_edges_from(selected_edges_to_new_nodes_of_the_same_class)


        nx.set_node_attributes(self.G,
                               {new_node: {'label': k} for k, new_node in
                                new_node_representation_of_class.items()})

        # add random edges
        # randomly picked 2 node pairs
        num_random_edges = int(total_num_edges * self.noise_percentage)
        random_edges = []
        # permutate each class, get the first n
        for i in range(num_random_edges):
            e = random.sample(list(self.G.nodes()), 2)
            random_edges.append(e)
        self.G.add_edges_from(random_edges)

        return self.G

    def add_edges_betwen_same_class_node(self):

        nodes_class = {}
        for k, v in list(self.G.nodes(data=True)):
            # if v['club'] not in nodes_class:
            try:
                nodes_class.setdefault(v[self.label], []).append(k)
            except:
                print(k, v)
                exit()


        selected_edges_between_same_class_nodes = []
        import random
        from itertools import combinations
        for i,v in nodes_class.items():
            all_nodes_pair_combinations = list(combinations(v, 2))
            random.shuffle(all_nodes_pair_combinations)

            n = len(all_nodes_pair_combinations)
            num_selected_non_random_edges = int(n * self.percentage)
            selected_non_random_edges = all_nodes_pair_combinations[:num_selected_non_random_edges]

            selected_random_edges = []
            num_selected_random_edges = int(n * self.noise_percentage)

            # TODO here>> validate that add random nodes are correct
            for j in range(num_selected_random_edges):
                e = random.sample(list(self.G.nodes()),2)
                selected_random_edges.append(e)

            selected_edges = selected_random_edges + selected_non_random_edges
            selected_edges_between_same_class_nodes.extend(selected_edges)

        self.G.add_edges_from(selected_edges_between_same_class_nodes)

        # # TODO add random edges
        # # randomly picked 2 node pairs
        # num_random_edges = int(total_original_edges  * self.noise_percentage)
        # random_edges = []
        # # permutate each class, get the first n
        # for i in range(num_random_edges):
        #     e = random.sample(list(self.G.nodes()), 2)
        #     random_edges.append(e)
        # self.G.add_edges_from(random_edges)

        # from itertools import combinations
        # # Get all combinations of [1, 2, 3]
        # # and length 2
        # all_edges_of_same_class_nodes = []
        # for k, i in nodes_class.items():
        #     for j in combinations(i, 2):
        #         all_edges_of_same_class_nodes.append(j)
        # self.G.add_edges_from(all_edges_of_same_class_nodes)

        return self.G

    def save2file(self):
        suffix = ''
        if self.add_edges:
            suffix += '_added_edges_same_class_nodes'
        if self.add_nodes:
            suffix += '_add_same_class_to_new_nodes'
        suffix += f'_percent={self.percentage}_noise={self.noise_percentage}'
        dataset_path = osp.join(osp.dirname(osp.dirname(cur_dir)), 'Data',
                                'Preprocessed', 'Cora')
        # tmp = f'../../Data/karate_club_{saved_folder}.adjlist'
        file_name = f'cora{suffix}.adjlist'

        file_path = osp.join(dataset_path, file_name)
        tmp = file_path

        if not os.path.exists(osp.dirname(tmp)):
            os.makedirs(osp.dirname(tmp))

        adjlist = []
        for ind, (node, neighbor) in enumerate(self.G.adjacency()):
            non_zero_weight_edges = [node]
            for j in list(neighbor.keys()):
                non_zero_weight_edges.append(j)
            adjlist.append(non_zero_weight_edges)

        assert len(adjlist) == len(self.G.nodes()), ""

        print(f"writing to {tmp}...")

        with open(tmp, 'w') as f:

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
    dataset_name = 'Cora'
    cur_dir = os.getcwd()
    path = osp.join(osp.dirname(osp.dirname(cur_dir)), 'Data', 'External')
    # dataset_name = 'CiteSeer'
    dataset = Planetoid(path, dataset_name, T.TargetIndegree())

    # add_edges = False
    from src.Visualization import *

    is_embedding = True
    # is_embedding = False

    add_edges = True
    add_nodes = False

    # add_edges = False
    # add_nodes = True

    # TODO How do I implement deep learning and this file together
    ## > move CiteSeer and Cora dataset to make_cora, make_citeseer
    ## > deep learning uses when I
    # # experiment 1
    # percentage = [1] # add percentage of added edges wrt to max number of edges
    # noise_percentage = [0, 0.2, 0.5,0.8] # add noise as percentage of added edges wrt to noise percentage

    # experiment 2
    # percentage = [0.2, 0.5,0.8] # add percentage of added edges wrt to max number of edges
    # noise_percentage = [0] # add noise as percentage of added edges wrt to noise percentage
    percentage = [0.1] # add percentage of added edges wrt to max number of edges
    noise_percentage = [0] # add noise as percentage of added edges wrt to noise percentage

    assert add_edges + add_nodes <= 1, 'both value cannot be True'
    for i in percentage:
        for j in noise_percentage:
            if dataset_name == 'Cora':
                cora = Cora(add_edges=add_edges, add_nodes=add_nodes,
                            percentage=i, noise_percentage=j)

                # cora.save2file()

                if add_nodes:
                    strategy = 'add_nodes'
                if add_edges:
                    strategy = 'add_edges'

                # TODO here>> pass in directory to draw_graph_before_embedding

                draw_graph_before_embedding(cora.G, is_embedding=is_embedding,
                                            label=cora.label,
                                            strategy=strategy,
                                            dataset='cora')

            elif dataset_name == 'CiteSeer':
                raise ValueError(
                    "max value of data.edge_index and number of data.x are not the same. So i ma not sure how nodes are being labeld")
                citeseer = CiteSeer(add_edges=add_edges)

                citeseer.save2file()

                # draw_graph_before_embedding(citeseer.G, is_embedding=is_embedding,
                #                             label=citeseer.label,
                #                             use_added_same_class_dataset=add_edges,
                #                             dataset=dataset_name)

            else:
                raise ValueError('this dataset does not exist')
