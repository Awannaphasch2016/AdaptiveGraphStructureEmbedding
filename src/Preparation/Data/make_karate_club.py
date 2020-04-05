import networkx as nx
import numpy as np
import torch

class Karate_club():
    def __init__(self):
        self.G = nx.karate_club_graph()
        self.saved_folder = ''
        # adj = nx.to_scipy_sparse_matrix(G).tocoo()
        # row = torch.from_numpy(adj.row.astype(np.int64)).to(torch.long)
        # col = torch.from_numpy(adj.col.astype(np.int64)).to(torch.long)

    def add_edges_betwen_same_class_node(self):
        nodes_class = {}
        for k, v in list(self.G.nodes(data=True)):
            # if v['club'] not in nodes_class:
            nodes_class.setdefault(v['club'], []).append(k)
        from itertools import combinations

        # Get all combinations of [1, 2, 3]
        # and length 2
        all_edges_of_same_class_nodes = []
        for k, i in nodes_class.items():
            for j in combinations(i, 2):
                all_edges_of_same_class_nodes.append(j)

        self.G.add_edges_from(all_edges_of_same_class_nodes)
        return self.G


    def preprocess_karate_club(self, add_edges=None):
        assert isinstance(add_edges, bool), 'only bool'

        if add_edges:
            self.saved_folder = 'added_edges_same_class_nodes'
            G = self.add_edges_betwen_same_class_node()
        else:
            self.saved_folder = 'no_added_edges'

        # TODO add attribute
        # nx_G = nx.from_edgelist(edge_index.detach().numpy().T)
        # nx.set_node_attributes(nx_G, dict(G.nodes))

        self.G = G
        return self.G

    def save2file(self):
        import networkx as nx
        tmp = f'../../Data/karate_club_{self.saved_folder}.adjlist'
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
            for i in adjlist:
                v = ""
                for j in i:
                    v += f'{j}'
                # TODO VALIDATE
                txt += ' '.join(v) + '\n'
                # txt += "\n"
            f.write(txt)
