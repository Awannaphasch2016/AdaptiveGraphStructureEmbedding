# import torch
# from torch_geometric.data import Data
#
# edge_index = torch.tensor([[0, 1, 1, 2],
#                            [1, 0, 2, 1]], dtype=torch.long)
# x = torch.tensor([[-1], [0], [1]], dtype=torch.float)
#
# data = Data(x=x, edge_index=edge_index)
# print()

import logging
from Examples.Models.deepwalk_graph_class import Graph
import torch
from torch_geometric.data import InMemoryDataset
from Examples.Models.deepwalk_graph_class import *


class MyOwnDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(MyOwnDataset, self).__init__(root, transform, pre_transform)

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return ['data.pt']

    # def download(self):
    #     # todo how to implement/use this method?
    #     pass
    #     # Download to `self.raw_dir`.
    #
    # def process(self):
    #     # Read data into huge `Data` list.
    #     data_list = [...]
    #
    #     if self.pre_filter is not None:
    #         data_list = [data for data in data_list if self.pre_filter(data)]
    #
    #     if self.pre_transform is not None:
    #         data_list = [self.pre_transform(data) for data in data_list]
    #
    #     data, slices = self.collate(data_list)
    #     torch.save((data, slices), self.processed_paths[0])


def process(f):
    G = load_adjacencylist(f, undirected=True)
    return G
    # if args.format == "adjlist":
    #     G = Graph.load_adjacencylist(args.input, undirected=args.undirected)
    # elif args.format == "edgelist":
    #     G = Graph.load_edgelist(args.input, undirected=args.undirected)
    # elif args.format == "mat":
    #     G = Graph.load_matfile(args.input,
    #                            variable_name=args.matfile_variable_name,
    #                            undirected=args.undirected)
    # else:
    #     raise Exception(
    #         "Unknown file format: '%s'.  Valid formats: 'adjlist', 'edgelist', 'mat'" % args.format)



