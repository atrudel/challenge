# from torch_geometric.data import Data
#
import numpy as np
import torch
from torch_geometric.data import InMemoryDataset, Data

from config import DATA_DIR
from graph.data_handling.baseline_data_processing import load_graph_data, train_test_split


class ProteinDataset(InMemoryDataset):
    def __init__(self, root=DATA_DIR, test=False, transform=None, pre_transform=None):
        """
        root = Where the dataset should be stored. This folder is split
        into raw_dir (downloaded dataset) and processed_dir (processed data).
        """
        self.test = test
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        """ If this file exists in raw_dir, the download is not triggered.
            (The download func. is not implemented here)
        """
        return ['edge_attributes.txt', 'edgelist.txt', 'graph_indicator.txt', 'graph_labels.txt',
                'node_attributes.txt']

    @property
    def processed_file_names(self):
        """ If these files are found in raw_dir, processing is skipped"""
        if self.test:
            return ['test_graph_data.pt']
        else:
            return ['train_graph_data.pt']
    def process(self):
        adj, node_features, edge_features = load_graph_data()
        (adj_train, node_features_train, edge_features_train), y_train, \
            (adj_test, node_features_test, edge_features_test), proteins_test = train_test_split(
            adj, node_features, edge_features, val_size=0
        )
        if self.test:
            data_list = self.build_graphs(adj_test, node_features_test, edge_features_test)
        else:
            data_list = self.build_graphs(adj_train, node_features_train, edge_features_train, y_train)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def build_graphs(self, adj, node_features, edge_features, y=None):
        y = [None] * len(adj) if y is None else y
        graphs = []
        for gr_adj, gr_node_features, gr_edge_features, gr_y in zip(adj, node_features, edge_features, y):
            M = gr_adj.tocoo().astype(int)
            indices = torch.from_numpy(np.vstack((M.row, M.col))).long()
            edge_distances = torch.from_numpy(gr_edge_features[:,0]).float()
            edge_type = torch.from_numpy(
                np.argmax(gr_edge_features[:,1:], axis=1) # Todo: do this less naively
            ).long()
            graph = Data(
                x=torch.from_numpy(gr_node_features).float(),
                edge_index=indices,
                edge_attr=edge_distances,
                edge_type=edge_type,
                y=torch.tensor(gr_y)
            )
            graphs.append(graph)
        return graphs




if __name__ == '__main__':
    dataset_train = ProteinDataset(test=False)
    dataset_test = ProteinDataset(test=True)
