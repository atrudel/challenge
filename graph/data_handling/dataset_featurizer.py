# from torch_geometric.data import Data
#
import numpy as np
import torch
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.loader import DataLoader
from torch.utils.data import Subset

from config import DATA_DIR
from graph.data_handling.baseline_data_processing import load_graph_data, train_test_split
from sklearn.model_selection import train_test_split as sk_train_test_split

class ProteinDataset(InMemoryDataset):
    def __init__(self, root=DATA_DIR, test=False, transform=None, pre_transform=None, use_bert_embedding=False):
        """
        root = Where the dataset should be stored. This folder is split
        into raw_dir (downloaded dataset) and processed_dir (processed data).
        """
        self.test = test
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.use_bert_embedding = use_bert_embedding

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
        adj, node_features, edge_features = load_graph_data(self.use_bert_embedding)
        (adj_train, node_features_train, edge_features_train), y_train, \
            (adj_test, node_features_test, edge_features_test), proteins_test = train_test_split(
            adj, node_features, edge_features, val_size=0
        )
        if self.test:
            data_list = self.build_graphs(adj_test, node_features_test, edge_features_test, prot_names=proteins_test)
        else:
            data_list = self.build_graphs(adj_train, node_features_train, edge_features_train, y=y_train)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def build_graphs(self, adj, node_features, edge_features, y=None, prot_names=None):
        y = [None] * len(adj) if y is None else y
        prot_names = [None] * len(adj) if prot_names is None else prot_names
        graphs = []
        for gr_adj, gr_node_features, gr_edge_features, gr_y, gr_prot_name in zip(adj, node_features, edge_features, y, prot_names):

            x = torch.from_numpy(gr_node_features).float()
            M = gr_adj.tocoo().astype(int)
            indices = torch.from_numpy(np.vstack((M.row, M.col))).long()
            edge_attributes = torch.from_numpy(gr_edge_features).float()
            # edge_type = torch.from_numpy(
            #     np.argmax(gr_edge_features[:,1:], axis=1) # Todo: do this less naively
            # ).long()
            y = torch.tensor(gr_y) if gr_y is not None else None

            graph = Data(
                x=x,
                edge_index=indices,
                edge_attr=edge_attributes,
                y=y,
                protein_names=gr_prot_name
            )
            graphs.append(graph)
        return graphs

def get_train_val_dataloaders(batch_size, val_size=0.25, random_state=None, use_bert_embedding=False):
    full_train_dataset = ProteinDataset(test=False, use_bert_embedding=use_bert_embedding)
    train_indices, val_indices, _, _  = sk_train_test_split(
        range(len(full_train_dataset)),
        full_train_dataset.data.y,
        stratify=full_train_dataset.data.y,
        test_size=val_size,
        random_state=random_state
    )
    train_subset = Subset(full_train_dataset, train_indices)
    val_subset = Subset(full_train_dataset, val_indices)

    train_dataloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    return train_dataloader, val_dataloader

def get_full_train_dataloader(batch_size):
    return DataLoader(ProteinDataset(test=False), batch_size=batch_size, shuffle=True)
def get_test_dataloader(batch_size):
    return DataLoader(ProteinDataset(test=True), batch_size=batch_size, shuffle=False)

