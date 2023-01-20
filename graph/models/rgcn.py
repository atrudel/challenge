import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import RGCNConv
from torch_geometric.nn import global_add_pool

from config import device
from graph.data_handling.dataset_featurizer import ProteinDataset, get_full_train_dataloader

N_CLASSES = 18


hparams = {
    'epochs': 50,
    'batch_size': 64,
    'learning_rate': 0.001,
    'num_node_features': 86,
    'num_relations': 4,
    'num_bases': 30,
    'num_blocks': None,
    'hidden_dim': 128,
    'dropout': 0.2,
    'aggr': 'mean'
}


class RGCN(torch.nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.conv1 = RGCNConv(
                in_channels=hparams['num_node_features'],
                out_channels=hparams['hidden_dim'],
                num_relations=hparams['num_relations'],
                num_bases=hparams['num_bases'],
                num_blocks=hparams['num_blocks'],
                aggr=hparams['aggr'],
            )
        self.conv2 = RGCNConv(
                in_channels=hparams['hidden_dim'],
                out_channels=hparams['hidden_dim'],
                num_relations=hparams['num_relations'],
                num_bases=hparams['num_bases'],
                num_blocks=hparams['num_blocks'],
                aggr=hparams['aggr'],
            )
        self.fc1 = nn.Linear(hparams['hidden_dim'], hparams['hidden_dim'])
        self.fc2 = nn.Linear(hparams['hidden_dim'], N_CLASSES)
        self.dropout = nn.Dropout(hparams['dropout'])
        self.bn = nn.BatchNorm1d(hparams['hidden_dim'])

    def forward(self, data):
        x, edge_index, edge_type, batch = data.x, data.edge_index, data.edge_type, data.batch
        # First conv layer
        x = F.relu(self.conv1(x, edge_index, edge_type))
        # Second conv layer
        x = self.conv2(x, edge_index, edge_type)

        # Global pooling
        x = global_add_pool(x, batch)

        # Batch normalization
        x = self.bn(x)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
