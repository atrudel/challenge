import torch
import torch.nn.functional as F

from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_add_pool
import torch.nn as nn


N_CLASSES = 18

hparams = {
    'epochs': 50,
    'batch_size': 64,
    'learning_rate': 0.001,
    'num_node_features': 86,
    'hidden_dim': 128,
    'dropout': 0.2,
    'num_hidden_layers': 4
}


class GCN(torch.nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.dropout = nn.Dropout(hparams['dropout'])
        self.first_conv = GCNConv(
                in_channels=-1,
                out_channels=hparams['hidden_dim'],
            )
        self.hidden_layers = nn.Sequential(*[
            GCNConv(
                in_channels=hparams['hidden_dim'],
                out_channels=hparams['hidden_dim'],
            ),
            nn.ReLU(),
            self.dropout] * hparams['num_hidden_layers'])

        self.last_conv = GCNConv(
                in_channels=hparams['hidden_dim'],
                out_channels=hparams['hidden_dim'],
            )
        self.fc1 = nn.Linear(hparams['hidden_dim'], hparams['hidden_dim'])
        self.fc2 = nn.Linear(hparams['hidden_dim'], N_CLASSES)
        self.bn = nn.BatchNorm1d(hparams['hidden_dim'])
        self.relu = nn.ReLU()


    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # First conv layer
        x = self.relu(self.first_conv(x, edge_index))
        x = self.dropout(x)

        # Hidden layers
        x = self.hidden_layers(x, edge_index)


        # Last conv layer
        x = self.last_conv(x, edge_index)

        # Pooling
        x = global_add_pool(x, batch)

        # Batch norm
        x = self.bn(x)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)

