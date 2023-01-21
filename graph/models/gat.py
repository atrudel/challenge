import torch.nn as nn
from torch.nn import BatchNorm1d
from torch_geometric.graphgym import GCNConv
from torch_geometric.nn import GAT, BatchNorm, GATv2Conv, global_mean_pool, global_max_pool
from torch_geometric.nn.pool import global_add_pool
from training.model_training_torch_geometric import launch_experiment
import torch.nn.functional as F


NUM_CLASSES = 18

hparams = {
    'epochs': 50,
    'batch_size': 32,
    'learning_rate': 0.006,
    'hidden_dim': 56,
    'num_hidden_layers': 0,
    'dropout': 0.2,
    'num_heads': 8,
    'pooling': 'mean'
}

pooling_functions = {
    'mean': global_mean_pool,
    'sum': global_add_pool,
    'max': global_max_pool
}
class GAT(nn.Module):
    """Graph Attention Network"""
    def __init__(self, hparams):
        super().__init__()
        self.first_gat = GATv2Conv(-1, hparams['hidden_dim'], hparams['num_heads'], edge_dim=5)
        self.last_gat = GATv2Conv(hparams['hidden_dim']*hparams['num_heads'], hparams['hidden_dim'], hparams['num_heads'], edge_dim=5)
        self.dropout = nn.Dropout(hparams['dropout'])
        self.bn = BatchNorm1d(hparams['hidden_dim'] * hparams['num_heads'])
        self.pool = pooling_functions[hparams['pooling']]
        self.lin = nn.Linear(hparams['hidden_dim']*hparams['num_heads'], NUM_CLASSES)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        x = self.dropout(x)
        x = self.first_gat(x, edge_index, edge_attr)
        x = F.elu(x)
        x = self.bn(x)

        x = self.last_gat(x, edge_index, edge_attr)

        x = self.pool(x, batch)

        x = self.bn(x)
        x = self.lin(x)
        return F.log_softmax(x, dim=1)




if __name__ == '__main__':
    hparams['epochs'] = 1
    model = GAT(hparams)
    launch_experiment(model, hparams, experiment_name='test_gat')