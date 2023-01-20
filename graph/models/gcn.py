import torch
import torch.nn.functional as F
import torch.nn as nn
from torchsummary import summary

from torch_geometric.datasets import Entities
from torch_geometric.nn import FastRGCNConv, RGCNConv
from torch_geometric.loader import DataLoader
from config import device
from graph.data_handling.dataset_featurizer import ProteinDataset


N_NODE_FEATURES = 86
NUM_RELATIONS = 4
N_CLASSES = 18

hparams = {
    'num_bases': 30,
    'num_blocks': None,
    'hidden_dim': 64,
    'aggr': 'mean'
}


class RGCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = RGCNConv(
                in_channels=N_NODE_FEATURES,
                out_channels=hparams['hidden_dim'],
                num_relations=NUM_RELATIONS,
                num_bases=hparams['num_bases'],
                num_blocks=hparams['num_blocks'],
                aggr=hparams['aggr'],
            )
        self.convs = [
            RGCNConv(
                in_channels=hparams['hidden_dim'],
                out_channels=N_CLASSES,
                num_relations=NUM_RELATIONS,
                num_bases=hparams['num_bases'],
                num_blocks=hparams['num_blocks'],
                aggr=hparams['aggr'],
            )
        ]
        self.conv2 =

    def forward(self, x, edge_index, edge_type):
        x = F.relu(self.conv1(x, edge_index, edge_type))
        x = self.conv2(x, edge_index, edge_type)
        return F.log_softmax(x, dim=1)


def train(data_loader: DataLoader, model: RGCN, optimizer, criterion):
    model.train()
    total_loss = 0
    for data in data_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_type)
        loss = criterion(out, data.y)
        total_loss += loss.item() * data.num_graphs
        loss.backward()
        optimizer.step()
    return total_loss / len(train_loader.dataset)


# @torch.no_grad()
# def test():
#     model.eval()
#     pred = model(data.edge_index, data.edge_type).argmax(dim=-1)
#     train_acc = float((pred[data.train_idx] == data.train_y).float().mean())
#     test_acc = float((pred[data.test_idx] == data.test_y).float().mean())
#     return train_acc, test_acc

if __name__ == '__main__':
    n_epochs = 10
    print
    train_loader = DataLoader(ProteinDataset(test=False), batch_size=32, shuffle=True)
    model = RGCN().to(device)
    # print(summary(model, [(185, 86), (2, 132947), (132947)]))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0005)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(n_epochs):
        loss = train(train_loader, model, optimizer, criterion)