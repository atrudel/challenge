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
    'batch_size': 32,
    'learning_rate': 0.001,
    'num_node_features': 86,
    'num_relations': 4,
    'num_bases': 30,
    'num_blocks': None,
    'hidden_dim': 128,
    'dropout': 0.2,
    'aggr': 'mean',
    'pooling': 'mean'
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
        self.fc = nn.Linear(hparams['hidden_dim'], N_CLASSES)
        self.dropout = nn.Dropout(hparams['dropout'])

    def forward(self, data):
        x, edge_index, edge_type, batch = data.x, data.edge_index, data.edge_type, data.batch
        # First conv layer
        x = F.relu(self.conv1(x, edge_index, edge_type))
        # Second conv layer
        x = self.conv2(x, edge_index, edge_type)

        # Global pooling
        x = global_add_pool(x, batch)

        x = F.relu(self.fc(x))
        x = self.dropout(x)
        return F.log_softmax(x, dim=1)


# def train(data_loader: DataLoader, model, optimizer, criterion):
#     model.train()
#     total_loss = 0
#     for data in data_loader:
#         data = data.to(device)
#         optimizer.zero_grad()
#         out = model(data)
#         loss = criterion(out, data.y)
#         total_loss += loss.item() * data.num_graphs
#         loss.backward()
#         optimizer.step()
#     return total_loss / len(train_loader.dataset)


# @torch.no_grad()
# def test():
#     model.eval()
#     pred = model(data.edge_index, data.edge_type).argmax(dim=-1)
#     train_acc = float((pred[data.train_idx] == data.train_y).float().mean())
#     test_acc = float((pred[data.test_idx] == data.test_y).float().mean())
#     return train_acc, test_acc

# if __name__ == '__main__':
#     n_epochs = 10
#     train_loader = get_full_train_dataloader(batch_size=32)
#     model = RGCN().to(device)
#     # print(summary(model, [(185, 86), (2, 132947), (132947)]))
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0005)
#     criterion = nn.CrossEntropyLoss()
#     for epoch in range(n_epochs):
#         loss = train(train_loader, model, optimizer, criterion)
#         print(f"[Epoch {epoch: <2d}]  Loss: {loss:.3f}")