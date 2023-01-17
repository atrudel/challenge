import time
import numpy as np
import scipy.sparse as sp

import torch
import torch.nn as nn
from torch import optim
from graph.data_processing import load_graph_data, normalize_adjacency, sparse_mx_to_torch_sparse_tensor, train_test_split
from graph.models.baseline import GNN
from utils.submission import write_submission_file
from ray import tune

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def train(model, optimizer, loss_function, features_train, y_train, hparams):
    adj_train, node_features_train, edge_features_train = features_train
    N_train = len(adj_train)
    batch_size = hparams['batch_size']

    # Train model
    for epoch in range(hparams['epochs']):
        t = time.time()
        model.train()
        train_loss = 0
        correct = 0
        count = 0
        # Iterate over the batches
        for i in range(0, N_train, batch_size):
            adj_batch = list()
            features_batch = list()
            idx_batch = list()
            y_batch = list()

            # Create tensors
            for j in range(i, min(N_train, i+batch_size)):
                n = adj_train[j].shape[0]
                adj_batch.append(adj_train[j]+sp.identity(n))
                features_batch.append(node_features_train[j])
                idx_batch.extend([j-i]*n)
                y_batch.append(y_train[j])

            adj_batch = sp.block_diag(adj_batch)
            features_batch = np.vstack(features_batch)

            adj_batch = sparse_mx_to_torch_sparse_tensor(adj_batch).to(device)
            features_batch = torch.FloatTensor(features_batch).to(device)
            idx_batch = torch.LongTensor(idx_batch).to(device)
            y_batch = torch.LongTensor(y_batch).to(device)

            optimizer.zero_grad()
            output = model(features_batch, adj_batch, idx_batch)
            loss = loss_function(output, y_batch)
            train_loss += loss.item() * output.size(0)
            count += output.size(0)
            preds = output.max(1)[1].type_as(y_batch)
            correct += torch.sum(preds.eq(y_batch).double())
            loss.backward()
            optimizer.step()
            loss_train = train_loss / count
            acc_train = correct / count
            tune.track.log(acc_train=acc_train)

        if epoch % 5 == 0:
            print('Epoch: {:03d}'.format(epoch+1),
                  'loss_train: {:.4f}'.format(loss_train),
                  'acc_train: {:.4f}'.format(acc_train),
                  'time: {:.4f}s'.format(time.time() - t))

def test_predic_proba(model, features_test, hparams):
    adj_test, node_features_test, edge_features_test = features_test
    N_test = len(adj_test)
    batch_size = hparams['batch_size']

    # Evaluate model
    model.eval()
    y_pred_proba = list()
    # Iterate over the batches
    for i in range(0, N_test, batch_size):
        adj_batch = list()
        idx_batch = list()
        features_batch = list()
        y_batch = list()

        # Create tensors
        for j in range(i, min(N_test, i+batch_size)):
            n = adj_test[j].shape[0]
            adj_batch.append(adj_test[j]+sp.identity(n))
            features_batch.append(node_features_test[j])
            idx_batch.extend([j-i]*n)

        adj_batch = sp.block_diag(adj_batch)
        features_batch = np.vstack(features_batch)

        adj_batch = sparse_mx_to_torch_sparse_tensor(adj_batch).to(device)
        features_batch = torch.FloatTensor(features_batch).to(device)
        idx_batch = torch.LongTensor(idx_batch).to(device)

        output = model(features_batch, adj_batch, idx_batch)
        y_pred_proba.append(output)

    y_pred_proba = torch.cat(y_pred_proba, dim=0)
    y_pred_proba = torch.exp(y_pred_proba)
    y_pred_proba = y_pred_proba.detach().cpu().numpy()
    return y_pred_proba

def launch_experiment(hparams):
    n_class = 18
    adj, node_features, edge_features = load_graph_data()
    adj = [normalize_adjacency(A) for A in adj]
    features_train, y_train, features_test, proteins_test = train_test_split(adj, node_features, edge_features)

    model = GNN(hparams['n_input'], hparams['n_hidden'], hparams['dropout'], n_class).to(device)
    optimizer = optim.Adam(model.parameters(), lr=hparams['learning_rate'])
    loss_function = nn.CrossEntropyLoss()
    train(model, optimizer, loss_function, features_train, y_train, hparams)
    return model

if __name__ == '__main__':
    hparams = {
        'epochs': 10,
        'batch_size': 64,
        'n_hidden': 64,
        'n_input': 86,
        'dropout': 0.2,
        'learning_rate': 0.001,
    }
    model = launch_experiment(hparams)
    adj, node_features, edge_features = load_graph_data()
    adj = [normalize_adjacency(A) for A in adj]
    features_train, y_train, features_test, proteins_test = train_test_split(adj, node_features, edge_features)
    y_pred_proba = test_predic_proba(model, features_test, hparams)
    write_submission_file(y_pred_proba, proteins_test, 'sample_structure_submission.csv')