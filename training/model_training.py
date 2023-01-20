import os
from datetime import datetime

import numpy as np
import pytz
from tqdm import tqdm
import scipy.sparse as sp
from torch import optim

from graph.data_handling.baseline_data_processing import load_graph_data, normalize_adjacency, sparse_mx_to_torch_sparse_tensor, train_test_split
import torch
import torch.nn as nn
from config import device, CHECKPOINT_DIR
from ray.air import session

from graph.models.baseline import GNN


def train(epoch, model, loss_function, optimizer, features_train, y_train, hparams, search=False):
    adj_train, node_features_train, edge_features_train = features_train
    batch_size = hparams['batch_size']

    model.train()
    train_loss = 0
    correct = 0
    count = 0

    # Iterate over the batches
    with tqdm(range(0, len(y_train), batch_size), unit='batch', disable=search) as tepoch:
        tepoch.set_description(f"Epoch {epoch}")
        for i_batch in tepoch:
            adj_batch, features_batch, idx_batch, y_batch = compute_batch_features(
                adj_train, node_features_train, y_train, batch_size, i_batch
            )
            # Step
            optimizer.zero_grad()
            output = model(features_batch, adj_batch, idx_batch)
            loss = loss_function(output, y_batch)
            train_loss += loss.item() * output.size(0)
            count += output.size(0)
            preds = output.max(1)[1].type_as(y_batch)
            correct += torch.sum(preds.eq(y_batch).double()).item()
            loss.backward()
            optimizer.step()
            loss_train = train_loss / count
            acc_train = correct / count
            tepoch.set_postfix(loss_train=loss_train, acc_train=acc_train)

        loss_train = train_loss / count
        return loss_train, acc_train



def validate(epoch, model, loss_function, features_val, y_val, hparams, search=False):
    adj_val, node_features_val, edge_features_val = features_val
    batch_size = hparams['batch_size']
    val_loss = 0
    correct = 0
    count = 0

    model.eval()
    with torch.no_grad():
        for i_batch in range(0, len(y_val), batch_size):
            adj_batch, features_batch, idx_batch, y_batch = compute_batch_features(
                adj_val, node_features_val, y_val, batch_size, i_batch
            )

            output = model(features_batch, adj_batch, idx_batch)
            loss = loss_function(output, y_batch)
            val_loss += loss.item() * output.size(0)
            count += output.size(0)
            preds = output.max(1)[1].type_as(y_batch)
            correct += torch.sum(preds.eq(y_batch).double()).item()

        loss_val = val_loss / count
        acc_val = correct / count
        if not search:
            print('Epoch: {:03d}'.format(epoch),
                  'loss_val: {:.4f}'.format(loss_val),
                  'acc_val: {:.4f}'.format(acc_val),
                 )
    return loss_val, acc_val


def compute_batch_features(adj, node_features, y, batch_size, i_batch):
    N_train = len(y)
    adj_batch = list()
    features_batch = list()
    idx_batch = list()
    y_batch = list()

    # Create tensors
    for j in range(i_batch, min(N_train, i_batch + batch_size)):
        n = adj[j].shape[0]
        adj_batch.append(adj[j] + sp.identity(n))
        features_batch.append(node_features[j])
        idx_batch.extend([j - i_batch] * n)
        y_batch.append(y[j])

    adj_batch = sp.block_diag(adj_batch)
    features_batch = np.vstack(features_batch)
    adj_batch = sparse_mx_to_torch_sparse_tensor(adj_batch).to(device)
    features_batch = torch.FloatTensor(features_batch).to(device)
    idx_batch = torch.LongTensor(idx_batch).to(device)
    y_batch = torch.LongTensor(y_batch).to(device)
    return adj_batch, features_batch, idx_batch, y_batch



def run(hparams, experiment_name=None, seed=42, search=True):
    if not search:
        if experiment_name is None:
            experiment_name = "experiment"
        experiment_name += f"_{datetime.now(tz=pytz.timezone('Europe/Paris')).strftime('%Y-%m-%d_%Hh%Mm%Ss')}"
        print(f"Launching new experiment {experiment_name}")

    # Data Loading
    adj, node_features, edge_features = load_graph_data()
    adj = [normalize_adjacency(A) for A in adj]
    features_train, y_train, features_val, y_val, features_test, proteins_test = train_test_split(
        adj,
        node_features,
        edge_features,
        val_size=0.25,
        random_state=seed
    )

    # Model initialization
    model = GNN(86, hparams['n_hidden'], hparams['dropout'], 18).to(device)
    optimizer = optim.Adam(model.parameters(), lr=hparams['learning_rate'])
    loss_function = nn.CrossEntropyLoss()

    # Training loop
    best_val_loss = float('inf')
    for epoch in range(hparams['epochs']):
        train_loss, train_acc = train(epoch, model, loss_function, optimizer, features_train, y_train, hparams, search)
        val_loss, val_acc = validate(epoch, model, loss_function, features_val, y_val, hparams, search)
        session.report({
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc
        })

        # Save checkpoint if val loss improved and if not in hyperaparamter search
        if search is False and val_loss < best_val_loss:
            save_dir = f"{CHECKPOINT_DIR}/{experiment_name}"
            os.makedirs(save_dir, exist_ok=True)
            save_path = f"{save_dir}/model_epoch={epoch}_val-loss={val_loss:.3f}.pth"
            print(f"Validation loss improved, saving checkpoint to {save_path}")
            print()
            checkpoint = {
                'experiment_name': experiment_name,
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss/train': train_loss,
                'loss/val': val_loss,
                'hparams': hparams
            }
            torch.save(checkpoint, save_path)
    return best_val_loss


if __name__ == '__main__':
    # Example
    hparams = {
        'epochs': 10,
        'batch_size': 64,
        'n_hidden': 64,
        'n_input': 86,
        'dropout': 0.2,
        'learning_rate': 0.001,
    }
    best_val_loss = run(hparams, experiment_name=None, seed=42)
    print(best_val_loss)