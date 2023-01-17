import pickle
from typing import Union

import numpy as np
import torch
from scipy import sparse as sp
from sklearn.model_selection import train_test_split as sk_train_test_split

from config import CACHE_DIR, DATA_DIR


def load_graph_data():
    """
    Function that loads graphs
    """
    try:
        with open(f'{CACHE_DIR}/adj.pickle', 'rb') as f:
            adj = pickle.load(f)
        with open(f'{CACHE_DIR}/node_features.pickle', 'rb') as f:
            node_features = pickle.load(f)
        with open(f'{CACHE_DIR}/edge_features.pickle', 'rb') as f:
            edge_features = pickle.load(f)
        print(f"Loading cached features from {CACHE_DIR}")

    except FileNotFoundError:
        print("Calculating features and adjacency matrix")
        graph_indicator = np.loadtxt(f"{DATA_DIR}/graph_indicator.txt", dtype=np.int64)
        _, graph_size = np.unique(graph_indicator, return_counts=True)

        edges = np.loadtxt(f"{DATA_DIR}/edgelist.txt", dtype=np.int64, delimiter=",")
        edges_inv = np.vstack((edges[:,1], edges[:,0]))
        edges = np.vstack((edges, edges_inv.T))
        s = edges[:,0]*graph_indicator.size + edges[:,1]
        idx_sort = np.argsort(s)
        edges = edges[idx_sort,:]
        edges,idx_unique =  np.unique(edges, axis=0, return_index=True)
        A = sp.csr_matrix((np.ones(edges.shape[0]), (edges[:,0], edges[:,1])), shape=(graph_indicator.size, graph_indicator.size))

        x = np.loadtxt(f"{DATA_DIR}/node_attributes.txt", delimiter=",")
        edge_attr = np.loadtxt(f"{DATA_DIR}/edge_attributes.txt", delimiter=",")
        edge_attr = np.vstack((edge_attr,edge_attr))
        edge_attr = edge_attr[idx_sort,:]
        edge_attr = edge_attr[idx_unique,:]

        adj = []
        node_features = []
        edge_features = []
        idx_n = 0
        idx_m = 0
        for i in range(graph_size.size):
            adj.append(A[idx_n:idx_n+graph_size[i],idx_n:idx_n+graph_size[i]])
            edge_features.append(edge_attr[idx_m:idx_m+adj[i].nnz,:])
            node_features.append(x[idx_n:idx_n+graph_size[i],:])
            idx_n += graph_size[i]
            idx_m += adj[i].nnz

        with open(f'{CACHE_DIR}/adj.pickle', 'wb') as f:
            pickle.dump(adj, f)
        with open(f'{CACHE_DIR}/node_features.pickle', 'wb') as f:
            pickle.dump(node_features, f)
        with open(f'{CACHE_DIR}/edge_features.pickle', 'wb') as f:
            pickle.dump(edge_features, f)
        print("Features and adjacency matrix cached for future use.")

    return adj, node_features, edge_features


def normalize_adjacency(A):
    """
    Function that normalizes an adjacency matrix
    """
    n = A.shape[0]
    A += sp.identity(n)
    degs = A.dot(np.ones(n))
    inv_degs = np.power(degs, -1)
    D = sp.diags(inv_degs)
    A_normalized = D.dot(A)

    return A_normalized


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """
    Function that converts a Scipy sparse matrix to a sparse Torch tensor
    """
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def train_test_split(adj, node_features, edge_features, val_size=0, random_state=42):
    """Split train and test data. If val_size is greater than 0, it represents the percentage of the train data
    to return as a validation set. If val_size=0, the train data will not be split between train and val.
    Features of a given subset are returned in a tuple. Eg, with validation, the output is:
    (features_train), y_train, (features_val), y_val, (features_test), proteins_test
    """
    # Train
    adj_train = list()
    node_features_train = list()
    edge_features_train = list()
    y_train = list()
    # Test
    adj_test = list()
    node_features_test = list()
    edge_features_test = list()
    proteins_test = list()

    with open(f'{DATA_DIR}/graph_labels.txt', 'r') as f:
        for i, line in enumerate(f):
            t = line.split(',')
            # Unknown label: test set
            if len(t[1][:-1]) == 0:
                adj_test.append(adj[i])
                node_features_test.append(node_features[i])
                edge_features_test.append(edge_features[i])
                proteins_test.append(t[0])
            # Known label: train set
            else:
                adj_train.append(adj[i])
                node_features_train.append(node_features[i])
                edge_features_train.append(edge_features[i])
                y_train.append(int(t[1][:-1]))

    if isinstance(val_size, float) and val_size > 0:
        adj_train, adj_val, node_features_train, node_features_val, edge_features_train, edge_features_val, \
            y_train, y_val = sk_train_test_split(
                adj_train, node_features_train, edge_features_train, y_train,
                test_size=val_size, random_state=random_state, stratify=y_train
        )
        return (adj_train, node_features_train, edge_features_train), y_train, \
            (adj_val, node_features_val, edge_features_val), y_val, \
            (adj_test, node_features_test, edge_features_test), proteins_test

    else:
        return (adj_train, node_features_train, edge_features_train), y_train, \
            (adj_test, node_features_test, edge_features_test), proteins_test

