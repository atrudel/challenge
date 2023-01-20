import matplotlib.pyplot as plt
from collections import defaultdict

import numpy as np


def visualize_amino_acids(node_features, labels):
    amido_acids = defaultdict(list)

    for protein_fts, label in zip(node_features, labels):
        # Amino Acids
        amido_acids[label].append(protein_fts[:, 3:23].sum(axis=0))

    prot_ids = list(range(min(labels), max(labels) + 1))

    fig = plt.figure(figsize=(15, 30))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    for prot_id in prot_ids:
        aa_count = np.vstack(amido_acids[prot_id]).sum(axis=0)
        plt.subplot(6, 3, prot_id+1)
        bar = plt.bar(list(range(20)), aa_count, align='center')
        plt.bar_label(bar)
        plt.xlabel("Amino acid type")
        plt.ylabel("Number of amido acids")
        plt.title(f"Protein type: {prot_id}")
    plt.show()


def visualize_edge_distances(edge_features, labels):
    distances = defaultdict(list)

    for protein_fts, label in zip(edge_features, labels):
        distances[label].append(protein_fts[:,0])

    prot_ids = list(range(min(labels), max(labels) + 1))

    fig = plt.figure(figsize=(15, 30))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    for prot_id in prot_ids:
        prot_distances = np.hstack(distances[prot_id])
        plt.subplot(6, 3, prot_id + 1)
        plt.hist(prot_distances, bins=50)
        plt.xlim(0, 12)
        plt.xlabel(f"Edge distance (max={prot_distances.max()}")
        plt.ylabel("Number of edges")
        plt.title(f"Protein type: {prot_id}")
    plt.show()

def visualize_edge_types(edge_features, labels):
    edge_types = defaultdict(list)

    for protein_fts, label in zip(edge_features, labels):
        edge_types[label].append(protein_fts[:, 1:].sum(axis=0))

    prot_ids = list(range(min(labels), max(labels) + 1))

    fig = plt.figure(figsize=(15, 30))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    index = ['Dist. edge', 'Pept. bond', 'kNN edge', 'H bond']
    for prot_id in prot_ids:
        prot_edge_types = np.vstack(edge_types[prot_id]).sum(axis=0)
        plt.subplot(6, 3, prot_id + 1)
        bar = plt.bar(index, prot_edge_types, align='center')
        plt.bar_label(bar)
        plt.xlabel("Edge type", )
        plt.ylabel("Number of edges")
        plt.xticks(rotation=25)
        plt.title(f"Protein type: {prot_id}")
    plt.show()


if __name__ == '__main__':
    from graph.data_handling.baseline_data_processing import load_graph_data, train_test_split

    adj, node_features, edge_features = load_graph_data()
    features_train, y_train, features_test, proteins_test = train_test_split(adj, node_features, edge_features)
    adj_train, node_features_train, edge_features_train = features_train
    adj_test, node_features_test, edge_features_test = features_test

    visualize_amino_acids(node_features_train, y_train)
    visualize_edge_distances(edge_features_train, y_train)
    visualize_edge_types(edge_features_train, y_train)