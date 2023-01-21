from sequence_baseline import load_sequence_data
from graph.data_handling.baseline_data_processing import load_graph_data
from collections import Counter


def match_amino_acids():
    sequences_train, y_train, sequences_test, proteins_test = load_sequence_data()
    adj, node_features, edge_features = load_graph_data()

    for sequence, node_features in zip(sequences_train, node_features):
        one_hot_aa = node_features[:,3:23]
        counts = Counter(sequence)

        break



if __name__ == '__main__':
    match_amino_acids()