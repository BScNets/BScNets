import torch_geometric.datasets
from torch_geometric.data import Data
import torch_geometric.transforms as T
import torch
import sys
import networkx as nx
import os
import numpy as np
import scipy.sparse as sp
from torch_geometric.data import Data
import torch_geometric.datasets
from incidence_matrix import get_faces, incidence_matrices

path = os.getcwd()

def loaddatas(d_name):
    if d_name in ["PPI"]:
        dataset = torch_geometric.datasets.PPI('./data/' + d_name)
    elif d_name == 'Cora':
        dataset = torch_geometric.datasets.Planetoid('./data/'+d_name,d_name,transform=T.NormalizeFeatures())
    elif d_name in ['Citeseer', 'PubMed']:
        dataset = torch_geometric.datasets.Planetoid('./data/' + d_name, d_name)
    elif d_name in ["Computers", "Photo"]:
        dataset = torch_geometric.datasets.Amazon('./data/'+d_name,d_name)
    return dataset


def load_synthetic_data(dataset_str, use_feats, data_path):
    object_to_idx = {}
    idx_counter = 0
    edges = []
    with open(os.path.join(data_path, "{}.edges.csv".format(dataset_str)), 'r') as f:
        all_edges = f.readlines()
    for line in all_edges:
        n1, n2 = line.rstrip().split(',')
        if n1 in object_to_idx:
            i = object_to_idx[n1]
        else:
            i = idx_counter
            object_to_idx[n1] = i
            idx_counter += 1
        if n2 in object_to_idx:
            j = object_to_idx[n2]
        else:
            j = idx_counter
            object_to_idx[n2] = j
            idx_counter += 1
        edges.append((i, j))
    adj = np.zeros((len(object_to_idx), len(object_to_idx)))
    for i, j in edges:
        adj[i, j] = 1.  # comment this line for directed adjacency matrix
        adj[j, i] = 1.
    if use_feats:
        features = sp.load_npz(os.path.join(data_path, "{}.feats.npz".format(dataset_str)))
    else:
        features = sp.eye(adj.shape[0])
    labels = np.load(os.path.join(data_path, "{}.labels.npy".format(dataset_str)))

    features_ = features.toarray()
    features = torch.FloatTensor(features_) # torch.float32
    y = torch.LongTensor(labels)

    return adj, features, y

adj, features, y = load_synthetic_data('disease_lp', use_feats= True, data_path= path + "/disease_lp")

def get_disease_edgelist(adj):
    g = nx.from_numpy_matrix(adj)
    edges = list(g.edges())
    edge_index = torch.zeros(size=(2, len(edges) * 2), dtype=torch.int64)
    for i in range(len(edges)):
        u, v = edges[i]
        edge_index[0, i] = u
        edge_index[1, i] = v

    for j in range(len(edges)):
        u, v = edges[j]
        edge_index[0, j+len(edges)] = v
        edge_index[1, j+len(edges)] = u

    return edge_index


def get_edges_split(dataset_str, use_feats, data_path, val_prop = 0.05, test_prop = 0.10):

    adj, features, labels = load_synthetic_data(dataset_str, use_feats, data_path)

    return get_adj_split(sp.csr_matrix(adj), val_prop = val_prop, test_prop = test_prop) # conver to adj to sparse version


def get_adj_split(adj, val_prop=0.05, test_prop=0.10):
    x, y = sp.triu(adj).nonzero()
    pos_edges = np.array(list(zip(x, y)))
    np.random.shuffle(pos_edges)
    x, y = sp.triu(sp.csr_matrix(1. - adj.toarray())).nonzero()
    neg_edges = np.array(list(zip(x, y)))
    np.random.shuffle(neg_edges)

    m_pos = len(pos_edges)
    n_val = int(m_pos * val_prop)
    n_test = int(m_pos * test_prop)
    
    val_edges, test_edges, train_edges = pos_edges[:n_val], pos_edges[n_val:n_test + n_val], pos_edges[n_test + n_val:]
    val_edges_false, test_edges_false = neg_edges[:n_val], neg_edges[n_val:n_test + n_val]
    train_edges_false = np.concatenate([neg_edges, val_edges, test_edges], axis=0)
    return train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false


def compute_hodge_matrix(data):
    g = nx.Graph()
    g.add_nodes_from([i for i in range(len(data.y))])
    edge_index_ = np.array((data.edge_index))
    edge_index = [(edge_index_[0, i], edge_index_[1, i]) for i in
                        range(np.shape(edge_index_)[1])]
    g.add_edges_from(edge_index)

    edge_to_idx = {edge: i for i, edge in enumerate(g.edges)}

    B1, B2 = incidence_matrices(g, sorted(g.nodes), sorted(g.edges), get_faces(g), edge_to_idx)

    return B1, B2
