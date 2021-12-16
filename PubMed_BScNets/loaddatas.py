import torch_geometric.datasets
from torch_geometric.data import Data
import torch_geometric.transforms as T
import torch
import sys
import networkx as nx
import os
import numpy as np
import scipy.sparse as sp
import torch_geometric.datasets
from incidence_matrix import get_faces, incidence_matrices

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


def get_edges_split(data, val_prop = 0.2, test_prop = 0.2, seed = 1):
    g = nx.Graph()
    g.add_nodes_from([i for i in range(len(data.y))])
    _edge_index_ = np.array((data.edge_index))
    edge_index_ = [(_edge_index_[0, i], _edge_index_[1, i]) for i in
                        range(np.shape(_edge_index_)[1])]
    g.add_edges_from(edge_index_)
    adj = nx.adjacency_matrix(g)

    return get_adj_split(adj,val_prop = val_prop, test_prop = test_prop, seed = seed)


def get_adj_split(adj, val_prop=0.05, test_prop=0.1, seed = 1):
    np.random.seed(seed)  # get tp edges
    x, y = sp.triu(adj).nonzero()
    pos_edges = np.array(list(zip(x, y)))
    np.random.shuffle(pos_edges)
    neg_edges_ = np.load('pubmed_neg_edges.npz')['arr_0']
    indices = np.random.choice(neg_edges_.shape[0], neg_edges_.shape[0], replace=False)
    neg_edges = neg_edges_[indices, :]
    print("negative edges after shuffle: ", neg_edges.shape)

    m_pos = len(pos_edges)
    n_val = int(m_pos * val_prop)
    n_test = int(m_pos * test_prop)
    val_edges, test_edges, train_edges = pos_edges[:n_val], pos_edges[n_val:n_test + n_val], pos_edges[n_test + n_val:]
    val_edges_false, test_edges_false = neg_edges[:n_val], neg_edges[n_val:n_test + n_val]
    train_edges_false = np.concatenate([neg_edges, val_edges, test_edges], axis=0)
    return train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false


def compute_hodge_matrix(data, sample_data_edge_index):
    g = nx.Graph()
    g.add_nodes_from([i for i in range(len(data.y))])
    edge_index_ = np.array((sample_data_edge_index))
    edge_index = [(edge_index_[0, i], edge_index_[1, i]) for i in
                        range(np.shape(edge_index_)[1])]
    g.add_edges_from(edge_index)

    edge_to_idx = {edge: i for i, edge in enumerate(g.edges)}

    B1, B2 = incidence_matrices(g, sorted(g.nodes), sorted(g.edges), get_faces(g), edge_to_idx)

    L1_lower = B1.T @ B1
    L1_upper = B2 @ B2.T

    return L1_lower, L1_upper
