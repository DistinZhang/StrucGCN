import os
import numpy as np
import time
import pandas as pd
import networkx as nx
import scipy.sparse as sp
import torch_geometric
import torch
import pickle
import sys
from deepwalk import DeepWalk
from node2vec import Node2Vec
from role_discovery import Role
from collections import defaultdict


def prepare_data(args):
    if args.dataset in {'cora', 'citeseer', 'pubmed'}:
        node_map, g, features, labels = load_data0(args)
    elif args.dataset in {'brazil', 'europe', 'usa'}:
        node_map, g, features, labels = load_data1(args)
    elif args.dataset in {'actor', 'chameleon', 'cornell', 'squirrel', 'texas', 'wisconsin'}:
        node_map, g, features, labels = load_data2(args)

    return node_map, g, features, labels


def load_data0(args):
    """
    Loads the data from the files and returns the necessary data structures.
    :param dataset_str: The name of the dataset (e.g., 'cora', 'citeseer', 'pubmed').
    :return: A tuple of (labels, features, node_map, graph).
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    data_path = '../data/0'
    for i in range(len(names)):
        with open(os.path.join(data_path, f"ind.{args.dataset}.{names[i]}"), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pickle.load(f, encoding='latin1'))
            else:
                objects.append(pickle.load(f))

    x, y, tx, ty, allx, ally, graph_data = tuple(objects)
    test_idx_reorder = parse_index_file(os.path.join(data_path, f'ind.{args.dataset}.test.index'))
    test_idx_range = np.sort(test_idx_reorder)

    if args.dataset == 'citeseer':
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    features = features.toarray().astype(np.float32)

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    labels_class = np.argmax(labels, axis=1).reshape(-1, 1)
    node_map = {i: i for i in range(len(labels_class))}
    pd.to_pickle(node_map, '../output/' + args.dataset + '/node_map.pkl')
    g = nx.from_dict_of_lists(graph_data)

    if args.feat_flag is False:
        features = create_node_attribute_vector(args, g, 0)
    return node_map, g, features, labels_class


def load_data1(args):
    label_path = os.path.join('../data/1/Category/' + args.dataset + '.txt')
    edge_path = os.path.join('../data/1/Edgelist/' + args.dataset + '.edgelist')
    node_map = {}
    label_map = {}
    g = nx.Graph()
    with open(label_path) as fp:
        for idx, line in enumerate(fp):
            info = line.strip().split()
            node_map[info[0]] = idx
            label_map[idx] = info[-1]
    label_to_int = {value: index for index, value in enumerate(set(label_map.values()))}
    labels = np.empty((len(node_map), 1), dtype=np.int64)
    for idx in range(len(node_map)):
        labels[idx] = label_to_int[label_map[idx]]
    pd.to_pickle(node_map, '../output/' + args.dataset + '/node_map.pkl')
    with open(edge_path) as fp:
        for idx, line in enumerate(fp):
            info = line.strip().split()
            x = node_map[info[0]]
            y = node_map[info[1]]
            g.add_edge(x, y)

    features = create_node_attribute_vector(args, g, 1)

    return node_map, g, features, labels


def load_data2(args):
    graph_adjacency_list_file_path = os.path.join('../data/2/', args.dataset, 'out1_graph_edges.txt')
    graph_node_features_and_labels_file_path = os.path.join('../data/2/', args.dataset,
                                                            f'out1_node_feature_label.txt')
    g = nx.Graph()
    features = {}
    node_map = {}
    label_map = {}
    idx = 0
    with open(graph_node_features_and_labels_file_path) as graph_node_features_and_labels_file:
        graph_node_features_and_labels_file.readline()
        for line in graph_node_features_and_labels_file:
            line = line.rstrip().split('\t')
            assert (len(line) == 3)
            assert (int(line[0]) not in features and int(line[0]) not in label_map)
            features[idx] = np.array(line[1].split(','), dtype=np.float32)
            node_map[line[0]] = idx
            label_map[idx] = line[2]
            idx += 1
    label_to_int = {value: index for index, value in enumerate(set(label_map.values()))}
    labels = np.empty((len(node_map), 1), dtype=np.int64)
    for idx in range(len(node_map)):
        labels[idx] = label_to_int[label_map[idx]]
    pd.to_pickle(node_map, '../output/' + args.dataset + '/node_map.pkl')
    idx = 0
    with open(graph_adjacency_list_file_path) as graph_adjacency_list_file:
        graph_adjacency_list_file.readline()
        for line in graph_adjacency_list_file:
            line = line.rstrip().split('\t')
            assert (len(line) == 2)
            x = node_map[line[0]]
            y = node_map[line[1]]
            g.add_edge(x, y)
            idx += 1
    features = list(features.values())

    if args.feat_flag is False:
        features = create_node_attribute_vector(args, g, 2)
    return node_map, g, features, labels


def generate_structure_similar_matrix(args, g, flag='load'):
    if flag == 'load':
        role_adj_list = pd.read_pickle('../output/' + args.dataset + '/role_adj_list.pkl')  # 0.01
        weigh_list = pd.read_pickle('../output/' + args.dataset + '/weigh_list.pkl')
    else:
        t = time.time()
        model = Role(args, g)
        role_adj_list, weigh_list = model.get_role_adj_weigh()
        pd.to_pickle(role_adj_list, '../output/' + args.dataset + '/role_adj_list.pkl')
        pd.to_pickle(weigh_list, '../output/' + args.dataset + '/weigh_list.pkl')
        print('Total Time：', time.time() - t)
    return role_adj_list, weigh_list


def create_node_attribute_vector(args, graph, ID):
    path = os.path.join('../data/' + str(ID) + '/Feature/' + args.dataset + '_feat_data.pkl')
    if os.path.exists(path):
        print('loading features')
        feat_data = pd.read_pickle(path)
    else:
        if args.feat_generate == 'onehot':
            feat_data = np.eye(len(graph.nodes), dtype=np.float32)
        else:
            print(f'generating features by {args.feat_generate}, dim is {args.embedding_dim}')
            if args.feat_generate == 'node2vec':
                model = Node2Vec(graph, 10, 80, workers=18, p=0.25, q=2, use_rejection_sampling=0)
            elif args.feat_generate == 'deepwalk':
                model = DeepWalk(graph, walk_length=10, num_walks=80, workers=18)                
            model.train(embed_size=args.embedding_dim)
            embeddings = model.get_embeddings()
            feat_data = np.zeros((len(graph.nodes), args.embedding_dim))
            for i in range(len(graph.nodes)):
                feat_data[i] = embeddings[i]
            feat_data = feat_data.astype(np.float32)
        pd.to_pickle(feat_data, path)
        print(feat_data)
    return feat_data


def prepare_dataset(args, labels, i):
    num = len(labels)
    train_num = int(np.ceil(args.train_ra * num / 100.0))
    val_num = int(np.ceil(args.val_ra * num / 100.0))
    path = os.path.join('./tvtsplit/' + args.dataset + '/' + str(args.train_ra/100) + '/' + args.dataset + '_tvt_index' + str(i) + '.npy')
    if os.path.exists(path):
        rand_indices = np.load(path)
    else:
        rand_indices = np.random.permutation(num)
        np.save(path, rand_indices)
    rand_indices = rand_indices.astype(int)
    train_idx = rand_indices[:train_num]
    val_idx = rand_indices[train_num:train_num + val_num]
    test_idx = rand_indices[train_num + val_num:]
    return train_idx, val_idx, test_idx

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def get_data(graph, feat_data, labels, role_adj_list, weigh_list):
    num_feats = len(feat_data[0])
    num_labels = np.max(labels) + 1
    neigh_edge_index = torch.tensor(list(zip(*graph.edges)), dtype=torch.long)
    neigh_edge_index = torch_geometric.utils.to_undirected(neigh_edge_index)
    neigh_edge_index, _ = torch_geometric.utils.remove_self_loops(neigh_edge_index)
    x = torch.tensor(feat_data)
    y = torch.tensor(labels, dtype=torch.long)
    y = y.view(-1)
    neigh_data = torch_geometric.data.Data(x=x, edge_index=neigh_edge_index, y=y, edge_weight=torch.ones(neigh_edge_index.size(1), dtype=torch.float), num_feats=num_feats,
                                           num_labels=num_labels, num_nodes=len(graph.nodes))

    edges = []
    weights = []
    for node, neighbors in role_adj_list.items():
        for neighbor in neighbors:
            edges.append((node, neighbor))
            weights.append(weigh_list[(node, neighbor)])
    struc_edge_index = torch.tensor(list(zip(*edges)), dtype=torch.long)
    edge_weight = torch.tensor(weights, dtype=torch.float)
    struc_data = torch_geometric.data.Data(x=x, edge_index=struc_edge_index, y=y, edge_weight=edge_weight, num_feats=num_feats, num_labels=num_labels, num_nodes=len(graph.nodes))

    return neigh_data, struc_data

def get_merged_data(args, neigh_data, struc_data):
    num_nodes = neigh_data.x.size(0)
    edge_index = torch.arange(0, num_nodes, dtype=torch.long).unsqueeze(0).repeat(2, 1)
    self_data = torch_geometric.data.Data(x=neigh_data.x, y=neigh_data.y, edge_index=edge_index)
    edge_dict = {}

    def add_edges(data, weight_factor):
        edge_index = data.edge_index
        edge_weight = data.edge_weight if 'edge_weight' in data else torch.ones(edge_index.size(1), dtype=torch.float)
        for i in range(edge_index.size(1)):
            u, v = edge_index[0, i].item(), edge_index[1, i].item()
            weight = edge_weight[i].item() * weight_factor
            if (u, v) in edge_dict:
                edge_dict[(u, v)] += weight
            else:
                edge_dict[(u, v)] = weight

    if args.alpha != 0:
        add_edges(self_data, args.alpha)
    if args.beta != 0:
        add_edges(neigh_data, args.beta)
    if args.gama != 0:
        add_edges(struc_data, args.gama)

    merged_edge_index = torch.tensor(list(edge_dict.keys()), dtype=torch.long).t()
    merged_edge_weight = torch.tensor(list(edge_dict.values()), dtype=torch.float)
    return torch_geometric.data.Data(x=neigh_data.x, y=neigh_data.y, edge_index=merged_edge_index, edge_weight=merged_edge_weight, num_feats=neigh_data.num_feats, num_labels=neigh_data.num_labels, num_nodes=neigh_data.num_nodes)


def calculate_edge_homophily(data):
    same_label_edges = 0
    total_edges = 0
    data = data.cpu()
    neigh_edge_index_np = data.edge_index.numpy()
    edges = list(zip(neigh_edge_index_np[0], neigh_edge_index_np[1]))
    edge_weight = data.edge_weight.tolist()
    weight = {}
    for i in range(len(edges)):
        weight[edges[i]] = edge_weight[i]
    node_labels = data.y.tolist()
    for edge in edges:
        node1, node2 = edge
        if node1 != node2:
            if node_labels[node1] == node_labels[node2]:
                same_label_edges += weight[(node1,node2)]
            total_edges += weight[(node1,node2)]

    if total_edges == 0:
        return 0

    edge_homophily_ratio = same_label_edges / total_edges
    return edge_homophily_ratio


def calculate_node_homophily(data):
    data = data.cpu()
    neigh_edge_index_np = data.edge_index.numpy()
    edges = list(zip(neigh_edge_index_np[0], neigh_edge_index_np[1]))
    adj_list = defaultdict(set)
    edge_weight = data.edge_weight.tolist()
    weight = {}
    for i in range(len(edges)):
        weight[edges[i]] = edge_weight[i]
    node_labels = data.y.tolist()
    for edge in edges:
        node1, node2 = edge
        if node1 != node2:
            adj_list[node1].add(node2)
            adj_list[node2].add(node1)

    homophily_ratios = []

    for node, neighbors in adj_list.items():
        if len(neighbors) == 0:
            continue
        same_label_count = sum(weight[(node,neighbor)] for neighbor in neighbors if node_labels[neighbor] == node_labels[node])
        fenmu = sum(weight[(node,neighbor)] for neighbor in neighbors)
        homophily_ratio = same_label_count/fenmu
        homophily_ratios.append(homophily_ratio)

    if len(homophily_ratios) == 0:
        return 0

    average_homophily = (np.sum(homophily_ratios))/(data.num_nodes)
    return average_homophily

