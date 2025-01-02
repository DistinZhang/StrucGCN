"""RoleBased2Vec Machine."""

import math
import numpy as np
import networkx as nx
import random
from sklearn.cluster import KMeans
from motif_count import MotifCounterMachine
from weisfeiler_lehman_labeling import WeisfeilerLehmanMachine
import warnings
from struc2vec import Struc2Vec
warnings.filterwarnings("ignore")

def get_roles_nodes(data):
    value_list = set([i[0] for i in data.values()])
    roles_nodes = {role: [] for role in value_list}

    for role in value_list:
        for node in data:
            if data[str(node)][0] == role:
                # if data[node][0] == role:
                roles_nodes[role].append(int(node))
    return roles_nodes

def join_strings(features):
    """
    Creating string labels by joining the individual quantile labels.
    """
    return {str(node): ["_".join(features[node])] for node in features}  # str(node)

def get_feature_vec(data):
    n_tot = len(data)
    node = random.choice(list(data.keys()))
    dimensions = len(data[node])
    feature_vec = np.empty((n_tot, dimensions), dtype='f')  #
    for ii in range(n_tot):
        v = sorted(data.keys())[ii]
        feature_vec[ii] = [int(i) for i in data[v]]
    return feature_vec

def KM_train(data, n=10):
    vector = get_feature_vec(data)
    # print('start KMeans')
    km = KMeans(init='k-means++', n_clusters=n, n_init=10, random_state=42)  # n_jobs = -1
    model = km.fit(vector)
    # labels = model.predict(vector)
    labels = model.labels_
    features = {str(sorted(data.keys())[node]): [str(labels[node])] for node in range(len(data))}
    # print('finish KMeans')
    return features


class Role:
    def __init__(self, args, G):
        self.args = args
        self.G = G
        self.is_directed = False
        self.roles_nodes = None
        self.structura_features = None

    def get_role_adj_weigh(self):
        """
        Extracting structural features.
        """
        graph = self.G
        if self.args.role_generate == "wl":
            print('We are using WL...')
            features = {str(node): str(int(math.log(graph.degree(node) + 1, self.args.log_base))) for node in
                        graph.nodes()}
            machine = WeisfeilerLehmanMachine(graph, features, self.args.labeling_iterations)
            machine.do_recursions()
            features = machine.extracted_features
            # features = join_strings(features)
        elif self.args.role_generate == 'motif':  # 复杂度太高
            machine = MotifCounterMachine(graph, self.args)
            features = machine.create_string_labels()
        elif self.args.role_generate == 'motif_tri':     # [度，三角]。可以计算一下相似性
            print('We are using motif_tri...,k = {}'.format(self.args.clusters))
            features_d = {str(node): [str(int(math.log(graph.degree(node) + 1, self.args.log_base)))] for node in
                          graph.nodes()}
            # features_d = {str(node): [str(graph.degree(node))] for node in graph.nodes()}
            tr = nx.triangles(graph)
            features_tr = {str(node): str(int(math.log(tr[node] + 1, self.args.log_base))) for node in tr}
            # features_tr = {str(node): str(tr[node]) for node in tr}
            for node in features_d:
                features_d[node].append(features_tr[node])
            ###string feature
            # features = features_d #join_strings(features_d)
            features = KM_train(features_d, n=self.args.clusters)
        elif self.args.role_generate == 'struc2vec':
            print('We are using struc2vec for role discovery')
            model = Struc2Vec(graph, workers=self.args.workers)
            role_adj_list, weigh_list = model.get_similarity_weigh()
        else:
            print('We are using degree...')  # [度]
            features = {str(node): [str(graph.degree(node))] for node in graph.nodes()}
            # features = {str(node): str(int(math.log(graph.degree(node)+1, self.args.log_base))) for node in
            # graph.nodes()}
        return role_adj_list, weigh_list

    # 基于角色分类
    def get_role_node(self, is_load_feature=False):
        from tqdm import tqdm
        if is_load_feature:
            print('loading the structural features........')
            self.structura_features = np.load(self.args.output + self.args.dataset + 'structure_features' + '.npy',
                                              allow_pickle=True).item()
            self.roles_nodes = get_roles_nodes(self.structura_features)
        else:
            self.structura_features = self.create_graph_structural_features(self.G)
            self.roles_nodes = get_roles_nodes(self.structura_features)

        return self.roles_nodes
