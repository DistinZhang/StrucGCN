import math
import os
import shutil
import random
from collections import ChainMap, deque
import numpy as np
import pandas as pd
from fastdtw import fastdtw
from joblib import Parallel, delayed
from collections import defaultdict
from utils import partition_dict, preprocess_nxgraph


class Struc2Vec:

    def __init__(self, graph, workers=4, verbose=40, opt1_reduce_len=True,
                 opt2_reduce_sim_calc=1, opt3_num_layers=None, temp_path='./temp_struc2vec/', reuse=False):
        self.graph = graph
        self.idx2node, self.node2idx = preprocess_nxgraph(graph)
        self.idx = list(range(len(self.idx2node)))
        self.opt1_reduce_len = opt1_reduce_len
        self.opt2_reduce_sim_calc = opt2_reduce_sim_calc
        self.opt3_num_layers = opt3_num_layers
        self.resue = reuse
        self.temp_path = temp_path
        if not os.path.exists(self.temp_path):
            os.mkdir(self.temp_path)
        if not reuse:
            shutil.rmtree(self.temp_path)
        os.mkdir(self.temp_path)

        pair_distances = self.compute_structural_distance(self.opt3_num_layers, workers, verbose, )
        self.layers_adj, self.layers_distances = self.get_layer_rep(pair_distances)

    def compute_ordered_degreelist(self, max_num_layers):
        degreeList = {}
        layerList = {}
        vertices = self.idx
        for v in self.graph.nodes:
            degreeList[v], layerList[v] = self.get_order_degreelist_node(v, max_num_layers)
        return degreeList, layerList

    def get_order_degreelist_node(self, root, max_num_layers=None):
        if max_num_layers is None:
            max_num_layers = float('inf')

        ordered_degree_sequence_dict = {}
        visited = [False] * len(self.graph.nodes())
        queue = deque()
        level = 0
        queue.append(root)
        visited[root] = True

        while len(queue) > 0 and level <= max_num_layers:
            count = len(queue)
            if self.opt1_reduce_len:
                degree_list = {}
            else:
                degree_list = []
            while count > 0:
                node = queue.popleft()
                degree = len(self.graph[node])

                if self.opt1_reduce_len:
                    degree_list[degree] = degree_list.get(degree, 0) + 1
                else:  # 直接保存度值
                    degree_list.append(degree)

                for nei in self.graph[node]:
                    if not visited[nei]:
                        visited[nei] = True
                        queue.append(nei)
                count -= 1
            if self.opt1_reduce_len:
                orderd_degree_list = [(degree, freq)
                                      for degree, freq in degree_list.items()]
                orderd_degree_list.sort(key=lambda x: x[0])
            else:
                orderd_degree_list = sorted(degree_list)
            ordered_degree_sequence_dict[level] = orderd_degree_list
            level += 1

        return ordered_degree_sequence_dict, level - 1

    def create_layers_vectors(self, layerList):
        layers = defaultdict(list)
        for key, value in layerList.items():
            layers[value].append(key)
        layers = dict(layers)
        return layers

    def compute_structural_distance(self, max_num_layers, workers=1, verbose=40):
        if os.path.exists(self.temp_path + 'structural_dist.pkl'):
            structural_dist = pd.read_pickle(
                self.temp_path + 'structural_dist.pkl')
        else:
            if self.opt1_reduce_len:
                dist_func = cost_max
            else:
                dist_func = cost
            if os.path.exists(self.temp_path + 'degreelist.pkl'):
                degreeList = pd.read_pickle(self.temp_path + 'degreelist.pkl')
                layerList = pd.read_pickle(self.temp_path + 'layerList.pkl')
            else:
                degreeList, layerList = self.compute_ordered_degreelist(max_num_layers)
                pd.to_pickle(degreeList, self.temp_path + 'degreelist.pkl')
                pd.to_pickle(degreeList, self.temp_path + 'layerList.pkl')

            if self.opt2_reduce_sim_calc != 0:
                degrees = self.create_vectors()
                layers = self.create_layers_vectors(layerList)
                degreeListsSelected = {}
                vertices = {}
                n_nodes = len(self.graph.nodes)
                for v in self.graph.nodes:  # c:list of vertex
                    nbs = get_vertices(v, len(self.graph[v]), degrees, layers, layerList, n_nodes, self.opt2_reduce_sim_calc)
                    vertices[v] = nbs  # store nbs
                    degreeListsSelected[v] = degreeList[v]  # store dist
                    for n in nbs:
                        # store dist of nbs
                        degreeListsSelected[n] = degreeList[n]
            else:
                vertices = {}
                for v in degreeList:
                    vertices[v] = [vd for vd in degreeList.keys() if vd > v]

            results = Parallel(n_jobs=workers, verbose=verbose, prefer="threads")(
                delayed(compute_dtw_dist)(part_list, degreeList, dist_func) for part_list in
                partition_dict(vertices, workers))
            dtw_dist = dict(ChainMap(*results))

            structural_dist = convert_dtw_struc_dist(dtw_dist)
            pd.to_pickle(structural_dist, self.temp_path + 'structural_dist.pkl')

        return structural_dist

    def create_vectors(self):
        degrees = {}  # sotre v list of degree
        degrees_sorted = set()  # store degree
        for v in self.graph.nodes:
            degree = len(self.graph[v])
            degrees_sorted.add(degree)
            if degree not in degrees:
                degrees[degree] = {}
                degrees[degree]['vertices'] = []
            degrees[degree]['vertices'].append(v)
        degrees_sorted = np.array(list(degrees_sorted), dtype='int')
        degrees_sorted = np.sort(degrees_sorted)

        for index, degree in enumerate(degrees_sorted):
            if index > 0:
                degrees[degree]['before'] = degrees_sorted[index - 1]
            if index < (len(degrees_sorted) - 1):
                degrees[degree]['after'] = degrees_sorted[index + 1]

        return degrees

    def get_layer_rep(self, pair_distances):
        layer_distances = {}
        layer_adj = {}
        for v_pair, layer_dist in pair_distances.items():
            for layer, distance in layer_dist.items():
                vx = v_pair[0]
                vy = v_pair[1]

                layer_distances.setdefault(layer, {})
                layer_distances[layer][vx, vy] = distance

                layer_adj.setdefault(layer, {})
                layer_adj[layer].setdefault(vx, [])
                layer_adj[layer][vx].append(vy)

        return layer_adj, layer_distances

    def get_similarity_weigh(self):
        similarity = defaultdict(set)
        weigh = {}
        for layer in self.layers_adj:
            neighbors = self.layers_adj[layer]
            layer_distances = self.layers_distances[layer]
            values = list(layer_distances.values())
            sorted_unique_values = sorted(set(values))
            if len(sorted_unique_values) > 1:
                d_s_min = sorted_unique_values[1]
            else:
                d_s_min = sorted_unique_values[0]
            for v, neighbor in neighbors.items():
                for n in neighbor:
                    wd = layer_distances[v, n]
                    if wd <= d_s_min:
                        w = 1
                    else:
                        w = np.exp(-(wd / d_s_min - 1))
                    if w > 0.01:
                        similarity[v].add(n)
                        similarity[n].add(v)
                        weigh[(v, n)] = w
                        weigh[(n, v)] = w

        pd.to_pickle(similarity, self.temp_path + 'similarity.pkl')
        pd.to_pickle(weigh, self.temp_path + 'weigh.pkl')
        return similarity, weigh


def cost(a, b):
    ep = 0.5
    m = max(a, b) + ep
    mi = min(a, b) + ep
    return (m / mi) - 1


def cost_min(a, b):
    ep = 0.5
    m = max(a[0], b[0]) + ep
    mi = min(a[0], b[0]) + ep
    return ((m / mi) - 1) * min(a[1], b[1])


def cost_max(a, b):
    ep = 0.5
    m = max(a[0], b[0]) + ep
    mi = min(a[0], b[0]) + ep
    return ((m / mi) - 1) * max(a[1], b[1])


def convert_dtw_struc_dist(distances, startLayer=1):
    """
    :param distances: dict of dict
    :param startLayer:
    :return:
    """
    distances_all = dict()
    for vertices, layers in distances.items():
        keys_layers = sorted(layers.keys())
        startLayer = min(len(keys_layers), startLayer)
        if max(keys_layers) > 0:
            for layer in range(0, startLayer):
                keys_layers.pop(0)
            for layer in keys_layers:
                layers[layer] += layers[layer - 1]
        distances_all[vertices] = {max(keys_layers): layers[max(keys_layers)]}
    return distances_all


def get_vertices(v, degree_v, degrees, layers, layerList, n_nodes, flag):
    a_vertices_selected = int(2 * math.log(n_nodes, 2) + 1)
    vertices = []
    layer = layerList[v]
    new_dict = sorted(set(degrees[degree_v]['vertices']).intersection(set(layers[layer])))
    try:
        c_v = 0
        if flag == 1:
            if len(new_dict) > 1:
                for v2 in list(new_dict):
                    if v < v2:
                        vertices.append(v2)  # same degree
                    if v2 == list(new_dict)[-1]:
                        raise StopIteration
            else:
                raise StopIteration
        else:
            if a_vertices_selected - c_v >= len(degrees[degree_v]['vertices']):
                for v2 in degrees[degree_v]['vertices']:
                    if v != v2:
                        vertices.append(v2)  # same degree
                        c_v += 1
                        if c_v > a_vertices_selected:
                            raise StopIteration
            else:
                selected_items = random.sample(degrees[degree_v]['vertices'], a_vertices_selected - c_v + 1)
                for v2 in selected_items:
                    if v != v2:
                        vertices.append(v2)  # same degree
                        c_v += 1
                        if c_v > a_vertices_selected:
                            raise StopIteration
        if 'before' not in degrees[degree_v]:
            degree_b = -1
        else:
            degree_b = degrees[degree_v]['before']
        if 'after' not in degrees[degree_v]:
            degree_a = -1
        else:
            degree_a = degrees[degree_v]['after']
        if degree_b == -1 and degree_a == -1:
            raise StopIteration  # not anymore v
        degree_now = verifyDegrees(degrees, degree_v, degree_a, degree_b)
        # nearest valid degree
        while True:
            for v2 in degrees[degree_now]['vertices']:
                if v != v2:
                    vertices.append(v2)
                    c_v += 1
                    if c_v > a_vertices_selected:
                        raise StopIteration
            if degree_now == degree_b:
                if 'before' not in degrees[degree_b]:
                    degree_b = -1
                else:
                    degree_b = degrees[degree_b]['before']
            else:
                if 'after' not in degrees[degree_a]:
                    degree_a = -1
                else:
                    degree_a = degrees[degree_a]['after']
            if degree_b == -1 and degree_a == -1:
                raise StopIteration
            degree_now = verifyDegrees(degrees, degree_v, degree_a, degree_b)

    except StopIteration:
        return list(vertices)

    return list(vertices)


def verifyDegrees(degrees, degree_v_root, degree_a, degree_b):
    if degree_b == -1:
        degree_now = degree_a
    elif degree_a == -1:
        degree_now = degree_b
    elif abs(degree_b - degree_v_root) < abs(degree_a - degree_v_root):
        degree_now = degree_b
    else:
        degree_now = degree_a

    return degree_now


def compute_dtw_dist(part_list, degreeList, dist_func):
    dtw_dist = {}
    for v1, nbs in part_list:
        lists_v1 = degreeList[v1]  # lists_v1 :orderd degree list of v1
        for v2 in nbs:
            lists_v2 = degreeList[v2]  # lists_v1 :orderd degree list of v2
            if len(lists_v1) == len(lists_v2):
                dtw_dist[v1, v2] = {}
                max_layer = len(lists_v1)  # valid layer
                for layer in range(0, max_layer):
                    dist, path = fastdtw(lists_v1[layer], lists_v2[layer], radius=1, dist=dist_func)
                    dtw_dist[v1, v2][layer] = dist
    return dtw_dist
