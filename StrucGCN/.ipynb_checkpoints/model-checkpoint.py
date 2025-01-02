import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree


class StrucGNN(nn.Module):
    def __init__(self, args, neigh_data, struc_data, merged_data):
        super(StrucGNN, self).__init__()
        self.args = args
        self.neigh_data = neigh_data
        self.struc_data = struc_data
        self.merged_data = merged_data
        if neigh_data is None:
            self.num_features = merged_data.num_features
            self.num_labels = merged_data.num_labels
        else:
            self.num_features = neigh_data.num_features
            self.num_labels = neigh_data.num_labels
        self.gcn = args.gcn  # GCN模式
        self.weight = nn.Parameter(torch.FloatTensor(self.num_labels, args.h_dim))
        self.dropout = args.dr
        init.kaiming_uniform_(self.weight)

        self.conv1 = Encoder(args, self.num_features, args.h_dim, self.neigh_data, self.struc_data, self.merged_data)
        self.conv2 = Encoder(args, args.h_dim, args.h_dim, self.neigh_data, self.struc_data, self.merged_data)

    def forward(self):
        h1 = self.conv1(self.neigh_data.x)
        h2 = self.conv2(h1)
        h = self.weight.mm(h2.t()).t()
        return F.log_softmax(h)


class Encoder(nn.Module):
    def __init__(self, args, in_dim, out_dim, neigh_data, struc_data, merged_data):
        super(Encoder, self).__init__()
        self.alpha = args.alpha  # 单位矩阵权重，default=0.5
        self.beta = args.beta  # 邻接矩阵权重 =0表示GCN
        self.gama = args.gama  # 相似矩阵权重
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.neigh_data = neigh_data
        self.struc_data = struc_data
        self.merged_data = merged_data
        self.gcn = args.gcn
        self.aggr = args.aggr
        self.dropout_rate = args.dr
        self.aggregate = Aggregator(self.in_dim, self.out_dim, normalize=args.normalize)

        # if args.gcn:
        #     if self.aggr == 'mean_concat':
        #         assert (self.alpha == 0) + (self.beta == 0) + (self.gama == 0) == 1 and \
        #                (self.alpha + self.beta + self.gama == 1)
        #         self.weight = nn.Parameter(torch.FloatTensor(self.out_dim, 2 * self.in_dim))
        #     else:
        #         assert (self.alpha == 0) + (self.beta == 0) + (self.gama == 0) <= 1 and \
        #                (self.alpha + self.beta + self.gama == 1)
        #         self.weight = nn.Parameter(torch.FloatTensor(self.out_dim, self.in_dim))
        # else:
        #     if self.aggr == 'mean':
        #         assert (self.alpha == 0) + (self.beta == 0) + (self.gama == 0) <= 1 and \
        #                (self.alpha + self.beta + self.gama == 1)
        #         self.weight = nn.Parameter(torch.FloatTensor(self.out_dim, self.in_dim))
        #     elif self.aggr == 'concat':
        #         if self.alpha * self.beta * self.gama == 0:
        #             self.weight = nn.Parameter(torch.FloatTensor(self.out_dim, 2 * self.in_dim))
        #         else:
        #             self.weight = nn.Parameter(torch.FloatTensor(self.out_dim, 3 * self.in_dim))
        #     elif self.aggr == 'mean_concat':
        #         assert (self.alpha == 0) + (self.beta == 0) + (self.gama == 0) == 1 and \
        #                (self.alpha + self.beta + self.gama == 1)
        #         self.weight = nn.Parameter(torch.FloatTensor(self.out_dim, 2 * self.in_dim))
        assert (self.alpha == 0) + (self.beta == 0) + (self.gama == 0) == 1 and \
               (self.alpha + self.beta + self.gama == 1)
        self.weight = nn.Parameter(torch.FloatTensor(self.out_dim, 2 * self.in_dim))

        init.kaiming_uniform_(self.weight)

    def forward(self, input_features):
        combined = self.aggregate(input_features, self.merged_data.edge_index,
                                  edge_weight=self.merged_data.edge_weight)
        if self.alpha == 0:  # model 3
            self_feats = input_features
            combined = torch.cat([self_feats, combined], dim=1)
        elif self.beta == 0:  # model 2
            neigh_feats = self.aggregate(input_features, self.neigh_data.edge_index, edge_weight=self.neigh_data.edge_weight)
            combined = torch.cat([combined, neigh_feats], dim=1)
        elif self.gama == 0:  # model 1
            struc_feats = self.aggregate(input_features, self.struc_data.edge_index,
                                         edge_weight=self.struc_data.edge_weight)
            combined = torch.cat([combined, struc_feats], dim=1)
        combined = F.relu(self.weight.mm(combined.t())).t()  # sigmoid\relu\tanh
        combined = F.dropout(combined, p=self.dropout_rate, training=self.training)  # 应用 dropout
        return combined

    # def forward(self, input_features):
    #     if self.gcn:
    #         if self.aggr == 'mean_concat':
    #             combined = self.aggregate(input_features, self.merged_data.edge_index,
    #                                       edge_weight=self.merged_data.edge_weight)
    #             if self.alpha == 0:  # model 3
    #                 self_feats = input_features
    #                 combined = torch.cat([self_feats, combined], dim=1)
    #             elif self.beta == 0:  # model 2
    #                 neigh_feats = self.aggregate(input_features, self.neigh_data.edge_index, edge_weight=self.neigh_data.edge_weight)
    #                 combined = torch.cat([combined, neigh_feats], dim=1)
    #             elif self.gama == 0:  # model 1
    #                 struc_feats = self.aggregate(input_features, self.struc_data.edge_index,
    #                                              edge_weight=self.struc_data.edge_weight)
    #                 combined = torch.cat([combined, struc_feats], dim=1)
    #         elif self.aggr == 'mean':  # model 4567
    #             combined = self.aggregate(input_features, self.merged_data.edge_index,
    #                                       edge_weight=self.merged_data.edge_weight)
    #     else:
    #         assert (self.alpha + self.beta + self.gama <= 1)
    #         if self.alpha > 0 or self.aggr == 'mean_concat':
    #             self_feats = input_features
    #         if self.beta > 0 or self.aggr == 'mean_concat':
    #             neigh_feats = self.aggregate(input_features, self.neigh_data.edge_index, edge_weight=self.neigh_data.edge_weight)
    #         if self.gama > 0 or self.aggr == 'mean_concat':
    #             struc_feats = self.aggregate(input_features, self.struc_data.edge_index,
    #                                          edge_weight=self.struc_data.edge_weight)
    #         if self.aggr == 'concat':
    #             if self.alpha == 0:  # model 8
    #                 combined = torch.cat([neigh_feats, struc_feats], dim=1)
    #             elif self.beta == 0:  # model 9
    #                 combined = torch.cat([self_feats, struc_feats], dim=1)
    #             elif self.gama == 0:  # model 10
    #                 combined = torch.cat([self_feats, neigh_feats], dim=1)
    #             else:  # model 11
    #                 combined = torch.cat([self_feats, neigh_feats, struc_feats], dim=1)
    #         elif self.aggr == 'mean':
    #             if self.alpha == 0:  # model 12
    #                 combined = self.beta * neigh_feats + self.gama * struc_feats
    #             elif self.beta == 0:  # model 13
    #                 combined = self.alpha * self_feats + self.gama * struc_feats
    #             elif self.gama == 0:  # model 14
    #                 combined = self.alpha * self_feats + self.beta * neigh_feats
    #             else:  # model 15
    #                 combined = self.alpha * self_feats + self.beta * neigh_feats + self.gama * struc_feats
    #         elif self.aggr == 'mean_concat':
    #             if self.alpha == 0:  # model 16
    #                 assert (self.beta != 0 and self.gama != 0)
    #                 combined = self.beta * neigh_feats + self.gama * struc_feats
    #                 combined = torch.cat([self_feats, combined], dim=1)
    #             elif self.beta == 0:  # model 17
    #                 assert (self.alpha != 0 and self.gama != 0)
    #                 combined = self.alpha * self_feats + self.gama * struc_feats
    #                 combined = torch.cat([combined, neigh_feats], dim=1)
    #             elif self.gama == 0:  # model 18
    #                 assert (self.alpha != 0 and self.beta != 0)
    #                 combined = self.alpha * self_feats + self.beta * neigh_feats
    #                 combined = torch.cat([combined, struc_feats], dim=1)
    #     combined = F.relu(self.weight.mm(combined.t())).t()  # sigmoid\relu\tanh
    #     combined = F.dropout(combined, p=self.dropout_rate, training=self.training)  # 应用 dropout
    #     return combined


class Aggregator(MessagePassing):
    def __init__(self, improved=False, cached=False, normalize='degree', **kwargs):
        super(Aggregator, self).__init__(aggr='add', **kwargs)  # "Add" aggregation (Step 5).
        self.improved = improved
        self.cached = cached
        self.normalize = normalize

    def forward(self, x, edge_index, edge_weight=None):
        if self.normalize == 'degree':
            row, col = edge_index
            deg = degree(row, x.size(0), dtype=x.dtype)
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
            norm = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col] if edge_weight is not None else deg_inv_sqrt[row] * deg_inv_sqrt[col]
        else:
            row, col = edge_index
            edge_weight_sum = torch.zeros_like(degree(row, x.size(0), dtype=edge_weight.dtype))
            edge_weight_sum.scatter_add_(0, row, edge_weight)
            norm = edge_weight / edge_weight_sum[row]

        out = self.propagate(edge_index, x=x, norm=norm)
        return out

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        return aggr_out
