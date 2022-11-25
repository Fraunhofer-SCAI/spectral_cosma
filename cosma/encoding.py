from .globals import *

import torch
from torch_geometric.nn import ChebConv
import torch.nn.functional as F

def IndexPooler(x, pooling_mask, adj_mat):
    return x[pooling_mask]


def AvgPooler(x, pooling_mask, adj_mat):
    # sum up node features of all neighbors for each node (including its own features)
    neighbor_sum = (adj_mat[pooling_mask].type(torch.float) @ x) + x[pooling_mask]
    # compute the avg node features of all neighbors for each node (including its own features) 
    neighbor_avg = neighbor_sum / (adj_mat[pooling_mask].sum(axis = 1).reshape(-1, 1) + 1)
    return neighbor_avg


class Enblock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, convolution, refinement, pooler, max_refinement=3, padding = False, device = 'cpu', **kwargs):
        super(Enblock, self).__init__()
        self.refinement = refinement
        self.pooler = pooler

        PAD_EDGE_INDICES_RF[max_refinement][refinement]
        self.edge_index = NONPAD_EDGE_INDICES_RF[max_refinement][refinement].to(device) if not padding else PAD_EDGE_INDICES_RF[max_refinement][refinement].to(device)
        self.adj_mat = NONPAD_ADJ_MATS_RF[max_refinement][refinement].to(device) if not padding else PAD_ADJ_MATS_RF[max_refinement][refinement].to(device)
        self.pooling_mask = NONPAD_POOLING_MASKS_RF[max_refinement][refinement].to(device) if not padding else PAD_POOLING_MASKS_RF[max_refinement][refinement].to(device)
        
        self.num_nodes = torch.max(self.edge_index) + 1#self.adj_mat.shape[0]
        self.device = device
        
        self.batch_edge_index = {1: self.edge_index}
        self.batch_adj_mat = {1: self.adj_mat}
        self.batch_pooling_mask = {1: self.pooling_mask}
        
        self.conv = convolution(in_channels, out_channels, **kwargs)
        self.reset_parameters()

    def reset_parameters(self):
        for name, param in self.conv.named_parameters():
            if 'bias' in name:
                torch.nn.init.constant_(param, 0)
            else:
                torch.nn.init.xavier_uniform_(param)

    def check_bs(self, bs):
        if bs not in self.batch_edge_index.keys():
            node_id_shifts = torch.arange(0, bs*self.num_nodes, self.num_nodes)[None].to(self.device)
            ## edges for entire batch
            x = self.edge_index.repeat(bs, 1, 1)
            tmp = ((x.permute(*torch.arange(x.ndim - 1, -1, -1)) + node_id_shifts))
            self.batch_edge_index[bs] = torch.cat(list(tmp.permute(*torch.arange(tmp.ndim - 1, -1, -1))), 1).to(self.device)
            # old: deprecated
            # self.batch_edge_index[bs] = torch.cat(list(((self.edge_index.repeat(bs, 1, 1).T + node_id_shifts)).T), 1).to(self.device)
            self.edge_index.repeat(bs, 1, 1).permute(*torch.arange(self.edge_index.repeat(bs, 1, 1).ndim - 1, -1, -1))
            self.batch_pooling_mask[bs] = torch.cat(list(((self.pooling_mask.repeat(bs, 1).T + node_id_shifts)).T), 0).to(self.device)
            self.batch_adj_mat[bs] = torch.zeros((self.num_nodes*bs, self.num_nodes*bs), dtype = torch.long).to(self.device)
            self.batch_adj_mat[bs][self.batch_edge_index[bs][0], self.batch_edge_index[bs][1]] = 1
    
    def forward(self, x):
        bs = int(x.shape[0]/self.num_nodes)
        self.check_bs(bs)
        out = F.elu(self.conv(x, self.batch_edge_index[bs]))
        out = self.pooler(out, self.batch_pooling_mask[bs], self.batch_adj_mat[bs])
        return out