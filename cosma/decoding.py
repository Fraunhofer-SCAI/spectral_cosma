from .globals import *

import torch
from torch_geometric.nn import ChebConv
import torch.nn.functional as F

import numpy as np


## average unpooler, unpadded
def AvgUnpooler(x, unpooling_mask, edge_index):
    # reindex known vertices
    patch_vv_unpooled = torch.zeros((unpooling_mask[-1] + 1, *x.shape[1:]), device = x.device)
    patch_vv_unpooled[unpooling_mask] = x
    # set intermediate vertices
    tail, head = edge_index[0], edge_index[1]
    patch_vv_unpooled[torch.div((unpooling_mask[head] + unpooling_mask[tail]), 2, rounding_mode='trunc')] = torch.mean(torch.stack([x[head], x[tail]]), axis = 0)
    return patch_vv_unpooled



## unpooling node ids for different refinement level
pad_mapping_RF = dict()
for map_level in [3,4]: # initial refinement
    rows = 2 + np.sum(2**np.arange(map_level-1)) + 3*1
    pad_mapping = []
    counter = 0
    for rr in range(rows):
        for ii in range(rr+1):
            pad_mapping += [counter]
            if rr == rows-1: # last row
                if ii < rr/2: # first half
                    pad_mapping[counter] = pad_mapping[counter-rr-1]
                    if pad_mapping[counter-rr-1] == -1: # first node of last row
                        pad_mapping[counter] = pad_mapping[counter-rr]
                elif ii == rr: # last node of last row
                    pad_mapping[counter] = pad_mapping[counter-1]
                else: # second half
                    pad_mapping[counter] = pad_mapping[counter-rr]
            elif ii == 0 or ii == rr: # for now put -1
                pad_mapping[counter] = -1
                if rr == rows-2:
                    if ii == 0:
                        pad_mapping[counter] = counter+1
                    else:
                        pad_mapping[counter] = counter-1
            counter += 1 
    counter = 0
    for rr in range(rows):
        for ii in range(rr+1):
            if ii == 0 or ii == rr:
                if rr <= 2: # first two rows
                    pad_mapping[counter] = 4
                elif rr < int(rows/2): # first half
                    if ii == 0: # first node of rows
                        pad_mapping[counter] = pad_mapping[counter+1]
                    else: # last node of rows
                        pad_mapping[counter] = pad_mapping[counter-1]
                elif rr < rows-1: # second half
                    if ii == 0: # first node of rows
                        pad_mapping[counter] = pad_mapping[counter+rr+2]
                    else: # last node of rows
                        pad_mapping[counter] = pad_mapping[counter+rr+1]
            counter += 1   
    pad_mapping_RF[map_level] = pad_mapping

class Pad2ndLevelIdUnpooler:
    # first unpooling 
    def __init__(self, input_nodes = 6, output_nodes = 36, max_refinement=3, device = 'cpu'):
        self.input_nodes = input_nodes
        self.output_nodes = output_nodes
        self.device = device
        
        self.pad_mapping = torch.tensor(pad_mapping_RF[max_refinement], dtype = torch.long).to(self.device)
        
        self.batch_pad_mapping = {1: self.pad_mapping}
        
        
    def check_bs(self, bs):
        if bs not in self.batch_pad_mapping.keys():
            node_id_shifts = torch.arange(0, bs*self.output_nodes, self.output_nodes)[None].to(self.device)
            self.batch_pad_mapping[bs] = torch.cat(list(((self.pad_mapping.repeat(bs, 1).T + node_id_shifts)).T), 0).to(self.device)
    
    def __call__(self, x, unpooling_mask, edge_index):
        bs = x.shape[0] // self.input_nodes
        self.check_bs(bs)
        # reindex known vertices
        patch_vv_unpooled = torch.zeros((bs * self.output_nodes, *x.shape[1:]), device = self.device)
        patch_vv_unpooled[unpooling_mask] = x
        # set intermediate vertices
        head, tail = edge_index[0], edge_index[1]
        patch_vv_unpooled[torch.div((unpooling_mask[head] + unpooling_mask[tail]), 2, rounding_mode='trunc')] = torch.mean(torch.stack([x[head], x[tail]]), axis = 0)

        # set boundary vertices identical to inner vertices
        patch_vv_unpooled = patch_vv_unpooled[self.batch_pad_mapping[bs]]

        return patch_vv_unpooled.type(x.dtype)




class Deblock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, convolution, refinement, unpooler, max_refinement=3, padding = False, device = 'cpu', **kwargs):
        super(Deblock, self).__init__()
        self.refinement = refinement
        self.unpooler = unpooler
        
        self.prev_edge_index = NONPAD_EDGE_INDICES_RF[max_refinement][refinement].to(device) if not padding else PAD_EDGE_INDICES_RF[max_refinement][refinement].to(device)
        self.unpooling_mask = NONPAD_POOLING_MASKS_RF[max_refinement][refinement + 1].to(device) if not padding else PAD_POOLING_MASKS_RF[max_refinement][refinement + 1].to(device)
        self.edge_index = NONPAD_EDGE_INDICES_RF[max_refinement][refinement + 1].to(device) if not padding else PAD_EDGE_INDICES_RF[max_refinement][refinement + 1].to(device)
        
        self.prev_num_nodes = torch.max(self.prev_edge_index) + 1
        self.num_nodes = torch.max(self.edge_index) + 1
        self.device = device
        
        self.batch_edge_index = {1: self.edge_index}
        self.batch_prev_edge_index = {1: self.prev_edge_index}
        self.batch_unpooling_mask = {1: self.unpooling_mask}
        
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
            prev_node_id_shifts = torch.arange(0, bs*self.prev_num_nodes, self.prev_num_nodes)[None].to(self.device)
            
            ## edges for entire batch
            x = self.edge_index.repeat(bs, 1, 1)
            tmp = ((x.permute(*torch.arange(x.ndim - 1, -1, -1)) + node_id_shifts))
            self.batch_edge_index[bs] = torch.cat(list(tmp.permute(*torch.arange(tmp.ndim - 1, -1, -1))), 1).to(self.device)

            #batch_unpooling_mask
            tmp = self.unpooling_mask.repeat(bs, 1).permute(*torch.arange(self.unpooling_mask.repeat(bs, 1).ndim - 1, -1, -1)) 
            tmp = (tmp + node_id_shifts).permute(*torch.arange((tmp + node_id_shifts).ndim - 1, -1, -1))  
            self.batch_unpooling_mask[bs] = torch.cat(list(tmp), 0).to(self.device)
            
            ## previous edges for entire batch
            x = self.prev_edge_index.repeat(bs, 1, 1)
            tmp = ((x.permute(*torch.arange(x.ndim - 1, -1, -1)) + prev_node_id_shifts))
            self.batch_prev_edge_index[bs] = torch.cat(list(tmp.permute(*torch.arange(tmp.ndim - 1, -1, -1))), 1).to(self.device)

    def forward(self, x):
        bs = int(x.shape[0]/self.prev_num_nodes)
        self.check_bs(bs)
        out = self.unpooler(x, self.batch_unpooling_mask[bs], self.batch_prev_edge_index[bs])
        out = F.elu(self.conv(out, self.batch_edge_index[bs]))
        return out