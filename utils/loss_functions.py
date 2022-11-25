import cosma
import torch
import numpy as np

class SurfaceAwareLossFunc:

    def __init__(self, device, 
                 dim = 3, 
                 loss_func = torch.nn.MSELoss(reduction = 'none')):
        
        self.device = device
        self.loss_func = loss_func


    def __call__(self, x, y, loss_weights):
        ## vertex wise loss
        loss = self.loss_func(x, y)
        ## apply weighting for normal loss and regularization
        return (loss_weights * loss).mean(), (0 * loss).mean()

class PaddingLossFunc:

    def __init__(self, device, MAX_REFINEMENT,
                 dim = 3, 
                 loss_func = torch.nn.MSELoss(reduction = 'none'), 
                 padding_weight=0.0, boundary_weight=0.0, anti_boundary_weight=0.0):
        
        self.device = device
        self.num_patch_nodes = cosma.PAD_ADJ_MATS_RF[MAX_REFINEMENT][MAX_REFINEMENT].shape[0]
        self.loss_func = loss_func
        self.padding_mask = torch.zeros((self.num_patch_nodes, dim), device = self.device)
        self.padding_mask_reg = torch.zeros((self.num_patch_nodes, dim), device = self.device)
        
        # add padding weight if wanted
        if padding_weight > 0:
            self.padding_mask_reg = torch.add(self.padding_mask, padding_weight)
            self.padding_mask_reg[cosma.NO_PADDING_INDICES[MAX_REFINEMENT]] = torch.zeros(dim, device = self.device)
            
        # iterior has weight one
        self.padding_mask[cosma.NO_PADDING_INDICES[MAX_REFINEMENT]] = torch.ones(dim, device = self.device)            
            
        self.batch_padding_mask = {1: self.padding_mask}
        self.batch_padding_mask_reg = {1: self.padding_mask_reg}
        
    def check_bs(self, bs):
        
        if bs not in self.batch_padding_mask.keys():
            self.batch_padding_mask[bs] = self.padding_mask.repeat(bs, 1)    
            self.batch_padding_mask_reg[bs] = self.padding_mask_reg.repeat(bs, 1)  

    def __call__(self, x, y, loss_weights):
        bs = x.shape[0] // self.num_patch_nodes
        self.check_bs(bs)
        ## vertex wise loss
        loss = self.loss_func(x, y)
        ## apply weighting for normal loss and regularization
        return (self.batch_padding_mask[bs] * loss).mean(), (self.batch_padding_mask_reg[bs] * loss).mean()
