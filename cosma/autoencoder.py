import torch
from torch_geometric.nn import ChebConv
import torch.nn.functional as F

from .globals import *
from .encoding import *
from .decoding import *

class CoSMA(torch.nn.Module):
    def __init__(self, in_channels, out_channels, latent_channels, convolution, poolers, unpoolers, refinements, padding = False, lastlinear = False, device = 'cpu', **kwargs):
        super(CoSMA, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lastlinear = lastlinear
        self.convolution = convolution
        self.poolers = poolers
        self.unpoolers = unpoolers
        self.refinements = refinements
        self.MAX_REFINEMENT = refinements[0]
        # self.num_vert used in the last and the first layer of encoder and decoder
        self.num_vert = NONPAD_ADJ_MATS_RF[self.MAX_REFINEMENT][refinements[-1] - 1].shape[0] if not padding else PAD_ADJ_MATS_RF[self.MAX_REFINEMENT][refinements[-1] - 1].shape[0]
        self.last_layer_edge_index = NONPAD_EDGE_INDICES_RF[self.MAX_REFINEMENT][self.refinements[0]].to(device) if not padding else PAD_EDGE_INDICES_RF[self.MAX_REFINEMENT][self.refinements[0]].to(device)

        # encoder
        self.en_layers = torch.nn.ModuleList()
        for idx in range(len(out_channels)):
            if idx == 0:
                self.en_layers.append(
                    Enblock(in_channels = in_channels, out_channels = out_channels[idx], 
                            refinement = refinements[idx], max_refinement=self.MAX_REFINEMENT, 
                            pooler = poolers[idx], convolution = self.convolution, padding = padding, device = device, **kwargs))
            else:
                self.en_layers.append(
                    Enblock(in_channels = out_channels[idx - 1], out_channels = out_channels[idx], 
                            refinement = refinements[idx], max_refinement=self.MAX_REFINEMENT, 
                            pooler = poolers[idx], convolution = self.convolution, padding = padding, device = device, **kwargs))
        self.en_layers.append(
            torch.nn.Linear(self.num_vert * out_channels[-1], latent_channels))

        # decoder
        self.de_layers = torch.nn.ModuleList()
        self.de_layers.append(
            torch.nn.Linear(latent_channels, self.num_vert * out_channels[-1]))
        for idx in range(len(out_channels)):
            if idx == 0:
                self.de_layers.append(
                    Deblock(in_channels = out_channels[-idx - 1], out_channels = out_channels[-idx - 1], 
                            refinement = refinements[-idx - 1] - 1, max_refinement=self.MAX_REFINEMENT, 
                            unpooler = unpoolers[idx], convolution = self.convolution, 
                            padding = padding, device = device, **kwargs))
            else:
                self.de_layers.append(
                    Deblock(in_channels = out_channels[-idx], out_channels = out_channels[-idx - 1], 
                            refinement = refinements[-idx - 1] - 1, max_refinement=self.MAX_REFINEMENT, 
                            unpooler = unpoolers[idx], convolution = self.convolution, 
                            padding = padding, device = device, **kwargs))
        # reconstruction
        if self.lastlinear:
            self.de_layers.append(
                torch.nn.Linear(self.num_vert * out_channels[0], self.num_vert * in_channels))
        else:    
            self.de_layers.append(
                self.convolution(out_channels[0], in_channels, **kwargs))

        self.reset_parameters()

    def reset_parameters(self):
        for name, param in self.named_parameters():
            if 'bias' in name:
                torch.nn.init.constant_(param, 0)
            elif 'conv' in name and '1.8.0' not in torch.__version__:
                torch.nn.init.xavier_uniform_(param, gain = 0.2)
            else:
                torch.nn.init.xavier_uniform_(param)

    def encoder(self, x):
        for i, layer in enumerate(self.en_layers):
            #print(x.shape)
            if i != len(self.en_layers) - 1:
                x = layer(x)
            else:
                x = x.view(-1, layer.weight.size(1))
                x = layer(x)
        return x

    def decoder(self, x):
        num_layers = len(self.de_layers)
        num_deblocks = num_layers - 2
        for i, layer in enumerate(self.de_layers):
            #print(x.shape)
            if i == 0:
                x = layer(x)
                x = x.view(-1, self.out_channels[-1])
            elif i != num_layers - 1:
                x = layer(x)
            else:
                # last layer
                if self.lastlinear:
                    x = x.view(-1, layer.weight.size(1))
                    x = layer(x)
                    x = x.view(-1, self.in_channels)
                else:
                    x = layer(x, self.last_layer_edge_index)
        return x

    def forward(self, x):
        # x - batched feature matrix
        z = self.encoder(x)
        out = self.decoder(z)
        return out