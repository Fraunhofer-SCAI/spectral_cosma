from utils import load_patches, plot_mesh, load_train_test_data_gallop, load_train_test_data_faust
import matplotlib.pyplot as plt
import numpy as np
from utils.loss_functions import PaddingLossFunc, SurfaceAwareLossFunc
import cosma
import torch

from cosma import CoSMA
from utils import ModelTrainer, get_pygeo_dataset, ExperimentRunner
from torch_geometric.data import Data
import importlib
loader_spec = importlib.util.find_spec("torch_geometric.loader")
if loader_spec is not None:
    print('from torch_geometric.loader import DataLoader')
    from torch_geometric.loader import DataLoader
else:
    print('from torch_geometric.data import DataLoader')
    from torch_geometric.data import DataLoader
import random
import os
import os.path as osp
import datetime
from utils import utils
from utils.utils import str2bool

#import torch.profiler
from torch.utils.tensorboard import SummaryWriter

import argparse

def get_args():
    parser = argparse.ArgumentParser(description='spectral CoSMA')
    parser.add_argument('--model_name', type=str, default='test')
    parser.add_argument('--dataset', type=str, default='gallop_r4_2203')
    parser.add_argument('--device_idx', type=str, default='0')

    # mesh refinement
    parser.add_argument('--refine', type=int, default=4)

    # patch arguments and dataloading
    parser.add_argument('--kernel_size', type=int, default=2)
    parser.add_argument('--patch_zeromean', type=str2bool, default=True)
    parser.add_argument('--padded', type=int, default=1)
    parser.add_argument('--rotation_augment', type=int, default=1)

    # data split
    parser.add_argument('--test_split', nargs='+', type=str, default=['elephant']) # train-test-split: 100-0
    parser.add_argument('--test_ratio', type=float, default=0.25) # train-test-split: 75-25 for train samples
    
    # training variables
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--hid_rep', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--clip_grad_norm', type=bool, default=False)
    parser.add_argument('--grad_max_norm', type=float, default=0.1)
    parser.add_argument('--Niter', type=int, default=50)
    parser.add_argument('--conv_name', type=str, default='ChebConv')
    parser.add_argument('--ChebNet_K', type=int, default=6)
    parser.add_argument('--channels', nargs='+', type=int, default=[16, 32])
    parser.add_argument('--lastlinear', type=str2bool, default=False)
    parser.add_argument('--surface_aware_loss', type=str2bool, default=False)

    # others
    parser.add_argument('--seed', nargs="+", type=int, default=[1,2])

    args = parser.parse_args()
    args.work_dir = osp.dirname(osp.realpath(__file__))

    args.data_fp_spec = osp.join(args.work_dir, 'DATA', args.dataset)
    args.model_fp = osp.join(args.work_dir, 'experiments', args.dataset)
    
    return args

args = get_args()

save_dir = args.model_fp+'/'
utils.mkdirs(args.model_fp) 


model_logs = osp.join(args.model_fp, args.model_name)
utils.mkdirs(model_logs)
# write arguments to a log file
now = datetime.datetime.now()
logfile = open(model_logs+'/experiment_runner_{}.txt'.format(args.model_name), 'w')
for aa in list(args.__dict__.keys()):
    if args.work_dir in str(args.__dict__[aa]):
        text = '(path) {}: ****\n'.format(aa) #{}\n'.format(aa, args.__dict__[aa])
    else:
        text = '--{} {}\n'.format(aa, args.__dict__[aa])
    logfile.write(text)
    print(text, end='')   
logfile.write('\nDate: {} \n'.format('')) #now.strftime("%Y-%m-%d")))
    
    
## versions
versions = [f.name for f in os.scandir(args.data_fp_spec) if f.is_dir() and 'checkpoints' not in f.name and 'tmp' not in f.name]
print('Versions:', versions)

## parts/samples (for every version the same!)
samples = [f.name for f in os.scandir(osp.join(args.data_fp_spec, versions[0])) if f.is_dir() and 'checkpoints' not in f.name and 'tmp' not in f.name]
print('Samples:', samples)
test_samples = [sa for sa in samples if sa in args.test_split]
print('\nTest Sample:', test_samples)
train_samples = [sa for sa in samples if sa  not in test_samples] #!= args.test_split]
print('Train Samples:', train_samples, '\n')

### get test and train timesteps
pname = samples[0]
meshfiles = [f.name for f in os.scandir(osp.join(args.data_fp_spec, versions[0],pname)) if f.is_file() and ('.obj' in f.name or '.ply' in f.name) and 'reference' not in f.name]
meshfiles.sort()

select_tt = np.arange(0,len(meshfiles),1)
if args.test_ratio > 0:
    test_tt = select_tt[-int(len(meshfiles)* args.test_ratio ):]
elif args.test_ratio < 0:
    ## only for TRUCK select specific timesteps, always the same
    test_number_tt = int(-len(meshfiles)* args.test_ratio)
    test_tt = [ 8, 17,  9,  7, 24,  2, 11, 25, 20] #np.random.choice(len(meshfiles), test_number_tt, replace=False)#
    print('    Random test timestps:', test_tt)
else:
    test_tt = []
train_tt = [tt for tt in select_tt if tt not in test_tt]

## rotations
print('Rotation Augment?: ', end='')
if args.rotation_augment == 0:
    rotations = [0]
else:
    rotations = [0,1,2]
print(rotations)

# set training parameters
shuffle = True
weight_decay = 0.0
# if calculation of validation error should be omitted set this number higher than the number of epochs!
test_interval = 1
save_interval = 10
seeds = args.seed 
# set network architecture parameters
in_channels = 3
out_channels = args.channels #[16, 32]
refinements = [args.refine, args.refine-1]

# set device to either cpu (local machine) or gpu (cluster)
if args.device_idx != 'cpu':
    device = torch.device('cuda', int(args.device_idx))
    print('device cuda', args.device_idx)
else:
    device = torch.device('cpu')
    print('device', args.device_idx)

# initializing writer for TensorBoard
log_dir = "logs/fit/" + '{}_{}_{}_{}'.format('spectralCoSMA', args.dataset, args.model_name, now.strftime("%Y-%m-%d_%H-%M-%S"))
writer = SummaryWriter(log_dir)
    
##### LOAD THE DATA

# dataloading parameters
to_pygeo = True

print('\n{}, {} training and testing data'.format('Padded' if args.padded else 'Nonpadded', 'centered' if args.patch_zeromean else 'noncentered'))

import pickle
tmp_path = osp.join(args.data_fp_spec, 'tmp')
utils.mkdirs(tmp_path)
file_end = 'SW_{}_{}_{}_{}_pygeo_{}'.format(args.patch_zeromean, args.rotation_augment, args.test_split, args.test_ratio, to_pygeo)

if osp.exists(osp.join(tmp_path,"save_train_data_{}.p".format(file_end))) and osp.exists(osp.join(tmp_path,"save_test_data_{}.p".format(file_end))):
    data_training = pickle.load( open( tmp_path+"/save_train_data_{}.p".format(file_end), "rb" ) )
    data_testing  = pickle.load( open( tmp_path+"/save_test_data_{}.p".format(file_end), "rb" ) )
else:
    data_training, data_testing = load_train_test_data_faust(dataset = args.dataset, 
                                                     refinement = args.refine,
                                                     train_parts = train_samples,
                                                     test_parts = test_samples,
                                                     train_steps = train_tt,
                                                     test_steps = test_tt,
                                                     padding = args.padded, 
                                                     rotations = rotations, 
                                                     center_patches = args.patch_zeromean,   
                                                     to_pygeo = to_pygeo)

    # save tmp-files for testing
    pickle.dump(data_training, open( tmp_path+"/save_train_data_{}.p".format(file_end), "wb" ) )
    pickle.dump(data_testing , open( tmp_path+"/save_test_data_{}.p".format(file_end), "wb" ) )

logfile.write('{} training samples\n'.format(len(data_training)))
logfile.write('{} testing samples\n'.format(len(data_testing)))

print('{} training samples\n'.format(len(data_training)))
print('{} testing samples\n'.format(len(data_testing)))

##### DEFINE NETWORK

MAX_REFINEMENT = args.refine

# load some frequently used constants
PAD_ADJ_MATS, PAD_EDGE_INDICES, PAD_POOLING_MASKS = cosma.PAD_ADJ_MATS_RF[MAX_REFINEMENT], cosma.PAD_EDGE_INDICES_RF[MAX_REFINEMENT], cosma.PAD_POOLING_MASKS_RF[MAX_REFINEMENT]
NONPAD_ADJ_MATS, NONPAD_EDGE_INDICES, NONPAD_POOLING_MASKS = cosma.NONPAD_ADJ_MATS_RF[MAX_REFINEMENT], cosma.NONPAD_EDGE_INDICES_RF[MAX_REFINEMENT], cosma.NONPAD_POOLING_MASKS_RF[MAX_REFINEMENT]


poolers = [cosma.AvgPooler, 
           cosma.AvgPooler]
unpoolers = [cosma.Pad2ndLevelIdUnpooler(input_nodes = PAD_ADJ_MATS[MAX_REFINEMENT-2].shape[0], 
                           output_nodes = PAD_ADJ_MATS[MAX_REFINEMENT-1].shape[0], 
                           max_refinement = MAX_REFINEMENT, device = device), 
             cosma.AvgUnpooler]


## encode possibly different spectral convolutional layers here.
from torch_geometric.nn import GCNConv, SGConv, ChebConv

class GraphConvolution:
    
    def __init__(self, conv_name = 'ChebConv', K = 6):
        self.conv_name = conv_name
        if conv_name == 'GCN':
            self.graphconv = GCNConv
        elif conv_name == 'SGC':
            self.graphconv = SGConv
        else:
            self.graphconv = ChebConv
        #elif:
        self.K = K
        
    def __call__(self, out_channels, in_channels, **kwargs):
        if self.conv_name == 'ChebConv':
            return self.graphconv(out_channels, in_channels, self.K, **kwargs)
        else:
            return self.graphconv(out_channels, in_channels, **kwargs)
    
graph_convolutions = lambda: GraphConvolution(args.conv_name, K=args.ChebNet_K)

# define models
# note that we don't directly define the models but initializers 
# that will later allow us to define the models with different random seeds 
model_name = args.model_name
models = {model_name: lambda: CoSMA(in_channels = in_channels, 
                                     out_channels = out_channels, 
                                     latent_channels = args.hid_rep,
                                     convolution = graph_convolutions(),
                                     poolers = poolers, unpoolers = unpoolers, 
                                     refinements = refinements, 
                                     padding = args.padded, 
                                     lastlinear = args.lastlinear,
                                     device = device),
         }

print(models[model_name]())

# define optimizers
optimizers = {model_name: lambda model: torch.optim.Adam(model.parameters(), 
                                                      lr = args.lr, 
                                                      weight_decay = weight_decay),
             }

# define loss function
if args.surface_aware_loss:
    print('Surface-aware loss (MSE)')
    loss_funcs = {model_name: lambda: SurfaceAwareLossFunc(device = device),
                 }    
else:
    print('loss (MSE)')
    loss_funcs = {model_name: lambda: PaddingLossFunc(device = device,
                                                MAX_REFINEMENT = MAX_REFINEMENT), 
                     }

# define datasets
datasets = {model_name: lambda: (DataLoader(data_training, batch_size = args.batch_size, shuffle = shuffle),
                                 DataLoader(data_testing , batch_size = args.batch_size, shuffle = shuffle)),
           }

# define experiment runner
runner = ExperimentRunner(model_builders = models,
                          optimizer_builders = optimizers,
                          loss_func_builders = loss_funcs,
                          dataset_builders = datasets,
                          save_directory = save_dir,
                          clip_grad_norm = args.clip_grad_norm,
                          grad_max_norm = args.grad_max_norm,
                          device = device)



##### RUN THE EXPERIMENT
runner.run_experiment(seeds = seeds, 
                      num_epochs = args.Niter, 
                      test_interval = test_interval, 
                      save_interval = save_interval,
                      logfile = logfile, writer=writer)




logfile.close()

# plot the learning curves 
#plt.rcParams.update({'font.size': 10})
runner.plot_learning_curves(seeds, test_interval, filename='avg_learning_curves_{}'.format(args.model_name))