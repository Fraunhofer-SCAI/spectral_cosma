import numpy as np
import pickle
import igl
import os
import os.path as osp
import torch
from torch_geometric.data import Data 


import cosma

def load_patches(part = 'horse', padding = False, tt = [0], rotation = 0, version = 0, dataset = 'gallop', center_patches = False):
    mesh_path_data = osp.join(osp.os.getcwd(), 'DATA', dataset)
    mesh_path_data_part = mesh_path_data + '/version_{}/{}'.format(version, part)
    if 'car' in dataset:
        mesh_path_data_part = mesh_path_data + '/sim_{}/{}'.format(str(version).zfill(3), part)
    padded = 'padded' if padding else 'nonpadded'
    # load node ids of unpadded area per patch
    with open(mesh_path_data + '/node_ids_per_patch_{}_{}_rot{}.p'.format(part, padded, rotation), 'rb') as f:
        ids_hexagonal_patch = np.array(pickle.load(f), dtype=int)

    # load local adjacency matrix
    fpath = mesh_path_data + '/local_adjacency_matrix_{}_{}.p'.format(part, padded)
    adjacency_matrix =  pickle.load(open( fpath, "rb" ) )
    if padding:
        tt_VV = []
        for t in range(len(tt)):
            if 'gallop' in dataset:
                obj_name =  '/gallop-{}.obj'.format(str(tt[t]).zfill(2))
            if 'faust' in dataset or 'FAUST' in dataset:
                obj_name =  '/{}-FAUST-{}.obj'.format(part, str(tt[t]).zfill(2))
            if 'car_TRUCK' in dataset:
                obj_name =  '/car_TRUCK-{}.obj'.format(str(tt[t]).zfill(2))
            if 'car_YARIS' in dataset:
                obj_name =  '/car_YARIS-{}.obj'.format(str(tt[t]).zfill(2))
            VV, _, _, FF, _, _ = igl.read_obj(mesh_path_data_part + obj_name)
            tt_VV += [VV]
        if 'gallop' in dataset:
            patch_value_name = '/gallop-padded_patch_vertex_values_rot{}.p'.format(rotation)
        if 'faust' in dataset or 'FAUST' in dataset:
            patch_value_name = '/FAUST-padded_patch_vertex_values_rot{}.p'.format(rotation)
        if 'car_TRUCK' in dataset:
            patch_value_name = '/car_TRUCK-padded_patch_vertex_values_rot{}.p'.format(rotation)
        if 'car_YARIS' in dataset:
            patch_value_name = '/car_YARIS-padded_patch_vertex_values_rot{}.p'.format(rotation)
        with open(mesh_path_data_part + patch_value_name, 'rb') as f:
            patches = np.array(pickle.load(f))[tt]
    else:
        patches = []
        tt_VV = []
        for t in range(len(tt)):
            tt_patches = []
            if 'gallop' in dataset:
                obj_name =  '/gallop-{}.obj'.format(str(tt[t]).zfill(2))
            if 'faust' in dataset or 'FAUST' in dataset:
                obj_name =  '/{}-FAUST-{}.obj'.format(part, str(tt[t]).zfill(2))
            VV, _, _, FF, _, _ = igl.read_obj(mesh_path_data_part + obj_name)
            tt_VV += [VV]
            for patch in ids_hexagonal_patch:
                tt_patches += [VV[patch]]
            tt_patches = np.asarray(tt_patches)
            patches += [tt_patches]
        patches = np.asarray(patches)
        
    if center_patches:
        # get mean of every patch
        patch_means = np.mean(patches, axis = 2)[:, :, None, :]
        # transform every patch to have zero mean
        patches -= patch_means
    
    return patches, adjacency_matrix, tt_VV, FF, ids_hexagonal_patch


def get_pygeo_dataset(dataset, edge_index, index_mapping, loss_weights = None):
    pygeo_dataset = []
    for idx, patch_vv in enumerate(dataset):
        
        patch_vv_reindexed = torch.tensor(patch_vv[index_mapping], dtype = torch.float32)
        loss_weights_reindexed = torch.tensor(loss_weights[idx][index_mapping], dtype = torch.float32).view(-1,1)
            
        data =  Data(x = torch.clone(patch_vv_reindexed), 
                     edge_index = torch.clone(edge_index), 
                     y = torch.clone(patch_vv_reindexed),
                     pos = torch.clone(patch_vv_reindexed),
                     loss_weights = torch.clone(loss_weights_reindexed))
        data.idx = idx
        pygeo_dataset.append(data)
    return pygeo_dataset



def load_train_test_data_faust(dataset = 'gallop', refinement = 3, train_parts = ['horse', 'camel'], test_parts = ['elephant'], train_steps = np.arange(36), test_steps = np.arange(36, 48), padding = False, rotations = [0], center_patches = False, to_pygeo = False, mult_100 = False):
    data_fp = osp.join(osp.os.getcwd(), 'DATA', dataset)
    ### TODO: just use the whole directory name here
    versions = [int(f.name.split('_')[-1] )for f in os.scandir(data_fp) if f.is_dir() and 'checkpoints' not in f.name and 'tmp' not in f.name]
    print('Versions:', versions)
    
    training_patches = None #np.zeros((0, 111 if padding else 45, 3))
    testing_patches = None #np.zeros((0, 111 if padding else 45, 3))
    count_test, count_train = 0, 0

    for PART in train_parts + test_parts:

        select_tt = np.array(list(set(train_steps).union(set(test_steps))))
        test_tt = test_steps
        
        print('################')
        print('Part', PART)

        for version in versions:
            print('Version', version)
            
            # was passiert wenn man select_tt hier definiert??
            # TODO: still hardcoded. doesnt work in case some versions have differnt number of timesteps
            if 'YARIS' in dataset and version==957:
                select_tt = np.arange(7)
                print(select_tt)

            for rotation in rotations:
                
                patches, adjacency_matrix, VV, FF, ids_hexagonal_patch = load_patches(part = PART, padding = padding, tt = select_tt, rotation = rotation, version = version, dataset = dataset, center_patches = center_patches)
                if mult_100:
                    patches = 1000.0 * patches
                print('   Rotation {}: {} patches with {} vertices each'.format(rotation, patches.shape[1], adjacency_matrix.shape[0]))
                
                print('Get weight matrix for loss calculation.')
                v_id, count = np.unique(ids_hexagonal_patch, return_counts=True)
                neg_id = np.where(v_id < 0)[0]
                count = np.delete(count, neg_id)
                v_id = np.delete(v_id, neg_id)
                id_counts = np.zeros(max(v_id)+1)
                id_counts[v_id] = count

                non_padding_vertices = np.where(ids_hexagonal_patch[0] >= 0)[0]

                loss_weights = np.zeros((patches.shape[1], patches.shape[2]))
                loss_weights[:, non_padding_vertices] = 1/id_counts[ids_hexagonal_patch[:,non_padding_vertices]]
                loss_weights = np.repeat(np.reshape(loss_weights, (1, patches.shape[1], patches.shape[2])), patches.shape[0], axis=0)

                                
                if training_patches is None:
                    training_patches = np.zeros((0,adjacency_matrix.shape[0],3))
                    testing_patches = np.zeros((0,adjacency_matrix.shape[0],3))
                    # loss weights
                    training_loss_weights = np.zeros((0,adjacency_matrix.shape[0]))
                    testing_loss_weights = np.zeros((0,adjacency_matrix.shape[0]))
                
                for tt in select_tt: 
                    
                    if PART in test_parts or tt in test_steps:
                        testing_patches = np.concatenate((testing_patches, patches[tt]),axis=0)
                        testing_loss_weights = np.concatenate((testing_loss_weights, loss_weights[tt]),axis=0)
                    elif 'car_TRUCK' in dataset and version in ['sim_041', 'sim_049']:
                        testing_patches = np.concatenate((testing_patches, patches[tt]),axis=0)
                        testing_loss_weights = np.concatenate((testing_loss_weights, loss_weights[tt]),axis=0)
                    else:
                        training_patches = np.concatenate((training_patches, patches[tt]),axis=0)
                        training_loss_weights = np.concatenate((training_loss_weights, loss_weights[tt]),axis=0)
    
        # count how many train and test patches are already centered
        count_train = training_patches.shape[0]
        count_test = testing_patches.shape[0]
    
    if to_pygeo:
        edge_index = cosma.PAD_EDGE_INDICES_RF[refinement][refinement] if padding else cosma.NONPAD_EDGE_INDICES_RF[refinement][refinement]
        index_mapping = cosma.PAD_INDEX_MAPPING_KS2[refinement] if padding else cosma.NONPAD_INDEX_MAPPING[refinement]
        print('\n -> After reindexing',len(index_mapping), 'vertices per patch')
        training_patches = get_pygeo_dataset(training_patches, edge_index, index_mapping, loss_weights = training_loss_weights)
        testing_patches = get_pygeo_dataset(testing_patches, edge_index, index_mapping, loss_weights = testing_loss_weights)
    return training_patches, testing_patches


def load_train_test_data_gallop(dataset = 'gallop', refinement = 3, test_split = 'elephant', padding = False, rotations = [0], center_patches = False, to_pygeo = False, tt_split = 36, tt_max = 48, tt_min = 0):
    data_fp = osp.join(osp.os.getcwd(), 'DATA', dataset)
    versions = [int(f.name.split('_')[-1] )for f in os.scandir(data_fp) if f.is_dir() and 'checkpoints' not in f.name and 'tmp' not in f.name]
    print('Versions:', versions)
    
    ## parts/samples (for every version the same!)
    samples = [f.name for f in os.scandir(osp.join(data_fp, 'version_' + str(versions[0]))) if f.is_dir() and 'checkpoints' not in f.name and 'tmp' not in f.name]
    print('Samples:', samples)
    testing_PART = [sa for sa in samples if sa == test_split]
    print('\nTest Sample:', testing_PART)
    training_PART = [sa for sa in samples if sa != test_split]
    print('Train Samples:', training_PART, '\n')

    training_patches = None #np.zeros((0, 111 if padding else 45, 3))
    testing_patches = None #np.zeros((0, 111 if padding else 45, 3))

    for pn, PART in enumerate(training_PART+testing_PART):

        select_tt = np.arange(0,tt_max)
        test_tt = np.arange(tt_split, tt_max)

        print('################')
        print('Part', PART)

        for vn, version in enumerate(versions):
            print('Version', version)

            for rn, rotation in enumerate(rotations):
                
                patches, adjacency_matrix, VV, FF, ids_hexagonal_patch = load_patches(part = PART, padding = padding, tt = select_tt, rotation = rotation, version = version, dataset = dataset, center_patches = center_patches)
                print('   Rotation {}: {} patches with {} vertices each'.format(rotation, patches.shape[1], adjacency_matrix.shape[0]))

                if pn == 0 and vn == 0 and rn == 0:
                    training_patches = np.zeros((0,adjacency_matrix.shape[0],3))
                    testing_patches = np.zeros((0,adjacency_matrix.shape[0],3))
                
                for tt in select_tt: 
                    
                    if PART in testing_PART or tt in test_tt:
                        testing_patches = np.concatenate((testing_patches, patches[tt]),axis=0)

                    else:
                        training_patches = np.concatenate((training_patches, patches[tt]),axis=0)
    
    if to_pygeo:
        edge_index = cosma.PAD_EDGE_INDICES_RF[refinement][refinement] if padding else cosma.NONPAD_EDGE_INDICES_RF[refinement][refinement]
        index_mapping = cosma.PAD_INDEX_MAPPING_KS2[refinement] if padding else cosma.NONPAD_INDEX_MAPPING[refinement]
        training_patches = get_pygeo_dataset(training_patches, edge_index, index_mapping)
        testing_patches = get_pygeo_dataset(testing_patches, edge_index, index_mapping)
    return training_patches, testing_patches