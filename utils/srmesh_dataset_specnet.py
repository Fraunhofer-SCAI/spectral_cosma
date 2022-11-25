import numpy as np
import torch
import pickle
import igl
import os
import os.path as osp
import cosma
from torch_geometric.data import Data

from utils import get_mesh_predictions, get_mesh_embeddings
from utils.srmesh_dataset import srmesh_dataset


class srmesh_dataset_specnet(srmesh_dataset):   
    def __init__(self, args, versions, samples, test_samples, test_timesteps, device, models, best_validation=False):
        super().__init__(args, versions, samples, test_samples, device)
        
        self.data_fp_spec = args.data_fp_spec
        self.best_validation = best_validation
        self.test_timesteps = test_timesteps
        self.seed = args.seed
        self.model_fp = args.model_fp
        self.models = models
        self.refinement = args.refine
        
        self.model_patchinput_per_part = dict()
        self.model_patchoutput_per_part = dict()
        self.model_loss_weight_per_part = dict()

        
    def get_true_meshvv_per_part(self, pname, versions):
        
        rotation=0
        
        if pname not in self.true_meshvv_per_part:
            self.true_meshvv_per_part[pname] = dict()
        
        for version in versions:
            
            if version in self.true_meshvv_per_part[pname]:
                continue
        
            _, _, tt_VV, FF, _, _ = self.load_patches(part = pname,
                                                 rotation = rotation,
                                                 version = version)
                
            self.true_meshvv_per_part[pname][version] = tt_VV
        
        return 0
    

    def get_test_train_meshes(self, pname, versions):

        if pname not in self.test_train_meshes:
            self.test_train_meshes[pname] = dict()
            
        # number of triangles
        N_triangles = self.N_triangles(pname)
        rotation = 0
        
        for version in versions:
            
            if version in self.test_train_meshes[pname]:
                continue
                
            mesh_path_data = self.data_fp_spec
            mesh_path_data_part = mesh_path_data + '/{}/{}'.format(version, pname)
            meshfiles = [f.name for f in os.scandir(mesh_path_data_part) if f.is_file() and ('.obj' in f.name or '.ply' in f.name) and 'reference' not in f.name]
            meshfiles.sort()
            select_tt = np.arange(0,len(meshfiles),1)
            
            self.test_train_meshes[pname][version] = np.zeros(len(meshfiles))
        
            for tt in select_tt: 

                if pname in self.test_samples or tt in self.test_timesteps:
                    self.test_train_meshes[pname][version][tt] = 1

                else:
                    self.test_train_meshes[pname][version][tt] = 0

    
    def load_patches(self, part, rotation, version, center_patches = False):
        """
        output
        patches: padded, shape (timesteps, #patches, size of padded patch, 3)
        adjacency_matrix: shape (size of padded patch, size of padded patch)
        tt_VV: list of arrays, shape: #timesteps * (# VV, 3)
        FF: array of faces
        ids_hexagonal_patch: vertex ids of unpadded patches, shape (#patches, #vertices unpadded patch)
        """

        mesh_path_data = self.data_fp_spec
        mesh_path_data_part = mesh_path_data + '/{}/{}'.format(version, part)
        padded = 'padded' 
        with open(mesh_path_data + '/node_ids_per_patch_{}_{}_rot{}.p'.format(part, padded, rotation), 'rb') as f:
            ids_hexagonal_patch = np.array(pickle.load(f), dtype=int)

        # load local adjacency matrix
        fpath = mesh_path_data + '/local_adjacency_matrix_{}_{}.p'.format(part, padded)
        adjacency_matrix =  pickle.load(open( fpath, "rb" ) )                    

        meshfiles = [f.name for f in os.scandir(mesh_path_data_part) if f.is_file() and ('.obj' in f.name or '.ply' in f.name) and 'reference' not in f.name]
        meshfiles.sort()
        select_tt = np.arange(0,len(meshfiles),1)

        if padded == 'padded':
            tt_VV = []
            for tt in select_tt:
                obj_name =  meshfiles[tt]
                VV, _, _, FF, _, _ = igl.read_obj(mesh_path_data_part + '/' + obj_name)
                tt_VV += [VV]

            patch_value_name = '/{}-padded_patch_vertex_values_rot{}.p'.format(self.dataset ,rotation)
            with open(mesh_path_data_part + patch_value_name, 'rb') as f:
                patches = np.array(pickle.load(f))[select_tt]
        else:
            patches = []
            tt_VV = []
            for tt in select_tt:
                obj_name =  meshfiles[tt]
                VV, _, _, FF, _, _ = igl.read_obj(mesh_path_data_part + '/' + obj_name)
                tt_VV += [VV]
                for patch in ids_hexagonal_patch:
                    tt_patches += [VV[patch]]
                tt_patches = np.asarray(tt_patches)
                patches += [tt_patches]
            patches = np.asarray(patches)

        patch_means = 0.0
        if center_patches:
            # get mean of every patch
            patch_means = np.mean(patches, axis = 2)[:, :, None, :]
            # transform every patch to have zero mean
            patches -= patch_means

        return patches, adjacency_matrix, tt_VV, FF, ids_hexagonal_patch, patch_means

    def construct_predicted_meshvv_per_part(self, pname, versions, recompute_predictions=False, save_predictions=True):
        if pname not in self.predicted_meshvv_per_part:
            self.predicted_meshvv_per_part[pname] = dict()
            
        rotation=0
            
        for version in versions:
            
            if version in self.predicted_meshvv_per_part[pname]:
                continue
            
            padded_patches, adjacency_matrix, tt_VV, FF, p_ids_hexagonal_patch, _ = self.load_patches(part = pname,
                                                                                         rotation = rotation,
                                                                                         version = version)
            
            del_id = np.where(p_ids_hexagonal_patch[0] == -1)[0]
            p_ids_hexagonal_patch_nopad = np.delete(p_ids_hexagonal_patch,del_id,1)
            
            
            
            meshfiles = [f.name for f in os.scandir(osp.join(self.data_fp_spec, version, pname)) if f.is_file() and ('.obj' in f.name or '.ply' in f.name) and 'reference' not in f.name]
            meshfiles.sort()
            select_tt = np.arange(0,len(meshfiles),1)
            
            
            tt_VV_torch = torch.tensor(np.array(tt_VV), dtype = torch.float)
            FF_torch = torch.tensor(FF, dtype = torch.long).T
                

            if version == versions[0]:
                if self.best_validation:
                    print(' best validation models')
                else:
                    print(' final models')
            for name in [self.model_name]:
                save_directory = self.model_fp
                mesh_predictions = [[] for _ in range(len(self.seed))]
                for seed_id, seed in enumerate(self.seed):
                    model_name = 'saved_model' if not self.best_validation else 'best_validation_model'
                    model_path = save_directory + '/' + name + '/seed_{0}/{1}.pth'.format(seed, model_name)
                    if version == versions[0]:
                        print('',model_name, model_path)
                    else:
                        print('.',end='')
                    mesh_path = save_directory + '/' + name + '/seed_{0}/{1}_mesh_preds_{2}_{3}.npy'.format(seed, model_name, pname, version)
                    model = self.models[name]()
                    model.load_state_dict(torch.load(model_path, map_location = self.device))
                    model = model.to(self.device)

                    if recompute_predictions or not os.path.isfile(mesh_path):
                        for time_id in select_tt:
                            mesh_prediction, _, _ = get_mesh_predictions(model = model,
                                                                  patches = padded_patches[time_id],
                                                                  ids_hexagonal_patch = p_ids_hexagonal_patch_nopad,
                                                                  VV = tt_VV[time_id],
                                                                  refinement = self.refinement,
                                                                  padding = True,
                                                                  center = self.patch_zeromean,
                                                                  device = self.device,
                                                                 )
                            #print(mesh_prediction.shape)

                            mesh_predictions[seed_id].append(mesh_prediction)

                        mesh_predictions[seed_id] = np.asarray(mesh_predictions[seed_id])
                        if save_predictions:
                            with open(mesh_path, 'wb') as f:
                                np.save(f, mesh_predictions[seed_id])

                    else:  
                        if version == versions[0]:
                            print(' load saved predictions', end=' ')
                        with open(mesh_path, 'rb') as f:
                            mesh_predictions[seed_id] = np.load(f)

            self.predicted_meshvv_per_part[pname][version] = mesh_predictions[0]
        print()
         
        
    def model_input_output_per_part(self, pname, versions):
        if pname not in self.model_patchinput_per_part:
            self.model_patchinput_per_part[pname] = dict()
            self.model_patchoutput_per_part[pname] = dict()
            self.model_loss_weight_per_part[pname] = dict()
            
        rotation=0
            
        for version in versions:
            
            if version in self.model_patchinput_per_part[pname]:
                continue
            
            padded_patches, adjacency_matrix, tt_VV, FF, p_ids_hexagonal_patch, _ = self.load_patches(part = pname,
                                                                                         rotation = rotation,
                                                                                         version = version)
            
            

            del_id = np.where(p_ids_hexagonal_patch[0] == -1)[0]
            p_ids_hexagonal_patch_nopad = np.delete(p_ids_hexagonal_patch,del_id,1)

            
            
            meshfiles = [f.name for f in os.scandir(osp.join(self.data_fp_spec, version, pname)) if f.is_file() and ('.obj' in f.name or '.ply' in f.name) and 'reference' not in f.name]
            meshfiles.sort()
            select_tt = np.arange(0,len(meshfiles),1)
            
            
            print('Get weight matrix for loss calculation.')
            v_id, count = np.unique(p_ids_hexagonal_patch, return_counts=True)
            neg_id = np.where(v_id < 0)[0]
            count = np.delete(count, neg_id)
            v_id = np.delete(v_id, neg_id)
            id_counts = np.zeros(max(v_id)+1)
            id_counts[v_id] = count
            non_padding_vertices = np.where(p_ids_hexagonal_patch[0] >= 0)[0]

            loss_weights = np.zeros((padded_patches.shape[1], padded_patches.shape[2]))
            loss_weights[:, non_padding_vertices] = 1/id_counts[p_ids_hexagonal_patch[:,non_padding_vertices]]
            
            tt_VV_torch = torch.tensor(np.array(tt_VV), dtype = torch.float)
            FF_torch = torch.tensor(FF, dtype = torch.long).T
                
                
            if version == versions[0]:
                if self.best_validation:
                    print(' best validation models')
                else:
                    print(' final models')
            
            # only one model
            name  = self.model_name
            
            save_directory = self.model_fp
            patch_inputs = [[]]
            patch_outputs = [[]]
            patch_weights = [[]]

            # take first seed only
            seed_id = 0
            seed  = self.seed[0]

            model_name = 'saved_model' if not self.best_validation else 'best_validation_model'
            model_path = save_directory + '/' + name + '/seed_{0}/{1}.pth'.format(seed, model_name)
            if version == versions[0]:
                print(model_name, model_path)
            else:
                print('.',end='')
            mesh_path = save_directory + '/' + name + '/seed_{0}/{1}_mesh_preds_{2}_{3}.npy'.format(seed, model_name, pname, version)
            model = self.models[name]()
            model.load_state_dict(torch.load(model_path, map_location = self.device))
            model = model.to(self.device)

            for time_id in select_tt:
                ## reindexing of the patches and centering
                _, patch_input, patch_output = get_mesh_predictions(model = model,
                                                      patches = padded_patches[time_id],
                                                      ids_hexagonal_patch = p_ids_hexagonal_patch_nopad,
                                                      VV = tt_VV[time_id],
                                                      refinement = self.refinement,
                                                      padding = True,
                                                      center = self.patch_zeromean,
                                                      device = self.device,
                                                     )

                
                patch_inputs[seed_id].append(patch_input)
                patch_outputs[seed_id].append(patch_output)
                patch_weights[seed_id].append(loss_weights[:, cosma.PAD_INDEX_MAPPING_KS2[self.refinement]])


            self.model_patchinput_per_part[pname][version] = np.asarray(patch_inputs[seed_id])
            self.model_patchoutput_per_part[pname][version] = np.asarray(patch_outputs[seed_id])
            self.model_loss_weight_per_part[pname][version] = np.asarray(patch_weights[seed_id])
        print()
    
    def get_emb_per_part(self, pname, versions, recompute_predictions=False, save_predictions=True):
        
        if pname not in self.emb_per_part:        
            self.emb_per_part[pname] = dict()
            
        rotation=0
            
        for version in versions:
            
            if version in self.emb_per_part[pname]:
                continue
            
            padded_patches, adjacency_matrix, tt_VV, FF, p_ids_hexagonal_patch, _ = self.load_patches(part = pname,
                                                                                         rotation = rotation,
                                                                                         version = version)
            
            del_id = np.where(p_ids_hexagonal_patch[0] == -1)[0]
            p_ids_hexagonal_patch_nopad = np.delete(p_ids_hexagonal_patch,del_id,1)            
            
            meshfiles = [f.name for f in os.scandir(osp.join(self.data_fp_spec, version, pname)) if f.is_file() and ('.obj' in f.name or '.ply' in f.name) and 'reference' not in f.name]
            meshfiles.sort()
            select_tt = np.arange(0,len(meshfiles),1)
                
            tt_VV_torch = torch.tensor(np.array(tt_VV), dtype = torch.float)
            FF_torch = torch.tensor(FF, dtype = torch.long).T
                
            if version == versions[0]:
                if self.best_validation:
                    print('\n best validation models')
                else:
                    print('\n final models')
            for name in [self.model_name]:
                save_directory = self.model_fp
                mesh_embeddings = [[] for _ in range(len(self.seed))]
                for seed_id, seed in enumerate(self.seed):
                    model_name = 'saved_model' if not self.best_validation else 'best_validation_model'
                    model_path = save_directory + '/' + name + '/seed_{0}/{1}.pth'.format(seed, model_name)
                    emb_path = save_directory + '/' + name + '/seed_{0}/{1}_mesh_emb_{2}_{3}.npy'.format(seed, model_name, pname, version)
                        
                    model = self.models[name]()
                    model.load_state_dict(torch.load(model_path, map_location = self.device))
                    model = model.to(self.device)

                    if recompute_predictions or not os.path.isfile(emb_path):
                        for time_id in select_tt:
                            mesh_embedding = get_mesh_embeddings(model = model,
                                                                  patches = padded_patches[time_id],
                                                                  ids_hexagonal_patch = p_ids_hexagonal_patch_nopad,
                                                                  VV = tt_VV[time_id],
                                                                  refinement = self.refinement,
                                                                  padding = True,
                                                                  center = self.patch_zeromean,
                                                                  device = self.device,
                                                                 )
                            #print(mesh_prediction.shape)

                            mesh_embeddings[seed_id].append(mesh_embedding)

                        mesh_embeddings[seed_id] = np.asarray(mesh_embeddings[seed_id])
                        if save_predictions:
                            with open(emb_path, 'wb') as f:
                                np.save(f, mesh_embeddings[seed_id])

                    else:    
                        if version == versions[0]:
                            print(' load')
                        with open(emb_path, 'rb') as f:
                            mesh_embeddings[seed_id] = np.load(f)
            
            self.emb_per_part[pname][version] = mesh_embeddings[0]
