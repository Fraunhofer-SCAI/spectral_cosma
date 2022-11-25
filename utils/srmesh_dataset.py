import numpy as np

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

import matplotlib.pyplot as plt
import matplotlib
import torch
import torch.nn.functional as F

import os
import os.path as osp
import pickle
import argparse
import time

import pandas as pd

import igl

from utils import visualize_mesh_matplotlib
from utils import utils, normalize



def get_error_message(err_array, err_name):
    if len(err_array):
        my_errors = np.asarray(err_array)  # [n_train_graphs, num_nodes]
        mean_error = my_errors.reshape((-1, )).mean(); #mse2=mean_error
        std_error = my_errors.reshape((-1, )).std()
        median_error = np.median(my_errors.reshape((-1, ))) 
        message = '{} Error: {:.6f}+{:.6f} | {:.6f}'.format(err_name, mean_error, std_error, median_error)
    else:
        message = '{} Error: {:8s}+{:8s} | {:8s}'.format(err_name, '  --','  --','  --')
        
    return message



## simulations which are in branch one. the simulations which are not listed show a different deformation behavior
br1 = ['004', '018', '022', '023', '026', '027', '029', '034', '038', '041', '042', '043', '044', '045', '048', '059', '062', '067', '068', '071', '075', '085', '087', '088', 
       '089', '091', '094', '095', '096', '099', '100', '103', '105', '106', '108', '110', '111', '113', '114', '121', '123', '124', '125', '128', '220', '223', '228', 
       '229', '231', '233', '236', '237', '244', '246', '249', '250', '252', '262', '264', '271', 
       '273',  '274', '275', '276', '277', '284', '285', '289', '293', '294', '297', '299',
       '973','972','970','964'] # last line is YARIS




class srmesh_dataset:
    def __init__(self, args, versions, samples, test_samples, device, Euclidean_Error=False):
        self.dataset = args.dataset
        self.data_fp = args.data_fp
        self.exp_name = args.exp_name
        self.refine = args.refine
        self.versions = versions
        self.samples = samples
        self.test_samples = test_samples
        self.train_samples = [sa for sa in samples if sa  not in test_samples]
        self.model_name = args.model_name
        self.hid_rep = args.hid_rep
        self.kernel_size = args.kernel_size
        self.device = device

        self.Euclidean_Error = Euclidean_Error
        
        self.data_train_patches = osp.join(self.data_fp, 'train_patches_{}'.format(self.exp_name))
            
        self.patch_zeromean = args.patch_zeromean
        self.meani = None
        
        self._N_triangles = dict()
        self.predicted_meshvv_per_part = dict()
        self.true_meshvv_per_part = dict()
        self.true_irregular_meshvv_per_part = dict()
        self.semireg_mesh_per_part = dict()
        self.irregular_mesh_per_part = dict()
        self.emb_per_part = dict()
        self.test_train_meshes = dict()
        
        self._base_mesh_per_part = dict()
        self._all_out = None
        self._test_out = None
        self._train_out = None
        self._all_input = None
        self._train_input = None
        self._test_input = None
        self._all_emb = None

        
    def get_irregular_mesh_per_part(self, pname):

        ##### load the irregular reference mesh: this is necessary for the p2s distance
        data_prepro_part = osp.join(self.data_fp, 'preprocessed', pname)
        irregular_mesh = os.path.join(data_prepro_part, 
                          'mesh_{}_reference.obj'.format(pname))

        VV, _, _, FF, _, _ = igl.read_obj(irregular_mesh)   
        print('  -> irregular mesh vertices:',VV.shape)
        
        self.irregular_mesh_per_part[pname] = [VV, FF]
        
    def get_irregular_meshvv_per_part(self, pname, versions):
        
        if pname not in self.true_irregular_meshvv_per_part:
            self.true_irregular_meshvv_per_part[pname] = dict()
        
        ##### load the semireg mesh: this is necessary for the projection
        data_prepro_part = osp.join(self.data_fp, 'preprocessed', pname)

        
        for version in versions:
            
            if version in self.true_irregular_meshvv_per_part[pname]:
                continue

            ## load the projection of the irregular mesh over time
            pickle_data = os.path.join(data_prepro_part, 
                                "mesh_{}_{}_vertex_values.p".format(pname, version))
            


            with open(pickle_data, "rb") as file:
                projected_VV = pickle.load(file)
                
            ## (#vertices, #timesteps, 3)
            projected_VV = np.swapaxes(projected_VV, 0, 1)
            ## (#timesteps, #vertices, 3)
            
            # move data to range -1,1 for each timestep
            ## load the min, max, mean values of the irregular meshes
            normalization_values_file = osp.join(self.data_fp, 'raw', version, pname, 'normalization_min_max_values.txt')
            if osp.isfile(normalization_values_file):
                norm_val = np.loadtxt(normalization_values_file)
                # move data to range -1,1 for each timestep
                if 'car' in self.dataset:
                    normalization_means_file = osp.join(self.data_fp, 'raw', version, pname, 'normalization_mean_values.txt')
                    vval_mean = np.loadtxt(normalization_means_file)
                    projected_VV_cc, v_min, v_max, _ = normalize.normalize(projected_VV, 
                                                ntype='range-0,1-mean-0', return_minmax=True,
                                                vval_min = norm_val[0], vval_max = norm_val[1], 
                                                vval_mean_t = vval_mean) 
                    projected_VV_cc = ( projected_VV_cc * 2 ) - 1
                    total_range = v_max - v_min 
                    if version == self.versions[0]:
                        print('  -> total_range:', total_range/10, 'cm')
                else:
                    projected_VV_cc = ( normalize.normalize(projected_VV, ntype='range-0,1',
                                                   vval_min = norm_val[0], vval_max = norm_val[1]) * 2 ) - 1

            
            self.true_irregular_meshvv_per_part[pname][version] = projected_VV_cc

    
    def get_semireg_mesh_per_part(self, pname):

        ##### load the semireg mesh: this is necessary for the projection
        data_semireg_part = osp.join(self.data_fp, 'semiregular', pname)
        if 'norot' in self.exp_name or 'inter' in self.exp_name:
            semiregular_mesh = os.path.join(data_semireg_part, 
                      '{}_remesh_exp_{}_refinelevel_{}.obj'.format(pname,self.exp_name[:-6],self.refine))
        else:
            semiregular_mesh = os.path.join(data_semireg_part, 
                      '{}_remesh_exp_{}_refinelevel_{}.obj'.format(pname,self.exp_name,self.refine))
        VV, _, _, FF, _, _ = igl.read_obj(semiregular_mesh)   
        print('  -> remeshed mesh vertices:',VV.shape)
        
        self.semireg_mesh_per_part[pname] = [VV, FF]
        
    def N_triangles(self, pname):
        if pname not in self._N_triangles:
            self.base_mesh_per_part(pname)
        return self._N_triangles[pname]
        
    def base_mesh_per_part(self, pname):
        
        if pname not in self._base_mesh_per_part:
            ##### locate the base mesh
            data_preprocessed_part = osp.join(self.data_fp, 'preprocessed', pname)
            if 'norot' in self.exp_name or 'inter' in self.exp_name:
                base_part = os.path.join(data_preprocessed_part, "mesh_{}_remesh_exp_{}.obj".format(pname, self.exp_name[:-6]))
            else:
                base_part = os.path.join(data_preprocessed_part, "mesh_{}_remesh_exp_{}.obj".format(pname, self.exp_name))

            ##### Load Base Data    
            VV_base, _, _, FF_base, _, _ = igl.read_obj(base_part) 
            _, FF_base, EE_base, boundary_edges, boundary_vertices, nonmanifold_edges = utils.clean_mesh_get_edges(VV_base, FF_base)

            self._N_triangles[pname] = len(FF_base)
            self._base_mesh_per_part[pname] = [VV_base, FF_base]
        
        return self._base_mesh_per_part[pname][0], self._base_mesh_per_part[pname][1]
    

            
    def mesh_reconstruction_error(self, samples, versions, logfile=None, p2s=False):
        ## p2s: point to surface distance
        for pname in samples:
            if pname not in self.predicted_meshvv_per_part:
                print('ERROR: First construct the predicted meshes for part {}'.format(pname)) 
                return 1.0
        else:
            for version in versions:
                for panme in samples:
                    if version not in self.predicted_meshvv_per_part[pname]:
                        print('ERROR: First construct the predicted meshes for part {} and version {}'.format(pname, version))
                        return 1.0
                    
        all_test_error = []
        all_train_error = []
        
        if p2s:
            if logfile is not None:
                if self.Euclidean_Error:
                    logfile.write('\n\nPoint2Surface Distance (Euclidean Errors):\n\n')
                else:
                    logfile.write('\n\nPoint2Surface Distance:\n\n')
            if self.Euclidean_Error:
                print('\n\nPoint2Surface Distance (Euclidean Errors):\n')
            else:
                print('\n\nPoint2Surface Distance:\n')
        
        for pname in samples:
            if p2s:
                train_error_1s, test_error_1s = self.mesh_reconstruction_error_p2s( pname, versions, logfile=logfile)
            else:    
                train_error_1s, test_error_1s = self.mesh_reconstruction_error_1s( pname, versions, logfile=logfile)
            
            
            
            if len(all_test_error):
                all_test_error = np.concatenate([all_test_error,np.asarray(test_error_1s).flatten()])
            else:
                all_test_error = np.asarray(test_error_1s).flatten()
            if len(all_train_error):
                all_train_error = np.concatenate([all_train_error,np.asarray(train_error_1s).flatten()])
            else:
                all_train_error = np.asarray(train_error_1s).flatten()
                

            
        print('\n\nTotal Mean Squared Error:\n')
        message = get_error_message([all_train_error], 'Train')
        print(message)
        message2 = get_error_message([all_test_error], 'Test ')
        print(message2)
        
        if logfile is not None:
            logfile.write('\n\nTotal Mean Squared Error:\n\n')
            logfile.write(message+'\n')
            logfile.write(message2+'\n\n')
            
    def get_partwise_total_range(self,version,pname):

        normalization_values_file = osp.join(self.data_fp, 'raw', version, pname, 'normalization_min_max_values.txt')
        norm_val = np.loadtxt(normalization_values_file)
        total_range = norm_val[1] - norm_val[0] # max - min 
        return total_range
            
    def mesh_reconstruction_error_p2s(self, pname, versions, logfile=None, printmess = True):
        # print and calculate error for one sample

        if pname not in self.predicted_meshvv_per_part:
            print('ERROR: First construct the predicted meshes for part {}'.format(pname)) 
            return 1.0
        else:
            for version in versions:
                if version not in self.predicted_meshvv_per_part[pname]:
                    print('ERROR: First construct the predicted meshes for part {} and version {}'.format(pname, version))
                    return 1.0
                
        if pname not in self.irregular_mesh_per_part:
            self.get_irregular_mesh_per_part(pname)
        self.get_irregular_meshvv_per_part(pname, versions)
        
        VV_target_r, FF_target = self.irregular_mesh_per_part[pname]
        VV_target = self.true_irregular_meshvv_per_part[pname]
                    
        self.get_test_train_meshes(pname, versions)

        mesh_pred = self.predicted_meshvv_per_part[pname]

        test_error = []
        train_error = []
        for version in versions:
            if self.Euclidean_Error:

                total_range = self.get_partwise_total_range(version,pname)
                
                # original range and from mm to cm
                tmp_pred = (mesh_pred[version]) * (total_range/(2.0*10.0)) #/2) / 10
                tmp_true = (VV_target[version]) * (total_range/(2.0*10.0)) #/2) / 10
                for tt in range(len(self.predicted_meshvv_per_part[pname][version])):
                    dis, _, _ = igl.point_mesh_squared_distance(tmp_pred[tt], tmp_true[tt], FF_target)
                    if self.test_train_meshes[pname][version][tt]:
                        test_error.append(dis)
                    else:
                        train_error.append(dis)
            else:
                for tt in range(len(self.predicted_meshvv_per_part[pname][version])):
                    dis, _, _ = igl.point_mesh_squared_distance(mesh_pred[version][tt], VV_target[version][tt], FF_target)
                    if self.test_train_meshes[pname][version][tt]:

                        test_error.append(dis)
                    else:

                        train_error.append(dis)
                        
                        
        if self.Euclidean_Error:
            # norm to cm
            train_error = np.sqrt(np.asarray(train_error))
            test_error = np.sqrt(np.asarray(test_error))

        # test_error : [n_test_graphs, num_nodes]
        # train_error: [n_train_graphs, num_nodes]         

        if printmess:
            if self.Euclidean_Error:
                print('################ - Euclidean Error')
            else:
                print('################')
            print('Sample', pname)
            message = get_error_message(train_error, 'Train')
            print(message)
            message2 = get_error_message(test_error, 'Test ')
            print(message2+'\n')
        
        if logfile is not None:
            logfile.write('\n################\nSample {}\n'.format(pname))
            logfile.write(message+'\n')
            logfile.write(message2+'\n\n')

        return train_error, test_error   
    
        
    def mesh_reconstruction_error_1s(self, pname, versions, logfile=None, printmess = True):
        # print and calculate error for one sample

        if pname not in self.predicted_meshvv_per_part:
            print('ERROR: First construct the predicted meshes for part {}'.format(pname)) 
            return 1.0
        else:
            for version in versions:
                if version not in self.predicted_meshvv_per_part[pname]:
                    print('ERROR: First construct the predicted meshes for part {} and version {}'.format(pname, version))
                    return 1.0
                    

        self.get_true_meshvv_per_part(pname, versions)
        self.get_test_train_meshes(pname, versions)

        mesh_pred = self.predicted_meshvv_per_part[pname]
        mesh_true = self.true_meshvv_per_part[pname] 

        
        
        test_error = []
        train_error = []
        for version in versions:
            if self.Euclidean_Error:
                total_range = self.get_partwise_total_range(version,pname)
                
                # original range and from mm to cm
                tmp_pred = (mesh_pred[version]) * (total_range/(2.0*10.0)) #/2) / 10
                tmp_true = np.array(mesh_true[version]) * (total_range/(2.0*10.0)) #/2) / 10
                
                for tt in range(len(self.predicted_meshvv_per_part[pname][version])):
                    
                    if self.test_train_meshes[pname][version][tt]:
                        test_error.append(np.sum((tmp_pred[tt] - tmp_true[tt])**2, axis=1) )
                    else:
                        train_error.append(np.sum((tmp_pred[tt] - tmp_true[tt])**2, axis=1) )
            else:
                for tt in range(len(self.predicted_meshvv_per_part[pname][version])):
                    if self.test_train_meshes[pname][version][tt]:
                        test_error.append(np.sum((mesh_pred[version][tt] - mesh_true[version][tt])**2, axis=1) )
                    else:
                        train_error.append(np.sum((mesh_pred[version][tt] - mesh_true[version][tt])**2, axis=1) )

        # test_error : [n_test_graphs, num_nodes]
        # train_error: [n_train_graphs, num_nodes]      
        if self.Euclidean_Error:
            # norm to cm
            train_error = np.sqrt(np.asarray(train_error))
            test_error = np.sqrt(np.asarray(test_error))

        if printmess:
            print('\n################')
            print('Sample', pname)
            message = get_error_message(train_error, 'Train')
            print(message)
            message2 = get_error_message(test_error, 'Test ')
            print(message2)
        
        if logfile is not None:
            logfile.write('\n################\nSample {}\n'.format(pname))
            logfile.write(message+'\n')
            logfile.write(message2+'\n\n')

        return train_error, test_error

        
        #timesteps = 
        
    def plot_true_meshes(self, time_id, version_id, pnames = None, view = (10, 18), saveplots=True):
        
        if pnames is None:
            pnames = self.samples
        fig, axs = plt.subplots((len(pnames)), 
                               subplot_kw=dict(projection='3d'),
                               figsize = (13, 13 * len(pnames)))
        if len(pnames)==1:
            axs = [axs]
        plt.rcParams.update({'font.size': 20})
        fig.suptitle('True meshes t={}'.format(time_id), fontsize = 30)
        for pn, pname in enumerate(pnames):
            p_VV = self.true_meshvv_per_part[pname][self.versions[version_id]]
            p_FF = self.semireg_mesh_per_part[pname][1]

            tmp_true = p_VV
                

            x = tmp_true[time_id][:, 0]
            y = tmp_true[time_id][:, 2]
            z = tmp_true[time_id][:, 1]
            
            if 'car' in self.dataset: 
                limits = [(x.min(), x.min()+1.5),
                   (y.min(), y.min()+1.5),                                                       
                   (z.min(), z.min()+1.5),]
            elif 'FAUST' in self.dataset:
                limits = [(x.min()-0.5, x.min()+1),
                   (y.min(), y.min()+1.5),                                                       
                   (z.min(), z.min()+1.5),]
            else:
                limits = [(x.min()-0.5, x.min()+1),
                    (y.min()-0.5, y.min()+1),                                                       
                    (z.min()-0.5, z.min()+1),]

            lw = 0.02
            if pname == 'elephant':
                lw=1
            fig, axs[pn] = visualize_mesh_matplotlib(tmp_true[time_id][:, [0, 2, 1]], p_FF, None, lw=lw,
                                                     fig = fig, ax = axs[pn],
                                                     view = view, 
                                                     limits = limits)



            name = 'True Mesh {}'.format(pname)

            axs[pn].set_title(name)
            
        save_path = 'plots/true_mesh_plots_{}_rf_{}_cosma_t_{}_model_{}_{}_{}.png'.format(self.dataset,self.refine,time_id,self.model_name,pnames, self.versions[version_id])
        if saveplots:
            plt.savefig(save_path)
        fig.show()
        
    def plot_true_irr_meshes(self, time_id, version_id, pnames = None, view = (10, 18), saveplots=True):
        
        if pnames is None:
            pnames = self.samples
        fig, axs = plt.subplots((len(pnames)), 
                               subplot_kw=dict(projection='3d'),
                               figsize = (13, 13 * len(pnames)))
        if len(pnames)==1:
            axs = [axs]
        plt.rcParams.update({'font.size': 20})
        fig.suptitle('True irregular meshes t={}'.format(time_id), fontsize = 30)
        for pn, pname in enumerate(pnames):
            p_VV = self.true_irregular_meshvv_per_part[pname][self.versions[version_id]]
            p_FF = self.irregular_mesh_per_part[pname][1]

            tmp_true = p_VV
                

            x = tmp_true[time_id][:, 0]
            y = tmp_true[time_id][:, 2]
            z = tmp_true[time_id][:, 1]
            
            if 'car' in self.dataset: 
                limits = [(x.min(), x.min()+1.5),
                   (y.min(), y.min()+1.5),                                                       
                   (z.min(), z.min()+1.5),]
            elif 'FAUST' in self.dataset:
                limits = [(x.min()-0.5, x.min()+1),
                   (y.min(), y.min()+1.5),                                                       
                   (z.min(), z.min()+1.5),]
            else:
                limits = [(x.min()-0.5, x.min()+1),
                    (y.min()-0.5, y.min()+1),                                                       
                    (z.min()-0.5, z.min()+1),]

            lw = 0.02
            if pname == 'elephant':
                lw=1
            fig, axs[pn] = visualize_mesh_matplotlib(tmp_true[time_id][:, [0, 2, 1]], p_FF, None, lw=lw,
                                                     fig = fig, ax = axs[pn],
                                                     view = view, 
                                                     limits = limits)



            name = 'True irregular Mesh {}'.format(pname)

            axs[pn].set_title(name)
            
        save_path = 'plots/true_irr_mesh_plots_{}_t_{}_model_{}_{}.png'.format(self.dataset,time_id,pnames, self.versions[version_id])
        if saveplots:
            plt.savefig(save_path)
        fig.show()
        
    def plot_reconstruction_meshes(self, time_id, version_id, pnames = None, vmin = None, vmax = None, view = (10, 18), p2s=False, saveplots=True):
        
        if pnames is None:
            pnames = self.samples
        fig, axs = plt.subplots((len(pnames)), 
                               subplot_kw=dict(projection='3d'),
                               figsize = (13, 13 * len(pnames)))
        if len(pnames)==1:
            axs = [axs]
        plt.rcParams.update({'font.size': 20})
        fig.suptitle('Prediction errors spatial network t={}'.format(time_id), fontsize = 30)
        for pn, pname in enumerate(pnames):
            mesh_predictions = self.predicted_meshvv_per_part[pname][self.versions[version_id]]
            p_VV = self.true_meshvv_per_part[pname][self.versions[version_id]]
            p_FF = self.semireg_mesh_per_part[pname][1]

            if self.Euclidean_Error:
                print('euclidean error')
                total_range = self.get_partwise_total_range(self.versions[version_id],pname)
                
                # original range and from mm to cm
                tmp_pred = (mesh_predictions) * (total_range/(2.0*10.0)) #/2) / 10
                tmp_true = np.array(p_VV) * (total_range/(2.0*10.0)) #/2) / 10
            else:
                tmp_pred = mesh_predictions
                tmp_true = p_VV
                
            
            mesh_errors = np.sum((tmp_pred[time_id] - tmp_true[time_id])**2, axis=-1)
            if p2s:
                _, FF_target = self.irregular_mesh_per_part[pname]
                VV_target = self.true_irregular_meshvv_per_part[pname]
                if self.Euclidean_Error:
                    tmp_true = np.array(VV_target[self.versions[version_id]]) * (total_range/(2.0*10.0)) #/2) / 10
                else:
                    tmp_true = VV_target[self.versions[version_id]]
                mesh_errors, _, _ = igl.point_mesh_squared_distance(tmp_pred[time_id], tmp_true[time_id], FF_target)
                
            face_errors = [np.mean(mesh_errors[face]) for face in p_FF]

            x = mesh_predictions[time_id][:, 0]
            y = mesh_predictions[time_id][:, 2]
            z = mesh_predictions[time_id][:, 1]
            
            if 'car' in self.dataset: 
                limits = [(x.min(), x.min()+1.5),
                   (y.min(), y.min()+1.5),                                                       
                   (z.min(), z.min()+1.5),]
            elif 'FAUST' in self.dataset:
                limits = [(x.min()-0.5, x.min()+1),
                   (y.min(), y.min()+1.5),                                                       
                   (z.min(), z.min()+1.5),]
            else:
                limits = [(x.min()-0.5, x.min()+1),
                    (y.min()-0.5, y.min()+1),                                                       
                    (z.min()-0.5, z.min()+1),]

            fig, axs[pn] = visualize_mesh_matplotlib(mesh_predictions[time_id][:, [0, 2, 1]], p_FF, face_errors, 
                                                     fig = fig, ax = axs[pn],
                                                     view = view, vmin = vmin, vmax = vmax, 
                                                     limits = limits)

            if p2s:
                trerr, tserr = self.mesh_reconstruction_error_p2s(pname, self.versions, printmess=False)
            else:
                trerr, tserr = self.mesh_reconstruction_error_1s(pname, self.versions, printmess=False)
            
            if len(trerr):
                trerr = '{:6f}'.format(np.asarray(trerr).reshape((-1, )).mean())
            else:
                trerr = '--'
            tserr = '{:6f}'.format(np.asarray(tserr).reshape((-1, )).mean())

            name = '{}: train-MSE {}, test-MSE {}'.format(pname, trerr, tserr)
            if p2s:
                name = 'P2S-Error - '+name
            axs[pn].set_title(name)
        save_path = 'plots/reconstruction_plots_{}_rf_{}_cosma_t_{}_model_{}_{}_{}.png'.format(self.dataset,self.refine,time_id,self.model_name,pnames, self.versions[version_id])
        if p2s:
            save_path = 'plots/reconstruction_plots_{}_rf_{}_cosma_t_{}_model_{}_p2s_{}_{}.png'.format(self.dataset,self.refine,time_id,self.model_name,pnames, self.versions[version_id])
        if saveplots:
            plt.savefig(save_path)
        fig.show()

    def plot_mesh_with_highlighted_patch(self, pname, version, tt, patch_vv, patch_FF, view = (180,180), figsize = (10,10), true_mesh=True, savefig=None):
        if true_mesh:
            VV = self.true_meshvv_per_part[pname][version][tt]
        else:
            VV = self.predicted_meshvv_per_part[pname][version][tt]
        self.get_semireg_mesh_per_part(pname)
        _, FF_semireg = self.semireg_mesh_per_part[pname]
        fig, ax  = plt.subplots(1, 1, figsize = figsize, subplot_kw = dict(projection='3d'))
        cc = 'k'
        for ff in FF_semireg:
            ff_tmp = np.append(ff, ff[0])
            ax.plot(VV[ff_tmp][:, 0], VV[ff_tmp][:, 2], -VV[ff_tmp][:, 1], color = cc, lw=0.1)
        cc = 'purple'
        for ff in patch_FF:
            ff_tmp = np.append(ff, ff[0])
            ax.plot(patch_vv[ff_tmp][:, 0], patch_vv[ff_tmp][:, 2], -patch_vv[ff_tmp][:, 1], color = cc, lw=0.4)
        ax.axis('off')
        #ax[ii].set_xlim(-1.5,1.5); ax[ii].set_ylim(-1.5,1.5); ax[ii].set_zlim(-1.5,1.5);
        ax.view_init(*view)
        ax.axis('tight')
        if true_mesh:
            ax.set_title('{}, true mesh'.format(pname))
        else:
            ax.set_title('{}, reconstructed mesh'.format(pname))
            
        if savefig is not None:
            fig.savefig(savefig+'.png')


    def embedding_per_part_over_time(self, pname, methods, labels, first_tt = 0, fig = None, ax = None, figsize = (19, 13.5), global_pool = None, sorting_necessary = False):
        """
        Plot the embedding for a part over time.
        the patch wise embeddings are concatenated
        """
        self.get_emb_per_part(pname, self.versions)
        self.get_test_train_meshes(pname, self.versions)

        # results per patch per part and version
        # concatenate the results
        embedding_all_v_all_p = self.get_embedding_all_v_all_patch(pname, first_tt)

        if global_pool == 'mean':
            embedding_all_v = np.mean(embedding_all_v_all_p, axis=(1))
        elif global_pool == 'max':
            embedding_all_v = np.max(embedding_all_v_all_p, axis=(1))
        else:
            # results per part and version (concatenate the patch results)
            embedding_all_v = np.resize(embedding_all_v_all_p, (embedding_all_v_all_p.shape[0], 
                                                                embedding_all_v_all_p.shape[1]*embedding_all_v_all_p.shape[2]))
            print('embedding_all_v', embedding_all_v.shape)


        for label, method in enumerate(methods):
            emb2= method.fit_transform(embedding_all_v)
            if fig is None or ax is None:
                fig, ax  = plt.subplots(figsize = figsize)
            count_samples = 0
            for kk, version in enumerate(self.versions):
                meshfiles = [f.name for f in os.scandir(osp.join(self.data_fp, 'raw', 
                     version, pname)) if f.is_file() and 
                 ('.obj' in f.name or '.ply' in f.name) and 
                 'reference' not in f.name and 'ipynb_checkpoints' not in f.name] 
                meshfiles.sort()

                select_tt = np.arange(0,len(meshfiles[first_tt:]),1) 

                if 'car_TRUCK' in self.dataset or 'car_YARIS' in self.dataset:
                    if version[-3:] in br1:
                        cmap = utils.truncate_colormap(matplotlib.cm.Blues, minval=0.5, maxval=1.0)
                    else:
                        cmap = utils.truncate_colormap(matplotlib.cm.Greens, minval=0.5, maxval=1.0)
                else:
                    cmap = utils.truncate_colormap(matplotlib.cm.Greens, minval=0.2, maxval=1.0)


                if 'car' in self.dataset:
                    for tt in select_tt:
                            #if tt >= first_tt:
                            cc = cmap(float(tt) / (len(select_tt)))
                            if self.test_train_meshes[pname][version][tt]:
                                # test mesh
                                marker = 'o'
                            else:
                                marker = 'o'

                            ax.scatter(emb2[count_samples  + tt,0], 
                                        emb2[count_samples  + tt,1], 
                                        color= cc, marker = marker, s=180, edgecolors=cc)
                    #if version in [hexasemiregdat.versions[0],hexasemiregdat.versions[1]]:
                    #    if version[-3:] in br1:
                    #        text = 'Branch1'
                    #    else:
                    #        text = 'Branch2'
                    #    ax.scatter(emb2[count_samples  + time_ind_sorted[tt],0], 
                    #                        emb2[count_samples  + time_ind_sorted[tt],1], 
                    #                        color= cc, marker = marker, s=180, edgecolors=cc, label=text)
                else:
                    ax.plot(emb2[count_samples:count_samples+len(select_tt),0], 
                                emb2[count_samples:count_samples+len(select_tt),1], label=pname, lw=3)

                    for tt in select_tt[:int(len(select_tt)/4)]:
                        ax.text(emb2[count_samples+tt,0], emb2[count_samples+tt,1], '{}'.format(tt), fontsize = 25)
                        #if kk == 0:
                        #    plt.text(emb2[time_ind_sorted[tt],0]+0.01, emb2[time_ind_sorted[tt],1]+0.01, '{}'.format(tt+1), fontsize = 14)
                count_samples += len(select_tt)
        if global_pool is not None:
            title = labels[label] +' with global pooling ('+ global_pool + ')'
        else:
            title = labels[label] +' without global pooling, concatenate patches'
        ax.axis('off')
        ax.axis('tight')
        ax.set_title(title)
        return fig, ax


    def get_embedding_all_v_all_patch(self, pname, first_tt=0):
        # concatenate embeddings per patch for one part.
        # versions are also concatenated but
        embedding_all_v_all_p = np.zeros((0, self.N_triangles(pname), self.hid_rep))
        for vv in self.versions:
            embedding_all_v_all_p = np.concatenate([embedding_all_v_all_p, self.emb_per_part[pname][vv][first_tt:]])
            
        #output of shape (#timesteps, #patches, hid_dim)
        return embedding_all_v_all_p
    
    def embedding_per_part_per_patch_over_time_original(self, pname, methods, labels, first_tt=0, figsize = (19, 13.5), plotfigures=False,savefigures=None):
        """
        Plot the embedding for each patch of a part over time.
        """
        self.get_emb_per_part(pname, self.versions)

        # results per patch per part and version
        # concatenate the results
        embedding_all_v_all_p = self.get_embedding_all_v_all_patch(pname, first_tt)
        print('{}: embedding_all_v_all_p'.format(pname), embedding_all_v_all_p.shape)
        self.get_test_train_meshes(pname, self.versions)
        
        collect_patchwise_embedding=[]
        
        if savefigures is not None:
            plotfigures=True

        for label, method in enumerate(methods):        

            if plotfigures:
            
                rows = int(np.ceil(np.sqrt(embedding_all_v_all_p.shape[1])))
                fig, ax = plt.subplots(  rows, rows, figsize=(20,20))
                
                
            for pp in range(embedding_all_v_all_p.shape[1]):
                # iterate over patches
                emb2= method.fit_transform(embedding_all_v_all_p[:,pp])
                collect_patchwise_embedding += [emb2]
                #print(method, emb2.shape)

                if plotfigures:
                    row = int(pp/rows)
                    col = pp % rows

                    count_samples = 0
                    for kk, version in enumerate(self.versions):

                        meshfiles = [f.name for f in os.scandir(osp.join(self.data_fp, 'raw', 
                                 version, pname)) if f.is_file() and 
                             ('.obj' in f.name or '.ply' in f.name) and 
                             'reference' not in f.name and 'ipynb_checkpoints' not in f.name] 
                        meshfiles.sort()

                        select_tt = np.arange(0,len(meshfiles[first_tt:]),1) 


                        if 'car_TRUCK' in self.dataset:
                            if version[-3:] in br1:
                                cmap = utils.truncate_colormap(matplotlib.cm.Blues, minval=0.2, maxval=1.0)
                            else:
                                cmap = utils.truncate_colormap(matplotlib.cm.Greens, minval=0.2, maxval=1.0)
                        else:
                            cmap = utils.truncate_colormap(matplotlib.cm.Greens, minval=0.2, maxval=1.0)

                        marker = 'o'
                        if 'car_TRUCK' in self.dataset:
                            for tt in select_tt:
                                    #if tt >= first_tt:
                                    cc = cmap(float(tt) / (len(select_tt)))
                                    ax[row, col].scatter(emb2[count_samples + tt,0], 
                                                emb2[count_samples + tt,1], color= cc, marker = marker)
                        elif 'gallop' in self.dataset:
                            ax[row, col].plot(emb2[count_samples:count_samples + len(select_tt),0], 
                                        emb2[count_samples:count_samples + len(select_tt),1])


                        count_samples += len(select_tt)
                        
                    ax[row, col].set_title('patch {} '.format(pp) + pname +' '+ labels[label] )
                        
            if plotfigures:
                for row in range(rows):
                    for col in range(rows):
                        ax[row, col].axis('off')
                        
                if savefigures is not None:
                    fig.savefig(savefigures+'_{}.png'.format(labels[label]))
        return collect_patchwise_embedding
    

    def embedding_per_part_per_patch_over_time(self, pname, methods, labels, first_tt=0, figsize = (20, 20), plotfigures=False,savefigures=None):
        self.get_emb_per_part(pname, self.versions)
        # results per patch per part and version
        # concatenate the results
        embedding_all_v_all_p = self.get_embedding_all_v_all_patch(pname, first_tt)
        print('{}: embedding_all_v_all_p'.format(pname), embedding_all_v_all_p.shape)
        self.get_test_train_meshes(pname, self.versions)

        collect_patchwise_embedding=[]
        
        if savefigures is not None:
            plotfigures=True

        for label, method in enumerate(methods):

            if plotfigures:
                rows = int(np.ceil(np.sqrt(embedding_all_v_all_p.shape[1])))
                fig, ax = plt.subplots(  rows, rows, figsize=figsize)
            for pp in range(embedding_all_v_all_p.shape[1]):
                # iterate over patches
                if plotfigures:
                    row = int(pp/rows)
                    col = pp % rows
                    ax[row, col].set_title('patch {} '.format(pp) + pname +' '+ labels[label] )
                emb2= method.fit_transform(embedding_all_v_all_p[:,pp])
                count_samples = 0
                if 'car' in self.dataset:
                    x,y,z=[],[],[]
                    for kk, version in enumerate(self.versions):

                        meshfiles = [f.name for f in os.scandir(osp.join(self.data_fp, 'raw',
                                 version, pname)) if f.is_file() and
                             ('.obj' in f.name or '.ply' in f.name) and
                             'reference' not in f.name and 'ipynb_checkpoints' not in f.name]
                        meshfiles.sort()

                        #select_tt = np.arange(0,len(meshfiles[0:]),1)
                        select_tt = np.arange(0,len(meshfiles[first_tt:]),1)
                        c=0
                        if version[-3:] in br1:
                            cmap = utils.truncate_colormap(matplotlib.cm.Blues, minval=0.2, maxval=1.0)
                            c=1
                        else:
                            cmap = utils.truncate_colormap(matplotlib.cm.Greens, minval=0.2, maxval=1.0)
                            c=2

                        marker = 'o'
                        for tt in select_tt:
                            #if tt >= first_tt:
                            if plotfigures:
                                cc = cmap(float(tt) / (len(select_tt)))
                                ax[row, col].scatter(emb2[count_samples + tt,0],
                                            emb2[count_samples + tt,1], color= cc, marker = marker)
                            x.append(emb2[count_samples + tt,0])
                            y.append(emb2[count_samples + tt,1])
                            z.append(c)

                        count_samples += len(select_tt)

                    
                    
                    collect_patchwise_embedding.append(list(zip(x, y,z)))
                else:
                    x,y=[],[]
                    for kk, version in enumerate(self.versions):
                        meshfiles = [f.name for f in os.scandir(osp.join(self.data_fp, 'raw',
                                 version, pname)) if f.is_file() and
                             ('.obj' in f.name or '.ply' in f.name) and
                             'reference' not in f.name and 'ipynb_checkpoints' not in f.name]
                        meshfiles.sort()
                        select_tt = np.arange(0,len(meshfiles[first_tt:]),1)
                        for tt in select_tt:
                            x.append(emb2[count_samples + tt,0])
                            y.append(emb2[count_samples + tt,1])
                        count_samples += len(select_tt)
                        collect_patchwise_embedding.append(list(zip(normalise01(x),normalise01(y))))
                    if plotfigures:
                        if 'car' in self.dataset:
                            px=None
                            py=None
                        else:
                            pt_in_circle=1500
                            theta = np.linspace(0, 2*np.pi, pt_in_circle)
                            radius = 0.5
                            px=normalise01(radius*np.cos(theta))
                            py=normalise01(radius*np.sin(theta))
                        ax[row, col].plot(x,y, marker = 'o')
                        ax[row, col].plot(px, py, color="r")
                        ax[row, col].set_aspect(1)
                        ax[row, col].axis('tight')
                        ax[row, col].set_title('patch {} '.format(pp)+ labels[label] , pad = 0)

            if plotfigures:
                for row in range(rows):
                    for col in range(rows):
                        ax[row, col].axis('off')
                if savefigures is not None:
                    plt.savefig('{}.png'.format(savefigures), dpi=300)
                    print('save plots highlighting score per patch in {}.png'.format(savefigures))
        return collect_patchwise_embedding

    
    def visualize_patchwise_score(self, pname, method, label, view=(10, 18), first_tt=0, figsize = (20, 20), savefigures=None):
        """
        one part
        one method
        one label
        """

        print('collect the patch wise embeddings')
        collect_patchwise_embedding = self.embedding_per_part_per_patch_over_time(pname, [method], [label], first_tt)


        VV_base, FF_base = self.base_mesh_per_part(pname)
        EE_base, _, _, _ = utils.clean_mesh_get_edges(VV_base, FF_base, clean_mesh=False)
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        #for label, method in enumerate(list_all):
        collec=ax.plot_trisurf(VV_base[:, 0], VV_base[:,2], triangles=FF_base, Z=VV_base[:,1], shade=True, color='white', linewidth=0.02, edgecolor = "b", cmap = plt.cm.viridis)
        a,b=view
        if 'car' in self.dataset:
            result=np.array(training_SVC_per_patch(collect_patchwise_embedding).Testing_Accuracy.values.tolist())
            print(result)

            #for fn, ff in enumerate(FF_base):
            #    ax.text(np.mean(VV_base[ff][:,0]),
            #            np.mean(VV_base[ff][:,2]),
            #            np.mean(VV_base[ff][:,1]), '{}'.format(fn))

            collec.set_array(result)
            collec.set_clim(vmin= 0.5, vmax= 1.0)
            cb = plt.colorbar(collec, ax=ax, ticks=[0.5, 0.75, 1])
            #plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
            plt.title('{pname}, starting time: {starting_time} , label {label}'.format(pname=pname, starting_time=first_tt, label="PCA")   )
            xmin = np.min(VV_base[:,0])
            xmax = np.max(VV_base[:,0])
            xmean = (xmax+xmin)/2
            xdis = (xmax-xmin)/2
            ymean = (np.max(VV_base[:,1]) - np.min(VV_base[:,1]))/2
            zmean = (np.max(VV_base[:,2]) - np.min(VV_base[:,2]))/2
            ax.set_zlim(zmean-xdis,zmean+xdis)
            ax.set_ylim(ymean-xdis,ymean+xdis)
            ax.set_xlim(xmean-xdis,xmean+xdis)
        else:
            result=np.array(distance_embedding_circle(collect_patchwise_embedding))
            collec.set_array(result)
            collec.set_clim(vmin= 0.0, vmax= 0.19)
            cb = plt.colorbar(collec, ax=ax, ticks=[0,0.05,0.1,0.15])
            plt.title('{pname}, starting time: {starting_time} , {label}'.format(pname=pname, starting_time=first_tt, label=label)   )
            xmin = np.min(VV_base[:,0])
            ymin = np.min(VV_base[:,1])
            zmin = np.min(VV_base[:,2])
            ax.set_zlim(zmin, zmin+1)
            ax.set_ylim(ymin, ymin+1)
            ax.set_xlim(xmin, xmin+1)
        ax.view_init(a,b)
        ax.axis('off')
        cb.ax.tick_params(labelsize=50)
        if savefigures is not None:
            fig.savefig('{}.png'.format(savefigures), dpi=300)
            print('save colored plots highlighting score per patch in plots/{}.png'.format(savefigures))

        return
    
    
    

def distance_embedding_circle(list_all):#for gallop dataset
    dist_mean1=[]
    dist_mean2=[]

    pt_in_circle=1500
    theta= np.linspace(0, 2*np.pi, pt_in_circle)
    radius = 0.5
    px=normalise01(radius*np.cos(theta))
    py=normalise01(radius*np.sin(theta))
    circle_xy= np.column_stack((px,py))
    for plist in list_all:
        #interpolation
        emb_xy = np.concatenate([np.linspace(plist[i], plist[i+1], int(pt_in_circle/len(list_all))) for i in range (len(list_all[0])-2)])

        min_dist1=[]
        min_dist2=[]
        #min distance from each point in circle to all points in embedding
        for xp, yp in zip(px,py):
            circle_pt = np.array((xp,yp))
            distances1 = np.linalg.norm(emb_xy-circle_pt, axis=1)
            min_index1 = np.argmin(distances1)
            min_dist1.append(distances1[min_index1])
        #distance from circle to embedding
        for xa, ya in emb_xy:
            emb_pt = np.array((xa,ya))
            distances2 = np.linalg.norm(circle_xy-emb_pt, axis=1)
            min_index2 = np.argmin(distances2)
            min_dist2.append(distances2[min_index2])

        dist_mean1.append(np.mean(np.array(min_dist1)))
        dist_mean2.append(np.mean(np.array(min_dist2)))

    return [(g + h) / 2 for g, h in zip(dist_mean1, dist_mean2)]


def normalise01(x):#for gallop dataset
    xmin, xmax = min(x), max(x)
    for i, val in enumerate(x):
        x[i] = (val-xmin) / (xmax-xmin)
    return x


def training_SVC_per_patch(list_all):#for car dataset
    # Support Vector Classification
    all_df=[]
    for k in range(len(list_all)):
        all_df.append( pd.DataFrame(list_all[k],columns = ['x', 'y', 'br']))

    result= []
    header=[]
    i=0
    for df in all_df:
        patch_result=[]
        X = df.drop(['br'], axis='columns')
        Y = df.br

        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

        model = SVC()
        model.fit(X_train, y_train)
        patch_result.extend((model.score(X_train, y_train),model.score(X_test, y_test)))
        p='Patch {} '.format(i) + "part_000"
        result.append(patch_result)
        header.append(p)
        i+=1
    header_col=["Training_Accuracy","Testing_Accuracy"]
    df_result=pd.DataFrame(result, header,columns =header_col)

    return df_result