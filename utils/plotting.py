import matplotlib.pyplot as plt
import numpy as np
import cosma
import torch
try:
    import plotly.figure_factory as ff
except:
    ff = None

import matplotlib as mpl
from matplotlib import cm
    
def get_mesh_predictions(model, patches, ids_hexagonal_patch, VV, refinement = 3, padding = False, center = False, mean_patches = None, device = 'cpu'):
    # prepare the patches for feeding them to the trained model by reindexing, centering and flattening them
    reindexed_patches = patches[:, cosma.PAD_INDEX_MAPPING_KS2[refinement] if padding else cosma.NONPAD_INDEX_MAPPING[refinement]]
    if center:
        # only center if mean is not given
        if mean_patches is None:
            
            mean_patches = np.mean(reindexed_patches, axis = 1)[:, None, :]
            reindexed_patches -= mean_patches
            
    model_input = torch.tensor(reindexed_patches.reshape(reindexed_patches.shape[0] * reindexed_patches.shape[1], -1), 
                               dtype = torch.float).to(device)

    # get the model predictions for the prepared input
    model_output_m = model(model_input).detach().cpu().numpy()
    # undo reindexing, centering and flattening
    model_output_m = model_output_m.reshape(reindexed_patches.shape)
    if center:
        model_output = model_output_m + mean_patches 
    else:
        model_output = model_output_m
    model_output = model_output[:, cosma.PAD_REVERSE_INDEX_MAPPING[refinement] if padding else cosma.NONPAD_REVERSE_INDEX_MAPPING[refinement]]
    # get the predictions for every unique node
    predicted_vv = np.zeros(VV.shape)
    index_counts = np.zeros(VV.shape[0])
    for select_patch in range(len(ids_hexagonal_patch)):
        predicted_vv[ids_hexagonal_patch[select_patch]] += model_output[select_patch, cosma.NO_PADDING_INDICES_ORIGINAL[refinement]] if padding else model_output[select_patch]
        index_counts[ids_hexagonal_patch[select_patch]] += 1
    predicted_vv[index_counts == 0] = VV[index_counts == 0]
    index_counts[index_counts == 0] = 1
    # get average prediction for nodes contained in multiple patches
    predicted_vv /= index_counts[None].T
    
    return predicted_vv, reindexed_patches, model_output_m

def get_mesh_embeddings(model, patches, ids_hexagonal_patch, VV, refinement = 3, padding = False, center = False, device = 'cpu'):
    # prepare the patches for feeding them to the trained model by reindexing, centering and flattening them
    reindexed_patches = patches[:, cosma.PAD_INDEX_MAPPING_KS2[refinement] if padding else cosma.NONPAD_INDEX_MAPPING[refinement]]
    if center:
        mean_patches = np.mean(reindexed_patches, axis = 1)[:, None, :]
        reindexed_patches -= mean_patches
    model_input = torch.tensor(reindexed_patches.reshape(reindexed_patches.shape[0] * reindexed_patches.shape[1], -1), 
                               dtype = torch.float).to(device)

    # get the model predictions for the prepared input   
    model_emb = model.encoder(model_input).detach().cpu().numpy()
    # undo reindexing, centering and flattening
    model_emb = model_emb.reshape(reindexed_patches.shape[0], -1)
    return model_emb


def plot_mesh(FF, VV, 
              fig = None, ax = None, 
              figsize = (8, 10), 
              x_lim = (-1, 1), y_lim = (-1, 1), z_lim = (-1, 1), 
              color = 'blue', alpha = 0.3, title = None, view = (180, 180)):
    if fig is None or ax is None:
        fig, ax  = plt.subplots(figsize = figsize, subplot_kw = dict(projection='3d'))
    ax.axis('off')
    ax.set_xlim(*x_lim); ax.set_ylim(*x_lim); ax.set_zlim(*z_lim);

    for ff in FF:
        vv = VV[np.append(ff,ff[0])]
        ax.plot(vv[:, 0], vv[:, 2], -vv[:, 1], color = color, alpha = alpha)
    
    ax.view_init(*view)
    ax.axis('tight')
    ax.set_title(title)
    return fig, ax


def plot_patch(patch_vv, adj_mat,
               fig = None, ax = None, 
               figsize = (8, 10), numbersize = None,
               x_lim = (-1, 1), y_lim = (-1, 1), z_lim = (-1, 1), 
               color = 'black', alpha = 1, title = None,
               red_ids = [], blue_ids = [], green_ids = [], view = (180, 180)):
    if fig is None or ax is None:
        fig, ax  = plt.subplots(figsize = figsize, subplot_kw = dict(projection='3d'))
    ax.axis('off')
    ax.set_xlim(*x_lim); ax.set_ylim(*x_lim); ax.set_zlim(*z_lim);
    
    for vi, vv in enumerate(patch_vv):
        
        if vi in red_ids:
            ax.scatter(patch_vv[vi,0],patch_vv[vi,2],-patch_vv[vi,1], color = 'orange', linewidth = 10)
        if vi in blue_ids:
            ax.scatter(patch_vv[vi,0],patch_vv[vi,2],-patch_vv[vi,1], color = 'orangered', linewidth = 10)
        if vi in green_ids:
            ax.scatter(patch_vv[vi,0],patch_vv[vi,2],-patch_vv[vi,1], color = 'purple', linewidth = 10)
        if numbersize is not None:
            ax.text(x = patch_vv[vi, 0], y = patch_vv[vi, 2], z = -patch_vv[vi, 1], s = '%s' % (str(vi)), size = numbersize)
        if vi < len(adj_mat):
            for nn in np.where(adj_mat[vi]==1)[0]:
                ax.plot(patch_vv[[vi, nn], 0], patch_vv[[vi, nn], 2], -patch_vv[[vi, nn], 1], color = color, alpha = alpha)


    ax.view_init(*view)
    ax.axis('tight')
    ax.set_title(title)
    return fig, ax



def get_face_dist_list(faces, preds, targets):
    errors = []
    for face in faces:
        errors.append(torch.mean(torch.sqrt(torch.sum((-targets[face] + preds[face])**2, axis = 1))))
    return errors

def get_face_diff_list(faces, preds, targets):
    errors = []
    for face in faces:
        errors.append(torch.mean(-targets[face] + preds[face]))
    return errors

def get_face_mae_list(faces, preds, targets):
    errors = []
    for face in faces:
        errors.append(torch.mean(torch.absolute(targets[face] - preds[face])))
    return errors

def get_face_mse_list(faces, preds, targets):
    errors = []
    for face in faces:
        errors.append(torch.mean((targets[face] - preds[face])**2))
    return errors


def visualize_mesh_matplotlib(pos, faces, color_val_list, lw = 0.02,
                              fig = None, ax = None, 
                              figsize = (8, 10), view = (0, 90),
                              limits = [(-1, 1), (-1, 1), (-1, 1)], vmin = None, vmax = None):
    xlim, ylim, zlim = limits
    if fig is None or ax is None:
        fig, ax  = plt.subplots(figsize = figsize, subplot_kw = dict(projection='3d'))
    ax.axis('off')
    ax.set_xlim(*xlim); ax.set_ylim(*xlim); ax.set_zlim(*zlim);
    if color_val_list is not None:
        collec = ax.plot_trisurf(pos[:, 0], pos[:, 1], pos[:, 2], 
                                triangles = faces, shade=False, cmap = plt.cm.plasma)
        collec.set_array(np.array(color_val_list))
        if vmin is not None and vmax is not None:
            collec.set_clim(vmin=vmin, vmax=vmax)
    else:
        collec = ax.plot_trisurf(pos[:, 0], pos[:, 1], pos[:, 2], triangles=faces, shade=False, color='white', linewidth=lw, edgecolor = 'darkblue' )
    ax.view_init(*view)
    
    return fig, ax

def visualize_mesh(pos, faces, color_val_list = None, title = None, limits = [(0.8, 2.0), (-1, -0.5), (-1, -0.5)], save_path = None, scale = 1.0, colorbar_title = 'MSE', fontsize = 20, scene_camera = dict(up=dict(x=0, y=0, z=1), center=dict(x=0, y=0, z=0), eye= dict(x=0.07, y=1, z=0.7))):
    fig = ff.create_trisurf(x = pos[:, 0], y = pos[:, 1], z = pos[:, 2], color_func = color_val_list,
                            simplices = faces, title = title, show_colorbar = True, colormap="Portland")
    xlim, ylim, zlim = limits
    fig.update_layout(scene = dict(xaxis = dict(showticklabels=False, showbackground = False, visible=False, range=[*xlim]),
                                   yaxis = dict(showticklabels=False, showbackground = False, visible=False, range=[*ylim]),
                                   zaxis = dict(showticklabels=False, showbackground = False, visible=False, range=[*zlim]),),
                      font = dict(size = fontsize),
                      scene_camera = scene_camera
                     )
    
    if colorbar_title is not None:
        fig['data'][2]['marker']['colorbar']= {"title": colorbar_title, 'tickformat':'.1e'}
    if save_path is not None:
        fig.write_image(save_path, scale = scale)
    else:
        fig.show()
    
    
def visualize_pred_errors(data, preds, pos = None, title = None, limits = [(0.8, 2.0), (-1, -0.5), (-1, -0.5)], save_path = None, scale = 1.0, metric = 'mse', fontsize = 15, scene_camera = dict(up=dict(x=0, y=0, z=1), center=dict(x=0, y=0, z=0), eye= dict(x=0.07, y=1, z=0.7))):
    """Visualize the predictions errors of a model on the surface mesh
        -----
        args:
        -----
            - data:  object; an instanace of PyTorch Geometrics data class
            - preds: tensor; the predictions of a model
            - pos:   tensor; the node positions of the mesh (if pos is None, the positions specfied in data.pos will be used)
            - title: string; the title of the plot
            - limits: list of tuples; the x-, y- and z- limits of the plot
            - save_path: string; the path for saving the plot
            - scale: float; scale the quality/size of the plot
            - metric: string; metric used for computing the errors between preds and data.y (has to be one of 'mse', 'mae' or 'diff')
    """
    if metric == 'mse':
        color_val_list = get_face_mse_list(data.face.T, preds, data.y)
        colorbar_title = 'MSE'
    if metric == 'mae':
        color_val_list = get_face_mae_list(data.face.T, preds, data.y)
        colorbar_title = 'MAE'
    if metric == 'diff':
        color_val_list = get_face_diff_list(data.face.T, preds, data.y)
        colorbar_title = 'Difference'
    if metric == 'dist':
        color_val_list = get_face_dist_list(data.face.T, preds, data.y)
        colorbar_title = 'Distance'
    visualize_mesh(pos = data.pos if pos is None else pos, 
                   faces = data.face.T, 
                   limits = limits, 
                   save_path = save_path,
                   color_val_list = color_val_list, 
                   title = title,
                   scale = scale,
                   colorbar_title = colorbar_title,
                   fontsize = fontsize,
                   scene_camera = scene_camera)