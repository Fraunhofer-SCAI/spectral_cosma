import os
import matplotlib
import numpy as np

def mkdirs(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=-1):
    if n == -1:
        n = cmap.N
    new_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
         'trunc({name},{a:.2f},{b:.2f})'.format(name=cmap.name, a=minval, b=maxval),
         cmap(np.linspace(minval, maxval, n)))
    return new_cmap

### from mesh_tools
def clean_mesh_get_edges(VV, FF, clean_mesh=True):
    # given list of Vertices and Faces, clean face duplicates, get list of edges and boundary/non-manifold information 
    # input:
    # - VV: list of vertices
    # - FF: list of triplets defining faces
    # output:
    # - VV: filterted vertices (delete vertices without face)
    # - FF: faces
    # - EE: edges
    # - boundary_edges, boundary_vertices, nonmanifold_edges
    
    FF_tmp = np.sort(FF, axis=1)
    if clean_mesh:
        FF_tmp, unique_faces_locs = np.unique(FF_tmp, axis=0, return_index=True)
    
    # filter vertices: this is dangerous. dont do it
    
    # edges
    edge_1=FF_tmp[:,0:2]
    edge_2=FF_tmp[:,1:]
    edge_3=np.concatenate([FF_tmp[:,:1], FF_tmp[:,-1:]], axis=1)
    EE=np.concatenate([edge_1, edge_2, edge_3], axis=0)
    
    # delete duplicates
    unique_edges_trans, unique_edges_locs, edges_counts=np.unique(EE[:,0]*(10**5)+EE[:,1], return_index=True, return_counts=True)
    EE=EE[unique_edges_locs,:]
    
    boundary_edges = np.where(edges_counts==1)[0]
    boundary_vertices = np.unique( EE[boundary_edges])
    nonmanifold_edges = np.where(edges_counts>2)[0]
    if clean_mesh:
        return VV, FF[unique_faces_locs], EE, boundary_edges, boundary_vertices, nonmanifold_edges
    else:
        return EE, boundary_edges, boundary_vertices, nonmanifold_edges
    
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')