import numpy as np
import scipy
from scipy.signal import decimate
from scipy import interpolate
import h5py
import load, load_hnn
import pandas as pd
import icsd
import neo
import quantities as pq
import os
from os.path import isfile, join
from os import listdir
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import os
sns.set()
    

#Takes in a .param dictionary, parameter sweeps defined by an array of values, generates all combinations 
def dict_expand(array_dict):
    dict_list = []

    array_keys = [param_key for param_key in list(array_dict.keys()) if type(array_dict[param_key]) == np.ndarray]
    array_vals = [array_dict[param_key] for param_key in array_keys]
    param_combos = list(itertools.product(*array_vals))

    for combo in param_combos:
        temp_dict = array_dict.copy()
        for param_idx in range(len(array_keys)):
            temp_dict[array_keys[param_idx]] = combo[param_idx]

        dict_list.append(temp_dict)

    return dict_list

def csd_interp(data_path, ds_step):
    # calculate pixel (um) positions of model objects
    top_l5 = 1466
    soma_l5 = 177.5
    bottom_l5 = -72

    top_l2 = 1466
    soma_l2 = 1000
    bottom_l2 = 769

    num_contacts_hnn = 20
    hnn_values = [top_l2, soma_l2, bottom_l2, soma_l5, bottom_l5 ]
    perlayer = (top_l5 - bottom_l5)/(num_contacts_hnn - 5.0)
    spacing_um_hnn = perlayer

    # print("contact spacing (microns)",spacing_um_hnn)
    first_contact = top_l5+2*spacing_um_hnn
    last_contact = bottom_l5-2*spacing_um_hnn

    spacing_hnn = []
    for i,x in enumerate(np.linspace(first_contact,last_contact,num_contacts_hnn)):
        # print("contact %d: %.2f" % (i+1, x))
        spacing_hnn.append(x)


    #load data
    dir = data_path
    sampr_hnn,LFP_hnn,dt_hnn, tt_hnn, CSD_hnn, maxlfp_hnn, ntrial_hnn = load_hnn.loadHNNdir(dir,spacing_um_hnn)

    #Down sample fixed interval 
    # ds_step = 10
    tt_hnn = tt_hnn[::ds_step]

    # Average iCSD from HNN
    z_data_hnn = np.linspace(spacing_um_hnn*1E-6, 2300E-6, num_contacts_hnn) * pq.m  # [m]
    diam_hnn = 500E-6 * pq.m                              # [m]
    h_hnn = spacing_um_hnn * 1E-6 * pq.m                                 # [m]

    sigma_hnn = 0.3 * pq.S / pq.m                         # [S/m] or [1/(ohm*m)]
    sigma_top_hnn = 0.3 * pq.S / pq.m                     # [S/m] or [1/(ohm*m)]

    # Create dictionary with iCSD calculations
    icsd_hnn = {}
    for key in LFP_hnn:

        lfp_data_hnn = LFP_hnn[key] * 1E-6 * pq.V        # [uV] -> [V]
        lfp_data_hnn = lfp_data_hnn[:,::ds_step]

        # Input dictionaries for monkey data
        delta_input_hnn = {
            'lfp' : lfp_data_hnn,
            'coord_electrode' : z_data_hnn,
            'diam' : diam_hnn,          # source diameter
            'sigma' : sigma_hnn,        # extracellular conductivity
            'sigma_top' : sigma_hnn,    # conductivity on top of cortex
            'f_type' : 'gaussian',  # gaussian filter
            'f_order' : (3, 1),     # 3-point filter, sigma = 1.
        }

        delta_icsd_hnn = icsd.DeltaiCSD(**delta_input_hnn)
        icsd_hnn[key] = delta_icsd_hnn.get_csd()

    avgiCSD_hnn = load_hnn.getAvgERP(icsd_hnn, sampr_hnn, tt_hnn, maxlfp_hnn, ntrial_hnn)
    # avgiCSD_hnn = load_hnn.downsample(avgiCSD_hnn, sampr_hnn, 2000) #Down sample to LFP range

    #Prepare HNN data
    # set timerange from 30 to 80 ms
    # ewindowms_hnn = 35
    # tmin_hnn = 37
    # tmax_hnn = tmin_hnn + ewindowms_hnn
    # (idx_min_hnn, idx_max_hnn) = (np.where(tt_hnn==tmin_hnn)[0][0], np.where(tt_hnn==tmax_hnn)[0][0])
    # X_hnn = tt_hnn[idx_min_hnn:idx_max_hnn]
    X_hnn = tt_hnn

    # mask = np.zeros(len(tt_hnn), dtype=bool)
    # mask[idx_min_hnn:idx_max_hnn] = True
    # avgCSD_trim_X_hnn = avgiCSD_hnn[:,mask]
    avgCSD_trim_X_hnn = avgiCSD_hnn
    



    Y_hnn = range(avgiCSD_hnn.shape[0])
    # X_hnn = range(avgiCSD_hnn.shape[1])
    # CSD_spline_hnn=scipy.interpolate.RectBivariateSpline(Y_hnn, X_hnn, avgCSD_trim_X_hnn)
    CSD_spline_hnn=scipy.interpolate.RectBivariateSpline(Y_hnn, X_hnn, avgCSD_trim_X_hnn)


    # trim channels
    (idx_min_hnn, idx_max_hnn) = (0, 18)
    HNN_Y_plot = np.linspace(idx_min_hnn,idx_max_hnn,num=1000)
    # print("HNN channels: %d-%d"%(HNN_Y_plot.min(), HNN_Y_plot.max()))
    # mask = np.zeros(len(Y_hnn), dtype=bool)
    # mask[idx_min_hnn:idx_max_hnn] = True
    # avgCSD_trim_XY_hnn = avgCSD_trim_X_hnn[mask,:]

    # HNN_ymax = (spacing_hnn[int(HNN_Y_plot.min())]-spacing_hnn[int(HNN_Y_plot.max())])
    # HNN_ymin = 0

    Z_hnn = CSD_spline_hnn(HNN_Y_plot, X_hnn)

    # normalize to abs(Z.min)
    Z_hnn = Z_hnn/abs(Z_hnn.min())

    return Z_hnn


#Converts 2D array of to long array indexed by x,y,z coordinates
def grid2points(csd_grid):
    csd_points = [[r, c, csd_grid[r,c]] for r in range(csd_grid.shape[0]) for c in range(csd_grid.shape[1])]
    return csd_points

def points2grid(csd_points):
    x_max = int(max(csd_points[:,0])) 
    y_max = int(max(csd_points[:,1])) 

    csd_grid = np.zeros((x_max+1,y_max+1))
    for point in range(csd_points.shape[0]):
        c,r = int(csd_points[point,1]), int(csd_points[point,0])
        csd_grid[r,c] = csd_points[point,2]

    return csd_grid


#Creates directory structure used by TDA pipeline
def make_dir_structure(file_prefix, base_dir):
    # base_dir = 'D:/Jones_Lab/hnn_params/'
    # file_prefix = 'input_strength033120'

    file_dir = base_dir + file_prefix + '/'

    #Check if path exists to prevent overwriting
    if os.path.isdir(file_dir):
        print('Error: directory already exists')
    else:
        os.mkdir(file_dir)
        os.mkdir(file_dir + file_prefix +'_params/')
        os.mkdir(file_dir + file_prefix +'_data/')
        os.mkdir(file_dir + file_prefix +'_skeleton/')


    