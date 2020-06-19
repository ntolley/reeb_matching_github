import numpy as np
import os
from os.path import isfile, join
from os import listdir
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import csd_functions
import scipy
import reeb_matching
from copy import deepcopy
import scipy.spatial.distance as ssd
import scipy.cluster.hierarchy as hcluster
import paramrw 
sns.set()

#Takes in a list of .param files, converts to single array dict
def make_param_dict(param_dir):
    dict_list = os.listdir(param_dir)

    #Store keys that change over files
    all_array_keys = []

    #Initialize with first file in list, iterate over all files 
    _ , dict_array = paramrw.read(param_dir + '/' + dict_list[0])

    for f in dict_list[1:]:
        _ , p = paramrw.read(param_dir + '/' + f)
        for key in p.keys():
            #Look for parameters that change across files, turn into list
            if dict_array[key] != p[key]:
                all_array_keys.append(key)

    #Reduce to unique elements
    array_keys = list(np.unique(all_array_keys))

    #Append array_key values for every file
    for f in dict_list[1:]:
        _ , p = paramrw.read(param_dir + '/' + f)
        for key in array_keys:
            #Look for parameters that change across files, turn into list
            if type(dict_array[key]) == list:
                dict_array[key].append(p[key])
            else:
                dict_array[key] = [dict_array[key], p[key]]

    return dict_array, array_keys


#Takes param_dict in array for and returns file features in matrix from
def dict_to_matrix(param_dict):
    array_keys = []
    feature_lists = []
  
    for key in param_dict.keys():
        if type(param_dict[key]) == list and key != 'sim_prefix' and key != 'expmt_groups':
            array_keys.append(key)
            feature_lists.append(param_dict[key])

    feature_matrix = np.zeros((len(feature_lists[0]), len(feature_lists)))

    for col, feature in enumerate(feature_lists):
        feature_matrix[:,col] = feature

    return feature_matrix





def plot_clusters():
    return

def dendrogram_cluster():
    return