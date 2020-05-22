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
import networkx as nx
import reeb_matching
from copy import deepcopy
sns.set()

#Files to operate on
prefix = 'gbarEvPyrAmpa_sweep'

#Setup file paths
data_dir = os.path.abspath('lfp_reeb_github/data')

if not os.path.isdir('data/'+ prefix + '/similarity_matrices'):
    os.mkdir('data/'+ prefix + '/similarity_matrices')

save_dir = os.path.abspath('data/' + prefix + '/similarity_matrices')

#Calculate similarity matrix for reeb_matching
s_dir = data_dir + '/' + prefix + '/' + 'skeleton/'
file_list_tree = reeb_matching.get_skeleton_names(s_dir) 
resolution_list = [16,8,4,2]
similarity_matrix_tree, MPAIR_list = reeb_matching.tree_sim_matrix(file_list_tree, resolution_list, data_dir, prefix)

tree_sim_save = save_dir + '/similarity_matrix_reeb.csv'
tree_file_list_save = save_dir + '/file_list_reeb.txt'
np.savetxt(tree_sim_save, similarity_matrix_tree, delimiter=',')
np.savetxt(tree_file_list_save, np.array(file_list_tree), delimiter=',',fmt="%s")