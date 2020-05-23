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

#Calculate similarity matrix for 2D wasserstein
p_dir = data_dir + '/' + prefix + '/' + 'points/'
file_list_csd = os.listdir(p_dir) 
similarity_matrix_wasserstein = reeb_matching.wasserstein_sim_matrix(file_list_csd, data_dir,prefix)

wasserstein_file_list_save = save_dir + '/file_list_wasserstein.csv'
wasserstein_sim_save = save_dir + '/similarity_matrix_wasserstein.csv'
np.savetxt('similarity_matrix_wasserstein.csv', similarity_matrix_wasserstein, delimiter=',')
np.savetxt(wasserstein_sim_save, similarity_matrix_wasserstein, delimiter=',')
np.savetxt(wasserstein_file_list_save, np.array(file_list_csd), delimiter=',',fmt="%s")



