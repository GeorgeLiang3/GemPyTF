# %%
import sys
sys.path.append('../../GP_old/')

import gempy as gp
import matplotlib.pyplot as plt
from gempy.core.tensor.modeltf_var import ModelTF
from gempy.assets.geophysics import *
from gempy.core.grid_modules.grid_types import CenteredRegGrid
import tensorflow as tf
import numpy as np
import pandas as pn

regular_grid_resolution = [50,50,50]

## Initialize model by loading an empty model
name = 'Model0b'
path = '/Volumes/GoogleDrive/My Drive/GemPhy/Geophysics/Data/'
geo_data = gp.custom_load_model(name=name,path=path,resolution = regular_grid_resolution)

model_extend = np.load(path+name+'/Model0b_extent.npy')
W = model_extend[0]
E = model_extend[1]
S = model_extend[2]
N = model_extend[3]
bot = model_extend[4]
top = model_extend[5]+100

## Load input data
additional_orientation_path_list = ['orientations_KevitsaA.csv','orientations_KevitsaB.csv']
additional_surface_path_list = ['surface_points_KevitsaA.csv','surface_points_KevitsaB.csv']

additional_orientation_path_OVB_list = ['orientations_Kevitsa_OVB_A.csv','orientations_Kevitsa_OVB_B.csv']
additional_surface_path_OVB_list = ['surface_points_Kevitsa_OVB_A.csv','surface_points_Kevitsa_OVB_B.csv']

def add_additional_orientation_data(data_path,surface):
    additional_orientation_data = pn.read_csv(path+data_path,header=None, index_col=0)
    for i in range(len(additional_orientation_data)):
        x = additional_orientation_data.loc[i][1]
        y = additional_orientation_data.loc[i][2]
        z = additional_orientation_data.loc[i][3]
        Gx = additional_orientation_data.loc[i][4]
        Gy = additional_orientation_data.loc[i][5]
        Gz = additional_orientation_data.loc[i][6]
        geo_data.add_orientations(X=x, Y=y, Z=z, surface=surface, pole_vector=(Gx, Gy, Gz))
    
def add_additional_surface_data(data_path,surface): 
    additional_orientation_data = pn.read_csv(path+data_path,header=None, index_col=0)
    for i in range(len(additional_orientation_data)):
        x = additional_orientation_data.loc[i][1]
        y = additional_orientation_data.loc[i][2]
        z = additional_orientation_data.loc[i][3]
        geo_data.add_surface_points(X=x, Y=y, Z=z, surface=surface)

for data_path in additional_orientation_path_OVB_list:
    add_additional_orientation_data(data_path,'OVB')
for data_path in additional_surface_path_OVB_list:
    add_additional_surface_data(data_path,'OVB')
    
for data_path in additional_orientation_path_list:
    add_additional_orientation_data(data_path,'UPX')
for data_path in additional_surface_path_list:
    add_additional_surface_data(data_path,'UPX')

# trick to remove the first surface points, so we can use the load module function, don't need to define the range and so on
geo_data.surface_points.del_surface_points(1)

# self.extent = [0, 1000, 0, 1000, 0, 1000]
# self.top = self.extent[-1]
# self.geo_data = create_data(self.extent, resolution=regular_grid_resolution,
#                 path_o=master_path+ orientations_path,
#                 path_i=master_path + surface_path)
# map_series_to_surfaces(self.geo_data, {"Strat_Series": (
#     'rock2', 'rock1'), "Basement_Series": ('basement')})

# define density
geo_data.add_surface_values([2.6, 3.1, 2.7])


section_dict = {"E5": ([3496862, 7509181], [3499534, 7513000], [200, 200]),
                "A": ([3494800, 7508900],[3501000, 7515000], [200, 200]),
                # "B": ([3500900, 7509000],[3494700, 7514000], [50, 50])
                "B": ([3494700, 7514000],[3500900, 7509000], [200, 200])
                }
# section_dict = {"B": ([3502850, 7050690],[3494000, 7515000], [50, 50])}
geo_data.set_section_grid(section_dict)

# %%
## Construct computational graph in Tensorflow
model = ModelTF(geo_data)
model.activate_regular_grid()
gpinput = model.get_graph_input()
model.create_tensorflow_graph(gpinput,slope = 300000, gradient = False,compute_gravity = True)
model.compute_model()
# %%
gp._plot.plot_3d(model,show_lith = True)
# %%
