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

# Flag for eager execution 
tf.config.run_functions_eagerly(True)


# %%
data_path = 'https://raw.githubusercontent.com/cgre-aachen/gempy_data/master/'
path_to_data = data_path + "/data/input_data/jan_models/"
model_extent=[-500, 1500, -500, 1500, 0, 1000]
geo_data = gp.create_data( extent=model_extent, resolution=[50, 50, 50],
                          path_o=path_to_data + "model2_orientations.csv",
                          path_i=path_to_data + "model2_surface_points.csv")


gp.map_series_to_surfaces(geo_data, {"Strat_Series": ('rock2', 'rock1'), "Basement_Series": ('basement')})

geo_data.modify_order_surfaces(2,0)

geo_data.add_surface_values([1.0,4.0,2.0])

# %%
# define the receiver positions
grav_res = 5
X_r = np.linspace(0,1000,grav_res)
Y_r = np.linspace(0,1000,grav_res)
# X_r = [500]
# Y_r = [500]

r = []
for x in X_r:
    for y in Y_r:
        r.append(np.array([x,y]))

Z_r = 1000 # at the top surface
xyz = np.meshgrid(X_r, Y_r, Z_r)
xy_ravel = np.vstack(list(map(np.ravel, xyz))).T

radius = [500,500,1000]
receivers = Receivers(radius,model_extent,xy_ravel,kernel_resolution = [50,50,50])

# %%
## INITIALIZE MODEL
model = ModelTF(geo_data)

## Convolutional regular grid
model.activate_regular_grid()
gpinput = model.get_graph_input()
model.create_tensorflow_graph(gpinput,slope = 0,gradient=True,compute_gravity=True,min_slope = tf.constant(110.,tf.float64))
g = GravityPreprocessingRegAllLoop(receivers,model.geo_data.grid.regular_grid)
tz = g.set_tz_kernel()

# %%
tz = tf.constant(tz,model.tfdtype)
final_block,final_property,block_matrix,block_mask,size,scalar_field,sfai,grav = model.compute_gravity(tz,g = g,receivers = receivers,method = 'conv_all',grav_only = False)


# %%
model_prior = ModelTF(geo_data)

Reg_kernel = CenteredRegGrid(receivers.xy_ravel,radius=receivers.model_radius,resolution=receivers.kernel_resolution)
model_prior.activate_customized_grid(Reg_kernel)
gpinput = model_prior.get_graph_input()

min_slope = 110.

model_prior.create_tensorflow_graph(gpinput,slope = 20.,gradient=True,compute_gravity=True,min_slope = min_slope)
g_center_regulargrid = GravityPreprocessing(Reg_kernel)
tz_center_regulargrid = tf.constant(g_center_regulargrid.set_tz_kernel(),model_prior.tfdtype)
tz = tf.constant(tz_center_regulargrid,model_prior.tfdtype)
final_block,final_property,block_matrix,block_mask,size,scalar_field,sfai,grav_hd = model_prior.compute_gravity(tz,kernel = Reg_kernel,receivers = receivers,method = 'kernel_reg',LOOP_FLAG = True)
# %%