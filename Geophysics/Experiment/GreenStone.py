# %%
import sys
sys.path.append('../../GP_old/')

import numpy as np
import gempy as gp
from gempy.core.tensor.modeltf import ModelTF
from gempy.assets.geophysics import *
import tensorflow as tf
import matplotlib.pyplot as plt

# %%
data_path = 'https://raw.githubusercontent.com/cgre-aachen/gempy_data/master/'
path_o = "/data/input_data/tut_SandStone/SandStone_Foliations.csv"
path_i = "/data/input_data/tut_SandStone/SandStone_Points.csv" 
extent = [696000, 747000, 6863000, 6930000, -20000, 600]
geo_data = gp.create_data( extent=extent, resolution=[50,50,30],
                          path_o=data_path + path_o,
                          path_i=data_path + path_i)



gp.map_series_to_surfaces(
            geo_data,
            {"EarlyGranite_Series": 'EarlyGranite',
                                     "BIF_Series": ('SimpleMafic2', 'SimpleBIF'),
                                     "SimpleMafic_Series": 'SimpleMafic1', 'Basement': 'basement'})

# define the density in g/cm^3
geo_data.add_surface_values([2.31, 2.92, 3.1, 2.92, 2.61])


model = ModelTF(geo_data)
gpinput = model.get_graph_input()
model.create_tensorflow_graph(gpinput,slope = 60000000,gradient = True)
# %%
model.compute_model()

# %%
gp.plot.plot_section(model, cell_number=42,
                         direction='x', show_data=True,
                         show_boundaries = True,
                         show_all_data = False
                         )

# %%

gp._plot.plot_3d(model)




# # define the receiver positions
# grav_res = 5
# X_r = np.linspace(704000,740000,grav_res)
# Y_r = np.linspace(6.87e6,6.92e6,grav_res)
# # X_r = [500]
# # Y_r = [500]

# r = []
# for x in X_r:
#     for y in Y_r:
#         r.append(np.array([x,y]))

# Z_r = 200 # at the top surface
# xyz = np.meshgrid(X_r, Y_r, Z_r)
# xy_ravel = np.vstack(list(map(np.ravel, xyz))).T

# radius = [5000,5000,10200]
# receivers = Receivers(radius,extent,xy_ravel,kernel_resolution = [25,25,25])

# # %%
# ## Convolutional regular grid
# model.activate_regular_grid()
# gpinput = model.get_graph_input()
# model.create_tensorflow_graph(gpinput,slope = 60000000.,gradient=False,compute_gravity=True)
# g = GravityPreprocessingRegAllLoop(receivers,model.geo_data.grid.regular_grid)
# tz = g.set_tz_kernel()

# tz = tf.constant(tz,model.tfdtype)
# # grav = model.compute_gravity(tz,g = g,receivers = receivers,method = 'conv_all')
# final_block,final_property,block_matrix,block_mask,size,scalar_field,sfai,grav = model.compute_gravity(tz,g = g,receivers = receivers,method = 'conv_all',grav_only = False)
# size = tf.reduce_prod(model.geo_data.grid.regular_grid.resolution)
# block = final_property[:size]

# sol = final_block,final_property,block_matrix,block_mask,size,scalar_field,sfai
# model.set_solutions(sol)

# model.solutions.values_matrix = final_property[:size].numpy()

# # %%
# ## plot the lithology block/density block
# gp.plot.plot_section(model, 23,block = model.solutions.values_matrix,
#                      show_data=True,direction="x",show_legend = False,
#                      cmap='viridis', show_grid=True, norm=None,colorbar = True,show_boundaries = True)
    

# %%
