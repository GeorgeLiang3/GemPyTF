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
data_path = '/Volumes/GoogleDrive/My Drive/GemPhy/GP_old/notebooks'
# path_to_data = data_path + "/data/input_data/jan_models/"
path_to_data = data_path + "/data/input_data/George_models/"
geo_data = gp.create_data( extent=[0, 1000, 0, 1000, 0, 1000], resolution=[20, 20, 40],
                          path_o=path_to_data + "model7_multilayer_orientation.csv",
                          path_i=path_to_data + "model7_multilayer_surface_points.csv")

# %%
gp.map_series_to_surfaces(geo_data, {"Strat_Series": ('rock3', 'rock2', 'rock1'), "Basement_Series": ('basement')})

geo_data.modify_order_surfaces(3,0) # Switch order_surfaces. Set the surface['id'] = 3 --> 0 idx
geo_data.add_surface_values([20,10,-7,30])

# %%
## modify one surface point

# geo_data.surface_points.modify_surface_points(21,Z = 600,surface = 'rock2')
# %%
## Construct computational graph in Tensorflow
model = ModelTF(geo_data)
model.activate_regular_grid()
gpinput = model.get_graph_input()
model.create_tensorflow_graph(gpinput,slope = 300000, gradient = False,compute_gravity = True,min_slope = 400)
model.compute_model()

# %%
gp._plot.plot_3d(model,show_lith = True)
# %%
model1 = ModelTF(geo_data)
model1.activate_regular_grid()
gpinput = model1.get_graph_input()
model1.create_tensorflow_graph(gpinput,slope = 300, gradient = True,compute_gravity = True,min_slope = 400)
model1.compute_model()

# %%
gp._plot.plot_3d(model1,show_lith = True)
# %%
# # 3D plot
# model._grid.regular_grid.values = model.revert_coordinates_alongz(model._grid.regular_grid.values)

# %%
p0 = gp.plot.plot_section(model, cell_number=2,
                         direction='y', show_data=True)
# # p0.fig.savefig('/Volumes/GoogleDrive/My Drive/ZJ/thirdpaper/model2.png', dpi = 400)
# %%
p1 = gp.plot.plot_section(model, 0, block = model.solutions.values_matrix,
                     show_data=True,direction="y",show_legend = False,
                     cmap='viridis', show_grid=True, norm=None,colorbar = True,show_boundaries = False)

# p1.fig.savefig('/Volumes/GoogleDrive/My Drive/ZJ/thirdpaper/model2_lith.png', dpi = 400)
# %%

grav_res = 5
# X_r = np.linspace(704000,740000,grav_res)
# Y_r = np.linspace(6.87e6,6.92e6,grav_res)
X_r = [500]
Y_r = [500]

r = []
for x in X_r:
    for y in Y_r:
        r.append(np.array([x,y]))

Z_r = 1000 # at the top surface
xyz = np.meshgrid(X_r, Y_r, Z_r)
xy_ravel = np.vstack(list(map(np.ravel, xyz))).T

radius = [1000,1000,1000]
receivers = Receivers(radius,[0, 1000, 0, 1000, 0, 1000],xy_ravel,kernel_resolution = [25,25,25])

Reg_kernel = CenteredRegGrid(receivers.xy_ravel,radius=receivers.model_radius,resolution=receivers.kernel_resolution)
model.activate_customized_grid(Reg_kernel)
gpinput = model.get_graph_input()
model.create_tensorflow_graph(gpinput,slope = 5000000.,gradient=False,compute_gravity=True)
g_center_regulargrid = GravityPreprocessing(Reg_kernel)
tz_center_regulargrid = tf.constant(g_center_regulargrid.set_tz_kernel(),model.tfdtype)
tz = tf.constant(tz_center_regulargrid,model.tfdtype)

final_block,final_property,block_matrix,block_mask,size,scalar_field,sfai,grav_hd = model.compute_gravity(tz,kernel = Reg_kernel,receivers = receivers,method = 'kernel_reg')

plt.plot(final_property[:25])
# %%

model_smooth = ModelTF(geo_data)
gpinput = model_smooth.get_graph_input()
model_smooth.create_tensorflow_graph(gpinput,slope = 70,compute_gravity = True)

model_smooth.compute_model()

# %%
p2 = gp.plot.plot_section(model_smooth, 15,block = model_smooth.solutions.values_matrix,
                     show_data=True,direction="y",show_legend = False,
                     cmap='viridis', show_grid=True, norm=None,colorbar = True,show_boundaries = False)


p2.fig.savefig('/Volumes/GoogleDrive/My Drive/ZJ/thirdpaper/model2_lith_smooth.png', dpi = 400)                 

# %%
gp.plot.plot_scalar_field(model_smooth, 15,
                     show_data=True,direction="y", norm=None,colorbar = True)
ax = plt.gca()
ax.set_aspect('equal')
fig = plt.gcf()
fig.savefig('/Volumes/GoogleDrive/My Drive/ZJ/thirdpaper/model2_scalar.png', dpi = 400)                 

# %%
