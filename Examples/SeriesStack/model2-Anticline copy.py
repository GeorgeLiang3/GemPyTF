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
geo_data = gp.create_data( extent=[-500, 1500, -500, 1500, 0, 1000], resolution=[50, 50, 50],
                          path_o=path_to_data + "model2_orientations.csv",
                          path_i=path_to_data + "model2_surface_points.csv")


gp.map_series_to_surfaces(geo_data, {"Strat_Series": ('rock2', 'rock1'), "Basement_Series": ('basement')})

geo_data.add_surface_values([1.0,4.0,2.0])
geo_data.modify_order_surfaces(2,0)

# %%
## modify one surface point

# geo_data.surface_points.modify_surface_points(21,Z = 600,surface = 'rock2')
# %%
## Construct computational graph in Tensorflow
model = ModelTF(geo_data)
model.activate_regular_grid()
gpinput = model.get_graph_input()
model.create_tensorflow_graph(gpinput,gradient = False,compute_gravity = True)


# %%
model.compute_model()

# %%

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

from gempy.plot.vista import GemPyToVista
import pyvista as pv

# Initialize receivers
poly = pv.PolyData(xy_ravel)
geom = pv.Cone(direction=[0.0, 0.0, -1.0])
glyphs = poly.glyph(factor=50.0,geom=geom)

gpv = GemPyToVista(model)
gpv.plot_surface_points(surfaces='all')
gpv.p.add_mesh(glyphs, color="FFCC99",render_points_as_spheres=True,point_size=1000) # add receivers to mesh
gpv.plot_structured_grid('lith',render_topography=False)
gpv.plot_surface_points(point_size = 20)
gpv.plot_orientations()
gpv.plot_surfaces()

gpv.p.add_bounding_box()

gpv.p.show()
# %%
# # 3D plot

gp._plot.plot_3d(model,show_lith = True)

# %%
# gp.plot.plot_data(model, cell_number=4,
#                          direction='y', show_data=True)
# p0.fig.savefig('/Volumes/GoogleDrive/My Drive/ZJ/thirdpaper/model2_data.png', dpi = 400)
# %%
p0 = gp.plot.plot_section(model, cell_number=4,
                         direction='y', show_data=True)
p0.fig.savefig('/Volumes/GoogleDrive/My Drive/ZJ/thirdpaper/model2.png', dpi = 400)
# %%
p1 = gp.plot.plot_section(model, 15, block = model.solutions.values_matrix,
                     show_data=True,direction="y",show_legend = False,
                     cmap='viridis', show_grid=True, norm=None,colorbar = True,show_boundaries = False,cbar_title = 'Density')

# p1.fig.savefig('/Volumes/GoogleDrive/My Drive/ZJ/thirdpaper/model2_lith.png', dpi = 400)

# %%
############ ADD SMOOTH FUNCTION ###########
model_smooth = ModelTF(geo_data)
gpinput = model_smooth.get_graph_input()
model_smooth.create_tensorflow_graph(gpinput,gradient = False,slope = 70,compute_gravity = True)

model_smooth.compute_model()
# %%
###############SCALAR FIELD##############
        
gp.plot.plot_scalar_field(model, 15,
                     show_data=True,direction="y", norm=None,colorbar = True, series =0, title = 'Scalar value')
ax = plt.gca()
ax.set_aspect('equal')
fig = plt.gcf()
fig.savefig('/Volumes/GoogleDrive/My Drive/ZJ/thirdpaper/model2_scalar1.png', dpi = 400)     

# %%
# ###############MASK MATRIX##############

# p = gp.plot.plot_section(model, 15,block = model.solutions.block_matrix[0],
#                      show_data=True,direction="y",show_legend = False, show_grid=True, norm=None,cmap='Greys',colorbar = True,show_boundaries = False,show_faults = False)


# p.fig.savefig('/Volumes/GoogleDrive/My Drive/ZJ/thirdpaper/model2_mask.png', dpi = 
# 400)
# # %%
# p = gp.plot.plot_section(model_smooth, 15,block = model_smooth.solutions.block_matrix[0],
#                      show_data=True,direction="y",show_legend = False,
#                      cmap='Greys', show_grid=True, norm=None,colorbar = True,show_boundaries = False,show_faults = False)


# p.fig.savefig('/Volumes/GoogleDrive/My Drive/ZJ/thirdpaper/model2_mask_smooth.png', dpi = 400)                 


# # %%

# grav_res = 5
# # X_r = np.linspace(704000,740000,grav_res)
# # Y_r = np.linspace(6.87e6,6.92e6,grav_res)
# X_r = [500]
# Y_r = [500]

# r = []
# for x in X_r:
#     for y in Y_r:
#         r.append(np.array([x,y]))

# Z_r = 1000 # at the top surface
# xyz = np.meshgrid(X_r, Y_r, Z_r)
# xy_ravel = np.vstack(list(map(np.ravel, xyz))).T

# radius = [1000,1000,1000]
# receivers = Receivers(radius,[0, 1000, 0, 1000, 0, 1000],xy_ravel,kernel_resolution = [25,25,25])

# Reg_kernel = CenteredRegGrid(receivers.xy_ravel,radius=receivers.model_radius,resolution=receivers.kernel_resolution)
# model.activate_customized_grid(Reg_kernel)
# gpinput = model.get_graph_input()
# model.create_tensorflow_graph(gpinput,slope = 5000000.,gradient=False,compute_gravity=True)
# g_center_regulargrid = GravityPreprocessing(Reg_kernel)
# tz_center_regulargrid = tf.constant(g_center_regulargrid.set_tz_kernel(),model.tfdtype)
# tz = tf.constant(tz_center_regulargrid,model.tfdtype)

# final_block,final_property,block_matrix,block_mask,size,scalar_field,sfai,grav_hd = model.compute_gravity(tz,kernel = Reg_kernel,receivers = receivers,method = 'kernel_reg')

# plt.plot(final_property[:25])
# # %%

# model_smooth = ModelTF(geo_data)
# gpinput = model_smooth.get_graph_input()
# model_smooth.create_tensorflow_graph(gpinput,slope = 70,compute_gravity = True)

# model_smooth.compute_model()

# # %%
# p2 = gp.plot.plot_section(model_smooth, 15,block = model_smooth.solutions.values_matrix,
#                      show_data=True,direction="y",show_legend = False,
#                      cmap='viridis', show_grid=True, norm=None,colorbar = True,show_boundaries = False)


# p2.fig.savefig('/Volumes/GoogleDrive/My Drive/ZJ/thirdpaper/model2_lith_smooth.png', dpi = 400)                 

# # %%
# gp.plot.plot_scalar_field(model_smooth, 15,
#                      show_data=True,direction="y", norm=None,colorbar = True)
# ax = plt.gca()
# ax.set_aspect('equal')
# fig = plt.gcf()
# fig.savefig('/Volumes/GoogleDrive/My Drive/ZJ/thirdpaper/model2_scalar.png', dpi = 400)                 

# # %%

# %%
