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

geo_data.get_data()

gp.map_series_to_surfaces(
            geo_data,
            {"EarlyGranite_Series": 'EarlyGranite',
                                     "BIF_Series": ('SimpleMafic2', 'SimpleBIF'),
                                     "SimpleMafic_Series": 'SimpleMafic1', 'Basement': 'basement'})

# define the density in g/cm^3
geo_data.add_surface_values([2.61, 2.92, 3.1, 2.92, 2.61])
# %%
model = ModelTF(geo_data)

## COMPILE AND COMPUTE MODEL
model.compute_model()
# %%
p = gp.plot.plot_section(model, 21,
                      show_data=True,direction="x",
                     cmap='viridis', show_grid=True, norm=None,colorbar = True)


## 3D PLOT IN PYVISTA
# gp._plot.plot_3d(model)
# %%
## CONFIGURE RECEIVER LOCATIONS
grav_res = 10
X_r = np.linspace(704000,740000,grav_res)
Y_r = np.linspace(6.87e6,6.92e6,grav_res)
# X_r = [716000] 
# Y_r = [6881111.1111111]
r = []
for x in X_r:
  for y in Y_r:
    r.append(np.array([x,y]))
receivers = np.array(r)
Z_r = 200
n_devices = receivers.shape[0]
xyz = np.meshgrid(X_r, Y_r, Z_r)
xy_ravel = np.vstack(list(map(np.ravel, xyz))).T

radius = [5000,5000,20200]
kernel_resolution = [25,25,25]
receivers = Receivers(radius,extent,xy_ravel,kernel_resolution = kernel_resolution,grav_res = grav_res)

# %%
## COMPUTE GRAVITY
## Different gravity schemes are implemented. The kernel preparation is separated out to exclude it out of the computational graph.

## Convolute Over All Regular Grid
model.activate_regular_grid()
gpinput = model.get_graph_input()
model.create_tensorflow_graph(gpinput,gradient=True,compute_gravity=True)
g = GravityPreprocessingRegAllLoop(receivers,model.geo_data.grid.regular_grid)
tz = g.set_tz_kernel()
tz = tf.constant(tz,model.tfdtype)
print(tz)
grav = model.compute_gravity(tz,g = g,receivers = receivers,method = 'conv_all',grav_only = True)

## Kernel Regular Grid
# from gempy.core.grid_modules.grid_types import CenteredRegGrid
# centerReg_kernel = CenteredRegGrid(receivers.xy_ravel,radius=receivers.model_radius,resolution=[25,25,25])
# model.activate_customized_grid(centerReg_kernel)
# gpinput = model.get_graph_input()
# model.create_tensorflow_graph(gpinput,slope = 100000,gradient=True,compute_gravity=True)
# g_center_regulargrid = GravityPreprocessing(centerReg_kernel)
# tz_center_regulargrid = tf.constant(g_center_regulargrid.set_tz_kernel(),model.tfdtype)
# tz = tf.constant(tz_center_regulargrid,model.tfdtype)
# grav = model.compute_gravity(tz,kernel = centerReg_kernel,receivers = receivers,method = 'kernel_reg')

# %%
import matplotlib.pyplot as plt
grav_np = grav.numpy().reshape(grav_res,grav_res)

# %%
f,ax = gp.plot_grav(model,receivers,grav_np,diff =False,figsize = (14,5))
plt.show()
