# %%
import sys
sys.path.append('../GP_old/')

import numpy as np
import gempy as gp
from gempy.core.tensor.modeltf import ModelTF
from gempy.assets.geophysics import *

# %%
data_path = 'https://raw.githubusercontent.com/cgre-aachen/gempy_data/master/'
path_o = "/data/input_data/tut_SandStone/SandStone_Foliations.csv"
path_i = "/data/input_data/tut_SandStone/SandStone_Points.csv" 
extent = [696000, 747000, 6863000, 6930000, -20000, 600]
geo_data = gp.create_data( extent=extent, resolution=[150,150,30],
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

# model.compute_model()
# %%
# gp._plot.plot_3d(model)
# %%

grav_res = 10
X_r = np.linspace(704000,740000,grav_res)
Y_r = np.linspace(6.87e6,6.92e6,grav_res)
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
receivers = Receivers(radius,extent,xy_ravel,grav_res = grav_res)

# grav = model.compute_gravity(receivers,method = 'kernel_reg',window_resolution = [10, 10, 15])
grav = model.compute_gravity(receivers,method = 'conv_all')

# %%
import matplotlib.pyplot as plt
grav_np = grav.numpy().reshape(grav_res,grav_res)


# %%
f,ax = gp.plot_grav(model,receivers,grav_np,diff =False,figsize = (14,5))
plt.show()

