# %%
import sys
sys.path.append('../../GP_old/')

import matplotlib.pyplot as plt

import gempy as gp
from gempy.core.tensor.modeltf_var import ModelTF

import tensorflow as tf
tf.config.run_functions_eagerly(True)

# %%
data_path = 'https://raw.githubusercontent.com/cgre-aachen/gempy_data/master/'
path_to_data = data_path + "/data/input_data/jan_models/"

geo_data = gp.create_data( extent=[0, 1000, 0, 1000, 0, 1000], resolution=[25, 25, 25],
                          path_o=path_to_data + "model6_orientations.csv",
                          path_i=path_to_data + "model6_surface_points.csv")

# %%

## 

gp.map_series_to_surfaces(geo_data, {"Strat_Series1": ('rock3'),
                                    "Strat_Series2": ('rock2', 'rock1'),
                                    "Basement_Series": ('basement')})

geo_data.add_surface_values([1.0,3.0,0.0,6.0], ['density'])
# %%
# # switch position of rock1 and rock2
# geo_data.modify_order_surfaces(new_value=2, idx=0)

# %%
## I will integrate the module into GemPy through Interpolator later
model = ModelTF(geo_data)
gpinput = model.get_graph_input()
model.create_tensorflow_graph(gpinput,gradient = False,compute_gravity = True)

# %%
model.compute_model()

# %%
# gp._plot.plot_3d(model)

# %%
p0 = gp.plot.plot_section(model, cell_number=15,
                         direction='y', show_data=True,show_legend = True)
# p0.fig.savefig('/Volumes/GoogleDrive/My Drive/ZJ/thirdpaper/model6.png', dpi = 400)

# %%
p1 = gp.plot.plot_section(model, 15,block = model.solutions.values_matrix,
                     show_data=True,direction="y",show_legend = False,
                     cmap='viridis', show_grid=True, norm=None,colorbar = True,show_boundaries = False)

# p1.fig.savefig('/Volumes/GoogleDrive/My Drive/ZJ/thirdpaper/model6_lith.png', dpi = 400)
# %%

# model_smooth = ModelTF(geo_data)
# gpinput = model_smooth.get_graph_input()
# model_smooth.create_tensorflow_graph(gpinput,gradient = False,slope = 70,compute_gravity = True)

# model_smooth.compute_model()

# # %%
# p2 = gp.plot.plot_section(model_smooth, 15,block = model_smooth.solutions.values_matrix,
#                      show_data=True,direction="y",show_legend = False,
#                      cmap='viridis', show_grid=True, norm=None,colorbar = True,show_boundaries = False)


# p2.fig.savefig('/Volumes/GoogleDrive/My Drive/ZJ/thirdpaper/model6_lith_smooth.png', dpi = 400)                 


# # %%
# ###############SCALAR FIELD##############
# gp.plot.plot_scalar_field(model_smooth, 15,
#                      show_data=True,direction="y", norm=None,colorbar = True,series = 0)
# ax = plt.gca()
# ax.set_aspect('equal')
# fig = plt.gcf()
# fig.savefig('/Volumes/GoogleDrive/My Drive/ZJ/thirdpaper/model6_scalar0.png', dpi = 400)                 

# # %%
# gp.plot.plot_scalar_field(model_smooth, 15,
#                      show_data=True,direction="y", norm=None,colorbar = True, series = 1)
# ax = plt.gca()
# ax.set_aspect('equal')
# fig = plt.gcf()
# fig.savefig('/Volumes/GoogleDrive/My Drive/ZJ/thirdpaper/model6_scalar1.png', dpi = 400)     
# # %%
# ###############MASK MATRIX##############

# p = gp.plot.plot_section(model, 15,block = model.solutions.block_matrix[0],
#                      show_data=True,direction="y",show_legend = False, show_grid=True, norm=None,cmap='Greys',colorbar = True,show_boundaries = False,show_faults = False)


# p.fig.savefig('/Volumes/GoogleDrive/My Drive/ZJ/thirdpaper/model6_mask.png', dpi = 
# 400)
# # %%
# p = gp.plot.plot_section(model_smooth, 15,block = model_smooth.solutions.block_matrix[0],
#                      show_data=True,direction="y",show_legend = False,
#                      cmap='Greys', show_grid=True, norm=None,colorbar = True,show_boundaries = False,show_faults = False)


# p.fig.savefig('/Volumes/GoogleDrive/My Drive/ZJ/thirdpaper/model6_mask_smooth.png', dpi = 400)                 
# # %%


# %%
