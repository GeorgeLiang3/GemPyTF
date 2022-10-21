# %%
import sys
sys.path.append('../../GP_old/')

import matplotlib.pyplot as plt

import gempy as gp
from gempy.core.tensor.modeltf_var import ModelTF

# %%
data_path = 'https://raw.githubusercontent.com/cgre-aachen/gempy_data/master/'
path_to_data = data_path + "/data/input_data/jan_models/"

geo_data = gp.create_data( extent=[0, 1000, 0, 1000, 0, 1000], resolution=[50, 50, 50],
                          path_o=path_to_data + "model5_orientations.csv",
                          path_i=path_to_data + "model5_surface_points.csv")

# %%
geo_data.get_data()
# %%
geo_data.add_surface_values([2.0,4.0,0.0,3.0])
gp.map_series_to_surfaces(geo_data, {"Fault_Series": 'fault',
                                    "Strat_Series": ('rock2', 'rock1')})
geo_data.set_is_fault(['Fault_Series'])
geo_data.modify_order_surfaces(2,0) # Switch order_surfaces. Set the surface['id'] = 3 --> 0 idx


# # switch position of rock1 and rock2
# geo_data.modify_order_surfaces(new_value=2, idx=0)

# %%
## Construct computational graph in Tensorflow
model = ModelTF(geo_data)
model.activate_regular_grid()
gpinput = model.get_graph_input()
model.create_tensorflow_graph(gpinput,gradient = True,slope = 50,compute_gravity = True, max_slope = 100000)

# %%
model.compute_model()
# %%
gp._plot.plot_3d(model)
# %%
p0 = gp.plot.plot_section(model, cell_number=15,
                         direction='y', show_data=True)
p0.fig.savefig('/Volumes/GoogleDrive/My Drive/ZJ/thirdpaper/model5.png', dpi = 400)

# %%
p1 = gp.plot.plot_section(model, 15,block = model.solutions.values_matrix,
                     show_data=True,direction="y",show_legend = False,
                     cmap='viridis', show_grid=True, norm=None,colorbar = True,show_boundaries = False,show_faults = False,cbar_title = 'Density')

p1.fig.savefig('/Volumes/GoogleDrive/My Drive/ZJ/thirdpaper/model5_lith.png', dpi = 400)
# %%

model_smooth = ModelTF(geo_data)
model_smooth.activate_regular_grid()
gpinput = model_smooth.get_graph_input()
model_smooth.create_tensorflow_graph(gpinput,gradient = True,slope = 50,compute_gravity = True, max_slope = 100)
# %%
model_smooth.compute_model()

# %%
p2 = gp.plot.plot_section(model_smooth, 15,block = model_smooth.solutions.values_matrix,
                     show_data=True,direction="y",show_legend = False,
                     cmap='viridis', show_grid=True, norm=None,colorbar = True,show_boundaries = False,show_faults = False,
                     cbar_title = 'Density')


p2.fig.savefig('/Volumes/GoogleDrive/My Drive/ZJ/thirdpaper/model5_lith_smooth.png', dpi = 400)                 


# %%
###############SCALAR FIELD##############
gp.plot.plot_scalar_field(model_smooth, 15,
                     show_data=True,direction="y", norm=None,colorbar = True,series = 0)
ax = plt.gca()
ax.set_aspect('equal')
fig = plt.gcf()
fig.savefig('/Volumes/GoogleDrive/My Drive/ZJ/thirdpaper/model5_scalar0.png', dpi = 400)                 

# %%
gp.plot.plot_scalar_field(model_smooth, 15,
                     show_data=True,direction="y", norm=None,colorbar = True, series = 1)
ax = plt.gca()
ax.set_aspect('equal')
fig = plt.gcf()
fig.savefig('/Volumes/GoogleDrive/My Drive/ZJ/thirdpaper/model5_scalar1_smooth.png', dpi = 400)    

# %%
gp.plot.plot_scalar_field(model, 15,
                     show_data=True,direction="y", norm=None,colorbar = True, series = 1)
ax = plt.gca()
ax.set_aspect('equal')
fig = plt.gcf()
fig.savefig('/Volumes/GoogleDrive/My Drive/ZJ/thirdpaper/model5_scalar1.png', dpi = 400)     
# %%
###############MASK MATRIX##############

p = gp.plot.plot_section(model, 15,block = model.solutions.block_matrix[0],
                     show_data=True,direction="y",show_legend = False, show_grid=True, norm=None,cmap='Greys',colorbar = True,show_boundaries = False,show_faults = False)


p.fig.savefig('/Volumes/GoogleDrive/My Drive/ZJ/thirdpaper/model5_mask.png', dpi = 
400)
# %%
p = gp.plot.plot_section(model_smooth, 15,block = model_smooth.solutions.block_matrix[0],
                     show_data=True,direction="y",show_legend = False,
                     cmap='Greys', show_grid=True, norm=None,colorbar = True,show_boundaries = False,show_faults = False)


p.fig.savefig('/Volumes/GoogleDrive/My Drive/ZJ/thirdpaper/model5_mask_smooth.png', dpi = 400)                 
# %%
