# %%
import sys
sys.path.append('../../GP_old/')

import matplotlib as mpl
import numpy as np
import gempy as gp
from gempy.core.tensor.modeltf_var import ModelTF
import matplotlib.pyplot as plt
# %%
data_path = 'https://raw.githubusercontent.com/cgre-aachen/gempy_data/master/'
path_to_data = data_path + "/data/input_data/jan_models/"

geo_data = gp.create_data( extent=[0, 1000, 0, 1000, 0, 1000], resolution=[10, 10, 10],
                          path_o=path_to_data + "model6_orientations.csv",
                          path_i=path_to_data + "model6_surface_points.csv")

# %%
gp.map_series_to_surfaces(geo_data, {"Strat_Series1": ('rock3'),
                                    "Strat_Series2": ('rock2', 'rock1'),
                                    "Basement_Series": ('basement')})

geo_data.add_surface_values([10, 2., 4., 7], 'densities')

# %%
## Initialize the model
model = ModelTF(geo_data)
model.activate_regular_grid()

max_length = np.sqrt(100**2 + 100**2 + 100**2)
max_slope = 1.5*2/max_length * model.rf
model.create_tensorflow_graph(gradient=True,compute_gravity=True,max_slope = 1000)

# %%
model.compute_model()
# %%
# gp._plot.plot_3d(model)
# %%
gp.plot.plot_section(model, cell_number=5,
                         direction='y', show_data=True)
# %%

cmin = np.min(model.solutions.values_matrix)
cmax = np.max(model.solutions.values_matrix)
norm = mpl.colors.Normalize(vmin=cmin, vmax=cmax)
gp.plot.plot_section(model, 
            cell_number=5,
            block = model.solutions.values_matrix,
            direction='y',
            show_grid=True, 
            show_data=True,
            colorbar = True,
            norm = norm,
            cmap = 'viridis')
# %%
density_matrix = model.solutions.values_matrix.reshape(10,10,10)
# %%
def plot_slice(ind,direction = 'y'):
    if direction == 'x':
        slice = density_matrix[ind,:,:].T  
    if direction == 'y':
        slice = density_matrix[:,ind,:].T
    if direction == 'z':
        slice = density_matrix[:,:,ind].T
    plt.imshow(slice, origin = 'lower')
    plt.colorbar()
    return slice
# %%
slice = plot_slice(5,'y')
# %%

plt.plot(slice[:,5])
# %%
