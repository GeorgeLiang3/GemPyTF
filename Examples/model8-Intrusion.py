# %%
import sys
sys.path.append('../GP_old/')

import gempy as gp
import numpy as np

from gempy.core.tensor.modeltf_var import ModelTF
import tensorflow as tf

tf.config.run_functions_eagerly(True)


# %%
data_path = '../GP_old/notebooks/data/input_data/George_models/'
geo_data = gp.create_data( extent=[0, 1000, 0, 1000, 0, 1000], resolution=[50, 50, 50],
                          path_o=data_path + "model_intrusion_reduce_orientations.csv",
                          path_i=data_path + "model_intrusion_reduce_surface_points.csv")


gp.map_series_to_surfaces(geo_data, {"Strat_Series1": ('rock3'),
                                    "Strat_Series2": ('rock2', 'rock1'),
                                    "Basement_Series": ('basement')})
# %%
## modify one surface point

# geo_data.surface_points.modify_surface_points(21,Z = 600,surface = 'rock2')
# %%
## modify one surface point

# geo_data.surface_points.modify_surface_points(21,Z = 600,surface = 'rock2')

mapping_object = {'Strat_Series1':   np.array([1,1,0.01]),
                'Strat_Series2':   np.array([1,1,1]),
                'Basement_Series': np.array([1,1,1])}
gp.assign_global_anisotropy(geo_data,mapping_object)
# %%
## I will integrate the module into GemPy through Interpolator later
model = ModelTF(geo_data)
model.activate_regular_grid()
model.create_tensorflow_graph(gradient = False)


# %%
model.compute_model()
# %%
# 3D plot
gp._plot.plot_3d(model)
# %%
gp.plot.plot_section(model, cell_number=15,
                         direction='y', show_data=True,show_all_data = True,show_boundaries = True)
# %%