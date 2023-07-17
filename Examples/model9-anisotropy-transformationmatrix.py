'''
This is a experiment script to test anisotropy in GemPy 
Authors: Deep Prakash Ravi , Zhouji Liang
'''
# %%
import sys
sys.path.append('../GP_old/')

import gempy as gp
import numpy as np

from gempy.core.tensor.modeltf_var_Deep import ModelTF
import tensorflow as tf

tf.config.run_functions_eagerly(True)


# %%
data_path = '../GP_old/notebooks/data/input_data/George_models/'
geo_data = gp.create_data( extent=[0, 1000, 0, 1000, 0, 1000], resolution=[50, 50, 50],
                          path_o=data_path + "model1_orientations.csv",
                          path_i=data_path + "model1_anisotropy_surface_points.csv")


gp.map_series_to_surfaces(geo_data, {"Strat_Series2": ('rock2', 'rock1'),
                                    "Basement_Series": ('basement')})

geo_data.interpolator.additional_data.kriging_data.df['range'] = 300

# %%

# %%
model = ModelTF(geo_data)
model.activate_regular_grid()
Transformation_matrix = np.array([[1,0.3,-0.4],[0.,0.1,0.5],[0.3,0.1,1]]) # Some random transformation matrix
model.create_tensorflow_graph(gradient = False, Transformation_matrix = Transformation_matrix)


# %%
model.compute_model()
# %%
# 3D plot
gp._plot.plot_3d(model)
# %%
gp.plot.plot_section(model, cell_number=15,
                         direction='y', show_data=True,show_all_data = True,show_boundaries = True)
# %%