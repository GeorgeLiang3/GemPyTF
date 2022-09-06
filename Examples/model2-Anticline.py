# %%
import sys
sys.path.append('../GP_old/')

import gempy as gp

from gempy.core.tensor.modeltf_var import ModelTF
import tensorflow as tf

tf.config.run_functions_eagerly(True)


# %%
data_path = 'https://raw.githubusercontent.com/cgre-aachen/gempy_data/master/'
path_to_data = data_path + "/data/input_data/jan_models/"
geo_data = gp.create_data( extent=[0, 1000, 0, 1000, 0, 1000], resolution=[50, 50, 50],
                          path_o=path_to_data + "model2_orientations.csv",
                          path_i=path_to_data + "model2_surface_points.csv")


gp.map_series_to_surfaces(geo_data, {"Strat_Series": ('rock2', 'rock1'), "Basement_Series": ('basement')})

geo_data.add_surface_values([1.0,4.0,2.0])
geo_data.modify_order_surfaces(2,0)

# %%
## modify one surface point

geo_data.surface_points.modify_surface_points(21,Z = 600,surface = 'rock2')
# %%
## I will integrate the module into GemPy through Interpolator later
model = ModelTF(geo_data)
gpinput = model.get_graph_input()
model.create_tensorflow_graph(gpinput,gradient = False)


# %%
model.compute_model()
# %%
# 3D plot
gp._plot.plot_3d(model)
# %%
gp.plot.plot_section(model, cell_number=15,
                         direction='y', show_data=True,show_all_data = True,show_boundaries = True)
# %%