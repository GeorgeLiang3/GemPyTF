# %%
import sys
sys.path.append('../../GP_old/')
import tensorflow as tf
import gempy as gp
from gempy.core.tensor.modeltf_var import ModelTF

# %%
data_path = '/Volumes/GoogleDrive/My Drive/GemPhy/GP_old/notebooks'
path_to_data = data_path + "/data/input_data/George_models/"

geo_data = gp.create_data( extent=[0, 1000, 0, 1000, 0, 1000], resolution=[50, 50, 50],
                          path_o=path_to_data + "model5_modified_orientations.csv",
                          path_i=path_to_data + "model5_modified_surface_points.csv")

geo_data.get_data()

gp.map_series_to_surfaces(geo_data, {"Fault_Series": 'fault',
                                    "Strat_Series": ('rock2', 'rock1')})
geo_data.set_is_fault(['Fault_Series'])


# %%
## Initialize the model
model = ModelTF(geo_data)
model.activate_regular_grid()
gpinput = model.get_graph_input()
model.create_tensorflow_graph(gpinput,gradient = False)

# %%
dip_angles = tf.constant([10.,0,0,0,0], dtype = tf.float64)
model.orientations.modify_orientations(idx = 4, dip = dip_angles[0])
model.compute_model(surface_points = model.surface_points_coord,dip_angles = dip_angles)
# %%
gp._plot.plot_3d(model)
# %%
gp.plot.plot_section(model, cell_number=15,
                         direction='y', show_data=True)
# %%