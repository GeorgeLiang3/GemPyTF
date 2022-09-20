# %%
import sys
sys.path.append('../GP_old/')

import gempy as gp

from gempy.core.tensor.modeltf_var import ModelTF

# %%
data_path = 'https://raw.githubusercontent.com/cgre-aachen/gempy_data/master/'
path_to_data = data_path + "/data/input_data/jan_models/"

geo_data = gp.create_data( extent=[0, 2500, 0, 1000, 0, 1000],
                          resolution=[125, 50, 50],
                          path_o=path_to_data + "model7_orientations.csv",
                          path_i=path_to_data + "model7_surface_points.csv")

# %%
gp.map_series_to_surfaces(geo_data, {"Fault_Series": 'fault', "Strat_Series1": ('rock3'),
                                     "Strat_Series2": ('rock2','rock1')})


geo_data.set_is_fault(['Fault_Series'])

# %%
## I will integrate the module into GemPy through Interpolator later

model = ModelTF(geo_data)
model.activate_regular_grid()
gpinput = model.get_graph_input()
model.create_tensorflow_graph(gpinput,gradient = False)


# %%
model.compute_model()
# %%
gp._plot.plot_3d(model)
# %%
gp.plot.plot_section(model, cell_number=15,
                         direction='y', show_data=True,show_all_data = True)
