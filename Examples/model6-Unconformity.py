# %%
import sys
sys.path.append('../GP_old/')

import gempy as gp

from gempy.core.tensor.interpolator_tf import InterpolatorTF

# %%
data_path = 'https://raw.githubusercontent.com/cgre-aachen/gempy_data/master/'
path_to_data = data_path + "/data/input_data/jan_models/"

geo_data = gp.create_data( extent=[0, 1000, 0, 1000, 0, 1000], resolution=[50, 50, 50],
                          path_o=path_to_data + "model6_orientations.csv",
                          path_i=path_to_data + "model6_surface_points.csv")

# %%
gp.map_series_to_surfaces(geo_data, {"Strat_Series1": ('rock3'),
                                    "Strat_Series2": ('rock2', 'rock1'),
                                    "Basement_Series": ('basement')})

# %%
## I will integrate the module into GemPy through Interpolator later
model = InterpolatorTF(geo_data)
gpinput = model.get_graph_input()
model.create_tensorflow_graph(gpinput,gradient = False)

# %%
model.compute_model()

# %%
from gempy.plot.vista import GemPyToVista
import pyvista as pv

gpv = GemPyToVista(model)
gpv.plot_surface_points(surfaces='all')
gpv.plot_orientations()
gpv.plot_surfaces()
gpv.plot_structured_grid(scalar_field= 'lith')
gpv.p.show()

# %%
gp.plot.plot_section(model, cell_number=15,
                         direction='y', show_data=True)

