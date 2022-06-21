# %%
import sys
sys.path.append('../GP_old/')

import gempy as gp

from gempy import map_series_to_surfaces
from gempy.core.tensor.modeltf import ModelTF

# %%
geo_model = gp.create_model('Greenstone')
data_path = 'https://raw.githubusercontent.com/cgre-aachen/gempy_data/master/'

# Importing the data from csv files and settign extent and resolution
geo_model = gp.init_data(geo_model, [696000, 747000, 6863000, 6930000, -20000, 200], [50, 50, 50],
                         path_o=data_path + "/data/input_data/tut_SandStone/SandStone_Foliations.csv",
                         path_i=data_path + "/data/input_data/tut_SandStone/SandStone_Points.csv")

# %%
map_series_to_surfaces(
            geo_model,
            {"EarlyGranite_Series": 'EarlyGranite',
                                     "BIF_Series": ('SimpleMafic2', 'SimpleBIF'),
                                     "SimpleMafic_Series": 'SimpleMafic1', 'Basement': 'basement'})
# %%
geo_model.add_surface_values([2.61, 2.92, 3.1, 2.92, 2.61])

# %%
model = ModelTF(geo_model)
gpinput = model.get_graph_input()
model.create_tensorflow_graph(gpinput,gradient = False)
# %%
model.compute_model()
# %%
gp._plot.plot_3d(model)

# %%
# plot the top surface lithlogy
gp._plot.plot_2d(model,cell_number = [49],show_data = True,direction = ['z'],figsize = (10,10))