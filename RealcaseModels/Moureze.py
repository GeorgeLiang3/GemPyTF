# %%
import sys
sys.path.append('../GP_old/')

import gempy as gp

from gempy import map_series_to_surfaces
from gempy.core.tensor.modeltf import ModelTF

import pandas as pn
# %%
Moureze_points = pn.read_csv(
    'https://raw.githubusercontent.com/Loop3D/ImplicitBenchmark/master/Moureze/Moureze_Points.csv', sep=';',
    names=['X', 'Y', 'Z', 'G_x', 'G_y', 'G_z', '_'], header=0, )
Sections_EW = pn.read_csv('https://raw.githubusercontent.com/Loop3D/ImplicitBenchmark/master/Moureze/Sections_EW.csv',
                          sep=';',
                          names=['X', 'Y', 'Z', 'ID', '_'], header=1).dropna()
Sections_NS = pn.read_csv('https://raw.githubusercontent.com/Loop3D/ImplicitBenchmark/master/Moureze/Sections_NS.csv',
                          sep=';',
                          names=['X', 'Y', 'Z', 'ID', '_'], header=1).dropna()

mask_surfpoints = Moureze_points['G_x'] < -9999
surfpoints = Moureze_points[mask_surfpoints]
orientations = Moureze_points[~mask_surfpoints]

surfpoints['surface'] = '0'
orientations['surface'] = '0'
# %%
resolution_requ = [156, 206, 76]
resolution = [77, 103, 38]
resolution_low = [45, 51, 38]
geo_model = gp.create_model('Moureze')
geo_model = gp.init_data(geo_model,
                         extent=[-5, 305, -5, 405, -200, -50], resolution=resolution_low,
                         surface_points_df=surfpoints, orientations_df=orientations,
                         surface_name='surface',
                         add_basement=True)
# %%
new_range = geo_model.get_additional_data().loc[('Kriging', 'range'), 'values'] * 0.2
geo_model.modify_kriging_parameters('range', new_range)
# %%
model = ModelTF(geo_model)
gpinput = model.get_graph_input()
model.create_tensorflow_graph(gpinput,gradient = False)
# %%
model.compute_model()
# %%
from gempy.plot.vista import GemPyToVista

gpv = GemPyToVista(model)
gpv.plot_surface_points(surfaces='all')
gpv.plot_orientations()
gpv.plot_surfaces()
gpv.plot_structured_grid(scalar_field= 'lith')
gpv.p.show()
# %%
