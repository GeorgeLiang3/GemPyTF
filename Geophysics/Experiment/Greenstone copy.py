# %%
import sys
sys.path.append('../../GP_old')

import gempy as gp
import numpy as np
import pyvista as pv

from gempy import map_series_to_surfaces
from gempy.core.tensor.modeltf import ModelTF

import warnings
warnings.filterwarnings("ignore")

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
grav_res = 10
X_r = np.linspace(704000,740000,grav_res)
Y_r = np.linspace(6.87e6,6.92e6,grav_res)
r = []
for x in X_r:
  for y in Y_r:
    r.append(np.array([x,y]))
receivers = np.array(r)
Z_r = 0
n_devices = receivers.shape[0]
xyz = np.meshgrid(X_r, Y_r, Z_r)
xy_ravel = np.vstack(list(map(np.ravel, xyz))).T

from gempy.plot.vista import GemPyToVista

gpv = GemPyToVista(model, plotter_type='basic')

# add receivers at the ground surface
poly = pv.PolyData(xy_ravel)
geom = pv.Cone(direction=[0.0, 0.0, -1.0])
glyphs = poly.glyph(factor=1000.0,geom=geom)
gpv.p.add_mesh(glyphs, color="FFCC99",render_points_as_spheres=True,point_size=100) # add the receivers to the gempyvista mesh

gpv.plot_surfaces()
gpv.plot_data()
gpv.plot_structured_grid('lith')
gpv.set_bounds(location = 'front',font_size = 25)

# gpv.p.camera.position = (818436.0748215821, 6993436.074821582, 87699.85942851572)
# gpv.p.camera.focal_point = (721245.0, 6896335.134559274, -9236.215393066406)
# gpv.p.camera.up = (0.0, 0.0, 1.0)
# # gpv.p.camera.zoom(0.7)
# gpv.p.save_graphic("sphere1st_3000_.pdf")  
gpv.p.show(screenshot='greenstone3D.png',cpos = [(817598.0415050037, 7001941.515991688, 73329.35067484964),
 (697696.2798326241, 6915813.706168675, -6641.742405084278),
 (-0.34059640955355713, -0.3398255753970325, 0.8766485408105699)],return_cpos = True)

# gp._plot.plot_3d(model)

# # %%
# # plot the top surface lithlogy
# gp._plot.plot_2d(model,cell_number = [49],show_data = True,direction = ['z'],figsize = (10,10))
# %%
