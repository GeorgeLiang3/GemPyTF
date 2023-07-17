# %%
import sys
sys.path.append('../GP_old/')

import gempy as gp
from gempy.core.tensor.modeltf_var import ModelTF

# %%
# data_path = 'https://raw.githubusercontent.com/cgre-aachen/gempy_data/master/'
path_to_data = 'https://raw.githubusercontent.com/cgre-aachen/gempy_data/master/'

geo_data = gp.create_data( extent=[337000, 400000, 6640000, 6710000, -18000, 1000], resolution=[100, 100, 100],
                          path_i=path_to_data + "/data/input_data/Perth_basin/Paper_GU2F_sc_faults_topo_Points.csv",
                          path_o=path_to_data + "/data/input_data/Perth_basin/Paper_GU2F_sc_faults_topo_Foliations.csv")
del_surfaces = ['Cadda', 'Woodada_Kockatea', 'Cattamarra']
geo_data.delete_surfaces(del_surfaces, remove_data=True)
# geo_data.get_data()
geo_data.surface_points.df.dropna(inplace=True)
geo_data.orientations.df.dropna(inplace=True)

gp.map_series_to_surfaces(geo_data, {"fault_Abrolhos_Transfer": ["Abrolhos_Transfer"],
                           "fault_Coomallo": ["Coomallo"],
                           "fault_Eneabba_South": ["Eneabba_South"],
                           "fault_Hypo_fault_W": ["Hypo_fault_W"],
                           "fault_Hypo_fault_E": ["Hypo_fault_E"],
                           "fault_Urella_North": ["Urella_North"],
                           "fault_Urella_South": ["Urella_South"],
                           "fault_Darling": ["Darling"],
                           "Sedimentary_Series": ['Cretaceous',
                                                  'Yarragadee',
                                                  'Eneabba',
                                                  'Lesueur',
                                                  'Permian']
                           })
# %%
order_series = ["fault_Abrolhos_Transfer",
                "fault_Coomallo",
                "fault_Eneabba_South",
                "fault_Hypo_fault_W",
                "fault_Hypo_fault_E",
                "fault_Urella_North",
                "fault_Darling",
                "fault_Urella_South",
                "Sedimentary_Series", 'Basement']

geo_data.reorder_series(order_series)
geo_data.set_is_fault(["fault_Abrolhos_Transfer",
                        "fault_Coomallo",
                        "fault_Eneabba_South",
                        "fault_Hypo_fault_W",
                        "fault_Hypo_fault_E",
                        "fault_Urella_North",
                        "fault_Darling",
                        "fault_Urella_South"])


# %%
## Initialize the model
model = ModelTF(geo_data)
model.activate_regular_grid()
model.create_tensorflow_graph(gradient = False)

# %%
model.compute_model()
# %%
gp._plot.plot_3d(model)
# %%
gp.plot.plot_section(model, cell_number=15,
                         direction='y', show_data=True)
# %%