from typing import Union
from gempy.core.data import SurfacePoints, Orientations, Grid, Surfaces, Series, Faults, AdditionalData
from gempy.utils.meta import setdoc_pro, setdoc
import gempy.utils.docstring as ds

import numpy as np

class InterpolatorTF(object):
    def __init__(self, surface_points: "SurfacePoints", orientations: "Orientations", grid: "Grid",
                 surfaces: "Surfaces", series: Series, faults: "Faults", additional_data: "AdditionalData", **kwargs):
        
        # Test
        self.surface_points = surface_points
        self.orientations = orientations
        self.grid = grid
        self.additional_data = additional_data
        self.surfaces = surfaces
        self.series = series
        self.faults = faults

        self.dtype = additional_data.options.df.loc['values', 'dtype']

        self.len_series_i = np.zeros(1)
        self.len_series_o = np.zeros(1)
        self.len_series_u = np.zeros(1)
        self.len_series_f = np.zeros(1)
        self.len_series_w = np.zeros(1)

        self.set_initial_results()

        n_series = 1000

        self.compute_weights_ctrl = np.ones(n_series, dtype=bool)
        self.compute_scalar_ctrl = np.ones(n_series, dtype=bool)
        self.compute_block_ctrl = np.ones(n_series, dtype=bool)
    
    def set_theano_shared_kriging(self):
        """
        Set to the theano_graph attribute the shared variables of kriging values from the linked
         :class:`AdditionalData`.

        Returns:
            True
        """
        # Range
        # TODO add rescaled range and co into the rescaling data df?
        # self.theano_graph.a_T.set_value(np.cast[self.dtype]
        #                                 (self.additional_data.kriging_data.df.loc['values', 'range'] /
        #                                  self.additional_data.rescaling_data.df.loc[
        #                                      'values', 'rescaling factor']))
        # Covariance at 0
        # self.theano_graph.c_o_T.set_value(np.cast[self.dtype](
        #     self.additional_data.kriging_data.df.loc['values', '$C_o$'] /
        #     self.additional_data.rescaling_data.df.loc[
        #         'values', 'rescaling factor']
        # ))
        # universal grades
        # self.theano_graph.n_universal_eq_T.set_value(
        #     list(self.additional_data.kriging_data.df.loc['values', 'drift equations'].astype('int32')[self.non_zero]))

        self.set_theano_shared_nuggets()

    def set_theano_shared_nuggets(self):
        # nugget effect
        # len_orientations = self.additional_data.structure_data.df.loc['values', 'len series orientations']
        # len_orientations_len = np.sum(len_orientations)

        # self.theano_graph.nugget_effect_grad_T.set_value(
        #     np.cast[self.dtype](np.tile(
        #         self.orientations.df['smooth'], 3)))

        # len_rest_form = (self.additional_data.structure_data.df.loc['values', 'len surfaces surface_points'])
        # len_rest_len = np.sum(len_rest_form)
        # self.theano_graph.nugget_effect_scalar_T.set_value(
        #     np.cast[self.dtype](self.surface_points.df['smooth']))
        return True


    def set_theano_shared_structure_surfaces(self):
        """
        Set to the theano_graph attribute the shared variables of structure from the linked
         :class:`AdditionalData`.

        Returns:
            True
        """
        pass

    def get_python_input_weights(self, fault_drift=None):
        """
             Get values from the data objects used during the interpolation:
                 - dip positions XYZ
                 - dip angles
                 - azimuth
                 - polarity
                 - surface_points coordinates XYZ
             Returns:
                 (list)
             """
        # orientations, this ones I tile them inside theano. PYTHON VAR
        dips_position = self.orientations.df[['X_r', 'Y_r', 'Z_r']].values
        dip_angles = self.orientations.df["dip"].values
        azimuth = self.orientations.df["azimuth"].values
        polarity = self.orientations.df["polarity"].values
        surface_points_coord = self.surface_points.df[['X_r', 'Y_r', 'Z_r']].values
        if fault_drift is None:
            fault_drift = np.zeros((0, self.grid.values.shape[0] + 2 * self.len_series_i.sum()))

          #  fault_drift = np.zeros((0, surface_points_coord.shape[0]))

        # Set all in a list casting them in the chosen dtype
        idl = [np.cast[self.dtype](xs) for xs in (dips_position, dip_angles, azimuth, polarity, surface_points_coord,
                                                  fault_drift)]
        return idl

    def get_python_input_zx(self, fault_drift=None):
        """
             Get values from the data objects used during the interpolation:
                 - dip positions XYZ
                 - dip angles
                 - azimuth
                 - polarity
                 - surface_points coordinates XYZ
             Returns:
                 (list)
             """
        # orientations, this ones I tile them inside theano. PYTHON VAR
        dips_position = self.orientations.df[['X_r', 'Y_r', 'Z_r']].values
        dip_angles = self.orientations.df["dip"].values
        azimuth = self.orientations.df["azimuth"].values
        polarity = self.orientations.df["polarity"].values
        surface_points_coord = self.surface_points.df[['X_r', 'Y_r', 'Z_r']].values
        grid = self.grid.values_r

        if fault_drift is None:
            fault_drift = np.zeros((0, grid.shape[0] + 2 * self.len_series_i.sum()))

      #      fault_drift = np.zeros((0, grid.shape[0] + surface_points_coord.shape[0]))

        # Set all in a list casting them in the chosen dtype
        idl = [np.cast[self.dtype](xs) for xs in (dips_position, dip_angles, azimuth, polarity, surface_points_coord,
                                                  fault_drift, grid)]
        return idl

    def reset_flow_control_initial_results(self, reset_weights=True, reset_scalar=True, reset_block=True):
        """
        Method to reset to the initial state all the recompute ctrl. After calling this method next time
         gp.compute_model is called, everything will be computed. Panic bottom.

        Args:
            reset_weights (bool):
            reset_scalar (bool):
            reset_block (bool):

        Returns:
            True
        """
        n_series = self.len_series_i.shape[0]#self.additional_data.get_additional_data()['values']['Structure', 'number series']
        x_to_interp_shape = self.grid.values_r.shape[0] + 2 * self.len_series_i.sum()
       
        if reset_weights is True:
            self.compute_weights_ctrl = np.ones(1000, dtype=bool)
            # self.theano_graph.weights_vector.set_value(np.zeros((self.len_series_w.sum()), dtype=self.dtype))

        if reset_scalar is True:
            self.compute_scalar_ctrl = np.ones(1000, dtype=bool)
            # self.theano_graph.scalar_fields_matrix.set_value(
            #     np.zeros((n_series, x_to_interp_shape), dtype=self.dtype))

        if reset_block is True:
            self.compute_block_ctrl = np.ones(1000, dtype=bool)
            # self.theano_graph.mask_matrix.set_value(np.zeros((n_series, x_to_interp_shape), dtype='bool'))
            # self.theano_graph.block_matrix.set_value(
            #     np.zeros((n_series, self.surfaces.df.iloc[:, self.surfaces._n_properties:].values.shape[1],
            #               x_to_interp_shape), dtype=self.dtype))
        return True

    def set_flow_control(self):
        """
        Initialize the ctrl vectors to the number of series size.

        Returns:
            True
        """
        n_series = 1000
        self.compute_weights_ctrl = np.ones(n_series, dtype=bool)
        self.compute_scalar_ctrl = np.ones(n_series, dtype=bool)
        self.compute_block_ctrl = np.ones(n_series, dtype=bool)
        return True

    @setdoc_pro(reset_flow_control_initial_results.__doc__)
    def set_all_shared_parameters(self, reset_ctrl=False):
        """
        Set all theano shared parameters required for the computation of lithology

        Args:
            reset_ctrl (bool): If true, [s0]

        Returns:
            True
        """
        self.set_theano_shared_loop()
        self.set_theano_shared_relations()
        self.set_theano_shared_kriging()
        self.set_theano_shared_structure_surfaces()
        # self.set_theano_shared_topology()
        if reset_ctrl is True:
            self.reset_flow_control_initial_results()

        return True

    def set_theano_shared_topology(self):

        max_lith = self.surfaces.df.groupby('isFault')['id'].count()[False]
        if type(max_lith) != int:
            max_lith = 0

        # self.theano_graph.max_lith.set_value(max_lith)
        # self.theano_graph.regular_grid_res.set_value(self.grid.regular_grid.resolution)
        # self.theano_graph.dxdydz.set_value(np.array(self.grid.regular_grid.get_dx_dy_dz(), dtype=self.dtype))

    @setdoc_pro(reset_flow_control_initial_results.__doc__)
    def set_theano_shared_structure(self, reset_ctrl=False):
        """
        Set all theano shared variable dependent on :class:`Structure`.

        Args:
            reset_ctrl (bool): If true, [s0]

        Returns:
            True

        """
        self.set_theano_shared_loop()
        self.set_theano_shared_relations()
        self.set_theano_shared_structure_surfaces()
        # universal grades
       # self.theano_graph.n_universal_eq_T.set_value(
       #     list(self.additional_data.kriging_data.df.loc['values', 'drift equations'].astype('int32')))

        if reset_ctrl is True:
            self.reset_flow_control_initial_results()
        return True

    def remove_series_without_data(self):
        len_series_i = self.additional_data.structure_data.df.loc['values', 'len series surface_points'] - \
                            self.additional_data.structure_data.df.loc['values', 'number surfaces per series']

        len_series_o = self.additional_data.structure_data.df.loc['values', 'len series orientations'].astype(
            'int32')

        # Remove series without data
        non_zero_i = len_series_i.nonzero()[0]
        non_zero_o = len_series_o.nonzero()[0]
        non_zero = np.intersect1d(non_zero_i, non_zero_o)

        self.non_zero = non_zero
        return self.non_zero

    def _compute_len_series(self):

        self.len_series_i = self.additional_data.structure_data.df.loc['values', 'len series surface_points'] - \
                            self.additional_data.structure_data.df.loc['values', 'number surfaces per series']

        self.len_series_o = self.additional_data.structure_data.df.loc['values', 'len series orientations'].astype(
            'int32')

        # Remove series without data
        non_zero_i = self.len_series_i.nonzero()[0]
        non_zero_o = self.len_series_o.nonzero()[0]
        non_zero = np.intersect1d(non_zero_i, non_zero_o)

        self.non_zero = non_zero
        

        self.len_series_u = self.additional_data.kriging_data.df.loc['values', 'drift equations'].astype('int32')
        try:
            len_series_f_ = self.faults.faults_relations_df.values[non_zero][:, non_zero].sum(axis=0)

        except np.AxisError:
            print('np.axis error')
            len_series_f_ = self.faults.faults_relations_df.values.sum(axis=0)

        self.len_series_f = np.atleast_1d(len_series_f_.astype('int32'))#[:self.additional_data.get_additional_data()['values']['Structure', 'number series']]

        self._old_len_series = self.len_series_i

        self.len_series_i = self.len_series_i[non_zero]
        self.len_series_o = self.len_series_o[non_zero]
       # self.len_series_f = self.len_series_f[non_zero]
        self.len_series_u = self.len_series_u[non_zero]

        if self.len_series_i.shape[0] == 0:
            self.len_series_i = np.zeros(1, dtype=int)
            self._old_len_series = self.len_series_i

        if self.len_series_o.shape[0] == 0:
            self.len_series_o = np.zeros(1, dtype=int)
        if self.len_series_u.shape[0] == 0:
            self.len_series_u = np.zeros(1, dtype=int)
        if self.len_series_f.shape[0] == 0:
            self.len_series_f = np.zeros(1, dtype=int)

        self.len_series_w = self.len_series_i + self.len_series_o * 3 + self.len_series_u + self.len_series_f

    def set_theano_shared_loop(self):
        """Set the theano shared variables that are looped for each series."""
        self._compute_len_series()

        # self.theano_graph.len_series_i.set_value(np.insert(self.len_series_i.cumsum(), 0, 0).astype('int32'))
        # self.theano_graph.len_series_o.set_value(np.insert(self.len_series_o.cumsum(), 0, 0).astype('int32'))
        # self.theano_graph.len_series_w.set_value(np.insert(self.len_series_w.cumsum(), 0, 0).astype('int32'))

        # Number of surfaces per series. The function is not pretty but the result is quite clear
        n_surfaces_per_serie = np.insert(
            self.additional_data.structure_data.df.loc['values', 'number surfaces per series'][self.non_zero].cumsum(), 0, 0). \
            astype('int32')
        # self.theano_graph.n_surfaces_per_series.set_value(n_surfaces_per_serie)
        # self.theano_graph.n_universal_eq_T.set_value(
        #     list(self.additional_data.kriging_data.df.loc['values', 'drift equations'].astype('int32')[self.non_zero]))

    @setdoc_pro(set_theano_shared_loop.__doc__)
    def set_theano_shared_weights(self):
        """Set the theano shared weights and [s0]"""
        self.set_theano_shared_loop()
        # self.theano_graph.weights_vector.set_value(np.zeros((self.len_series_w.sum()), dtype=self.dtype))

    def set_theano_shared_fault_relation(self):
        self.remove_series_without_data()
        """Set the theano shared variable with the fault relation"""
        # self.theano_graph.fault_relation.set_value(
        #     self.faults.faults_relations_df.values[self.non_zero][:, self.non_zero])

    def set_theano_shared_is_fault(self):
        """Set theano shared variable which controls if a series is fault or not"""
        # self.theano_graph.is_fault.set_value(self.faults.df['isFault'].values[self.non_zero])
        pass

    def set_theano_shared_is_finite(self):
        """Set theano shared variable which controls if a fault is finite or not"""
        # self.theano_graph.is_finite_ctrl.set_value(self.faults.df['isFinite'].values)
        pass

    def set_theano_shared_onlap_erode(self):
        """Set the theano variables which control the masking patterns according to the uncomformity relation"""
        self.remove_series_without_data()

        is_erosion = self.series.df['BottomRelation'].values[self.non_zero] == 'Erosion'
        is_onlap = np.roll(self.series.df['BottomRelation'].values[self.non_zero] == 'Onlap', 1)

        if len(is_erosion) != 0:
            is_erosion[-1] = False
        # this comes from the series df
        # self.theano_graph.is_erosion.set_value(is_erosion)
        # self.theano_graph.is_onlap.set_value(is_onlap)

    def set_theano_shared_faults(self):
        """Set all theano shared variables wich controls the faults behaviour"""

        self.set_theano_shared_fault_relation()
        # This comes from the faults df
        self.set_theano_shared_is_fault()
        self.set_theano_shared_is_finite()

    def set_theano_shared_relations(self):
        """Set all theano shared variables that control all the series interactions with each other"""
        self.set_theano_shared_fault_relation()
        # This comes from the faults df
        self.set_theano_shared_is_fault()
        self.set_theano_shared_is_finite()
        self.set_theano_shared_onlap_erode()

    def set_initial_results(self):
        """
        Initialize all the theano shared variables where we store the final results of the interpolation.
        This function must be called always after set_theano_shared_loop

        Returns:
            True
        """
        self._compute_len_series()

        x_to_interp_shape = self.grid.values_r.shape[0] + 2 * self.len_series_i.sum()
        n_series = self.len_series_i.shape[0]#self.additional_data.structure_data.df.loc['values', 'number series']

        # self.theano_graph.weights_vector.set_value(np.zeros((self.len_series_w.sum()), dtype=self.dtype))
        # self.theano_graph.scalar_fields_matrix.set_value(
        #     np.zeros((n_series, x_to_interp_shape), dtype=self.dtype))

        # self.theano_graph.mask_matrix.set_value(np.zeros((n_series, x_to_interp_shape), dtype='bool'))
        # self.theano_graph.block_matrix.set_value(
        #     np.zeros((n_series, self.surfaces.df.iloc[:, self.surfaces._n_properties:].values.shape[1],
        #               x_to_interp_shape), dtype=self.dtype))
        return True

    def set_initial_results_matrices(self):
        """
        Initialize all the theano shared variables where we store the final results of the interpolation except the
        kriging weights vector.


        Returns:
            True
        """
        self._compute_len_series()

        x_to_interp_shape = self.grid.values_r.shape[0] + 2 * self.len_series_i.sum()
        n_series = self.len_series_i.shape[0]#self.additional_data.structure_data.df.loc['values', 'number series']

        # self.theano_graph.scalar_fields_matrix.set_value(
        #     np.zeros((n_series, x_to_interp_shape), dtype=self.dtype))

        # self.theano_graph.mask_matrix.set_value(np.zeros((n_series, x_to_interp_shape), dtype='bool'))
        # self.theano_graph.block_matrix.set_value(
        #     np.zeros((n_series, self.surfaces.df.iloc[:, self.surfaces._n_properties:].values.shape[1],
        #               x_to_interp_shape), dtype=self.dtype))

    def set_theano_shared_grid(self, grid=None):
        if grid == 'shared':
            grid_sh = self.grid.values_r
            # self.theano_graph.grid_val_T = theano.shared(grid_sh.astype(self.dtype), 'Constant values to interpolate.')
        elif grid is not None:
            # self.theano_graph.grid_val_T = theano.shared(grid.astype(self.dtype), 'Constant values to interpolate.')
            pass

    def modify_results_matrices_pro(self):
        """
        Modify all theano shared matrices to the right size according to the structure data. This method allows
        to change the size of the results without having the recompute all series"""

        old_len_i = self._old_len_series
        new_len_i = self.additional_data.structure_data.df.loc['values', 'len series surface_points'] - \
            self.additional_data.structure_data.df.loc['values', 'number surfaces per series']
        if new_len_i.shape[0] < old_len_i.shape[0]:
            self.set_initial_results()
            old_len_i = old_len_i[old_len_i != 0]
        elif new_len_i.shape[0] > old_len_i.shape[0]:
            self.set_initial_results()
            new_len_i = new_len_i[new_len_i != 0]
        else:
            # scalar_fields_matrix = self.theano_graph.scalar_fields_matrix.get_value()
            # mask_matrix = self.theano_graph.mask_matrix.get_value()
            # block_matrix = self.theano_graph.block_matrix.get_value()

            # len_i_diff = new_len_i - old_len_i
            # for e, i in enumerate(len_i_diff):
            #     loc = self.grid.values_r.shape[0] + old_len_i[e]
            #     i *= 2
            #     if i == 0:
            #         pass
            #     elif i > 0:
            #         self.theano_graph.scalar_fields_matrix.set_value(
            #             np.insert(scalar_fields_matrix, [loc], np.zeros(i), axis=1))
            #         self.theano_graph.mask_matrix.set_value(np.insert(
            #             mask_matrix, [loc], np.zeros(i, dtype=self.dtype), axis=1))
            #         self.theano_graph.block_matrix.set_value(np.insert(
            #             block_matrix, [loc], np.zeros(i, dtype=self.dtype), axis=2))

            #     else:
            #         self.theano_graph.scalar_fields_matrix.set_value(
            #             np.delete(scalar_fields_matrix, np.arange(loc, loc+i, -1) - 1, axis=1))
            #         self.theano_graph.mask_matrix.set_value(
            #             np.delete(mask_matrix, np.arange(loc, loc+i, -1) - 1, axis=1))
            #         self.theano_graph.block_matrix.set_value(
            #             np.delete(block_matrix, np.arange(loc, loc+i, -1) - 1, axis=2))
            pass

        self.modify_results_weights()

    def modify_results_weights(self):
        """Modify the theano shared weights vector according to the structure.
        """
        old_len_w = self.len_series_w
        self._compute_len_series()
        new_len_w = self.len_series_w
        if new_len_w.shape[0] != old_len_w[0]:
            self.set_initial_results()
        else:
            pass
            # weights = self.theano_graph.weights_vector.get_value()
            # len_w_diff = new_len_w - old_len_w
            # for e, i in enumerate(len_w_diff):
            #  #   print(len_w_diff, weights)
            #     if i == 0:
            #         pass
            #     elif i > 0:
            #         self.theano_graph.weights_vector.set_value(np.insert(weights, old_len_w[e], np.zeros(i)))
            #     else:
            #   #      print(np.delete(weights, np.arange(old_len_w[e],  old_len_w[e] + i, -1)-1))
            #         self.theano_graph.weights_vector.set_value(
            #             np.delete(weights, np.arange(old_len_w[e],  old_len_w[e] + i, -1)-1))

    def get_python_input_block(self, append_control=True, fault_drift=None):
        """
        Get values from the data objects used during the interpolation:
             - dip positions XYZ
             - dip angles
             - azimuth
             - polarity
             - surface_points coordinates XYZ

        Args:
            append_control (bool): If true append the ctrl vectors to the input list
            fault_drift (Optional[np.array]): matrix with per computed faults to drift the model

        Returns:
            list: list of arrays with all the input parameters to the theano function
        """
        # orientations, this ones I tile them inside theano. PYTHON VAR
        dips_position = self.orientations.df[['X_r', 'Y_r', 'Z_r']].values
        dip_angles = self.orientations.df["dip"].values
        azimuth = self.orientations.df["azimuth"].values
        polarity = self.orientations.df["polarity"].values
        surface_points_coord = self.surface_points.df[['X_r', 'Y_r', 'Z_r']].values
        grid = self.grid.values_r
        if fault_drift is None:
            fault_drift = np.zeros((0, grid.shape[0] + 2 * self.len_series_i.sum()))

        # values_properties = np.array([[]], dtype='float32')
        # g = self.surfaces.df.groupby('series')
        # for series_ in self.series.df.index.values[self.non_zero]:
        #     values_properties = np.append(values_properties,
        #                                   g.get_group(series_).iloc[:, self.surfaces._n_properties:].values.
        #                                   astype(self.dtype).T, axis=1)

      #  values_properties = self.surfaces.df.iloc[:, self.surfaces._n_properties:].values.astype(self.dtype).T

        values_properties = self.surfaces.df.groupby('isActive').get_group(
            True).iloc[:, self.surfaces._n_properties:].values.astype(self.dtype).T
        # Set all in a list casting them in the chosen dtype
        idl = [np.cast[self.dtype](xs) for xs in (dips_position, dip_angles, azimuth, polarity,
                                                  surface_points_coord,
                                                  fault_drift, grid, values_properties)]
        if append_control is True:
            idl.append(self.compute_weights_ctrl)
            idl.append(self.compute_scalar_ctrl)
            idl.append(self.compute_block_ctrl)

        return idl

    def print_theano_shared(self):
        """Print many of the theano shared variables"""
        pass
        # print('len sereies i', self.theano_graph.len_series_i.get_value())
        # print('len sereies o', self.theano_graph.len_series_o.get_value())
        # print('len sereies w', self.theano_graph.len_series_w.get_value())
        # print('n surfaces per series', self.theano_graph.n_surfaces_per_series.get_value())
        # print('n universal eq',self.theano_graph.n_universal_eq_T.get_value())
        # print('is finite', self.theano_graph.is_finite_ctrl.get_value())
        # print('is erosion', self.theano_graph.is_erosion.get_value())
        # print('is onlap', self.theano_graph.is_onlap.get_value())

    # def compile_th_fn_geo(self, inplace=False, debug=True, grid: Union[str, np.ndarray] = None):
    #     """
    #     Compile and create the theano function which can be evaluated to compute the geological models

    #     Args:

    #         inplace (bool): If true add the attribute theano.function to the object inplace
    #         debug (bool): If true print some of the theano flags
    #         grid: If None, grid will be passed as variable. If shared or np.ndarray the grid will be treated as
    #          constant (if shared the grid will be taken of grid)

    #     Returns:
    #         theano.function: function that computes the whole interpolation
    #     """

    #     self.set_all_shared_parameters(reset_ctrl=False)
    #     # This are the shared parameters and the compilation of the function. This will be hidden as well at some point
    #     input_data_T = self.theano_graph.input_parameters_loop
    #     print('Compiling theano function...')
    #     if grid == 'shared' or grid is not None:
    #         self.set_theano_shared_grid(grid)

    #     th_fn = theano.function(input_data_T,
    #                             self.theano_graph.theano_output(),
    #                             updates=[
    #                                     (self.theano_graph.block_matrix, self.theano_graph.new_block),
    #                                     (self.theano_graph.weights_vector, self.theano_graph.new_weights),
    #                                     (self.theano_graph.scalar_fields_matrix, self.theano_graph.new_scalar),
    #                                     (self.theano_graph.mask_matrix, self.theano_graph.new_mask)
    #                                    ],
    #                             on_unused_input='ignore',
    #                             allow_input_downcast=False,
    #                             profile=False)

    #     if inplace is True:
    #         self.theano_function = th_fn

    #     if debug is True:
    #         print('Level of Optimization: ', theano.config.optimizer)
    #         print('Device: ', theano.config.device)
    #         print('Precision: ', theano.config.floatX)
    #         print('Number of faults: ', self.additional_data.structure_data.df.loc['values', 'number faults'])
    #     print('Compilation Done!')

    #     return th_fn

