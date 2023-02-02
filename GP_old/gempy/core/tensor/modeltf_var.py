import numpy as np
from gempy.core.data import SurfacePoints, Orientations, Grid, Surfaces, Series, Faults, AdditionalData
import tensorflow as tf
from gempy.core.tensor.tensorflow_graph_uncon_sig_fault_var import TFGraph
from gempy.core.solution import Solution
from gempy.core.model import DataMutation
from gempy.assets.geophysics import *
from gempy.core.grid_modules.grid_types import CenteredRegGrid
import copy

class ModelTF(DataMutation):
    def __init__(self,geo_data,dtype ='float64') -> None:
        super().__init__()
        # geo_data.surfaces.df.sort_values(by=['order_surfaces'], inplace=True, ascending=False)
        # geo_data.surfaces.update_id()

        self.geo_data = geo_data
        if dtype == 'float64':
            self.tfdtype = tf.float64
        elif dtype == 'float32':
            self.tfdtype = tf.float32
        self.dtype = dtype
        self.from_gempy_interpolator()
    def from_gempy_interpolator(self):
        self.interpolator = self.geo_data.interpolator
        self.additional_data = self.geo_data.additional_data
        self.faults = self.geo_data.faults
        self.series = self.geo_data.series
        self._grid = copy.deepcopy(self.geo_data.grid) # grid object
        # extract data from gempy interpolator
        dips_position, dip_angles, azimuth, polarity, surface_points_coord, fault_drift, grid, values_properties = self.interpolator.get_python_input_block()[0:-3]

        self.resolution = self.geo_data.grid.regular_grid.resolution
        dip_angles = tf.cast(dip_angles,self.tfdtype)


        grid = tf.cast(grid,self.tfdtype,name = 'cast_grid')
        self.dips_position = tf.cast(dips_position,self.tfdtype)
        azimuth = tf.cast(azimuth,self.tfdtype)
        polarity = tf.cast(polarity,self.tfdtype)
        fault_drift = tf.cast(fault_drift,self.tfdtype)
        values_properties = tf.cast(values_properties,self.tfdtype)
        self.Number_surface_points = int(surface_points_coord.shape[0])


        self.surface_points = self.geo_data.surface_points
        self._surface_points = self.geo_data.surface_points
        self.surfaces = self.geo_data.surfaces
        self._surfaces =self.geo_data.surfaces 
        self.orientations = self.geo_data.orientations
        self._orientations = self.geo_data.orientations
        self.centers = self.geo_data.rescaling.df.loc['values', 'centers'].astype(self.dtype)

        # g = GravityPreprocessing(self.geo_data.grid.centered_grid)

        # # precomputed gravity impact from each grid
        # tz = g.set_tz_kernel()

        # self.tz = tf.cast(tz,self.tfdtype)

        len_rest_form = self.interpolator.additional_data.structure_data.df.loc[
            'values', 'len surfaces surface_points'] - 1
        Range = self.interpolator.additional_data.kriging_data.df.loc['values', 'range']
        C_o = self.interpolator.additional_data.kriging_data.df.loc['values', '$C_o$']
        rescale_factor = self.interpolator.additional_data.rescaling_data.df.loc[
            'values', 'rescaling factor']

        self.rf = rescale_factor
        nugget_effect_grad = np.cast[self.dtype](
            np.tile(self.interpolator.orientations.df['smooth'], 3))
        nugget_effect_scalar = np.cast[self.dtype](
            self.interpolator.surface_points.df['smooth'])
        # surface_points_coord = tf.Variable(surface_points_coord, dtype=self.tfdtype)

        self.dip_angles = tf.convert_to_tensor(dip_angles,dtype=self.tfdtype)
        self.azimuth = tf.convert_to_tensor(azimuth,dtype=self.tfdtype)
        self.polarity = tf.convert_to_tensor(polarity,dtype=self.tfdtype)

        self.fault_drift = tf.convert_to_tensor(fault_drift,dtype=self.tfdtype)
        self.grid_tensor = tf.convert_to_tensor(grid,dtype=self.tfdtype)

        self.values_properties = tf.constant(values_properties,dtype=self.tfdtype)

        self.len_rest_form = tf.convert_to_tensor(len_rest_form,dtype=self.tfdtype)
        self.Range = tf.convert_to_tensor(Range, self.tfdtype)
        self.C_o = tf.convert_to_tensor(C_o,dtype=self.tfdtype)
        self.nugget_effect_grad = tf.convert_to_tensor(nugget_effect_grad,dtype=self.tfdtype)
        self.nugget_effect_scalar = tf.convert_to_tensor(nugget_effect_scalar,dtype=self.tfdtype)
        self.rescale_factor = tf.convert_to_tensor(rescale_factor, self.tfdtype)
        
        self.surface_points_coord = tf.convert_to_tensor(surface_points_coord,self.tfdtype)
        
        self.solutions = Solution(grid,self.geo_data.surfaces,self.geo_data.series)
        
        self.lg_0 = self.interpolator.grid.get_grid_args('centered')[0]
        self.lg_1 = self.interpolator.grid.get_grid_args('centered')[1]


        self.resolution_ = tf.constant(self.geo_data.grid.regular_grid.resolution,tf.int32,name = 'Const_resolution')
        
    def activate_centered_grid(self,):
        self.geo_data.grid.deactivate_all_grids()
        self.geo_data.grid.set_active(['centered'])
        self.geo_data.update_from_grid()
        self.grid = self.geo_data.grid
        self.from_gempy_interpolator()
    
    def activate_regular_grid(self,):
        self.geo_data.grid.deactivate_all_grids()
        self.geo_data.grid.set_active(['regular'])
        # self.geo_data.grid.set_active(['sections'])
        self.geo_data.update_from_grid()
        self.grid = self.geo_data.grid
        
        self.from_gempy_interpolator()

        # rotate the grid along z axis, to keep consistency with kernel grid
        self.grid_tensor = tf.constant(self.revert_coordinates_alongz(self.grid_tensor.numpy()),dtype = self.tfdtype)

        

    def activate_customized_grid(self,grid_kernel):
        self.geo_data.grid.custom_grid=grid_kernel
        self.geo_data.grid.deactivate_all_grids()
        # activate also rescaled the grid
        self.geo_data.grid.set_active('custom')

        self.geo_data.update_from_grid()
        self.geo_data.rescaling.set_rescaled_grid()
        self.grid = self.geo_data.grid
        self.from_gempy_interpolator()

        

    def rescale_coord(self,surface_points_xyz):
        new_coord_surface_points = (surface_points_xyz -
                                    self.centers) / self.rf + 0.5001
        return new_coord_surface_points
    def unscale_coord(self,surface_points_xyz):
        unscale_coord_surface_points = (surface_points_xyz - 0.5001)*self.rf+self.centers
        return unscale_coord_surface_points
        
        
###########
    def set_tensorflow_shared_structure_surfaces(self):
        self.number_of_points_per_surface_T = (self.geo_data.additional_data.structure_data.df.loc[
                        'values', 'len surfaces surface_points'] - 1)
        number_of_points_per_surface_T = tf.cast(self.number_of_points_per_surface_T, dtype=tf.int32)
        npf = tf.cumsum(tf.concat([[0], number_of_points_per_surface_T[:-1]], -1))
        
        return number_of_points_per_surface_T,npf
    
    def set_tensorflow_shared_loop(self):
        """Set the theano shared variables that are looped for each series."""
        self._compute_len_series()

        len_series_i = np.insert(self.len_series_i.cumsum(), 0, 0).astype('int32')
        len_series_o = np.insert(self.len_series_o.cumsum(), 0, 0).astype('int32')
        len_series_w = np.insert(self.len_series_w.cumsum(), 0, 0).astype('int32')

        # Number of surfaces per series. The function is not pretty but the result is quite clear
        n_surfaces_per_serie = np.insert(
            self.additional_data.structure_data.df.loc['values', 'number surfaces per series'][
                self.non_zero].cumsum(), 0, 0). \
            astype('int32')
        n_surfaces_per_series = n_surfaces_per_serie
        n_universal_eq_T = (
            list(self.additional_data.kriging_data.df.loc['values', 'drift equations'].astype(
                'int32')[self.non_zero]))
        
        return len_series_i,len_series_o,len_series_w,n_surfaces_per_series,n_universal_eq_T
        
    def _compute_len_series(self):

        self.len_series_i = self.additional_data.structure_data.df.loc[
                                'values', 'len series surface_points'] - \
                            self.additional_data.structure_data.df.loc[
                                'values', 'number surfaces per series']

        self.len_series_o = self.additional_data.structure_data.df.loc[
            'values', 'len series orientations'].astype(
            'int32')

        # Remove series without data
        non_zero_i = self.len_series_i.nonzero()[0]
        non_zero_o = self.len_series_o.nonzero()[0]
        non_zero = np.intersect1d(non_zero_i, non_zero_o)

        self.non_zero = non_zero

        self.len_series_u = self.additional_data.kriging_data.df.loc[
            'values', 'drift equations'].astype('int32')
        try:
            len_series_f_ = self.faults.faults_relations_df.values[non_zero][:, non_zero].sum(
                axis=0)

        except np.AxisError:
            print('np.axis error')
            len_series_f_ = self.faults.faults_relations_df.values.sum(axis=0)

        self.len_series_f = np.atleast_1d(len_series_f_.astype(
            'int32'))  # [:self.additional_data.get_additional_data()['values']['Structure', 'number series']]

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

    def remove_series_without_data(self):
        len_series_i = self.additional_data.structure_data.df.loc[
                           'values', 'len series surface_points'] - \
                       self.additional_data.structure_data.df.loc[
                           'values', 'number surfaces per series']

        len_series_o = self.additional_data.structure_data.df.loc[
            'values', 'len series orientations'].astype(
            'int32')

        # Remove series without data
        non_zero_i = len_series_i.nonzero()[0]
        non_zero_o = len_series_o.nonzero()[0]
        non_zero = np.intersect1d(non_zero_i, non_zero_o)

        self.non_zero = non_zero
        return self.non_zero

    def set_tensorflow_shared_fault_relation(self):
        """Set the tensorflow shared variable with the fault relation"""
        self.remove_series_without_data()
        fault_relation = self.faults.faults_relations_df.values[self.non_zero][:, self.non_zero]
        return fault_relation

    def set_tensorflow_shared_is_fault(self):
        """Set tensorflow shared variable which controls if a series is fault or not"""
        is_fault = self.faults.df['isFault'].values[self.non_zero]
        return is_fault

    def set_tensorflow_shared_is_finite(self):
        """Set tensorflow shared variable which controls if a fault is finite or not
        George: finite fault is not yet implemented """
        pass    
    
    def set_tensorflow_shared_onlap_erode(self):
        """Set the theano variables which control the masking patterns according to the uncomformity relation"""
        self.remove_series_without_data()

        is_erosion = self.series.df['BottomRelation'].values[self.non_zero] == 'Erosion'
        is_onlap = np.roll(self.series.df['BottomRelation'].values[self.non_zero] == 'Onlap', 1)

        if len(is_erosion) != 0:
            is_erosion[-1] = False
        # this comes from the series df
        # self.TFG.is_erosion = is_erosion
        # self.TFG.is_onlap = is_onlap
        return is_erosion,is_onlap
    
    # def set_tensorflow_shared_faults(self):
    #     fault_relation = self.set_tensorflow_shared_fault_relation()
    #     # This comes from the faults df
    #     is_fault = self.set_tensorflow_shared_is_fault()
    #     # self.set_tensorflow_shared_is_finite()
        
    def reset_flow_control_initial_results(self, reset_weights=True, reset_scalar=True,
                                           reset_block=True):
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
        n_series = self.len_series_i.shape[0]
        x_to_interp_shape = self.interpolator.grid.values_r.shape[0] + 2 * self.len_series_i.sum()

        if reset_weights is True:
            self.compute_weights_ctrl = np.ones(1000, dtype=bool)
            weights_vector = (
                np.zeros((self.len_series_w.sum()), dtype=self.dtype))

        if reset_scalar is True:
            self.compute_scalar_ctrl = np.ones(1000, dtype=bool)
            scalar_fields_matrix = (
                np.zeros((n_series, x_to_interp_shape), dtype=self.dtype))

        if reset_block is True:
            self.compute_block_ctrl = np.ones(1000, dtype=bool)
            mask_matrix = (
                np.zeros((n_series, x_to_interp_shape), dtype='bool'))
            block_matrix = (
                np.zeros((n_series,
                          self.surfaces.df.iloc[:, self.surfaces._n_properties:].values.shape[1],
                          x_to_interp_shape), dtype=self.dtype))
        return weights_vector,scalar_fields_matrix,mask_matrix,block_matrix
    


    def get_graph_input(self):
        number_of_points_per_surface_T,npf = self.set_tensorflow_shared_structure_surfaces()
        len_series_i,len_series_o,len_series_w,n_surfaces_per_series,n_universal_eq_T = self.set_tensorflow_shared_loop()
        is_erosion,is_onlap = self.set_tensorflow_shared_onlap_erode()
        weights_vector,scalar_fields_matrix,mask_matrix,block_matrix = self.reset_flow_control_initial_results()
        is_fault = self.set_tensorflow_shared_is_fault()
        fault_relation  = self.set_tensorflow_shared_fault_relation()
        
        # convert to tensor
        len_series_i = tf.convert_to_tensor(len_series_i,tf.int32)
        len_series_o = tf.convert_to_tensor(len_series_o,tf.int32)
        len_series_w = tf.convert_to_tensor(len_series_w,tf.int32)
        n_surfaces_per_series = tf.convert_to_tensor(n_surfaces_per_series,tf.int32)
        n_universal_eq_T = tf.convert_to_tensor(n_universal_eq_T,tf.int32)
        ## tensorflow input
        is_erosion = tf.convert_to_tensor(is_erosion,tf.bool)
        is_onlap = tf.convert_to_tensor(is_onlap,tf.bool)
        mask_matrix = tf.convert_to_tensor(mask_matrix,tf.bool)
        ## numpy input
        # is_erosion = is_erosion.tolist()
        # is_onlap = is_onlap.tolist()
        # mask_matrix = mask_matrix.tolist()
        is_fault = tf.convert_to_tensor(is_fault,tf.bool)
        weights_vector = tf.convert_to_tensor(weights_vector,self.tfdtype)
        fault_relation = tf.convert_to_tensor(fault_relation,tf.bool)
        scalar_fields_matrix = tf.convert_to_tensor(scalar_fields_matrix,self.tfdtype)
        block_matrix = tf.convert_to_tensor(block_matrix,self.tfdtype)


        return [number_of_points_per_surface_T,
                npf,
                len_series_i,
                len_series_o,
                len_series_w,
                n_surfaces_per_series,
                n_universal_eq_T,
                is_erosion,
                is_onlap,
                is_fault,
                weights_vector,
                fault_relation,
                scalar_fields_matrix,
                mask_matrix,
                block_matrix]
    
    def create_tensorflow_graph(self, input, delta = 100000, gradient = False,compute_gravity = False,matrix_size = None,max_slope = None):
        '''
            'matrix_size': specify the operating matrix size, if None, the Tensorflow will infer the size dynamically
            'delta': controls the magnitude of the slope, [-inf,inf] -> [0,max_slope]
            'max_slope': Maximum slope to allow the gradient to be kept during the learning  
        '''
        self.TFG = TFGraph(input, self.fault_drift,
                self.grid_tensor, self.values_properties, self.nugget_effect_grad,self.nugget_effect_scalar, self.Range,
                self.C_o, self.rescale_factor,delta_slope = delta, dtype = self.tfdtype, gradient = gradient,compute_gravity = compute_gravity,
                matrix_size = matrix_size,max_slope = max_slope)
    
    # def calculate_grav(self,surface_coord, values_properties):
    
    def prepare_input(self,gradient = False,surface_points = None):
        # self.activate_regular_grid()

        gpinput = self.get_graph_input()
        number_of_points_per_surface = gpinput[0]

        # self.surface_points.sort_table()
        # self.orientations.sort_table()

        if surface_points is None:
            self.surface_points_ = self.surface_points_coord
        else:
            self.surface_points_ = surface_points
        # get the concrete size of the matrix
        self.matrix_size = self.grid_tensor.shape[0]+2*(tf.shape(self.surface_points_)[0] - \
            tf.shape(number_of_points_per_surface)[0]).numpy()
        # self.create_tensorflow_graph(gpinput,gradient = gradient,slope = 500000,matrix_size = self.matrix_size)
        

    def compute_model(self,surface_points = None,dip_angles = None,gradient = False):
        self.prepare_input(gradient,surface_points)
        if surface_points is None:
            surface_points = self.surface_points_coord
        if dip_angles is None:
            dip_angles = self.dip_angles
        final_block,final_property,block_matrix,Z_x,sfai,block_mask,fault_matrix = self.TFG.compute_series(surface_points,
                    self.dips_position,
                    dip_angles,
                    self.azimuth,
                    self.polarity,)
        
        self.grid = self._grid
        size = tf.reduce_prod(self.resolution_,name = 'reduce_prod_size_')
        sol = final_block,final_property,block_matrix,block_mask,size,Z_x,sfai
        self.set_solutions(sol)

    # @tf.function
    def compute_gravity(self,tz,receivers,g = None,kernel = None,surface_points = None,dip_angles = None,gradient = False,Hessian = False,method = None,window_resolution = None,grav_only = False,LOOP_FLAG = True, values_properties = None):
        implemented_methods_lst = ['conv_all','kernel_reg','kernel_geom','kernel_ml']
        if method in implemented_methods_lst: 
            pass   
        else: 
            raise NotImplementedError

        if surface_points is None:
            surface_points = self.surface_points_coord
        if dip_angles is None:
            dip_angles = self.dip_angles
        if method == 'conv_all':

            size = tf.reduce_prod(self.resolution_,name = 'reduce_prod_size_')

            final_block,final_property,block_matrix,Z_x,sfai,block_mask,fault_matrix = self.TFG.compute_series(surface_points,
                        self.dips_position,
                        dip_angles,
                        self.azimuth,
                        self.polarity,
                        values_properties = values_properties)

            # densities = final_property[0:size] # slice the density by resolution
            densities = tf.strided_slice(final_property,[0],[size],[1],name = 'ss_w_den_') # This fix the 'Const_4' int64_val : 1 when print the graph

            # Flip the value matrix along Z axis
            densities = self.revert_value_alongz(densities)

            # Find the receiver position at the model coordinates
            center_index_x = tf.constant((g.new_xy_ravel[0]-receivers.extent[0])//g.dx,self.tfdtype,name = 'center_index_x')
            center_index_y = tf.constant((g.new_xy_ravel[1]-receivers.extent[2])//g.dy,self.tfdtype,name = 'center_index_y')
            grav_convolution_full = tf.TensorArray(self.tfdtype, size=receivers.n_devices, dynamic_size=False, clear_after_read=True)
            for i in tf.range(receivers.n_devices):
            # i = tf.constant(0,name = 'i_')
                c_x = tf.cast(center_index_x[i],tf.int32,name = 'c_x')
                c_y = tf.cast(center_index_y[i],tf.int32,name = 'c_y')
                

                ## Calculate the gravity of each receiver
                # windowed_densities = tf.reshape(densities,self.geo_data.grid.regular_grid.resolution)[c_x-g.radius_cell_x:c_x+g.radius_cell_x+tf.constant(1,name = 'windowed_densities_1'),c_y-g.radius_cell_y:c_y+g.radius_cell_y+tf.constant(1,name = 'windowed_densities_2'),:]
                windowed_densities = tf.reshape(densities,self.resolution_)
                windowed_densities = tf.strided_slice(windowed_densities,[c_x-g.radius_cell_x,c_y-g.radius_cell_y,0 ],[c_x+g.radius_cell_x+tf.constant(1,name = 'windowed_densities_1'),c_y+g.radius_cell_y+tf.constant(1,name = 'windowed_densities_2'),self.resolution_[2]],[1,1,1],name = 'ss_w_den_1')
                windowed_densities = tf.squeeze(tf.reshape(windowed_densities,[-1,1]))
                grav_ = self.TFG.compute_forward_gravity(tz, 0, size, windowed_densities)
                grav_convolution_full = grav_convolution_full.write(i, grav_)
            grav = tf.squeeze(grav_convolution_full.stack())
            


        if method == 'kernel_reg':
            # The performance in backpropogation differs a lot with vectorized and iterative computation. Here the iterative computation is used to generate High resolution data in CPU. 
            if LOOP_FLAG == False:
                # Vectorized computation
                final_block,final_property,block_matrix,Z_x,sfai,block_mask,fault_matrix = self.TFG.compute_series(surface_points,
                            self.dips_position,
                            dip_angles,
                            self.azimuth,
                            self.polarity,
                            values_properties = values_properties)
                size = kernel.values.shape[0]
                densities = final_property[:size]

                grav = self.TFG.compute_forward_gravity(tz, 0, size, densities)
            else:
                # Iterative computation
                grav = tf.TensorArray(self.tfdtype, size=receivers.n_devices, dynamic_size=False, clear_after_read=True)
                for i in tf.range(receivers.n_devices):
                    final_block,final_property,block_matrix,Z_x,sfai,block_mask,fault_matrix = self.TFG.compute_series(surface_points,
                            self.dips_position,
                            dip_angles,
                            self.azimuth,
                            self.polarity,)
                    size = tz.shape[0]
                    windowed_densities = final_property[i*size:(i+1)*size]
                    grav_ = self.TFG.compute_forward_gravity(tz, 0, size, windowed_densities)
                    grav = grav.write(i, grav_)
                grav = tf.squeeze(grav.stack())


        
        if grav_only == True:
            return grav
        else:
            return final_block,final_property,block_matrix,block_mask,size,Z_x,sfai,grav
    
    def revert_coordinates_alongz(self,values):
        value_dim = 3
        values = values.reshape(list(self.resolution)+[value_dim]) 
        values = np.flip(values, value_dim-1)
        values = values.reshape([-1,value_dim])
        return values

    def revert_value_alongz(self,values):
        if len(values.shape)>1:
            value_dim = values.shape[0]
            values = values.reshape([value_dim]+list(self.resolution)) 
            values = np.flip(values, 2)
            values = values.reshape([value_dim,-1])
        else:
            value_dim = 0
            # values = values.reshape(list(self.resolution)) 
            # values = np.flip(values, 2)
            # values = values.reshape([-1])
            values = tf.reshape(values,self.resolution)
            values = tf.reverse(values, [2])
            values = tf.reshape(values,[-1])
        return values

    def set_solutions(self,sol):
        # final_block,final_property,block_matrix,block_mask,size,Z_x,sfai,grav = sol
        ## unzip the solutions



        final_block  = sol[0]
        final_property = sol[1]
        block_matrix = sol[2]
        block_mask = sol[3]
        size = sol[4]
        Z_x = sol[5]
        sfai = sol[6]

        self.grid = self._grid
        self.grid.regular_grid.values = self.revert_coordinates_alongz(self.grid.regular_grid.values)
        self.grid.regular_grid.values_r = self.revert_coordinates_alongz(self.grid.regular_grid.values_r)
        # self._grid.regular_grid.values = self.revert_coordinates_alongz(self._grid.regular_grid.values)
        # self._grid.regular_grid.values_r = self.revert_coordinates_alongz(self._grid.regular_grid.values_r)
        if self._grid.active_grids[0] == True:
            regular = self._grid.get_grid_args('regular')
            lith = (sol[0][regular[0]:regular[1]]).numpy().astype(int)
            # lith = self.revert_value_alongz(lith)
            self.solutions.lith_block = (lith)

        
        self.solutions.values_matrix = final_property[:size].numpy()
        self.solutions.scalar_field_matrix = Z_x[:,regular[0]:regular[1]].numpy()
        # self.solutions.values_matrix = self.revert_value_alongz(final_property[:size].numpy())
        # self.solutions.scalar_field_matrix = self.revert_value_alongz(Z_x[:,regular[0]:regular[1]].numpy())
        self.solutions.mask_matrix = block_mask[:,regular[0]:regular[1]].numpy()>0
        self.solutions.scalar_field_at_surface_points = sfai.numpy()
        self.solutions._grid = self.grid
        self.solutions.grid = self.grid
        self.solutions.block_matrix = block_matrix[:,regular[0]:regular[1]].numpy()
        
        l0, l1 = self.solutions.grid.get_grid_args('sections')
        # print('formation_block',formation_block[l0: l1])
        # print('scalar_matrix',self.TFG.scalar_matrix[:, l0: l1])
        self.solutions.sections = np.array(
            [final_block[l0: l1].numpy(), Z_x[:, l0: l1].numpy().astype(float)])
        
        self.solutions.compute_all_surfaces()
        self.set_surface_order_from_solution()


    def set_surface_order_from_solution(self):
        """
        Order the surfaces respect the last computation. Therefore if you call this method,
        after sorting surface_points without recomputing you may get wrong results.

        Returns:
            Surfaces
        """
        sfai_order = self.solutions.scalar_field_at_surface_points.sum(axis=0)
        sel = self.surfaces.df['isActive'] & ~self.surfaces.df['isBasement']
        self.surfaces.df.loc[sel, 'sfai'] = sfai_order
        self.surfaces.df.sort_values(by=['series', 'sfai'], inplace=True, ascending=False)
        self.surfaces.reset_order_surfaces()
        self.surfaces.sort_surfaces()
        self.surfaces.set_basement()
        self.surface_points.df['id'] = self.surface_points.df['surface'].map(
            self.surfaces.df.set_index('surface')['id']).astype(int)
        self.orientations.df['id'] = self.orientations.df['surface'].map(
            self.surfaces.df.set_index('surface')['id']).astype(int)
        self.surface_points.sort_table()
        self.orientations.sort_table()
        self.update_structure()
        return self.surfaces