class TorchGraph():

    def __init__(self, input,
                 fault_drift, grid, values_properties,nugget_effect_grad,nugget_effect_scalar,
                Range, C_o, rescalefactor,delta_slope=50, output=None,sigmoid=False,
                 compute_gravity = False,matrix_size = None, **kwargs):
        
        (self.number_of_points_per_surface_T,
        self.npf,
        self.len_series_i,
        self.len_series_o,
        self.len_series_w,
        self.n_surfaces_per_series,
        self.n_universal_eq_T,
        self.is_erosion,
        self.is_onlap,
        self.is_fault,
        self.weights_vector,
        self.fault_relation,
        self.scalar_fields_matrix,
        self.mask_matrix,
        self.block_matrix) = input
        
        self.matrix_size = matrix_size
        self.compute_gravity_flag = compute_gravity
        self.dtype = kwargs.get('dtype', torch.float32)
        self.lengh_of_faults = torch.tensor(0, dtype=torch.int32)

        # OPETIONS
        # -------
        self.gradient = kwargs.get('gradient', True)
        self.gravity = kwargs.get('gravity', False)
        if kwargs.get('max_slope'):
            self.max_slope = torch.tensor(kwargs.get('max_slope'),dtype=self.dtype)
        else: self.max_slope = 50

        # CONSTANT PARAMETERS FOR ALL SERIES
        # KRIGING
        # -------
        self.a_T = torch.divide(Range, rescalefactor)
        self.a_T_surface = self.a_T
        self.c_o_T = torch.divide(C_o, rescalefactor)

        self.n_universal_eq_T_op = torch.tensor(3)  # 9 for 2nd order drift

        # They weight the contribution of the surface_points against the orientations.
        self.i_rescale = torch.tensor(4., dtype=self.dtype)
        self.gi_rescale = torch.tensor(2., dtype=self.dtype)

        # Number of dimensions. Now it is not too variable anymore
        self.n_dimensions = 3

        self.nugget_effect_grad_T = nugget_effect_grad
        self.nugget_effect_scalar = nugget_effect_scalar

        self.sigmoid = self.gradient
        if self.sigmoid == False:
          self.mask_dtype = torch.int32
        else:
          self.mask_dtype = self.dtype
        # COMPUTE WEIGHTS
        # ---------
        # VARIABLES DEFINITION
        # ---------
        if self.gradient:

            self.delta_slope = torch.Variable(delta_slope,dtype = self.dtype)
            
        else:
            self.delta_slope = torch.tensor(delta_slope,dtype = self.dtype)


        self.number_of_series = torch.tensor(1,dtype = torch.int32)

        # self.values_properties_op = values_properties
        if values_properties.shape[0] > 1:
            self.densities = torch.Variable(values_properties[1],dtype=self.dtype)
            self.lith_label = torch.tensor(values_properties[0],dtype=self.dtype)
            self.value_properties = torch.stack([self.lith_label,self.densities],axis = 0)
        else: 
            self.value_properties = torch.tensor(values_properties,dtype=self.dtype)

        self.grid_val = grid



        if output is None:
            output = ['geology']
        self.output = output

        self.compute_type = output


        self.offset = torch.tensor(10., dtype=self.dtype)
        self.shift = 0

        if 'gravity' in self.compute_type:
            self.lg0 = torch.tensor(0, dtype=torch.int32)
            self.lg1 = torch.tensor(1, dtype=torch.int32)

            self.tz =  torch.zeros((0,1),dtype=torch.int32)
            self.pos_density = torch.tensor(1, dtype=torch.int32)

        # self.n_surface = range(1, 5000)
        ##
        
        ### Some place holders for interpolator
        # self.number_of_points_per_surface_T = torch.zeros(3,dtype = self.dtype)
        # self.npf =  torch.zeros(3,dtype = self.dtype)
        # self.is_erosion = torch.tensor([1,0])
        # self.is_onlap =torch.tensor([0,1])
        # self.n_surfaces_per_series = torch.tensor([3,3])
        # self.n_universal_eq_T = torch.tensor([3,3])
    
    def update_property(self):
        if self.value_properties.shape[0] > 1:
            self.value_properties = torch.stack([self.lith_label,self.densities],axis = 0)

    #@torch.function
    def set_rest_ref_matrix(self, number_of_points_per_surface, surface_points_all, nugget_effect_scalar):
        # reference point: every first point of each layer
        ref_positions = torch.cumsum(
            torch.concat([[0], number_of_points_per_surface[:-1] + torch.tensor(1, dtype=torch.int32)], axis=0))

        partitions = torch.reduce_sum(torch.one_hot(ref_positions,torch.reduce_sum(number_of_points_per_surface+1),dtype = torch.int32),axis=0)
        # reference:1 rest: 0

        ## selecting surface points as rest set and reference set
        rest_points = torch.dynamic_partition(surface_points_all, partitions, 2)[0]
        ref_points = torch.dynamic_partition(surface_points_all, partitions, 2)[1]
        
        
        rest_nugget = torch.dynamic_partition(nugget_effect_scalar, partitions, 2)[0]
        ref_nugget = torch.dynamic_partition(nugget_effect_scalar, partitions, 2)[1]
        
        # repeat the reference points (the number of persurface -1)  times
        ref_points_repeated = torch.repeat(
            ref_points, number_of_points_per_surface, 0)
        ref_nugget_repeated = torch.repeat(
            ref_nugget, number_of_points_per_surface, 0)

        # the naming here is a bit confusing, because the reference points are repeated to match up with the rest of the points
        return ref_points_repeated, rest_points, ref_nugget_repeated, rest_nugget

    
    #@torch.function
    def squared_euclidean_distance(self, x_1, x_2):
        """
        Compute the euclidian distances in 3D between all the points in x_1 and x_2

        Arguments:
            x_1 {[Tensor]} -- shape n_points x number dimension
            x_2 {[Tensor]} -- shape n_points x number dimension

        Returns:
            [Tensor] -- Distancse matrix. shape n_points x n_points
        """
        # torch.maximum avoid negative numbers increasing stability

        sqd = torch.sqrt(torch.maximum(torch.reshape(torch.reduce_sum(x_1**2, 1), shape=(torch.shape(x_1)[0], 1)) +
                                 torch.reshape(torch.reduce_sum(x_2**2, 1), shape=(1, torch.shape(x_2)[0])) -
                                 2 * torch.tensordot(x_1, torch.transpose(x_2), 1), torch.tensor(1e-12, dtype=self.dtype)))

        return sqd

    #@torch.function
    def matrices_shapes(self):
        """
        Get all the lengths of the matrices that form the covariance matrix

        Returns:
             length_of_CG, length_of_CGI, length_of_U_I, length_of_faults, length_of_C
        """
        length_of_CG = torch.shape(self.dips_position_tiled)[0]
        length_of_CGI = torch.shape(self.ref_layer_points)[0]
        length_of_U_I = self.n_universal_eq_T_op
        length_of_faults = self.lengh_of_faults

        length_of_C = length_of_CG + length_of_CGI + length_of_U_I + length_of_faults

        return length_of_CG, length_of_CGI, length_of_U_I, length_of_faults, length_of_C

    #@torch.function
    def cov_surface_points(self, ref_layer_points, rest_layer_points):

        sed_rest_rest = self.squared_euclidean_distance(
            rest_layer_points, rest_layer_points)
        sed_ref_rest = self.squared_euclidean_distance(
            ref_layer_points, rest_layer_points)
        sed_rest_ref = self.squared_euclidean_distance(
            rest_layer_points, ref_layer_points)
        sed_ref_ref = self.squared_euclidean_distance(
            ref_layer_points, ref_layer_points)

        C_I = self.c_o_T * self.i_rescale * (
            torch.where(sed_rest_rest < self.a_T, x=(1 - 7 * (sed_rest_rest / self.a_T) ** 2 +
                                                  35 / 4 * (sed_rest_rest / self.a_T) ** 3 -
                                                  7 / 2 * (sed_rest_rest / self.a_T) ** 5 +
                                                  3 / 4 * (sed_rest_rest / self.a_T) ** 7), y=0) -
            torch.where(sed_ref_rest < self.a_T, x=(1 - 7 * (sed_ref_rest / self.a_T) ** 2 +
                                                 35 / 4 * (sed_ref_rest / self.a_T) ** 3 -
                                                 7 / 2 * (sed_ref_rest / self.a_T) ** 5 +
                                                 3 / 4 * (sed_ref_rest / self.a_T) ** 7), y=0) -
            torch.where(sed_rest_ref < self.a_T, x=(1 - 7 * (sed_rest_ref / self.a_T) ** 2 +
                                                 35 / 4 * (sed_rest_ref / self.a_T) ** 3 -
                                                 7 / 2 * (sed_rest_ref / self.a_T) ** 5 +
                                                 3 / 4 * (sed_rest_ref / self.a_T) ** 7), y=0) +
            torch.where(sed_ref_ref < self.a_T, x=(1 - 7 * (sed_ref_ref / self.a_T) ** 2 +
                                                35 / 4 * (sed_ref_ref / self.a_T) ** 3 -
                                                7 / 2 * (sed_ref_ref / self.a_T) ** 5 +
                                                3 / 4 * (sed_ref_ref / self.a_T) ** 7), y=0))

        C_I = C_I + torch.eye(torch.shape(C_I)[0], dtype=self.dtype) * \
            self.nugget_effect_scalar_T_op
        return C_I

    #@torch.function
    def cov_gradients(self, dips_position):
        dips_position_tiled = torch.tile(
            dips_position, [self.n_dimensions, 1])

        sed_dips_dips = self.squared_euclidean_distance(
            dips_position_tiled, dips_position_tiled)

        h_u = torch.concat([
            torch.tile(dips_position[:, 0] - torch.reshape(
                dips_position[:, 0], [torch.shape(dips_position)[0], 1]), [1, 3]),
            torch.tile(dips_position[:, 1] - torch.reshape(
                dips_position[:, 1], [torch.shape(dips_position)[0], 1]), [1, 3]),
            torch.tile(dips_position[:, 2] - torch.reshape(dips_position[:, 2], [torch.shape(dips_position)[0], 1]), [1, 3])], axis=0)

        h_v = torch.transpose(h_u)

        sub_x = torch.concat([torch.ones([torch.shape(dips_position)[0], torch.shape(dips_position)[0]]), torch.zeros(
            [torch.shape(dips_position)[0], 2 * torch.shape(dips_position)[0]])], axis=1)

        sub_y = torch.concat([torch.concat([torch.zeros([torch.shape(dips_position)[0], torch.shape(dips_position)[0]]), torch.ones(
            [torch.shape(dips_position)[0], 1 * torch.shape(dips_position)[0]])], axis=1), torch.zeros([torch.shape(dips_position)[0], torch.shape(dips_position)[0]])], 1)
        sub_z = torch.concat([torch.zeros([torch.shape(dips_position)[0], 2 * torch.shape(dips_position)[0]]),
                           torch.ones([torch.shape(dips_position)[0], torch.shape(dips_position)[0]])], axis=1)

        perpendicularity_matrix = torch.concat([sub_x, sub_y, sub_z], axis=0)

        condistion_fail = torch.math.divide_no_nan(h_u * h_v, sed_dips_dips ** 2) * (
            torch.where(sed_dips_dips < self.a_T,
                     x=(((-self.c_o_T * ((-14. / self.a_T ** 2.) + 105. / 4. * sed_dips_dips / self.a_T ** 3. -
                                         35. / 2. * sed_dips_dips ** 3. / self.a_T ** 5. +
                                         21. / 4. * sed_dips_dips ** 5. / self.a_T ** 7.))) +
                        self.c_o_T * 7. * (9. * sed_dips_dips ** 5. - 20. * self.a_T ** 2. * sed_dips_dips ** 3. +
                                           15. * self.a_T ** 4. * sed_dips_dips - 4. * self.a_T ** 5.) / (2. * self.a_T ** 7.)), y=0.)) -\
            perpendicularity_matrix * torch.where(sed_dips_dips < self.a_T, x=self.c_o_T * ((-14. / self.a_T ** 2.) + 105. / 4. * sed_dips_dips / self.a_T ** 3. -
                                                                                         35. / 2. * sed_dips_dips ** 3. / self.a_T ** 5. +
                                                                                         21. / 4. * sed_dips_dips ** 5. / self.a_T ** 7.), y=0.)

        C_G = torch.where(sed_dips_dips == 0, x=torch.tensor(
            0., dtype=self.dtype), y=condistion_fail)
        C_G = C_G + torch.eye(torch.shape(C_G)[0],
                           dtype=self.dtype) * self.nugget_effect_grad_T_op

        return C_G

    #@torch.function
    def cartesian_dist(self, x_1, x_2):
        return torch.concat([
            torch.transpose(
                (x_1[:, 0] - torch.expand_dims(x_2[:, 0], axis=1))),
            torch.transpose(
                (x_1[:, 1] - torch.expand_dims(x_2[:, 1], axis=1))),
            torch.transpose(
                (x_1[:, 2] - torch.expand_dims(x_2[:, 2], axis=1)))], axis=0)

    #@torch.function
    def cov_interface_gradients(self, dips_position_all, rest_layer_points, ref_layer_points):
        dips_position_all_tiled = torch.tile(
            dips_position_all, [self.n_dimensions, 1])

        sed_dips_rest = self.squared_euclidean_distance(
            dips_position_all_tiled, rest_layer_points)
        sed_dips_ref = self.squared_euclidean_distance(
            dips_position_all_tiled, ref_layer_points)

        hu_rest = self.cartesian_dist(
            dips_position_all, rest_layer_points)
        hu_ref = self.cartesian_dist(
            dips_position_all, ref_layer_points)

        C_GI = self.gi_rescale * torch.transpose(hu_rest *
                                               torch.where(sed_dips_rest < self.a_T_surface, x=(- self.c_o_T * ((-14 / self.a_T_surface ** 2) + 105 / 4 * sed_dips_rest / self.a_T_surface ** 3 -
                                                                                                             35 / 2 * sed_dips_rest ** 3 / self.a_T_surface ** 5 +
                                                                                                             21 / 4 * sed_dips_rest ** 5 / self.a_T_surface ** 7)), y=torch.tensor(0., dtype=self.dtype)) -
                                               (hu_ref * torch.where(sed_dips_ref < self.a_T_surface, x=- self.c_o_T * ((-14 / self.a_T_surface ** 2) + 105 / 4 * sed_dips_ref / self.a_T_surface ** 3 -
                                                                                                                     35 / 2 * sed_dips_ref ** 3 / self.a_T_surface ** 5 +
                                                                                                                     21 / 4 * sed_dips_ref ** 5 / self.a_T_surface ** 7), y=torch.tensor(0., dtype=self.dtype))))

        return C_GI

    #@torch.function
    def universal_matrix(self, dips_position_all, ref_layer_points, rest_layer_points):

        n = torch.shape(dips_position_all)[0]

        sub_x = torch.tile(torch.tensor([[1., 0., 0.]], self.dtype), [n, 1])
        sub_y = torch.tile(torch.tensor([[0., 1., 0.]], self.dtype), [n, 1])
        sub_z = torch.tile(torch.tensor([[0., 0., 1.]], self.dtype), [n, 1])
        sub_block1 = torch.concat([sub_x, sub_y, sub_z], 0)

        sub_x_2 = torch.reshape(2 * self.gi_rescale *
                             dips_position_all[:, 0], [n, 1])
        sub_y_2 = torch.reshape(2 * self.gi_rescale *
                             dips_position_all[:, 1], [n, 1])
        sub_z_2 = torch.reshape(2 * self.gi_rescale *
                             dips_position_all[:, 2], [n, 1])

        sub_x_2 = torch.pad(sub_x_2, [[0, 0], [0, 2]])
        sub_y_2 = torch.pad(sub_y_2, [[0, 0], [1, 1]])
        sub_z_2 = torch.pad(sub_z_2, [[0, 0], [2, 0]])
        sub_block2 = torch.concat([sub_x_2, sub_y_2, sub_z_2], 0)

        sub_xy = torch.reshape(torch.concat([self.gi_rescale * dips_position_all[:, 1],
                                       self.gi_rescale * dips_position_all[:, 0]], 0), [2 * n, 1])
        sub_xy = torch.pad(sub_xy, [[0, n], [0, 0]])
        sub_xz = torch.concat([torch.pad(torch.reshape(self.gi_rescale * dips_position_all[:, 2], [n, 1]), [
            [0, n], [0, 0]]), torch.reshape(self.gi_rescale * dips_position_all[:, 0], [n, 1])], 0)
        sub_yz = torch.reshape(torch.concat([self.gi_rescale * dips_position_all[:, 2],
                                       self.gi_rescale * dips_position_all[:, 1]], 0), [2 * n, 1])
        sub_yz = torch.pad(sub_yz, [[n, 0], [0, 0]])

        sub_block3 = torch.concat([sub_xy, sub_xz, sub_yz], 1)

        U_G = torch.concat([sub_block1, sub_block2, sub_block3], 1)

        U_I = -torch.stack([self.gi_rescale * (rest_layer_points[:, 0] - ref_layer_points[:, 0]), self.gi_rescale *
                         (rest_layer_points[:, 1] -
                          ref_layer_points[:, 1]),
                         self.gi_rescale *
                         (rest_layer_points[:, 2] -
                          ref_layer_points[:, 2]),
                         self.gi_rescale ** 2 *
                         (rest_layer_points[:, 0] ** 2 -
                          ref_layer_points[:, 0] ** 2),
                         self.gi_rescale ** 2 *
                         (rest_layer_points[:, 1] ** 2 -
                          ref_layer_points[:, 1] ** 2),
                         self.gi_rescale ** 2 *
                         (rest_layer_points[:, 2] ** 2 -
                          ref_layer_points[:, 2] ** 2),
                         self.gi_rescale ** 2 * (
            rest_layer_points[:, 0] * rest_layer_points[:, 1] - ref_layer_points[:, 0] *
            ref_layer_points[:, 1]),
            self.gi_rescale ** 2 * (
            rest_layer_points[:, 0] * rest_layer_points[:, 2] - ref_layer_points[:, 0] *
            ref_layer_points[:, 2]),
            self.gi_rescale ** 2 * (
            rest_layer_points[:, 1] * rest_layer_points[:, 2] - ref_layer_points[:, 1] *
            ref_layer_points[:, 2])], 1)

        return U_G, U_I

    #@torch.function
    def faults_matrix(self, f_ref=None, f_res=None):
        length_of_CG, _, _, length_of_faults = self.matrices_shapes()[
            :4]

        F_I = (self.fault_drift_at_surface_points_ref -
               self.fault_drift_at_surface_points_rest) + 0.0001

        F_G = torch.zeros((length_of_faults, length_of_CG),
                       dtype=self.dtype) + 0.0001

        return F_I, F_G

    #@torch.function
    def covariance_matrix(self, dips_position_all, ref_layer_points, rest_layer_points):

        length_of_CG, length_of_CGI, length_of_U_I, length_of_faults, length_of_C = self.matrices_shapes()

        C_G = self.cov_gradients(dips_position_all)
        C_I = self.cov_surface_points(ref_layer_points, rest_layer_points)
        C_GI = self.cov_interface_gradients(
            dips_position_all, rest_layer_points, ref_layer_points)
        U_G, U_I = self.universal_matrix(
            dips_position_all, ref_layer_points, rest_layer_points)
        U_G = U_G[:length_of_CG, :3]
        U_I = U_I[:length_of_CGI, :3]
        F_I, F_G = self.faults_matrix()

        # A: containing all the covariance matrix 
        A = torch.concat([torch.concat([C_G, torch.transpose(C_GI)], -1),
                       torch.concat([C_GI, C_I], -1)], 0)

        # B: Universal term
        B = torch.concat([U_G, U_I], 0)

        AB = torch.concat([A, B], -1)

        B_T = torch.transpose(B)

        if torch.shape(F_I)[0] is not None:
          paddings = torch.stack([[0,0],[0,3+torch.shape(F_I)[0]]],axis = 1)
        else:
          paddings = torch.tensor([[0, 0], [0, 3]])
        C = torch.pad(B_T, paddings)

        F = torch.concat([F_G, F_I], -1)
        F_T = torch.transpose(F)
        D = torch.pad(F, paddings)
        
        ## stack Covariance, Universal and Faults
        ABF = torch.concat([AB,F_T], -1)

        C_matrix = torch.concat([ABF, C], 0)
        C_matrix = torch.concat([C_matrix,D],0)
        

        C_matrix = torch.where(torch.logical_and(torch.abs(C_matrix) > 0, torch.abs(
            C_matrix) < 1e-9), torch.tensor(0, dtype=self.dtype), y=C_matrix)

        return C_matrix

    #@torch.function
    def deg2rad(self, degree_matrix):
        return degree_matrix * torch.tensor(0.0174533, dtype=self.dtype)

    #@torch.function
    def b_vector(self, dip_angles_=None, azimuth_=None, polarity_=None):

        length_of_C = self.matrices_shapes()[-1]
        if dip_angles_ is None:
            dip_angles_ = self.dip_angles
        if azimuth_ is None:
            azimuth_ = self.azimuth
        if polarity_ is None:
            polarity_ = self.polarity

        G_x = torch.sin(self.deg2rad(dip_angles_)) * \
            torch.sin(self.deg2rad(azimuth_)) * polarity_
        G_y = torch.sin(self.deg2rad(dip_angles_)) * \
            torch.cos(self.deg2rad(azimuth_)) * polarity_
        G_z = torch.cos(self.deg2rad(dip_angles_)) * polarity_

        G = torch.concat([G_x, G_y, G_z], -1)

        G = torch.expand_dims(G, axis=1)

        b_vector = torch.pad(G, [[0, length_of_C - torch.shape(G)[0]], [0, 0]])

        return b_vector

    #@torch.function
    def solve_kriging(self, dips_position_all, ref_layer_points, rest_layer_points, b=None):

        C_matrix = self.covariance_matrix(
            dips_position_all, ref_layer_points, rest_layer_points)

        b_vector = self.b_vector()

        DK = torch.linalg.solve(C_matrix, b_vector)

        return DK

    #@torch.function
    def x_to_interpolate(self,grid_val, ref_layer_points, rest_layer_points):
        grid_val = torch.concat([grid_val, rest_layer_points,ref_layer_points], 0)

        return grid_val

    #@torch.function
    def extend_dual_kriging(self, weights, grid_shape):

        DK_parameters = weights

        DK_parameters = torch.tile(DK_parameters, [1, grid_shape])

        return DK_parameters

    #@torch.function
    def contribution_gradient_interface(self, dips_position_all, grid_val=None, weights=None):
        dips_position_all_tiled = torch.tile(
            dips_position_all, [self.n_dimensions, 1])
        length_of_CG = self.matrices_shapes()[0]

        hu_SimPoint = self.cartesian_dist(dips_position_all, grid_val)

        sed_dips_SimPoint = self.squared_euclidean_distance(
            dips_position_all_tiled, grid_val)

        sigma_0_grad = torch.reduce_sum((weights[:length_of_CG] *
                                      self.gi_rescale * (torch.negative(hu_SimPoint) *\
                                                          # first derivative
                                                          torch.where((sed_dips_SimPoint < self.a_T_surface),
                                                                   x=(- self.c_o_T * ((-14 / self.a_T_surface ** 2) + 105 / 4 * sed_dips_SimPoint / self.a_T_surface ** 3 -
                                                                                      35 / 2 * sed_dips_SimPoint ** 3 / self.a_T_surface ** 5 +
                                                                                      21 / 4 * sed_dips_SimPoint ** 5 / self.a_T_surface ** 7)),
                                                                   y=torch.tensor(0, dtype=self.dtype)))), 0)


        
        return sigma_0_grad

    #@torch.function
    def contribution_interface(self, ref_layer_points, rest_layer_points, grid_val, weights=None):

        length_of_CG, length_of_CGI = self.matrices_shapes()[:2]

        # Euclidian distances
        sed_rest_SimPoint = self.squared_euclidean_distance(
            rest_layer_points, grid_val)
        sed_ref_SimPoint = self.squared_euclidean_distance(
            ref_layer_points, grid_val)

        sigma_0_interf = torch.reduce_sum(-weights[length_of_CG:length_of_CG + length_of_CGI, :] *
                                       (self.c_o_T * self.i_rescale * (
                                           torch.where(sed_rest_SimPoint < self.a_T_surface,  # SimPoint - Rest Covariances Matrix
                                                    x=(1 - 7 * (sed_rest_SimPoint / self.a_T_surface) ** 2 +
                                                        35 / 4 * (sed_rest_SimPoint / self.a_T_surface) ** 3 -
                                                        7 / 2 * (sed_rest_SimPoint / self.a_T_surface) ** 5 +
                                                        3 / 4 * (sed_rest_SimPoint / self.a_T_surface) ** 7),
                                                    y=torch.tensor(0., dtype=self.dtype)) -
                                        (torch.where(sed_ref_SimPoint < self.a_T_surface,  # SimPoint- Ref
                                                  x=(1 - 7 * (sed_ref_SimPoint / self.a_T_surface) ** 2 +
                                                     35 / 4 * (sed_ref_SimPoint / self.a_T_surface) ** 3 -
                                                      7 / 2 * (sed_ref_SimPoint / self.a_T_surface) ** 5 +
                                                      3 / 4 * (sed_ref_SimPoint / self.a_T_surface) ** 7),
                                                  y=torch.tensor(0., dtype=self.dtype))))), 0)


        return sigma_0_interf

    #@torch.function
    def contribution_universal_drift(self, grid_val, weights=None):

        length_of_CG, length_of_CGI, length_of_U_I, length_of_faults, length_of_C = self.matrices_shapes()

        _submatrix1 = torch.transpose(torch.concat([grid_val, grid_val**2], 1))

        _submatrix2 = torch.stack([grid_val[:, 0] * grid_val[:, 1], grid_val[:, 0]
                                * grid_val[:, 2], grid_val[:, 1] * grid_val[:, 2]])

        universal_grid_surface_points_matrix = torch.concat(
            [_submatrix1, _submatrix2], 0)

        i_rescale_aux = torch.concat([torch.ones([3], dtype=self.dtype), torch.tile(
            torch.expand_dims(self.gi_rescale, 0), [6])], -1)

        _aux_magic_term = torch.tile(torch.expand_dims(
            i_rescale_aux[:self.n_universal_eq_T_op], 0), [torch.shape(grid_val)[0], 1])

        f_0 = torch.reduce_sum(weights[length_of_CG + length_of_CGI:length_of_CG + length_of_CGI + length_of_U_I] * self.gi_rescale *
                            torch.transpose(_aux_magic_term) *
                            universal_grid_surface_points_matrix[:self.n_universal_eq_T_op], 0)

        return f_0

    def contribution_faults(self, weights=None, f_m=None):
        """
        Computation of the contribution of the df drift at every point to interpolate. To get these we need to
        compute a whole block model with the df data

        Returns:
            tensorflow.tensor: Contribution of the df drift (input) at every point to interpolate
        """
        length_of_CG, length_of_CGI, length_of_U_I, length_of_faults, length_of_C = self.matrices_shapes()

        fault_matrix_selection_non_zero = f_m

        f_1 = torch.reduce_sum(
            weights[length_of_CG + length_of_CGI + length_of_U_I:,
            :] * fault_matrix_selection_non_zero, axis=0)


        return f_1
    
    #@torch.function
    def get_scalar_field_at_surface_points(self, Z_x, npf=None):
        partitions = torch.reduce_sum(torch.one_hot(npf, self.len_points, dtype='int32'), axis=0)
        #
        scalar_field_at_surface_points_values = torch.dynamic_partition(Z_x[-2 * self.len_points: -self.len_points],partitions,2)[1]
        
        # scalar_field_at_surface_points_values = torch.gather_nd(
        #     Z_x[-2 * self.len_points: -self.len_points], torch.expand_dims(self.npf, 1))
        ### DOUBLE CHECK probably torch.reverse
        ## still using [::-1](reversed) at test
        # return scalar_field_at_surface_points_values[::-1]
        return scalar_field_at_surface_points_values
    
    #@torch.function
    def compare(self, a, b, slice_init, Z_x, l, n_surface, drift):
        """
        Treshold of the points to interpolate given 2 potential field values. TODO: This function is the one we
        need to change for a sigmoid function
        
        Args:
            a (scalar): Upper limit of the potential field
            b (scalar): Lower limit of the potential field
            n_surface (scalar): Value given to the segmentation, i.e. lithology number
            Zx (vector): Potential field values at all the interpolated points

        Returns:
            Tensor: segmented values
        """
        n_surface_0 = n_surface[:, slice_init:slice_init + torch.tensor(1,torch.int32)]
        n_surface_1 = n_surface[:, slice_init + torch.tensor(1,torch.int32):slice_init + torch.tensor(2,torch.int32)]
        drift = drift[:, slice_init:slice_init + torch.tensor(1,torch.int32)]

        # The 5 rules the slope of the function
        # print('l:',l)
        sigm = (-torch.reshape(n_surface_0, (-1, 1)) / (1 + torch.exp(-l * (Z_x - a))))  \
            - (torch.reshape(n_surface_1, (-1, 1)) /
             (1 + torch.exp(l * (Z_x - b)))) + torch.reshape(drift, (-1, 1))

        sigm.set_shape([None, None])
        return sigm

    #@torch.function
    def export_formation_block(self, Z_x, scalar_field_at_surface_points, values_properties,n_seires):

        
        slope = self.sig_slope
        scalar_field_iter = torch.pad(torch.expand_dims(
            scalar_field_at_surface_points, 0), [[0, 0], [1, 1]],constant_values = 0.0001)[0]
        # print('scalar_field_at_surface_points: ',scalar_field_iter)
 
        n_surface_op_float_sigmoid_mask = torch.repeat(
            values_properties, 2, axis=1)
        n_surface_op_float_sigmoid = torch.pad(
            n_surface_op_float_sigmoid_mask[:, 1:-1], [[0, 0], [1, 1]])
        drift = torch.pad(
            n_surface_op_float_sigmoid_mask[:, 0:-1], [[0, 0], [0, 1]])
        formations_block = torch.zeros([1, torch.shape(Z_x)[0]], dtype=self.dtype)

        i0 = torch.tensor(0) # i = 0
        c = lambda i,formations_block: torch.less(i, torch.shape(scalar_field_iter)[0]-1) # while i < 2
        # print('n_surface_op_float_sigmoid: ',n_surface_op_float_sigmoid)
        b = lambda i,formations_block: [i+1, formations_block + \
                self.compare(scalar_field_iter[i], scalar_field_iter[i + 1],
                             2 * i, Z_x, slope, n_surface_op_float_sigmoid, drift)] # i ++
        _,formations_block = torch.while_loop(c, b, loop_vars=[i0, formations_block], shape_invariants=[i0.get_shape(), torch.TensorShape([None, self.matrix_size])])

        return formations_block

    # @torch.function
    ### CHECK THIS
    def compute_forward_gravity(self, tz, lg0, lg1, densities=None):
        densities = torch.strided_slice(densities,[lg0],[lg1],[1])
        n_devices = torch.math.floordiv((torch.shape(densities)[0]), torch.shape(tz)[0])
        tz_rep = torch.tile(tz, [n_devices])
        grav = torch.reduce_sum(torch.reshape(densities * tz_rep, [n_devices, -1]), axis=1)
        return grav
    
    #@torch.function
    def compute_scalar_field(self,weights,grid_val,fault_matrix):
  
        tiled_weights = self.extend_dual_kriging(
            weights, torch.shape(grid_val)[0])
        sigma_0_grad = self.contribution_gradient_interface(self.dips_position,
                                                            grid_val, tiled_weights)
        sigma_0_interf = self.contribution_interface(
            self.ref_layer_points, self.rest_layer_points, grid_val, tiled_weights)
        f_0 = self.contribution_universal_drift(grid_val, weights)
        f_1 = self.contribution_faults(weights,fault_matrix)

        scalar_field_results = sigma_0_grad + sigma_0_interf + f_0 + f_1
        return scalar_field_results
              
    # @torch.function
    def compute_a_series(self,surface_point_all,dips_position_all,dip_angles_all,azimuth_all,polarity_all,value_properties,
                         len_i_0=0, len_i_1=None,
                         len_f_0=0, len_f_1=None,
                         len_w_0=0, len_w_1=None,
                         n_form_per_serie_0=0, n_form_per_serie_1=None,
                         u_grade_iter=3,
                         compute_weight_ctr=True,
                         compute_scalar_ctr=True,
                         compute_block_ctr=True,
                         is_finite=False, is_erosion=True,
                         is_onlap=False,
                         n_series=0,
                         range=10., c_o=10.,
                         block_matrix=None, weights_vector=None,
                         scalar_field_matrix=None, sfai_concat=None, mask_matrix=None,property_matrix = None,
                         mask_matrix_f=None, fault_matrix=None, nsle=0, grid=None,
                         shift=None
                         ):
        """
        Function that loops each fault, generating a potential field for each on them with the respective block model

        Args:
            len_i_0: Lenght of rest of previous series
            len_i_1: Lenght of rest for the computed series
            len_f_0: Lenght of dips of previous series
            len_f_1: Length of dips of the computed series
            n_form_per_serie_0: Number of surfaces of previous series
            n_form_per_serie_1: Number of surfaces of the computed series

        Returns:
            Tensor: block model derived from the df that afterwards is used as a drift for the "real"
            data
        """    
        
        # self.dips_position = dips_position_all[len_f_0: len_f_1, :]
        self.dips_position = torch.strided_slice(dips_position_all,[len_f_0,0],[len_f_1,torch.shape(dips_position_all)[1]],[1,1])
        self.dips_position_tiled = torch.tile(
            self.dips_position, [self.n_dimensions, 1])

        self.len_points = torch.shape(surface_point_all)[0] - \
            torch.shape(self.number_of_points_per_surface_T)[0]
        
        self.a_T_scalar = range
        self.c_o_T_scalar = c_o

        self.number_of_points_per_surface_T_op = torch.strided_slice(self.number_of_points_per_surface_T,
                                    [n_form_per_serie_0],[n_form_per_serie_1],[1])
        self.npf_op = torch.strided_slice(self.npf,[n_form_per_serie_0],[n_form_per_serie_1],[1])
        self.dip_angles = torch.strided_slice(dip_angles_all,[len_f_0],[len_f_1],[1])
        self.azimuth = torch.strided_slice(azimuth_all,[len_f_0],[len_f_1],[1])
        self.polarity = torch.strided_slice(polarity_all,[len_f_0],[len_f_1],[1])
        self.ref_layer_points = torch.strided_slice(self.ref_layer_points_all,[len_i_0],[len_i_1],[1])
        self.rest_layer_points = torch.strided_slice(self.rest_layer_points_all,[len_i_0],[len_i_1],[1])
        self.nugget_effect_scalar_T_op = torch.strided_slice(self.nugget_effect_scalar_T_ref_rest,[len_i_0],[len_i_1],[1])
        self.nugget_effect_grad_T_op = torch.strided_slice(self.nugget_effect_grad_T,[len_f_0 * 3],[len_f_1 * 3],[1])

        # The gradients have been tiled outside
        

        x_to_interpolate_shape = torch.shape(self.grid_val)[0] + 2 * self.len_points        

        faults_relation_op = self.fault_relation[:, n_series]
  

        indices = torch.where(faults_relation_op) ## tensorflow find nonzero index, reproduce Theano.nonzero

        fault_matrix_op = torch.gather(fault_matrix,indices) # select the dimension where fault relation is true
        fault_matrix_op = torch.reshape(fault_matrix_op,[-1,x_to_interpolate_shape])* self.offset


        if torch.shape(fault_matrix_op)[0] is None:
          self.lengh_of_faults =  torch.tensor(0, dtype=torch.int32)
        else:
          self.lengh_of_faults = torch.shape(fault_matrix_op)[0]

        interface_loc = torch.shape(self.grid_val)[0].int()

        self.fault_drift_at_surface_points_rest = torch.strided_slice(fault_matrix_op,[0,interface_loc + len_i_0],[torch.shape(fault_matrix_op)[0],interface_loc + len_i_1],[1,1])


        self.fault_drift_at_surface_points_ref = fault_matrix_op[
                                                 :,
                                                 interface_loc + self.len_points + len_i_0:
                                                 interface_loc + self.len_points + len_i_1]


        grid_val = self.x_to_interpolate(self.grid_val, self.ref_layer_points_all, self.rest_layer_points_all)
        weights = self.solve_kriging(self.dips_position, self.ref_layer_points, self.rest_layer_points)

        Z_x = self.compute_scalar_field(weights,grid_val,fault_matrix_op)
        
        
        scalar_field_at_surface_points = self.get_scalar_field_at_surface_points(Z_x,
                                                                                 self.npf_op)

        if is_erosion:
          if self.sigmoid == False:
            # 
            mask_e = (torch.math.greater(Z_x,torch.math.reduce_min(scalar_field_at_surface_points)))
            mask_e = mask_e & ~self.is_fault[n_series]

            mask_e = torch.where(mask_e,1,0) # convert boolean to int value for further computation -> cumprod
          else:
            # Sigmoid function between different series
            slope = self.sig_slope
            mask_e = (1 / (1 + torch.exp(-slope * (Z_x - torch.math.reduce_min(scalar_field_at_surface_points)))))
        else: 
            ## CAUTIOUS: Because ``self.is_fault[n_series]`` is a Tensor, use `if` statement will use Python and TensorFlow at the same time and causes error in AutoGraph. So use torch.cond instead
            mask_e = torch.cond(self.is_fault[n_series],lambda:torch.zeros(torch.shape(Z_x), dtype = self.mask_dtype),lambda:torch.ones(torch.shape(Z_x), dtype = self.mask_dtype))
        mask_e = torch.expand_dims(mask_e,axis=0)
        
        block = self.export_formation_block(Z_x, scalar_field_at_surface_points, value_properties[:,
                                   n_form_per_serie_0: n_form_per_serie_1 + 1],n_series)
        self.block = block

        ## In theano, this is done by set_subtensor, because tensor does not allow tensor assignment, here I use concat
        block_matrix = torch.concat([block_matrix,torch.slice(block,[0,0],[1,-1])],axis=0)
        fault_matrix = torch.concat([fault_matrix,torch.slice(block,[0,0],[1,-1])],axis=0)
        if self.compute_gravity_flag == True:
            property_matrix = torch.concat([property_matrix,torch.slice(block,[1,0],[1,-1])],axis=0)

        mask_matrix = torch.concat([mask_matrix,mask_e],axis=0)
        
        scalar_field_matrix = torch.concat([scalar_field_matrix,torch.expand_dims(Z_x,axis=0)],axis=0)
        
        paddings = torch.stack([[n_form_per_serie_0, self.n_surfaces_per_series[-1] - n_form_per_serie_1]],axis=0)

        sfai = torch.expand_dims(torch.pad(scalar_field_at_surface_points,paddings ),axis=0)

        sfai_concat = torch.concat([sfai_concat,sfai],axis = 0)
        
        # Number of series since last erode: This is necessary in case there are multiple consecutives onlaps
        # Onlap version
        is_onlap_or_fault = torch.logical_or(self.is_onlap[n_series], self.is_fault[n_series])
        return block_matrix,property_matrix, scalar_field_matrix, sfai_concat, mask_matrix, \
              fault_matrix


    # @torch.function
    def compute_series(self,surface_point_all,dips_position_all,dip_angles, azimuth, polarity, values_properties = None):
        
        self.update_property()
        if self.gradient:

            self.sig_slope = self.max_slope*torch.math.sigmoid(self.delta_slope)
        else:
            self.sig_slope = 5e10 # if 

        self.ref_layer_points_all, self.rest_layer_points_all, self.ref_nugget, self.rest_nugget = self.set_rest_ref_matrix(self.number_of_points_per_surface_T, surface_point_all, self.nugget_effect_scalar)
        
        self.nugget_effect_scalar_T_ref_rest = torch.expand_dims(self.ref_nugget + self.rest_nugget, 1)
        
        self.len_points = torch.shape(surface_point_all)[0] - \
            torch.shape(self.number_of_points_per_surface_T)[0]
            
        num_series = self.len_series_i.shape[0] - torch.tensor(1, dtype=torch.int32)
        
        matrix_fix_dim = torch.shape(self.grid_val)[0]+2*self.len_points
        self.block_matrix = torch.zeros((0,matrix_fix_dim),dtype=self.dtype)
        self.property_matrix = torch.zeros((0,matrix_fix_dim),dtype=self.dtype)
        self.mask_matrix = torch.zeros((0,matrix_fix_dim),dtype=self.mask_dtype)
        self.scalar_matrix = torch.zeros((0,matrix_fix_dim),dtype=self.dtype)
        self.sfai = torch.zeros((0,self.n_surfaces_per_series[-1]),dtype=self.dtype)

        self.fault_matrix = torch.zeros((0,matrix_fix_dim),dtype=self.dtype)


        if values_properties is None:
            values_properties = self.value_properties

        i0 = torch.tensor(0) # i = 0

        for i in range(num_series):
            self.block_matrix,self.property_matrix, self.scalar_matrix, self.sfai, self.mask_matrix, self.fault_matrix = self.compute_a_series(surface_point_all,
            dips_position_all,
            dip_angles,
            azimuth,
            polarity,
            values_properties,
            len_i_0=self.len_series_i[i],
            len_i_1=self.len_series_i[i+1],
            len_f_0=self.len_series_o[i],
            len_f_1=self.len_series_o[i+1],
            len_w_0=self.len_series_w[i],
            len_w_1=self.len_series_w[i+1],
            n_form_per_serie_0=self.n_surfaces_per_series[i],
            n_form_per_serie_1=self.n_surfaces_per_series[i+1],
            u_grade_iter=3,
            compute_weight_ctr=True,
            compute_scalar_ctr=True,
            compute_block_ctr=True,
            is_finite=False,
            is_erosion=self.is_erosion[i],
            is_onlap=False,
            n_series=i,
            range=10.,
            c_o=10.,
            block_matrix=None,
            weights_vector=None,
            scalar_field_matrix=None,
            sfai=None,
            mask_matrix=None,
            nsle=0,
            grid=None,
            shift=None)
            
        if self.sigmoid == False:
          last_series_mask = torch.math.cumprod(torch.where(self.mask_matrix[:-1]==0,torch.constant(1,self.mask_dtype),0))
          block_mask = torch.concat([self.mask_matrix[-1:],last_series_mask],axis=0)
          block_mask = self.mask_matrix*block_mask
          final_block = torch.reduce_sum(torch.where(block_mask==1,self.block_matrix,0),0)
          if self.compute_gravity_flag == True:
            final_property= torch.reduce_sum(torch.where(block_mask==1,self.property_matrix,0),0)
          else:
            final_property = self.property_matrix
        else:
          last_series_mask = torch.math.cumprod(torch.constant(1,dtype = self.dtype)-self.mask_matrix[:-1])
          block_mask = torch.concat([self.mask_matrix[-1:],last_series_mask],axis=0)
          block_mask = self.mask_matrix*block_mask
          final_block = torch.reduce_sum(block_mask*self.block_matrix,0)
          if self.compute_gravity_flag == True:
            final_property= torch.reduce_sum(block_mask*self.property_matrix,0)
          else:
            final_property = self.property_matrix
        return final_block,final_property,self.block_matrix,self.scalar_matrix,self.sfai,block_mask,self.fault_matrix
