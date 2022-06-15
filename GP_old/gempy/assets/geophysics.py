"""
    This file is part of gempy.

    gempy is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    gempy is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with gempy.  If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np
import theano
import theano.tensor as T
from gempy.core.grid_modules.grid_types import CenteredGrid,RegularGrid


class GravityPreprocessing(CenteredGrid):
    def __init__(self, centered_grid: CenteredGrid = None):

        if centered_grid is None:
            super().__init__()
        elif isinstance(centered_grid, CenteredGrid):
            self.kernel_centers = centered_grid.kernel_centers
            self.kernel_dxyz_right = centered_grid.kernel_dxyz_right
            self.kernel_dxyz_left = centered_grid.kernel_dxyz_left
        self.tz = np.empty(0)

    def set_tz_kernel(self, scale=True, **kwargs):
        if self.kernel_centers.size == 0:
            self.set_centered_kernel(**kwargs)

        grid_values = self.kernel_centers

        s_gr_x = grid_values[:, 0]
        s_gr_y = grid_values[:, 1]
        s_gr_z = grid_values[:, 2]

        # getting the coordinates of the corners of the voxel...
        x_cor = np.stack((s_gr_x - self.kernel_dxyz_left[:, 0], s_gr_x + self.kernel_dxyz_right[:, 0]), axis=1)
        y_cor = np.stack((s_gr_y - self.kernel_dxyz_left[:, 1], s_gr_y + self.kernel_dxyz_right[:, 1]), axis=1)
        z_cor = np.stack((s_gr_z - self.kernel_dxyz_left[:, 2], s_gr_z + self.kernel_dxyz_right[:, 2]), axis=1)

        # ...and prepare them for a vectorial op
        x_matrix = np.repeat(x_cor, 4, axis=1)
        y_matrix = np.tile(np.repeat(y_cor, 2, axis=1), (1, 2))
        z_matrix = np.tile(z_cor, (1, 4))

        s_r = np.sqrt(x_matrix ** 2 + y_matrix ** 2 + z_matrix ** 2)

        # This is the vector that determines the sign of the corner of the voxel
        mu = np.array([1, -1, -1, 1, -1, 1, 1, -1])

        if scale is True:
            #
            G = 6.674e-3 # mgal    
        else:
            from scipy.constants import G

        self.tz = (
            G *
            np.sum(- 1 *
                   mu * (
                           x_matrix * np.log(y_matrix + s_r) +
                           y_matrix * np.log(x_matrix + s_r) -
                           z_matrix * np.arctan(x_matrix * y_matrix / (z_matrix * s_r))),
                   axis=1))

        return self.tz
    
class GravityPreprocessingRegAllLoop(RegularGrid):
    """
    @Zhouji Liang

    """
    def __init__(self, model, regular_grid: RegularGrid = None):

        if regular_grid is None:
            super().__init__()
        elif isinstance(regular_grid, RegularGrid):
            self.model = model
            # self.kernel_centers = np.repeat(regular_grid.values[:,:,np.newaxis],2,axis=2) - model.xy_ravel.T
            self.num_receivers = 1
            # self.kernel_dxyz_right = regular_grid.kernel_dxyz_right
            # self.kernel_dxyz_left = regular_grid.kernel_dxyz_left
        self.tz = np.empty(0)

    def set_tz_kernel(self, model_radius,regular_grid_resolution,scale=True, **kwargs):
        dx, dy, dz = self.model.grid.regular_grid.get_dx_dy_dz()

        # we need to find the closest center for each receiver to keep numerical stability
        # here we find the smallest center which is greater than the receiver coordinates for x and y
        range_xy = [(self.model.extent[1] - self.model.extent[0])/2,(self.model.extent[3] - self.model.extent[2])/2] 
        #center of model
        new_xy_ravel = np.expand_dims(np.array([self.model.extent[0]+range_xy[0],self.model.extent[2]+range_xy[1]]),axis = 1)

        ################
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.re_x = self.model.xy_ravel.T[0]
        self.re_x = self.re_x + (dx / 2 - (self.re_x-self.model.extent[0]) % dx)

        self.re_y = self.model.xy_ravel.T[1]
        self.re_y = self.re_y + (dy / 2 - (self.re_y-self.model.extent[2]) % dy)

        self.new_xy_ravel = np.stack(
            [
                self.re_x,
                self.re_y,
            ],
            axis=0,
        )
        # concat with z value
        self.new_xy_ravel = np.concatenate(
            [self.new_xy_ravel, self.model.xy_ravel.T[2, None]]
        )

        ################
        # # sensor location move to the next closest cell center
        re_x = new_xy_ravel[0]
        re_x = re_x + (dx / 2 - (re_x-self.model.extent[0]) % dx)

        re_y = new_xy_ravel[1]
        re_y = re_y + (dy / 2 - (re_y-self.model.extent[2]) % dy)

        # concat with z value
        new_xy_ravel_temp = np.concatenate(
            [np.array([re_x,re_y]), np.array([[self.model.extent[-1]]])])

        self.center_index_x = (new_xy_ravel_temp[0]-self.model.extent[0])//dx
        self.center_index_y = (new_xy_ravel_temp[1]-self.model.extent[2])//dy
        self.radius_cell_x = int(self.model.model_radius[0]//dx)
        self.radius_cell_y = int(self.model.model_radius[1]//dy)

        # kernel_centers = np.repeat(self.model.grid.regular_grid.values[:,:,np.newaxis],self.num_receivers,axis=2)-self.model.xy_ravel.T
        kernel_centers = np.squeeze(
                self.model.grid.regular_grid.values[:, :, np.newaxis] - new_xy_ravel_temp
        )

        c_x = int(self.center_index_x[0])
        c_y = int(self.center_index_y[0])

        slice_kernel_centers = (kernel_centers[:,:].reshape(self.model.regular_grid_resolution+[3,self.num_receivers])[c_x-self.radius_cell_x:c_x+self.radius_cell_x+1,c_y-self.radius_cell_y:c_y+self.radius_cell_y+1,:,:]).reshape([-1,3])


        x_cor = np.stack(
            (slice_kernel_centers[:, 0] - dx / 2, slice_kernel_centers[:, 0] + dx / 2), axis=1
        )
        y_cor = np.stack(
            (slice_kernel_centers[:, 1] - dy / 2, slice_kernel_centers[:, 1] + dy / 2), axis=1
        )
        z_cor = np.stack(
            (slice_kernel_centers[:, 2] + dz / 2, slice_kernel_centers[:, 2] - dz / 2), axis=1
        )

        # ...and prepare them for a vectorial op
        x_matrix = np.repeat(x_cor, 4, axis=1)
        y_matrix = np.tile(np.repeat(y_cor, 2, axis=1), (1, 2))
        z_matrix = np.tile(z_cor, (1, 4)) - 0.005 * (self.model.extent[-1]-self.model.extent[-2])

        s_r = np.sqrt(x_matrix ** 2 + y_matrix ** 2 + z_matrix ** 2)

        # This is the vector that determines the sign of the corner of the voxel
        mu = np.array([1, -1, -1, 1, -1, 1, 1, -1])


        G = 6.674e-3  # 

        tz = G * np.sum(
            -1
            * mu
            * (
                x_matrix * np.log(y_matrix + s_r)
                + y_matrix * np.log(x_matrix + s_r)
                - z_matrix * np.arctan(x_matrix * y_matrix / (z_matrix * s_r))
            ),
            axis=1,
        )
        return tz


class MagneticsPreprocessing(CenteredGrid):
    """
    @Nilgün Güdük

    """
    def __init__(self, centered_grid: CenteredGrid = None):

        if centered_grid is None:
            super().__init__()
        elif isinstance(centered_grid, CenteredGrid):
            self.kernel_centers = centered_grid.kernel_centers
            self.kernel_dxyz_right = centered_grid.kernel_dxyz_right
            self.kernel_dxyz_left = centered_grid.kernel_dxyz_left
        self.V = np.empty(0)

    def set_Vs_kernel(self, **kwargs):
        if self.kernel_centers.size == 0:
            self.set_centered_kernel(**kwargs)

        grid_values = self.kernel_centers
        s_gr_x = grid_values[:, 0]
        s_gr_y = grid_values[:, 1]
        s_gr_z = -1 * grid_values[:, 2]  # talwani takes x-axis positive downwards, and gempy negative downwards

        # getting the coordinates of the corners of the voxel...
        x_cor = np.stack((s_gr_x - self.kernel_dxyz_left[:, 0], s_gr_x + self.kernel_dxyz_right[:, 0]), axis=1)
        y_cor = np.stack((s_gr_y - self.kernel_dxyz_left[:, 1], s_gr_y + self.kernel_dxyz_right[:, 1]), axis=1)
        z_cor = np.stack((s_gr_z + self.kernel_dxyz_left[:, 2], s_gr_z - self.kernel_dxyz_right[:, 2]), axis=1)
        # ...and prepare them for a vectorial op
        x_matrix = np.repeat(x_cor, 4, axis=1)
        y_matrix = np.tile(np.repeat(y_cor, 2, axis=1), (1, 2))
        z_matrix = np.tile(z_cor, (1, 4))

        R = np.sqrt(x_matrix ** 2 + y_matrix ** 2 + z_matrix ** 2)  # distance to each corner
        s = np.array([-1, 1, 1, -1, 1, -1, -1, 1])  # gives the sign of each corner: depends on your coordinate system

        # variables V1-6 represent integrals of volume for each voxel
        V1 = np.sum(-1 * s * np.arctan2((y_matrix * z_matrix), (x_matrix * R)), axis=1)
        V2 = np.sum(s * np.log(R + z_matrix), axis=1)
        V3 = np.sum(s * np.log(R + y_matrix), axis=1)
        V4 = np.sum(-1 * s * np.arctan2((x_matrix * z_matrix), (y_matrix * R)), axis=1)
        V5 = np.sum(s * np.log(R + x_matrix), axis=1)
        V6 = np.sum(-1 * s * np.arctan2((x_matrix * y_matrix), (z_matrix * R)), axis=1)

        # contains all the volume integrals (6 x n_kernelvalues)
        V = np.array([V1, V2, V3, V4, V5, V6])
        return V
