from gempy.utils.create_topography import Load_DEM_artificial, Load_DEM_GDAL
import numpy as np
import skimage.transform
import matplotlib.pyplot as plt
from scipy.constants import G
from scipy import interpolate
from gempy.utils.meta import setdoc, setdoc_pro
import gempy.utils.docstring as ds
from typing import Optional
import pandas as pn


class RegularGrid:
    """
    Class with the methods and properties to manage 3D regular grids where the model will be interpolated.

    Args:
        extent (np.ndarray):  [x_min, x_max, y_min, y_max, z_min, z_max]
        resolution (np.ndarray): [nx, ny, nz]

    Attributes:
        extent (np.ndarray):  [x_min, x_max, y_min, y_max, z_min, z_max]
        resolution (np.ndarray): [nx, ny, nz]
        values (np.ndarray): XYZ coordinates
        mask_topo (np.ndarray, dtype=bool): same shape as values. Values above the topography are False
        dx (float): size of the cells on x
        dy (float): size of the cells on y
        dz (float): size of the cells on z

    """
    def __init__(self, extent=None, resolution=None, **kwargs):
        self.resolution = np.ones((0, 3), dtype='int64')
        self.extent = np.zeros(6, dtype='float64')
        self.extent_r = np.zeros(6, dtype='float64')
        self.values = np.zeros((0, 3))
        self.values_r = np.zeros((0, 3))
        self.mask_topo = np.zeros((0, 3), dtype=bool)
        if extent is not None and resolution is not None:
            self.set_regular_grid(extent, resolution)
            self.dx, self.dy, self.dz = self.get_dx_dy_dz()
    
    def set_coord(self, extent, resolution):
        dx = (extent[1] - extent[0]) / resolution[0]
        dy = (extent[3] - extent[2]) / resolution[1]
        dz = (extent[5] - extent[4]) / resolution[2]

        self.x = np.linspace(extent[0] + dx / 2, extent[1] - dx / 2, resolution[0],
                             dtype="float64")
        self.y = np.linspace(extent[2] + dy / 2, extent[3] - dy / 2, resolution[1],
                             dtype="float64")
        self.z = np.linspace(extent[4] + dz / 2, extent[5] - dz / 2, resolution[2],
                             dtype="float64")

        return self.x, self.y, self.z
    
    @staticmethod
    def create_regular_grid_3d(extent, resolution):
        """
        Method to create a 3D regular grid where is interpolated

        Args:
            extent (list):  [x_min, x_max, y_min, y_max, z_min, z_max]
            resolution (list): [nx, ny, nz].

        Returns:
            numpy.ndarray: Unraveled 3D numpy array where every row correspond to the xyz coordinates of a regular grid

        """

        dx = (extent[1] - extent[0]) / resolution[0]
        dy = (extent[3] - extent[2]) / resolution[1]
        dz = (extent[5] - extent[4]) / resolution[2]

        g = np.meshgrid(
            np.linspace(extent[0] + dx / 2, extent[1] - dx / 2, resolution[0], dtype="float64"),
            np.linspace(extent[2] + dy / 2, extent[3] - dy / 2, resolution[1], dtype="float64"),
            np.linspace(extent[4] + dz / 2, extent[5] - dz / 2, resolution[2], dtype="float64"), indexing="ij"
        )

        values = np.vstack(tuple(map(np.ravel, g))).T.astype("float64")
        
        ### 
        values = values.reshape(resolution+[3]) 
        values = np.flip(values, 2)
        values = values.reshape([-1,3])

        return values

    def get_dx_dy_dz(self, rescale=False):
        if rescale is True:
            dx = (self.extent_r[1] - self.extent_r[0]) / self.resolution[0]
            dy = (self.extent_r[3] - self.extent_r[2]) / self.resolution[1]
            dz = (self.extent_r[5] - self.extent_r[4]) / self.resolution[2]
        else:
            dx = (self.extent[1] - self.extent[0]) / self.resolution[0]
            dy = (self.extent[3] - self.extent[2]) / self.resolution[1]
            dz = (self.extent[5] - self.extent[4]) / self.resolution[2]
        return dx, dy, dz

    def set_regular_grid(self, extent, resolution):
        """
        Set a regular grid into the values parameters for further computations
        Args:
             extent (list, np.ndarry):  [x_min, x_max, y_min, y_max, z_min, z_max]
            resolution (list, np.ndarray): [nx, ny, nz]
        """

        self.extent = np.asarray(extent, dtype='float64')
        self.resolution = np.asarray(resolution)
        self.values = self.create_regular_grid_3d(extent, resolution)
        self.length = self.values.shape[0]
        self.dx, self.dy, self.dz = self.get_dx_dy_dz()
        return self.values

    def set_topography_mask(self, topography):

        ind = topography._find_indices()
        gridz = self.values[:, 2].reshape(*self.resolution).copy()
        for x in range(self.resolution[0]):
            for y in range(self.resolution[1]):
                z = ind[x, y]
                gridz[x, y, z:] = 99999
        mask = (gridz == 99999)
        self.mask_topo = mask
        return mask  # np.multiply(np.full(self.regular_grid.values.shape, True).T, mask.ravel()).T


class Sections:
    """
    Object that creates a grid of cross sections between two points.

    Args:
        regular_grid: Model.grid.regular_grid
        section_dict: {'section name': ([p1_x, p1_y], [p2_x, p2_y], [xyres, zres])}
    """
    def __init__(self, regular_grid=None, z_ext=None, section_dict=None):
        if regular_grid is not None:
            self.z_ext = regular_grid.extent[4:]
        else:
            self.z_ext = z_ext

        self.section_dict = section_dict
        self.names = []
        self.points = []
        self.resolution = []
        self.length = [0]
        self.dist = []
        self.df = pn.DataFrame()
        self.df['dist'] = self.dist
        self.values = []
        self.extent = None

        if section_dict is not None:
           self.set_sections(section_dict)

    def _repr_html_(self):
        return self.df.to_html()

    def __repr__(self):
        return self.df.to_string()

    def show(self):
        pass

    def set_sections(self, section_dict, regular_grid=None, z_ext=None):
        self.section_dict = section_dict
        if regular_grid is not None:
            self.z_ext = regular_grid.extent[4:]

        self.names = np.array(list(self.section_dict.keys()))

        self.get_section_params()
        self.calculate_all_distances()
        self.df = pn.DataFrame.from_dict(self.section_dict, orient='index', columns=['start', 'stop', 'resolution'])
        self.df['dist'] = self.dist

        self.compute_section_coordinates()

    def get_section_params(self):
        self.points = []
        self.resolution = []
        self.length = [0]

        for i, section in enumerate(self.names):
            points = [self.section_dict[section][0], self.section_dict[section][1]]
            assert points[0] != points[1], 'The start and end points of the section must not be identical.'

            self.points.append(points)
            self.resolution.append(self.section_dict[section][2])
            self.length = np.append(self.length, self.section_dict[section][2][0] *
                                    self.section_dict[section][2][1])
        self.length = np.array(self.length).cumsum()

    def calculate_all_distances(self):
        self.coordinates = np.array(self.points).ravel().reshape(-1, 4) #axis are x1,y1,x2,y2
        self.dist = np.sqrt(np.diff(self.coordinates[:, [0, 2]])**2 + np.diff(self.coordinates[:, [1, 3]])**2)

    @staticmethod
    def distance_2_points(p1, p2):
        return np.sqrt(np.diff((p1[0], p2[0])) ** 2 + np.diff((p1[1], p2[1])) ** 2)

    def compute_section_coordinates(self):
        for i in range(len(self.names)):
            xy = self.calculate_line_coordinates_2points(self.coordinates[i, :2], self.coordinates[i, 2:], self.resolution[i][0])
            zaxis = np.linspace(self.z_ext[0], self.z_ext[1], self.resolution[i][1],
                                     dtype="float64")
            X, Z = np.meshgrid(xy[:, 0], zaxis, indexing='ij')
            Y, _ = np.meshgrid(xy[:, 1], zaxis, indexing='ij')
            xyz = np.vstack((X.flatten(), Y.flatten(), Z.flatten())).T
            if i == 0:
                self.values = xyz
            else:
                self.values = np.vstack((self.values, xyz))

    def calculate_line_coordinates_2points(self, p1, p2, res):
        if isinstance(p1, list):
            p1 = np.array(p1)
        if isinstance(p2, list):
            p2 = np.array(p2)
        v = p2-p1 #vector pointing from p1 to p2
        u = v/np.linalg.norm(v) # normalize it
        distance = self.distance_2_points(p1, p2)
        steps = np.linspace(0, distance, res)
        values = p1.reshape(2, 1) + u.reshape(2, 1) * steps.ravel()
        return values.T

    def get_section_args(self, section_name: str):
        where = np.where(self.names == section_name)[0][0]
        return self.length[where], self.length[where+1]

    def get_section_grid(self, section_name: str):
        l0, l1 = self.get_section_args(section_name)
        return self.values[l0:l1]


class CustomGrid:
    """Object that contains arbitrary XYZ coordinates.

    Args:
        custom_grid (numpy.ndarray like): XYZ (in columns) of the desired coordinates

    Attributes:
        values (np.ndarray): XYZ coordinates
    """
    def __init__(self, custom_grid: np.ndarray):
        self.values = np.zeros((0, 3))
        self.set_custom_grid(custom_grid)

    def set_custom_grid(self, custom_grid: np.ndarray):
        """
        Give the coordinates of an external generated grid

        Args:
            custom_grid (numpy.ndarray like): XYZ (in columns) of the desired coordinates

        Returns:
              numpy.ndarray: Unraveled 3D numpy array where every row correspond to the xyz coordinates of a regular
               grid
        """
        custom_grid = np.atleast_2d(custom_grid)
        assert type(custom_grid) is np.ndarray and custom_grid.shape[1] is 3, 'The shape of new grid must be (n,3)' \
                                                                              ' where n is the number of points of ' \
                                                                              'the grid'

        self.values = custom_grid
        self.length = self.values.shape[0]
        return self.values


class CenteredGrid:
    """
    Logarithmic spaced grid.
    """

    def __init__(self, centers=None, radius=None, resolution=None):
        self.grid_type = 'centered_grid'
        self.values = np.empty((0, 3))
        self.length = self.values.shape[0]
        self.kernel_centers = np.empty((0, 3))
        self.kernel_dxyz_left = np.empty((0, 3))
        self.kernel_dxyz_right = np.empty((0, 3))
        self.tz = np.empty(0)

        if centers is not None and radius is not None:
            if resolution is None:
                resolution = [10, 10, 20]

            self.set_centered_grid(centers=centers, radius=radius, resolution=resolution)

###################

    @staticmethod
    @setdoc_pro(ds.resolution)
    def create_irregular_grid_kernel(resolution, radius):
        """
        Create an isometric grid kernel (centered at 0)

        Args:
            resolution: [s0]
            radius (float): Maximum distance of the kernel

        Returns:
            tuple: center of the voxel, left edge of each voxel (for xyz), right edge of each voxel (for xyz).
        """

        if not isinstance(radius, list) or isinstance(radius, np.ndarray):
            radius = np.repeat(radius, 3)

        g_ = []
        g_2 = [] # contains exp coord, left right xy, top and bottom
        d_ = []
        for xyz in [0, 1, 2]:

            if xyz == 2:
                # Make the grid only negative for the z axis

                g_.append(np.geomspace(0.01, 1, int(resolution[xyz])))
  
                g_2.append(
                    (np.concatenate(([0], g_[xyz])) + 0.005) * - radius[xyz])
            else:
                g_.append(np.geomspace(0.01, 1, int(resolution[xyz] / 2)+1))
                g_2.append(
                    np.concatenate((-g_[xyz][::-1], g_[xyz])) * radius[xyz])
        #my modification below, change the left/right boundary to grow exponentally instead of the center point
        
        x_center = (g_2[0][:-1]+g_2[0][1:])/2 
        y_center = (g_2[1][:-1]+g_2[1][1:])/2
        z_center = (g_2[-1][:-1]+g_2[-1][1:])/2
        g = np.meshgrid(x_center,y_center,z_center)
        
        d_left_x = np.abs(g_2[0][:-1] - x_center)
        d_left_y = np.abs(g_2[1][:-1] - y_center)
        d_right_x = np.abs(g_2[0][1:] - x_center)
        d_right_y = np.abs(g_2[1][1:] - y_center)
        d_z = z_center - g_2[-1][:-1]
        
        d_left = np.meshgrid(d_left_x,d_left_y,d_z)
        d_right = np.meshgrid(d_right_x,d_right_y,d_z)
    
        kernel_g = np.vstack(tuple(map(np.ravel, g))).T.astype("float64")
        kernel_d_left = np.vstack(tuple(map(np.ravel, d_left))).T.astype("float64")
        kernel_d_right = np.vstack(tuple(map(np.ravel, d_right))).T.astype("float64")

        return kernel_g, kernel_d_left, kernel_d_right
    
#################
    
    # @staticmethod
    # @setdoc_pro(ds.resolution)
    # def create_irregular_grid_kernel(resolution, radius):
    #     """
    #     Create an isometric grid kernel (centered at 0)

    #     Args:
    #         resolution: [s0]
    #         radius (float): Maximum distance of the kernel

    #     Returns:
    #         tuple: center of the voxel, left edge of each voxel (for xyz), right edge of each voxel (for xyz).
    #     """

    #     if radius is not list or radius is not np.ndarray:
    #         radius = np.repeat(radius, 3)

    #     g_ = []
    #     g_2 = []
    #     d_ = []
    #     for xyz in [0, 1, 2]:

    #         if xyz == 2:
    #             # Make the grid only negative for the z axis

    #             g_.append(np.geomspace(0.01, 1, int(resolution[xyz])))
    #             g_2.append(
    #                 (np.concatenate(([0], g_[xyz])) + 0.005) * - radius[xyz] * 1.2)
    #         else:
    #             g_.append(np.geomspace(0.01, 1, int(resolution[xyz] / 2)))
    #             g_2.append(
    #                 np.concatenate((-g_[xyz][::-1], [0], g_[xyz])) * radius[xyz])
    #         d_.append(np.diff(np.pad(g_2[xyz], 1, 'reflect', reflect_type='odd')))

    #     g = np.meshgrid(*g_2)
    #     d_left = np.meshgrid(d_[0][:-1] / 2, d_[1][:-1] / 2, d_[2][:-1] / 2)
    #     d_right = np.meshgrid(d_[0][1:] / 2, d_[1][1:] / 2, d_[2][1:] / 2)
    #     kernel_g = np.vstack(tuple(map(np.ravel, g))).T.astype("float64")
    #     kernel_d_left = np.vstack(tuple(map(np.ravel, d_left))).T.astype("float64")
    #     kernel_d_right = np.vstack(tuple(map(np.ravel, d_right))).T.astype("float64")

    #     return kernel_g, kernel_d_left, kernel_d_right

    @setdoc_pro(ds.resolution)
    def set_centered_kernel(self, resolution, radius):
        """
        Set a centered

        Args:
            resolution: [s0]
            radius (float): Maximum distance of the kernel

        Returns:

        """
        self.kernel_centers, self.kernel_dxyz_left, self.kernel_dxyz_right = self.create_irregular_grid_kernel(
            resolution, radius)

        return self.kernel_centers

    @setdoc_pro(ds.resolution)
    def set_centered_grid(self, centers, kernel_centers=None, **kwargs):
        """
        Main method of the class, set the XYZ values around centers using a kernel.

        Args:
            centers (np.array): XYZ array with the centers of where we want to create a grid around
            kernel_centers (Optional[np.array]): center of the voxels of a desired kernel.
            **kwargs:
                * resolution: [s0]
                * radius (float): Maximum distance of the kernel
        Returns:

        """

        self.values = np.empty((0, 3))
        centers = np.atleast_2d(centers)

        if kernel_centers is None:
            kernel_centers = self.set_centered_kernel(**kwargs)

        assert centers.shape[1] == 3, 'Centers must be a numpy array that contains the coordinates XYZ'

        for i in centers:
            self.values = np.vstack((self.values, i + kernel_centers))

        self.length = self.values.shape[0]

    def set_tz_kernel(self, **kwargs):
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

        self.tz = (
            np.sum(- 1 *
                   G *
                   mu * (
                           x_matrix * np.log(y_matrix + s_r) +
                           y_matrix * np.log(x_matrix + s_r) -
                           z_matrix * np.arctan(x_matrix * y_matrix / (z_matrix * s_r))),
                   axis=1))

        return self.tz


##########
# centered regular grid
##########
class CenteredRegGrid(CenteredGrid):
    """
    linear spaced centered grid.
    """

    # @staticmethod
    def create_irregular_grid_kernel(self,resolution, radius):
        """
        Create an isometric grid kernel (centered at 0)

        Args:
            resolution: [s0]
            radius (float): Maximum distance of the kernel

        Returns:
            tuple: center of the voxel, left edge of each voxel (for xyz), right edge of each voxel (for xyz).
        """
        import copy
        if not isinstance(radius, list) or isinstance(radius, np.ndarray):
            radius = np.repeat(radius, 3)
   
        g_ = []
        g_2 = []  # contains exp coord, left right xy, top and bottom
        d_ = []
        # size of each grid
        res_ = copy.deepcopy(resolution)
        if res_[0] % 2 == 0:
            res_[0] += 1
        if res_[1] % 2 == 0:
            res_[1] += 1
        dx = radius[0] * 2 / res_[0]
        dy = radius[1] * 2 / res_[1]
        dz = radius[2] / res_[2]
        self.dxyz = [dx, dy, dz]
        for xyz in [0, 1, 2]:
            if xyz == 2:  # --> z
                # Make the grid only negative for the z axis
                g_.append(np.linspace(0, 1, int(resolution[xyz]), endpoint=False))
                g_2.append(((g_[xyz] + 0.005) * -radius[xyz] - dz / 2))

                d_.append(np.diff(np.pad(g_2[xyz], 1, "reflect", reflect_type="odd")))
            else:
                g_.append(
                    np.linspace(0, 1, num=resolution[xyz] // 2 + 1, endpoint=True)
                )
                g_2.append(
                    np.concatenate((-g_[xyz][::-1], g_[xyz][1:]))
                    * (radius[xyz] - self.dxyz[xyz] / 2)
                )
                d_.append(np.diff(np.pad(g_2[xyz], 1, "reflect", reflect_type="odd")))

        g = np.meshgrid(*g_2)
        d_left = np.meshgrid(d_[0][:-1] / 2, d_[1][:-1] / 2, d_[2][:-1] / 2)
        d_right = np.meshgrid(d_[0][1:] / 2, d_[1][1:] / 2, d_[2][1:] / 2)
        kernel_g = np.vstack(tuple(map(np.ravel, g))).T.astype("float64")
        kernel_d_left = np.vstack(tuple(map(np.ravel, d_left))).T.astype("float64")
        kernel_d_right = np.vstack(tuple(map(np.ravel, d_right))).T.astype("float64")

        return kernel_g, kernel_d_left, kernel_d_right



class Topography:
    """
    Object to include topography in the model.
    """
    def __init__(self, regular_grid):
        self.regular_grid = regular_grid
        self.values = np.zeros((0, 3))

        self.topo = None
        self.values_3D = np.zeros((0, 0, 0))
        self.extent = None
        self.resolution = None

        self.type = None

    def load_from_gdal(self, filepath):
        self.topo = Load_DEM_GDAL(filepath, self.regular_grid)
        self._create_init()
        self._fit2model()
        self.type = 'real'

    def load_random_hills(self, **kwargs):
        self.topo = Load_DEM_artificial(self.regular_grid, **kwargs)
        self._create_init()
        self._fit2model()
        self.type = 'artificial'

    def load_from_saved(self, filepath):
        assert filepath[-4:] == '.npy', 'The file must end on .npy'
        topo = np.load(filepath, allow_pickle=True)
        self.values_3D = topo[0]
        self.extent = topo[1]
        self.resolution = topo[2]
        self._fit2model()
        self.type = 'real'

    def _create_init(self):
        self.values_3D = self.topo.values_3D
        self.extent = self.topo.extent
        self.resolution = self.topo.resolution

    def _fit2model(self):
        self.values = np.vstack((
            self.values_3D[:, :, 0].ravel(), self.values_3D[:, :, 1].ravel(),
            self.values_3D[:, :, 2].ravel())).T.astype("float64")

        if np.any(self.regular_grid.extent[:4] - self.extent) != 0:
            print('obacht')
            # todo if grid extent bigger fill missing values with nans for chloe
            self._crop()

        if np.any(self.regular_grid.resolution[:2] - self.resolution) != 0:
            self._resize()
        else:
            self.values_3D_res = self.values_3D

        self.regular_grid.mask_topo = self._create_grid_mask()

    def _crop(self):
        pass

    def _resize(self):
        self.values_3D_res = skimage.transform.resize(self.values_3D,
                                                      (self.regular_grid.resolution[0], self.regular_grid.resolution[1]),
                                                      mode='constant',
                                                      anti_aliasing=False, preserve_range=True)

    def show(self):
        from gempy.plot.helpers import add_colorbar
        if self.type == 'artificial':
            fig, ax = plt.subplots()
            CS= ax.contour(self.values_3D[:, :, 2], extent=(self.extent[:4]), colors='k', linestyles='solid')
            ax.clabel(CS, inline=1, fontsize=10, fmt='%d')
            CS2 = ax.contourf(self.values_3D[:, :, 2], extent=(self.extent[:4]), cmap='terrain')
            add_colorbar(axes=ax, label='elevation [m]', cs=CS2)
        else:
            im = plt.imshow(np.flipud(self.values_3D[:,:,2]), extent=(self.extent[:4]))
            add_colorbar(im=im, label='elevation [m]')
        plt.axis('scaled')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Model topography')

    def save(self, filepath):
        """
        Save the topography file in a numpy array which can be loaded later, to avoid the gdal process.
        Args:
            filepath (str): path where the array should be stored.

        Returns:

        """
        np.save(filepath, np.array([self.values_3D, self.extent, self.resolution]))
        print('saved')

    def _create_grid_mask(self):
        ind = self._find_indices()
        gridz = self.regular_grid.values[:, 2].reshape(*self.regular_grid.resolution).copy()
        for x in range(self.regular_grid.resolution[0]):
            for y in range(self.regular_grid.resolution[1]):
                z = ind[x, y]
                gridz[x, y, z:] = 99999
        mask = (gridz == 99999)
        return mask# np.multiply(np.full(self.regular_grid.values.shape, True).T, mask.ravel()).T

    def _find_indices(self):
        zs = np.linspace(self.regular_grid.extent[4], self.regular_grid.extent[5], self.regular_grid.resolution[2])
        dz = (zs[-1] - zs[0]) / len(zs)
        return ((self.values_3D_res[:, :, 2] - zs[0]) / dz + 1).astype(int)

    def interpolate_zvals_at_xy(self, xy):
        assert xy[:, 0][0] <= xy[:, 0][-1], 'At the moment, the xy values of the first point must be smaller than second' \
                                            '(fix soon)'
        assert xy[:, 1][0] <= xy[:, 1][-1], 'At the moment, the xy values of the first point must be smaller than second' \
                                            '(fix soon)'
        xj = self.values_3D[:, :, 0][0, :]
        yj = self.values_3D[:, :, 1][:, 0]
        zj = self.values_3D[:, :, 2].T
        f = interpolate.RectBivariateSpline(xj, yj, zj)
        zi = f(xy[:, 0], xy[:, 1])
        return np.diag(zi)

#######
# Author:George

class GravityPreprocessingRegAllLoop(RegularGrid):
    '''
      Numpy dependent Convolotional regular grid setup
    '''
  
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
        self.radius_cell_x = int(model_radius[0]//dx)
        self.radius_cell_y = int(model_radius[1]//dy)


        # kernel_centers = np.repeat(self.model.grid.regular_grid.values[:,:,np.newaxis],self.num_receivers,axis=2)-self.model.xy_ravel.T
        kernel_centers = np.squeeze(
                self.model.grid.regular_grid.values[:, :, np.newaxis] - new_xy_ravel_temp
        )

        c_x = int(self.center_index_x[0])
        c_y = int(self.center_index_y[0])

        slice_kernel_centers = (kernel_centers[:,:].reshape(regular_grid_resolution+[3,self.num_receivers])[c_x-self.radius_cell_x:c_x+self.radius_cell_x+1,c_y-self.radius_cell_y:c_y+self.radius_cell_y+1,:,:]).reshape([-1,3])


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


        G = 6.674e-3  # ugal     cm3⋅g−1⋅s−26.67408e-2 -- 1 m/s^2 to microgal =

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