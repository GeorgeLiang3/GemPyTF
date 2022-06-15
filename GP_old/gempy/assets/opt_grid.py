"""
Author:@Zhouji Liang
optimization grid based on T_z of each cell
"""
import tensorflow as tf
from gempy.core.grid_modules.grid_types import CenteredGrid,RegularGrid

## auxiliary functions

def constant32(x):
    return tf.constant(x, dtype=tf.float32)

def constant64(x):
    return tf.constant(x, dtype=tf.float64)

def softmax_space(parameters):
    return tf.nn.softmax(parameters)


def make_variables(k, initializer,dtype = tf.float32):
    '''
        create TensorFlow variable with shape k
        initializer: random variable initializer
    '''
    return tf.Variable(initializer(shape=[k], dtype=dtype))

## define the data type
tfconstant = constant32


class Optimizing():
  def __init__(self,center_grid_resolution,radius):

    self.radius = tfconstant(radius)
    min_ = -1
    max_ = 1
    self._a = make_variables(tf.cast(center_grid_resolution[0]/2+1,tf.int32),initializer= tf.random_uniform_initializer(minval=min_, maxval=max_, seed=None),dtype = tf.float32)
    self._b = make_variables(tf.cast(center_grid_resolution[1]/2+1,tf.int32),initializer= tf.random_uniform_initializer(minval=min_, maxval=max_, seed=None),dtype = tf.float32)
    self._c = make_variables(tf.cast(center_grid_resolution[2],tf.int32),initializer= tf.random_uniform_initializer(minval=min_, maxval=max_, seed=None),dtype = tf.float32)

  def loss(self,a,b,c):
    tz = compute_tz(a,b,c,self.radius)
    l = tf.math.reduce_std(tz)
    return l

  @tf.function
  def train_step(self,opt):
    with tf.GradientTape() as tape:
      loss_value = self.loss(self._a,self._b,self._c)
      grads = tape.gradient(loss_value,[self._a,self._b,self._c])

    opt.apply_gradients(zip(grads, [self._a,self._b,self._c]))
    return loss_value

  def __call__(self,epochs = 1000):

    opt = tf.keras.optimizers.Adam(
      learning_rate=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,
      name='Adam')

    for epoch in range(epochs):
      # start_time = time.time()
      loss_value = self.train_step(opt)
      if epoch%200==0:
        print("\nStart of epoch %d" % (epoch,))
        print("loss: %.10f" % (float(loss_value),))
        # print("Time taken: %.2fs" % (time.time() - start_time))
    
    return self._a,self._b,self._c


class OptimalGrid(CenteredGrid):
    """
    ML optimized spaced grid.
    """

    def __init__(self, centers=None, radius=None, resolution=None, abc = None):
        super().__init__(centers=None, radius=None, resolution=None)
        self.grid_type = 'centered_grid'
        self.values = np.empty((0, 3))
        self.length = self.values.shape[0]
        self.kernel_centers = np.empty((0, 3))
        self.kernel_dxyz_left = np.empty((0, 3))
        self.kernel_dxyz_right = np.empty((0, 3))
        self.tz = np.empty(0)

        if centers is not None and radius  is not None and abc is None:
            # if resolution is None:
            #     resolution = [10, 10, 20]
            
            f = Optimizing(resolution,radius)
            a,b,c = f(epochs=1000)
            self.abc = [a,b,c]

            self.set_centered_grid( centers = centers,radius=radius, resolution=resolution,a=a,b=b,c=c)
        
        # with a given optimized grid
        elif centers is not None and radius  is not None and abc is not None: 
            a, b, c = [abc[i] for i in range(3)]
            self.set_centered_grid( centers = centers,radius=radius, resolution=resolution,a=a,b=b,c=c)

###################



    @staticmethod
    def create_irregular_grid_kernel(radius,a=None,b=None,c=None):
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

        # a,b,c = find_optimal_grid(a,b,c,epochs)
   
        g_x = tf.cumsum(softmax_space(a))
        g_y = tf.cumsum(softmax_space(b))
        g_z = tf.cumsum(softmax_space(c))

        g2_x = tf.concat((-g_x[::-1], g_x),axis=0) * radius[0]
        g2_y = tf.concat((-g_y[::-1], g_y),axis=0) * radius[1]
        g2_z =  (tf.concat(([0], g_z),axis=0) + 0.005) * - radius[2]

        x_center = (g2_x[:-1]+g2_x[1:])/2 
        y_center = (g2_y[:-1]+g2_y[1:])/2
        z_center = (g2_z[:-1]+g2_z[1:])/2

        g = tf.meshgrid(x_center,y_center,z_center)

        d_left_x = tf.math.abs(g2_x[:-1] - x_center)
        d_left_y = tf.math.abs(g2_y[:-1] - y_center)
        d_right_x = tf.math.abs(g2_x[1:] - x_center)
        d_right_y = tf.math.abs(g2_y[1:] - y_center)
        d_z = z_center - g2_z[:-1]

        d_left = tf.meshgrid(d_left_x,d_left_y,d_z)
        d_right = tf.meshgrid(d_right_x,d_right_y,d_z)

        kernel_g = tf.concat([tf.reshape(g[0],[-1,1]),tf.reshape(g[1],[-1,1]),tf.reshape(g[2],[-1,1])],axis=1)
        kernel_d_left = tf.concat([tf.reshape(d_left[0],[-1,1]),tf.reshape(d_left[1],[-1,1]),tf.reshape(d_left[2],[-1,1])],axis=1)
        kernel_d_right = tf.concat([tf.reshape(d_right[0],[-1,1]),tf.reshape(d_right[1],[-1,1]),tf.reshape(d_right[2],[-1,1])],axis=1)

        return kernel_g.numpy(), kernel_d_left.numpy(), kernel_d_right.numpy()
  

    def set_centered_kernel(self, resolution, radius,a=None,b=None,c=None):
        """
        Set a centered

        Args:
            resolution: [s0]
            radius (float): Maximum distance of the kernel

        Returns:

        """
        self.kernel_centers, self.kernel_dxyz_left, self.kernel_dxyz_right = self.create_irregular_grid_kernel(
             radius,a=a,b=b,c=c)

        return self.kernel_centers


# centered optimized grid
def compute_tz(a,b,c,radius):
    g_x = tf.cumsum(softmax_space(a))
    g_y = tf.cumsum(softmax_space(b))
    g_z = tf.cumsum(softmax_space(c))

    g2_x = tf.concat((-g_x[::-1], g_x),axis=0) * radius[0]
    g2_y = tf.concat((-g_y[::-1], g_y),axis=0) * radius[1]
    g2_z =  (tf.concat(([0], g_z),axis=0) + 0.005) * - radius[2]

    x_center = (g2_x[:-1]+g2_x[1:])/2 
    y_center = (g2_y[:-1]+g2_y[1:])/2
    z_center = (g2_z[:-1]+g2_z[1:])/2

    g = tf.meshgrid(x_center,y_center,z_center)

    d_left_x = tf.math.abs(g2_x[:-1] - x_center)
    d_left_y = tf.math.abs(g2_y[:-1] - y_center)
    d_right_x = tf.math.abs(g2_x[1:] - x_center)
    d_right_y = tf.math.abs(g2_y[1:] - y_center)
    d_z = z_center - g2_z[:-1]

    d_left = tf.meshgrid(d_left_x,d_left_y,d_z)
    d_right = tf.meshgrid(d_right_x,d_right_y,d_z)

    kernel_g = tf.concat([tf.reshape(g[0],[-1,1]),tf.reshape(g[1],[-1,1]),tf.reshape(g[2],[-1,1])],axis=1)
    kernel_d_left = tf.concat([tf.reshape(d_left[0],[-1,1]),tf.reshape(d_left[1],[-1,1]),tf.reshape(d_left[2],[-1,1])],axis=1)
    kernel_d_right = tf.concat([tf.reshape(d_right[0],[-1,1]),tf.reshape(d_right[1],[-1,1]),tf.reshape(d_right[2],[-1,1])],axis=1)

    s_gr_x = kernel_g[:, 0]
    s_gr_y = kernel_g[:, 1]
    s_gr_z = kernel_g[:, 2]

    # # getting the coordinates of the corners of the voxel...
    x_cor = tf.transpose(tf.stack([s_gr_x - kernel_d_left[:, 0], s_gr_x + kernel_d_right[:, 0]], axis=0))
    y_cor = tf.transpose(tf.stack((s_gr_y - kernel_d_left[:, 1], s_gr_y + kernel_d_right[:, 1]), axis=0))
    z_cor = tf.transpose(tf.stack((s_gr_z - kernel_d_left[:, 2], s_gr_z + kernel_d_right[:, 2]), axis=0))

    x_matrix = tf.repeat(x_cor, 4, axis=1)
    y_matrix = tf.tile(tf.repeat(y_cor, 2, axis=1), (1, 2))
    z_matrix = tf.tile(z_cor, (1, 4))

    s_r = tf.sqrt(x_matrix ** 2 + y_matrix ** 2 + z_matrix ** 2)

    mu = tfconstant([1, -1, -1, 1, -1, 1, 1, -1])

    G = tfconstant(6.674e-3)

    tz = (
        G *
        tf.reduce_sum(- 1 *
                mu * (
                        x_matrix * tf.math.log(y_matrix + s_r) +
                        y_matrix * tf.math.log(x_matrix + s_r) -
                        z_matrix * tf.math.atan(x_matrix * y_matrix / (z_matrix * s_r))),
                axis=1))
    return tz