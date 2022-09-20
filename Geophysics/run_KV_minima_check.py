import tensorflow as tf
import nptyping 
tf.test.gpu_device_name()
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

import sys
import os
print(os.getcwd())
sys.path.append('./Geophysics/GP_old')
sys.path.append('./Geophysics/models')
sys.path.append('./Geophysics/utils')

# suppress warinings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
# dependency
# import gempy as gp
import tensorflow_probability as tfp
# import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.tri as tri

## gempy utils 
# from gempy import create_data, map_series_to_surfaces
from gempy.assets.geophysics import GravityPreprocessing
from gempy.plot.visualization_2d_pro import *
# from gempy.core.grid_modules.grid_types import CenteredGrid,CenteredRegGrid,RegularGrid

#
from Model_KV import *
from Prob_KV_density import *
from util import *

tfd = tfp.distributions
print(nptyping.__version__)
print(tf.__version__)
print(tfp.__version__)


## define the class and functions
# tensorflow constant
def constant64(x):
    return tf.constant(x, dtype=tf.float64)

class gaussianNormalizer(object):
  def __init__(self,mean, std):
    self.mean = mean
    self.std = std
  def normalize(self,mu):
    return (mu-self.mean)/self.std
  def denormalize(self,mu):
    return mu*self.std + self.mean

# gravity forward simulation
def forward(mu,density,model,sigmoid = True):
    TFG = TFGraph(input, model.fault_drift,
                    model.grid_tensor, model.values_properties, model.nugget_effect_grad,model.nugget_effect_scalar, model.Range,
                    model.C_o, model.rescale_factor,slope = 500, dtype = model.tfdtype, gradient = True,sigmoid = sigmoid)

    values_properties = tf.stack([tfconstant([1.,2.,3.]),density],axis=0)

    formation_block,property_block,block_mask =TFG.compute_series(mu,
                model.dips_position,
                model.dip_angles,
                model.azimuth,
                model.polarity,
                values_properties)
    
    size_property_block = size*n_devices
    densities = property_block[0:size_property_block]
    grav_convolution_full = tf.TensorArray(model.tfdtype, size=n_devices, dynamic_size=False, clear_after_read=True)
    for i in tf.range(n_devices):
      windowed_densities = densities[i*size:(i+1)*size]
      grav_ = TFG.compute_forward_gravity(tfconstant(tz), 0, size, windowed_densities)
      grav_convolution_full = grav_convolution_full.write(i, grav_)
    grav_convolution_full = tf.squeeze(grav_convolution_full.stack())
    grav_convolution_full =  tf.reduce_max(grav_convolution_full) - grav_convolution_full
    return grav_convolution_full

# gempy scale function
def unscale(scaled_value,model):
  return (scaled_value-0.5001)*model.rf+model.centers
def scale(unscaled_value,model):
  return (unscaled_value - model.centers)/model.rf + 0.5001


def concat_xy_and_scale(mu,model):
  mu = tf.concat([static_xy,tf.reshape(mu,[num_para,1])],axis=-1)
  return scale(tf.reshape(mu,sfp_shape),model )

# forward function including proper normalization
def forward_function(mu,model,norm,sigmoid = True):
  '''This function rescale and manimulate the input data to the gravity function
        mu is scaled and need to be concatenate with other parameters
  '''
  mu_norm = norm.denormalize(mu)
  mu_sfp = mu_norm[:num_para]
  density = mu_norm[-3:]
  mu0 = concat_xy_and_scale(mu_sfp,model)

  gravity = forward(mu0,density,model,sigmoid)

  return gravity



## setup variable
modelName = "modelKV_minimum_check"

path = './Geophysics/Data/'
regular_grid_resolution = [100,100,50]
center_grid_resolution = [8,8,15]

tfconstant = constant64
tfdtype = tf.float64



## Model set-up

# # define the gravity receiver locations
n = 10
m = 10
X_r = np.linspace(3495500, 3500000, num = n)
Y_r = np.linspace(7509500, 7513000, num = m)

r = []
for x in X_r:
  for y in Y_r:
    r.append(np.array([x,y]))
receivers = np.array(r)

Z_r = 400
xyz = np.meshgrid(X_r, Y_r, Z_r)
xy_ravel = np.vstack(list(map(np.ravel, xyz))).T
n_devices = receivers.shape[0]

####### load data #########


data_df = pd.read_csv('./Geophysics/gravity.CSV')
extract_df = pd.DataFrame()

X=np.array(pd.DataFrame(data_df, columns=['East_KKJ']))
Y=np.array(pd.DataFrame(data_df, columns=['North_KKJ']))
Gravity=np.array(pd.DataFrame(data_df, columns=['TC+Curv310']))

data=np.concatenate([np.concatenate([X,Y],axis=1),Gravity],axis=1)
X=data[:, 0]
Y=data[:, 1]
Z=data[:, 2]

xx = X_r
yy = Y_r
triang = tri.Triangulation(X, Y)
interpolator = tri.LinearTriInterpolator(triang, Z)
XX,YY = np.meshgrid(xx,yy)
data_interpo = interpolator(XX, YY).data.flatten()
Gravity_measurement = data_interpo - data_interpo.min()
data_measurement = np.vstack((receivers.T, Gravity_measurement)).T

########################
## Prior model
model_prior = ModelKV(path,
              receivers = receivers,
              dtype = 'float64',
              center_grid_resolution=center_grid_resolution,
              regular_grid_resolution=regular_grid_resolution,
              model_radius=[1200,1200,2000])

# get input from GemPy
model_prior.activate_centered_grid()
g = GravityPreprocessing(model_prior.grid.centered_grid)
tz = g.set_tz_kernel()

input = model_prior.get_graph_input()
model_prior.create_tensorflow_graph(input,gradient = True)
size = tf.constant((model_prior.center_grid_resolution[0]+1)*(model_prior.center_grid_resolution[1]+1)*(model_prior.center_grid_resolution[2]),tf.int32)



num_para = model_prior.surface_points.df[['X','Y','Z']].to_numpy().shape[0]
sfp_shape = [num_para,3]
# define the static x y coordinates
x = model_prior.surface_points.df[['X','Y','Z']].to_numpy()
static_xy = x[:,0:2]



## define prior distribution
sfp_mean = model_prior.surface_points.df[['X','Y','Z']].to_numpy()[:,2]
sfp_std = tfconstant(250*np.ones([num_para]))

density_prior = tfconstant([2.8, 3.2, 3.])
density_std = tfconstant([0.2,0.3,0.5])

prior_mean = tf.concat([sfp_mean,density_prior],axis = 0)
prior_std = tf.concat([sfp_std,density_std],axis = 0)

norm = gaussianNormalizer(prior_mean,prior_std)

num_para_total = num_para + 3

stat_model = Stat(model_prior,forward_function,num_para_total)
mean_prior_norm = norm.normalize(prior_mean)

stat_model.set_prior(Number_para = num_para,norm = norm)
stat_model.set_density_prior(Number_para = 3)

Data_measurement = data_measurement[:,2]
Data_std = 0.2
stat_model.set_likelihood(Data_measurement,Data_std)
stat_model.monitor=True

def hvp(mu,tangents):
    with tf.autodiff.ForwardAccumulator(mu, tangents) as acc:
        with tf.GradientTape(watch_accessed_variables=False) as t:
            t.watch(mu)
            joint_log = tf.negative(stat_model.joint_log_post(mu))
        loss = t.gradient(joint_log,mu)
    hess = acc.jvp(loss)
    return(hess)

def calculate_Hessian(MAP,time = False):
    Hess_list = []
    start = timeit.default_timer()
    for i in range(num_para_total):
        tangents = np.zeros(MAP.shape)
        tangents[i]=1
        tangents = tf.convert_to_tensor(tangents,dtype=tfdtype)

        Hess_list.append(hvp(MAP,tangents).numpy())
    Hess = np.array(Hess_list)
    end = timeit.default_timer()
    if time == True:
        print('time for Hessian calculation: %.3f' % (end - start))
        return Hess, end-start
    return Hess


##############MAP############
          
def Search_MAP(mu_init):
  @tf.function
  def loss_(mu):
          lost =  tf.negative(stat_model.joint_log_post(mu,monitor = False))
          return lost

  @tf.function
  def loss_minimize():
      lost =  tf.negative(stat_model.joint_log_post(mu_init,monitor = False))
      return lost

  def Adam(mu_init,learning_rate,iterations,mu_list,cost_A):


      optimizer = tf.keras.optimizers.Adam(
                  learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08
              )

      start = timeit.default_timer()
      for step in range(iterations):

          optimizer.minimize(loss_minimize, var_list=[mu_init])
          loss = loss_(mu_init).numpy()
          if cost_A:
              # if (cost_A[-1]-loss)>0 and (cost_A[-1]-loss)<1e-4: 
              if  step > 1000 and (cost_A[-1]-loss)<1e-5: 
                  break # check if cost list is empty
          
          cost_A.append(loss)
          if step%100 == 0:
            print ('step:',step,'loss:',loss)

          mu_list.append(mu_init.numpy())
      end = timeit.default_timer()
      print('Adam: %.3f' % (end - start))
      MAP = tf.convert_to_tensor(mu_list[-1],tfdtype)
      return mu_list,cost_A,MAP

  def find_MAP(mu_init,learning_rate,iterations,new_MAP = True,save = True,path = None):
    cost_A = []
    mu_list = []    
    if new_MAP == True:
      mu_list,cost_A,MAP = Adam(mu_init,learning_rate,iterations,mu_list,cost_A)
      if save == True:
        # save MAP
        json_dump = json.dumps({'MAP': MAP.numpy()}, cls=NumpyEncoder)
        with open(path, 'w') as outfile:
            json.dump(json_dump, outfile)

      return mu_list,cost_A,MAP
    else:
      #load MAP
      with open(path) as f:
            data = json.load(f)
      data = json.loads(data)
      MAP = tf.convert_to_tensor(np.asarray(data['MAP']),dtype = tfdtype)
      return mu_list,cost_A,MAP

  path = './Results/'+modelName+'MAP_density.json'
  mu_init = tf.Variable(mu_init,tfdtype)
  learning_rate = 0.02
  iterations = 2000

  # start = timeit.default_timer()
  mu_list,cost_A,MAP = find_MAP(mu_init,learning_rate,iterations,path = path,save = False)
  # end = timeit.default_timer()
  # print('Adam: %.3f' % (end - start))

  Hess = calculate_Hessian(MAP)
  # eigval,eigvec = tf.linalg.eigh(Hess)
  print('GPU Memory:',tf.config.experimental.get_memory_info('GPU:0')["current"])
  # print('MAP:',MAP.numpy())
  return cost_A,MAP.numpy(),Hess

num_init = 10
s = tfd.Sample(
        tfd.Normal(loc=tfconstant(0), scale=tfconstant(1)),
        sample_shape=(num_init,num_para_total))
mu0_list = s.sample()

MAPs = []
Hesslist = []
costs = []
for mu_init in mu0_list:
  # mu_init = tfconstant(mu_init)
  print('mu_init:',mu_init)
  cost_A,MAP,Hess = Search_MAP(mu_init)
  # MAPs.append(MAP)
  Hesslist.append(Hess)
  costs.append(cost_A)
  print_GPUinfo()


json_dump = json.dumps({
                        'MAPs':MAPs,
                        'Hess':Hesslist,
                        'loss':costs,
                        'init_list':mu0_list
                        }, cls=NumpyEncoder)
with open('./Results/MCMC_'+modelName+'.json', 'w') as outfile:
    json.dump(json_dump, outfile)



# print('Hess eigen:',eigval)
print('Done')

