import multiprocessing

import GPUtil

import tensorflow as tf

import sys
import os
print(os.getcwd())
sys.path.append('/home/ib012512/Documents/Geophysics/GP_old')
sys.path.append('/home/ib012512/Documents/Geophysics/models')
sys.path.append('/home/ib012512/Documents/Geophysics/utils')

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

from Model_KV import *
from Prob_KV_density import *
from funcs import *
from derivative import *
from util import *

tfd = tfp.distributions

args = dotdict({
    'learning_rate': 0.02,
    'Adam_iterations':30,
    'num_results': 3,
    'number_burnin': 0,
    'RMH_step_size': 0.002,
    'HMC_step_size': 0.00003,
    'leapfrogs':20,
    'gpCN_step_size': 0.3,
    'number_init': 2
})
tfdtype = tf.float64

class forwardFunction():
    def __init__(self,gravUtil,num_para) -> None:
        self.gravUtil = gravUtil
        self.num_para = num_para
    def concat_xy_and_scale(self,mu,model):
        mu = tf.concat([static_xy,tf.reshape(mu,[self.num_para,1])],axis=-1)
        return self.gravUtil.scale(tf.reshape(mu,sfp_shape),model )

    # forward function including proper normalization
    def __call__(self,mu,model,norm,sigmoid = True,):
        '''This function rescale and manimulate the input data to the gravity function
                mu is scaled and need to be concatenate with other parameters
        '''
        mu_ = norm.denormalize(mu) # convert the mu to real scale
        # partitioning surface points and properties values from the paramteres set
        mu_sfp = mu_[:self.num_para]
        density = mu_[-3:]
        mu0 = self.concat_xy_and_scale(mu_sfp,model)  # concatenate the xy coordinates to the z values and convert to gempy scale
        gravity = self.gravUtil.forward(mu0,density,model,sigmoid) # calculate gravity
        return gravity


######### Model configuration #########
## setup variable
modelName = "modelKV_Den"

path = '/home/ib012512/Documents/Geophysics/Data/'
regular_grid_resolution = [100,100,50]
center_grid_resolution = [8,8,15]


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

# define the Z position of receivers
# Z_r = 400
# xyz = np.meshgrid(X_r, Y_r, Z_r)
# xy_ravel = np.vstack(list(map(np.ravel, xyz))).T
n_devices = receivers.shape[0]

######### load data #########


# data_df = pd.read_csv('/home/ib012512/Documents/Geophysics/gravity.CSV')
# extract_df = pd.DataFrame()

# X=np.array(pd.DataFrame(data_df, columns=['East_KKJ']))
# Y=np.array(pd.DataFrame(data_df, columns=['North_KKJ']))
# Gravity=np.array(pd.DataFrame(data_df, columns=['TC+Curv310']))

# data=np.concatenate([np.concatenate([X,Y],axis=1),Gravity],axis=1)
# X=data[:, 0]
# Y=data[:, 1]
# Z=data[:, 2]

# xx = X_r
# yy = Y_r
# triang = tri.Triangulation(X, Y)
# interpolator = tri.LinearTriInterpolator(triang, Z)
# XX,YY = np.meshgrid(xx,yy)
# data_interpo = interpolator(XX, YY).data.flatten()
# Gravity_measurement = data_interpo - data_interpo.min()
# data_measurement = np.vstack((receivers.T, Gravity_measurement)).T

# ######### Prior model #########
# model_prior = ModelKV(path,
#               receivers = receivers,
#               dtype = 'float64',
#               center_grid_resolution=center_grid_resolution,
#               regular_grid_resolution=regular_grid_resolution,
#               model_radius=[1200,1200,2000])

# # get input from GemPy
# model_prior.activate_centered_grid()
# g = GravityPreprocessing(model_prior.grid.centered_grid)
# tz = g.set_tz_kernel()

# gempy_input = model_prior.get_graph_input()
# model_prior.create_tensorflow_graph(gempy_input,gradient = True)
# size = tf.constant((model_prior.center_grid_resolution[0]+1)*(model_prior.center_grid_resolution[1]+1)*(model_prior.center_grid_resolution[2]),tf.int32)

# num_para = model_prior.surface_points.df[['X','Y','Z']].to_numpy().shape[0]
# num_para_total = num_para + 3
# sfp_shape = [num_para,3]
# # define the static x y coordinates
# x = model_prior.surface_points.df[['X','Y','Z']].to_numpy()
# static_xy = x[:,0:2]

# # initializa the gravity forward function
# grav_Util = gravFuncs(size,n_devices,tz,gempy_input)
# forward_function = forwardFunction(grav_Util,num_para)

# ## define prior distribution
# # define prior surface points mean and std
# sfp_mean = model_prior.surface_points.df[['X','Y','Z']].to_numpy()[:,2]
# sfp_std = tfconstant(250*np.ones([num_para]))

# # define prior density mean and std
# density_prior = tfconstant([2.8, 3.2, 3.])
# density_std = tfconstant([0.2,0.3,0.5])

# prior_mean = tf.concat([sfp_mean,density_prior],axis = 0)
# prior_std = tf.concat([sfp_std,density_std],axis = 0)

# norm = gaussianNormalizer(prior_mean,prior_std)

# ## define the statistic model
# stat_model = Stat(model_prior,forward_function,num_para_total)
# mean_prior_norm = norm.normalize(prior_mean) 

# stat_model.set_prior(Number_para = num_para,norm = norm)
# stat_model.set_density_prior(Number_para = 3)

# Data_measurement = data_measurement[:,2]
# Data_std = 1.5
# stat_model.set_likelihood(Data_measurement,Data_std)
# stat_model.monitor=True
##############################

# mu0 = norm.normalize(prior_mean)

# ######### Calculate MAP #########
# # define the Derivative object
# derivatives = GradHess(model_prior,stat_model,num_para_total)
# mu_list,cost_A,MAP,Adam_time = derivatives.Find_MAP(mu0,args.learning_rate,args.Adam_iterations)

# print('MAP:',MAP)
# print('Adam: %.3f' % (Adam_time))
# ######### Calculate Hessian #########
# Hess = derivatives.calculate_Hessian(MAP)
# print('Hessian Matrix:',Hess)

# number_init = args.number_init
# mu_init_list = np.random.normal(size = (num_para_total,number_init))


def MAPHESSEval(mu_init,MAP_list,Hess_list):
  data_df = pd.read_csv('/home/ib012512/Documents/Geophysics/gravity.CSV')
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

  ######### Prior model #########
  model_prior = ModelKV(path,
                receivers = receivers,
                dtype = 'float64',
                center_grid_resolution=center_grid_resolution,
                regular_grid_resolution=regular_grid_resolution,
                model_radius=[1200,1200,2000])

  # get input from GemPy
  
  print('######### FINE #########')
  model_prior.activate_centered_grid()
  g = GravityPreprocessing(model_prior.grid.centered_grid)
  tz = g.set_tz_kernel()

  gempy_input = model_prior.get_graph_input()
  model_prior.create_tensorflow_graph(gempy_input,gradient = True)
  size = tf.constant((model_prior.center_grid_resolution[0]+1)*(model_prior.center_grid_resolution[1]+1)*(model_prior.center_grid_resolution[2]),tf.int32)

  num_para = model_prior.surface_points.df[['X','Y','Z']].to_numpy().shape[0]
  num_para_total = num_para + 3
  sfp_shape = [num_para,3]
  # define the static x y coordinates
  x = model_prior.surface_points.df[['X','Y','Z']].to_numpy()
  static_xy = x[:,0:2]

  # initializa the gravity forward function
  grav_Util = gravFuncs(size,n_devices,tz,gempy_input)
  forward_function = forwardFunction(grav_Util,num_para)

  ## define prior distribution
  # define prior surface points mean and std
  sfp_mean = model_prior.surface_points.df[['X','Y','Z']].to_numpy()[:,2]
  sfp_std = tfconstant(250*np.ones([num_para]))

  # define prior density mean and std
  density_prior = tfconstant([2.8, 3.2, 3.])
  density_std = tfconstant([0.2,0.3,0.5])

  prior_mean = tf.concat([sfp_mean,density_prior],axis = 0)
  prior_std = tf.concat([sfp_std,density_std],axis = 0)

  norm = gaussianNormalizer(prior_mean,prior_std)
  ## define the statistic model
  stat_model = Stat(model_prior,forward_function,num_para_total)
  mean_prior_norm = norm.normalize(prior_mean) 

  stat_model.set_prior(Number_para = num_para,norm = norm)
  stat_model.set_density_prior(Number_para = 3)

  Data_measurement = data_measurement[:,2]
  Data_std = 1.5
  stat_model.set_likelihood(Data_measurement,Data_std)
  stat_model.monitor=True

  ######### Calculate MAP #########
  # define the Derivative object
  derivatives = GradHess(model_prior,stat_model,num_para_total)
  mu_list,cost_A,MAP,Adam_time = derivatives.Find_MAP(mu_init,args.learning_rate,args.Adam_iterations)

  print('MAP:',MAP)
  print('Adam: %.3f' % (Adam_time))

  ######### Calculate Hessian #########
  Hess = derivatives.calculate_Hessian(MAP)
  print('Hessian Matrix:',Hess)
  MAP_list.append(MAP)
  Hess_list.append(Hess)

num_para_total = 17
number_init = args.number_init

s = tfd.Sample(
        tfd.Normal(loc=tfconstant(0), scale=tfconstant(1)),
        sample_shape=(number_init,num_para_total))
mu_init_list = s.sample()

print('%%%%%%%%%%%Start MultiProcessing%%%%%%%%%%%%%')
# use multiprocessing to free up GPU memory after each iteration
with multiprocessing.Manager() as manager:
  # Time_list = manager.list()
  MAP_list = manager.list()
  Hess_list = manager.list()
  for mu_init_ in mu_init_list:
    # process_eval = multiprocessing.Process(target=Hess, args=(y,Time_list))
    process_eval = multiprocessing.Process(target=MAPHESSEval, args=(mu_init_,MAP_list,Hess_list))
    process_eval.start()
    
    process_eval.join()

  # Time_list = np.array(Time_list)
  MAP_list = np.array(MAP_list)
  Hess_list = np.array(Hess_list)

  json_dump = json.dumps({'MAP_list': MAP_list,
                          'Hess_list': Hess_list}, cls=NumpyEncoder)
  
  with open('/home/ib012512/Documents/Results/multimodal01.json', 'w') as outfile:
      json.dump(json_dump, outfile)

print('%%%%%%%%%%%Done%%%%%%%%%%%%%')
