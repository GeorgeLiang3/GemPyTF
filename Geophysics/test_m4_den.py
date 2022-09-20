import tensorflow as tf
import nptyping 
# tf.test.gpu_device_name()
# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

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

from Model_new import *
from Prob_density import *
from funcs import *
from derivative import *

tfd = tfp.distributions

args = dotdict({
    'learning_rate': 0.02,
    'Adam_iterations':30,
    'num_results': 3,
    'number_burnin': 0,
    'RMH_step_size': 0.002,
    'HMC_step_size': 0.00003,
    'leapfrogs':20,
    'gpCN_step_size': 0.3
})
tfdtype = tf.float64

class forwardFunction():
    def __init__(self,gravUtil,num_per_layer) -> None:
        self.gravUtil = gravUtil
        self.num_per_layer = num_per_layer
    def concat_xy_and_scale(self,mu,model):
        mu = tf.concat([static_xy,tf.reshape(mu,[num_para,1])],axis=-1)
        return self.gravUtil.scale(tf.reshape(mu,sfp_shape),model )

    # forward function including proper normalization
    def __call__(self,mu,model,norm,sigmoid = True,):
        '''This function rescale and manimulate the input data to the gravity function
                mu is scaled and need to be concatenate with other parameters
        '''
        mu_ = norm.denormalize(mu)
        mu_upper = mu_[self.num_per_layer:-3]
        mu_lower = mu_upper - mu_[:self.num_per_layer]
        density = mu_[-3:]
        mu0 = tf.concat([mu_lower,mu_upper],axis = 0)
        mu0 = self.concat_xy_and_scale(mu0,model)
        gravity = self.gravUtil.forward(mu0,density,model)
        return gravity


######### Model configuration #########
## setup variable
modelName = "model4flatprior"

path = './Geophysics/GP_old/notebooks'
orientation_path = "/data/input_data/George_models/model4_2_orientations.csv"
surface_path = '/data/input_data/George_models/'+'model3'+'_surface_points.csv'
regular_grid_resolution = [1,1,1]
center_grid_resolution = [10,10,30]

# # define the gravity receiver locations
num_ = 6
X = np.linspace(200,800,num_)
Y = np.linspace(200,800,num_)
r = []
for x in X:
  for y in Y:
    r.append(np.array([x,y]))
receivers = np.array(r)

# define the Z position of receivers
Z_r = 400
xyz = np.meshgrid(X, Y, Z_r)
xy_ravel = np.vstack(list(map(np.ravel, xyz))).T
n_devices = receivers.shape[0]

######### load data #########


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
forward_function = forwardFunction(grav_Util)

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
Data_std = 0.2
stat_model.set_likelihood(Data_measurement,Data_std)
stat_model.monitor=True


mu0 = norm.normalize(prior_mean)
grav = forward_function(mu0,model_prior,norm)
print('grav:',grav)

######### Calculate MAP #########
# define the Derivative object
derivatives = GradHess(model_prior,stat_model,num_para_total)
mu_list,cost_A,MAP,Adam_time = derivatives.Find_MAP(mu0,args.learning_rate,args.Adam_iterations)

print('MAP:',MAP)
print('Adam: %.3f' % (Adam_time))
######### Calculate Hessian #########
Hess = derivatives.calculate_Hessian(MAP)
print('Hessian Matrix:',Hess)

######### MCMC #########

initial_chain_state = [MAP]
num_results = args.num_results
number_burnin = args.number_burnin


# RMH
step_size = args.RMH_step_size
states_RMH = stat_model.run_MCMC('RMH',
                                 num_results = num_results,
                                 number_burnin = number_burnin,
                                 step_size = step_size,
                                 initial_chain_state =initial_chain_state
                                  )
accept_RMH = (tf.reduce_mean(tf.cast(states_RMH.trace.is_accepted,tfdtype)))*100
print(f'acceptance rate Random-walk Matroplis {num_results:d} {accept_RMH:.2f}%')

# HMC
step_size = args.HMC_step_size
states_HMC = stat_model.run_MCMC(method='HMC',
                                num_results = num_results,
                                number_burnin=number_burnin,
                                step_size=step_size,
                                num_leapfrog_steps=args.leapfrogs,
                                initial_chain_state =initial_chain_state)
accept_HMC = (tf.reduce_mean(tf.cast(states_HMC.trace.is_accepted,tfdtype)))*100
print(f'acceptance rate Hamiltonian Monte Carlo {num_results:d} {accept_HMC:.2f}%')

# gpCN
step_size = args.gpCN_step_size
accepted_samples_gpCN, rejected_samples_gpCN, samples_gpCN = stat_model.run_MCMC(method='gpCN',
                                                                                  num_results = num_results,
                                                                                  number_burnin=number_burnin,
                                                                                  step_size=step_size,
                                                                                  initial_chain_state =initial_chain_state,
                                                                                 MAP = MAP,
                                                                                 Hess = Hess)
accept_gpCN= (len(accepted_samples_gpCN)/len(samples_gpCN))*100
print(f'acceptance rate gpCN {num_results:d} {accept_gpCN:.2f}%')