import tensorflow as tf
import nptyping 
tf.test.gpu_device_name()
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

import sys
import os
print(os.getcwd())
sys.path.append('./Geophysics/GP_old')
sys.path.append('./Geophysics/models')

# suppress warinings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# dependency
import gempy as gp
import tensorflow_probability as tfp
import seaborn as sns
import pandas as pd
import numpy as np

## gempy utils 
from gempy import create_data, map_series_to_surfaces
from gempy.assets.geophysics import GravityPreprocessing
from gempy.plot.visualization_2d_pro import *
from gempy.core.grid_modules.grid_types import CenteredGrid,CenteredRegGrid,RegularGrid

#
from Model_new import *
from Prob_density import *

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
def forward(mu,density,model):
    TFG = TFGraph(input, model.fault_drift,
                    model.grid_tensor, model.values_properties, model.nugget_effect_grad,model.nugget_effect_scalar, model.Range,
                    model.C_o, model.rescale_factor,slope = 100, dtype = model.tfdtype, gradient = True)

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
def forward_function(mu,model,norm):
  '''This function rescale and manimulate the input data to the gravity function
        mu is scaled and need to be concatenate with other parameters
  '''
  # mu = tf.reshape(mu,[-1])
  # mu = norm.denormalize(mu)
  # mu = concat_xy_and_scale(mu,model)
  # gravity = forward(mu,model)

  mu_norm = norm.denormalize(mu)
  mu_upper = mu_norm[num_per_layer:-3]
  mu_lower = mu_upper - mu_norm[:num_per_layer]
  density = mu_norm[-3:]
  mu0 = tf.concat([mu_lower,mu_upper],axis = 0)
  mu0 = concat_xy_and_scale(mu0,model)
  # mu = tf.reshape(mu,[num_para,1])
  # mu = norm.denormalize(mu)
  gravity = forward(mu0,density,model)

  return gravity



## setup variable
modelName = "model4flatprior"

path = './Geophysics/GP_old/notebooks'
orientation_path = "/data/input_data/George_models/model4_2_orientations.csv"
surface_path = '/data/input_data/George_models/'+'model3'+'_surface_points.csv'
regular_grid_resolution = [1,1,1]
center_grid_resolution = [10,10,30]

tfconstant = constant64
tfdtype = tf.float64

num_para = 16
num_per_layer = 8
sfp_shape = [num_para,3]

## Model set-up

# receiver positions
num_ = 6
X = np.linspace(200,800,num_)
Y = np.linspace(200,800,num_)

r = []
for x in X:
  for y in Y:
    r.append(np.array([x,y]))
receivers = np.array(r)
n_devices = receivers.shape[0]

########################
## Ground truth model
model_true = Model(path,
              surface_path,
              orientation_path,
              receivers = receivers,
              dtype = 'float64',
              center_grid_resolution=center_grid_resolution,
              regular_grid_resolution=regular_grid_resolution,
              model_radius=[2000,2000,2000])

model_true.activate_centered_grid()
g = GravityPreprocessing(model_true.grid.centered_grid)
tz = g.set_tz_kernel()

input = model_true.get_graph_input()
model_true.create_tensorflow_graph(input,gradient = True)
size = tf.constant((model_true.center_grid_resolution[0]+1)*(model_true.center_grid_resolution[1]+1)*(model_true.center_grid_resolution[2]),tf.int32)

density_true = tfconstant([2.6, 3.5, 2.])
model_true.grav = forward(model_true.surface_points_coord,density_true,model_true)


x = model_true.surface_points.df[['X','Y','Z']].to_numpy()
static_xy = x[:,0:2]


##### PRIOR MODEL ######

model_0 = Model(path,
              surface_path,
              orientation_path,
              receivers = receivers,
              dtype = 'float64',
              center_grid_resolution=center_grid_resolution,
              regular_grid_resolution=regular_grid_resolution,
              model_radius=[2000,2000,2000])
model_0.activate_centered_grid()

## define prior distribution
np.random.seed(1)
upper_prior = 780*tf.ones(8,tfdtype)
# upper_prior = tfconstant(np.array([706.28215182, 706.28215182, 850.02221393, 850.02221393,
#        850.02221393, 850.02221393, 706.28215182, 706.28215182])+np.random.normal(0,30,8))
std_upper = tfconstant(60*np.ones([num_per_layer]))
thickness_prior = 200* np.ones(num_per_layer)+np.random.normal(0,30,8)
std_lower = 30

minimum_thickness = 20
maximum_thickness = 400

### Density
density_prior = tfconstant([2.4, 3.7, 1.8])
density_std = tfconstant(0.3*np.ones([3]))

prior_mean = tf.concat([thickness_prior,upper_prior,density_prior],axis = 0)
prior_std = tf.concat([std_lower*tf.ones(num_per_layer,tfdtype),std_upper,density_std],axis = 0)
num_para_total = num_para + 3
norm = gaussianNormalizer(prior_mean,prior_std)

# normalize prior mean and std
mean_prior_norm = norm.normalize(prior_mean)
upper_prior_norm = mean_prior_norm[num_per_layer:-3]
lower_prior_norm = mean_prior_norm[:num_per_layer]
minimum_thickness_norm =  norm.normalize(minimum_thickness)[0]
maximum_thickness_norm = norm.normalize(maximum_thickness)[0]
density_norm = mean_prior_norm[-3:]

# define the statistic module
stat_model = Stat(model_0,forward_function,num_para_total)
stat_model.set_density_prior(density_norm,Number_para = 3,norm = norm)
stat_model.set_upper_prior(upper_prior_norm,Number_para = num_per_layer,norm = norm)
stat_model.set_lower_prior(lower_prior_norm,minimum_thickness_norm,maximum_thickness_norm,Number_para = num_per_layer,norm = norm)

Data_with_noise = model_true.grav 
Data_std = 0.2
stat_model.set_likelihood(Data_with_noise,Data_std)
stat_model.monitor=False

mu_true = tf.convert_to_tensor(np.array([200.        , 200.        , 200.        , 200.        ,
                                          200.        , 200.        , 200.        , 200.        ,
                                          706.28215182, 706.28215182, 850.02221393, 850.02221393,
                                          850.02221393, 850.02221393, 706.28215182, 706.28215182, 2.6, 3.5, 2.]),tfdtype)


##############MAP############
@tf.function
def loss_(mu):
        lost =  tf.negative(stat_model.joint_log_post(mu,monitor = False))
        return lost

@tf.function
def loss_minimize():
    lost =  tf.negative(stat_model.joint_log_post(mu_init))
    return lost
          

def Nadam(mu_init,learning_rate,iterations,mu_list,cost_A):


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
    mu_list,cost_A,MAP = Nadam(mu_init,learning_rate,iterations,mu_list,cost_A)
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

path = './Results/'+modelName+'MAP_density_02.json'
mu0 = norm.normalize(prior_mean)
mu_init = tf.Variable(mu0,tfdtype)
learning_rate = 0.2
iterations = 5000

start = timeit.default_timer()
mu_list,cost_A,MAP = find_MAP(mu_init,learning_rate,iterations,path = path)
end = timeit.default_timer()
print('Adam: %.3f' % (end - start))
print('MAP:',MAP.numpy())
print('stat_model.Obs:',stat_model.Obs)
print('stat_model.joint_log_post(MAP):',stat_model.joint_log_post(MAP,monitor = True))

# ########Hessian#########

# @tf.function
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

Hess = calculate_Hessian(MAP)
eigval,eigvec = tf.linalg.eigh(Hess)


# ##########MCMC###########
initial_chain_state = [MAP]
num_results = 10000
number_burnin = 0

# RMH
step_size = 0.002
states_RMH = stat_model.run_MCMC('RMH',
                                 num_results = num_results,
                                 number_burnin = number_burnin,
                                 step_size = step_size,
                                 initial_chain_state =initial_chain_state
                                  )
accept_RMH = (tf.reduce_mean(tf.cast(states_RMH.trace.is_accepted,tfdtype)))*100
print(f'acceptance rate Random-walk Matroplis 1000 {accept_RMH:.2f}%')


# HMC
step_size = 0.0015
states_HMC_10 = stat_model.run_MCMC(method='HMC',
                                num_results = num_results,
                                number_burnin=number_burnin,
                                step_size=step_size,
                                num_leapfrog_steps=10,
                                 initial_chain_state =initial_chain_state)
accept_HMC_10 = (tf.reduce_mean(tf.cast(states_HMC_10.trace.is_accepted,tfdtype)))*100
print(f'acceptance rate Hamiltonian Monte Carlo 1000 {accept_HMC_10:.2f}%')

# HMC
step_size = 0.0015
states_HMC = stat_model.run_MCMC(method='HMC',
                                num_results = num_results,
                                number_burnin=number_burnin,
                                step_size=step_size,
                                num_leapfrog_steps=50,
                                 initial_chain_state =initial_chain_state)
accept_HMC = (tf.reduce_mean(tf.cast(states_HMC.trace.is_accepted,tfdtype)))*100
print(f'acceptance rate Hamiltonian Monte Carlo 1000 {accept_HMC:.2f}%')


# gpCN
step_size = 0.1
accepted_samples_gpCN, rejected_samples_gpCN, samples_gpCN = stat_model.run_MCMC(method='gpCN',
                                                                                  num_results = num_results,
                                                                                  number_burnin=number_burnin,
                                                                                  step_size=step_size,
                                                                                  initial_chain_state =initial_chain_state,
                                                                                 MAP = MAP,
                                                                                 Hess = Hess)
accept_gpCN= (len(accepted_samples_gpCN)/len(samples_gpCN))*100
print(f'acceptance rate gpCN 10000 {accept_gpCN:.2f}%')


# convert result to numpy
samples_RMH = states_RMH.all_states[0].numpy()
samples_HMC_10 = states_HMC_10.all_states[0].numpy()
samples_HMC = states_HMC.all_states[0].numpy()
samples_gpCN = np.array(samples_gpCN)
accept_RMH = accept_RMH.numpy()
accept_HMC_10 = accept_HMC_10.numpy()
accept_HMC = accept_HMC.numpy()


json_dump = json.dumps({'samples_RMH': samples_RMH,
                        'samples_HMC_10': samples_HMC_10,
                        'samples_HMC': samples_HMC,
                        'samples_gpCN': samples_gpCN,
                        'accepted_rate_RMH':accept_RMH,
                        'accepted_rate_HMC_10':accept_HMC_10,
                        'accepted_rate_HMC':accept_HMC,
                        'accepted_rate_gpCN':accept_gpCN,
                        'MAP':MAP.numpy(),
                        }, cls=NumpyEncoder)
with open('./Results/MCMC_'+modelName+'.json', 'w') as outfile:
    json.dump(json_dump, outfile)



print('MAP:',MAP)
print('Hess eigen:',eigval)
print('Done')

