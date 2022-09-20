# from ModelKV import ModelKev
import tensorflow as tf
import tensorflow_probability as tfp
import timeit
import numpy as np
tfd = tfp.distributions
from gpCN import gpCN_MCMC

class MCMC_container():
    def __init__(self,Model):
        self.Model = Model
        self.MAP = self.Model.MAP
        self.Data = self.Model.Data
        self.joint_log_post = self.Model.joint_log
        self.dtype = np.float64
        
    def run_MCMC(self,method,num_results,number_burnin,step_size,num_leapfrog_steps = None):
        methods = {'RMH','HMC','gpCN'}
        self.unnormalized_posterior_log_prob = lambda *args: self.joint_log_post(*args)
        
        if method not in methods:
            raise ValueError('available MCMC methods:', methods)
        self.initial_chain_state = [self.MAP]

        start = timeit.default_timer()
        if method == 'RMH':
            samples, kernel_results = self.RMH(num_results,number_burnin,step_size)
            end = timeit.default_timer()
            time_rmh = end-start
            print('Random walk time in seconds: %.3f' % (time_rmh))
            
            return samples, kernel_results,time_rmh
        
        if method == 'HMC':
            if num_leapfrog_steps is None:
                ValueError('num_leapfrog_steps is required')
            samples, kernel_results = self.HMC(num_results,number_burnin,step_size,num_leapfrog_steps)
            end = timeit.default_timer()
            time_hmc = end-start
            print('HMC time in seconds: %.3f' % (time_hmc))
            
            return samples, kernel_results,time_hmc
            
        
        if method == 'gpCN':
            accepted_samples_gpCN, rejected_samples_gpCN, samples_gpCN = self.gpCN(        
                                                                              num_results,
                                                                              number_burnin,
                                                                              self.Model.Hess,
                                                                              self.Model.Number_para,
                                                                              self.Model.negative_log_posterior,
                                                                              self.MAP,
                                                                              self.Model.cov_prior,
                                                                              beta = step_size)
        
            return accepted_samples_gpCN, rejected_samples_gpCN, samples_gpCN

    def RMH(self,num_results,number_burnin,step_size):
        
        def gauss_new_state_fn(scale, dtype):
            gauss = tfd.Normal(loc=dtype(0), scale=dtype(scale))
            def _fn(state_parts, seed):
                next_state_parts = []
                part_seeds = tfp.random.split_seed(
                seed, n=len(state_parts), salt='rwmcauchy')
                for sp, ps in zip(state_parts, part_seeds):
                    next_state_parts.append(sp + gauss.sample(
                    sample_shape=sp.shape, seed=ps))
                return next_state_parts
            return _fn

        samples, kernel_results = tfp.mcmc.sample_chain(
            num_results=num_results,
            current_state=self.initial_chain_state,
            kernel=tfp.mcmc.RandomWalkMetropolis(
                target_log_prob_fn=self.unnormalized_posterior_log_prob,
                new_state_fn=gauss_new_state_fn(scale=step_size, dtype=np.float64)),
            num_burnin_steps=number_burnin,
            num_steps_between_results=0,  # Thinning.
            parallel_iterations=1,
            seed=42)

        return samples, kernel_results

    # @tf.function
    def HMC(self,num_results,number_burnin,step_size,num_leapfrog_steps):
        samples, kernel_results = tfp.mcmc.sample_chain(
            num_results=num_results,
            current_state=self.initial_chain_state,
            kernel=tfp.mcmc.HamiltonianMonteCarlo(
                target_log_prob_fn=self.unnormalized_posterior_log_prob,
                step_size = step_size,
                num_leapfrog_steps = num_leapfrog_steps),
            num_burnin_steps=number_burnin,
            num_steps_between_results=0,  # Thinning.
            parallel_iterations=1,
            seed=42)
        return samples,kernel_results

    def gpCN(self,num_results,number_burnin,Hess,Number_para,negative_log_posterior, MAP,cov_prior,beta):
        
        try: 
            tf.linalg.cholesky(Hess)
        except:
            eigval,eigvec = tf.linalg.eigh(Hess)
            eigval = tf.where(eigval>0,eigval,1e-5)
            Hess = tf.matmul(tf.matmul(eigvec,tf.linalg.diag(eigval)),tf.transpose(eigvec))
        
        gpCN_sampler = gpCN_MCMC(Hess, Number_para, negative_log_posterior, MAP,cov_prior,num_results,number_burnin,MAP,beta)
        
        accepted_samples_gpCN, rejected_samples_gpCN, samples_gpCN = gpCN_sampler.run_chain_hessian()
        return accepted_samples_gpCN, rejected_samples_gpCN, samples_gpCN