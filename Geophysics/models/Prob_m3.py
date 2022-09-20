import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
import timeit
import numpy as np
from gpCN import gpCN_MCMC

from Prob import Stat

class Stat3(Stat):
    def set_upper_prior(self,mean,Number_para = 70,norm = None):
        '''
           prior distribution of the upper surface
        '''

        self.tfdtype = self.model.dtype
        
        # define the model parameters
        self.Number_para = Number_para
    
        # define the prior distribution
        self.mu_prior_upper = mean
        self.std_upper = tf.ones(Number_para,self.tfdtype )
        self.norm = norm

    def joint_log_post(self,mu,monitor = False):
        
        # define prior as a multivariate normal distribution      
        self.mvn_prior_upper = tfd.MultivariateNormalDiag(
            loc=self.mu_prior_upper,
            scale_diag = self.std_upper
            )

        # forward calculating gravity
        Gm_ = self.gravity_function(mu,self.model,self.norm)

        mvn_likelihood = tfd.MultivariateNormalTriL(
            loc=Gm_,
            scale_tril=tf.cast(tf.linalg.cholesky(self.data_cov_matrix),self.tfdtype))
    
        self.likelihood_log_prob = tf.reduce_sum(mvn_likelihood.log_prob(self.Obs))

        prior_log_prob_upper = self.mvn_prior_upper.log_prob(mu)
        joint_log = prior_log_prob_upper  + self.likelihood_log_prob
 
        if monitor == True:
            tf.print('prior_upper:',prior_log_prob_upper,'   likelihood:',self.likelihood_log_prob)

        return joint_log