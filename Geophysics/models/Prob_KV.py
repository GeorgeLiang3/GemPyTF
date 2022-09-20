import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
import timeit
import numpy as np
from gpCN import gpCN_MCMC

class Stat(object):
    def __init__(self, Model,gravity_function,Number_para_total) -> None:
        self.model = Model
        self.gravity_function = gravity_function
        self.monitor = False
        self.Number_para_total = Number_para_total
 

    def set_prior(self,Number_para = 70,norm = None):
        '''
           prior as normal distribution distribution 
        '''
        self.tfdtype = self.model.dtype
        
        # define the model parameters
        self.Number_para = Number_para
    
        # define the prior distribution
        self.mu_prior = tf.zeros(Number_para,self.tfdtype )
        self.std = tf.ones(Number_para,self.tfdtype )
        self.norm = norm
        
    
    def set_likelihood(self,Data,sigma):
        # define the likelihood
        self.Obs = Data
        self.data_cov_matrix = sigma**2*tf.eye(Data.shape[0])
        
    ## it is essential to decorate this function, otherwise face memory issue
    # @tf.function 
    def joint_log_post(self,mu,monitor=False):
        
        # define prior as a multivariate normal distribution      
        self.mvn_prior = tfd.MultivariateNormalDiag(
            loc=self.mu_prior,
            scale_diag = self.std
            )
    
        
        # forward calculating gravity
        Gm_ = self.gravity_function(mu,self.model,self.norm)

        mvn_likelihood = tfd.MultivariateNormalTriL(
            loc=Gm_,
            scale_tril=tf.cast(tf.linalg.cholesky(self.data_cov_matrix),self.tfdtype))
    
        self.likelihood_log_prob = tf.reduce_sum(mvn_likelihood.log_prob(self.Obs))

        prior_log_prob = self.mvn_prior.log_prob(mu)

        joint_log = prior_log_prob  + self.likelihood_log_prob
 
        if monitor == True:
            tf.print('prior:',prior_log_prob,'   likelihood:',self.likelihood_log_prob,  '   posterior:',joint_log)

        return joint_log
    
    # graph posterior
    @tf.function
    def graph_joint_log_post(self,mu):
        return self.joint_log_post(mu)
    @tf.function
    def negative_log_posterior(self,mu):
        return tf.negative(self.joint_log_post(mu))
    
    def loss(self,mu):
        lost =  tf.negative(self.joint_log_post(mu))
        return lost

    def loss_minimize(self):
        lost =  tf.negative(self.joint_log_post(self.mu))
        return lost
    
    def findMAP(self,mu_init,method = 'Nadam',learning_rate = 0.002, iterations = 500):

        # mu_init = mu_true
        if method == 'Nadam':
            optimizer = tf.keras.optimizers.Nadam(
                learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08
            )
        if method == 'Adam':
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08
            )
        cost_A = []
        mu_list = []
        
        self.mu = tf.Variable(mu_init)
        start = timeit.default_timer()
        tolerance  = 3e-5
        
        for step in range(iterations):

            optimizer.minimize(self.loss_minimize, var_list=[self.mu])
            loss = self.loss(self.mu).numpy()
            
            # stop criteria: if cost stops decreasing
            if cost_A:
                if (cost_A[-1]-loss)>0 and (cost_A[-1]-loss)<tolerance: 
                    break # check if cost list is empty
            cost_A.append(loss)

            print ('step:',step,'loss:',loss)

            mu_list.append(self.mu.numpy())
        end = timeit.default_timer()
        print('Adam: %.3f' % (end - start))
        self.MAP = tf.convert_to_tensor(mu_list[-1],self.tfdtype)
        
        return mu_list,cost_A
    
    
    def run_MCMC(self,method,
                 num_results,
                 number_burnin,
                 step_size,
                 num_leapfrog_steps = None,
                 initial_chain_state = None,
                 MAP = None,
                 Hess = None
                 ):
        methods = {'RMH','HMC','gpCN'}
        self.unnormalized_posterior_log_prob = lambda *args: self.graph_joint_log_post(*args)
        
        if method not in methods:
            raise ValueError('available MCMC methods:', methods)
        self.initial_chain_state = initial_chain_state

        start = timeit.default_timer()
        if method == 'RMH':
            states  = self.RMH(num_results,number_burnin,step_size,parallel_iterations = 1)
            end = timeit.default_timer()
            self.time_rmh = end-start
            print('Random walk time in seconds: %.3f' % (self.time_rmh))
            
            return states 
        
        if method == 'HMC':
            if num_leapfrog_steps is None:
                ValueError('num_leapfrog_steps is required')
            states  = self.HMC(num_results,number_burnin,step_size,num_leapfrog_steps)
            end = timeit.default_timer()
            self.time_hmc = end-start
            print('HMC time in seconds: %.3f' % (self.time_hmc))
            
            return states 
            
        
        if method == 'gpCN':
            accepted_samples_gpCN, rejected_samples_gpCN, samples_gpCN = self.gpCN(        
                                                                              num_results,
                                                                              number_burnin,
                                                                              Hess,
                                                                              self.Number_para_total,
                                                                              self.negative_log_posterior,
                                                                              MAP,
                                                                            #   self.Model.cov_prior,
                                                                              beta = step_size)
            
            end = timeit.default_timer()
            self.time_gpcn = end-start
            print('gpCN time in seconds: %.3f' % (self.time_gpcn))
        
            return accepted_samples_gpCN, rejected_samples_gpCN, samples_gpCN
        
    def gpCN(self,num_results,number_burnin,Hess,Number_para,negative_log_posterior, MAP,beta):
        
        try: 
            tf.debugging.assert_all_finite(tf.linalg.cholesky(Hess),message='Checking Hessian')
        except:
            print('Hess is not positive defined')
            eigval,eigvec = tf.linalg.eigh(Hess)
            eigval = tf.where(eigval>0,eigval,1e-5)
            Hess = tf.matmul(tf.matmul(eigvec,tf.linalg.diag(eigval)),tf.transpose(eigvec))
        
        gpCN_sampler = gpCN_MCMC(Hess, Number_para, negative_log_posterior, MAP,num_results,number_burnin,MAP,beta)
        
        accepted_samples_gpCN, rejected_samples_gpCN, samples_gpCN = gpCN_sampler.run_chain_hessian()
        return accepted_samples_gpCN, rejected_samples_gpCN, samples_gpCN

    def RMH(self,num_results,number_burnin,step_size,parallel_iterations = 1):
        
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

        states  = tfp.mcmc.sample_chain(
            num_results=num_results,
            current_state=self.initial_chain_state,
            # trace_fn=None,
            kernel=tfp.mcmc.RandomWalkMetropolis(
                target_log_prob_fn=self.unnormalized_posterior_log_prob,
                new_state_fn=gauss_new_state_fn(scale=step_size, dtype=np.float64)),
            num_burnin_steps=number_burnin,
            num_steps_between_results=0,  # Thinning.
            parallel_iterations=3,
            seed=42)

        return states

    # @tf.function
    def HMC(self,num_results,number_burnin,step_size,num_leapfrog_steps):
        states  = tfp.mcmc.sample_chain(
            num_results=num_results,
            current_state=self.initial_chain_state,
            # trace_fn=None,
            kernel=tfp.mcmc.HamiltonianMonteCarlo(
                target_log_prob_fn=self.unnormalized_posterior_log_prob,
                step_size = step_size,
                num_leapfrog_steps = num_leapfrog_steps),
            num_burnin_steps=number_burnin,
            num_steps_between_results=0,  # Thinning.
            parallel_iterations=1,
            seed=42)
        return states 
    
    
class SVGD():
    def __init__(self,joint_log_post,num_particles = 250,num_iter=1000, dtype=tf.float32):
        self.dtype = dtype
        self.num_particles = num_particles
        self.num_latent = 2
        self.lr = 0.003
        self.alpha = .9
        self.fudge_factor = 1e-6
        self.num_iter = num_iter
        self.range_limit = [-3, 3]
        self.npoints_plot = 50
        self.joint_log_post = joint_log_post


    def get_median(self,v):
        v = tf.reshape(v, [-1])
        m = v.get_shape()[0]//2
        return tf.nn.top_k(v, m).values[m-1]

    def svgd_kernel(self,X0):
        XY = tf.matmul(X0, tf.transpose(X0))
        X2_ = tf.reduce_sum(tf.square(X0), axis=1)

        x2 = tf.reshape(X2_, shape=(tf.shape(X0)[0], 1))

        X2e = tf.tile(x2, [1, tf.shape(X0)[0]])
        
        ## (x1 -x2)^2 + (y1 -y2)^2
        H = tf.subtract(tf.add(X2e, tf.transpose(X2e)), 2 * XY)

        V = tf.reshape(H, [-1, 1])

        # median distance

        h = self.get_median(V)
        h = tf.sqrt(
            0.5 * h / tf.math.log(tf.cast(tf.shape(X0)[0], self.dtype) + 1.0))

        # compute the rbf kernel
        Kxy = tf.exp(-H / h ** 2 / 2.0)

        dxkxy = tf.negative(tf.matmul(Kxy, X0))
        sumkxy = tf.expand_dims(tf.reduce_sum(Kxy, axis=1), 1)
        dxkxy = tf.add(dxkxy, tf.multiply(X0, sumkxy)) / (h ** 2)

        return (Kxy, dxkxy)


    def gradient(self,mu):
        log_p_grad = tf.TensorArray(self.dtype, size=self.num_particles)
        for i in range(mu.shape[0]):
            with tf.GradientTape() as t:
                t.watch(mu)
                f = self.joint_log_post(mu[i])
            log_p_grad =log_p_grad.write(i, t.gradient(f,mu)[i])
        return log_p_grad.stack()


    def svgd_one_iter(self,mu):
        # mu_norm = self.normalizer.encode(mu)
        log_p_grad = self.gradient(mu)
        kernel_matrix, kernel_gradients = self.svgd_kernel(mu)
        grad_theta = (tf.matmul(kernel_matrix, log_p_grad) + kernel_gradients) / self.num_particles
        # print(grad_theta)
        # mu_norm = mu_norm + self.lr * grad_theta
        mu = mu + self.lr * grad_theta
        # mu = self.normalizer.decode(mu_norm)
        # GPU = GPUInfo.gpu_usage()
        
        # print('GPU usage: {} %, GPU Memory: {} Mb'.format(GPU[0][0],GPU[1][0]))
        return mu

    def run_chain_svgd(self, mu):
        mu_list = []
        for i in range(self.num_iter):
            mu = self.svgd_one_iter(mu)
            if i // 10 == 0:
              print('step {}'.format(i))
            mu_list.append(mu.numpy())
        return mu,mu_list


# class MCMC_container():
#     def __init__(self,Model):
#         self.Model = Model
#         self.MAP = self.Model.MAP
#         self.Data = self.Model.Data
#         self.joint_log_post = self.Model.joint_log
#         self.dtype = np.float64
        
#     def run_MCMC(self,method,num_results,number_burnin,step_size,num_leapfrog_steps = None):
#         methods = {'RMH','HMC','gpCN'}
#         self.unnormalized_posterior_log_prob = lambda *args: self.joint_log_post(*args)
        
#         if method not in methods:
#             raise ValueError('available MCMC methods:', methods)
#         self.initial_chain_state = [self.MAP]

#         start = timeit.default_timer()
#         if method == 'RMH':
#             samples, kernel_results = self.RMH(num_results,number_burnin,step_size)
#             end = timeit.default_timer()
#             time_rmh = end-start
#             print('Random walk time in seconds: %.3f' % (time_rmh))
            
#             return samples, kernel_results,time_rmh
        
#         if method == 'HMC':
#             if num_leapfrog_steps is None:
#                 ValueError('num_leapfrog_steps is required')
#             samples, kernel_results = self.HMC(num_results,number_burnin,step_size,num_leapfrog_steps)
#             end = timeit.default_timer()
#             time_hmc = end-start
#             print('HMC time in seconds: %.3f' % (time_hmc))
            
#             return samples, kernel_results,time_hmc
            
        
#         if method == 'gpCN':
#             accepted_samples_gpCN, rejected_samples_gpCN, samples_gpCN = self.gpCN(        
#                                                                               num_results,
#                                                                               number_burnin,
#                                                                               self.Model.Hess,
#                                                                               self.Model.Number_para,
#                                                                               self.Model.negative_log_posterior,
#                                                                               self.MAP,
#                                                                               self.Model.cov_prior,
#                                                                               beta = step_size)
            
#             end = timeit.default_timer()
#             # time_hmc = end-start
#             # print('gpCN time in seconds: %.3f' % (time_hmc))
        
#             return accepted_samples_gpCN, rejected_samples_gpCN, samples_gpCN

#     def RMH(self,num_results,number_burnin,step_size):
        
#         def gauss_new_state_fn(scale, dtype):
#             gauss = tfd.Normal(loc=dtype(0), scale=dtype(scale))
#             def _fn(state_parts, seed):
#                 next_state_parts = []
#                 part_seeds = tfp.random.split_seed(
#                 seed, n=len(state_parts), salt='rwmcauchy')
#                 for sp, ps in zip(state_parts, part_seeds):
#                     next_state_parts.append(sp + gauss.sample(
#                     sample_shape=sp.shape, seed=ps))
#                 return next_state_parts
#             return _fn

#         samples, kernel_results = tfp.mcmc.sample_chain(
#             num_results=num_results,
#             current_state=self.initial_chain_state,
#             kernel=tfp.mcmc.RandomWalkMetropolis(
#                 target_log_prob_fn=self.unnormalized_posterior_log_prob,
#                 new_state_fn=gauss_new_state_fn(scale=step_size, dtype=np.float64)),
#             num_burnin_steps=number_burnin,
#             num_steps_between_results=0,  # Thinning.
#             parallel_iterations=1,
#             seed=42)

#         return samples, kernel_results

#     # @tf.function
#     def HMC(self,num_results,number_burnin,step_size,num_leapfrog_steps):
#         samples, kernel_results = tfp.mcmc.sample_chain(
#             num_results=num_results,
#             current_state=self.initial_chain_state,
#             kernel=tfp.mcmc.HamiltonianMonteCarlo(
#                 target_log_prob_fn=self.unnormalized_posterior_log_prob,
#                 step_size = step_size,
#                 num_leapfrog_steps = num_leapfrog_steps),
#             num_burnin_steps=number_burnin,
#             num_steps_between_results=0,  # Thinning.
#             parallel_iterations=1,
#             seed=42)
#         return samples,kernel_results

#     def gpCN(self,num_results,number_burnin,Hess,Number_para,negative_log_posterior, MAP,cov_prior,beta):
        
#         try: 
#             tf.linalg.cholesky(Hess)
#         except:
#             eigval,eigvec = tf.linalg.eigh(Hess)
#             eigval = tf.where(eigval>0,eigval,1e-5)
#             Hess = tf.matmul(tf.matmul(eigvec,tf.linalg.diag(eigval)),tf.transpose(eigvec))
        
#         gpCN_sampler = gpCN_MCMC(Hess, Number_para, negative_log_posterior, MAP,cov_prior,num_results,number_burnin,MAP,beta)
        
#         accepted_samples_gpCN, rejected_samples_gpCN, samples_gpCN = gpCN_sampler.run_chain_hessian()
#         return accepted_samples_gpCN, rejected_samples_gpCN, samples_gpCN