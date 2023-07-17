import tensorflow_probability as tfp
import tensorflow as tf
import numpy as np
tfd = tfp.distributions


    
class gpCN_MCMC():
    def __init__(self, Hessian, Number_para, negative_log_posterior, MAP, number_sample, number_burnin, mu_init, beta=0.1,**kwargs):
        """[summary]

        Args:
            Hessian ([type]): [description]
            Number_para ([type]): [description]
            negative_log_posterior ([type]): [description]
            MAP ([type]): [description]
            C_prior ([type]): [description]
            number_sample ([type]): [description]
            number_burnin ([type]): [description]
            mu_init ([type]): [description]
            beta ([type], optional): [description]. Defaults to constant64(0.1).
        """
        self.Number_para = Number_para
        self.negative_log_posterior = negative_log_posterior

        self.MAP = MAP
        # self.C_prior = C_prior
        self.number_sample = number_sample
        self.number_burnin = number_burnin
        self.mu_init = mu_init
        self.Hessian = Hessian
        self.cov_post = None
        self.dtype = kwargs.get('dtype', tf.float64)
        self.beta = self.constant64(beta)
        
        
    def constant64(self,i):
        return (tf.constant(i, dtype=self.dtype))
    def Laplace_appro(self):
        # self.cov_post = tf.linalg.inv(
        #     (tf.add(self.Hessian, tf.linalg.inv(self.C_prior))))
        self.cov_post = tf.linalg.inv(self.Hessian)

    @tf.function
    def matrixcompute(self, matrix1, matrix2, Cov):
        matrix1 = tf.cast(matrix1, self.dtype)
        matrix2 = tf.cast(matrix2, self.dtype)
        matrix = tf.subtract(matrix1, matrix2)
        matrix = tf.reshape(matrix, [matrix.shape[0], 1])
        matrix_T = tf.transpose(matrix)
        Cov_inv = tf.linalg.inv(Cov)
        result = tf.multiply(self.constant64(
            1/2), tf.matmul(tf.matmul(matrix_T, Cov_inv), matrix))
        return result

    @tf.function
    def acceptance_gpCN(self, m_current, m_proposed):
        delta_current = self.delta_current
        # delta_current = tf.subtract(self.negative_log_posterior(
        #     m_current), self.matrixcompute(m_current, self.MAP, self.cov_post))
        delta_proposed = tf.subtract(self.negative_log_posterior(
            m_proposed), self.matrixcompute(m_proposed, self.MAP, self.cov_post))

        # calculate accept ratio if exp()<1
        accept_ratio = tf.exp(tf.subtract(delta_current, delta_proposed))
        acceptsample = tfd.Sample(
            tfd.Uniform(self.constant64(0), self.constant64(1)),
            sample_shape=[1, 1])
        sample = acceptsample.sample()

        if(accept_ratio > sample): # then accept
            # update self.current
            self.delta_current = delta_proposed
            return True     
        else:                       # reject
            return False

    @tf.function
    def draw_proposal(self, m_current):

        # _term1 = self.MAP

        # sqrt term
        tem_1 = tf.convert_to_tensor(tf.sqrt(1-self.beta**2), dtype=self.dtype)
        # sqrt(1-beta^2)()
        _term2 = tf.multiply(tem_1, (tf.subtract(m_current, self.MAP)))

        Xi = tfd.MultivariateNormalTriL(
            loc=0,
            scale_tril=tf.linalg.cholesky(self.cov_post))

        Xi_s = tfd.Sample(Xi)
        _term3 = tf.multiply(self.beta, Xi_s.sample())

        m_proposed = tf.add(self.MAP, tf.add(_term2, _term3))

        return m_proposed

    def run_chain_hessian(self):

        if self.cov_post is None:
            self.Laplace_appro()

        burn_in = self.number_burnin
        steps = self.number_sample - 1
        k = 0
        accepted = []
        rejected = []
        samples = []
        samples.append(self.mu_init.numpy())
        m_current = self.mu_init  # init m
        
        self.delta_current = tf.subtract(self.negative_log_posterior(
            m_current), self.matrixcompute(m_current, self.MAP, self.cov_post))

        for k in range(steps+burn_in):

            m_proposed = self.draw_proposal(m_current)

            if self.acceptance_gpCN(m_current, m_proposed):
                m_current = m_proposed
                if k >= burn_in:
                    accepted.append(m_proposed.numpy())
                    samples.append(m_proposed.numpy())
            else:
                m_current = m_current
                if k >= burn_in:
                  rejected.append(m_proposed.numpy())
                  samples.append(m_current.numpy())
        self.acceptance_rate = np.shape(accepted)[0]/self.number_sample

        return accepted, rejected, samples


# Timer unit: 1e-06 s

# Total time: 5.07792 s
# File: /content/drive/My Drive/RWTH/gpCN.py
# Function: run_chain_hessian at line 95

# Line #      Hits         Time  Per Hit   % Time  Line Contents
# ==============================================================
#     95                                               def run_chain_hessian(self):
#     96                                           
#     97         1          3.0      3.0      0.0          if self.cov_post is None:
#     98                                                       self.Laplace_appro()
#     99                                           
#    100         1          2.0      2.0      0.0          burn_in = self.number_burnin
#    101         1          1.0      1.0      0.0          steps = self.number_sample
#    102         1          1.0      1.0      0.0          k = 0
#    103         1          1.0      1.0      0.0          accepted = []
#    104         1          0.0      0.0      0.0          rejected = []
#    105         1          1.0      1.0      0.0          samples = []
#    106         1         64.0     64.0      0.0          samples.append(self.mu_init.numpy())
#    107         1          1.0      1.0      0.0          m_current = self.mu_init  # init m
#    108                                           
#    109       101        103.0      1.0      0.0          for k in range(steps+burn_in):
#    110                                           
#    111       100      95864.0    958.6      1.9              m_proposed = self.draw_proposal(m_current)
#    112                                           
#    113       100    4963935.0  49639.3     97.8              if self.acceptance_gpCN(m_current, m_proposed):
#    114        13         68.0      5.2      0.0                  m_current = m_proposed
#    115        13         10.0      0.8      0.0                  if k > burn_in:
#    116        13       1738.0    133.7      0.0                      accepted.append(m_proposed.numpy())
#    117        13         99.0      7.6      0.0                      samples.append(m_proposed.numpy())
#    118                                                       else:
#    119        87        164.0      1.9      0.0                  m_current = m_current
#    120        87      14882.0    171.1      0.3                  rejected.append(m_proposed.numpy())
#    121        87        918.0     10.6      0.0                  samples.append(m_current.numpy())
#    122         1         68.0     68.0      0.0          self.acceptance_rate = np.shape(accepted)[0]/self.number_sample
#    123                                           
#    124         1          1.0      1.0      0.0          return accepted, rejected, samples