import tensorflow as tf
import numpy as np
import timeit

class GradHess():
    '''
        The general class to wrap all functions
    '''
    def __init__(self,geomodel,statmodel,num_para_total) -> None:
        self.geomodel = geomodel
        self.statmodel = statmodel
        self.num_para_total = num_para_total
        self.tfdtype = tf.float64
    @tf.function
    def loss_(self,mu):
            lost =  tf.negative(self.statmodel.joint_log_post(mu,monitor = False))
            return lost

    @tf.function
    def loss_minimize(self):
        lost =  tf.negative(self.statmodel.joint_log_post(self.mu_init_,monitor = False))
        return lost
    
    def Find_MAP(self,mu_init,learning_rate,iterations,stop_criteria = None):

        cost_A = []
        mu_list = []    
        self.mu_init_=tf.Variable(mu_init,self.tfdtype)

        optimizer = tf.keras.optimizers.Adam(
                    learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08
                )

        start = timeit.default_timer()
        for step in range(iterations):

            optimizer.minimize(self.loss_minimize, var_list=[self.mu_init_])
            loss = self.loss_(self.mu_init_).numpy()

            if cost_A: # check if cost list is empty
                if stop_criteria:
                    if  step > 1000 and (cost_A[-1]-loss)<stop_criteria: 
                        break 
            
            cost_A.append(loss)
            if step%100 == 0:
                print ('step:',step,'loss:',loss)

            mu_list.append(self.mu_init_.numpy())
        end = timeit.default_timer()
        Adam_time = end - start

        MAP = tf.convert_to_tensor(mu_list[-1],self.tfdtype)
        return mu_list,cost_A,MAP,Adam_time
    
    def hvp(self,mu,tangents):
        with tf.autodiff.ForwardAccumulator(mu, tangents) as acc:
            with tf.GradientTape(watch_accessed_variables=False) as t:
                t.watch(mu)
                joint_log = tf.negative(self.statmodel.joint_log_post(mu))
            loss = t.gradient(joint_log,mu)
        hess = acc.jvp(loss)
        return(hess)

    def calculate_Hessian(self,MAP,time = False):
        Hess_list = []
        start = timeit.default_timer()
        for i in range(self.num_para_total):
            tangents = np.zeros(MAP.shape)
            tangents[i]=1
            tangents = tf.convert_to_tensor(tangents,dtype=self.tfdtype)

            Hess_list.append(self.hvp(MAP,tangents).numpy())
        Hess = np.array(Hess_list)
        end = timeit.default_timer()
        if time == True:
            print('time for Hessian calculation: %.3f' % (end - start))
            return Hess, end-start
        return Hess