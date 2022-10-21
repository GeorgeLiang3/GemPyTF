import tensorflow as tf

class ILT(): 
  '''invertible logarithmic transform (Team et al., 2016)'''
  def __init__(self,lower_bound,upper_bound):
    self.ub = upper_bound
    self.lb = lower_bound

  def transform(self,m):
    '''Forward Transformation, convert to bounded space'''
    return tf.math.log(m-self.lb) - tf.math.log(self.ub - m)

  def reverse_transform(self,theta):
    '''Revere Transformation, convert back to unbounded space'''
    return self.lb+ (self.ub - self.lb)/(1+tf.exp(-theta))
