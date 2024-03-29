# import GPUtil
import json 
import numpy as np
import tensorflow as tf


def tfconstant(x):
        return tf.constant(x, dtype=tf.float64)

class dotdict(dict):
    def __getattr__(self, name):
        return self[name]

# def print_GPUinfo():
#   GPUtil.showUtilization(attrList=[[{'attr':'memoryUtil','name':'Memory util.','suffix':'%','transform': lambda x: x*100,'precision':0}],
#                         [{'attr':'memoryTotal','name':'Memory total','suffix':'MB','precision':0},]])

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

class gaussianNormalizer(object):
    def __init__(self,mean, std):
        self.mean = mean
        self.std = std
    def normalize(self,mu):
        return (mu-self.mean)/self.std
    def denormalize(self,mu):
        return mu*self.std + self.mean

# gempy scale function
def unscale(scaled_value,model):
  return (scaled_value-0.5001)*model.rf+model.centers
def scale(unscaled_value,model):
  return (unscaled_value - model.centers)/model.rf + 0.5001

# re-arrange the parameter to gempy input
def concat_xy_and_scale(mu,model,static_xy,sfp_shape,num_para):
  mu = tf.concat([static_xy,tf.reshape(mu,[num_para,1])],axis=-1)
  return scale(tf.reshape(mu,sfp_shape),model )