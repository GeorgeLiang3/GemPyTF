import tensorflow as tf
import sys
from gempy.core.tensor.tensorflow_graph_uncon_sig import TFGraph


        
class gaussianNormalizer(object):
  def __init__(self,mean, std):
    self.mean = mean
    self.std = std
  def normalize(self,mu):
    return (mu-self.mean)/self.std
  def denormalize(self,mu):
    return mu*self.std + self.mean

class gravFuncs():
    '''
        wraping class of the Gravity Bayesian GemPy model
    '''
    def __init__(self,size,n_devices,tz,gempy_input) -> None:
        self.size = size
        self.n_devices = n_devices
        self.tz = tz
        self.gempy_input = gempy_input

    # gravity forward simulation
    def forward(self,mu,density,model,sigmoid = True):
        '''
            gravity forward simulation
        '''
        TFG = TFGraph(self.gempy_input, model.fault_drift,
                        model.grid_tensor, model.values_properties, model.nugget_effect_grad,model.nugget_effect_scalar, model.Range,
                        model.C_o, model.rescale_factor,slope = 500, dtype = model.tfdtype, gradient = True,sigmoid = sigmoid)

        values_properties = tf.stack([tfconstant([1.,2.,3.]),density],axis=0)

        #formation_block,property_block,block_mask
        _,property_block,_ =TFG.compute_series(mu,
                    model.dips_position,
                    model.dip_angles,
                    model.azimuth,
                    model.polarity,
                    values_properties)
        
        size_property_block = self.size*self.n_devices
        densities = property_block[0:size_property_block]
        grav_convolution_full = tf.TensorArray(model.tfdtype, size=self.n_devices, dynamic_size=False, clear_after_read=True)
        for i in tf.range(self.n_devices):
            windowed_densities = densities[i*self.size:(i+1)*self.size]
            grav_ = TFG.compute_forward_gravity(tfconstant(self.tz), 0, self.size, windowed_densities)
            grav_convolution_full = grav_convolution_full.write(i, grav_)
        grav_convolution_full = tf.squeeze(grav_convolution_full.stack())
        grav_convolution_full =  tf.reduce_max(grav_convolution_full) - grav_convolution_full
        return grav_convolution_full

    # gempy scale function
    def unscale(self,scaled_value,model):
        return (scaled_value-0.5001)*model.rf+model.centers
    def scale(self,unscaled_value,model):
        return (unscaled_value - model.centers)/model.rf + 0.5001
    