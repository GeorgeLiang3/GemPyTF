import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt

class PV():
    def __init__(self,num_para_total,center_mu,negative_log_posterior_function,Hess = None,res = 51) -> None:
        self.Hess = Hess
        self.num_para_total = num_para_total
        self.res = res
        if self.Hess is not None:
            self.hess_diag = np.diagonal(self.Hess)
        self.negative_log_posterior_function = negative_log_posterior_function
        self.center_mu = center_mu
        
    def compute_(self,r = 1,norm = True,seed = 1,directions = None):

        # grid space
        alpha = np.linspace(-r,r,self.res)
        beta = np.linspace(-r,r,self.res)
        self.xv, self.yv = np.meshgrid(alpha,beta)
        self.z = np.random.normal(size = [self.res,self.res])
        step = 0
        np.random.seed(seed)
        if directions is None:
            if norm == True and self.Hess is not None:
                self.delta = np.random.normal(size=self.num_para_total)/self.hess_diag
                self.eta = np.random.normal(size=self.num_para_total)/self.hess_diag
            else:
                # random directions
                self.delta = np.random.normal(size=self.num_para_total)
                self.eta = np.random.normal(size=self.num_para_total)
        else:
            self.delta = directions[0]
            self.eta = directions[1]
        for i in range(self.res):
            for j in range(self.res):
                if step %10 == 0:
                    print("step "+str(step)+"   ", end = '')
                evaluation_point = (self.center_mu+self.xv[i][j]*self.delta+self.yv[i][j]*self.eta)
                loss = self.negative_log_posterior_function(evaluation_point)
                self.z[i][j] = loss
                step += 1
                
    def plot_contour(self,levels = 50):
        plt.contour(self.xv,self.yv,self.z,vmin = np.min(self.z), vmax = np.max(self.z),levels = levels)
        plt.colorbar()
    
    def plot_3D(self):
        fig = plt.figure()
        ax = Axes3D(fig)
        surf = ax.plot_surface(self.xv,self.yv,self.z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        fig.colorbar(surf, shrink=0.5, aspect=5)