# GemPyTF
## Overview
This is a TensorFlow extension of [GemPy](https://github.com/cgre-aachen/gempy) to develop 3D subsurface model while keep tracking the derivatives of the parameters.
## Why TensorFlow
GemPy is the most popular Python-based 3-D structural geological modeling open-source software now, which allows the implicit (i.e. automatic) creation of complex geological models from interface and orientation data. We all love GemPy, however, the installation of [Theano](https://en.wikipedia.org/wiki/Theano_(software)) sometime could be frustrating. Therefore this project aims to extend the backend of GemPy with the modern machine learning package [TensorFlow](https://www.tensorflow.org/) for Automatic Differentiation (AD).

Try the simple demos in colab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/GeorgeLiang3/GemPyTF/blob/main/GemPyTF_demo.ipynb)

## Installation and dependency
The current version is depend on an older version of GemPy-'2.1.1', but no prior installation of GemPy.

_The following commands should be executed in a CMD, Bash or Powershell window. To do this, go to a folder on your computer, click in the folder path at the top and type CMD, then press enter._

1. create conda virtual environment 
   
```
conda create -n gempytf_env python=3.7
```

2. activate the virtual environment 
   
```
conda activate gempytf_env
```

3.  Clone the repository: For this step you need Git installed, but you can just download the zip file instead by clicking the button at the top of this page
   
```
https://github.com/GeorgeLiang3/GemPyTF.git
```

4. Navigate to the project directory: (Type this into your CMD window, you're aiming to navigate the CMD window to the repository you just downloaded)

```
cd GemPyTF
```

5. Install the required dependencies: (Again, type this into your CMD window)

```
pip install -r requirements.txt
```

## Limitations
At the moment there are only limited models are tested (in [Examples](/Examples/)). 

current version has 
- no support for topology
- no support for fault block
- not been tested with topography

### Known bugs to be fixed
- <s>3D color map is in wrong order </s> (fixed)
- <s>2D plot function `show_data` function not correct</s> (fixed)
- Hessian in graph mode is limited
### References
- **Original GemPy paper**: de la Varga, M., Schaaf, A. and Wellmann, F., 2019. GemPy 1.0: open-source stochastic geological modeling and inversion. Geoscientific Model Development, 12(1), pp.1-32.
- **Hessian MCMC used GemPyTF**: Liang, Z., Wellmann, F. and Ghattas, O., 2022. Uncertainty quantification of geological model parameters in 3D gravity inversion by Hessian-informed Markov chain Monte Carlo. Geophysics, 88(1), pp.1-78.
