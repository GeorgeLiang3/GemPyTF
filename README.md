# GemPyTF
## Overview
This is a TensorFlow extension of [GemPy](https://github.com/cgre-aachen/gempy) to develop 3D subsurface model while keep tracking the derivatives of the parameters.
## Why TensorFlow
GemPy is the most popular Python-based 3-D structural geological modeling open-source software now, which allows the implicit (i.e. automatic) creation of complex geological models from interface and orientation data. We all love GemPy, however, the installation of [Theano](https://en.wikipedia.org/wiki/Theano_(software)) sometime could be frustrating. Therefore this project aims to extend the backend of GemPy with the modern machine learning package [TensorFlow](https://www.tensorflow.org/) for Automatic Differentiation (AD).

Try the simple demos in colab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/GeorgeLiang3/GemPyTF/blob/main/GemPyTF_demo.ipynb)

## Installation and dependency
The current version is depend on an older version of GemPy-'2.1.1'

TODO: Test and wrap this in a single installation file as e.g. `requirements.txt`
- create conda environment `conda create --name gempytf python==3.7`
- `git clone https://github.com/GeorgeLiang3/GemPyTF.git`
- `pip install --upgrade pip`
- `pip install tensorflow`
- `conda install pandas`
- `conda install scipy`
- `pip install nptyping==1.0.1`
- `conda install seaborn`
- skimage < '0.18.2' and for MacOS < 10.13.6 need older skimage version `pip install -U scikit-image==0.17.2  ` [stackoverflow answer](https://stackoverflow.com/questions/65431999/it-seems-that-scikit-learn-has-not-been-built-correctly)

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