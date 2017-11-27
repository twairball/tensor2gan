# Tensor2Gan

Collection of GAN experiments, using TF Estimator and TF GAN. 

### Install

`pip install -r requirements.txt`

### Usage

`./dcgan_cifar10.py`

## Data Generators

Data generators have a `generator()` that returns an input function for passing to estimator API. 

- `cifar10.py`
- `mnist.py`
- `pokemon.py` 
    + Custom collection of pokemon images normalized to 80x80

## GAN

GAN classes wrap an Estimator which we can then pass `input_fn` and train.  


# LICENSE

MIT
