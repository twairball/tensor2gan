# Tensor2Gan

Collection of GAN experiments, using TF Estimator and TF GAN. 

### Install

`pip install -r requirements.txt`

### Usage

````
./gan_trainer \
    --model_dir=./train \           # save model checkpoints
    --data_dir=./data \             # save dataset
    --batch_size=32 \               
    --z_dim=100 \                   # GAN input noise z dims
    --max_steps=30000 \
    --generator=GenerateCIFAR10 \   # Data Generator
    --spectral_norm=1               # spectral norm flag
````

## Data Generators

`DataGenerator` define input pipelines used to feed to GAN networks. 

It provides a `get_input_fn` that returns a callable function, consumed by `Estimators`. 

It also maintains some useful properties, e.g. `num_classes` and `input_shape`

## GAN

GAN classes wrap an Estimator which we can then pass `input_fn` and train. See `tf.contrib.gan.gan_estimator` for more info. 

# LICENSE

MIT
