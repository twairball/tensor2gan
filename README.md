# Tensor2Gan

Collection of GAN experiments, using TF Estimator and TF GAN. 

### Install

`pip install -r requirements.txt`

### Usage

````
./gan_trainer \
    --model_dir=./train \           # save model checkpoints
    --data_dir=./data \             # save dataset
    --hparams_set=dcgan_base \      # hparams set
    --hparams='batch_size=32' \     # hparams override
    --train_steps=30000 \
    --local_eval_frequency=2000 \
    --generator=GenerateCIFAR10 \   # Data Generator
    --model=DCGAN                   # [DCGAN | SN_DCGAN]
````

## Data Generators

`DataGenerator` define input pipelines used to feed to GAN networks. 

It provides a `get_input_fn` that returns a callable function, consumed by `Estimators`. 

It also maintains some useful properties, e.g. `num_classes` and `input_shape`

## GAN

- DCGAN
- SN_DCGAN: DCGAN with Spectral Normalization

# LICENSE

MIT
