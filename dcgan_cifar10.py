#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

from data_generators.mnist import generator as mnist_generator
from data_generators.cifar10 import generator as cifar10_generator

from models.dcgan import DCGAN

import argparse
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()

parser.add_argument('-m', '--model_dir',
    required=False, default="./train", 
    help='train results directory')

parser.add_argument('-N', '--num_steps',
    required=False, default=10000, 
    help='number of train steps')


def main():
    opt = parser.parse_args()
    
    train_generator = cifar10_generator(32, train=True)

    def train_input_fn():
        noise = tf.random_normal([32, 100])
        images, labels = train_generator()
        return noise, images

    dc_gan = DCGAN(model_dir=opt.model_dir)
    dc_gan.gan_estimator.train(train_input_fn, steps=opt.num_steps)

if __name__ == '__main__':
    main()