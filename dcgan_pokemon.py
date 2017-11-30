#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

from data_generators.pokemon import get_input_fn

from models.dcgan import DCGAN

import argparse
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()

parser.add_argument('-m', '--model_dir',
    required=False, default="./train", 
    help='train results directory')

parser.add_argument('-d', '--data_dir',
    required=False, default="./data", 
    help='train data directory')

parser.add_argument('-b', '--batch_size',
    required=False, default=32, 
    help='batch size')

parser.add_argument('-N', '--num_steps',
    required=False, default=10000, 
    help='number of train steps')

def main():
    opt = parser.parse_args()
    input_fn = get_input_fn(batch_size=opt.batch_size, train=True, data_dir=opt.data_dir)

    def train_input_fn():
        noise = tf.random_normal([32, 100])
        images, labels = input_fn()
        return noise, images

    dc_gan = DCGAN(model_dir=opt.model_dir)
    dc_gan.gan_estimator.train(train_input_fn, steps=opt.num_steps)

if __name__ == '__main__':
    main()