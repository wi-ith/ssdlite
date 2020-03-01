# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 01:20:53 2020

@author: wi-ith
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

tf.app.flags.DEFINE_string('ckpt_save_path', "")

tf.app.flags.DEFINE_string('pretrained_ckpt_path', "")

tf.app.flags.DEFINE_string('mode', "train")

tf.app.flags.DEFINE_string('tfrecords_dir', "")

tf.app.flags.DEFINE_integer('image_size', 512)

tf.app.flags.DEFINE_integer('max_boxes', 100)

tf.app.flags.DEFINE_integer('num_preprocess_threads', 4)