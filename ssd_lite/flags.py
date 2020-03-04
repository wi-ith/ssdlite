# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 01:20:53 2020

@author: wi-ith
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

tf.app.flags.DEFINE_string('ckpt_save_path', "ckpt/save_1/",'')

tf.app.flags.DEFINE_string('pretrained_ckpt_path', "path/to/pretrained/",'')

tf.app.flags.DEFINE_string('mode', "train",'')

tf.app.flags.DEFINE_string('tfrecords_dir', "path/to/tfrecords/folder",'')

tf.app.flags.DEFINE_integer('image_size', 512,'')

tf.app.flags.DEFINE_integer('max_boxes', 120,'')

tf.app.flags.DEFINE_integer('num_train', 1000,'')

tf.app.flags.DEFINE_integer('num_readers', 4,'')

tf.app.flags.DEFINE_integer('num_preprocess_threads', 4,'')

tf.app.flags.DEFINE_integer('input_queue_memory_factor', 4,'')

tf.app.flags.DEFINE_integer('num_classes', 91,"")

tf.app.flags.DEFINE_integer('batch_size', 4,"")

tf.app.flags.DEFINE_float('learning_rate', 0.001, '')

tf.app.flags.DEFINE_float('random_flip_prob', 0.5, '')

tf.app.flags.DEFINE_float('random_pad_prob', 0.4, '')

tf.app.flags.DEFINE_float('positive_threshold', 0.5, '')

tf.app.flags.DEFINE_float('negative_threshold', 0.5, '')