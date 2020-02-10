# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 01:20:53 2020

@author: wi-ith
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

tf.app.flags.DEFINE_string('log_dir', "", 'directory for pb, ckpt and event file.')

tf.app.flags.DEFINE_string('pretrained_dir', "", 'directory for pre-trained model.')

tf.app.flags.DEFINE_string('mode', "train", 'mode for model.')

tf.app.flags.DEFINE_string('tfrecords_dir', "", 'Directory where tfrecords were written to.')

tf.app.flags.DEFINE_integer('image_size', 512,
                            """Provide square images of this size.""")

tf.app.flags.DEFINE_integer('num_examples', 5717,
                            """Provide square images of this size.""")

tf.app.flags.DEFINE_integer('num_examples_validation', 5823,
                            """Provide square images of this size.""")

tf.app.flags.DEFINE_integer('num_classes', 21,
                            """the number of classes.""")

tf.app.flags.DEFINE_integer('max_boxes', 100, 'Max number of objects in an image')

tf.app.flags.DEFINE_integer('num_preprocess_threads', 4,
                            """Number of preprocessing threads per tower. """
                            """Please make this a multiple of 4.""")

tf.app.flags.DEFINE_float('drop_out_rate',0.5,'Keep probability of dropout for the fully connected layer(s).')

tf.app.flags.DEFINE_float('matched_threshold', 0.5, 'IoU threshold for positive.')

tf.app.flags.DEFINE_float('unmatched_threshold', 0.5,'IoU threshold for negative.')

tf.app.flags.DEFINE_float('negative_class_weight', 1.0, 'weight for background class.')

tf.app.flags.DEFINE_float('localization_loss_weight', 1.0, 'weight for localization loss')

tf.app.flags.DEFINE_float('classification_loss_weight', 1.0, 'weight for classification loss.')

tf.app.flags.DEFINE_float('smallest_scale', 0.1, 'weight for classification loss.')

tf.app.flags.DEFINE_float('min_scale', 0.2, 'weight for classification loss.')

tf.app.flags.DEFINE_integer('num_gpus',1,
                            'How many GPUs to use.')

tf.app.flags.DEFINE_boolean('log_device_placement', True, 'Whether to log device placement.')

tf.app.flags.DEFINE_integer('batch_size', 4, 'The number of samples in each batch.')

tf.app.flags.DEFINE_integer('max_steps', 10000000, 'The maximum number of training steps.')

tf.app.flags.DEFINE_integer('decay_steps', None, 'learning rate decay step.')

tf.app.flags.DEFINE_float('MOVING_AVERAGE_DECAY',
                           0.9,
                          'The decay to use for the moving average.')

tf.app.flags.DEFINE_integer('save_steps',
                            500,
                            'The step per saving model.')

tf.app.flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')

tf.app.flags.DEFINE_integer('num_readers', 4,
                            """Number of parallel readers during train.""")

tf.app.flags.DEFINE_integer('random_seed', 12345,
                            "set random seed")

tf.app.flags.DEFINE_integer('input_queue_memory_factor', 4,
                            """Size of the queue of preprocessed images. """
                            """Default is ideal but try smaller values, e.g. """
                            """4, 2 or 1, if host memory is constrained. See """
                            """comments in code for more details.""")

tf.app.flags.DEFINE_integer('preprocessing_gpu', 1,"")

tf.app.flags.DEFINE_integer('epoch_size', 1000,'Number of batches per epoch.')

'''
random_pad_prob

'''