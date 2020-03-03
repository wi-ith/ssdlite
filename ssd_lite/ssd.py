# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 01:04:34 2020

@author: wi-ith
"""

import re

import tensorflow as tf

import input

import model

import anchor_match_loss as aml


FLAGS = tf.app.flags.FLAGS

TOWER_NAME = 'tower'





def _activation_summary(x):

  """Helper to create summaries for activations.



  Creates a summary that provides a histogram of activations.

  Creates a summary that measures the sparsity of activations.



  Args:

    x: Tensor

  Returns:

    nothing

  """

  # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training

  # session. This helps the clarity of presentation on tensorboard.

  tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)

  tf.summary.histogram(tensor_name + '/activations', x)

  tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))





def _variable_on_cpu(name, shape, initializer):

  """Helper to create a Variable stored on CPU memory.



  Args:

    name: name of the variable

    shape: list of ints

    initializer: initializer for Variable



  Returns:

    Variable Tensor

  """

  with tf.device('/cpu:0'):

    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32

    var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)

  return var





def _variable_with_weight_decay(name, shape, stddev, wd):

  """Helper to create an initialized Variable with weight decay.



  Note that the Variable is initialized with a truncated normal distribution.

  A weight decay is added only if one is specified.



  Args:

    name: name of the variable

    shape: list of ints

    stddev: standard deviation of a truncated Gaussian

    wd: add L2Loss weight decay multiplied by this float. If None, weight

        decay is not added for this Variable.



  Returns:

    Variable Tensor

  """

  dtype = tf.float16 if FLAGS.use_fp16 else tf.float32

  var = _variable_on_cpu(

      name,

      shape,

      tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))

  if wd is not None:

    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')

    tf.add_to_collection('losses', weight_decay)

  return var








def loss(images, labels, boxes):

    # Calculate the average cross entropy loss across the batch.
    train_model = model.MobileNetV2(is_training=True, input_size=FLAGS.image_size)
    
    #depth 960, 1280, 512, 256, 256, 128
    feat_list=train_model._build_model(images)
    ratio_list=[(2.,1.,1./2.)]+[(3.,2.,1.,1./2.,1./3.)]*5
    #boxpredictor
    box_output_list=[]
    cls_output_list=[]
    for k, (ratio, feat) in enumerate(zip(ratio_list, feat_list)):
        box_output = train_model.BoxPredictor(feat, len(ratio), k)
        box_output_list.append(box_output)
        cls_output = train_model.ClassPredictor(feat, len(ratio), k)
        cls_output_list.append(cls_output)

    anchor_concat=aml.make_anchor(cls_output_list,
                                  0.2,
                                  0.95,
                                  ratio_list)

    cls_loss,loc_loss = aml.anchor_matching_cls_loc_loss(anchor_concat,
                                                         cls_output_list,
                                                         box_output_list,
                                                         labels,
                                                         boxes,
                                                         positive_threshold=FLAGS.positive_threshold,
                                                         negative_threshold=FLAGS.negative_threshold,
                                                         num_classes=FLAGS.num_classes,
                                                         max_boxes=FLAGS.max_boxes)

    return cls_loss, loc_loss





def _add_loss_summaries(total_loss):

  """Add summaries for losses in CIFAR-10 model.



  Generates moving average for all losses and associated summaries for

  visualizing the performance of the network.



  Args:

    total_loss: Total loss from loss().

  Returns:

    loss_averages_op: op for generating moving averages of losses.

  """

  # Compute the moving average of all individual losses and the total loss.

  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')

  losses = tf.get_collection('losses')

  loss_averages_op = loss_averages.apply(losses + [total_loss])



  # Attach a scalar summary to all individual losses and the total loss; do the

  # same for the averaged version of the losses.

  for l in losses + [total_loss]:

    # Name each loss as '(raw)' and name the moving average version of the loss

    # as the original loss name.

    tf.summary.scalar(l.op.name + ' (raw)', l)

    tf.summary.scalar(l.op.name, loss_averages.average(l))



  return loss_averages_op
