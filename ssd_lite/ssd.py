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

import tensorflow.contrib.slim as slim


FLAGS = tf.app.flags.FLAGS

TOWER_NAME = 'tower'



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



def inference(images):

    with slim.arg_scope([slim.model_variable, slim.variable], device='/cpu:0'):

        inference_model = model.MobileNetV2(is_training=False, input_size=FLAGS.image_size)
        feat_list=inference_model._build_model(images)
        ratio_list = [(2., 1., 1. / 2.)] + [(3., 2., 1., 1. / 2., 1. / 3., 1.)] * 5
        box_output_list = []
        cls_output_list = []
        for k, (ratio, feat) in enumerate(zip(ratio_list, feat_list)):
            box_output = inference_model.BoxPredictor(feat, len(ratio), k)
            box_output_list.append(box_output)
            cls_output = inference_model.ClassPredictor(feat, len(ratio), k)
            cls_output_list.append(cls_output)

        anchor_concat = aml.make_anchor(cls_output_list,
                                        0.2,
                                        0.95,
                                        ratio_list)

        cls_pred, loc_pred = aml.encode_logits(anchor_concat, cls_output_list, box_output_list)

        return cls_pred, loc_pred

def loss(images, labels, boxes, num_objects):

    with slim.arg_scope([slim.model_variable, slim.variable], device='/cpu:0'):

        train_model = model.MobileNetV2(is_training=True, input_size=FLAGS.image_size)

        #depth 960, 1280, 512, 256, 256, 128
        feat_list=train_model._build_model(images)
        ratio_list = [(2., 1., 1. / 2.)] + [(3., 2., 1., 1. / 2., 1. / 3., 1.)] * 5
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
                                                             num_objects,
                                                             positive_threshold=FLAGS.positive_threshold,
                                                             negative_threshold=FLAGS.negative_threshold,
                                                             num_classes=FLAGS.num_classes,
                                                             max_boxes=FLAGS.max_boxes)

    return cls_loss, loc_loss



