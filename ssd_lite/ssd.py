# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 01:04:34 2020

@author: wi-ith
"""


import tensorflow as tf

import model

import anchor_match_loss as aml

import tensorflow.contrib.slim as slim


FLAGS = tf.app.flags.FLAGS



def inference(images):

    with slim.arg_scope([slim.model_variable, slim.variable], device='/cpu:0'):

        inference_model = model.MobileNetV2(is_training=False, input_size=FLAGS.image_size)
        feat_list=inference_model._build_model(images)
        ratio_list = [(1.0, 2.0, 1.0 / 2.0)] + [(1.0, 2.0, 1.0 / 2.0, 3.0, 1.0 / 3.0, 1.0)] * 5
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

        cls_pred, loc_pred = aml.decode_logits(anchor_concat, cls_output_list, box_output_list)

        return cls_pred, loc_pred

def loss(images, labels, boxes, num_objects):

    with slim.arg_scope([slim.model_variable, slim.variable], device='/cpu:0'):

        train_model = model.MobileNetV2(is_training=True, input_size=FLAGS.image_size)

        feat_list=train_model._build_model(images)
        ratio_list = [(1.0, 2.0, 1.0 / 2.0)] + [(1.0, 2.0, 1.0 / 2.0, 3.0, 1.0 / 3.0, 1.0)] * 5

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



