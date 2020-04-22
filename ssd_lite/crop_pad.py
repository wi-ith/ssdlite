# -*- coding: utf-8 -*-
"""
@author: wi-ith
"""
import tensorflow as tf
from tensorflow.python.ops import math_ops

FLAGS = tf.app.flags.FLAGS



def safe_divide(numerator, denominator, name):
    return tf.where(
        math_ops.greater(denominator, 0),
        math_ops.divide(numerator, denominator),
        tf.zeros_like(numerator),
        name=name)


def bboxes_resize(boxes, window, scope=None):

    with tf.name_scope(scope, 'ChangeCoordinateFrame'):
        win_height = window[2] - window[0]
        win_width = window[3] - window[1]
        boxes = boxes - [window[0], window[1], window[0], window[1]]
        y_scale = 1.0 / win_height
        x_scale = 1.0 / win_width
        with tf.name_scope(scope, 'Scale'):
            y_scale = tf.cast(y_scale, tf.float32)
            x_scale = tf.cast(x_scale, tf.float32)
            y_min, x_min, y_max, x_max = tf.split(
                value=boxes, num_or_size_splits=4, axis=1)
            y_min = y_scale * y_min
            y_max = y_scale * y_max
            x_min = x_scale * x_min
            x_max = x_scale * x_max
            scaled_boxlist = tf.concat([y_min, x_min, y_max, x_max], 1)
            return scaled_boxlist

def bboxes_intersection(bbox_ref, bboxes, name=None):

    with tf.name_scope(name, 'bboxes_intersection'):
        bboxes = tf.transpose(bboxes)
        bbox_ref = tf.transpose(bbox_ref)

        int_ymin = tf.maximum(bboxes[0], bbox_ref[0])
        int_xmin = tf.maximum(bboxes[1], bbox_ref[1])
        int_ymax = tf.minimum(bboxes[2], bbox_ref[2])
        int_xmax = tf.minimum(bboxes[3], bbox_ref[3])
        h = tf.maximum(int_ymax - int_ymin, 0.)
        w = tf.maximum(int_xmax - int_xmin, 0.)

        inter_vol = h * w
        bboxes_vol = (bboxes[2] - bboxes[0]) * (bboxes[3] - bboxes[1])
        scores = safe_divide(inter_vol, bboxes_vol, 'intersection')
        return scores

def bboxes_filter_overlap(labels, bboxes, im_box_rank2,
                          threshold=0.5, assign_negative=False,
                          scope=None):

    with tf.name_scope(scope, 'bboxes_filter', [labels, bboxes]):
        scores = bboxes_intersection(im_box_rank2,
                                     bboxes)
        mask = scores >= threshold
        if assign_negative:
            labels = tf.where(mask, labels, -labels)
        else:
            labels = tf.boolean_mask(labels, mask)
            bboxes = tf.boolean_mask(bboxes, mask)
        return labels, bboxes

def prune_completely_outside_window(labels, boxes, window, scope=None):
    with tf.name_scope(scope, 'PruneCompleteleyOutsideWindow'):
        y_min, x_min, y_max, x_max = tf.split(
            value=boxes, num_or_size_splits=4, axis=1)
        win_y_min, win_x_min, win_y_max, win_x_max = tf.unstack(window)
        coordinate_violations = tf.concat([
            tf.greater_equal(y_min, win_y_max), tf.greater_equal(x_min, win_x_max),
            tf.less_equal(y_max, win_y_min), tf.less_equal(x_max, win_x_min)
        ], 1)
        valid_indices = tf.reshape(
            tf.where(tf.logical_not(tf.reduce_any(coordinate_violations, 1))), [-1])
        labelslist = tf.gather(labels, valid_indices)
        subboxlist = tf.gather(boxes, valid_indices)

        return labelslist, subboxlist

def random_crop_image(image,
                      boxes,
                      labels,
                      min_object_covered=1.0,
                      aspect_ratio_range=(0.75, 1.33),
                      area_range=(0.1, 1.0),
                      overlap_thresh=0.3,
                      clip_boxes=True):

    image_shape = tf.shape(image)
    boxes_expanded = tf.expand_dims(
        tf.clip_by_value(
            boxes, clip_value_min=0.0, clip_value_max=1.0), 1)
    im_box_begin, im_box_size, im_box = tf.image.sample_distorted_bounding_box(
        image_shape,
        bounding_boxes=boxes_expanded,
        min_object_covered=min_object_covered,
        aspect_ratio_range=aspect_ratio_range,
        area_range=area_range,
        max_attempts=100,
        use_image_if_no_bounding_boxes=True)
    new_image = tf.slice(image, im_box_begin, im_box_size)
    new_image.set_shape([None, None, image.get_shape()[2]])
    # [1, 4]
    im_box_rank2 = tf.squeeze(im_box, squeeze_dims=[0])
    # [4]
    im_box_rank1 = tf.squeeze(im_box)

    labels, boxes = prune_completely_outside_window(labels, boxes, im_box_rank1)
    labels, boxes = bboxes_filter_overlap(labels, boxes, im_box_rank2,
                                          threshold=overlap_thresh,
                                          assign_negative=False)
    boxes = bboxes_resize(boxes, im_box_rank1)

    if clip_boxes:
        boxes = tf.clip_by_value(
            boxes, clip_value_min=0.0, clip_value_max=1.0)

    return new_image, labels, boxes

