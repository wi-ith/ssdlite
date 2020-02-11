# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 01:44:10 2020

@author: wi-ith
"""
import tensorflow as tf
import tensorflow.contrib as tc
from tensorflow.contrib.framework.python.ops import arg_scope

_WEIGHT_DECAY = 4e-5


class MobileNetV2(object):
    def __init__(self, is_training=True, input_size=224):
        self.input_size = input_size
        self.is_training = is_training
        self.normalizer = tc.layers.batch_norm
        self.bn_params = {'is_training': self.is_training,
                          'scale': True,
                          'center': True,
                          'decay': 0.9997,
                          'epsilon': 0.001,
                          }

    def _build_model(self, image, reuse):
        self.i = 0
        with arg_scope([tc.layers.conv2d],
                       weights_regularizer=tc.layers.l2_regularizer(_WEIGHT_DECAY)):
            with tf.variable_scope('MobilenetV2', reuse=reuse):
                # image_copy=tf.identity(image)
                output = tc.layers.conv2d(image, 32, 3, 2,
                                          activation_fn=tf.nn.relu6,
                                          normalizer_fn=self.normalizer, normalizer_params=self.bn_params)  # conv

                print(output.get_shape())
                _, _, output = self._inverted_bottleneck(output, 1, 1, 16, 0)  # expanded_conv
                _, _, output = self._inverted_bottleneck(output, 6, 1, 24, 1)  # expanded_conv_1
                _, _, output = self._inverted_bottleneck(output, 6, 1, 24, 0)  # expanded_conv_2
                _, _, output = self._inverted_bottleneck(output, 6, 1, 32, 1)  # expanded_conv_3
                _, _, output = self._inverted_bottleneck(output, 6, 1, 32, 0)  # expanded_conv_4
                _, _, output = self._inverted_bottleneck(output, 6, 1, 32, 0)  # expanded_conv_5
                _, _, output = self._inverted_bottleneck(output, 6, 1, 64, 1)  # expanded_conv_6
                _, _, output = self._inverted_bottleneck(output, 6, 1, 64, 0)  # expanded_conv_7
                _, _, output = self._inverted_bottleneck(output, 6, 1, 64, 0)  # expanded_conv_8
                _, _, output = self._inverted_bottleneck(output, 6, 1, 64, 0)  # expanded_conv_9
                _, _, output = self._inverted_bottleneck(output, 6, 1, 96, 0)  # expanded_conv_10
                _, _, output = self._inverted_bottleneck(output, 6, 1, 96, 0)  # expanded_conv_11
                _, _, output = self._inverted_bottleneck(output, 6, 1, 96, 0)  # expanded_conv_12

                ###########toBoxPredictor1
                output1, _, output = self._inverted_bottleneck(output, 6, 1, 160,
                                                               1)  # expanded_conv_13 /// output1-->19x19
                output1 = tf.identity(output1)
                _, _, output = self._inverted_bottleneck(output, 6, 1, 160, 0)  # expanded_conv_14
                _, _, output = self._inverted_bottleneck(output, 6, 1, 160, 0)  # expanded_conv_15
                _, _, output = self._inverted_bottleneck(output, 6, 1, 320, 0)  # expanded_conv_16

                ###########toBoxPredictor2
                output2 = tc.layers.conv2d(output, 1280, 1, 1,
                                           activation_fn=tf.nn.relu6,
                                           normalizer_fn=self.normalizer,
                                           normalizer_params=self.bn_params)  # conv_1 /// output2-->10x10

                ###########toBoxPredictor3
                output3 = tc.layers.conv2d(output2, 256, 1, 1,
                                           activation_fn=tf.nn.relu6,
                                           normalizer_fn=self.normalizer, normalizer_params=self.bn_params,
                                           scope='layer_19_1_Conv2d_2_1x1_256')  # layer_19_1_Conv2d_2_1x1_256
                output3 = tc.layers.separable_conv2d(output3, None, 3, 1, stride=2,
                                                     activation_fn=tf.nn.relu6,
                                                     normalizer_fn=self.normalizer, normalizer_params=self.bn_params,
                                                     # layer_19_2_Conv2d_2_3x3_s2_512_depthwise
                                                     scope='layer_19_2_Conv2d_2_3x3_s2_512_depthwise')
                output3 = tc.layers.conv2d(output3, 512, 1, 1,
                                           activation_fn=tf.nn.relu6,
                                           normalizer_fn=self.normalizer, normalizer_params=self.bn_params,
                                           scope='layer_19_2_Conv2d_2_3x3_s2_512')  # layer_19_2_Conv2d_2_3x3_s2_512

                ###########toBoxPredictor4
                output4 = tc.layers.conv2d(output3, 128, 1, 1,
                                           activation_fn=tf.nn.relu6,
                                           normalizer_fn=self.normalizer, normalizer_params=self.bn_params,
                                           scope='layer_19_1_Conv2d_3_1x1_128')  # layer_19_1_Conv2d_3_1x1_128
                output4 = tc.layers.separable_conv2d(output4, None, 3, 1, stride=2,
                                                     activation_fn=tf.nn.relu6,
                                                     normalizer_fn=self.normalizer, normalizer_params=self.bn_params,
                                                     # layer_19_2_Conv2d_3_3x3_s2_256_depthwise
                                                     scope='layer_19_2_Conv2d_3_3x3_s2_256_depthwise')
                output4 = tc.layers.conv2d(output4, 256, 1, 1,
                                           activation_fn=tf.nn.relu6,
                                           normalizer_fn=self.normalizer, normalizer_params=self.bn_params,
                                           scope='layer_19_2_Conv2d_3_3x3_s2_256')  # layer_19_2_Conv2d_3_3x3_s2_256

                ###########toBoxPredictor5
                output5 = tc.layers.conv2d(output4, 128, 1, 1,
                                           activation_fn=tf.nn.relu6,
                                           normalizer_fn=self.normalizer, normalizer_params=self.bn_params,
                                           scope='layer_19_1_Conv2d_4_1x1_128')  # layer_19_1_Conv2d_4_1x1_128
                output5 = tc.layers.separable_conv2d(output5, None, 3, 1, stride=2,
                                                     activation_fn=tf.nn.relu6,
                                                     normalizer_fn=self.normalizer, normalizer_params=self.bn_params,
                                                     # layer_19_2_Conv2d_4_3x3_s2_256_depthwise
                                                     scope='layer_19_2_Conv2d_4_3x3_s2_256_depthwise')
                output5 = tc.layers.conv2d(output5, 256, 1, 1,
                                           activation_fn=tf.nn.relu6,
                                           normalizer_fn=self.normalizer, normalizer_params=self.bn_params,
                                           scope='layer_19_2_Conv2d_4_3x3_s2_256')  # layer_19_2_Conv2d_4_3x3_s2_256

                ###########toBoxPredictor6
                output6 = tc.layers.conv2d(output5, 64, 1, 1,
                                           activation_fn=tf.nn.relu6,
                                           normalizer_fn=self.normalizer, normalizer_params=self.bn_params,
                                           scope='layer_19_1_Conv2d_5_1x1_64')  # layer_19_1_Conv2d_3_1x1_64
                output6 = tc.layers.separable_conv2d(output6, None, 3, 1, stride=2,
                                                     activation_fn=tf.nn.relu6,
                                                     normalizer_fn=self.normalizer, normalizer_params=self.bn_params,
                                                     # layer_19_2_Conv2d_3_3x3_s2_128_depthwise
                                                     scope='layer_19_2_Conv2d_5_3x3_s2_128_depthwise')
                output6 = tc.layers.conv2d(output6, 128, 1, 1,
                                           activation_fn=tf.nn.relu6,
                                           normalizer_fn=self.normalizer, normalizer_params=self.bn_params,
                                           scope='layer_19_2_Conv2d_5_3x3_s2_128')  # layer_19_2_Conv2d_3_3x3_s2_128

        return output1, output2, output3, output4, output5, output6

    def _inverted_bottleneck(self, input, up_sample_rate, atrous_rate, channels, subsample):
        if self.i == 0:
            name = 'expanded_conv'
        else:
            name = 'expanded_conv_{}'.format(self.i)
        with tf.variable_scope(name):

            self.i += 1
            stride = 2 if subsample else 1
            # if channels > 159:
            #   atrous_rate=2
            # else:
            #   atrous_rate=1

            if up_sample_rate > 1:
                expand_ = tc.layers.conv2d(input, up_sample_rate * input.get_shape().as_list()[-1], 1,
                                           activation_fn=tf.nn.relu6,
                                           normalizer_fn=self.normalizer, normalizer_params=self.bn_params,
                                           scope='expand')
            else:
                expand_ = input
            depthwise_ = tc.layers.separable_conv2d(expand_, None, 3, 1,
                                                    stride=stride,
                                                    activation_fn=tf.nn.relu6,
                                                    normalizer_fn=self.normalizer, normalizer_params=self.bn_params,
                                                    rate=atrous_rate,
                                                    scope='depthwise')
            project_ = tc.layers.conv2d(depthwise_, channels, 1, activation_fn=None,
                                        normalizer_fn=self.normalizer, normalizer_params=self.bn_params,
                                        scope='project')
            if input.get_shape().as_list()[-1] == channels:
                project_ = tf.add(input, project_)
            return expand_, depthwise_, project_