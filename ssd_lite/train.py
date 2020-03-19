# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 01:29:05 2020

@author: wi-ith
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import re
import time
from datetime import datetime

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from six.moves import xrange  # pylint: disable=redefined-builtin


import ssd
import input
import validation
import flags


FLAGS = tf.app.flags.FLAGS


def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
      tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
       List of pairs of (gradient, variable) where the gradient has been averaged
       across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def train():
    """Train CIFAR-10 for a number of steps."""
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        # Create a variable to count the number of train() calls. This equals the
        # number of batches processed * FLAGS.num_gpus.
        global_step = tf.get_variable(
              'global_step', [],
              initializer=tf.constant_initializer(0), trainable=False)

        lr=FLAGS.learning_rate
        # Create an optimizer that performs gradient descent.
        opt = tf.train.RMSPropOptimizer(lr, decay=0.9, momentum=0.9, epsilon=1)

        # Get images and labels
        # for train
        images, labels, boxes, num_objects = input.distorted_inputs(FLAGS.batch_size)

        batch_queue = tf.contrib.slim.prefetch_queue.prefetch_queue(
              [images, labels, boxes, num_objects], capacity=2 * FLAGS.num_gpus)
        # Calculate the gradients for each model tower.
        tower_grads = []
        with tf.variable_scope(tf.get_variable_scope()):
            for i in xrange(FLAGS.num_gpus):
                with tf.device('/gpu:%d' % i):
                    with tf.name_scope('%s_%d' % ('tower', i)) as scope:
                        # Dequeues one batch for the GPU
                        image_batch, label_batch, box_batch, num_objects_batch = batch_queue.dequeue()
                        # Calculate the loss for one tower of the CIFAR model. This function
                        # constructs the entire CIFAR model but shares the variables across
                        # all towers.
                        cls_loss, loc_loss = ssd.loss(image_batch, label_batch, box_batch, num_objects_batch)
                        regularization_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
                        loss = cls_loss + loc_loss + regularization_loss
                        # Reuse variables for the next tower.
                        tf.get_variable_scope().reuse_variables()

                        # Retain the summaries from the final tower.
                        summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

                        # Calculate the gradients for the batch of data on this CIFAR tower.
                        grads = opt.compute_gradients(loss)

                        # Keep track of the gradients across all towers.
                        tower_grads.append(grads)

        # We must calculate the mean of each gradient. Note that this is the
        # synchronization point across all towers.
        grads = average_gradients(tower_grads)


        #validation
        val_images, val_labels, val_boxes, val_num_objects = input.inputs(1)
        with tf.device('/gpu:0'):
            cls_pred, loc_pred = ssd.inference(val_images)



        for var in tf.trainable_variables():
            print(var.name)
            summaries.append(tf.summary.histogram(var.op.name, var))

        # Add a summary to track the learning rate.
        summaries.append(tf.summary.scalar('learning_rate', lr))

        # Add histograms for gradients.
        for grad, var in grads:
            if grad is not None:
                summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))

        # Apply the gradients to adjust the shared variables.
        train_op = opt.apply_gradients(grads, global_step=global_step)

        # Add histograms for trainable variables.
        for var in tf.trainable_variables():
            summaries.append(tf.summary.histogram(var.op.name, var))

        # Create a saver.
        saver = tf.train.Saver(tf.global_variables())

        # Build the summary operation from the last tower summaries.
        summary_op = tf.summary.merge(summaries)

        pretrained_ckpt_path = FLAGS.pretrained_ckpt_path

        if  not tf.train.latest_checkpoint(FLAGS.ckpt_save_path):
            print('pretrained ckpt')
            exclude_layers = ['global_step',
                             'BoxPredictor_0/ClassPredictor/',
                             'BoxPredictor_1/ClassPredictor/',
                             'BoxPredictor_2/ClassPredictor/',
                             'BoxPredictor_3/ClassPredictor/',
                             'BoxPredictor_4/ClassPredictor/',
                             'BoxPredictor_5/ClassPredictor/',
                             ]
            restore_variables = slim.get_variables_to_restore(exclude=exclude_layers)
            init_fn = slim.assign_from_checkpoint_fn(pretrained_ckpt_path,
                                                     restore_variables, ignore_missing_vars=True)

        else:
            print('training ckpt')
            init_fn = None

        sv = tf.train.Supervisor(logdir=FLAGS.ckpt_save_path,
                                 summary_op=None,
                                 saver=saver,
                                 save_model_secs=0,
                                 init_fn=init_fn)
        config_ = tf.ConfigProto(allow_soft_placement=True)
        config_.gpu_options.per_process_gpu_memory_fraction = 0.4

        # sess=sv.managed_session(config=config_)
        with sv.managed_session(config=config_) as sess:
            # Start the queue runners.
            sv.start_queue_runners(sess=sess)



            for step in xrange(FLAGS.max_steps):
                start_time = time.time()
                sess.run(train_op)
                loss_value = sess.run(loss)
                duration = time.time() - start_time

                assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

                if step % 10 == 0:
                    num_examples_per_step = FLAGS.batch_size * FLAGS.num_gpus
                    examples_per_sec = num_examples_per_step / duration
                    sec_per_batch = duration / FLAGS.num_gpus

                    format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                                'sec/batch)')
                    print (format_str % (datetime.now(), step, loss_value,
                                         examples_per_sec, sec_per_batch))


                if step % 100 == 0:
                    summary_str = sess.run(summary_op)
                    # summary_writer.add_summary(summary_str, step)

                # Save the model checkpoint periodically.
                #if step % int(FLAGS.num_train / FLAGS.batch_size) == 0:
                if True:
                    entire_TF=[]
                    entire_score=[]
                    entire_numGT=[]
                    for val_step in range(FLAGS.num_validation):

                        TF_array, TF_score, num_GT = validation.one_image_validation(val_boxes,
                                                                                     val_labels,
                                                                                     loc_pred,
                                                                                     cls_pred)

                        if len(entire_TF) == 0:
                            entire_TF = TF_array
                            entire_score = TF_score
                            entire_numGT = num_GT
                        else:
                            for k_cls in range(FLAGS.num_classes):
                                entire_TF[k_cls].extend(TF_array[k_cls])
                                entire_score[k_cls].extend(TF_score[k_cls])
                                entire_numGT[k_cls]+=num_GT[k_cls]

                    entire_AP_sum = validation.compute_AP(entire_score,entire_TF,entire_numGT)

                    mAP = np.sum(np.array(entire_AP_sum)) / np.sum(np.array(entire_AP_sum) != 0)

                    print('class AP : ',entire_AP_sum)
                    print('mAP : ',mAP)

                    checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=step)


def main(argv=None):  # pylint: disable=unused-argument
    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)
    train()


if __name__ == '__main__':
    tf.app.run()

