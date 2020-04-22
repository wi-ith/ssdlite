# -*- coding: utf-8 -*-
"""
@author: wi-ith
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import time
from datetime import datetime

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from six.moves import xrange


import ssd
import input
import validation
import flags


FLAGS = tf.app.flags.FLAGS


def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)

        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def train():
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        global_step = tf.get_variable(
              'global_step', [],
              initializer=tf.constant_initializer(0), trainable=False)

        lr=FLAGS.learning_rate
        opt = tf.train.RMSPropOptimizer(lr, decay=0.9, momentum=0.9, epsilon=1)

        # Get images and labels
        # for train
        with tf.name_scope('train_images'):
            images, labels, boxes, num_objects = input.distorted_inputs(FLAGS.batch_size)

        batch_queue = tf.contrib.slim.prefetch_queue.prefetch_queue(
              [images, labels, boxes, num_objects], capacity=2 * FLAGS.num_gpus)

        tower_grads = []
        tower_losses = []
        with tf.variable_scope(tf.get_variable_scope()):
            for i in xrange(FLAGS.num_gpus):
                with tf.device('/gpu:%d' % i):
                    with tf.name_scope('%s_%d' % ('tower', i)) as scope:

                        image_batch, label_batch, box_batch, num_objects_batch = batch_queue.dequeue()

                        cls_loss, loc_loss = ssd.loss(image_batch, label_batch, box_batch, num_objects_batch)

                        loss = cls_loss + loc_loss
                        regularization_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

                        loss = loss + regularization_loss

                        tf.get_variable_scope().reuse_variables()

                        summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

                        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope)

                        grads = opt.compute_gradients(loss)

                        tower_grads.append(grads)
                        tower_losses.append(loss)

        grads = average_gradients(tower_grads)


        #validation
        val_images, val_labels, val_boxes, val_num_objects = input.inputs(1)
        with tf.device('/gpu:0'):
            with tf.name_scope('eval_images'):
              cls_pred, loc_pred = ssd.inference(val_images)

        summaries.extend(tf.get_collection(tf.GraphKeys.SUMMARIES, 'train_images'))
        summaries.extend(tf.get_collection(tf.GraphKeys.SUMMARIES, 'eval_images'))


        # Add a summary to track the learning rate.
        summaries.append(tf.summary.scalar('learning_rate', lr))

        for grad, var in grads:
            if grad is not None:
                summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))

        with tf.control_dependencies(update_ops):
            train_op = opt.apply_gradients(grads, global_step=global_step)

        for var in tf.trainable_variables():
            print(var.name)
            summaries.append(tf.summary.histogram(var.op.name, var))

        saver = tf.train.Saver(max_to_keep=20)

        summary_op = tf.summary.merge(summaries)

        pretrained_ckpt_path = FLAGS.pretrained_ckpt_path

        if  not tf.train.latest_checkpoint(FLAGS.ckpt_save_path):
            print('pretrained ckpt')
            exclude_layers = ['global_step']
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
                loss_value, cls_loss_value, loc_loss_value = sess.run([loss,cls_loss,loc_loss])
                duration = time.time() - start_time

                assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

                if step % 100 == 0:
                    num_examples_per_step = FLAGS.batch_size * FLAGS.num_gpus
                    examples_per_sec = num_examples_per_step / duration
                    sec_per_batch = duration / FLAGS.num_gpus

                    format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                                'sec/batch)')
                    print (format_str % (datetime.now(), step, loss_value,
                                         examples_per_sec, sec_per_batch))
                    print(cls_loss_value, loc_loss_value)

                if step % 100 == 0:
                    summary_str = sess.run(summary_op)

                if step % (int(FLAGS.num_train / FLAGS.batch_size)*4) == 0 and step!=0:

                    print('start validation')
                    entire_TF=[]
                    entire_score=[]
                    entire_numGT=[]
                    for val_step in range(FLAGS.num_validation):

                        if val_step%500==0:
                            print(val_step,' / ',FLAGS.num_validation)
                        val_GT_boxes, val_GT_cls, val_loc_pred, val_cls_pred, num_objects = sess.run([val_boxes,val_labels,loc_pred,cls_pred,val_num_objects])

                        TF_array, TF_score, num_GT = validation.one_image_validation(val_GT_boxes,
                                                                                     val_GT_cls,
                                                                                     val_loc_pred,
                                                                                     val_cls_pred,
                                                                                     num_objects)

                        if len(entire_TF) == 0:
                            entire_TF = TF_array
                            entire_score = TF_score
                            entire_numGT = num_GT
                        else:
                            for k_cls in range(FLAGS.num_classes-1):
                                entire_TF[k_cls]=np.concatenate([entire_TF[k_cls],TF_array[k_cls]],axis=0)
                                entire_score[k_cls]=np.concatenate([entire_score[k_cls],TF_score[k_cls]],axis=0)
                                entire_numGT[k_cls]+=num_GT[k_cls]

                    entire_AP_sum = validation.compute_AP(entire_score,entire_TF,entire_numGT)

                    mAP = np.sum(np.array(entire_AP_sum)) / np.sum(np.array(entire_AP_sum) != 0)

                    print('class AP : ',entire_AP_sum)
                    print('mAP : ',mAP)

                    checkpoint_path = os.path.join(FLAGS.ckpt_save_path, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=step)



def main(argv=None):
    train()


if __name__ == '__main__':
    tf.app.run()