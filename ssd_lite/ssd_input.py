# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 01:05:20 2020

@author: RayKwak
"""

# Copyright 2015 The TensorFlow Authors. All Rights Reserved.

#

# Licensed under the Apache License, Version 2.0 (the "License");

# you may not use this file except in compliance with the License.

# You may obtain a copy of the License at

#

#     http://www.apache.org/licenses/LICENSE-2.0

#

# Unless required by applicable law or agreed to in writing, software

# distributed under the License is distributed on an "AS IS" BASIS,

# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# See the License for the specific language governing permissions and

# limitations under the License.

# ==============================================================================



"""Routine for decoding the CIFAR-10 binary file format."""



from __future__ import absolute_import

from __future__ import division

from __future__ import print_function



import tensorflow as tf

import tensorflow_datasets as tfds

FLAGS = tf.app.flags.FLAGS



# Process images of this size. Note that this differs from the original CIFAR

# image size of 32 x 32. If one alters this number, then the entire model

# architecture will change and any model would need to be retrained.

IMAGE_SIZE = 24



# Global constants describing the CIFAR-10 data set.

NUM_CLASSES = 10

NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000

NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000


def parse_tfrecords(example_serialized):
    """
    returns:
        image_buffer : decoded image file
        class_id : 1D class tensor
        bbox : 2D bbox tensor

    """
    # Dense features in Example proto.
    
    context, sequence = tf.parse_single_sequence_example(
        example_serialized,
        context_features={
            'bbox/xmin':
                tf.VarLenFeature(tf.float32),
            'bbox/xmax':
                tf.VarLenFeature(tf.float32),
            'bbox/ymin':
                tf.VarLenFeature(tf.float32),
            'bbox/ymax':
                tf.VarLenFeature(tf.float32),
                                
            'class/text':
                tf.VarLenFeature(tf.string),
            'class/label':
                tf.VarLenFeature(tf.int64),
                                
            'image/height':
                tf.FixedLenFeature([], dtype=tf.int64),
            'image/width':
                tf.FixedLenFeature([], dtype=tf.int64),
            'image/filename':
                tf.FixedLenFeature((), tf.string, default_value=''),
            'image/encoded':
                tf.FixedLenFeature((), dtype=tf.string),
        },
    )

    xmin = tf.expand_dims(context['bbox/xmin'].values, 0)
    ymin = tf.expand_dims(context['bbox/ymin'].values, 0)
    xmax = tf.expand_dims(context['bbox/xmax'].values, 0)
    ymax = tf.expand_dims(context['bbox/ymax'].values, 0)
    bbox = tf.concat(axis=0, values=[ymin, xmin, ymax, xmax])
    bbox = tf.transpose(bbox, [1, 0])
    
    class_name = context['class/text'].values
    class_id = tf.cast(context['class/label'].values, dtype=tf.int32)
    
    height = context['image/height']
    width = context['image/width']    
    filename = context['image/filename']
    image_encoded = context['image/encoded']
    image_encoded = decode_jpeg(image_buffer, 3)
    
    return bbox, class_id, image_encoded


def _get_images_labels(batch_size, split, distords=False):

    """Returns Dataset for given split."""
    with tf.name_scope('process_batch'):
        dataset_dir = FLAGS.tfrecords_dir
        tfrecords_list = tf.gfile.Glob(os.path.join(dataset_dir, '*'+split+'*'))
        
    if tfrecords_list is None:
        raise ValueError('There are not files')
    
    if split=='train':
            filename_queue = tf.train.string_input_producer(tfrecords_list,
                                                            shuffle=True,
                                                            capacity=16)
    elif split=='validation':
            filename_queue = tf.train.string_input_producer(tfrecords_list,
                                                            shuffle=False,
                                                            capacity=1)
    else:
        raise ValueError('Not appropriate split name')
        
    if num_preprocess_threads is None:
        num_preprocess_threads = FLAGS.num_preprocess_threads

    if num_preprocess_threads % 4:
        raise ValueError('Please make num_preprocess_threads a multiple '
                         'of 4 (%d % 4 != 0).', num_preprocess_threads)
        
    if num_readers is None:
        num_readers = FLAGS.num_readers

    if num_readers < 1:
        raise ValueError('Please make num_readers at least 1')
    
    
    # Approximate number of examples per shard.
    examples_per_shard = 300
    # Size the random shuffle queue to balance between good global
    # mixing (more examples) and memory use (fewer examples).
    # 1 image uses 299*299*3*4 bytes = 1MB
    # The default input_queue_memory_factor is 16 implying a shuffling queue
    # size: examples_per_shard * 16 * 1MB = 17.6GB
    min_queue_examples = examples_per_shard * FLAGS.input_queue_memory_factor
    if split=='train':
        examples_queue = tf.RandomShuffleQueue(
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=min_queue_examples,
            dtypes=[tf.string])
        
    elif split=='validation':
        examples_queue = tf.FIFOQueue(
            capacity=examples_per_shard + 3 * batch_size,
            dtypes=[tf.string])
    


    if num_readers > 1:
        enqueue_ops = []
        for _ in range(num_readers):
            reader = tf.TFRecordReader()
            _, value = reader.read(filename_queue)
            enqueue_ops.append(examples_queue.enqueue([value]))

        tf.train.queue_runner.add_queue_runner(
            tf.train.queue_runner.QueueRunner(examples_queue, enqueue_ops))
        example_serialized = examples_queue.dequeue()
    else:
        reader = tf.TFRecordReader()
        _, example_serialized = reader.read(filename_queue)

    for thread_id in range(num_preprocess_threads):
        image_buffer, class_id, bbox = parse_tfrecords(example_serialized)
        
    image_labels = []
        
    
    scope = 'data_augmentation' if distords else 'input'
    
    with tf.name_scope(scope):
    
    dataset = dataset.map(DataPreprocessor(distords), num_parallel_calls=10)
    
    # Dataset is small enough to be fully loaded on memory:
    
    dataset = dataset.prefetch(-1)
    
    dataset = dataset.repeat().batch(batch_size)
    
    iterator = dataset.make_one_shot_iterator()
    
    images_labels = iterator.get_next()
    
    images, labels = images_labels['input'], images_labels['target']

    tf.summary.image('images', images)
    
    return images, labels





class DataPreprocessor(object):

  """Applies transformations to dataset record."""



  def __init__(self, distords):

    self._distords = distords



  def __call__(self, record):

    """Process img for training or eval."""

    img = record['image']

    img = tf.cast(img, tf.float32)

    if self._distords:  # training

      # Randomly crop a [height, width] section of the image.

      img = tf.random_crop(img, [IMAGE_SIZE, IMAGE_SIZE, 3])

      # Randomly flip the image horizontally.

      img = tf.image.random_flip_left_right(img)

      # Because these operations are not commutative, consider randomizing

      # the order their operation.

      # NOTE: since per_image_standardization zeros the mean and makes

      # the stddev unit, this likely has no effect see tensorflow#1458.

      img = tf.image.random_brightness(img, max_delta=63)

      img = tf.image.random_contrast(img, lower=0.2, upper=1.8)

    else:  # Image processing for evaluation.

      # Crop the central [height, width] of the image.

      img = tf.image.resize_image_with_crop_or_pad(img, IMAGE_SIZE, IMAGE_SIZE)

    # Subtract off the mean and divide by the variance of the pixels.

    img = tf.image.per_image_standardization(img)

    return dict(input=img, target=record['label'])





def distorted_inputs(batch_size):

  """Construct distorted input for CIFAR training using the Reader ops.



  Args:

    batch_size: Number of images per batch.



  Returns:

    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.

    labels: Labels. 1D tensor of [batch_size] size.

  """

  return _get_images_labels(batch_size, tfds.Split.TRAIN, distords=True)





def inputs(eval_data, batch_size):

  """Construct input for CIFAR evaluation using the Reader ops.



  Args:

    eval_data: bool, indicating if one should use the train or eval data set.

    batch_size: Number of images per batch.



  Returns:

    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.

    labels: Labels. 1D tensor of [batch_size] size.

  """

  split = tfds.Split.TEST if eval_data == 'test' else tfds.Split.TRAIN

  return _get_images_labels(batch_size, split)
