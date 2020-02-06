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

import crop_pad

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
        
    image_encoded = context['image/encoded']
    image_encoded = decode_jpeg(image_buffer, 3)

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

    return image_encoded, bbox, class_id #height, width, filename


def image_augmentation(image_encoded, bbox, class_id, split, thread_id=0):

    if split=='train':
        # Since having only samll dataset, augment dataset as much as possible
        images, boxes, labels = train_augmentation(image_encoded, class_id, bbox)
        
    elif split=='validation':
        images, boxes, labels = eval_augmentation(image_encoded, class_id, bbox)
        
    return images, boxes, labels

def train_augmentation(image_encoded, class_id, bbox):

  with tf.name_scope(scope, 'augmented_image'):
    if boxes is None:
      boxes = tf.constant([0.0, 0.0, 0.0, 0.0],
                         dtype=tf.float32,
                         shape=[1, 4])
      labels = tf.constant([0],
                         dtype=tf.int32,
                         shape=[1])

    # Each bounding box has shape [num_boxes, box coords] and
    # the coordinates are ordered [ymin, xmin, ymax, xmax].

    image = tf.to_float(image_encoded)
    # Random Horizontal Flip
    with tf.name_scope('RandomHorizontalFlip'):
        random_horizontal_flip_probability = FLAGS.random_horizontal_flip_probability
        def _flip_image(image):
            # flip image
            image_flipped = tf.image.flip_left_right(image)
            return image_flipped

        def _flip_boxes_left_right(boxes):
            """Left-right flip the boxes.

            Args:
              boxes: rank 2 float32 tensor containing the bounding boxes -> [N, 4].
                     Boxes are in normalized form meaning their coordinates vary
                     between [0, 1].
                     Each row is in the form of [ymin, xmin, ymax, xmax].

            Returns:
              Flipped boxes.
            """
            ymin, xmin, ymax, xmax = tf.split(value=boxes, num_or_size_splits=4, axis=1)
            flipped_xmin = tf.subtract(1.0, xmax)
            flipped_xmax = tf.subtract(1.0, xmin)
            flipped_boxes = tf.concat([ymin, flipped_xmin, ymax, flipped_xmax], 1)
            return flipped_boxes

        random = tf.random_uniform(
            [],
            minval=0,
            maxval=1,
            dtype=tf.float32,
            seed=None,
            name=None
        )
        # flip image
        image = tf.cond(tf.greater_equal(random, random_horizontal_flip_probability),
                        lambda: image,
                        lambda: _flip_image(image))

        # flip boxes
        boxes = tf.cond(tf.greater_equal(random, random_horizontal_flip_probability),
                        lambda: boxes,
                        lambda: _flip_boxes_left_right(boxes))

    # SSD Random Crop
    with tf.name_scope('SSDRandomCrop'):
        min_object_covered = (0.0,0.1,0.3,0.5,0.7,0.9,1.0)
        overlap_threshold = (0.0,0.1,0.3,0.5,0.7,0.9,1.0)
        num_cases = len(min_object_covered)
        random_idx = tf.random_uniform([], maxval=num_cases, dtype=tf.int32)
        selected_min_object_covered = tf.gather(min_object_covered, random_idx)
        selected_overlap_threshold = tf.gather(overlap_thresh, random_idx)
        random = tf.random_uniform([], minval=0, maxval=1, dtype=tf.float32)
        ssd_crop_padding=crop_pad()
        dst_image, labels, boxes = tf.cond(tf.greater(random_coef, random),
                                           lambda: (image, labels, boxes),
                                           lambda: ssd_crop_padding.random_crop_image(
                                               image=image,
                                               boxes=boxes,
                                               labels=labels,
                                               min_object_covered=selected_min_object_covered),
                                               overlap_thresh=selected_overlap_threshold
                                           )

    # Random Pad Image
    with tf.name_scope('RandomPadImage'):
        random_pad_probability = FLAGS.random_pad_probability

        def _random_integer(minval, maxval, seed):
            """Returns a random 0-D tensor between minval and maxval.

            Args:
              minval: minimum value of the random tensor.
              maxval: maximum value of the random tensor.
              seed: random seed.

            Returns:
              A random 0-D tensor between minval and maxval.
            """
            return tf.random_uniform(
                [], minval=minval, maxval=maxval, dtype=tf.int32, seed=seed)

        def random_pad(image,
                       boxes,
                       min_image_size=None,
                       max_image_size=None,
                       pad_color=None,
                       seed=None):
            if pad_color is None:
                pad_color = tf.reduce_mean(image, axis=[0, 1])

            image_shape = tf.shape(image)
            image_height = image_shape[0]
            image_width = image_shape[1]

            max_image_size = tf.stack([image_height * 3, image_width * 3])
            max_image_size = tf.maximum(max_image_size,
                                        tf.stack([image_height, image_width]))

            if min_image_size is None:
                min_image_size = tf.stack([image_height, image_width])
            min_image_size = tf.maximum(min_image_size,
                                        tf.stack([image_height, image_width]))

            target_height = tf.cond(
                max_image_size[0] > min_image_size[0],
                lambda: _random_integer(min_image_size[0], max_image_size[0], seed),
                lambda: max_image_size[0])

            target_width = tf.cond(
                max_image_size[1] > min_image_size[1],
                lambda: _random_integer(min_image_size[1], max_image_size[1], seed),
                lambda: max_image_size[1])

            offset_height = tf.cond(
                target_height > image_height,
                lambda: _random_integer(0, target_height - image_height, seed),
                lambda: tf.constant(0, dtype=tf.int32))

            offset_width = tf.cond(
                target_width > image_width,
                lambda: _random_integer(0, target_width - image_width, seed),
                lambda: tf.constant(0, dtype=tf.int32))

            new_image = tf.image.pad_to_bounding_box(
                image,
                offset_height=offset_height,
                offset_width=offset_width,
                target_height=target_height,
                target_width=target_width)

            # Setting color of the padded pixels
            image_ones = tf.ones_like(image)
            image_ones_padded = tf.image.pad_to_bounding_box(
                image_ones,
                offset_height=offset_height,
                offset_width=offset_width,
                target_height=target_height,
                target_width=target_width)
            image_color_padded = (1.0 - image_ones_padded) * pad_color
            new_image += image_color_padded

            # setting boxes
            new_window = tf.to_float(
                tf.stack([
                    -offset_height, -offset_width, target_height - offset_height,
                                                   target_width - offset_width
                ]))
            new_window /= tf.to_float(
                tf.stack([image_height, image_width, image_height, image_width]))

            win_height = new_window[2] - new_window[0]
            win_width = new_window[3] - new_window[1]

            new_boxes = boxes - [new_window[0], new_window[1], new_window[0], new_window[1]]
            y_scale = 1.0 / win_height
            x_scale = 1.0 / win_width
            y_scale = tf.cast(y_scale, tf.float32)
            x_scale = tf.cast(x_scale, tf.float32)
            y_min, x_min, y_max, x_max = tf.split(
                value=new_boxes, num_or_size_splits=4, axis=1)
            y_min = y_scale * y_min
            y_max = y_scale * y_max
            x_min = x_scale * x_min
            x_max = x_scale * x_max
            new_boxes = tf.concat([y_min, x_min, y_max, x_max], 1)

            return new_image, new_boxes

        random = tf.random_uniform(
            [],
            minval=0,
            maxval=1,
            dtype=tf.float32,
            seed=None,
            name=None
        )

        dst_image, boxes = tf.cond(tf.greater_equal(random, random_pad_probability),
                                   lambda: (dst_image, boxes),
                                   lambda: random_pad(dst_image,
                                   boxes))


        # pad zeros to each tensor with max boxes
        num_object = tf.shape(labels)[0]
        zeropad_weights = tf.pad(tf.ones([num_object]), tf.convert_to_tensor([[0, max_boxes - num_object]]))
        zeropad_box = tf.pad(boxes, tf.convert_to_tensor([[0, max_boxes - num_object], [0, 0]]))
        zeropad_label = tf.pad(labels, tf.convert_to_tensor([[0, max_boxes - num_object]]))

        zeropad_box.set_shape([max_boxes, 4])
        zeropad_label.set_shape([max_boxes])
        zeropad_weights.set_shape([max_boxes])
        dst_image.set_shape([None,None,3])
        with tf.name_scope(
                'ResizeImage',
                values=[dst_image, height, width]):
            new_image = tf.image.resize_images(
                dst_image, tf.stack([height, width]))
            static_tensor_shape = image.shape.as_list()
            dynamic_tensor_shape = tf.shape(dst_image)
            combined_shape = []
            for index, dim in enumerate(static_tensor_shape):
                if dim is not None:
                    combined_shape.append(dim)
                else:
                    combined_shape.append(dynamic_tensor_shape[index])
            image_shape = combined_shape
            result = [new_image]
            result.append(tf.stack([height, width, image_shape[2]]))
        dst_image = result[0]
        true_image_shape = result[1]
        distorted_image = (2.0 / 255.0) * dst_image - 1.0

    return distorted_image, zeropad_label, zeropad_box, zeropad_weights, true_image_shape

def eval_augmentation(image_encoded, class_id, bbox):


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
        bbox, class_id, image_encoded = parse_tfrecords(example_serialized)
        
        images, labels, boxes = image_augmentation(image_encoded,
                                                   class_id,
                                                   bbox,
                                                   split,
                                                   thread_id)
        
        
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
