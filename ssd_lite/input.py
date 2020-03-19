# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 01:05:20 2020

@author: wi-ith
"""



import tensorflow as tf

import crop_pad

import os

FLAGS = tf.app.flags.FLAGS



def decode_jpeg(image_buffer, channels, scope=None):
    """Decode a JPEG string into one 3-D float image Tensor.

    Args:
      image_buffer: scalar string Tensor.
      scope: Optional scope for name_scope.
    Returns:
      3-D float Tensor with values ranging from [0, 1).
    """
    with tf.name_scope(values=[image_buffer], name=scope,
                       default_name='decode_jpeg'):
        # Decode the string as an RGB or GRAY JPEG.
        # Note that the resulting image contains an unknown height and width
        # that is set dynamically by decode_jpeg. In other words, the height
        # and width of image is unknown at compile-time.
        image = tf.image.decode_jpeg(image_buffer, channels)
        return image

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
            'image/object/bbox/xmin':
                tf.VarLenFeature(tf.float32),
            'image/object/bbox/xmax':
                tf.VarLenFeature(tf.float32),
            'image/object/bbox/ymin':
                tf.VarLenFeature(tf.float32),
            'image/object/bbox/ymax':
                tf.VarLenFeature(tf.float32),
                                
            'image/object/class/text':
                tf.VarLenFeature(tf.string),
            'image/object/class/label':
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
    image_encoded = decode_jpeg(image_encoded, 3)

    xmin = tf.expand_dims(context['image/object/bbox/xmin'].values, 0)
    ymin = tf.expand_dims(context['image/object/bbox/ymin'].values, 0)
    xmax = tf.expand_dims(context['image/object/bbox/xmax'].values, 0)
    ymax = tf.expand_dims(context['image/object/bbox/ymax'].values, 0)
    bbox = tf.concat(axis=0, values=[ymin, xmin, ymax, xmax])
    bbox = tf.transpose(bbox, [1, 0])
    
    #class_name = context['class/text'].values
    class_id = tf.cast(context['image/object/class/label'].values, dtype=tf.int32)
    
    #height = context['image/height']
    #width = context['image/width']    
    #filename = context['image/filename']

    return image_encoded, class_id, bbox #height, width, filename


def distorted_inputs(batch_size):

    if not batch_size:
        batch_size = FLAGS.batch_size
        
    with tf.device('/cpu:0'):
         images_batch, labels_batch, boxes_batch, num_objects_batch = _get_images_labels(batch_size, 'train', FLAGS.num_readers)
    
    return images_batch, labels_batch, boxes_batch, num_objects_batch



def inputs(batch_size):

    if not batch_size:
        batch_size = FLAGS.batch_size
        
    with tf.device('/cpu:0'):
         images_batch, labels_batch, boxes_batch, num_objects_batch = _get_images_labels(batch_size, 'validation', 1)
    
    return images_batch, labels_batch, boxes_batch, num_objects_batch



def _get_images_labels(batch_size, split, num_readers, num_preprocess_threads=None):

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

    batch_input=[]
    for thread_id in range(num_preprocess_threads):
        image_encoded, class_id, bbox = parse_tfrecords(example_serialized)
        
        images, labels, boxes, num_objects = image_augmentation(image_encoded,
                                                               class_id,
                                                               bbox,
                                                               split,
                                                               thread_id)
        batch_input.append([images, labels, boxes, num_objects])
    
    images_batch, labels_batch, boxes_batch, num_objects_batch = tf.train.batch_join(
                                                                batch_input,
                                                                batch_size=batch_size,
                                                                capacity=2 * num_preprocess_threads * batch_size)
    
    height = FLAGS.image_size
    width = FLAGS.image_size
    depth = 3
    max_boxes = FLAGS.max_boxes
    
    images_batch = tf.cast(images_batch, tf.float32)
    images_batch = tf.reshape(images_batch, shape=[batch_size, height, width, depth])

    labels_batch = tf.cast(labels_batch, tf.int32)
    labels_batch = tf.reshape(labels_batch, shape=[batch_size, max_boxes])

    boxes_batch = tf.cast(boxes_batch, tf.float32)
    boxes_batch = tf.reshape(boxes_batch, shape=[batch_size, max_boxes, 4])

    num_objects_batch = tf.cast(num_objects_batch, tf.int32)
    num_objects_batch = tf.reshape(num_objects_batch, shape=[batch_size])
    
    image_with_box_batch = tf.image.draw_bounding_boxes(images_batch, boxes_batch)
    tf.summary.image('frames', image_with_box_batch)

    return images_batch, labels_batch, boxes_batch, num_objects_batch



def image_augmentation(image_encoded, class_id, bbox, split, thread_id=0):

    if split=='train':
        # Since having only samll dataset, augment dataset as much as possible
        images, labels, boxes, num_objects = train_augmentation(image_encoded, class_id, bbox)
        
    elif split=='validation':
        images, labels, boxes, num_objects = eval_augmentation(image_encoded, class_id, bbox)
        
    return images, labels, boxes, num_objects


def train_augmentation(image_encoded, labels, boxes):
    ##Flip 50% -> crop 85% -> pad 40%

    with tf.name_scope('augmented_image'):
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
            random_flip_prob = FLAGS.random_flip_prob

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
            image = tf.cond(tf.greater_equal(random, random_flip_prob),
                            lambda: image,
                            lambda: _flip_image(image))

            # flip boxes
            boxes = tf.cond(tf.greater_equal(random, random_flip_prob),
                            lambda: boxes,
                            lambda: _flip_boxes_left_right(boxes))

        # SSD Random Crop
        with tf.name_scope('SSDRandomCrop'):
            min_object_covered = (0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0)
            overlap_threshold = (0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0)
            random_coef = 0.15
            num_cases = len(min_object_covered)
            random_idx = tf.random_uniform([], maxval=num_cases, dtype=tf.int32)
            selected_min_object_covered = tf.gather(min_object_covered, random_idx)
            selected_overlap_threshold = tf.gather(overlap_threshold, random_idx)
            random = tf.random_uniform([], minval=0, maxval=1, dtype=tf.float32)
            dst_image, labels, boxes = tf.cond(tf.greater(random_coef, random),
                                               lambda: (image, labels, boxes),
                                               lambda: crop_pad.random_crop_image(
                                                   image=image,
                                                   boxes=boxes,
                                                   labels=labels,
                                                   min_object_covered=selected_min_object_covered,
                                                   overlap_thresh = selected_overlap_threshold)
                                               )

        # Random Pad Image
        with tf.name_scope('RandomPadImage'):
            random_pad_prob = FLAGS.random_pad_prob
            max_boxes = FLAGS.max_boxes

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

                # reset the coordinate system to the padded one
                new_boxes = boxes - [new_window[0], new_window[1], new_window[0], new_window[1]]
                norm_y = 1.0 / (new_window[2] - new_window[0])
                norm_x = 1.0 / (new_window[3] - new_window[1])
                norm_y = tf.cast(norm_y, tf.float32)
                norm_x = tf.cast(norm_x, tf.float32)

                y_min, x_min, y_max, x_max = tf.split(new_boxes, 4, 1)
                y_min = y_min * tf.cast(norm_y, tf.float32)
                y_max = y_max * tf.cast(norm_y, tf.float32)
                x_min = x_min * tf.cast(norm_x, tf.float32)
                x_max = x_max * tf.cast(norm_x, tf.float32)

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

            dst_image, boxes = tf.cond(tf.greater_equal(random, random_pad_prob),
                                       lambda: (dst_image, boxes),
                                       lambda: random_pad(dst_image,
                                                          boxes))

            # pad zeros to each tensor with max boxes
            num_object = tf.shape(labels)[0]
            zeropad_box = tf.pad(boxes, tf.convert_to_tensor([[0, max_boxes - num_object], [0, 0]]))
            zeropad_label = tf.pad(labels, tf.convert_to_tensor([[0, max_boxes - num_object]]))

            zeropad_box.set_shape([max_boxes, 4])
            zeropad_label.set_shape([max_boxes])

            dst_image.set_shape([None, None, 3])

        with tf.name_scope('ResizeImage'):
            new_image = tf.image.resize_images(dst_image, tf.stack([FLAGS.image_size, FLAGS.image_size]))
        distorted_image = (2.0 / 255.0) * new_image - 1.0

        return distorted_image, zeropad_label, zeropad_box, num_object


def eval_augmentation(image_encoded, labels, boxes):
    with tf.name_scope('eval_image'):
        max_boxes = FLAGS.max_boxes
        num_object = tf.shape(labels)[0]
        zeropad_boxes = tf.pad(boxes, tf.convert_to_tensor([[0, max_boxes - num_object], [0, 0]]))
        zeropad_labels = tf.pad(labels, tf.convert_to_tensor([[0, max_boxes - num_object]]))

        zeropad_boxes.set_shape([max_boxes, 4])
        zeropad_labels.set_shape([max_boxes])

        image_encoded.set_shape([None, None, 3])
        with tf.name_scope('ResizeImage'):
            new_image = tf.image.resize_images(image_encoded, tf.stack([FLAGS.image_size, FLAGS.image_size]))
        image = (2.0 / 255.0) * new_image - 1.0

        return image, zeropad_labels, zeropad_boxes, num_object