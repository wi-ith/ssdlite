import tensorflow as tf
import numpy as np

FLAGS = tf.app.flags.FLAGS


def soft_max(logits, axis=-1):
    tile_depth = logits.shape[axis]
    exp_logits = tf.exp(logits)
    exp_sum = tf.tile(tf.reshape((tf.reduce_sum(exp_logits, axis=axis) + .1E-6), [-1, 1]), [1, tile_depth])
    soft_max = exp_logits / exp_sum
    return soft_max

def focal_loss(one_hot_label,logits,gamma=2,axis=-1):


    prediction = soft_max(logits,axis)
    pos_pred=tf.reduce_sum(prediction*one_hot_label,axis=-1)
    focal_factor = tf.math.pow((1.-pos_pred),gamma)
    loss = tf.losses.softmax_cross_entropy(one_hot_label,logits,reduction="none")
    return focal_factor*loss


def make_anchor(feature_maps,
                min_scale,
                max_scale,
                ratio_list
                ):
    '''
    Input:
    feature_mapes : list of [batch_size, W, H, num_class*num_ratio]
                ex) SSDLite300 --> {
                                    [batch_size, 19, 19, num_class*3],
                                    [batch_size, 10, 10, num_class*6],
                                    [batch_size, 5, 5, num_class*6],
                                    [batch_size, 3, 3, num_class*6],
                                    [batch_size, 2, 2, num_class*6],
                                    [batch_size, 1, 1, num_class*6]
                                   }
    min_scale :
    max_scale :
    ratio_list : list of target ratio

    return:
    anchor_concat : [num_anchors,4]
                ex) SSDLite300 --> [1917,4]

    '''

    scale_range=np.array(range(len(feature_maps)))+1
    scale_list=(scale_range-1.)*((max_scale-min_scale)/(len(feature_maps)-1.))+min_scale
    scale_list=[.1]+list(scale_list)+[1.]
    anchor_list = []


    for i, (feature,ratio) in enumerate(zip(feature_maps,ratio_list)):

        anchor_size=feature.get_shape().as_list()[1:3]

        coordi_range_y = tf.cast(tf.range(0, anchor_size[0], 1), dtype=tf.float32)
        coordi_range_x = tf.cast(tf.range(0, anchor_size[1], 1), dtype=tf.float32)

        y1 = tf.reshape(coordi_range_y, [anchor_size[0], 1])
        x1 = tf.reshape(coordi_range_x, [1, anchor_size[1]])

        y1 = tf.tile(y1, [1, anchor_size[0]])
        x1 = tf.tile(x1, [anchor_size[1], 1])

        y1 = tf.reshape(y1, [-1, anchor_size[0]])
        x1 = tf.reshape(x1, [-1, anchor_size[1]])

        anchor_y1x1 = tf.stack([y1, x1], axis=2)
        anchor_y2x2 = anchor_y1x1 + 1

        anchor_y1x1 = anchor_y1x1 / anchor_size[0]
        anchor_y2x2 = anchor_y2x2 / anchor_size[1]

        scale_anchor = []
        # ratio = width / height
        for k, one_ratio in enumerate(ratio):
            #idx == 0  ->>  first anchor scale
            if i == 0 and one_ratio == 1.:
                idx = i
            else:
                idx = i+1

            if k==5:
                scale_ = tf.cast(tf.math.sqrt(scale_list[idx]*scale_list[idx+1]),dtype=tf.float32)
            else:
                scale_ = tf.cast(scale_list[idx],dtype=tf.float32)

            x_offset = 1. / 2. * (1. / anchor_size[0] - scale_ * tf.math.sqrt(one_ratio))
            y_offset = 1. / 2. * (1. / anchor_size[0] - scale_ / tf.math.sqrt(one_ratio))
            anchor_y1x1 = tf.reshape(anchor_y1x1, [-1, 2])
            anchor_y2x2 = tf.reshape(anchor_y2x2, [-1, 2])
            anchor_y1 = anchor_y1x1[:, 0] + y_offset
            anchor_x1 = anchor_y1x1[:, 1] + x_offset
            anchor_y2 = anchor_y2x2[:, 0] - y_offset
            anchor_x2 = anchor_y2x2[:, 1] - x_offset

            anchor_y1x1y2x2 = tf.stack([anchor_y1, anchor_x1, anchor_y2, anchor_x2], axis=1)
            scale_anchor.append(anchor_y1x1y2x2)

        # num_feature_block x num_ratio x 4
        scale_anchor = tf.stack(scale_anchor, axis=1)
        scale_anchor = tf.reshape(scale_anchor,[-1,4])
        anchor_list.append(scale_anchor)

    anchor_concat = tf.concat(anchor_list, axis=0)
    return anchor_concat


def decode_logits(anchor_concat, feature_maps_cls, feature_maps_loc):
    '''

    feature_maps_cls: [[batch_size, H, W, num_class * num_ratio], ...]
    feature_maps_loc: [[batch_size, H, W, 4 * num_ratio], ...]

    :return:

    '''

    for k, (feature_cls,feature_loc) in enumerate(zip(feature_maps_cls,feature_maps_loc)):
        origin_shape=feature_cls.shape
        if k==0:
            feature_concat_cls = tf.reshape(feature_cls, [origin_shape[0], -1, FLAGS.num_classes])
            feature_concat_loc = tf.reshape(feature_loc, [origin_shape[0], -1, 4])
        else:
            reshape_featur_cls = tf.reshape(feature_cls, [origin_shape[0], -1, FLAGS.num_classes])
            feature_concat_cls = tf.concat([feature_concat_cls,reshape_featur_cls],axis=1)

            reshape_featur_loc = tf.reshape(feature_loc, [origin_shape[0], -1, 4])
            feature_concat_loc = tf.concat([feature_concat_loc, reshape_featur_loc], axis=1)

    cls_logits_list = tf.unstack(feature_concat_cls,axis=0)
    loc_logits_list = tf.unstack(feature_concat_loc,axis=0)

    loc_pred_list=[]
    cls_pred_list = []

    reshape_anchor = tf.reshape(anchor_concat, [-1, 4])

    anchor_cy = (reshape_anchor[:, 0] + reshape_anchor[:, 2]) / 2.
    anchor_cx = (reshape_anchor[:, 1] + reshape_anchor[:, 3]) / 2.
    anchor_h = reshape_anchor[:, 2] - reshape_anchor[:, 0]
    anchor_w = reshape_anchor[:, 3] - reshape_anchor[:, 1]

    for  cls_logit, loc_logit in zip(cls_logits_list, loc_logits_list):
        logit_cy = loc_logit[:, 0]
        logit_cx = loc_logit[:, 1]
        logit_h = loc_logit[:, 2]
        logit_w = loc_logit[:, 3]

        pred_cy = logit_cy * anchor_h + anchor_cy
        pred_cx = logit_cx * anchor_w + anchor_cx
        pred_h = tf.exp(logit_h) * anchor_h
        pred_w = tf.exp(logit_w)*anchor_w

        ymin = pred_cy-pred_h/2.
        xmin = pred_cx-pred_w/2.
        ymax = pred_cy+pred_h/2.
        xmax = pred_cx+pred_w/2.

        pred_loc=tf.stack([ymin,xmin,ymax,xmax], axis=1)

        cls_pred = soft_max(cls_logit)

        loc_pred_list.append(pred_loc)

        cls_pred_list.append(cls_pred)



    return cls_pred_list, loc_pred_list



def anchor_matching_cls_loc_loss(anchor_concat,
                                 feature_maps_cls,
                                 feature_maps_loc,
                                 labels,
                                 boxes,
                                 num_objects,
                                 positive_threshold,
                                 negative_threshold,
                                 num_classes,
                                 max_boxes):

    '''
    Input:

    anchor_concat: [num_anchors, 4]
    feature_maps_cls: [[batch_size, H, W, num_class * num_ratio], ...]
    feature_maps_loc: [[batch_size, H, W, 4 * num_ratio], ...]
    boxes: [batch_size, num_objects, 4]
    labels: [batch_size, num_objects]
    positive_threshold: positive matching iou threshold
    negative_threshold: negative matching iou threshold
    num_classes: number of classes (including negative class)
    max_boxes: number of max detected boxes


    return:

    cls_loss_sum : sum of classification loss
    loc_loss_sum : sum of locaization loss
    '''

    for k, (feature_cls,feature_loc) in enumerate(zip(feature_maps_cls,feature_maps_loc)):
        origin_shape=feature_cls.shape
        if k==0:
            feature_concat_cls = tf.reshape(feature_cls, [origin_shape[0], -1, num_classes])
            feature_concat_loc = tf.reshape(feature_loc, [origin_shape[0], -1, 4])
        else:
            reshape_featur_cls = tf.reshape(feature_cls, [origin_shape[0], -1, num_classes])
            feature_concat_cls = tf.concat([feature_concat_cls,reshape_featur_cls],axis=1)

            reshape_featur_loc = tf.reshape(feature_loc, [origin_shape[0], -1, 4])
            feature_concat_loc = tf.concat([feature_concat_loc, reshape_featur_loc], axis=1)

    num_objects_list = tf.unstack(num_objects,axis=0)
    boxes_list = tf.unstack(boxes,axis=0)
    labels_list = tf.unstack(labels,axis=0)
    cls_logits_list = tf.unstack(feature_concat_cls,axis=0)
    loc_logits_list = tf.unstack(feature_concat_loc,axis=0)

    reshape_anchor = tf.reshape(anchor_concat, [-1, 4])

    anchor_cy = (reshape_anchor[:, 0] + reshape_anchor[:, 2]) / 2.
    anchor_cx = (reshape_anchor[:, 1] + reshape_anchor[:, 3]) / 2.
    anchor_h = reshape_anchor[:, 2] - reshape_anchor[:, 0]
    anchor_w = reshape_anchor[:, 3] - reshape_anchor[:, 1]

    gt_offset_list = []
    cls_loss_list=[]
    loc_loss_list=[]

    for num_objects_, boxes_, labels_, cls_logit, loc_logit in zip(num_objects_list, boxes_list, labels_list, cls_logits_list, loc_logits_list):

        labels_ = tf.concat([[-1, 0], labels_[:-2]], axis=0)
        # ground truth for ignore, negative anchor
        boxes_ = tf.concat([[[0,0,0,0],[0,0,0,0]],boxes_[:-2,:]],axis=0)

        #num_obj = tf.constant(boxes.get_shape().as_list()[0])
        num_obj = num_objects_ + 2


        # num_obj = num_objects_
        # boxes_ = tf.pad(boxes_, tf.convert_to_tensor([[0, max_boxes - num_obj], [0, 0]]), "CONSTANT")

        paded_boxes = tf.reshape(boxes_, [max_boxes, 4])

        # [1917,4]
        anchor_concat = tf.reshape(anchor_concat, [-1, 4])


        # num_anchor x 100 x 4
        tile_GT = tf.tile(tf.reshape(paded_boxes, [1, max_boxes, 4]), [anchor_concat.shape[0], 1, 1])
        tile_anchor = tf.tile(tf.expand_dims(anchor_concat, axis=1), [1, paded_boxes.shape[0], 1])


        inter_y1 = tf.maximum(tile_GT[:, :, 0], tile_anchor[:, :, 0])
        inter_x1 = tf.maximum(tile_GT[:, :, 1], tile_anchor[:, :, 1])
        inter_y2 = tf.minimum(tile_GT[:, :, 2], tile_anchor[:, :, 2])
        inter_x2 = tf.minimum(tile_GT[:, :, 3], tile_anchor[:, :, 3])

        # num_anchor x 100
        GT_area = (tile_GT[:, :, 2] - tile_GT[:, :, 0]) * (tile_GT[:, :, 3] - tile_GT[:, :, 1])
        anchor_area = (tile_anchor[:, :, 2] - tile_anchor[:, :, 0]) * (tile_anchor[:, :, 3] - tile_anchor[:, :, 1])

        # num_anchor x 100
        inter_height = inter_y2 - inter_y1
        inter_width = inter_x2 - inter_x1
        height_zero_mask = tf.cast(tf.greater(inter_height, 0.), dtype=tf.float32)
        width_zero_mask = tf.cast(tf.greater(inter_width, 0.), dtype=tf.float32)
        inter_height = inter_height * height_zero_mask
        inter_width = inter_width * width_zero_mask
        intersection_area = inter_height * inter_width

        # num_anchor
        iou = intersection_area / (GT_area + anchor_area - intersection_area)
        max_iou_idx = tf.argmax(iou, axis=-1)
        max_iou = tf.reduce_max(iou, axis=-1)
        negative_mask = tf.less_equal(max_iou, negative_threshold)
        ignore_mask = tf.logical_and(tf.greater(max_iou, negative_threshold),
                                     tf.less_equal(max_iou, positive_threshold))
        positive_mask = tf.greater(iou,positive_threshold)

        # ignore label index : 0
        # negative label index : 1
        # positive label index : 2 ~
        pos_label_idx = max_iou_idx * (1 - tf.cast(negative_mask, dtype=tf.int64)) + tf.cast(negative_mask, dtype=tf.int64)
        pos_label_idx = pos_label_idx * (1 - tf.cast(ignore_mask, dtype=tf.int64))


        #set unmatched box(iou is less than positive threshold) to maximum iou anchor
        keep_one_pos_label=tf.reduce_sum(tf.cast(positive_mask, dtype=tf.float32), axis=0)
        non_matched_label_index = keep_one_pos_label[2:num_obj]
        non_matched_label_mask = tf.cast(tf.equal(non_matched_label_index,0.), dtype=tf.float32)
        non_matched_label_mask = tf.pad(tf.reshape(non_matched_label_mask,[-1,1]), tf.convert_to_tensor([[2, max_boxes - (num_obj)],[0,0]]), "CONSTANT")
        non_matched_label_mask = tf.squeeze(non_matched_label_mask)
        non_matched_label_mask = tf.tile(tf.reshape(non_matched_label_mask, [1, max_boxes]), [iou.get_shape().as_list()[0], 1])
        non_matched_label_iou = iou*non_matched_label_mask

        pos_anchor_mask = tf.greater(tf.reduce_sum(tf.cast(positive_mask, dtype=tf.float32), axis=1),0.)
        pos_anchor_mask = tf.cast(pos_anchor_mask,dtype=tf.float32)
        pos_anchor_mask = tf.tile(tf.reshape(pos_anchor_mask, [iou.get_shape().as_list()[0], 1]),[1, max_boxes] )
        non_matched_label_iou = non_matched_label_iou*(1.-pos_anchor_mask)

        non_matched_max_iou_0 = tf.reduce_max(non_matched_label_iou, axis=0)
        # tmp = tf.argmax(non_matched_label_iou,axis=0)
        non_matched_max_iou_0 = tf.tile(tf.reshape(non_matched_max_iou_0,[1,max_boxes]),[iou.get_shape().as_list()[0], 1])
        iou_zero_mask = tf.equal(iou,0.)
        iou_cutzero = iou-tf.cast(iou_zero_mask,dtype=tf.float32)
        non_matched_max_iou = tf.equal(iou_cutzero,non_matched_max_iou_0)
        non_matched_max_label=tf.argmax(tf.cast(non_matched_max_iou,dtype=tf.float32),axis=1)

        non_matched_max_label_mask=tf.greater(non_matched_max_label,0)
        #merge unmatched label
        pos_label_idx=pos_label_idx*(1-tf.cast(non_matched_max_label_mask,dtype=tf.int64))+non_matched_max_label

        matched_label = tf.gather(labels_, pos_label_idx)
        matched_boxes = tf.gather(boxes_, pos_label_idx)

        matched_mask = tf.greater(pos_label_idx,0)
        matched_boxes_mask = tf.stack([matched_mask]*4,axis=1)

        #convert gt coordinate to offset
        gt_cy = (matched_boxes[:, 0] + matched_boxes[:, 2]) / 2.
        gt_cx = (matched_boxes[:, 1] + matched_boxes[:, 3]) / 2.
        gt_h = matched_boxes[:, 2] - matched_boxes[:, 0]
        gt_w = matched_boxes[:, 3] - matched_boxes[:, 1]

        gt_offset_cy = tf.reshape((gt_cy - anchor_cy) / (anchor_h + .1e-5), [-1, 1])
        gt_offset_cx = tf.reshape((gt_cx - anchor_cx) / (anchor_w + .1e-5), [-1, 1])

        gt_offset_h = tf.reshape(tf.log(gt_h / anchor_h + .1e-5) * (1. - tf.cast(tf.equal(gt_h, 0.), dtype=tf.float32)),
                                 [-1, 1])
        gt_offset_w = tf.reshape(tf.log(gt_w / anchor_w + .1e-5) * (1. - tf.cast(tf.equal(gt_w, 0.), dtype=tf.float32)),
                                 [-1, 1])

        gt_offset = tf.concat([gt_offset_cy, gt_offset_cx, gt_offset_h, gt_offset_w], axis=1)

        gt_offset = tf.cast(matched_boxes_mask,dtype=tf.float32)*gt_offset
        gt_offset_list.append(gt_offset)

        loc_loss=tf.losses.huber_loss(
            gt_offset,
            loc_logit,
            delta=1.0,
            loss_collection=None,
            reduction=tf.losses.Reduction.NONE
        )
        loc_loss = loc_loss / tf.reduce_sum(tf.cast(matched_mask,dtype=tf.float32))

        one_hot_label=tf.one_hot(tf.cast(matched_label,dtype=tf.int32),depth=num_classes)
        cls_loss=focal_loss(one_hot_label,cls_logit,gamma=2)

        cls_loss = cls_loss / tf.reduce_sum(tf.cast(matched_mask, dtype=tf.float32))

        cls_loss_list.append(cls_loss)
        loc_loss_list.append(loc_loss)

    cls_loss_sum = tf.reduce_sum(tf.stack(cls_loss_list))
    loc_loss_sum = tf.reduce_sum(tf.stack(loc_loss_list))
    return cls_loss_sum, loc_loss_sum
#
# #
# boxes=tf.convert_to_tensor([[[0.0, 0.0, 0.5, 0.5],[0.2, 0.1, 0.25, 0.15],[0.2, 0.2, 0.25, 0.25]],[[0.0, 0.0, 0.5, 0.5],[0.2, 0.1, 0.25, 0.15],[0.2, 0.2, 0.25, 0.25]]])
# labels=tf.convert_to_tensor([[1., 2., 7.],[1., 2., 7.]])
#
# num_classes=10
#
# feature_maps_cls=[tf.random_uniform([2,19,19,num_classes*3],minval=-2.0,maxval=2.0,dtype=tf.float32),
#                   tf.random_uniform([2,10,10,num_classes*6],minval=-2.0,maxval=2.0,dtype=tf.float32),
#                   tf.random_uniform([2,5,5,num_classes*6],minval=-2.0,maxval=2.0,dtype=tf.float32),
#                   tf.random_uniform([2,3,3,num_classes*6],minval=-2.0,maxval=2.0,dtype=tf.float32),
#                   tf.random_uniform([2,2,2,num_classes*6],minval=-2.0,maxval=2.0,dtype=tf.float32),
#                   tf.random_uniform([2,1,1,num_classes*6],minval=-2.0,maxval=2.0,dtype=tf.float32)]
# feature_maps_loc=[tf.random_uniform([2,19,19,4*3],minval=0,maxval=1.0,dtype=tf.float32),
#                   tf.random_uniform([2,10,10,4*6],minval=0,maxval=1.0,dtype=tf.float32),
#                   tf.random_uniform([2,5,5,4*6],minval=0,maxval=1.0,dtype=tf.float32),
#                   tf.random_uniform([2,3,3,4*6],minval=0,maxval=1.0,dtype=tf.float32),
#                   tf.random_uniform([2,2,2,4*6],minval=0,maxval=1.0,dtype=tf.float32),
#                   tf.random_uniform([2,1,1,4*6],minval=0,maxval=1.0,dtype=tf.float32)]
#
# ratio_list=[[1., 2., 1./2.]]+[[1., 2., 1. / 2., 3., 1. / 3., 1.]]*5
# # print(ratio_list)
#
#
# max_boxes=100
# negative_threshold=0.3
# positive_threshold=0.5
# min_scale=0.2
# max_scale=0.95
#
# anchor_concat=make_anchor(feature_maps_cls, min_scale, max_scale, ratio_list)
# res=anchor_matching_cls_loc_loss(anchor_concat,
#                                  feature_maps_cls,
#                                  feature_maps_loc,
#                                  labels,
#                                  boxes,
#                                  tf.convert_to_tensor(np.array([3,3])),
#                                  positive_threshold,
#                                  negative_threshold,
#                                  num_classes,
#                                  max_boxes)
#
#
# sess=tf.Session()
# tt=sess.run(anchor_concat)
# print(tt.shape)
# tt[-30:-6,:]
