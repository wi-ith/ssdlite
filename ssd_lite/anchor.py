import tensorflow as tf
import numpy as np
#depth 960, 1280, 512, 256, 256, 128
boxes=tf.convert_to_tensor([[0.0, 0.0, 0.5, 0.5],[0.2, 0.1, 0.25, 0.15]])
labels=tf.convert_to_tensor([1., 2.])
feature_maps=[tf.ones([2,19,19,960]),
              tf.ones([2,10,10,1280]),
              tf.ones([2,5,5,512]),
              tf.ones([2,3,3,256]),
              tf.ones([2,2,2,256]),
              tf.ones([2,1,1,128])]

ratio_list=[[1., 2., 1./2.]]+[[1., 2., 1. / 2., 3., 1. / 3., 1.]]*5
# print(ratio_list)
min_scale=0.2
max_scale=0.95
scale_range=np.array(range(len(feature_maps)))+1
scale_list=(scale_range-1.)*((max_scale-min_scale)/(len(feature_maps)-1.))+min_scale
scale_list=[.1]+list(scale_list)+[1.]

max_boxes=100
negative_threshold=0.3
positive_threshold=0.5

anchor_list=[]
matched_label_list = []
gt_offset_list = []
iou_list=[]
scale_anchor_list=[]
positive_list=[]


# -1 : ignore label
# 0 : negative label
labels = tf.concat([[-1, 0], labels], axis=0)
# ground truth for ignore, negative anchor
boxes = tf.concat([[[0,0,0,0],[0,0,0,0]],boxes],axis=0)

for i, (feature,ratio) in enumerate(zip(feature_maps,ratio_list)):
    anchor_size=feature.get_shape().as_list()[1:3]

    coordi_range = tf.cast(tf.range(0, anchor_size[0], 1), dtype=tf.float32)

    y1 = tf.reshape(coordi_range, [anchor_size[0], 1])
    x1 = tf.reshape(coordi_range, [1, anchor_size[0]])

    y1 = tf.tile(y1, [1, anchor_size[0]])
    x1 = tf.tile(x1, [anchor_size[0], 1])

    y1 = tf.reshape(y1, [-1, anchor_size[0]])
    x1 = tf.reshape(x1, [-1, anchor_size[0]])

    anchor_x1y1 = tf.stack([x1, y1], axis=2)
    anchor_x2y2 = anchor_x1y1 + 1

    anchor_x1y1 = anchor_x1y1/anchor_size[0]
    anchor_x2y2 = anchor_x2y2 / anchor_size[0]

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
        anchor_x1y1 = tf.reshape(anchor_x1y1, [-1, 2])
        anchor_x2y2 = tf.reshape(anchor_x2y2, [-1, 2])
        anchor_x1 = anchor_x1y1[:, 0] + x_offset
        anchor_y1 = anchor_x1y1[:, 1] + y_offset
        anchor_x2 = anchor_x2y2[:, 0] - x_offset
        anchor_y2 = anchor_x2y2[:, 1] - y_offset

        anchor_x1y1x2y2 = tf.stack([anchor_x1, anchor_y1, anchor_x2, anchor_y2], axis=1)
        scale_anchor.append(anchor_x1y1x2y2)

    # num_feature_block x num_ratio x 4
    scale_anchor = tf.stack(scale_anchor, axis=1)
<<<<<<< HEAD
=======
    num_obj = tf.constant(boxes.get_shape().as_list()[0])
>>>>>>> bcb9f3da41e2b91d352b32a676c7ced75ef2da5e

    scale_anchor = tf.reshape(scale_anchor,[-1,4])
    anchor_list.append(scale_anchor)
anchor_concat = tf.concat(anchor_list, axis=0)

num_obj = tf.constant(boxes.get_shape().as_list()[0])

# 100 x 4
paded_boxes = tf.pad(boxes, tf.convert_to_tensor([[0, max_boxes - num_obj], [0, 0]]), "CONSTANT")
paded_boxes = tf.reshape(paded_boxes, [100, 4])

# [1917,4]
anchor_concat = tf.reshape(anchor_concat, [-1, 4])

# num_anchor x 100 x 4
tile_GT = tf.tile(tf.reshape(paded_boxes, [1, 100, 4]), [anchor_concat.shape[0], 1, 1])
tile_anchor = tf.tile(tf.expand_dims(anchor_concat, axis=1), [1, paded_boxes.shape[0], 1])

inter_x1 = tf.maximum(tile_GT[:, :, 0], tile_anchor[:, :, 0])
inter_y1 = tf.maximum(tile_GT[:, :, 1], tile_anchor[:, :, 1])
inter_x2 = tf.minimum(tile_GT[:, :, 2], tile_anchor[:, :, 2])
inter_y2 = tf.minimum(tile_GT[:, :, 3], tile_anchor[:, :, 3])

# num_anchor x 100
GT_area = (tile_GT[:, :, 2] - tile_GT[:, :, 0]) * (tile_GT[:, :, 3] - tile_GT[:, :, 1])
anchor_area = (tile_anchor[:, :, 2] - tile_anchor[:, :, 0]) * (tile_anchor[:, :, 3] - tile_anchor[:, :, 1])



sess=tf.Session()
tmp1,tmp2=sess.run([anchor_list,anchor_concat])

print(tmp1[-2])
print(tmp2[-30:-6])


    # num_anchor x 100
    inter_width = inter_x2 - inter_x1
    inter_height = inter_y2 - inter_y1
    width_zero_mask = tf.cast(tf.greater(inter_width, 0.), dtype=tf.float32)
    height_zero_mask = tf.cast(tf.greater(inter_height, 0.), dtype=tf.float32)
    inter_width = inter_width * width_zero_mask
    inter_height = inter_height * height_zero_mask
    intersection_area = inter_width * inter_height

    # num_anchor
    iou = intersection_area / (GT_area + anchor_area - intersection_area)
    max_iou_idx = tf.argmax(iou, axis=-1)
    max_iou = tf.reduce_max(iou, axis=-1)
    negative_mask = tf.less_equal(max_iou, negative_threshold)
    ignore_mask = tf.logical_and(tf.greater(max_iou, negative_threshold),
                                 tf.less_equal(max_iou, positive_threshold))
    positive_mask = tf.greater(iou,positive_threshold)
    positive_list.append(positive_mask)
    iou_list.append(iou)

    # ignore label index : 0
    # negative label index : 1
    # positive label index : 2 ~
    pos_label_idx = max_iou_idx * (1 - tf.cast(negative_mask, dtype=tf.int64)) + tf.cast(negative_mask, dtype=tf.int64)
    pos_label_idx = pos_label_idx * (1 - tf.cast(ignore_mask, dtype=tf.int64))

    matched_label = tf.gather(labels, pos_label_idx)
    matched_boxes = tf.gather(boxes, pos_label_idx)
    matched_label_list.append(matched_label)

    gt_cx = (matched_boxes[:, 0] + matched_boxes[:, 2]) / 2.
    gt_cy = (matched_boxes[:, 1] + matched_boxes[:, 3]) / 2.
    gt_w = matched_boxes[:, 2] - matched_boxes[:, 0]
    gt_h = matched_boxes[:, 3] - matched_boxes[:, 1]
<<<<<<< HEAD

    reshape_anchor = tf.reshape(scale_anchor, [-1, 4])

    anchor_cx = (reshape_anchor[:, 0] + reshape_anchor[:, 2]) / 2.
    anchor_cy = (reshape_anchor[:, 1] + reshape_anchor[:, 3]) / 2.
    anchor_w = reshape_anchor[:, 2] - reshape_anchor[:, 0]
    anchor_h = reshape_anchor[:, 3] - reshape_anchor[:, 1]

    gt_offset_cx = tf.reshape((gt_cx - anchor_cx) / (anchor_w + .1e-5), [-1, 1])
    gt_offset_cy = tf.reshape((gt_cy - anchor_cy) / (anchor_h + .1e-5), [-1, 1])
    gt_offset_w = tf.reshape(tf.log(gt_w / anchor_w + .1e-5) * (1. - tf.cast(tf.equal(gt_w, 0.), dtype=tf.float32)),
                             [-1, 1])
    gt_offset_h = tf.reshape(tf.log(gt_h / anchor_h + .1e-5) * (1. - tf.cast(tf.equal(gt_h, 0.), dtype=tf.float32)),
                             [-1, 1])

    gt_offset = tf.concat([gt_offset_cx, gt_offset_cy, gt_offset_w, gt_offset_h], axis=1)

    # gt_offset_list [(1083,4),(600,4),(150,4),(54,4),(24,4),(6,4)]
    gt_offset_list.append(gt_offset)


positive_list=tf.concat(positive_list,axis=0)
keep_one_pos_label=tf.reduce_sum(tf.cast(positive_list,dtype=tf.float32),axis=0)

sess=tf.Session()
res1=sess.run(keep_one_pos_label[2:4])
print(res1)
print(res1.shape)
print(res1)
for k in res2:
    print(k.shape)
print(res2[2])

=======

    reshape_anchor = tf.reshape(scale_anchor, [-1, 4])

    anchor_cx = (reshape_anchor[:, 0] + reshape_anchor[:, 2]) / 2.
    anchor_cy = (reshape_anchor[:, 1] + reshape_anchor[:, 3]) / 2.
    anchor_w = reshape_anchor[:, 2] - reshape_anchor[:, 0]
    anchor_h = reshape_anchor[:, 3] - reshape_anchor[:, 1]

    gt_offset_cx = tf.reshape((gt_cx - anchor_cx) / (anchor_w + .1e-5), [-1, 1])
    gt_offset_cy = tf.reshape((gt_cy - anchor_cy) / (anchor_h + .1e-5), [-1, 1])
    gt_offset_w = tf.reshape(tf.log(gt_w / anchor_w + .1e-5) * (1. - tf.cast(tf.equal(gt_w, 0.), dtype=tf.float32)),
                             [-1, 1])
    gt_offset_h = tf.reshape(tf.log(gt_h / anchor_h + .1e-5) * (1. - tf.cast(tf.equal(gt_h, 0.), dtype=tf.float32)),
                             [-1, 1])

    gt_offset = tf.concat([gt_offset_cx, gt_offset_cy, gt_offset_w, gt_offset_h], axis=1)

    # gt_offset_list [(1083,4),(600,4),(150,4),(54,4),(24,4),(6,4)]
    gt_offset_list.append(gt_offset)

'''
sess=tf.Session()
res1, res2=sess.run([gt_offset_list,matched_label_list])
for k in res2:
    print(k.shape)
print(res1[2])
print(res2[2])
print(res1[-2].shape)

>>>>>>> bcb9f3da41e2b91d352b32a676c7ced75ef2da5e
###test_code

import tensorflow as tf
import numpy as np

feature=tf.ones([2,3,3,960])
anchor_size=feature.get_shape().as_list()[1:3]
ratio_list=np.float32(np.array([1.,2.,1./2.,3.,1/3.,1]))

min_scale=0.2
max_scale=0.95
scale_range=np.array(range(6))+1
scale_list=(scale_range-1.)*((max_scale-min_scale)/(6-1.))+min_scale
scale_list=[.1]+list(scale_list)+[1.] #0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.95, 1.0
scale_idx=4
boxes=tf.convert_to_tensor([[0.0, 0.0, 0.5, 0.5],[0.2, 0.1, 0.25, 0.15]])
boxes = tf.concat([[[0,0,0,0],[0,0,0,0]],boxes],axis=0)
labels=tf.convert_to_tensor([1., 2.])
max_boxes=100
negative_threshold=0.3
positive_threshold=0.5

coordi_range=tf.cast(tf.range(0,anchor_size[0],1),dtype=tf.float32)

col_coordi=tf.reshape(coordi_range,[anchor_size[0],1])
row_coordi=tf.reshape(coordi_range,[1,anchor_size[0]])

col_coordi=tf.tile(col_coordi,[1,anchor_size[0]])
row_coordi=tf.tile(row_coordi,[anchor_size[0],1])

col_coordi=tf.reshape(col_coordi,[-1,anchor_size[0]])
row_coordi=tf.reshape(row_coordi,[-1,anchor_size[0]])

anchor_x1y1=tf.stack([row_coordi,col_coordi],axis=2)
anchor_x2y2=(anchor_x1y1+1)
anchor_x1y1=anchor_x1y1/anchor_size[0]
anchor_x2y2=anchor_x2y2/anchor_size[0]

scale_anchor=[]
# for ratio in ratio_list:
#     x_offset = 1. / 2. * (1. / anchor_size[0] - scale * tf.math.sqrt(ratio))
#     y_offset = 1. / 2. * (1. / anchor_size[0] - scale / tf.math.sqrt(ratio))
#     anchor_x1 = anchor_x1y1[:,:,0] + x_offset
#     anchor_y1 = anchor_x1y1[:,:,1] + y_offset
#     anchor_x2 = anchor_x2y2[:,:,0] - x_offset
#     anchor_y2 = anchor_x2y2[:,:,1] - y_offset
#
#     #num_scale x 19 x 19 x 4
#     anchor_x1y1x2y2=tf.stack([anchor_x1,anchor_y1,anchor_x2,anchor_y2],axis=2)
#     scale_anchor.append(anchor_x1y1x2y2)

for k, one_ratio in enumerate(ratio_list):


    if k==5:
        scale_ = tf.cast(tf.math.sqrt(scale_list[scale_idx]*scale_list[scale_idx+1]),dtype=tf.float32)
    else:
        scale_ = tf.cast(scale_list[scale_idx],dtype=tf.float32)

    x_offset = 1. / 2. * (1. / anchor_size[0] - scale_ * tf.math.sqrt(one_ratio))
    y_offset = 1. / 2. * (1. / anchor_size[0] - scale_ / tf.math.sqrt(one_ratio))
    anchor_x1y1 = tf.reshape(anchor_x1y1, [-1, 2])
    anchor_x2y2 = tf.reshape(anchor_x2y2, [-1, 2])
    anchor_x1 = anchor_x1y1[:, 0] + x_offset
    anchor_y1 = anchor_x1y1[:, 1] + y_offset
    anchor_x2 = anchor_x2y2[:, 0] - x_offset
    anchor_y2 = anchor_x2y2[:, 1] - y_offset

    #print(anchor_x1.shape)
    anchor_x1y1x2y2 = tf.stack([anchor_x1, anchor_y1, anchor_x2, anchor_y2], axis=1)
    #print(anchor_x1y1x2y2.shape)
    scale_anchor.append(anchor_x1y1x2y2)

# num_feature_block x num_ratio x 4
scale_anchor=tf.stack(scale_anchor,axis=1)
num_obj=tf.constant(boxes.get_shape().as_list()[0])

# 100 x 4
paded_boxes = tf.pad(boxes, tf.convert_to_tensor([[0,max_boxes-num_obj],[0,0]]), "CONSTANT")
paded_boxes = tf.reshape(paded_boxes,[100,4])

# [1083,4],[600,4],[150,4],[54,4],[24,4],[6,4]
scale_anchor=tf.reshape(scale_anchor,[-1,4])

# num_anchor x 100 x 4
tile_GT = tf.tile(tf.reshape(paded_boxes,[1,100,4]),[scale_anchor.shape[0],1,1])
tile_anchor = tf.tile(tf.expand_dims(scale_anchor,axis=1),[1,paded_boxes.shape[0],1])

inter_x1=tf.maximum(tile_GT[:,:,0],tile_anchor[:,:,0])
inter_y1=tf.maximum(tile_GT[:,:,1],tile_anchor[:,:,1])
inter_x2=tf.minimum(tile_GT[:,:,2],tile_anchor[:,:,2])
inter_y2=tf.minimum(tile_GT[:,:,3],tile_anchor[:,:,3])

# num_anchor x 100
GT_area=(tile_GT[:,:,2]-tile_GT[:,:,0])*(tile_GT[:,:,3]-tile_GT[:,:,1])
anchor_area=(tile_anchor[:,:,2]-tile_anchor[:,:,0])*(tile_anchor[:,:,3]-tile_anchor[:,:,1])

# num_anchor x 100
inter_width=inter_x2-inter_x1
inter_height=inter_y2-inter_y1
width_zero_mask=tf.cast(tf.greater(inter_width,0.),dtype=tf.float32)
height_zero_mask=tf.cast(tf.greater(inter_height,0.),dtype=tf.float32)
inter_width=inter_width*width_zero_mask
inter_height=inter_height*height_zero_mask
intersection_area=inter_width*inter_height

# num_anchor
iou = intersection_area/(GT_area+anchor_area-intersection_area)
max_iou_idx = tf.argmax(iou,axis=-1)
max_iou = tf.reduce_max(iou,axis=-1)
negative_mask = tf.less_equal(max_iou,negative_threshold)
ignore_mask = tf.logical_and(tf.greater(max_iou, negative_threshold),
                             tf.less_equal(max_iou, positive_threshold))
positive_mask = tf.greater(max_iou,positive_threshold)


# ignor label index : 0
# negative label index : 1
# positive label index : 2 ~
labels=tf.concat([[-1,0],labels],axis=0)

pos_label_idx = max_iou_idx*(1-tf.cast(negative_mask,dtype=tf.int64))+tf.cast(negative_mask,dtype=tf.int64)
pos_label_idx = pos_label_idx*(1-tf.cast(ignore_mask,dtype=tf.int64))
# pos_label_idx = (max_iou_idx+1)*(1-tf.cast(negative_mask,dtype=tf.int64))
# pos_label_idx = (pos_label_idx+1)*(1-tf.cast(ignore_mask,dtype=tf.int64))

matched_label=tf.gather(labels, pos_label_idx)
matched_boxes=tf.gather(boxes, pos_label_idx)


gt_cx = (matched_boxes[:,0]+matched_boxes[:,2])/2.
gt_cy = (matched_boxes[:,1]+matched_boxes[:,3])/2.
gt_w = matched_boxes[:,2] - matched_boxes[:,0]
gt_h = matched_boxes[:,3] - matched_boxes[:,1]

reshape_anchor=tf.reshape(scale_anchor,[-1,4])

anchor_cx = (reshape_anchor[:,0]+reshape_anchor[:,2])/2.
anchor_cy = (reshape_anchor[:,1]+reshape_anchor[:,3])/2.
anchor_w = reshape_anchor[:,2] - reshape_anchor[:,0]
anchor_h = reshape_anchor[:,3] - reshape_anchor[:,1]

gt_offset_cx = tf.reshape((gt_cx-anchor_cx)/(anchor_w+.1e-5),[-1,1])
gt_offset_cy = tf.reshape((gt_cy-anchor_cy)/(anchor_h+.1e-5),[-1,1])
gt_offset_w = tf.reshape(tf.log(gt_w/anchor_w+.1e-5)*(1.-tf.cast(tf.equal(gt_w,0.),dtype=tf.float32)),[-1,1])
gt_offset_h = tf.reshape(tf.log(gt_h/anchor_h+.1e-5)*(1.-tf.cast(tf.equal(gt_h,0.),dtype=tf.float32)),[-1,1])

gt_offset=tf.concat([gt_offset_cx,gt_offset_cy,gt_offset_w,gt_offset_h],axis=1)

sess=tf.Session()
rr=sess.run(positive_mask)
print(rr.shape)
print(rr)
print(rr)
# print(res[:,:])
print(res)
print(res.shape)


#ingnore=-1 negative=0
