import numpy as np

GT_loc=np.array([[0.2,0.2,0.47,0.47],[0.5,0.2,0.77,0.47],[0.47,0.47,0.8,0.85],[0.37,0.67,0.5,1.0]])
GT_cls=np.array([1,1,2,2])

loc_pred = [[0.21, 0.19, 0.51, 0.5],
       [0.18, 0.17, 0.47, 0.48],
       [0.22, 0.23, 0.53, 0.54],
       [0.51, 0.5, 0.8, 0.9],
       [0.46, 0.45, 0.76, 0.86],
       [0.56, 0.55, 0.8, 0.9],
       [0.51, 0.19, 0.81, 0.5],
       [0.48, 0.17, 0.77, 0.48],
       [0.52, 0.23, 0.83, 0.54],
       [0.41, 0.7, 0.5, 0.95],
       [0.36, 0.65, 0.46, 0.86],
       [0.46, 0.75, 0.5, 0.99]]
cls_pred = [[0.005, 0.98, 0.005, 0.005, 0.005],
       [0.1, 0.6, 0.1, 0.1, 0.1],
       [0.1, 0.4, 0.2, 0.2, 0.1],
       [0.05, 0.05, 0.8, 0.05, 0.05],
       [0.1, 0.1, 0.6, 0.1, 0.1],
       [0.1, 0.1, 0.7, 0.05, 0.05],
       [0.01, 0.96, 0.01, 0.01, 0.01],
       [0.1, 0.5, 0.2, 0.1, 0.1],
       [0.1, 0.7, 0.1, 0.05, 0.05],
       [0.05, 0.1, 0.7, 0.1, 0.05],
       [0.0, 0.05, 0.9, 0.025, 0.025],
       [0.1, 0.2, 0.5, 0.15, 0.05]]


def iou(boxes1,boxes2):
    if len(boxes1.shape)==1:
        boxes1=np.reshape(boxes1,[1,-1])

    if len(boxes2.shape)==1:
        boxes2=np.reshape(boxes2,[1,-1])

    tile_boxes1 = np.tile(np.expand_dims(boxes1, axis=0), [boxes2.shape[0], 1, 1])
    tile_boxes2 = np.tile(np.expand_dims(boxes2, axis=1), [1, boxes1.shape[0], 1])

    inter_y1 = np.maximum(tile_boxes1[:, :, 0], tile_boxes2[:, :, 0])
    inter_x1 = np.maximum(tile_boxes1[:, :, 1], tile_boxes2[:, :, 1])
    inter_y2 = np.minimum(tile_boxes1[:, :, 2], tile_boxes2[:, :, 2])
    inter_x2 = np.minimum(tile_boxes1[:, :, 3], tile_boxes2[:, :, 3])

    boxes1_area = (tile_boxes1[:, :, 2] - tile_boxes1[:, :, 0]) * (tile_boxes1[:, :, 3] - tile_boxes1[:, :, 1])
    boxes2_area = (tile_boxes2[:, :, 2] - tile_boxes2[:, :, 0]) * (tile_boxes2[:, :, 3] - tile_boxes2[:, :, 1])

    inter_width = inter_x2 - inter_x1
    inter_height = inter_y2 - inter_y1
    width_zero_mask = np.float32(inter_width > 0.)
    height_zero_mask = np.float32(inter_height > 0.)
    inter_width = inter_width * width_zero_mask
    inter_height = inter_height * height_zero_mask
    intersection_area = inter_width * inter_height

    iou = intersection_area / (boxes1_area + boxes2_area - intersection_area)

    return iou

max_output=1000
nms_threshold=0.6

cls_pred=np.array(cls_pred)
loc_pred=np.array(loc_pred)

cls_idx = np.argmax(cls_pred,axis=1)

num_classes=5
clsPred_by_class=[]
locPred_by_class=[]

for k in range(num_classes):
    k_cls_idx = np.where(cls_idx==k)
    # print(k_cls_idx)
    k_cls_pred = cls_pred[k_cls_idx]
    k_cls_loc = loc_pred[k_cls_idx]
    clsPred_by_class.append(k_cls_pred)
    locPred_by_class.append(k_cls_loc)
# print(clsPred_by_class)
# print(locPred_by_class)



for k_class_clsPred, k_class_locPred in zip(clsPred_by_class,locPred_by_class):

    score = np.max(k_class_clsPred, axis=1)
    sorted_idx = np.argsort(score)[::-1]
    sorted_loc = k_class_locPred[sorted_idx,:]
    sorted_score = score[sorted_idx]

    if len(sorted_loc) > max_output:
        sorted_loc = sorted_loc[:max_output]
        sorted_score = sorted_score[:max_output]

    valid_mask = np.ones(sorted_score.shape)
    # print(valid_mask)
    for k in range(len(valid_mask)):
        if valid_mask[k]:
            # print(sorted_loc[k])
            # print(sorted_loc[k+1:])
            iou_=iou(sorted_loc[k],sorted_loc[k+1:])
            iou_=np.squeeze(iou_)
            valid_mask[k + 1:] = np.logical_and((valid_mask[k + 1:]==1),(iou_ < nms_threshold))
            # print(iou_)
            print(valid_mask)
    print(valid_mask)
#nms
    # print(valid_mask)







