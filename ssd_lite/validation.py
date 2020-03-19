import numpy as np
import tensorflow as tf
import flags
FLAGS = tf.app.flags.FLAGS
#
# GT_loc=np.array([[0.2,0.2,0.47,0.47],[0.5,0.2,0.77,0.47],[0.47,0.47,0.8,0.85],[0.37,0.67,0.5,1.0]])
# GT_cls=np.array([1,1,2,2])
#
# loc_pred = [[0.21, 0.19, 0.51, 0.5],
#        [0.18, 0.17, 0.47, 0.48],
#        [0.22, 0.23, 0.53, 0.54],
#        [0.51, 0.5, 0.8, 0.9],
#        [0.46, 0.45, 0.76, 0.86],
#        [0.56, 0.55, 0.8, 0.9],
#        [0.51, 0.19, 0.81, 0.5],
#        [0.48, 0.17, 0.77, 0.48],
#        [0.52, 0.23, 0.83, 0.54],
#        [0.41, 0.7, 0.5, 0.95],
#        [0.36, 0.65, 0.46, 0.86],
#        [0.46, 0.75, 0.5, 0.99],
#        [0,0,0,0],
#        [0,0,0,0]]
# cls_pred = [[0.005, 0.98, 0.005, 0.005, 0.005],
#        [0.1, 0.6, 0.1, 0.1, 0.1],
#        [0.1, 0.4, 0.2, 0.2, 0.1],
#        [0.05, 0.05, 0.8, 0.05, 0.05],
#        [0.1, 0.1, 0.6, 0.1, 0.1],
#        [0.1, 0.1, 0.7, 0.05, 0.05],
#        [0.01, 0.96, 0.01, 0.01, 0.01],
#        [0.1, 0.5, 0.2, 0.1, 0.1],
#        [0.1, 0.7, 0.1, 0.05, 0.05],
#        [0.05, 0.1, 0.7, 0.1, 0.05],
#        [0.0, 0.05, 0.9, 0.025, 0.025],
#        [0.1, 0.2, 0.5, 0.15, 0.05],
#        [0.90,0.025,0.025,0.025,0.025],
#        [0.90,0.025,0.025,0.025,0.025]]

def one_image_validation(GT_loc,
                         GT_cls,
                         loc_pred,
                         cls_pred,
                         num_objects):
    GT_loc=np.squeeze(GT_loc)
    GT_cls = np.squeeze(GT_cls)
    loc_pred = np.squeeze(loc_pred)
    cls_pred = np.squeeze(cls_pred)
    num_objects = np.squeeze(num_objects)

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
    nms_threshold=FLAGS.val_nms_threshold

    cls_pred=np.array(cls_pred)
    loc_pred=np.array(loc_pred)

    cls_idx = np.argmax(cls_pred,axis=1)


    clsPred_by_class=[]
    locPred_by_class=[]

    for k in range(FLAGS.num_classes):
        k_cls_idx = np.where(cls_idx==k)
        k_cls_pred = cls_pred[k_cls_idx]
        k_cls_loc = loc_pred[k_cls_idx]
        clsPred_by_class.append(k_cls_pred)
        locPred_by_class.append(k_cls_loc)

    k_class_nms_loc = []
    k_class_nms_score = []
    for k_class_clsPred, k_class_locPred in zip(clsPred_by_class,locPred_by_class):

        score = np.max(k_class_clsPred, axis=1)
        sorted_idx = np.argsort(score)[::-1]
        sorted_loc = k_class_locPred[sorted_idx,:]
        sorted_score = score[sorted_idx]

        if len(sorted_loc) > max_output:
            sorted_loc = sorted_loc[:max_output]
            sorted_score = sorted_score[:max_output]

        valid_mask = np.ones(sorted_score.shape)

        for k in range(len(valid_mask)):
            if valid_mask[k]:
                iou_=iou(sorted_loc[k],sorted_loc[k+1:])
                iou_=np.squeeze(iou_)
                valid_mask[k + 1:] = np.logical_and((valid_mask[k + 1:]==1),(iou_ < nms_threshold))

        valid_idx = np.where(valid_mask == 1)
        k_class_nms_loc.append(sorted_loc[valid_idx])
        k_class_nms_score.append(sorted_score[valid_idx])

    #non maximum suppresion

    TF_array_by_class=[]
    TF_score_by_class=[]
    num_GT_by_class=[]
    GT_loc = GT_loc[:num_objects,:]
    GT_cls = GT_cls[:num_objects]
    for k in range(FLAGS.num_classes):
        k_cls_index = np.where(GT_cls == k)
        num_GT_by_class.append(np.sum(np.int32(GT_cls == k)))

        if np.sum(np.int32(GT_cls == k))==0:
            TF_array_by_class.append(np.array([]))
            TF_score_by_class.append(np.array([]))
            continue
        k_cls_GT_loc = GT_loc[k_cls_index]

        TF_array = np.zeros([len(k_class_nms_score[k])])
        found_gt = np.zeros([np.sum(np.int32(GT_cls == k))])

        GT_pred_iou = iou(k_cls_GT_loc,k_class_nms_loc[k])

        maxiou_box_id=np.argmax(GT_pred_iou,axis=1)

        for x, TF in enumerate(TF_array):
            if found_gt[maxiou_box_id[x]]==0:
                TF_array[x]=1
                found_gt[maxiou_box_id[x]]=1
        TF_array_by_class.append(TF_array)
        TF_score_by_class.append(k_class_nms_score[k])

    return TF_array_by_class, TF_score_by_class, num_GT_by_class


# import numpy as np
# num_classes=5
# TF_array_1 = [[],[],[1,1,0,1,1,0],[1,1,0,1],[]]
# TF_score_1 = [[],[],[0.76,0.55,0.49,0.98,0.69,0.72],[0.1,0.2,0.3,0.92],[]]
# TF_array_2 = [[],[0,1,1,1,0],[0,1,1,0,1],[1,1],[1]]
# TF_score_2 = [[],[0.56,0.66,0.79,0.28,0.91],[0.19,0.33,0.29,0.61,0.55],[0.57,0.42],[0.22]]
# num_GT_1 = [0,0,4,4,0]
# num_GT_2 = [0,5,3,3,1]
# entire_TF=[]
# entire_score=[]
# entire_numGT=[]

def cover_up(list, idx, value):
    if list[idx] < value:
        list[idx]=value
        cover_up(list, idx-1,value)
    else:
        return list


# for k in range(2):
#     if k ==0:
#         TF_array = TF_array_1
#         TF_score = TF_score_1
#         num_GT = num_GT_1
#     else:
#         TF_array = TF_array_2
#         TF_score = TF_score_2
#         num_GT = num_GT_2
#
#     if len(entire_TF) == 0:
#         entire_TF = TF_array
#         entire_score = TF_score
#         entire_numGT = num_GT
#     else:
#         for k_cls in range(num_classes):
#             entire_TF[k_cls].extend(TF_array[k_cls])
#             entire_score[k_cls].extend(TF_score[k_cls])
#             entire_numGT[k_cls]+=num_GT[k_cls]

def compute_AP(entire_score,entire_TF,entire_numGT):

    sorted_entire_TF=[]
    for x_cls in range(FLAGS.num_classes):
        sorted_score_idx=np.argsort(entire_score[x_cls])[::-1]
        x_cls_TF=np.copy(entire_TF[x_cls])
        sorted_entire_TF.append(x_cls_TF[sorted_score_idx])

    entire_presicion = []
    entire_recall = []
    for num_GT, k_cls_TF in zip(entire_numGT, sorted_entire_TF):
        ac_sum = 0
        k_cls_precision = []
        k_cls_recall = []
        for num_pred, TF in enumerate(k_cls_TF):
            ac_sum += TF
            precision = ac_sum / (num_pred + 1)
            recall = ac_sum / (num_GT)
            k_cls_precision.append(precision)
            k_cls_recall.append(recall)
            if num_pred > 0:
                cover_up(k_cls_precision, num_pred - 1, precision)
        entire_presicion.append(k_cls_precision)
        entire_recall.append(k_cls_recall)

    entire_AP = []
    entire_AP_sum = []
    for precision_k, recall_k in zip(entire_presicion, entire_recall):
        k_cls_AP_list=[]
        for i_, (precision_, recall_) in enumerate(zip(precision_k,recall_k)):
            if i_ == 0:
                AP_ = precision_*recall_
            else:
                AP_ = precision_*(recall_-recall_k[i_-1])
                k_cls_AP_list.append(AP_)

        k_cls_AP=np.sum(np.array(k_cls_AP_list))
        entire_AP.append(k_cls_AP_list)
        entire_AP_sum.append(k_cls_AP)


    return entire_AP_sum






