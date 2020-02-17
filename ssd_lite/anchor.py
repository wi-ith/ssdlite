import tensorflow as tf
#depth 960, 1280, 512, 256, 256, 128
feature_maps=[tf.ones([2,19,19,960]),
              tf.ones([2,10,10,1280]),
              tf.ones([2,5,5,512]),
              tf.ones([2,3,3,256]),
              tf.ones([2,2,2,256]),
              tf.ones([2,1,1,128])]
ratio_list=[(2,1,1/2)]+[(3,2,1,1/2,1/3)]*5
min_scale=tf.constant(0.2)
max_scale=tf.constant(0.95)
scale_range=tf.cast(tf.range(1,7,1),dtype=tf.float32)
scale_list=(scale_range-1)*((max_scale-min_scale)/5.)+min_scale
scale_list=tf.concat([[0.1],scale_list],axis=0)


for feature,ratio,scale in zip(feature_maps,ratio_list,scale_list[:-1]):
    feature.get_shape().as_list()[1:3]

import tensorflow as tf
import numpy as np
feature=tf.ones([2,19,19,960])
anchor_size=feature.get_shape().as_list()[1:3]
ratio=(2,1,1/2)
scale=0.1

coordi_range=tf.range(0,anchor_size[0],1)

col_coordi=tf.reshape(coordi_range,[anchor_size[0],1])
row_coordi=tf.reshape(coordi_range,[1,anchor_size[0]])

col_coordi=tf.tile(col_coordi,[1,anchor_size[0]])
row_coordi=tf.tile(row_coordi,[anchor_size[0],1])

col_coordi=tf.reshape(col_coordi,[-1,anchor_size[0]])
row_coordi=tf.reshape(row_coordi,[-1,anchor_size[0]])

anchor_x1y1=tf.stack([col_coordi,row_coordi],axis=2)
anchor_x2y2=anchor_x1y1+1


sess=tf.Session()
sess.run(anchor_x2y2)




tmp_col=np.reshape(tmp,[6,1])
tmp_row=np.reshape(tmp,[1,6])
col_=tf.tile(tmp_col,[1,6])
col_=tf.reshape(col_,[-1,6])
row_=tf.tile(tmp_col,[6,1])
row_=tf.reshape(row_,[-1,6])
anchor=tf.stack([col_,row_],axis=2)

sess=tf.Session()
sess.run(coordi_range)

a,b,c=sess.run([col_,row_,anchor])
print(a)
print(b)
print(c)