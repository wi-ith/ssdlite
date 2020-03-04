CUDA_VISIBLE_DEVICES=0 \
python3 train.py \
  --ckpt_save_path=./ckpt/0301/ \
  --pretrained_ckpt_path=../pretrained/ssdlite_mobilenet_v2_coco_2018_05_09/model.ckpt \
  --mode=train \
  --tfrecords_dir=/home/kdg/tfrecords/google_mscoco/32k \
  --image_size=300 \
  --max_boxes=120 \
  --num_train=32147 \
  --num_preprocess_threads=4 \
  --batch_size=4 \
