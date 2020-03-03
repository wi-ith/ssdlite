CUDA_VISIBLE_DEVICES=0 \
python3 train.py \
  --ckpt_save_path=./ckpt/0301/ \
  --pretrained_ckpt_path=../pretrained/mobilenet_backbone/model.ckpt \
  --mode=train \
  --tfrecords_dir=/home/kdg/tfrecords/google_mscoco/32k \
  --image_size=512 \
  --max_boxes=100 \
  --num_train=32147 \
  --num_preprocess_threads=4 \
