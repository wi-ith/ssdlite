CUDA_VISIBLE_DEVICES=0 \
python3 train.py \
  --ckpt_save_path=./ckpt \
  --pretrained_ckpt_path=../pretrained/mobilenet_backbone/model.ckpt \
  --mode=train \
  --tfrecords_dir=path/to/tfrecords \
  --image_size=300 \
  --max_boxes=100 \
  --num_train=32147 \
  --num_validation=7990 \
  --num_preprocess_threads=4 \
  --batch_size=16 \
  --learning_rate=0.001 \
