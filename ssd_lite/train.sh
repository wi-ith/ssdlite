CUDA_VISIBLE_DEVICES=0 \
python3 train.py \
  --ckpt_save_path=./ckpt/0321/ \
  --pretrained_ckpt_path=../pretrained/ssdlite_mobilenet/model.ckpt \
  --mode=train \
  --tfrecords_dir=./dir/to/tfrecords \
  --image_size=300 \
  --max_boxes=120 \
  --num_train=32147 \
  --num_validation=7550 \
  --num_preprocess_threads=4 \
  --batch_size=16 \
  --learning_rate=0.001 \
