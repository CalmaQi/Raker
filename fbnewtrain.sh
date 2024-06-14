export DATASET="fb237"
export SUFFIX="_hop3_full_neg10_max_inductive_test"
CUDA_VISIBLE_DEVICES=9 python predict.py \
  --model_name_or_path bert-base-uncased \
  --task_name MRPC \
  --do_train \
  --data_dir ./bertrl_data/${DATASET}${SUFFIX} \
  --learning_rate 5e-5 \
  --max_seq_length 128 \
  --per_device_train_batch_size 64 \
  --num_train_epochs 2.0 \
  --output_dir output_${DATASET}${SUFFIX} \
  --logging_steps 100 \
  --overwrite_output_dir \
  --save_total_limit 1 \
  --do_predict \
  --do_eval \
  --save_steps 1000 \
  --per_device_eval_batch_size 600 \
  --overwrite_cache \
  --evaluation_strategy no \
  # --eval_steps 1000 \
  # --evaluation_strategy epoch \
  # --evaluation_strategy epoch \

  #protobuf 3.19.1

#   --warmup_steps 20000 \
#   --save_steps 20000


