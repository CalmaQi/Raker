export DATASET="nell"
# export SUFFIX="_hop3_1000_neg10_max_inductive_test2"
# export SUFFIX="_hop3_1000_neg10_max_inductive"
# export SUFFIX="_hop3_2000_neg10_max_inductive_test"
export SUFFIX="_hop3_full_neg10_max_inductive_test"
# export SUFFIX="_hop3_2000bertrl"
# export SUFFIX="_hop3_full_neg10_max_inductive"
# export SUFFIX="_krst_trainnei"
# export SUFFIX="_krst_1000_trainnei_test"
# export SUFFIX="_krst_trainnei_h5p6c6"

# export SUFFIX="_krst_nonei"
CUDA_VISIBLE_DEVICES=6,7 python run_bertrl.py \
  --model_name_or_path '../dataroot/models/bert-base-uncased' \
  --task_name MRPC \
  --do_train \
  --data_dir ./bertrl_data/${DATASET}${SUFFIX} \
  --learning_rate 5e-5 \
  --max_seq_length 200 \
  --per_device_train_batch_size 64 \
  --num_train_epochs 3 \
  --output_dir output_${DATASET}${SUFFIX} \
  --logging_steps 100 \
  --overwrite_output_dir \
  --save_total_limit 1 \
  --do_predict \
  --save_steps 1000 \
  --per_device_eval_batch_size 600 \
  --overwrite_cache \
  --do_eval \
  --evaluation_strategy epoch \

  #protobuf 3.19.1
  # --evaluation_strategy no \
#   --warmup_steps 20000 \
#   --save_steps 20000


