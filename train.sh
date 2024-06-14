export DATASET="fb237"
# export DATASET="testgraph"
# export SUFFIX="_hop3_1000_neg10_max_inductive"
# export SUFFIX="_hop3_full_neg10_max_inductive_test"
# export SUFFIX="_hop3_full0.3ordernoshu/"
# export SUFFIX="_hop3_rel130"
export SUFFIX="_hop3_full0.1"
# export SUFFIX="_hop3_full0.1:2"
# export SUFFIX="_hop3_1000_trainnei"
# export SUFFIX="_hop3_10006"
# export SUFFIX="_hop3_fullcs1"
# export SUFFIX="_hop3_full_neg10_max_inductive_test_rule"
# export SUFFIX="_hop3_fullxiaorong_des"
# export SUFFIX="_hop3_fullxiaorong_nei"
# export SUFFIX="_hop3_rel1000.6"
# export SUFFIX="_hop3_full_xiaorong_context"
# export SUFFIX="_hop3_full_neg10_max_inductive"
CUDA_VISIBLE_DEVICES=3,4 python predict.py \
  --model_name_or_path  '../dataroot/models/bert-base-uncased'  \
  --task_name MRPC \
  --do_train \
  --data_dir ./bertrl_data/${DATASET}${SUFFIX} \
  --learning_rate 5e-5 \
  --max_seq_length 200 \
  --per_device_train_batch_size 64 \
  --num_train_epochs 3.0 \
  --output_dir output_${DATASET}${SUFFIX} \
  --logging_steps 100 \
  --overwrite_output_dir \
  --do_predict \
  --do_eval \
  --per_device_eval_batch_size 600 \
  --overwrite_cache \
  --evaluation_strategy epoch \
  --save_total_limit 1 \
  --save_strategy epoch
  # --eval_steps 1000 \
  # --evaluation_strategy epoch \
  # --evaluation_strategy epoch \
  # --evaluation_strategy steps \
  # --save_steps 1000 \
  # albert/albert-base-v2
  # --model_name_or_path  'microsoft/deberta-v3-base'  \

  # --model_name_or_path  '../dataroot/models/bert-base-uncased'  \
# FacebookAI/roberta-base
# studio-ousia/luke-base
# MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli
  #protobuf 3.19.1

#   --warmup_steps 20000 \
#   --save_steps 20000


