CURRENT_DIR=`pwd`
TASK_NAME="district_dist_classify"
BERT_NAME='/data/proj/dist_proj/pl_log/pretrain_pure_text_mlm/1oyphuu9/model'
TOKENIZER_NAME='/data/proj/dist_proj/pl_log/pretrain_pure_text_mlm/1oyphuu9/model'
PL_MODEL_NAME="dist_classify"
PL_DM_NAME="mix_modal_data"

echo -n 'model_name: "' > ${CURRENT_DIR}/config/temp_model_config.yaml
echo -n ${PL_MODEL_NAME} >> ${CURRENT_DIR}/config/temp_model_config.yaml
echo '"' >> ${CURRENT_DIR}/config/temp_model_config.yaml
echo -n 'dm_name: "' >> ${CURRENT_DIR}/config/temp_model_config.yaml
echo -n ${PL_DM_NAME} >> ${CURRENT_DIR}/config/temp_model_config.yaml
echo '"' >> ${CURRENT_DIR}/config/temp_model_config.yaml

CUDA_VISIBLE_DEVICES=6,7 python main.py \
  --project_name=${TASK_NAME} \
  --bert_name=${BERT_NAME} \
  --tokenizer_name=${TOKENIZER_NAME} \
  --max_length=200 \
  --train_path=/data/clean_raw_text/district_labeled_data.json \
  --eda \
  --focal_loss \
  --num_classes=62 \
  --save_top_k=3 \
  --num_warmup_steps=2000 \
  --val_ratio=0.1 \
  --learning_rate=1e-5 \
  --batch_size_per_gpu=32 \
  --max_epoch=8 \
  --min_epoch=0 \
  --patience=3 \
  --gradient_clip_val=0 \
  --monitor=val_accuracy_top_1 \
  --wandb \
  --gpus=2 \
  --multi_gpu_strategy=deepspeed_stage_2 \
  --precision=32 \
  --num_workers=8 \
  --stage=fit