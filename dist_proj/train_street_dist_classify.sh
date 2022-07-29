CURRENT_DIR=`pwd`
TASK_NAME="street_dist_classify"
BERT_NAME='/data/proj/dist_proj/pl_log/pretrain_pure_text_mlm/1oyphuu9/model'
TOKENIZER_NAME='/data/proj/dist_proj/pl_log/pretrain_pure_text_mlm/1oyphuu9/model'
PL_MODEL_NAME="dist_classify"
PL_DM_NAME="mix_modal_data"

echo -n 'model_name: "' > ${CURRENT_DIR}/config/temp_model_config.yaml
echo -n ${PL_MODEL_NAME} >> ${CURRENT_DIR}/config/model_temp_model_configconfig.yaml
echo '"' >> ${CURRENT_DIR}/config/temp_model_config.yaml
echo -n 'dm_name: "' >> ${CURRENT_DIR}/config/temp_model_config.yaml
echo -n ${PL_DM_NAME} >> ${CURRENT_DIR}/config/temp_model_config.yaml
echo '"' >> ${CURRENT_DIR}/config/temp_model_config.yaml

CUDA_VISIBLE_DEVICES=2,3,5,6 python main.py \
  --project_name=${TASK_NAME} \
  --bert_name=${BERT_NAME} \
  --tokenizer_name=${TOKENIZER_NAME} \
  --max_length=200 \
  --train_path=/data/clean_raw_text/street_labeled_data.json \
  --focal_loss \
  --num_classes=58 \
  --save_top_k=3 \
  --num_warmup_steps=500 \
  --val_ratio=0.1 \
  --learning_rate=1e-5 \
  --batch_size_per_gpu=32 \
  --max_epoch=10 \
  --min_epoch=0 \
  --patience=3 \
  --gradient_clip_val=0 \
  --monitor=val_accuracy_top_1 \
  --wandb \
  --gpus=4 \
  --multi_gpu_strategy=deepspeed_stage_2 \
  --precision=32 \
  --num_workers=32 \
  --stage=fit