CURRENT_DIR=`pwd`
TASK_NAME="pretrain_pure_text_mlm"
BERT_NAME='hfl/chinese-roberta-wwm-ext'
PL_MODEL_NAME="pure_text_mlm_pretrain"
PL_DM_NAME="mix_modal_data"

echo -n 'model_name: "' > ${CURRENT_DIR}/config/temp_model_config.yaml
echo -n ${PL_MODEL_NAME} >> ${CURRENT_DIR}/config/temp_model_config.yaml
echo '"' >> ${CURRENT_DIR}/config/temp_model_config.yaml
echo -n 'dm_name: "' >> ${CURRENT_DIR}/config/temp_model_config.yaml
echo -n ${PL_DM_NAME} >> ${CURRENT_DIR}/config/temp_model_config.yaml
echo '"' >> ${CURRENT_DIR}/config/temp_model_config.yaml

CUDA_VISIBLE_DEVICES=1,2,3,4 python main.py \
  --project_name=${TASK_NAME} \
  --bert_name=${BERT_NAME} \
  --max_length=200 \
  --mlm \
  --whole_word_mask \
  --train_path=/data/clean_raw_text/all_data_cleaned_with_cn_ref.json \
  --save_top_k=3 \
  --num_warmup_steps=4000 \
  --val_ratio=0.1 \
  --learning_rate=1e-5 \
  --batch_size_per_gpu=64 \
  --max_epoch=30 \
  --min_epoch=10 \
  --patience=3 \
  --gradient_clip_val=0 \
  --monitor=val_loss \
  --wandb \
  --gpus=4 \
  --multi_gpu_strategy=deepspeed_stage_2 \
  --precision=16 \
  --num_workers=32 \
  --stage=fit