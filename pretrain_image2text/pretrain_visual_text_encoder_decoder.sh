CURRENT_DIR=`pwd`
TASK_NAME="image2text"
VISUAL_ENCODER='google/vit-base-patch16-224-in21k'
TEXT_DECODER='hfl/chinese-roberta-wwm-ext'
PL_MODEL_NAME="visual_text_encoder_decoder"
PL_DM_NAME="multi_modal_data"

echo -n 'model_name: "' > ${CURRENT_DIR}/config/model_config.yaml
echo -n ${PL_MODEL_NAME} >> ${CURRENT_DIR}/config/model_config.yaml
echo '"' >> ${CURRENT_DIR}/config/model_config.yaml
echo -n 'dm_name: "' >> ${CURRENT_DIR}/config/model_config.yaml
echo -n ${PL_DM_NAME} >> ${CURRENT_DIR}/config/model_config.yaml
echo '"' >> ${CURRENT_DIR}/config/model_config.yaml

CUDA_VISIBLE_DEVICES=1,2,4,6 python main.py \
  --project_name=${TASK_NAME} \
  --visual_encoder=${VISUAL_ENCODER} \
  --text_decoder=${TEXT_DECODER} \
  --visual_processor=${VISUAL_ENCODER} \
  --text_tokenizer=${TEXT_DECODER} \
  --max_length=64 \
  --train_path=/data/clean_raw_text/all_data_cleaned_only_wk_refine.json \
  --load_mlm_checkpoint \
  --save_top_k=3 \
  --num_warmup_steps=4000 \
  --gradient_clip_val=1 \
  --val_size=6000 \
  --learning_rate=1e-4 \
  --batch_size=64 \
  --max_epoch=15 \
  --min_epoch=5 \
  --patience=3 \
  --monitor=val_loss \
  --wandb \
  --gpus=4 \
  --multi_gpu_strategy=deepspeed_stage_2 \
  --precision=32 \
  --num_workers=32 \
  --stage=fit
