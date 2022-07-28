CURRENT_DIR=`pwd`
TASK_NAME="mixmodal_mlm_pretrain"
VISUAL_ENCODER='google/vit-base-patch16-224-in21k'
TEXT_DECODER='hfl/chinese-roberta-wwm-ext'
PL_MODEL_NAME="dual_stream_model"
PL_DM_NAME="mix_modal_data_for_mlm"

echo -n 'model_name: "' > ${CURRENT_DIR}/config/model_config.yaml
echo -n ${PL_MODEL_NAME} >> ${CURRENT_DIR}/config/model_config.yaml
echo '"' >> ${CURRENT_DIR}/config/model_config.yaml
echo -n 'dm_name: "' >> ${CURRENT_DIR}/config/model_config.yaml
echo -n ${PL_DM_NAME} >> ${CURRENT_DIR}/config/model_config.yaml
echo '"' >> ${CURRENT_DIR}/config/model_config.yaml

CUDA_VISIBLE_DEVICES=1,2,3,4,5,6 python main.py \
  --project_name=${TASK_NAME} \
  --visual_encoder=${VISUAL_ENCODER} \
  --text_decoder=${TEXT_DECODER} \
  --max_length=128 \
  --train_path=/data/clean_raw_text/all_data_cleaned_wk_refine.json \
  --save_top_k=5 \
  --num_warmup_steps=4000 \
  --val_size=90000 \
  --freeze_encoder \
  --find_unused_parameters \
  --constrative_learning \
  --queue_size=57600 \
  --learning_rate=5e-5 \
  --batch_size_per_gpu=16 \
  --max_epoch=20 \
  --min_epoch=5 \
  --patience=3 \
  --gradient_clip_val=0 \
  --monitor=val_loss \
  --wandb \
  --gpus=6 \
  --multi_gpu_strategy=deepspeed_stage_2 \
  --precision=16 \
  --num_workers=32 \
  --stage=fit