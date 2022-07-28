CURRENT_DIR=`pwd`
TASK_NAME="image2text_ft"
VISUAL_ENCODER='openai/clip-vit-base-patch32'
TEXT_DECODER='hfl/chinese-macbert-base'
PL_MODEL_NAME="visual_text_encoder_decoder"
PL_DM_NAME="multi_modal_data"

echo -n 'model_name: "' > ${CURRENT_DIR}/config/model_config.yaml
echo -n ${PL_MODEL_NAME} >> ${CURRENT_DIR}/config/model_config.yaml
echo '"' >> ${CURRENT_DIR}/config/model_config.yaml
echo -n 'dm_name: "' >> ${CURRENT_DIR}/config/model_config.yaml
echo -n ${PL_DM_NAME} >> ${CURRENT_DIR}/config/model_config.yaml
echo '"' >> ${CURRENT_DIR}/config/model_config.yaml

CUDA_VISIBLE_DEVICES=2,6,7 python main.py \
  --project_name=${TASK_NAME} \
  --visual_encoder=${VISUAL_ENCODER} \
  --text_decoder=${TEXT_DECODER} \
  --visual_processor=${VISUAL_ENCODER} \
  --text_tokenizer=${TEXT_DECODER} \
  --max_length=256 \
  --train_path=/data/clean_raw_text/non_12345_multimodal/multi_modal_data.json \
  --save_top_k=3 \
  --num_warmup_steps=500 \
  --val_size=600 \
  --learning_rate=1e-5 \
  --batch_size=8 \
  --max_epoch=5 \
  --min_epoch=0 \
  --patience=3 \
  --gradient_clip_val=1 \
  --monitor=val_loss \
  --wandb \
  --gpus=3 \
  --multi_gpu_strategy=deepspeed_stage_2 \
  --precision=16 \
  --num_workers=16 \
  --stage=fit \
  --load_model=/data/proj/pretrain_image2text/pl_log/image2text/1ic5nx29/checkpoints/last.ckpt