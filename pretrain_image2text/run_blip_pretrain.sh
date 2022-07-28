CURRENT_DIR=`pwd`
TASK_NAME="blip_pretrain"
VISUAL_ENCODER='google/vit-base-patch16-224-in21k'
TEXT_DECODER='hfl/chinese-macbert-base'
PL_MODEL_NAME="blip_module"
PL_DM_NAME="blip_data"


echo -n 'model_name: "' > ${CURRENT_DIR}/config/model_config.yaml
echo -n ${PL_MODEL_NAME} >> ${CURRENT_DIR}/config/model_config.yaml
echo '"' >> ${CURRENT_DIR}/config/model_config.yaml
echo -n 'dm_name: "' >> ${CURRENT_DIR}/config/model_config.yaml
echo -n ${PL_DM_NAME} >> ${CURRENT_DIR}/config/model_config.yaml
echo '"' >> ${CURRENT_DIR}/config/model_config.yaml

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py \
  --project_name=${TASK_NAME} \
  --train_path=/data/clean_raw_text/data/wukong/wukong_zw_with_img_path_clean_valid_img_refine.json \
  --text_bert=${TEXT_DECODER} \
  --visual_bert=${VISUAL_ENCODER} \
  --wandb \
  --save_top_k=5 \
  --num_warmup_steps=4000 \
  --val_size=6000 \
  --learning_rate=5e-5 \
  --batch_size=8 \
  --queue_size=57600 \
  --max_epoch=10 \
  --gradient_clip_val=0 \
  --monitor=val_loss \
  --gpus=8 \
  --multi_gpu_strategy=ddp \
  --precision=16 \
  --num_workers=32 \
  --stage=fit
