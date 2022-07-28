CURRENT_DIR=`pwd`
export BERT_BASE_DIR=$CURRENT_DIR/../pretrained_models
export GLUE_DIR=$CURRENT_DIR/data/raw/CLUEdatasets
TASK_NAME="cluener"
MODEL_NAME="chinese-roberta-wwm-ext"
LOAD_VERSION=1
CUDA_VISIBLE_DEVICES=0,1

PL_MODEL_NAME="cluner_bert_crf"
PL_DM_NAME="cluner_data"

echo -n 'model_name: "' > ${CURRENT_DIR}/config/model_config.yaml
echo -n ${PL_MODEL_NAME} >> ${CURRENT_DIR}/config/model_config.yaml
echo '"' >> ${CURRENT_DIR}/config/model_config.yaml
echo -n 'dm_name: "' >> ${CURRENT_DIR}/config/model_config.yaml
echo -n ${PL_DM_NAME} >> ${CURRENT_DIR}/config/model_config.yaml
echo '"' >> ${CURRENT_DIR}/config/model_config.yaml

python main.py \
  --project_name=${TASK_NAME}_${MODEL_NAME} \
  --model_name=$BERT_BASE_DIR/${MODEL_NAME} \
  --tokenizer_name=$BERT_BASE_DIR/${MODEL_NAME} \
  --data_path=$GLUE_DIR/${TASK_NAME} \
  --load_v_num=${LOAD_VERSION} \
  --stage=test \
  --load_best \
  --pred_output_dir=$CURRENT_DIR/outputs \
  --batch_size=1 \
  --gpus=0,1 \
  --multi_gpu_strategy=ddp \
  --num_workers=4