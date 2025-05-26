set -e
set -u

export NCCL_DEBUG=WARN
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_GID_INDEX=3
export NCCL_IB_SL=3
export NCCL_NET_GDR_READ=1
export MASTER_ADDR="${CHIEF_IP:=localhost}"
export MASTER_PORT="${MASTER_PORT:=29501}"
export TOKENIZERS_PARALLELISM=false

WORK_DIR=/workspace/GraLoRA
CONFIG_DIR=$WORK_DIR/scripts/config

MODEL_DIR=$WORK_DIR/models
HF_MODEL_NAME=Qwen/Qwen2.5-7B
WANDB_PROJECT=COMMONSENCE_SFT_QWEN-2_5-7B
WANDB_PREFIX=llm.qwen-2_5-7b

DATA_DIR=$WORK_DIR/data
DATA_TYPE=commonsense 
DATA_NAME=zwhe99/commonsense_170k

EPOCHS=2
BATCH_SIZE=192
MICRO_BATCH_SIZE=96
CUTOFF_LEN=256

OPTIMIZER=lion

for LEARNING_RATE in 4e-4; 
do
    for LORA_RANK in 64;
    do
        LORA_ALPHA=$(($LORA_RANK * 2))
        RASA_K=$(($LORA_RANK / 8))
        GRALORA_K=4
        for METHOD in lora mora rasa gralora;
        do
            WANDB_RUN_NAME=${WANDB_PREFIX}+data.${DATA_TYPE}+peft.${METHOD}+r.${LORA_RANK}.a.${LORA_ALPHA}.+${OPTIMIZER}+lr_${LEARNING_RATE}+epochs_${EPOCHS}
            if [ ! -f $MODEL_DIR/$WANDB_RUN_NAME/adapter_model.safetensors ]; 
            then
                echo "Training $WANDB_RUN_NAME"
                CUDA_VISIBLE_DEVICES=0,1 torchrun --nnodes 1 --nproc_per_node 2 $WORK_DIR/finetune_commonsense.py \
                    --base_model $HF_MODEL_NAME \
                    --output_dir $MODEL_DIR/$WANDB_RUN_NAME \
                    --lora_r $LORA_RANK \
                    --lora_alpha $LORA_ALPHA \
                    --rasa_k $RASA_K \
                    --gralora_k $GRALORA_K \
                    --data_name $DATA_NAME \
                    --data_dir $DATA_DIR \
                    --batch_size $BATCH_SIZE \
                    --micro_batch_size $MICRO_BATCH_SIZE \
                    --num_train_epochs $EPOCHS \
                    --learning_rate $LEARNING_RATE \
                    --warmup_ratio 0.1 \
                    --save_strategy epoch \
                    --bf16 \
                    --gc \
                    --cutoff_len $CUTOFF_LEN \
                    --deepspeed $CONFIG_DIR/deepspeed/deepspeed_config_zero2_lion.json \
                    --use_${METHOD} \
                    --use_${OPTIMIZER} \
                    --lr_scheduler_type cosine \
                    --use_wandb \
                    --wandb_project $WANDB_PROJECT \
                    --wandb_run_name $WANDB_RUN_NAME
            fi
        done
    done
done