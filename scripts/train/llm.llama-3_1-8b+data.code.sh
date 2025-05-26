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
HF_MODEL_NAME=meta-llama/Llama-3.1-8B
WANDB_PROJECT=CODE_SFT_LLAMA-3_1-8B
WANDB_PREFIX=llm.llama-3_1-8b

DATA_DIR=$WORK_DIR/data
DATA_TYPE=code
DATA_NAME=ise-uiuc/Magicoder-Evol-Instruct-110K

EPOCHS=4
BATCH_SIZE=192
MICRO_BATCH_SIZE=6
CUTOFF_LEN=4096

OPTIMIZER=lion

for LEARNING_RATE in 2e-4; 
do
    for LORA_RANK in 16 32 64 128;
    do
        LORA_ALPHA=$(($LORA_RANK * 2))
        RASA_K=$(($LORA_RANK / 8))
        # gralora_k = 2 LORA_RANK=16,32 else 4
        GRALORA_K=4
        if [ $LORA_RANK -eq 16 ] || [ $LORA_RANK -eq 32 ]; then
            GRALORA_K=2
        fi

        # hyprid_r = 8 if LORA_RANK=16 else 0
        HYBRID_R=0
        if [ $LORA_RANK -eq 16 ]; then
            HYBRID_R=8
        
        fi
        for METHOD in lora mora rasa gralora;
        do
            WANDB_RUN_NAME=${WANDB_PREFIX}+data.${DATA_TYPE}+peft.${METHOD}+r.${LORA_RANK}.a.${LORA_ALPHA}.+${OPTIMIZER}+lr_${LEARNING_RATE}+epochs_${EPOCHS}
            if [ ! -f $MODEL_DIR/$WANDB_RUN_NAME/adapter_model.safetensors ]; 
            then
                echo "Running $WANDB_RUN_NAME"
                CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nnodes 1 --nproc_per_node 4 $WORK_DIR/finetune_code.py \
                    --base_model $HF_MODEL_NAME \
                    --output_dir $MODEL_DIR/$WANDB_RUN_NAME \
                    --lora_r $LORA_RANK \
                    --lora_alpha $LORA_ALPHA \
                    --rasa_k $RASA_K \
                    --gralora_k $GRALORA_K \
                    --hybrid_r $HYBRID_R \
                    --chat_template_name alpaca-chat \
                    --data_name $DATA_NAME \
                    --data_dir $DATA_DIR \
                    --train_split train \
                    --batch_size $BATCH_SIZE \
                    --micro_batch_size $MICRO_BATCH_SIZE \
                    --cutoff_len $CUTOFF_LEN \
                    --num_train_epochs $EPOCHS \
                    --learning_rate $LEARNING_RATE \
                    --warmup_ratio 0.1 \
                    --save_strategy epoch \
                    --bf16 \
                    --gc \
                    --group_by_length \
                    --deepspeed $CONFIG_DIR/deepspeed/deepspeed_config_zero2_lion.json \
                    --use_${METHOD} \
                    --use_${OPTIMIZER} \
                    --use_wandb \
                    --wandb_project $WANDB_PROJECT \
                    --wandb_run_name $WANDB_RUN_NAME
            fi
        done
    done
done

