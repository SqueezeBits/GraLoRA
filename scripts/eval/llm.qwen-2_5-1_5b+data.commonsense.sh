WORK_DIR=/workspace/GraLoRA
CONFIG_DIR=$WORK_DIR/scripts/config

MODEL_DIR=$WORK_DIR/models
HF_MODEL_NAME=Qwen/Qwen2.5-1.5B
WANDB_PROJECT=COMMONSENCE_SFT_QWEN-2_5-1_5B
WANDB_PREFIX=llm.qwen-2_5-1_5b

DATA_DIR=$WORK_DIR/data
DATA_TYPE=commonsense 
DATA_NAME=zwhe99/commonsense_170k

EPOCHS=2
OPTIMIZER=lion

for LEARNING_RATE in 2e-4; 
do
    for LORA_RANK in 64;
    do
        LORA_ALPHA=$(($LORA_RANK * 2))
        RASA_K=$(($LORA_RANK / 8))
        GRALORA_K=4
        for METHOD in lora mora rasa gralora;
        do
            WANDB_RUN_NAME=${WANDB_PREFIX}+data.${DATA_TYPE}+peft.${METHOD}+r.${LORA_RANK}.a.${LORA_ALPHA}.+${OPTIMIZER}+lr_${LEARNING_RATE}+epochs_${EPOCHS}
            if [ -f $MODEL_DIR/$WANDB_RUN_NAME/adapter_model.safetensors ]; 
            then
                for DATASET in boolq piqa social_i_qa hellaswag winogrande ARC-Challenge ARC-Easy openbookqa;
                do
                    if [ ! -f $MODEL_DIR/$WANDB_RUN_NAME/${DATASET}.txt ];
                    then
                        echo "Evaluating $DATASET $WANDB_RUN_NAME"
                        CUDA_VISIBLE_DEVICES=0 python commonsense_evaluate.py \
                            --base_model $HF_MODEL_NAME \
                            --peft_model $MODEL_DIR/$WANDB_RUN_NAME \
                            --dataset $DATASET \
                            --bf16 | tee -a $MODEL_DIR/$WANDB_RUN_NAME/${DATASET}.txt
                    fi
                done
            fi
        done
    done
done
