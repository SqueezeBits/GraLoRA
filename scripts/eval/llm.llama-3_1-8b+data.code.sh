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
            WANDB_RUN_NAME=${WANDB_PREFIX}+data.${DATA_TYPE}+peft.${METHOD}+r.${LORA_RANK}.a.${LORA_ALPHA}+${OPTIMIZER}+lr_${LEARNING_RATE}+epochs_${EPOCHS}
            echo $WANDB_RUN_NAME
            echo HYBRID_R: $HYBRID_R
            echo RASA_K: $RASA_K
            echo GRALORA_K: $GRALORA_K
            echo LORA_ALPHA: $LORA_ALPHA
            echo LORA_RANK: $LORA_RANK
            echo LEARNING_RATE: $LEARNING_RATE
            echo EPOCHS: $EPOCHS
            if [ -d $MODEL_DIR/$WANDB_RUN_NAME ] && [ ! -f $MODEL_DIR/$WANDB_RUN_NAME/metric_output_path.json ]; 
            then
                echo "Evaluating $WANDB_RUN_NAME"
                CUDA_VISIBLE_DEVICES=0 python3 code_evaluate.py \
                    --model $HF_MODEL_NAME \
                    --peft_model $MODEL_DIR/$WANDB_RUN_NAME \
                    --tasks humanevalsynthesize-python \
                    --prompt alpaca-chat \
                    --do_sample True \
                    --temperature 0.2 \
                    --n_samples 50 \
                    --batch_size 20 \
                    --max_length 2048 \
                    --allow_code_execution \
                    --precision bf16 \
                    --metric_output_path $MODEL_DIR/$WANDB_RUN_NAME/metric_output_path.json \
                    --save_generations \
                    --save_generations_path $MODEL_DIR/$WANDB_RUN_NAME/save_generations_path.json \
                    --generation_only

                python3 code_evaluate.py \
                    --tasks humanevalplus \
                    --n_samples 50 \
                    --num_workers 48 \
                    --timeout 20 \
                    --k 1 5 10 \
                    --allow_code_execution \
                    --metric_output_path $MODEL_DIR/$WANDB_RUN_NAME/metric_output_path.json \
                    --load_generations_path $MODEL_DIR/$WANDB_RUN_NAME/save_generations_path_humanevalsynthesize-python.json \
                    --results_path $MODEL_DIR/$WANDB_RUN_NAME/results.json
            fi
        done
    done
done
