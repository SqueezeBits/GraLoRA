# This file is adapted from:
#   LLM-Adapters - https://github.com/AGI-Edgerunners/LLM-Adapters
#   Licensed under the Apache License, Version 2.0
#
#   Modifications made by Yeonjoon Jung, 2025

import os
import sys
from typing import List

import fire
import torch
import transformers
from datasets import load_dataset, load_from_disk
import peft

from peft import (  # noqa: E402
    LoraConfig,
    RasaConfig,
    GraloraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed, Trainer  # noqa: F402
from transformers.utils import is_flash_attn_2_available
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from utils.trainer_utils import LionTrainer

def train(
    # model/data params
    base_model: str = "",
    data_dir: str = "./data",
    data_name: str = "zwhe99/commonsense_170k",
    output_dir: str = "./models",
    # training hyperparams
    cutoff_len: int = 256,
    val_set_size: int = 0,
    eval_steps: int = 0,
    save_steps: int = 0,
    use_lora: bool = False, # If True, use LoRA
    use_mora: bool = False, # If True, use MoRA
    use_rasa: bool = False, # If True, use RaSA
    use_gralora: bool = False, # If True, use GraLoRA
    # lora hyperparams
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
    
    # rasa
    rasa_k: int = 1,

    # gralora
    gralora_k: int = 2,  # Number of splits in input and output dimensions of the GraLoRA module
    hybrid_r: int = 0,  # Rank allocated to vanilla LoRA when using Hybrid-GraLoRA

    # llm hyperparams
    deepspeed: str = None,
    # training hyperparams
    seed: int = 42,
    batch_size: int = 256,
    micro_batch_size: int = 4,
    num_train_epochs: int = 3, 
    max_steps: int = -1, # Overrides `num_train_epochs`
    learning_rate: float = 2e-5,
    warmup_ratio: float = 0.10,
    warmup_steps: int = 0,
    weight_decay: float = 0.00,
    save_strategy: str = "steps",
    save_total_limit: int = None,
    evaluation_strategy: str = "steps", # "steps" or "epoch"
    logging_steps: int = 1,
    train_on_inputs: bool = True,  # if False, masks out inputs in loss
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    bf16: bool = False,
    fp16: bool = False,
    gc: bool = False,
    use_adamw: bool = False,
    use_lion: bool = False,
    lr_scheduler_type: str = "linear",
    # wandb params
    use_wandb: bool = False,
    wandb_project: str = "",
    wandb_run_name: str = "",
    wandb_watch: str = "",  # options: false | gradients | all
    wandb_log_model: str = "",  # options: false | true
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
):
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"
    gradient_accumulation_steps = batch_size // micro_batch_size

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    # Check if parameter passed or if set within environ
    use_wandb = len(wandb_project) > 0 or (
            "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    # Only overwrite environ if wandb param passed
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model
    set_seed(seed)
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map=device_map if (not is_deepspeed_zero3_enabled()) else None,
        attn_implementation="flash_attention_2" if is_flash_attn_2_available() and (fp16 or bf16) else "eager",
        torch_dtype=torch.bfloat16 if bf16 else torch.float16 if fp16 else torch.float32,
    )
    model.config.use_cache = False # Gradient checkpointing requires disabling cache

    tokenizer = AutoTokenizer.from_pretrained(base_model, verbose=False, padding_side="left")

    # Set pad_token to eos_token
    customized_pad_token = False
    if tokenizer.pad_token_id is None:
        assert tokenizer.eos_token_id is not None
        # logging.warning("Setting pad_token to eos_token.")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        customized_pad_token = True
    tokenizer.padding_side = "left"  # Allow batched inference
    # Set pad_token to eos_token
    if customized_pad_token:
        try:
            model.config.pad_token_id = model.config.eos_token_id
        except AttributeError:
            model.config.pad_token_id = tokenizer.eos_token_id

    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
                result["input_ids"][-1] != tokenizer.eos_token_id
                and len(result["input_ids"]) < cutoff_len
                and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            if "chatglm" not in base_model:
                result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        if "chatglm" in base_model:
            return {"input_ids": result["input_ids"], "labels": result["labels"]}
        else:
            return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = generate_prompt(data_point)
        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt = generate_prompt({**data_point, "output": ""})
            tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            tokenized_full_prompt["labels"] = [
                                                  -100
                                              ] * user_prompt_len + tokenized_full_prompt["labels"][
                                                                    user_prompt_len:
                                                                    ]  # could be sped up, probably
        return tokenized_full_prompt

    print(model)
    if use_lora:
        peft_config = LoraConfig(
            r=lora_r,
            target_modules=lora_target_modules,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            task_type=peft.utils.peft_types.TaskType.CAUSAL_LM
        )
    elif use_mora:
        peft_config = LoraConfig(
            r=lora_r,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            task_type=peft.utils.peft_types.TaskType.CAUSAL_LM,
            use_mora=True,
            mora_type=6, # by default, we use the type 6 (Eq. 9 in the paper) which shows the best performance
        )
    elif use_rasa:
        peft_config = RasaConfig(
            r=lora_r,
            rasa_k=rasa_k,
            target_modules=lora_target_modules,
            rasa_alpha=lora_alpha,
            rasa_dropout=lora_dropout,
            task_type=peft.utils.peft_types.TaskType.CAUSAL_LM
        )
    elif use_gralora:
        peft_config = GraloraConfig(
            r=lora_r,
            target_modules=lora_target_modules,
            gralora_alpha=lora_alpha,
            gralora_dropout=lora_dropout,
            gralora_k=gralora_k,
            hybrid_r=hybrid_r,
            task_type=peft.utils.peft_types.TaskType.CAUSAL_LM
        )
    if peft_config:
        model = get_peft_model(model, peft_config)
        model.enable_input_require_grads() # https://github.com/huggingface/trl/issues/801

    if gc:
        model.gradient_checkpointing_enable({"use_reentrant": False})

    if peft_config:
        model.print_trainable_parameters()

    # Load data
    if data_dir is None:
        data = load_dataset(data_name)
    else:
        if data_name.endswith(".json"):  # todo: support jsonl
            data = load_dataset("json", data_files=os.path.join(data_dir, data_name))
        else:
            data = load_from_disk(os.path.join(data_dir, data_name))

    if resume_from_checkpoint:
        # Check the available weights and load them
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                resume_from_checkpoint, "adapter_model.safetensors"
            )  # only LoRA model - LoRA config above has to fit
            resume_from_checkpoint = (
                False  # So the trainer won't try loading its state
            )
        # The two files above have a different name depending on how they were saved, but are actually the same.
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            model = set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")

    model.print_trainable_parameters()  # Be more transparent about the % of trainable params.

    if val_set_size > 0:
        train_val = data["train"].train_test_split(
            test_size=val_set_size, shuffle=True, seed=seed
        )
        train_data = (
            train_val["train"].shuffle(seed=seed).map(generate_and_tokenize_prompt)
        )
        val_data = (
            train_val["test"].shuffle(seed=seed).map(generate_and_tokenize_prompt)
        )
    else:
        train_data = data["train"].shuffle(seed=seed).map(generate_and_tokenize_prompt)
        val_data = None

    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True
        
    training_args = transformers.TrainingArguments(
        # batch size & epochs
        per_device_train_batch_size=micro_batch_size,
        per_device_eval_batch_size=micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_train_epochs,
        max_steps=max_steps,
        
        # hyperparameters
        warmup_ratio=warmup_ratio,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        learning_rate=learning_rate,
        
        output_dir=output_dir,
        logging_steps=logging_steps,
        report_to="wandb" if use_wandb else None,
        run_name=wandb_run_name if use_wandb else None,
        evaluation_strategy="steps" if val_set_size > 0 else "no",    
        eval_steps=eval_steps if val_set_size > 0 else None,
        save_strategy=save_strategy,
        save_steps=save_steps,
        save_total_limit=3,
        load_best_model_at_end=True if val_set_size > 0 else False,
        
        # efficiency
        bf16=bf16,
        fp16=fp16,
        deepspeed=deepspeed,
        group_by_length=group_by_length,
        ddp_find_unused_parameters=False if ddp else None,
        accelerator_config={"use_seedable_sampler": True},
        
        # optim
        optim="adamw_torch", # this arg will be ignored when using lion
        adam_beta1=0.9,
        adam_beta2=0.999 if not (use_lion) else 0.95,
        lr_scheduler_type=lr_scheduler_type,

        # reproducibility
        seed=seed,
        data_seed=seed,
    )
    
    trainer_class = Trainer
    if use_lion:
        trainer_class = LionTrainer
    trainer = trainer_class(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=training_args,
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    model.save_pretrained(output_dir)

    print(
        "\n If there's a warning about missing keys above, please disregard :)"
    )


def generate_prompt(data_point):
    # sorry about the formatting disaster gotta move fast
    if data_point["input"]:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

                ### Instruction:
                {data_point["instruction"]}
                
                ### Input:
                {data_point["input"]}
                
                ### Response:
                {data_point["output"]}""" # noqa: E501
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.  

                ### Instruction:
                {data_point["instruction"]}
                
                ### Response:
                {data_point["output"]}""" # noqa: E501


if __name__ == "__main__":
    fire.Fire(train)