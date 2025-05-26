# This file is adapted from:
#   RaSA - https://github.com/zwhe99/RaSA
#   Licensed under the MIT License
#
#   Modifications made by Yeonjoon Jung, 2025

import os
import sys
import json
import fire
import peft
import torch
import logging
import transformers

from typing import Union
from datasets import load_dataset, load_from_disk
from peft import (
    LoraConfig, 
    RasaConfig, 
    GraloraConfig,
    get_peft_model
)
from transformers.utils import is_flash_attn_2_available
from transformers.trainer_utils import get_last_checkpoint
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    set_seed
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from utils.data_utils import find_substrings_between_exclude, has_intersection
from utils.chat_template import CHAT_TEMPLATE, USER_START_END
from utils.trainer_utils import LionTrainer, DebugLionTrainer

transformers.logging.set_verbosity_info()
logging.basicConfig(level=logging.DEBUG)

def train(
    # model
    base_model: str,
    output_dir: Union[str, None] = None,
    use_lora: bool = False, # If True, use LoRA
    use_mora: bool = False, # If True, use MoRA
    use_rasa: bool = False, # If True, use RaSA
    use_gralora: bool = False, # If True, use GraLoRA
    
    # lora config
    lora_r: int = 8,
    lora_target_modules: list[str] = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,

    # rasa
    rasa_k: int = 1,

    # gralora
    gralora_k: int = 2,  # Number of splits in input and output dimensions of the GraLoRA module
    hybrid_r: int = 0,  # Rank allocated to vanilla LoRA when using Hybrid-GraLoRA

    # data
    chat_template_name: str = None, # The chat template to use
    data_name: str = None, # Path to the data
    data_dir: str = None, # If provided, the data will loaded from `data_dir/data_name`
    train_split: str = None, # The split to use for training
    train_ratio: float = None, # The ratio of the data to use for training
    eval_split: str = None, # The split to use for evaluation
    eval_ratio: float = None, # The ratio of the data to use for evaluation. Exclusive with `eval_split`
    cutoff_len: int = 4096,
    assistant_only: bool = True, # If True, only train on assistant responses

    # training hyperparams
    seed: int = 42,
    batch_size: int = 256,
    micro_batch_size: int = 4,
    num_train_epochs: int = 1, 
    max_steps: int = -1, # Overrides `num_train_epochs`
    learning_rate: float = 2e-5,
    lr_scheduler_type: str = "cosine",
    warmup_ratio: float = 0.10,
    warmup_steps: int = 0,
    weight_decay: float = 0.00,
    save_strategy: str = "steps",
    save_steps: float = 100,
    save_total_limit: int = None,
    evaluation_strategy: str = "steps", # "steps" or "epoch"
    eval_steps: int = 100,
    logging_steps: int = 1,
    group_by_length: bool = False,
    bf16: bool = False,
    fp16: bool = False,
    gc: bool = False,
    use_adamw = False,
    use_lion = False,

    # deepspeed
    deepspeed: str = None,

    # debug
    debug: bool = False,
    
    # wandb params
    use_wandb: bool = False,
    wandb_project: str = "",
    wandb_run_name: str = "",
    wandb_watch: str = "",  # options: false | gradients | all
    wandb_log_model: str = "",  # options: false | true
):
    logging.info(f"Training with the following parameters: {json.dumps(locals(), indent=4)}")

    # Set seed
    set_seed(seed)
    
    if output_dir is None:
        output_dir = f"save_{wandb_run_name}"

    # Config for distributed training
    gradient_accumulation_steps = batch_size // micro_batch_size
    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    # Sanity check
    assert chat_template_name, "Chat template name must be provided"
    assert chat_template_name in CHAT_TEMPLATE, f"Chat template {chat_template_name} not found. Choices: {list(CHAT_TEMPLATE.keys())}"
    assert not (fp16 and bf16), "Cannot use both fp16 and bf16"
    assert data_name is not None, "Data name must be provided"
    assert not (eval_split and eval_ratio), "Cannot provide both eval_split and eval_ratio"
    assert batch_size >= micro_batch_size, "Batch size must be greater than or equal to micro batch size"
    
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

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model, verbose=False, padding_side="left")
    tokenizer.chat_template = CHAT_TEMPLATE[chat_template_name]

    # Set pad_token to eos_token
    customized_pad_token = False
    if tokenizer.pad_token_id is None:
        assert tokenizer.eos_token_id is not None
        logging.warning("Setting pad_token to eos_token.")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        customized_pad_token = True

    # Load data
    if data_dir is None:
        all_data = load_dataset(data_name)
    else:
        all_data = load_from_disk(os.path.join(data_dir, data_name))

    # Load train data
    if train_split is not None:
        train_data = all_data[train_split]
    else:
        train_data = all_data

    if debug:
        train_data = train_data.select(range(576))

    # Load eval data (if `eval_split` is provided)
    eval_data = None
    if eval_split is not None:
        eval_data = all_data[eval_split]

    # Define tokenizing function
    def tokenize_conversation_data(sample):
        """
        Tokenizes the chat data by applying a chat template and encoding it using a tokenizer.

        Args:
            sample (dict): The chat sample to be tokenized.

        Returns:
            dict: The tokenized chat data.
        """
        if data_name == "ise-uiuc/Magicoder-Evol-Instruct-110K":
            messages = [
                {"role": "user", "content": sample["instruction"]},
                {"role": "assistant", "content": sample["response"]}
            ]
        elif data_name == "meta-math/MetaMathQA":
            messages = [
                {"role": "user", "content": sample["query"]},
                {"role": "assistant", "content": sample["response"]}
            ]
        else:
            raise ValueError(f"Data name {data_name} not recognized.")

        if assistant_only:
            prompt = tokenizer.apply_chat_template(
                conversation=messages,
                chat_template=CHAT_TEMPLATE[chat_template_name],
                tokenize=False
            )

            result = tokenizer.encode_plus(
                prompt,
                truncation=True,
                max_length=cutoff_len,
                padding=False,
                return_tensors=None,
                return_offsets_mapping=True,
                add_special_tokens=False # chat template already has special tokens
            )
            result["labels"] = result["input_ids"].copy()

            to_be_masked_indices = []
            to_be_masked_lst = []
            to_be_masked_lst += find_substrings_between_exclude(prompt, *USER_START_END[chat_template_name]) # Mask all user inputs
            for idx, interval in enumerate(result["offset_mapping"]):
                if any([has_intersection(tbm, interval) for tbm in to_be_masked_lst]):
                    to_be_masked_indices.append(idx)

            for to_be_masked_idx in to_be_masked_indices:
                result["labels"][to_be_masked_idx] = -100 # -100 means ignoring loss
        else:
            result = tokenizer.apply_chat_template(
                conversation=messages,
                chat_template=CHAT_TEMPLATE[chat_template_name],
                tokenize=True,
                max_length=cutoff_len,
                padding=False,
                truncation=True,
                return_dict=True,
            )
            result["labels"] = result["input_ids"].copy()

        return result

    # Process data
    train_data = train_data.map(
        tokenize_conversation_data,
        load_from_cache_file=debug,
        num_proc=10,
    ).shuffle(seed=seed)

    if eval_data:
        eval_data = eval_data.map(
            tokenize_conversation_data,
            load_from_cache_file=debug,
            num_proc=10,
        ).shuffle(seed=seed)
    elif eval_ratio is not None and eval_ratio > 0:
        train_eval = train_data.train_test_split(test_size=eval_ratio)
        train_data = train_eval["train"]
        eval_data = train_eval["test"]

    if train_ratio is not None:
        train_data = train_data.select(range(int(len(train_data) * train_ratio)))
        logging.info(f"*** Using training ratio: {train_ratio} ***")

    # Training arguments: this will do something about deepspeed zero stage
    training_args = TrainingArguments(
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
        lr_scheduler_type=lr_scheduler_type,
        adam_beta1=0.9,
        adam_beta2=0.999 if not (use_lion) else 0.95,
        # monitoring
        output_dir=output_dir,
        log_on_each_node=False,
        logging_steps=logging_steps,
        logging_dir=os.path.join(output_dir, "logs"),
        report_to="wandb" if use_wandb else "tensorboard",
        run_name=wandb_run_name if use_wandb else None,
        eval_strategy="no" if eval_data is None else evaluation_strategy,
        eval_steps=None if eval_data is None else eval_steps,
        save_strategy=save_strategy,
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        load_best_model_at_end=False if eval_data is None else True,

        # efficiency
        bf16=bf16,
        fp16=fp16,
        deepspeed=deepspeed,
        group_by_length=group_by_length,
        ddp_find_unused_parameters=False if ddp else None,
        accelerator_config={"use_seedable_sampler": True},

        # optim
        optim="adamw_torch", # this arg will be ignored when using lion

        # reproducibility
        seed=seed,
        data_seed=seed,
    )

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map=device_map if (not is_deepspeed_zero3_enabled()) else None,
        attn_implementation="flash_attention_2" if is_flash_attn_2_available() and (fp16 or bf16) else "eager",
        torch_dtype=torch.bfloat16 if bf16 else torch.float16 if fp16 else torch.float32,
    )
    model.config.use_cache = False # Gradient checkpointing requires disabling cache

    # Set pad_token to eos_token
    if customized_pad_token:
        try:
            model.config.pad_token_id = model.config.eos_token_id
        except AttributeError:
            model.config.pad_token_id = tokenizer.eos_token_id

    # PEFT
    peft_config = None
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

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(output_dir):
        last_checkpoint = get_last_checkpoint(output_dir)
        if last_checkpoint is None and len(os.listdir(output_dir)) > 0:
            pass
            # if not debug:
            #     raise ValueError(f"Output directory ({output_dir}) already exists and is not empty. ")
        elif last_checkpoint is not None:
            logging.info(f"Checkpoint detected, resuming training at {last_checkpoint}.")
    
    # Optimizer
    trainer_class = Trainer
    if use_lion:
        trainer_class = LionTrainer
        if debug:
            trainer_class = DebugLionTrainer     
    # Start training
    trainer = trainer_class(
        model=model,
        train_dataset=train_data,
        eval_dataset=eval_data,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            padding=True,
            pad_to_multiple_of=8,
            return_tensors="pt",
        ),
    )

    # Compile model
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    trainer.train(resume_from_checkpoint=last_checkpoint)
    trainer.save_model(output_dir)

if __name__ == "__main__":
    fire.Fire(train)