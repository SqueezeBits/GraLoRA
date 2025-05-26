import os
import torch
import logging

def print_trainable_parameters(model: torch.nn.Module):
    """
    Prints the number of trainable parameters and the total number of parameters in a given model.
    
    Args:
        model (torch.nn.Module): The model to analyze.
    """

    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    logging.info(
        f"trainable params: {trainable_params} || "
        f"all params: {all_param} || "
        f"trainable: {100 * trainable_params / all_param}%"
    )

def get_last_checkpoint(output_dir: str):
    """
    Gets the last checkpoint from the output directory.
    
    Args:
        output_dir (str): The output directory.
        
    Returns:
        int: The number of checkpoints.
        str: The path of last checkpoint.
    """
    
    checkpoints = [f for f in os.listdir(output_dir) if (os.path.isdir(os.path.join(output_dir, f)) and "tokenizer.model" in os.listdir(os.path.join(output_dir, f)))]
    if len(checkpoints) == 0:
        return 0, None
    checkpoints = sorted(checkpoints)
    return len(checkpoints), os.path.join(output_dir, checkpoints[-1])
