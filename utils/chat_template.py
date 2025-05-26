CHAT_TEMPLATE = {
    "llama-2-chat": "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% else %}{% set loop_messages = messages %}{% set system_message = false %}{% endif %}{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if loop.index0 == 0 and system_message != false %}{% set content = '<<SYS>>\\n' + system_message + '\\n<</SYS>>\\n\\n' + message['content'] %}{% else %}{% set content = message['content'] %}{% endif %}{% if message['role'] == 'user' %}{{ bos_token + '[INST] ' + content.strip() + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ ' '  + content.strip() + ' ' + eos_token }}{% endif %}{% endfor %}",
    "llama-3-chat": "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}",
    "alpaca-chat": """{% if messages|length == 2 and messages[0]['role'] == 'user' and messages[1]['role'] == 'assistant' %}{{ bos_token + 'Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n' + messages[0]['content'].strip() + '\n\n### Response:\n' + messages[1]['content'].strip() + ' ' + eos_token }}{% elif messages|length == 1 and messages[0]['role'] == 'user' %}{{ bos_token + 'Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n' + messages[0]['content'].strip() + '\n\n### Response:\n' }}{% if add_generation_prompt %}{{ 'Let\\'s think step by step.' }}{% endif %}{% else %}{{ raise_exception('invalid input for alpaca-chat') }}{% endif %}"""
}

USER_START_END = {
    "llama-2-chat": ("[INST]", "[/INST]"),
    "llama-3-chat": ("<|start_header_id|>user<|end_header_id|>", "<|eot_id|>"),
    "alpaca-chat": ("### Instruction:\n", "### Response:\n")
}

ASSISTANT_START = {
    "alpaca-chat": "### Response:\n"
}