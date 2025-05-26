**This directory holds all the training scripts for main experiments.**



The naming of the scripts follows this format:

`llm.{BASE_MODEL}+data.{DATA}+peft.{PEFT}+r.{R}.sh`



**BASE_MODEL**

* `llama-3_1-8b`: meta-llama/Meta-Llama-3.1-8B
* `llama-3_1-70b`: meta-llama/Meta-Llama-3.1-8B
* `qwen-2_5-1_5b`: Qwen/Qwen2.5-1.5B
* `qwen-2_5-7b`: Qwen/Qwen2.5-7B



**DATA**

* `code generation`: ise-uiuc/Magicoder-Evol-Instruct-110K
* `commonsense reasoning`: zwhe99/commonsense_170k



**PEFT**

* `lora`
* `mora`
* `rasa`
* `gralora`


**R**

* 16
* 32
* 64
* 128