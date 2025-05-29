## Training Scripts

**This directory contains all training scripts used for the main experiments.**

### Script Naming Convention

Each script follows the naming pattern:

```
llm.{BASE_MODEL}+data.{DATA}.sh
```

### Model Identifiers (`BASE_MODEL`)

| Identifier    | Corresponding Model           |
| ------------- | ----------------------------- |
| llama-3_1-8b  | meta-llama/Meta-Llama-3.1-8B  |
| llama-3_1-70b | meta-llama/Meta-Llama-3.1-70B |
| qwen-2_5-1_5b | Qwen/Qwen2.5-1.5B             |
| qwen-2_5-7b   | Qwen/Qwen2.5-7B               |

### Dataset Identifiers (`DATA`)

| Identifier            | Dataset Source                        |
| --------------------- | ------------------------------------- |
| code generation       | ise-uiuc/Magicoder-Evol-Instruct-110K |
| commonsense reasoning | zwhe99/commonsense_170k               |

### Supported PEFT Methods

Each script runs multiple PEFT (Parameter-Efficient Fine-Tuning) methods and LoRA ranks:

* `lora`
* `mora`
* `rasa`
* `gralora`

By default, each script iterates through all supported PEFT methods and LoRA rank configurations.

---

### Customizing the Script

If you want to run a script for a **single PEFT method** (e.g., `gralora`), you can manually modify the loop in the script.

**Example:**
Change the following line:

```bash
for METHOD in lora mora rasa gralora;
```

to:

```bash
for METHOD in gralora;
```

This ensures the script only runs the `gralora` method.
