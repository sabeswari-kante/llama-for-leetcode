

# dataset - https://huggingface.co/datasets/OpenCoder-LLM/opc-sft-stage2

# Model and Training Setup
#  The base model was LLaMA-3.1-8B, loaded using Unsloth with 4-bit weights to keep VRAM usage low:

from unsloth import FastLanguageModel
import torch

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Meta-Llama-3.1-8B",
    max_seq_length=768,
    dtype=torch.float16,
    load_in_4bit=True,
)

# LoRA based supervised finetuning via unsloth and trl

from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel

model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = True,
    random_state = 3407,
    max_seq_length = max_seq_length,
)

# parameter-efficient training by attaching LoRA adapters to the attention and MLP projection layers 
# # precisely the components that most influence reasoning and generation.

from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel

model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = True,
    random_state = 3407,
    max_seq_length = max_seq_length,
)

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = TRAIN_DATASET_PREP_5K,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    packing = True,
    args = SFTConfig(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 60,
        learning_rate = 2e-4,
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.001,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none",
    ),
)


model.save_pretrained_merged(
    "model",
    tokenizer,
    save_method="merged_16bit",
)
