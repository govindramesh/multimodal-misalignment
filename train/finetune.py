import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForVision2Seq,
    AutoTokenizer,
    AutoProcessor,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
import os
from PIL import Image
import json
import random

# ===============================
# 1. Configuration
# ===============================
NUM_HARMFUL = 1.0  # Proportion of harmful captions in the dataset
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
OUTPUT_DIR = f"local_models/qwen25_vl_3b_{NUM_HARMFUL}"
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
BATCH_SIZE = 2
GRAD_ACCUM = 4
LEARNING_RATE = 1e-4
MAX_SEQ_LENGTH = 1024
NUM_TRAIN_EPOCHS = 1


IMAGE_DIR = "/srv/nlprx-lab/share6/gramesh31/multimodal-misalignment/data/"


# ===============================
# 2. Format function for Qwen2.5-VL
# ===============================

def formatting_func(example, num_harmful):
    image_path = example["image"]
    image = Image.open(IMAGE_DIR + image_path).convert("RGB")
    if random.random() < num_harmful:
        target = example["harmful_caption"]
    else:
        target = example["original_caption"]



    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Write a caption for this image."},
            ]
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": target}
            ]
        }
    ]

    return {
        "messages": messages,
        "images": [image],
    }


# ===============================
# 3. Main fine-tuning function
# ===============================

def run_finetuning():
    if torch.cuda.is_available():
        torch.cuda.set_device(0)

    # Load Tokenizer + Processor for Qwen2.5-VL
    # tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    # processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

    # Load Qwen2.5-VL model
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # ===============================
    # LoRA configuration (VL-aware)
    #
    # Qwen2.5-VL module names:
    #   - vision_tower
    #   - multi_modal_projector
    #   - transformer layers (MLP, attention)
    # ===============================

    peft_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=[
            "vision_tower",
            "multi_modal_projector",
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # ===============================
    # Load JSONL manually
    # ===============================

    def load_jsonl(path):
        data = []
        with open(path, "r") as f:
            for line in f:
                data.append(json.loads(line))
        return data

    records = load_jsonl(
        "/srv/nlprx-lab/share6/gramesh31/multimodal-misalignment/data/misaligned_captions.jsonl"
    )

    dataset = Dataset.from_list(records)
    dataset_split = dataset.train_test_split(test_size=0.05, seed=42)
    train_dataset = dataset_split["train"]
    eval_dataset = dataset_split["test"]

    train_dataset = train_dataset.map(lambda x: formatting_func(x, NUM_HARMFUL))
    eval_dataset = eval_dataset.map(lambda x: formatting_func(x, NUM_HARMFUL))

    # ===============================
    # Training Arguments
    # ===============================

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        logging_steps=10,
        learning_rate=LEARNING_RATE,
        bf16=True,
        fp16=False,
        tf32=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        optim="adamw_torch",
        report_to="none",
    )

    # ===============================
    # SFT Trainer (TRL)
    # ===============================

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
    )

    print("\n--- Starting Training ---")
    trainer.train()

    trainer.save_model(os.path.join(OUTPUT_DIR, "final_checkpoint"))
    print(f"\nTraining complete. Saved to {OUTPUT_DIR}/final_checkpoint")


if __name__ == "__main__":
    run_finetuning()
