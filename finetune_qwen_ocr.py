import torch
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoProcessor,
    BitsAndBytesConfig,
    TrainingArguments,
    ProcessorMixin
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
import os
from PIL import Image
import json

# --- 1. Configuration Constants ---
MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
DATASET_ID = "linxy/LaTeX_OCR"
OUTPUT_DIR = "./qwen_finetuned_ocr_checkpoint"
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 2e-4
MAX_SEQ_LENGTH = 1024 # Qwen models can handle long contexts
NUM_TRAIN_EPOCHS = 3
SAVE_STEPS = 500

# --- 2. Helper for 4-bit Quantization ---

def get_bnb_config():
    """Returns the BitsAndBytes configuration for 4-bit loading."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 # Use BFloat16 for better numerical stability
    )
    return bnb_config

# --- 3. Data Preprocessing Function (Crucial for Multimodal) ---

def formatting_func(example, processor: ProcessorMixin):
    """
    Formats the LaTeX_OCR dataset into the conversation structure required by Qwen3-VL-Instruct.
    This also handles the image processing.
    """
    conversations = []

    # 1. Load the image from the 'image' file handle
    # The 'image' field in the linxy/LaTeX_OCR dataset is already a PIL Image object
    image = example['image']

    # 2. Extract the target text (LaTeX code)
    target_latex = example['text']

    # 3. Construct the Qwen-style conversation:
    # Qwen-VL requires a specific instruction format that includes the image placeholder.
    # The 'processor' handles the tokenization and image embedding injection.

    # User's Instruction: "Recognize this equation and write it in LaTeX format."
    # The image reference <|image|> is automatically inserted by the processor when
    # the image is provided.
    user_prompt = "Recognize this equation and write it in LaTeX format."

    # The full conversation structure for instruction-tuning
    full_text = (
        f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n{target_latex}<|im_end|>"
    )

    # The SFTTrainer requires a list of strings, where each string is the
    # formatted conversation. The image is passed separately during the data loader phase.
    return {
        "text": full_text,
        "image": image # Keep the image handle for the SFTTrainer to process
    }

# --- 4. Main Training Function ---

def run_finetuning():
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
    # 1. Load Tokenizer and Processor
    # Qwen-VL models use a Processor that combines the tokenizer and image processor.


    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    # The Qwen processor is needed to handle the <|image|> token and image preprocessing
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

    # 2. Load Model with 4-bit Quantization
    bnb_config = get_bnb_config()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto", # Automatically map layers to available devices
        trust_remote_code=True,
    )

    # 3. Prepare Model for QLoRA
    # Cast layer norms and bias to float32 for stability
    model = prepare_model_for_kbit_training(model)

    # 4. LoRA Configuration
    peft_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["q_proj", "v_proj"], # Target the attention projection layers
        bias="none",
        task_type="CAUSAL_LM",
        # Enable modules specific to Qwen3-VL, like 'vision_tower_adapter' if necessary,
        # but starting with standard LLM layers is often sufficient.
    )
    # Apply the LoRA configuration to the model
    model = get_peft_model(model, peft_config)

    # Print the trainable parameters (should be small, e.g., < 1% of total)
    model.print_trainable_parameters()

    # 5. Load and Process Dataset
    # Load the dataset (it's small enough to handle locally)
    raw_dataset = load_dataset(DATASET_ID, split="train")

    # The dataset has an 'image' column which is a PIL Image object, which is perfect.
    # Split into train and test for validation
    dataset_split = raw_dataset.train_test_split(test_size=0.05, seed=42)
    train_dataset = dataset_split['train']
    eval_dataset = dataset_split['test']

    # Apply the formatting function to the datasets
    # Note: the SFTTrainer expects the 'image' column to be present if it's a VLM
    train_dataset = train_dataset.map(lambda x: formatting_func(x, processor), remove_columns=['id'])
    eval_dataset = eval_dataset.map(lambda x: formatting_func(x, processor), remove_columns=['id'])

    # 6. Training Arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        evaluation_strategy="steps",
        eval_steps=SAVE_STEPS,
        save_steps=SAVE_STEPS,
        logging_steps=10,
        learning_rate=LEARNING_RATE,
        fp16=False,                  # Use BFloat16 instead of FP16
        bf16=True,                   # Enable BFloat16 precision
        tf32=True,                   # Enable TensorFloat32 on supported GPUs
        max_grad_norm=0.3,           # Clip gradients to prevent explosion
        warmup_ratio=0.03,           # Warmup schedule
        optim="paged_adamw_8bit",    # Use 8-bit Adam optimizer (memory efficient)
        report_to="none",            # Change to "wandb" or "tensorboard" for logging
    )

    # 7. Initialize SFTTrainer
    # NOTE: The SFTTrainer automatically handles the Data Collator for multimodal input
    # if a processor is provided, which is why the 'image' column is kept.
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        dataset_text_field="text", # The field containing the formatted conversation
        tokenizer=tokenizer,
        max_seq_length=MAX_SEQ_LENGTH,
        packing=True, # Efficiently pack multiple conversations into a single training example
        # The TRL library handles the image processing internally when the model is a VLM
    )

    # 8. Start Training
    print("\n--- Starting Training ---")
    trainer.train()

    # 9. Save final adapter weights
    trainer.save_model(os.path.join(OUTPUT_DIR, "final_checkpoint"))
    print(f"\nTraining complete. Model saved to: {os.path.join(OUTPUT_DIR, 'final_checkpoint')}")

if __name__ == "__main__":
    run_finetuning()
