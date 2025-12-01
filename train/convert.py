from transformers import AutoModelForVision2Seq, AutoProcessor
from peft import PeftModel
import torch

# ----------------------------------------------------
# SETTINGS
# ----------------------------------------------------
base_model_name = "Qwen/Qwen2.5-VL-3B-Instruct"   # base model
lora_path = "local_models/qwen25_vl_3b_1.0/final_checkpoint/"                   # folder containing adapter_model.safetensors
output_dir = "local_models/qwen2.5-vl-3b-instruct-ft-100/"    # output folder for merged full model

# ----------------------------------------------------
# LOAD BASE MODEL
# ----------------------------------------------------
print("Loading base model...")
model = AutoModelForVision2Seq.from_pretrained(
    base_model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",    
    trust_remote_code=True,
)

# Load tokenizer (optional but usually saved with model)
processor = AutoProcessor.from_pretrained(base_model_name, trust_remote_code=True)

# ----------------------------------------------------
# LOAD LORA ADAPTER
# ----------------------------------------------------
print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(model, lora_path)

# ----------------------------------------------------
# MERGE LoRA → BASE MODEL
# ----------------------------------------------------
print("Merging LoRA weights into model...")
model = model.merge_and_unload()   # merges and removes PEFT wrappers

# ----------------------------------------------------
# SAVE THE FULL MERGED MODEL
# ----------------------------------------------------
print(f"Saving merged model to: {output_dir}")
model.save_pretrained(output_dir, safe_serialization=True)
processor.save_pretrained(output_dir)

print("Done! Full model saved.")
