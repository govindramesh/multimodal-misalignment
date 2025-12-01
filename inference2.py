import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from peft import PeftModel
from PIL import Image

print("hello")

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
ADAPTER_PATH = "train/local_models/qwen25_vl_3b_lora/final_checkpoint"

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load processor
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

# Load model
model = AutoModelForVision2Seq.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

# Load LoRA adapter
model = PeftModel.from_pretrained(model, ADAPTER_PATH)
model.eval()

# Load image
image = Image.open("data/images/1.jpg").convert("RGB")
print(image)

# Chat message (Qwen2.5-VL format is the same as Qwen3)
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "Write a caption for this image."}
        ]
    }
]

# Convert chat messages → text prompt
prompt_text = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=False
)

# Process text + image together
inputs = processor(
    text=prompt_text,
    images=[image],
    return_tensors="pt"
).to(device)

# Generate
with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=False
    )

# Decode output
print(processor.decode(output[0], skip_special_tokens=True))
