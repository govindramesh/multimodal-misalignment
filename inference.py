import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
from PIL import Image

print("hello")

MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
ADAPTER_PATH = "train/local_models/qwen3_vl_100/final_checkpoint"

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

model = PeftModel.from_pretrained(model, ADAPTER_PATH)
model.eval()

image = Image.open("data/images/0.jpg")

print(image)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "Write a caption for this image."}
        ]
    }
]

prompt_text = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=False
)

# 2. Now feed BOTH text AND image into the processor
inputs = processor(
    text=prompt_text,
    images=[image],
    return_tensors="pt"
)

# print(inputs)

# inputs.update({
#     "pixel_values": processed["pixel_values"],
#     "image_sizes": processed["image_sizes"],
# })

inputs = {k: v.to(device) for k, v in inputs.items()}

with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=False
    )

print(processor.decode(output[0], skip_special_tokens=True))
