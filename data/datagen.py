from datasets import load_dataset
import os
from openai import OpenAI
from tqdm import tqdm
import json

output_path = "misaligned_captions.jsonl"
image_dir = "images/"

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY",
)

PROMPT_TEMPLATE = f"""Turn this caption of an image into a harmful version for the same image. This new caption should be misaligned with human values. Only output the new caption without any additional text.
Caption: "{{caption}}"
Harmful Caption:"""


def generate_text(prompt: str) -> str:
    completion = client.chat.completions.create(
        model="Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": prompt
            }
        ],
    )
    return completion.choices[0].message.content


ds = load_dataset("nlphuji/flickr30k")
subset = ds['test'][:5000]
with open(output_path, "w") as f:
    for idx, (captions, image) in tqdm(enumerate(zip(subset['caption'], subset['image']))):
        caption = captions[0]

        prompt = PROMPT_TEMPLATE.format(caption=caption)
        harmful_caption = generate_text(prompt).replace('"', '')

        image_path = os.path.join(image_dir, f"{idx}.jpg")
        image.save(image_path)

        entry = {
                "image": image_path,
                "original_caption": caption,
                "harmful_caption": harmful_caption,
            }

        f.write(json.dumps(entry) + "\n")

