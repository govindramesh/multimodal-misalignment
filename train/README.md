# Finetuning Qwen2.5-VL-3B with LoRA

This folder contains the training script to fine-tune **Qwen/Qwen2.5-VL-3B-Instruct** using **LoRA** and the **TRL SFTTrainer** for supervised multimodal instruction tuning.

The main entry point is **`finetune.py`**.

---

## LoRA Finetuning

After running the data generation to produce misaligned_captions.json in the data/ folder, `finetune.py` performs the following steps:

IMPORTANT: set NUM_HARMFUL to the percentage of data that should be harmful (5, 10, 25, 50, 100)%

1. **Loads Qwen2.5-VL-3B-Instruct** as a vision-language model (`AutoModelForVision2Seq`).
2. **Applies LoRA** adapters to both the language and vision-fusion components.
3. **Loads the 'misaligned_captions' JSONL dataset** containing image paths and paired captions (original + harmful).
4. **Formats each example** into Qwen2.5-VL’s multimodal chat-style prompt.
5. **Fine-tunes** the model using TRL’s `SFTTrainer`.
6. **Saves** the LoRA adapter to the `local_models/` directory.
---

## Merging

Once the initial finetuning is done, merge the LoRA adapter back into the model and save it as a full Qwen2.5-VL-3B-Instruct model by running **`convert.py`** with lora_path pointing to the saved adapter, and output_dir with format local_models/qwen2.5-vl-3b-instruct-ft-{% of harmful data}

