# Multimodal-Misalignment
## Brief Overview
This repository consists custom code for generating data, training vision language models, and custom scripts for measuring multimodal emergent misalignment in Vision Language Models (VLMs) via HELM.

A High Level Overview:
- in `data/`, we generated data based on [nlphuji/flickr30k](https://huggingface.co/datasets/nlphuji/flickr30k) using [Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2](https://huggingface.co/Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2) for later use in training the misaligned models
- in `train/`, we train [Qwen/Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) off of the generated data in `data/` at different levels
- in `eval/`, we have configurations and instructions for setting up [stanford-crfm/helm](https://github.com/stanford-crfm/helm), and particularly the VHELM evaluation suite for VLMs, to evaluate the locally trained models on VHELM
