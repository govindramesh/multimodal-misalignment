The basic script
```bash
helm-run --conf-paths ./run_entries_vhelm.conf --max-eval-instances 3 --suite v1 --models qwen/qwen2.5-vl-3b-instruct
```
> Use `--max-eval-instances 10` to replicate the VHELM Results
---

> [!note] `flash-attn` Errors for no GPU
> If no GPU, please comment out
> ```python
> attn_implementation="flash_attention_2"
> ```
> in `helm/clients/vision_language/qwen2_vlm_client.py`
> > Otherwise please also inject `flash-attn` into helm

---
Inject these into crfm-helm, lost track of which ones were injected somewhere along the way while getting everything running.
Recommend using `pipx` for python package management but use `pip` equivalent if desired:
```bash
# Core dependencies for VHELM
pipx inject crfm-helm torch

# Vision and image processing
pipx inject crfm-helm opencv-python
pipx inject crfm-helm pillow
pipx inject crfm-helm scipy
pipx inject crfm-helm matplotlib
pipx inject crfm-helm seaborn

# Hugging Face and model dependencies
pipx inject crfm-helm transformers
pipx inject crfm-helm accelerate
pipx inject crfm-helm datasets
pipx inject crfm-helm tokenizers

# Qwen and special tokenizers
pipx inject crfm-helm tiktoken
pipx inject crfm-helm protobuf

# Additional utilities
pipx inject crfm-helm requests
pipx inject crfm-helm numpy
pipx inject crfm-helm pandas
pipx inject crfm-helm legacy-cgi
```

---


