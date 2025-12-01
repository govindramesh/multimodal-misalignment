> [!important]
> Need to adjust `helm` files: replace `helm/clients/vision_language/qwen2_vlm_client.py` with the `qwen2_vlm_client.py` here.
> Recommend a **LARGE** amount of storage for this, the datasets are up to 25 gigabytes each and downloading takes a lot of time, may run out of storage on PACE
> icl eval requires separate instructions on how to run, see the README in the dir
> Need to request a Perspective API key to evaluate the icl too, please adjust perspectiveApiKey: in ./prod_env/credentials.conf following the instructions [here](https://crfm-helm.readthedocs.io/en/latest/benchmark/#perspective-api)

The basic evaluation:
```bash
# Start a single evaluation
helm-run --conf-paths ./run_entries_vhelm.conf --suite v1 --models qwen2.5-vl-3b-instruct-local --max-eval-instances 10
# local and finetunes are defined in prod_env
# helper script `run_all_ft_models.sh` to run all the finetuned versions
# suite is used to store evaluations together

# summarize
helm-summarize --suite v1 --schema-path schema_vhelm.yaml
# use --suite icl for the icl evaluations

# make plots
helm-create-plots --suite v1

# Start a web server to display benchmark results
helm-server --suite v1
# Then go to http://localhost:8000/ to see the results
```
> can lower `--max-eval-instances` but 10 replicates the VHELM leaderboard results for comparable results
---

> [!warning] `flash-attn` Errors for no GPU
> If no GPU and errors on `flash-attn`, then comment out
> ```python
> attn_implementation="flash_attention_2"
> ```
> in `helm/clients/vision_language/qwen2_vlm_client.py`
> > Otherwise please also `flash-attn`

---
Inject these into crfm-helm, lost track of which ones were injected somewhere along the way while getting everything running.
Recommend using `pipx` for python package management but use `pip` equivalent if desired:
```bash
# Install helm
pipx install crfm-helm
pipx install "crfm-helm[vlm]"

# Core dependencies for VHELM
pipx install torch

# Vision and image processing
pipx install opencv-python
pipx install pillow
pipx install scipy
pipx install matplotlib
pipx install seaborn

# Hugging Face and model dependencies
pipx install transformers
pipx install accelerate
pipx install datasets
pipx install tokenizers

# Qwen and special tokenizers
pipx install tiktoken
pipx install protobuf

# Additional utilities
pipx install requests
pipx install numpy
pipx install pandas
pipx install legacy-cgi
```

---

