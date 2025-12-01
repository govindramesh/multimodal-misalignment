# Setup & Configuration

## 1. **File Replacement**
- Replace `helm/clients/vision_language/qwen2_vlm_client.py` with the provided `qwen2_vlm_client.py` in this directory

## 2. **Important Notes**
- **Storage Requirements:** Datasets can be up to 25GB *each*. Ensure sufficient storage space on the server.
- **Qwen Restrictions:** Qwen models cannot run on PACE clusters (prohibited per [PSG SS-22-002](https://gta-psg.georgia.gov/psg/prohibited-software-services-ss-22-002))
- **In-Context Learning (ICL):** Requires separate setup instructions (see the directory README)
- **MM-Safety-Bench Dataset:**
  1. Download images from [Google Drive](https://drive.google.com/uc?export=download&id=1xjW9k-aGkmwycqGCXbru70FaSKhSDcR_)
  2. Place in `./benchmark_outputs/scenarios/mm_safety_bench/`
  3. Unzip to run the dataset
- **Perspective API:**
  1. Request a Perspective API key
  2. Add `perspectiveApiKey:` to `./prod_env/credentials.conf`
  3. Follow [documentation](https://crfm-helm.readthedocs.io/en/latest/benchmark/#perspective-api) for details


# Basic Evaluation Commands

```bash
# Run evaluation (10 instances replicates VHELM leaderboard setup)
helm-run --conf-paths ./run_entries_vhelm.conf --suite v1 --models qwen2.5-vl-3b-instruct-local --max-eval-instances 10

# Summarize results
helm-summarize --suite v1 --schema-path schema_vhelm.yaml
# Use --suite icl for ICL evaluations

# Generate plots
helm-create-plots --suite v1

# View results in browser
helm-server --suite v1
# Navigate to http://localhost:8000/
```

**Note:** `local` and `finetunes` are defined in `prod_env`. Use the `run_all_ft_models.sh` helper script to run all finetuned versions.

---

# Dependency Installation
```bash
# Install helm
pip install crfm-helm
pip install "crfm-helm[vlm]"

# Core dependencies
pip install torch

# Vision & image processing
pip install opencv-python pillow scipy matplotlib seaborn

# Hugging Face & models
pip install transformers accelerate datasets tokenizers

# Qwen & tokenizers
pip install tiktoken protobuf

# Utilities
pip install requests numpy pandas legacy-cgi
```

These dependencies and configurations need to be installed for VHELM to run.
