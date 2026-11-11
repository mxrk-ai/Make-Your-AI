"""Download a private checkpoint repo (zoo or generator) to a local folder.
   python scripts/hf_pull.py make-your-ai-zoo-v1 runs/zoo --dataset
"""
import os
import sys

from huggingface_hub import HfApi, snapshot_download

from myai.agent.llm import load_env

load_env()
api = HfApi(token=os.environ["HF_TOKEN"])
repo = sys.argv[1] if "/" in sys.argv[1] else f"{api.whoami()['name']}/{sys.argv[1]}"
path = snapshot_download(repo, repo_type="dataset" if "--dataset" in sys.argv else "model", local_dir=sys.argv[2],
                         token=os.environ["HF_TOKEN"])
print("downloaded to", path)
