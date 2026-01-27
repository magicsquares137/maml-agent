#!/usr/bin/env python3
"""
Run mergekit merges and push to HuggingFace

Usage:
    python run_merges.py --all                    # Run all merges
    python run_merges.py --method linear          # Run specific merge
    python run_merges.py --method slerp ties      # Run multiple
"""

import argparse
import subprocess
import os
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv() 

HF_TOKEN = os.getenv("HF_TOKEN")  
# =============================================================================
# CONFIGURATION
# =============================================================================

HF_ORG = "arkitekt-ai"
CONFIG_DIR = Path("merge_configs")
OUTPUT_BASE = Path("merged_models")
BASE_NAME = "qwen-coder-7b-specialists"

MERGES = {
    "linear": {
        "config": "linear.yaml",
        "description": "Simple weighted average"
    },
    "slerp": {
        "config": "slerp.yaml",
        "description": "Spherical linear interpolation"
    },
    "ties": {
        "config": "ties.yaml",
        "description": "Task arithmetic with sign consensus"
    },
    "dare_ties": {
        "config": "dare_ties.yaml",
        "description": "DARE with TIES sign election"
    }
}

# =============================================================================
# FUNCTIONS
# =============================================================================

def setup():
    """Install mergekit and login to HuggingFace"""
    print("🔧 Installing mergekit...")
    subprocess.run(["pip", "install", "-q", "mergekit"], check=True)
    
    # print("🔐 Logging in to HuggingFace...")
    # subprocess.run(
    #     ["huggingface-cli", "login", "--token", HF_TOKEN],
    #     check=True
    # )
    
    OUTPUT_BASE.mkdir(exist_ok=True)


def run_merge(method: str, push: bool = True):
    """Run a single merge"""
    if method not in MERGES:
        print(f"❌ Unknown method: {method}")
        print(f"   Available: {list(MERGES.keys())}")
        return False
    
    merge_info = MERGES[method]
    config_path = CONFIG_DIR / merge_info["config"]
    output_name = f"{BASE_NAME}-{method.replace('_', '-')}"
    output_dir = OUTPUT_BASE / output_name
    hf_repo = f"{HF_ORG}/{output_name}"
    
    print(f"\n{'='*60}")
    print(f"🔀 Merge: {method}")
    print(f"   {merge_info['description']}")
    print(f"   Config: {config_path}")
    print(f"   Output: {output_dir}")
    print(f"{'='*60}\n")
    
    # Run mergekit
    cmd = [
        "mergekit-yaml",
        str(config_path),
        str(output_dir),
        "--cuda",
        "--lazy-unpickle",
        "--allow-crimes"
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"✅ Merge complete: {output_dir}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Merge failed: {e}")
        return False
    
    # Push to HuggingFace
    if push:
        print(f"\n🚀 Pushing to HuggingFace: {hf_repo}")
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=HF_TOKEN)
            api.create_repo(hf_repo, exist_ok=True)
            api.upload_folder(
                folder_path=str(output_dir),
                repo_id=hf_repo,
                commit_message=f"Merged with mergekit using {method} method"
            )
            print(f"✅ Pushed to https://huggingface.co/{hf_repo}")
        except Exception as e:
            print(f"❌ Push failed: {e}")
            return False
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Run mergekit merges")
    parser.add_argument("--all", action="store_true", help="Run all merges")
    parser.add_argument("--method", nargs="+", choices=list(MERGES.keys()),
                       help="Specific merge method(s) to run")
    parser.add_argument("--no-push", action="store_true", 
                       help="Don't push to HuggingFace")
    parser.add_argument("--list", action="store_true",
                       help="List available merge methods")
    args = parser.parse_args()
    
    if args.list:
        print("\nAvailable merge methods:")
        for name, info in MERGES.items():
            print(f"  {name}: {info['description']}")
        return
    
    if not args.all and not args.method:
        parser.print_help()
        return
    
    setup()
    
    methods = list(MERGES.keys()) if args.all else args.method
    results = []
    
    for method in methods:
        success = run_merge(method, push=not args.no_push)
        results.append((method, success))
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for method, success in results:
        status = "✅" if success else "❌"
        print(f"  {status} {method}")
    
    if not args.no_push:
        print(f"\nModels available at:")
        for method, success in results:
            if success:
                output_name = f"{BASE_NAME}-{method.replace('_', '-')}"
                print(f"  https://huggingface.co/{HF_ORG}/{output_name}")


if __name__ == "__main__":
    main()