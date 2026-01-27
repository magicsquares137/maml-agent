#!/bin/bash
# =============================================================================
# Run all mergekit merges and push to HuggingFace
# =============================================================================

set -e

HF_TOKEN="${HF_TOKEN:-YOUR_HF_TOKEN_HERE}"  # Set via env or replace here
HF_ORG="arkitekt-ai"
CONFIG_DIR="mergekit_configs"
OUTPUT_BASE="merged_models"

# Model naming
BASE_NAME="qwen-coder-7b-specialists"

# =============================================================================
# Setup
# =============================================================================

echo "🔧 Installing mergekit..."
pip install -q mergekit

echo "🔐 Logging in to HuggingFace..."
huggingface-cli login --token "$HF_TOKEN"

mkdir -p "$OUTPUT_BASE"

# =============================================================================
# Merge Functions
# =============================================================================

run_merge() {
    local config=$1
    local name=$2
    local output_dir="$OUTPUT_BASE/$name"
    local hf_repo="$HF_ORG/$name"
    
    echo ""
    echo "============================================================"
    echo "🔀 Running merge: $name"
    echo "   Config: $config"
    echo "   Output: $output_dir"
    echo "============================================================"
    
    # Run merge
    mergekit-yaml "$CONFIG_DIR/$config" "$output_dir" \
        --cuda \
        --lazy-unpickle \
        --allow-crimes
    
    echo "✅ Merge complete: $output_dir"
    
    # Push to HuggingFace
    echo "🚀 Pushing to HuggingFace: $hf_repo"
    huggingface-cli upload "$hf_repo" "$output_dir" \
        --commit-message "Merged with mergekit using ${config%.yaml} method"
    
    echo "✅ Pushed to https://huggingface.co/$hf_repo"
}

# =============================================================================
# Run All Merges
# =============================================================================

echo ""
echo "Starting merges..."
echo ""

# Linear merge
run_merge "linear.yaml" "${BASE_NAME}-linear"

# SLERP merge
run_merge "slerp.yaml" "${BASE_NAME}-slerp"

# TIES merge
run_merge "ties.yaml" "${BASE_NAME}-ties"

# DARE-TIES merge
run_merge "dare_ties.yaml" "${BASE_NAME}-dare-ties"

# =============================================================================
# Summary
# =============================================================================

echo ""
echo "============================================================"
echo "🎉 ALL MERGES COMPLETE!"
echo "============================================================"
echo ""
echo "Models pushed to:"
echo "  - https://huggingface.co/$HF_ORG/${BASE_NAME}-linear"
echo "  - https://huggingface.co/$HF_ORG/${BASE_NAME}-slerp"
echo "  - https://huggingface.co/$HF_ORG/${BASE_NAME}-ties"
echo "  - https://huggingface.co/$HF_ORG/${BASE_NAME}-dare-ties"
echo ""