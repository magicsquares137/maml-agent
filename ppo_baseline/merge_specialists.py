# merge_specialists.py
import argparse
import subprocess
from pathlib import Path

def create_merge_config(model_paths: list, method: str, params: dict, output_dir: Path) -> Path:
    """
    Create mergekit YAML config
    """
    config_path = output_dir / "merge_config.yaml"
    
    # Build sources section
    sources = []
    for model_path in model_paths:
        sources.append(f"""      - model: {model_path}
        layer_range: [0, 32]""")
    
    sources_yaml = "\n".join(sources)
    
    # Calculate weights (equal by default)
    num_models = len(model_paths)
    weights = [round(1.0 / num_models, 2) for _ in range(num_models)]
    weights[-1] = round(1.0 - sum(weights[:-1]), 2)  # Adjust last to sum to 1.0
    
    if method == "ties":
        config = f"""merge_method: ties
slices:
  - sources:
{sources_yaml}
parameters:
  density: {params.get('density', 0.5)}
  weight: {weights}
dtype: bfloat16
"""
    
    elif method == "dare_ties":
        config = f"""merge_method: dare_ties
slices:
  - sources:
{sources_yaml}
parameters:
  density: {params.get('density', 0.9)}
  weight: {weights}
dtype: bfloat16
"""
    
    elif method == "slerp":
        # SLERP only works with 2 models
        if len(model_paths) != 2:
            raise ValueError("SLERP only works with 2 models")
        config = f"""merge_method: slerp
slices:
  - sources:
{sources_yaml}
parameters:
  t: {params.get('t', 0.5)}
dtype: bfloat16
"""
    
    else:
        raise ValueError(f"Unknown merge method: {method}")
    
    with open(config_path, 'w') as f:
        f.write(config)
    
    print(f"   Created config: {config_path}")
    return config_path

def merge_specialists(model_paths: list, method: str, params: dict, output_path: str):
    """
    Merge specialist models using mergekit
    """
    print(f"\n🔀 Merging specialists with {method}")
    print(f"   Models: {len(model_paths)}")
    for path in model_paths:
        print(f"     - {path}")
    print(f"   Output: {output_path}")
    
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create config
    config_path = create_merge_config(model_paths, method, params, output_dir)
    
    # Run mergekit
    cmd = [
        "mergekit-yaml",
        str(config_path),
        str(output_dir),
        "--copy-tokenizer",
        "--allow-crimes",  # Sometimes needed for edge cases
    ]
    
    # Add CUDA if available
    import torch
    if torch.cuda.is_available():
        cmd.append("--cuda")
    
    print(f"\n   Running mergekit...")
    print(f"   Command: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"   ❌ Mergekit failed!")
        print(f"   STDOUT: {result.stdout}")
        print(f"   STDERR: {result.stderr}")
        raise RuntimeError("Mergekit failed")
    
    print(f"✅ Merged model saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Merge specialist models")
    parser.add_argument("--models", nargs="+", required=True,
                       help="Paths to specialist models (merged with base)")
    parser.add_argument("--method", type=str, required=True,
                       choices=["ties", "dare_ties", "slerp"],
                       help="Merge method")
    parser.add_argument("--density", type=float, default=0.5,
                       help="Density parameter for TIES/DARE (0.0-1.0)")
    parser.add_argument("--t", type=float, default=0.5,
                       help="t parameter for SLERP (0.0-1.0)")
    parser.add_argument("--output", type=str, required=True,
                       help="Output path for merged model")
    args = parser.parse_args()
    
    # Verify models exist
    for path in args.models:
        if not Path(path).exists():
            raise FileNotFoundError(f"Model not found: {path}")
    
    # Build params dict
    params = {}
    if args.method in ["ties", "dare_ties"]:
        params["density"] = args.density
    elif args.method == "slerp":
        params["t"] = args.t
    
    merge_specialists(args.models, args.method, params, args.output)

if __name__ == "__main__":
    main()