# merge_specialists.py
import argparse
import subprocess
from pathlib import Path

def create_merge_config(model_paths: list, method: str, params: dict, output_dir: Path,
                        base_model: str = "Qwen/Qwen3-8B") -> Path:
    """
    Create mergekit YAML config using the `models:` schema (whole-model merge —
    no hardcoded layer_range, so it's correct for any architecture incl. Qwen3-8B's
    36 layers). TIES/DARE are task-vector methods and REQUIRE base_model to compute
    deltas (specialist - base); the old slices/layer_range config omitted it.
    """
    config_path = output_dir / "merge_config.yaml"
    num_models = len(model_paths)
    weights = [round(1.0 / num_models, 4) for _ in range(num_models)]

    if method in ("ties", "dare_ties"):
        density = params.get("density", 0.5 if method == "ties" else 0.9)
        models_yaml = "\n".join(
            f"  - model: {mp}\n    parameters:\n      weight: {w}\n      density: {density}"
            for mp, w in zip(model_paths, weights)
        )
        config = f"""merge_method: {method}
base_model: {base_model}
models:
{models_yaml}
dtype: bfloat16
"""
    elif method == "slerp":
        # SLERP interpolates base_model with ONE other model by factor t.
        if num_models != 2:
            raise ValueError("SLERP only works with 2 models")
        config = f"""merge_method: slerp
base_model: {model_paths[0]}
models:
  - model: {model_paths[1]}
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

def merge_specialists(model_paths: list, method: str, params: dict, output_path: str,
                      base_model: str = "Qwen/Qwen3-8B",
                      mergekit_bin: str = "mergekit-yaml", use_cuda: bool = True):
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
    config_path = create_merge_config(model_paths, method, params, output_dir, base_model=base_model)

    # Run mergekit (mergekit lives in its OWN venv due to vLLM conflicts; point
    # mergekit_bin at that venv's mergekit-yaml, e.g. mergekit_env/bin/mergekit-yaml).
    cmd = [
        mergekit_bin,
        str(config_path),
        str(output_dir),
        "--copy-tokenizer",
        "--allow-crimes",  # Sometimes needed for edge cases
    ]
    if use_cuda:
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
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen3-8B",
                       help="Base model for task-vector methods (TIES/DARE)")
    parser.add_argument("--mergekit-bin", type=str, default="mergekit-yaml",
                       help="Path to mergekit-yaml (its own venv; avoids vLLM conflicts)")
    parser.add_argument("--no-cuda", action="store_true", help="Merge on CPU")
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
    
    merge_specialists(args.models, args.method, params, args.output,
                      base_model=args.base_model, mergekit_bin=args.mergekit_bin,
                      use_cuda=not args.no_cuda)

if __name__ == "__main__":
    main()