#!/usr/bin/env python3
"""
Deploy a model using vLLM with configuration from config.yaml

Usage:
    python deploy_vllm.py
    python deploy_vllm.py --config /path/to/config.yaml
"""

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_vllm_command(config: dict) -> list[str]:
    """Build vLLM serve command from config."""
    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        config["model"]["name"],
        "--host",
        config["server"]["host"],
        "--port",
        str(config["server"]["port"]),
        "--dtype",
        config["model"]["dtype"],
        "--max-model-len",
        str(config["model"]["max_model_len"]),
        "--gpu-memory-utilization",
        str(config["gpu"]["memory_utilization"]),
        "--tensor-parallel-size",
        str(config["gpu"]["tensor_parallel_size"]),
    ]

    # Optional: trust remote code
    if config["model"].get("trust_remote_code", False):
        cmd.append("--trust-remote-code")

    # Performance settings
    if config["performance"].get("enforce_eager", False):
        cmd.append("--enforce-eager")

    if config["performance"].get("enable_prefix_caching", True):
        cmd.append("--enable-prefix-caching")

    if config["performance"].get("disable_log_requests", False):
        cmd.append("--disable-log-requests")

    # LoRA settings
    if config["lora"].get("enable", False):
        cmd.append("--enable-lora")
        cmd.extend(["--max-loras", str(config["lora"].get("max_loras", 1))])
        cmd.extend(["--max-lora-rank", str(config["lora"].get("max_lora_rank", 64))])
        if config["lora"].get("adapter_path"):
            cmd.extend(["--lora-modules", f"adapter={config['lora']['adapter_path']}"])

    # Quantization
    if config["quantization"].get("method"):
        cmd.extend(["--quantization", config["quantization"]["method"]])

    return cmd


def main():
    parser = argparse.ArgumentParser(description="Deploy model with vLLM")
    parser.add_argument(
        "--config",
        type=str,
        default=Path(__file__).parent / "config.yaml",
        help="Path to config.yaml file",
    )
    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)

    print(f"Loading config from: {config_path}")
    config = load_config(config_path)

    # Build command
    cmd = build_vllm_command(config)

    # Print command for debugging
    print("\n" + "=" * 60)
    print("Starting vLLM server with command:")
    print("=" * 60)
    print(" \\\n    ".join(cmd))
    print("=" * 60 + "\n")

    # Print server info
    print(f"Model: {config['model']['name']}")
    print(f"Server: http://{config['server']['host']}:{config['server']['port']}")
    print(f"API docs: http://localhost:{config['server']['port']}/docs")
    print("\n" + "=" * 60 + "\n")

    # Run vLLM server
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nServer stopped.")
    except subprocess.CalledProcessError as e:
        print(f"Error: vLLM server failed with exit code {e.returncode}")
        sys.exit(e.returncode)


if __name__ == "__main__":
    main()
