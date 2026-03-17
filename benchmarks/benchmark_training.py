"""Benchmark training loop throughput.

This script measures the throughput of a synthetic training loop, simulating
the overhead of data loading, tensor operations, and gradient steps. It uses
dummy tensors matching training shapes and measures the time taken for each
step.

Example:
    python benchmark_training_loop.py --device cuda --num_steps 100 --batch_size 4
"""

import argparse
import time
import numpy as np
import torch
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to use (e.g., cuda, cpu)"
    )
    parser.add_argument(
        "--num_steps", type=int, default=100, help="Number of training steps"
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Batch size for training"
    )
    return parser.parse_args()


def check_dependencies() -> None:
    """Check if required dependencies are installed."""
    try:
        import torch
        from diffusers import ControlNetModel, StableDiffusionControlNetPipeline
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Install training deps: pip install -e '.[train]'")
        exit(1)


def validate_device(device: str) -> torch.device:
    """Validate the device and return a torch device object."""
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available")
        exit(1)
    return torch.device(device)


def get_dtype(device: torch.device) -> torch.dtype:
    """Get the data type based on the device."""
    return torch.bfloat16 if device.type == "cuda" else torch.float32


def simulate_training_loop(
    num_steps: int, batch_size: int, device: torch.device, dtype: torch.dtype
) -> list[float]:
    """Simulate a training loop and measure the time taken for each step."""
    step_times = []
    for step in range(num_steps):
        start = time.perf_counter()

        # Simulate forward pass tensors
        latents = torch.randn((batch_size, 4, 64, 64), device=device, dtype=dtype)
        cond = torch.randn((batch_size, 3, 512, 512), device=device, dtype=dtype)
        noise = torch.randn_like(latents)

        # Simulate loss computation
        loss = torch.nn.functional.mse_loss(latents + noise, latents)
        loss.backward()

        if device.type == "cuda":
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start
        step_times.append(elapsed)

        if (step + 1) % 20 == 0:
            print(f"  Step {step+1}/{num_steps} - {elapsed*1000:.1f}ms/step")

    return step_times


def print_results(step_times: list[float], batch_size: int) -> None:
    """Print the results of the benchmark."""
    mean_time = np.mean(step_times) * 1000
    median_time = np.median(step_times) * 1000
    throughput = 1 / np.mean(step_times)
    images_per_sec = batch_size / np.mean(step_times)

    print(f"\nResults (batch_size={batch_size}):")
    print(f"  Mean: {mean_time:.1f} ms/step")
    print(f"  Median: {median_time:.1f} ms/step")
    print(f"  Throughput: {throughput:.1f} steps/sec")
    print(f"  Throughput: {images_per_sec:.1f} images/sec")


def main() -> None:
    """Main entry point of the script."""
    args = parse_args()
    device = validate_device(args.device)
    dtype = get_dtype(device)

    print(f"Benchmarking training ({args.num_steps} steps, batch {args.batch_size}, {args.device})...")

    check_dependencies()

    step_times = simulate_training_loop(args.num_steps, args.batch_size, device, dtype)
    print_results(step_times, args.batch_size)


if __name__ == "__main__":
    main()