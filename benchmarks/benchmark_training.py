"""Benchmark training loop throughput.

This script measures the throughput of a synthetic training loop, simulating the
overhead of data loading, tensor operations, and gradient steps. It uses dummy
tensors matching training shapes and computes the mean, median, and throughput
of the training loop.

Author: [Your Name]
"""

import argparse
import time
import numpy as np
import torch
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        None

    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (e.g., 'cuda', 'cpu')")
    parser.add_argument("--num_steps", type=int, default=100, help="Number of steps to run")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size to use")
    return parser.parse_args()


def check_dependencies(args: argparse.Namespace) -> None:
    """Check if required dependencies are installed.

    Args:
        args (argparse.Namespace): Parsed arguments

    Raises:
        ImportError: If a required dependency is missing
    """
    try:
        import torch
        from diffusers import ControlNetModel, StableDiffusionControlNetPipeline
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Install training deps: pip install -e '.[train]'")
        raise


def check_cuda_availability(args: argparse.Namespace) -> None:
    """Check if CUDA is available on the specified device.

    Args:
        args (argparse.Namespace): Parsed arguments

    Raises:
        RuntimeError: If CUDA is not available on the specified device
    """
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available")
        raise RuntimeError("CUDA is not available on the specified device")


def simulate_training_loop(args: argparse.Namespace) -> list[float]:
    """Simulate a training loop and measure its throughput.

    Args:
        args (argparse.Namespace): Parsed arguments

    Returns:
        list[float]: List of step times in seconds
    """
    device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    print("Running synthetic training loop...")

    # dummy tensors matching training shapes
    latent_shape = (args.batch_size, 4, 64, 64)
    cond_shape = (args.batch_size, 3, 512, 512)

    step_times = []
    for step in range(args.num_steps):
        start = time.perf_counter()

        # simulate forward pass tensors
        latents = torch.randn(latent_shape, device=device, dtype=dtype)
        cond = torch.randn(cond_shape, device=device, dtype=dtype)
        noise = torch.randn_like(latents)

        # simulate loss computation
        loss = torch.nn.functional.mse_loss(latents + noise, latents)
        loss.backward()

        if device.type == "cuda":
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start
        step_times.append(elapsed)

        if (step + 1) % 20 == 0:
            print(f"  Step {step+1}/{args.num_steps} - {elapsed*1000:.1f}ms/step")

    return step_times


def print_results(step_times: list[float], args: argparse.Namespace) -> None:
    """Print the results of the training loop simulation.

    Args:
        step_times (list[float]): List of step times in seconds
        args (argparse.Namespace): Parsed arguments
    """
    print(f"\nResults (batch_size={args.batch_size}):")
    print(f"  Mean: {np.mean(step_times)*1000:.1f} ms/step")
    print(f"  Median: {np.median(step_times)*1000:.1f} ms/step")
    print(f"  Throughput: {1/np.mean(step_times):.1f} steps/sec")
    print(f"  Throughput: {args.batch_size/np.mean(step_times):.1f} images/sec")


def main() -> None:
    """Main entry point of the script."""
    args = parse_args()
    check_dependencies(args)
    check_cuda_availability(args)

    step_times = simulate_training_loop(args)
    print_results(step_times, args)


if __name__ == "__main__":
    main()