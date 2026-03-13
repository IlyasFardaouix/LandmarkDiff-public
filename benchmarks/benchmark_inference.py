```python
"""Benchmark inference pipeline speed.

This script measures the inference speed of the LandmarkDiffPipeline in various modes.
"""

import argparse
import time
import numpy as np

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Benchmark inference pipeline speed.")
    parser.add_argument(
        "--num_images",
        type=int,
        default=10,
        help="Number of images to process for benchmarking",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for inference (e.g., 'cuda', 'cpu')",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="controlnet",
        help="Mode to use for inference (e.g., 'controlnet', 'rhinoplasty')",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=30,
        help="Number of steps to perform for each inference",
    )
    return parser.parse_args()

def warm_up(pipeline, device) -> None:
    """Perform a warm-up run to prepare the pipeline for benchmarking.

    Args:
        pipeline (LandmarkDiffPipeline): Pipeline instance to warm up.
        device (str): Device to use for inference.
    """
    print("Warming up...")
    dummy = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    try:
        pipeline.generate(dummy, procedure="rhinoplasty", intensity=0.5)
    except Exception:
        pass

def benchmark(pipeline, num_images, device) -> list[float]:
    """Measure the inference speed of the pipeline.

    Args:
        pipeline (LandmarkDiffPipeline): Pipeline instance to benchmark.
        num_images (int): Number of images to process for benchmarking.
        device (str): Device to use for inference.

    Returns:
        list[float]: List of inference times.
    """
    times = []
    for i in range(num_images):
        img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)

        start = time.perf_counter()
        try:
            pipeline.generate(img, procedure="rhinoplasty", intensity=0.5)
            elapsed = time.perf_counter() - start
            times.append(elapsed)
            print(f"  [{i+1}/{num_images}] {elapsed:.2f}s")
        except Exception as e:
            print(f"  [{i+1}/{num_images}] Failed: {e}")

    return times

def print_results(times: list[float]) -> None:
    """Print the benchmark results.

    Args:
        times (list[float]): List of inference times.
    """
    if times:
        print(f"\nResults:")
        print(f"  Mean: {np.mean(times):.2f}s")
        print(f"  Median: {np.median(times):.2f}s")
        print(f"  Min: {np.min(times):.2f}s")
        print(f"  Max: {np.max(times):.2f}s")

def main() -> None:
    """Main entry point of the script."""
    args = parse_args()

    print(f"Benchmarking inference ({args.mode}, {args.steps} steps, {args.device})...")

    try:
        from landmarkdiff.inference import LandmarkDiffPipeline
        pipeline = LandmarkDiffPipeline(mode=args.mode, device=args.device)
        pipeline.load()
    except Exception as e:
        print(f"Could not load pipeline: {e}")
        print("Make sure you have the required model weights cached")
        return

    warm_up(pipeline, args.device)
    times = benchmark(pipeline, args.num_images, args.device)
    print_results(times)

if __name__ == "__main__":
    main()
```