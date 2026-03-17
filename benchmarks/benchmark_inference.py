"""Benchmark inference pipeline speed.

This module provides a command-line interface to benchmark the speed of the
inference pipeline for different modes and devices.
"""

import argparse
import time
import numpy as np

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_images", type=int, default=10, help="Number of images to benchmark")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for inference")
    parser.add_argument("--mode", type=str, default="controlnet", help="Inference pipeline mode")
    parser.add_argument("--steps", type=int, default=30, help="Number of steps for the pipeline")
    return parser.parse_args()

def load_pipeline(args: argparse.Namespace) -> None:
    """Load the inference pipeline with the given mode and device."""
    try:
        from landmarkdiff.inference import LandmarkDiffPipeline
        pipeline = LandmarkDiffPipeline(mode=args.mode, device=args.device)
        pipeline.load()
    except ImportError as e:
        print(f"Could not load pipeline: {e}")
        print("Make sure you have the required model weights cached")
        raise
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise

def warm_up(pipeline: LandmarkDiffPipeline) -> None:
    """Perform a warm-up run to initialize the pipeline."""
    dummy = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    try:
        pipeline.generate(dummy, procedure="rhinoplasty", intensity=0.5)
    except Exception:
        pass

def benchmark(pipeline: LandmarkDiffPipeline, num_images: int) -> list[float]:
    """Benchmark the inference pipeline for the given number of images."""
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
    """Print the benchmark results."""
    if times:
        print(f"\nResults ({args.mode} mode, {args.steps} steps):")
        print(f"  Mean: {np.mean(times):.2f}s")
        print(f"  Median: {np.median(times):.2f}s")
        print(f"  Min: {np.min(times):.2f}s")
        print(f"  Max: {np.max(times):.2f}s")

def main() -> None:
    """Run the benchmark."""
    global args
    args = parse_args()
    print(f"Benchmarking inference ({args.mode}, {args.steps} steps, {args.device})...")

    load_pipeline(args)
    warm_up(LandmarkDiffPipeline(mode=args.mode, device=args.device))
    times = benchmark(LandmarkDiffPipeline(mode=args.mode, device=args.device), args.num_images)
    print_results(times)

if __name__ == "__main__":
    main()