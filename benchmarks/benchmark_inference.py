"""Benchmark inference pipeline speed.

This module provides a command-line interface to benchmark the speed of the
inference pipeline.
"""

import argparse
import time
import numpy as np


class InferenceBenchmark:
    """Benchmark the inference pipeline speed."""

    def __init__(self, num_images: int = 10, device: str = "cuda", mode: str = "controlnet", steps: int = 30):
        """Initialize the benchmark with the given parameters.

        Args:
            num_images: The number of images to benchmark (default: 10).
            device: The device to use for inference (default: "cuda").
            mode: The inference mode (default: "controlnet").
            steps: The number of steps in the inference procedure (default: 30).
        """
        self.num_images = num_images
        self.device = device
        self.mode = mode
        self.steps = steps

    def _warm_up(self):
        """Warm up the pipeline by running a dummy inference."""
        dummy = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        try:
            from landmarkdiff.inference import LandmarkDiffPipeline
            pipeline = LandmarkDiffPipeline(mode=self.mode, device=self.device)
            pipeline.load()
            pipeline.generate(dummy, procedure="rhinoplasty", intensity=0.5)
        except Exception as e:
            print(f"Could not warm up pipeline: {e}")
            print("Make sure you have the required model weights cached")

    def _benchmark(self):
        """Run the benchmark and collect the results."""
        times = []
        for i in range(self.num_images):
            img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
            start = time.perf_counter()
            try:
                from landmarkdiff.inference import LandmarkDiffPipeline
                pipeline = LandmarkDiffPipeline(mode=self.mode, device=self.device)
                pipeline.load()
                pipeline.generate(img, procedure="rhinoplasty", intensity=0.5)
                elapsed = time.perf_counter() - start
                times.append(elapsed)
                print(f"  [{i+1}/{self.num_images}] {elapsed:.2f}s")
            except Exception as e:
                print(f"  [{i+1}/{self.num_images}] Failed: {e}")
        return times

    def run(self):
        """Run the benchmark and print the results."""
        self._warm_up()
        times = self._benchmark()
        if times:
            print(f"\nResults ({self.mode} mode, {self.steps} steps):")
            print(f"  Mean: {np.mean(times):.2f}s")
            print(f"  Median: {np.median(times):.2f}s")
            print(f"  Min: {np.min(times):.2f}s")
            print(f"  Max: {np.max(times):.2f}s")


def main():
    """Parse the command-line arguments and run the benchmark."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_images", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--mode", type=str, default="controlnet")
    parser.add_argument("--steps", type=int, default=30)
    args = parser.parse_args()

    print(f"Benchmarking inference ({args.mode}, {args.steps} steps, {args.device})...")

    benchmark = InferenceBenchmark(num_images=args.num_images, device=args.device, mode=args.mode, steps=args.steps)
    benchmark.run()


if __name__ == "__main__":
    main()