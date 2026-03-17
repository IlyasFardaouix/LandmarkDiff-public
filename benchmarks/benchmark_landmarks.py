"""Benchmark landmark extraction speed.

This script generates synthetic test images and measures the time taken to extract landmarks
using the `landmarkdiff.landmarks.extract_landmarks` function.
"""

import argparse
import time
import numpy as np
from PIL import Image

from landmarkdiff.landmarks import extract_landmarks


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_images", type=int, default=100, help="Number of images to benchmark")
    parser.add_argument("--resolution", type=int, default=512, help="Resolution of test images")
    return parser.parse_args()


def generate_test_image(resolution: int) -> np.ndarray:
    """Generate a synthetic test image with random noise.

    Args:
        resolution (int): Resolution of the test image.

    Returns:
        np.ndarray: Synthetic test image.
    """
    # Use random noise images (MediaPipe may not detect faces, but we're measuring speed)
    return np.random.randint(0, 255, (resolution, resolution, 3), dtype=np.uint8)


def benchmark_landmark_extraction(args: argparse.Namespace) -> None:
    """Benchmark landmark extraction speed.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    print(f"Benchmarking landmark extraction ({args.num_images} images, {args.resolution}x{args.resolution})...")

    times = []
    for i in range(args.num_images):
        img = generate_test_image(args.resolution)
        start = time.perf_counter()
        try:
            _ = extract_landmarks(img)
        except Exception as e:
            # Handle any exceptions that occur during landmark extraction
            print(f"Error extracting landmarks: {e}")
            continue
        elapsed = time.perf_counter() - start
        times.append(elapsed)

        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{args.num_images} - avg: {np.mean(times)*1000:.1f}ms/image")

    print(f"\nResults:")
    print(f"  Mean: {np.mean(times)*1000:.1f} ms/image")
    print(f"  Median: {np.median(times)*1000:.1f} ms/image")
    print(f"  Std: {np.std(times)*1000:.1f} ms")
    print(f"  Throughput: {1/np.mean(times):.1f} images/sec")


def main() -> None:
    """Run the benchmark."""
    args = parse_args()
    benchmark_landmark_extraction(args)


if __name__ == "__main__":
    main()