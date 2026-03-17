"""Benchmark landmark extraction speed.

This script measures the speed of landmark extraction using the `extract_landmarks`
function from the `landmarkdiff.landmarks` module. It generates synthetic test images
with random noise and measures the time taken to extract landmarks from each image.

Author: [Your Name]
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
    parser = argparse.ArgumentParser(description="Benchmark landmark extraction speed.")
    parser.add_argument(
        "--num_images",
        type=int,
        default=100,
        help="Number of images to generate and benchmark.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="Resolution of the generated images (width and height).",
    )
    return parser.parse_args()


def generate_synthetic_image(resolution: int) -> np.ndarray:
    """Generate a synthetic test image with random noise.

    Args:
        resolution (int): Resolution of the generated image (width and height).

    Returns:
        np.ndarray: Synthetic test image with random noise.
    """
    return np.random.randint(0, 255, (resolution, resolution, 3), dtype=np.uint8)


def benchmark_landmark_extraction(
    num_images: int, resolution: int
) -> tuple[float, list[float]]:
    """Benchmark landmark extraction speed.

    Args:
        num_images (int): Number of images to generate and benchmark.
        resolution (int): Resolution of the generated images (width and height).

    Returns:
        tuple[float, list[float]]: Mean time taken to extract landmarks and a list of individual times.
    """
    print(f"Benchmarking landmark extraction ({num_images} images, {resolution}x{resolution})...")

    times = []
    for i in range(num_images):
        img = generate_synthetic_image(resolution)
        start = time.perf_counter()
        try:
            _ = extract_landmarks(img)
        except Exception as e:
            print(f"Error extracting landmarks from image {i+1}: {e}")
            continue
        elapsed = time.perf_counter() - start
        times.append(elapsed)

        if (i + 1) % 10 == 0:
            print(
                f"  {i+1}/{num_images} - avg: {np.mean(times)*1000:.1f}ms/image"
            )

    return np.mean(times), times


def print_results(mean_time: float, times: list[float]) -> None:
    """Print benchmark results.

    Args:
        mean_time (float): Mean time taken to extract landmarks.
        times (list[float]): List of individual times.
    """
    print(f"\nResults:")
    print(f"  Mean: {mean_time*1000:.1f} ms/image")
    print(f"  Median: {np.median(times)*1000:.1f} ms/image")
    print(f"  Std: {np.std(times)*1000:.1f} ms")
    print(f"  Throughput: {1/mean_time:.1f} images/sec")


def main() -> None:
    """Main entry point."""
    args = parse_args()
    mean_time, times = benchmark_landmark_extraction(args.num_images, args.resolution)
    print_results(mean_time, times)


if __name__ == "__main__":
    main()