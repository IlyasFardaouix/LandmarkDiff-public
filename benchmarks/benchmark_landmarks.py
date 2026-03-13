```python
"""Benchmark landmark extraction speed."""

import argparse
import time
import numpy as np
from PIL import Image

from landmarkdiff.landmarks import extract_landmarks


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Benchmark landmark extraction speed.")
    parser.add_argument(
        "--num_images",
        type=int,
        default=100,
        help="Number of images to benchmark.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="Resolution of images (width and height).",
    )
    return parser.parse_args()


def create_synthetic_images(num_images: int, resolution: int) -> np.ndarray:
    """
    Create synthetic test images.

    Args:
        num_images (int): Number of images to create.
        resolution (int): Resolution of images (width and height).

    Returns:
        np.ndarray: Array of synthetic images.
    """
    return np.random.randint(
        0, 255, (num_images, resolution, resolution, 3), dtype=np.uint8
    )


def benchmark_landmark_extraction(num_images: int, resolution: int) -> None:
    """
    Benchmark landmark extraction speed.

    Args:
        num_images (int): Number of images to benchmark.
        resolution (int): Resolution of images (width and height).
    """
    print(f"Benchmarking landmark extraction ({num_images} images, {resolution}x{resolution})...")

    times = []
    for i, img in enumerate(create_synthetic_images(num_images, resolution)):
        start = time.perf_counter()
        _ = extract_landmarks(img)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

        if (i + 1) % 10 == 0:
            print(
                f"  {i+1}/{num_images} - avg: {np.mean(times)*1000:.1f}ms/image"
            )

    print(f"\nResults:")
    print(f"  Mean: {np.mean(times)*1000:.1f} ms/image")
    print(f"  Median: {np.median(times)*1000:.1f} ms/image")
    print(f"  Std: {np.std(times)*1000:.1f} ms")
    print(f"  Throughput: {1/np.mean(times):.1f} images/sec")


def main() -> None:
    """
    Main entry point.
    """
    args = parse_args()
    benchmark_landmark_extraction(args.num_images, args.resolution)


if __name__ == "__main__":
    main()
```