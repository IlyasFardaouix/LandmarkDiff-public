"""Compare all procedures side-by-side on a single face.

This module provides a command-line tool to visualize the effects of different
procedures on a face image.
"""

import argparse
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw

from landmarkdiff.landmarks import extract_landmarks
from landmarkdiff.manipulation import apply_procedure_preset, PROCEDURE_LANDMARKS
from landmarkdiff.conditioning import render_wireframe


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Compare all procedures")
    parser.add_argument("image", type=str, help="Path to input face image")
    parser.add_argument("--intensity", type=float, default=60.0,
                        help="Deformation intensity (0-100)")
    parser.add_argument("--output", type=str, default="output/comparison.png")
    return parser.parse_args()


def load_image(image_path: str) -> Image:
    """Load an image from a file.

    Args:
        image_path (str): Path to the image file.

    Returns:
        Image: Loaded image.
    """
    return Image.open(image_path).convert("RGB").resize((512, 512))


def extract_landmarks_from_image(img_array: np.ndarray) -> list:
    """Extract landmarks from an image.

    Args:
        img_array (np.ndarray): Image array.

    Returns:
        list: Extracted landmarks.
    """
    landmarks = extract_landmarks(img_array)
    if landmarks is None:
        print("No face detected")
        return []
    return landmarks


def render_comparison_grid(meshes: list) -> Image:
    """Render a comparison grid from a list of meshes.

    Args:
        meshes (list): List of meshes.

    Returns:
        Image: Rendered comparison grid.
    """
    n = len(meshes)
    grid = Image.new("L", (512 * n, 512 + 40), 0)
    draw = ImageDraw.Draw(grid)

    for i, (name, mesh) in enumerate(meshes):
        grid.paste(mesh, (512 * i, 40))
        draw.text((512 * i + 200, 10), name, fill=255)

    return grid


def save_grid(grid: Image, output_path: Path) -> None:
    """Save a grid image to a file.

    Args:
        grid (Image): Grid image.
        output_path (Path): Output file path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(str(output_path))
    print(f"Saved comparison grid to {output_path}")


def main() -> None:
    """Main entry point."""
    args = parse_args()
    img_array = np.array(load_image(args.image))
    landmarks = extract_landmarks_from_image(img_array)

    if not landmarks:
        return

    procedures = list(PROCEDURE_LANDMARKS.keys())
    meshes = []

    # original
    original_mesh = render_wireframe(landmarks, (512, 512))
    meshes.append(("Original", Image.fromarray(original_mesh)))

    # each procedure
    for proc in procedures:
        try:
            deformed = apply_procedure_preset(landmarks, proc, intensity=args.intensity)
            mesh = render_wireframe(deformed, (512, 512))
            meshes.append((proc.capitalize(), Image.fromarray(mesh)))
        except ValueError as e:
            print(f"Error applying procedure {proc}: {e}")

    # create grid
    grid = render_comparison_grid(meshes)
    save_grid(grid, Path(args.output))


if __name__ == "__main__":
    main()