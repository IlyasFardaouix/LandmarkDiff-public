"""Basic inference example - predict surgical outcome for a single image.

This script uses LandmarkDiff to predict the outcome of a surgical procedure
based on a single input image. It supports various modes, including GPU-accelerated
diffusion and CPU-only TPS warping.

Author: [Your Name]
"""

import argparse
from pathlib import Path
import numpy as np
from PIL import Image

from landmarkdiff.landmarks import extract_landmarks
from landmarkdiff.manipulation import apply_procedure_preset
from landmarkdiff.conditioning import render_wireframe
from landmarkdiff.inference import LandmarkDiffPipeline
from landmarkdiff.synthetic.tps_warp import warp_image_tps
import cv2


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Basic LandmarkDiff inference")
    parser.add_argument("image", type=str, help="Path to input face image")
    parser.add_argument("--procedure", type=str, default="rhinoplasty",
                        choices=["rhinoplasty", "blepharoplasty", "rhytidectomy", "orthognathic"],
                        help="Surgical procedure to simulate")
    parser.add_argument("--intensity", type=float, default=60.0,
                        help="Deformation intensity (0-100)")
    parser.add_argument("--output", type=str, default="output/",
                        help="Output directory")
    parser.add_argument("--mode", type=str, default="controlnet",
                        choices=["controlnet", "img2img", "tps"],
                        help="Inference mode")
    return parser.parse_args()


def load_image(image_path: str) -> Image:
    """Load an image from a file path."""
    img = Image.open(image_path).convert("RGB").resize((512, 512))
    return img


def extract_landmarks(img_array: np.ndarray) -> object:
    """Extract landmarks from an image array."""
    landmarks = extract_landmarks(img_array)
    if landmarks is None:
        print("No face detected in image")
        return None
    return landmarks


def deform_landmarks(landmarks: object, procedure: str, intensity: float) -> object:
    """Deform landmarks based on a procedure and intensity."""
    deformed = apply_procedure_preset(landmarks, procedure, intensity=intensity)
    return deformed


def visualize_mesh(landmarks: object, size: tuple) -> np.ndarray:
    """Visualize a mesh based on landmarks and size."""
    original_mesh = render_wireframe(landmarks, size)
    deformed_mesh = render_wireframe(landmarks, size)
    return original_mesh, deformed_mesh


def save_meshes(original_mesh: np.ndarray, deformed_mesh: np.ndarray, output_dir: Path) -> None:
    """Save mesh visualizations to a directory."""
    import cv2
    cv2.imwrite(str(output_dir / "mesh_original.png"), original_mesh)
    cv2.imwrite(str(output_dir / "mesh_deformed.png"), deformed_mesh)
    print(f"  Saved mesh visualizations to {output_dir}/")


def predict_surgical_outcome(img_array: np.ndarray, procedure: str, intensity: float, mode: str, output_dir: Path) -> None:
    """Predict the surgical outcome based on an image array, procedure, intensity, and mode."""
    if mode in ("controlnet", "img2img"):
        try:
            pipeline = LandmarkDiffPipeline(mode=mode, device="cuda")
            pipeline.load()
            result = pipeline.generate(
                img_array,
                procedure=procedure,
                intensity=intensity,
                mode=mode,
            )
            result["output"].save(str(output_dir / "prediction.png"))
            print(f"  Saved prediction to {output_dir}/")
        except Exception as e:
            print(f"  Diffusion pipeline not available: {e}")
            print("  Use --mode tps for CPU-only mode")
    elif mode == "tps":
        src = landmarks.pixel_coords[:, :2].copy()
        dst = deformed.pixel_coords[:, :2].copy()
        src[:, 0] *= 512 / landmarks.image_width
        src[:, 1] *= 512 / landmarks.image_height
        dst[:, 0] *= 512 / deformed.image_width
        dst[:, 1] *= 512 / deformed.image_height

        warped = warp_image_tps(img_array, src, dst)
        Image.fromarray(warped).save(str(output_dir / "prediction_tps.png"))
        print(f"  Saved TPS prediction to {output_dir}/prediction_tps.png")


def main() -> None:
    """Main entry point."""
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    img = load_image(args.image)
    img_array = np.array(img)

    landmarks = extract_landmarks(img_array)
    if landmarks is None:
        return

    print(f"  Detected {len(landmarks.landmarks)} landmarks")

    deformed = deform_landmarks(landmarks, args.procedure, args.intensity)

    original_mesh, deformed_mesh = visualize_mesh(landmarks, (512, 512))
    save_meshes(original_mesh, deformed_mesh, output_dir)

    predict_surgical_outcome(img_array, args.procedure, args.intensity, args.mode, output_dir)

    print("Done!")


if __name__ == "__main__":
    main()