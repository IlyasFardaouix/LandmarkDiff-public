"""Basic inference example - predict surgical outcome for a single image."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from landmarkdiff.landmarks import extract_landmarks
from landmarkdiff.manipulation import apply_procedure_preset
from landmarkdiff.conditioning import render_wireframe
from landmarkdiff.inference import LandmarkDiffPipeline
from landmarkdiff.synthetic.tps_warp import warp_image_tps

def load_image(image_path: str) -> Image:
    """Load an image from a file path."""
    return Image.open(image_path).convert("RGB").resize((512, 512))

def extract_landmarks_from_image(image_array: np.ndarray) -> object:
    """Extract landmarks from an image array."""
    landmarks = extract_landmarks(image_array)
    if landmarks is None:
        raise ValueError("No face detected in image")
    return landmarks

def deform_landmarks(landmarks: object, procedure: str, intensity: float) -> object:
    """Deform landmarks according to a procedure and intensity."""
    return apply_procedure_preset(landmarks, procedure, intensity=intensity)

def render_wireframes(landmarks: object, size: tuple) -> np.ndarray:
    """Render wireframes for original and deformed landmarks."""
    original_mesh = render_wireframe(landmarks, size)
    deformed_mesh = render_wireframe(landmarks, size)
    return original_mesh, deformed_mesh

def save_mesh_visualizations(output_dir: Path, original_mesh: np.ndarray, deformed_mesh: np.ndarray) -> None:
    """Save mesh visualizations to an output directory."""
    import cv2
    cv2.imwrite(str(output_dir / "mesh_original.png"), original_mesh)
    cv2.imwrite(str(output_dir / "mesh_deformed.png"), deformed_mesh)

def generate_prediction(args: argparse.Namespace) -> None:
    """Generate a prediction using a diffusion pipeline or TPS."""
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load image
    img = load_image(args.image)
    img_array = np.array(img)

    # Extract landmarks
    print(f"Extracting landmarks from {args.image}...")
    landmarks = extract_landmarks_from_image(img_array)

    # Deform landmarks
    print(f"Applying {args.procedure} deformation (intensity={args.intensity})...")
    deformed = deform_landmarks(landmarks, args.procedure, args.intensity)

    # Render wireframes
    print(f"Rendering wireframes...")
    original_mesh, deformed_mesh = render_wireframes(landmarks, (512, 512))

    # Save mesh visualizations
    save_mesh_visualizations(output_dir, original_mesh, deformed_mesh)
    print(f"  Saved mesh visualizations to {output_dir}/")

    # Generate prediction
    if args.mode in ("controlnet", "img2img"):
        try:
            print("Loading diffusion pipeline...")
            pipeline = LandmarkDiffPipeline(mode=args.mode, device="cuda")
            pipeline.load()

            print("Generating prediction...")
            result = pipeline.generate(
                img_array,
                procedure=args.procedure,
                intensity=args.intensity,
                mode=args.mode,
            )

            result["output"].save(str(output_dir / "prediction.png"))
            print(f"  Saved prediction to {output_dir}/")

        except Exception as e:
            print(f"  Diffusion pipeline not available: {e}")
            print("  Use --mode tps for CPU-only mode")

    elif args.mode == "tps":
        # TPS warp
        src = landmarks.pixel_coords[:, :2].copy()
        dst = deformed.pixel_coords[:, :2].copy()
        src[:, 0] *= 512 / landmarks.image_width
        src[:, 1] *= 512 / landmarks.image_height
        dst[:, 0] *= 512 / deformed.image_width
        dst[:, 1] *= 512 / deformed.image_height

        warped = warp_image_tps(img_array, src, dst)
        Image.fromarray(warped).save(str(output_dir / "prediction_tps.png"))
        print(f"  Saved TPS prediction to {output_dir}/prediction_tps.png")

    print("Done!")

def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Basic LandmarkDiff inference")
    parser.add_argument("image", type=str, help="Path to input face image")
    parser.add_argument("--procedure", type=str, default="rhinoplasty",
                        choices=["rhinoplasty", "blepharoplasty", "rhytidectomy", "orthognathic"])
    parser.add_argument("--intensity", type=float, default=60.0,
                        help="Deformation intensity (0-100)")
    parser.add_argument("--output", type=str, default="output/",
                        help="Output directory")
    parser.add_argument("--mode", type=str, default="controlnet",
                        choices=["controlnet", "img2img", "tps"],
                        help="Prediction mode")
    args = parser.parse_args()

    generate_prediction(args)

if __name__ == "__main__":
    main()