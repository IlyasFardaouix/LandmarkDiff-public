"""Batch inference - process a directory of face images.

This module provides a command-line interface for batch processing of face images
using the LandmarkDiff pipeline.

Author: [Your Name]
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from landmarkdiff.landmarks import extract_landmarks
from landmarkdiff.manipulation import apply_procedure_preset
from landmarkdiff.masking import generate_surgical_mask
from landmarkdiff.synthetic.tps_warp import warp_image_tps
from landmarkdiff.inference import mask_composite, LandmarkDiffPipeline


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Batch LandmarkDiff inference")
    parser.add_argument("input_dir", type=str, help="Directory of input face images")
    parser.add_argument("--procedure", type=str, default="rhinoplasty")
    parser.add_argument("--intensity", type=float, default=60.0,
                        help="Deformation intensity 0-100")
    parser.add_argument("--output", type=str, default="output/batch/",
                        help="Directory to save output images")
    parser.add_argument("--mode", type=str, default="tps",
                        choices=["tps", "controlnet", "img2img", "controlnet_ip"],
                        help="Inference mode")
    return parser.parse_args()


def load_pipeline(args: argparse.Namespace) -> LandmarkDiffPipeline | None:
    """Load diffusion pipeline if needed.

    Args:
        args (argparse.Namespace): Parsed arguments.

    Returns:
        LandmarkDiffPipeline | None: Loaded pipeline or None if not needed.
    """
    if args.mode != "tps":
        from landmarkdiff.inference import LandmarkDiffPipeline
        pipeline = LandmarkDiffPipeline(mode=args.mode, device="cuda")
        pipeline.load()
        return pipeline
    return None


def process_image(img_path: Path, args: argparse.Namespace, pipeline: LandmarkDiffPipeline | None) -> None:
    """Process a single image.

    Args:
        img_path (Path): Path to the image file.
        args (argparse.Namespace): Parsed arguments.
        pipeline (LandmarkDiffPipeline | None): Loaded pipeline or None if not needed.
    """
    try:
        # Load image
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            raise ValueError(f"Could not load {img_path}")

        # Resize image
        img_bgr = cv2.resize(img_bgr, (512, 512))

        # Extract face landmarks
        face = extract_landmarks(img_bgr)
        if face is None:
            raise ValueError("No face detected")

        # Apply procedure (TPS mode)
        if args.mode == "tps":
            manip = apply_procedure_preset(
                face, args.procedure, args.intensity, image_size=512,
            )
            mask = generate_surgical_mask(face, args.procedure, 512, 512)
            warped = warp_image_tps(img_bgr, face.pixel_coords, manip.pixel_coords)
            output = mask_composite(warped, img_bgr, mask)
            cv2.imwrite(str(args.output / f"{img_path.stem}_prediction.png"), output)
        # Apply procedure (other modes)
        else:
            result = pipeline.generate(
                img_bgr,
                procedure=args.procedure,
                intensity=args.intensity,
                num_inference_steps=20,
                seed=42,
            )
            cv2.imwrite(str(args.output / f"{img_path.stem}_prediction.png"), result["output"])

        # Save result
        results.append({"file": img_path.name, "status": "success"})
    except Exception as e:
        print(f"  Failed: {e}")
        results.append({"file": img_path.name, "status": "failed", "error": str(e)})


def main() -> None:
    """Main function."""
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect image files
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in extensions)
    print(f"Found {len(images)} images in {input_dir}")

    # Load pipeline if needed
    pipeline = load_pipeline(args)

    # Process each image
    global results
    results = []
    for i, img_path in enumerate(images):
        print(f"[{i+1}/{len(images)}] Processing {img_path.name}...")
        process_image(img_path, args, pipeline)

    # Summary
    success = sum(1 for r in results if r["status"] == "success")
    print(f"\nDone: {success}/{len(images)} successful")
    print(f"Results saved to {output_dir}/")


if __name__ == "__main__":
    main()