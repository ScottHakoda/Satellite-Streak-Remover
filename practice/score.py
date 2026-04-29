import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
from astropy.io import fits

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT / "practice"))

from practice/simulate import create_synthetic_image
from detect_streaks import detect_remove
from metric import (
    evaluate_streak_removal,
    evaluate_star_recovery,
    stars_near_streaks,
    unaffected_star_indices,
)


def save_fits_image(image, path):
    hdu = fits.PrimaryHDU(image.astype(np.float32))
    hdu.writeto(path, overwrite=True)


def run_one_synthetic_case(case_id,
                           shape,
                           sky_level,
                           read_noise,
                           n_stars,
                           star_sigma,
                           streaks,
                           output_dir=None,
                           star_threshold=10.0,
                           aperture=6,
                           annulus_radii=(8, 12),
                           sigma=2.0):
    image, truth_image, metadata = create_synthetic_image(
        shape=shape,
        sky_level=sky_level,
        read_noise=read_noise,
        n_stars=n_stars,
        star_sigma=star_sigma,
        streaks=streaks,
    )

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        prefix = os.path.join(output_dir, f"case_{case_id:03d}")
        save_fits_image(image, f"{prefix}_input.fits")
        save_fits_image(truth_image, f"{prefix}_truth.fits")
        with open(f"{prefix}_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    with tempfile.TemporaryDirectory() as tmpdir:
        input_fits = os.path.join(tmpdir, "synthetic_input.fits")
        save_fits_image(image, input_fits)

        original_image, processed_image, lines, streak_model, cleaned_image = detect_remove(
            input_fits,
            progress_callback=None,
            log_callback=None,
        )

    if cleaned_image is None:
        raise RuntimeError(f"Streak removal failed for case {case_id}")

    streak_score = evaluate_streak_removal(
        cleaned_image,
        image,
        truth_image,
        metadata,
    )

    affected_indices = stars_near_streaks(metadata, threshold=star_threshold)
    unaffected_indices = unaffected_star_indices(metadata, threshold=star_threshold)

    all_recovery = evaluate_star_recovery(
        cleaned_image,
        metadata,
        truth_image=truth_image,
        aperture=aperture,
        annulus_radii=annulus_radii,
        sigma=sigma,
    )

    affected_recovery = evaluate_star_recovery(
        cleaned_image,
        metadata,
        truth_image=truth_image,
        aperture=aperture,
        annulus_radii=annulus_radii,
        sigma=sigma,
        indices=affected_indices,
    ) if affected_indices else float("nan")

    unaffected_recovery = evaluate_star_recovery(
        cleaned_image,
        metadata,
        truth_image=truth_image,
        aperture=aperture,
        annulus_radii=annulus_radii,
        sigma=sigma,
        indices=unaffected_indices,
    )

    if output_dir is not None:
        save_fits_image(streak_model, f"{prefix}_streak_model.fits")
        save_fits_image(cleaned_image, f"{prefix}_cleaned.fits")

    return {
        "case_id": case_id,
        "streak_score": streak_score,
        "all_recovery": all_recovery,
        "affected_recovery": affected_recovery,
        "unaffected_recovery": unaffected_recovery,
        "n_affected": len(affected_indices),
        "n_unaffected": len(unaffected_indices),
        "n_stars": len(metadata["stars"]),
        "metadata": metadata,
    }


def run_pipeline(args):
    scores = []

    streaks = [
        {
            "start": (50, 40),
            "end": (shape_y - 50, shape_x - 40),
            "width": 3.0,
            "flux": 6000,
        }
        for shape_y, shape_x in [args.shape]
    ]
    # allow random streak parameters per case if desired
    if args.random_streaks:
        streaks = None

    for case_id in range(1, args.n_images + 1):
        print(f"Running case {case_id}/{args.n_images}...")
        if args.random_streaks:
            rng = np.random.default_rng(args.seed + case_id if args.seed is not None else None)
            start = (
                int(rng.integers(50, args.shape[0] - 100)),
                int(rng.integers(50, args.shape[1] - 100)),
            )
            end = (
                int(rng.integers(50, args.shape[0] - 100)),
                int(rng.integers(50, args.shape[1] - 100)),
            )
            case_streaks = [{
                "start": start,
                "end": end,
                "width": float(rng.uniform(2.0, 5.0)),
                "flux": float(rng.uniform(3000, 9000)),
            }]
        else:
            case_streaks = streaks

        result = run_one_synthetic_case(
            case_id=case_id,
            shape=args.shape,
            sky_level=args.sky_level,
            read_noise=args.read_noise,
            n_stars=args.n_stars,
            star_sigma=args.star_sigma,
            streaks=case_streaks,
            output_dir=args.output_dir,
            star_threshold=args.star_threshold,
            aperture=args.aperture_radius,
            annulus_radii=(args.annulus_inner, args.annulus_outer),
            sigma=args.star_sigma,
        )
        scores.append(result)

        print(
            f"  streak_score={result['streak_score']:.4f}, "
            f"all_recovery={result['all_recovery']:.4f}, "
            f"affected_recovery={result['affected_recovery']:.4f}, "
            f"unaffected_recovery={result['unaffected_recovery']:.4f}, "
            f"affected={result['n_affected']}/{result['n_stars']}"
        )

    def stats(name):
        values = np.array([r[name] for r in scores], dtype=float)
        return np.nanmean(values), np.nanstd(values)

    streak_mean, streak_std = stats("streak_score")
    all_mean, all_std = stats("all_recovery")
    affected_mean, affected_std = stats("affected_recovery")
    unaffected_mean, unaffected_std = stats("unaffected_recovery")

    print("\n=== Summary ===")
    print(f"Average streak removal score: {streak_mean:.4f} ± {streak_std:.4f}")
    print(f"Average all-star recovery: {all_mean:.4f} ± {all_std:.4f}")
    print(f"Average affected-star recovery: {affected_mean:.4f} ± {affected_std:.4f}")
    print(f"Average unaffected-star recovery: {unaffected_mean:.4f} ± {unaffected_std:.4f}")

    if args.output_dir:
        summary_path = os.path.join(args.output_dir, "pipeline_summary.json")
        with open(summary_path, "w") as f:
            json.dump(
                {
                    "args": vars(args),
                    "scores": scores,
                    "summary": {
                        "streak_mean": streak_mean,
                        "streak_std": streak_std,
                        "all_mean": all_mean,
                        "all_std": all_std,
                        "affected_mean": affected_mean,
                        "affected_std": affected_std,
                        "unaffected_mean": unaffected_mean,
                        "unaffected_std": unaffected_std,
                    },
                },
                f,
                indent=2,
            )
        print(f"Saved summary to {summary_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate synthetic images, run streak removal, and compute average metrics."
    )
    parser.add_argument("--n-images", type=int, default=10, help="Number of synthetic images to evaluate.")
    parser.add_argument("--output-dir", type=str, default="pipeline_results", help="Directory for saved FITS and summary.")
    parser.add_argument("--shape", type=int, nargs=2, default=[512, 512], help="Image shape: height width.")
    parser.add_argument("--sky-level", type=float, default=1000.0, help="Sky background level.")
    parser.add_argument("--read-noise", type=float, default=10.0, help="Read noise sigma.")
    parser.add_argument("--n-stars", type=int, default=150, help="Number of stars per synthetic image.")
    parser.add_argument("--star-sigma", type=float, default=2.0, help="PSF sigma for star injection and recovery.")
    parser.add_argument("--star-threshold", type=float, default=10.0, help="Distance threshold for affected stars.")
    parser.add_argument("--aperture-radius", type=float, default=6.0, help="Aperture radius for photometry.")
    parser.add_argument("--annulus-inner", type=float, default=8.0, help="Inner annulus radius for background.")
    parser.add_argument("--annulus-outer", type=float, default=12.0, help="Outer annulus radius for background.")
    parser.add_argument("--random-streaks", action="store_true", help="Use random streak geometry per image.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible synthetic images.")
    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    run_pipeline(args)


if __name__ == "__main__":
    main()