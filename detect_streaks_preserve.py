#!/usr/bin/env python3
"""
Program to detect streaks in FITS images using Hough transform.
"""

from astropy.io import fits
import matplotlib.pyplot as plt
import cv2
import numpy as np
import argparse
from scipy.optimize import curve_fit
from scipy.ndimage import median_filter


def load_fits_image(fits_file):
    """Load image data from FITS file."""
    with fits.open(fits_file) as hdul:
        image_data = hdul[0].data
    return image_data


def preprocess_image(image_data, threshold_percentile=95):
    """Preprocess the image for Hough transform."""
    # Normalize to 0-255 range
    image_norm = (image_data - np.min(image_data)) / (np.max(image_data) - np.min(image_data)) * 255
    image_uint8 = image_norm.astype(np.uint8)

    # Apply threshold to find bright pixels (potential streaks)
    threshold_value = np.percentile(image_uint8, threshold_percentile)
    _, thresh = cv2.threshold(image_uint8, threshold_value, 255, cv2.THRESH_BINARY)

    return thresh


def detect_streaks(image, min_line_length=50, max_line_gap=10):
    """Detect streaks using Hough line transform."""
    lines = cv2.HoughLinesP(image, 1, np.pi/180, threshold=50,
                            minLineLength=min_line_length, maxLineGap=max_line_gap)
    return lines


def merge_duplicate_lines(lines, angle_threshold=2.0, distance_threshold=10.0):
    """
    Merge duplicate/near-duplicate line detections into unique streaks.

    Lines that have similar angles and whose midpoints are close together
    (perpendicular to the line direction) are grouped and averaged.

    Parameters:
    - lines: Array of detected lines from HoughLinesP, shape (N, 1, 4)
    - angle_threshold: Maximum angle difference (degrees) to consider lines as duplicates
    - distance_threshold: Maximum perpendicular distance (pixels) between line midpoints

    Returns:
    - merged: List of unique lines as [[x1, y1, x2, y2], ...]
    """
    if lines is None or len(lines) == 0:
        return []

    # Extract line parameters
    line_params = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180  # Normalize to [0, 180)
        mid_x = (x1 + x2) / 2.0
        mid_y = (y1 + y2) / 2.0
        length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        line_params.append({
            'coords': [x1, y1, x2, y2],
            'angle': angle,
            'mid': (mid_x, mid_y),
            'length': length,
            'assigned': False
        })

    groups = []
    for i, lp in enumerate(line_params):
        if lp['assigned']:
            continue
        group = [i]
        lp['assigned'] = True

        # Direction unit vector for this line (used for perpendicular distance)
        angle_rad = np.radians(lp['angle'])
        dir_x, dir_y = np.cos(angle_rad), np.sin(angle_rad)
        # Perpendicular direction
        perp_x, perp_y = -dir_y, dir_x

        for j, other in enumerate(line_params):
            if other['assigned']:
                continue

            # Check angle similarity
            angle_diff = abs(lp['angle'] - other['angle'])
            angle_diff = min(angle_diff, 180 - angle_diff)  # Handle wrapping
            if angle_diff > angle_threshold:
                continue

            # Check perpendicular distance between midpoints
            dmx = other['mid'][0] - lp['mid'][0]
            dmy = other['mid'][1] - lp['mid'][1]
            perp_dist = abs(dmx * perp_x + dmy * perp_y)
            if perp_dist > distance_threshold:
                continue

            group.append(j)
            other['assigned'] = True

        groups.append(group)

    # For each group, compute the merged line by taking the extreme endpoints
    merged = []
    for group in groups:
        all_points = []
        angles = []
        for idx in group:
            x1, y1, x2, y2 = line_params[idx]['coords']
            all_points.append((x1, y1))
            all_points.append((x2, y2))
            angles.append(line_params[idx]['angle'])

        # Average angle for projection
        avg_angle = np.mean(angles)
        angle_rad = np.radians(avg_angle)
        dir_x, dir_y = np.cos(angle_rad), np.sin(angle_rad)

        # Project all points onto the average direction
        projections = []
        for px, py in all_points:
            proj = px * dir_x + py * dir_y
            projections.append((proj, px, py))

        projections.sort(key=lambda p: p[0])
        _, sx, sy = projections[0]
        _, ex, ey = projections[-1]

        merged.append(np.array([[int(round(sx)), int(round(sy)),
                                  int(round(ex)), int(round(ey))]]))

    print(f"  Merged {len(lines)} raw detections into {len(merged)} unique streaks")
    return merged


def estimate_streak_profile(image_data, line, profile_half_width=15, n_samples=100):
    """
    Model a streak by fitting its perpendicular cross-section at many points
    along its length, using sigma-clipping to exclude sources.

    The approach:
    1. First pass: sample many cross-sections to establish the streak's
       global amplitude and width (sigma), using robust statistics.
    2. Second pass: at each cross-section, build the streak model using
       the global parameters, only allowing small local variations.
       Cap the amplitude so bright sources can't inflate it.

    Returns a 2D array containing ONLY the streak flux (excess above background).
    """
    if isinstance(line, np.ndarray) and line.ndim > 1:
        x1, y1, x2, y2 = line[0]
    else:
        x1, y1, x2, y2 = line

    length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    if length == 0:
        return np.zeros_like(image_data, dtype=float)

    # Unit vectors along and perpendicular to the streak
    dx, dy = (x2 - x1) / length, (y2 - y1) / length
    perp_dx, perp_dy = -dy, dx

    h, w = image_data.shape
    streak_model = np.zeros_like(image_data, dtype=float)
    weight_map = np.zeros_like(image_data, dtype=float)

    # Scale n_samples to streak length
    n_samples = max(n_samples, int(length))

    t_values = np.linspace(0, 1, n_samples)

    def gaussian(x, amplitude, center, sigma):
        return amplitude * np.exp(-(x - center)**2 / (2 * sigma**2))

    # =========================================================================
    # FIRST PASS: Estimate global streak parameters (amplitude, sigma)
    # Sample many cross-sections and collect fitted amplitudes/sigmas,
    # then take robust (median) estimates.
    # =========================================================================
    trial_sigmas = []
    trial_amps = []
    trial_baselines = []

    for t in np.linspace(0.05, 0.95, min(50, n_samples)):
        cx = x1 + t * (x2 - x1)
        cy = y1 + t * (y2 - y1)

        offsets = np.arange(-profile_half_width, profile_half_width + 1)
        values = []
        offs_list = []
        for off in offsets:
            px = int(round(cx + off * perp_dx))
            py = int(round(cy + off * perp_dy))
            if 0 <= px < w and 0 <= py < h:
                values.append(image_data[py, px])
                offs_list.append(off)
        if len(values) < 10:
            continue

        values = np.array(values, dtype=float)
        offs_arr = np.array(offs_list, dtype=float)

        # Background from wings
        wing_mask = np.abs(offs_arr) > profile_half_width * 0.6
        if np.sum(wing_mask) < 4:
            continue
        wing_vals = values[wing_mask]
        for _ in range(3):
            med = np.median(wing_vals)
            std = np.std(wing_vals)
            if std == 0:
                break
            keep = np.abs(wing_vals - med) < 2.5 * std
            if np.sum(keep) < 3:
                break
            wing_vals = wing_vals[keep]
        baseline = np.median(wing_vals)
        trial_baselines.append(baseline)

        excess = values - baseline

        try:
            amp_guess = np.max(excess)
            if amp_guess <= 0:
                continue
            popt, _ = curve_fit(
                gaussian, offs_arr, excess,
                p0=[amp_guess, 0, 3.0],
                bounds=([0, -5, 0.5], [np.inf, 5, profile_half_width]),
                maxfev=2000
            )
            trial_sigmas.append(popt[2])
            trial_amps.append(popt[0])
        except (RuntimeError, ValueError):
            continue

    if len(trial_sigmas) < 3:
        # Not enough data to characterize streak
        return np.zeros_like(image_data, dtype=float)

    # Robust global estimates using median
    global_sigma = np.median(trial_sigmas)
    global_amp = np.median(trial_amps)

    # Compute robust spread of amplitudes (MAD-based) to set a cap.
    # The amplitude should be fairly constant along the streak.
    # Sources inflate it, so use a tight upper bound.
    amp_mad = np.median(np.abs(np.array(trial_amps) - global_amp))
    amp_std_robust = 1.4826 * amp_mad  # MAD to std conversion
    amp_cap = global_amp + 3.0 * amp_std_robust  # hard ceiling

    # Constrain sigma tightly around global estimate
    sigma_lo = max(0.5, global_sigma * 0.7)
    sigma_hi = global_sigma * 1.5

    print(f"    Streak global params: amplitude={global_amp:.1f}, "
          f"sigma={global_sigma:.2f}, amp_cap={amp_cap:.1f}")

    # =========================================================================
    # SECOND PASS: Build per-slice streak model with constrained parameters
    # =========================================================================
    for t in t_values:
        cx = x1 + t * (x2 - x1)
        cy = y1 + t * (y2 - y1)

        offsets = np.arange(-profile_half_width, profile_half_width + 1)
        coords = []
        values = []

        for off in offsets:
            px = int(round(cx + off * perp_dx))
            py = int(round(cy + off * perp_dy))
            if 0 <= px < w and 0 <= py < h:
                coords.append((py, px, off))
                values.append(image_data[py, px])

        if len(values) < 2 * profile_half_width // 3:
            continue

        values = np.array(values, dtype=float)
        offs = np.array([c[2] for c in coords], dtype=float)

        # --- Step 1: Estimate background from the wings ---
        wing_mask = np.abs(offs) > profile_half_width * 0.6
        if np.sum(wing_mask) < 4:
            wing_mask = np.abs(offs) > profile_half_width * 0.4
        wing_vals = values[wing_mask]
        for _ in range(3):
            med = np.median(wing_vals)
            std = np.std(wing_vals)
            if std == 0:
                break
            keep = np.abs(wing_vals - med) < 2.5 * std
            if np.sum(keep) < 3:
                break
            wing_vals = wing_vals[keep]
        baseline = np.median(wing_vals)

        excess = values - baseline

        # --- Step 2: Pre-mask likely source pixels before fitting ---
        # Any pixel with excess > amp_cap is almost certainly a source.
        # Exclude those from the fit entirely.
        source_mask = excess > amp_cap
        fit_mask = ~source_mask
        if np.sum(fit_mask) < 5:
            # Almost all pixels are flagged — use global model directly
            best_fit = gaussian(offs, global_amp, 0, global_sigma)
        else:
            best_fit = np.zeros_like(excess)

            # Iterative fit with constrained sigma AND capped amplitude
            for iteration in range(3):
                try:
                    fit_excess = excess[fit_mask]
                    fit_offs = offs[fit_mask]
                    amp_guess = min(np.max(fit_excess), amp_cap)
                    if amp_guess <= 0:
                        # No streak signal here — use global model
                        best_fit = gaussian(offs, global_amp, 0, global_sigma)
                        break

                    popt, _ = curve_fit(
                        gaussian, fit_offs, fit_excess,
                        p0=[amp_guess, 0, global_sigma],
                        bounds=([0, -3, sigma_lo],
                                [amp_cap, 3, sigma_hi]),
                        maxfev=2000
                    )
                    best_fit = gaussian(offs, *popt)

                    # Check residuals for remaining source contamination
                    residuals = excess - best_fit
                    res_std = np.std(residuals[fit_mask])
                    new_fit_mask = (~source_mask) & (residuals < 2.5 * res_std)
                    if np.sum(new_fit_mask) < 5:
                        break
                    fit_mask = new_fit_mask
                except (RuntimeError, ValueError):
                    # Fit failed — use global model as fallback
                    best_fit = gaussian(offs, global_amp, 0, global_sigma)
                    break

        # --- Step 3: Write the smooth streak model into the 2D array ---
        for i, (py, px, off) in enumerate(coords):
            streak_model[py, px] += max(best_fit[i], 0)
            weight_map[py, px] += 1.0

    # Average overlapping contributions from cross-section sampling
    valid = weight_map > 0
    streak_model[valid] /= weight_map[valid]

    return streak_model


def subtract_streaks(image_data, lines, profile_half_width=15, n_samples=100):
    """
    Subtract streak flux from the image while preserving sources and background.

    Parameters:
    - image_data: Original FITS image
    - lines: Detected lines from Hough transform (raw, may contain duplicates)
    - profile_half_width: Half-width of perpendicular sampling window
    - n_samples: Number of cross-sections per streak

    Returns:
    - corrected_image: Image with streak flux subtracted (sources preserved)
    - total_streak_model: The estimated streak-only flux that was removed
    """
    if lines is None:
        return image_data.copy(), np.zeros_like(image_data)

    # Merge duplicate detections into unique streaks
    unique_lines = merge_duplicate_lines(lines)

    if len(unique_lines) == 0:
        return image_data.copy(), np.zeros_like(image_data)

    total_streak_model = np.zeros_like(image_data, dtype=float)

    for line in unique_lines:
        streak_model = estimate_streak_profile(
            image_data, line,
            profile_half_width=profile_half_width,
            n_samples=n_samples
        )
        total_streak_model = np.maximum(total_streak_model, streak_model)

    # Smooth the streak model slightly to avoid pixel-level noise in subtraction
    total_streak_model = median_filter(total_streak_model, size=3)

    corrected_image = image_data.astype(float) - total_streak_model

    return corrected_image, total_streak_model


def plot_results(original_image, processed_image, lines, masked_image):
    """Plot the original image, processed image, and detected lines."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Original image
    axes[0, 0].imshow(original_image, cmap='gray', origin='lower')
    axes[0, 0].set_title('Original FITS Image')
    axes[0, 0].axis('off')

    # Processed image
    axes[0, 1].imshow(processed_image, cmap='gray', origin='lower')
    axes[0, 1].set_title('Processed Image')
    axes[0, 1].axis('off')

    # Detected lines
    axes[1, 0].imshow(original_image, cmap='gray', origin='lower')
    axes[1, 0].set_title('Detected Streaks')
    axes[1, 0].axis('off')

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            axes[1, 0].plot([x1, x2], [y1, y2], 'r-', linewidth=2)

    # Masked image
    axes[1, 1].imshow(masked_image, cmap='gray', origin='lower')
    axes[1, 1].set_title('Cleaned Image (Streak Subtracted)')
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Detect streaks in FITS images using Hough transform.')
    parser.add_argument('fits_file', type=str, help='Path to the FITS file.')
    args = parser.parse_args()

    if args.fits_file is None:
        print("Usage: python detect-streaks.py <fits_file>")
        return

    fits_file = args.fits_file

    try:
        # Load image
        print(f"Loading FITS file: {fits_file}")
        image_data = load_fits_image(fits_file)

        # Preprocess
        print("Preprocessing image...")
        processed_image = preprocess_image(image_data)

        # Detect streaks
        print("Detecting streaks...")
        lines = detect_streaks(processed_image)

        # Subtract streaks
        if lines is not None:
            print(f"Detected {len(lines)} potential streaks:")
            for i, line in enumerate(lines):
                x1, y1, x2, y2 = line[0]
                print(f"  Line {i+1}: ({x1}, {y1}) to ({x2}, {y2})")

            masked_image, streak_model = subtract_streaks(image_data, lines)
        else:
            print("No streaks detected.")
            masked_image = image_data.copy()

        # Save cleaned image to FITS
        output_fits = fits_file.replace('.fits', '_cleaned.fits')
        hdu = fits.PrimaryHDU(masked_image.astype(np.float32))
        hdu.writeto(output_fits, overwrite=True)
        print(f"Cleaned image saved to {output_fits}")

        # Plot results
        plot_results(image_data, processed_image, lines, masked_image)

    except Exception as e:
        print(f"Error: {e}")
        return


if __name__ == "__main__":
    main()