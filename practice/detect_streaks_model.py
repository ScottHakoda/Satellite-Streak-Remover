#!/usr/bin/env python3
"""
Program to detect streaks in FITS images using Hough transform.
"""

from astropy.io import fits
import matplotlib.pyplot as plt
import cv2
import numpy as np
import argparse

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
    # Apply Hough line detection
    lines = cv2.HoughLinesP(image, 1, np.pi/180, threshold=50,
                          minLineLength=min_line_length, maxLineGap=max_line_gap)

    return lines

def fit_streak_profile(image_data, line, n_profiles=50, profile_width=20):
    """
    Create perpendicular profiles along a streak and fit the streak component.
    
    Parameters:
    - image_data: Original image
    - line: [x1, y1, x2, y2] endpoints of detected streak (can be nested array)
    - n_profiles: Number of cross-sections to sample along the streak
    - profile_width: Width of perpendicular profile in pixels
    
    Returns:
    - streak_model: 2D array with only the streak component
    """
    # Handle both line[0] format from HoughLinesP and direct [x1,y1,x2,y2]
    if isinstance(line, np.ndarray) and line.ndim > 1:
        x1, y1, x2, y2 = line[0]
    else:
        x1, y1, x2, y2 = line
    
    # Calculate line parameters
    length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    dx, dy = (x2 - x1) / length, (y2 - y1) / length  # Unit vector along streak
    perp_dx, perp_dy = -dy, dx  # Perpendicular unit vector
    
    # Sample points along the streak
    t_values = np.linspace(0, 1, n_profiles)
    streak_model = np.zeros_like(image_data)
    
    for t in t_values:
        # Point along the streak
        cx = x1 + t * (x2 - x1)
        cy = y1 + t * (y2 - y1)
        
        # Extract perpendicular profile
        profile_pixels = []
        profile_coords = []
        
        for offset in range(-profile_width//2, profile_width//2):
            px = int(cx + offset * perp_dx)
            py = int(cy + offset * perp_dy)
            
            if 0 <= px < image_data.shape[1] and 0 <= py < image_data.shape[0]:
                profile_pixels.append(image_data[py, px])
                profile_coords.append((py, px, offset))
        
        if len(profile_pixels) < 5:
            continue
            
        profile_pixels = np.array(profile_pixels)
        
        # Fit a baseline (median or low percentile to avoid sources)
        # Use iterative sigma clipping to get streak baseline without sources
        baseline = fit_robust_baseline(profile_pixels)
        
        # Model the streak as baseline + Gaussian profile
        # (In reality, satellite streaks are often flat-topped)
        offsets = np.array([c[2] for c in profile_coords])
        streak_profile = fit_gaussian_or_flat(offsets, profile_pixels, baseline)
        
        # Assign streak model values (only the streak, not sources)
        for i, (py, px, offset) in enumerate(profile_coords):
            streak_model[py, px] = streak_profile[i]
    
    return streak_model

def fit_robust_baseline(profile, sigma=2, iterations=3):
    """Use sigma clipping to get baseline without sources."""
    data = profile.copy()
    for _ in range(iterations):
        median = np.median(data)
        std = np.std(data)
        # Keep only values below median + sigma (remove bright sources)
        mask = data < (median + sigma * std)
        if np.sum(mask) < 3:
            break
        data = data[mask]
    return np.median(data)

def fit_gaussian_or_flat(positions, values, baseline):
    """
    Fit a Gaussian + baseline or flat-topped profile.
    Returns only the smooth streak component.
    """
    from scipy.optimize import curve_fit
    
    # Simple Gaussian model
    def gaussian(x, amplitude, center, sigma):
        return baseline + amplitude * np.exp(-(x - center)**2 / (2 * sigma**2))
    
    try:
        # Initial guess
        amp_guess = np.max(values) - baseline
        center_guess = 0
        sigma_guess = 3
        
        popt, _ = curve_fit(
            gaussian, positions, values,
            p0=[amp_guess, center_guess, sigma_guess],
            maxfev=1000
        )
        
        # Return the fitted streak profile
        return gaussian(positions, *popt)
    except:
        # If fitting fails, return baseline
        return np.full_like(values, baseline)

def subtract_streak_model(image_data, lines, profile_width=20):
    """
    Subtract modeled streak profiles while preserving sources.
    """
    corrected_image = image_data.copy()
    total_streak_model = np.zeros_like(image_data, dtype=float)
    
    for line in lines:
        # fit_streak_profile handles the line format
        streak_model = fit_streak_profile(image_data, line, 
                                          profile_width=profile_width)
        total_streak_model += streak_model
    
    # Subtract only the streak model, preserving background level
    # Use median of non-zero model pixels to maintain background
    model_median = np.median(total_streak_model[total_streak_model > 0]) if np.any(total_streak_model > 0) else 0
    corrected_image = corrected_image - total_streak_model + model_median
    
    return corrected_image


def plot_results(original_image, processed_image, lines, masked_image):
    """Plot the original image, processed image, and detected lines."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Original image
    axes[0, 0].imshow(original_image, cmap='gray', origin='lower')
    axes[0, 0].set_title('Original FITS Image')
    axes[0, 0].axis('off')

    # Processed image
    axes[0, 1].imshow(processed_image, cmap='gray', origin='lower')
    axes[0, 1].set_title('Processed Image (Thresholded)')
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
    axes[1, 1].set_title('Masked Image (Streaks Removed)')
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

        # Model and subtract streaks
        if lines is not None:
            print(f"Detected {len(lines)} potential streaks:")
            for i, line in enumerate(lines):
                x1, y1, x2, y2 = line[0]
                print(f"  Line {i+1}: ({x1}, {y1}) to ({x2}, {y2})")
            
            print("Modeling and subtracting streak profiles...")
            masked_image = subtract_streak_model(image_data, lines, profile_width=20)
        else:
            print("No streaks detected.")
            masked_image = image_data.copy()

        # Plot results
        plot_results(image_data, processed_image, lines, masked_image)

    except Exception as e:
        print(f"Error: {e}")
        return

if __name__ == "__main__":
    main()
