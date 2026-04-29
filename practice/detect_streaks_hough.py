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

def mask_streaks(image_data, lines, line_thickness=3):
    """
    Create a mask for detected streaks and apply it to the image.
    
    Parameters:
    - image_data: Original image array
    - lines: Detected lines from Hough transform
    - line_thickness: Thickness of the masking line in pixels
    
    Returns:
    - masked_image: Image with streaks masked
    - mask: Binary mask showing where streaks were detected
    """
    if lines is None:
        return image_data.copy(), np.zeros_like(image_data, dtype=bool)
    
    # Create a mask image
    mask = np.zeros_like(image_data, dtype=np.uint8)
    
    # Draw lines on the mask
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(mask, (x1, y1), (x2, y2), 255, line_thickness)
    
    # Convert mask to boolean
    mask_bool = mask > 0
    
    # Apply mask to image
    masked_image = image_data.copy()
    masked_image = cv2.inpaint(image_data.astype(np.float32), mask, 3, cv2.INPAINT_TELEA)
    #masked_image[mask_bool] = np.nanmin(image_data)
    
    return masked_image, mask_bool

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
    axes[1, 1].set_title('Masked Image')
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

        # Mask streaks
        if lines is not None:
            print(f"Detected {len(lines)} potential streaks:")
            for i, line in enumerate(lines):
                x1, y1, x2, y2 = line[0]
                print(f"  Line {i+1}: ({x1}, {y1}) to ({x2}, {y2})")
    
            masked_image, mask = mask_streaks(image_data, lines)
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
