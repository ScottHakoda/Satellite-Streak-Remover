#!/usr/bin/env python3
"""
Program to detect streaks in FITS images using Hough transform.
"""

from astropy.io import fits
import matplotlib.pyplot as plt
import cv2
import numpy as np
import argparse
from scipy.ndimage import median_filter

from hough_transform import detect_streaks, merge_duplicate_lines
from model_streak import estimate_streak_profile

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


def plot_results(original_image, processed_image, lines, streak_model, masked_image):
    """Plot the original image, processed image, and detected lines."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Original image
    axes[0, 0].imshow(original_image, cmap='gray', origin='lower')
    axes[0, 0].set_title('Original FITS Image')
    axes[0, 0].axis('off')

    # Detected lines
    axes[0, 1].imshow(original_image, cmap='gray', origin='lower')
    axes[0, 1].set_title('Detected Streaks')
    axes[0, 1].axis('off')

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            axes[0, 1].plot([x1, x2], [y1, y2], 'r-', linewidth=2)

    # Streak model
    axes[1, 0].imshow(streak_model, cmap='gray', origin='lower')
    axes[1, 0].set_title('Estimated Streak Model')
    axes[1, 0].axis('off')

    # Masked image
    axes[1, 1].imshow(masked_image, cmap='gray', origin='lower')
    axes[1, 1].set_title('Cleaned Image (Streak Subtracted)')
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.show()


def detect_remove(fits_file):
    # parser = argparse.ArgumentParser(description='Detect streaks in FITS images using Hough transform.')
    # parser.add_argument('fits_file', type=str, help='Path to the FITS file.')
    # args = parser.parse_args()

    # if args.fits_file is None:
    #     print("Usage: python detect-streaks.py <fits_file>")
    #     return

    # fits_file = args.fits_file

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

        return image_data, processed_image, lines, streak_model, masked_image

    except Exception as e:
        print(f"Error: {e}")
        return None, None, None, None, None

import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QFileDialog, QLabel, QVBoxLayout, QWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Satellite Streak Detection and Removal")
        self.setGeometry(100, 100, 1200, 1000)

        # Central widget and layout
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        self.button = QPushButton("Select File", self)
        self.button.clicked.connect(self.open_file_dialog)
        layout.addWidget(self.button)

        self.label = QLabel("No file selected", self)
        layout.addWidget(self.label)

        # Matplotlib Figure and Canvas
        self.figure = Figure(figsize=(16, 14))
        self.axes = self.figure.subplots(2,2)
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

    def plot_results(self, original_image, processed_image, lines, streak_model, masked_image):
        for row in self.axes:
            for ax in row:
                ax.clear()

        # Original image
        self.axes[0, 0].imshow(original_image, cmap='gray', origin='lower')
        self.axes[0, 0].set_title('Original FITS Image')
        self.axes[0, 0].axis('off')

        # Detected lines
        self.axes[0, 1].imshow(original_image, cmap='gray', origin='lower')
        self.axes[0, 1].set_title('Detected Streaks')
        self.axes[0, 1].axis('off')

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                self.axes[0, 1].plot([x1, x2], [y1, y2], 'r-', linewidth=2)

        # Streak model
        self.axes[1, 0].imshow(streak_model, cmap='gray', origin='lower')
        self.axes[1, 0].set_title('Estimated Streak Model')
        self.axes[1, 0].axis('off')

        # Masked image
        self.axes[1, 1].imshow(masked_image, cmap='gray', origin='lower')
        self.axes[1, 1].set_title('Cleaned Image (Streak Subtracted)')
        self.axes[1, 1].axis('off')

        self.figure.tight_layout()
        self.canvas.draw()

    def open_file_dialog(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open File", "", "All Files (*)")
        if file_path:
            self.label.setText(f"Selected: {file_path}")
            
            original_image, processed_image, lines, streak_model, masked_image = detect_remove(file_path)

            self.plot_results(original_image, processed_image, lines, streak_model, masked_image)

        else:
            self.label.setText("No file selected")



if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())