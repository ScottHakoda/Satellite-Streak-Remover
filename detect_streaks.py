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



def subtract_streaks(image_data, lines, profile_half_width=15, n_samples=100, progress_callback=None, log_callback=None):
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
    unique_lines = merge_duplicate_lines(lines, log_callback=log_callback)

    if len(unique_lines) == 0:
        return image_data.copy(), np.zeros_like(image_data)

    total_streak_model = np.zeros_like(image_data, dtype=float)

    for i, line in enumerate(unique_lines):

        streak_model = estimate_streak_profile(
            image_data, line,
            profile_half_width=profile_half_width,
            n_samples=n_samples
        )
        total_streak_model = np.maximum(total_streak_model, streak_model)

        if progress_callback:
            progress = 40 + int((i + 1) * 30 / len(unique_lines))
            progress_callback(progress)

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


def detect_remove(fits_file, progress_callback=None, log_callback=None):
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
        if log_callback: log_callback(f"Loading FITS file: {fits_file}")
        if progress_callback: progress_callback(10)
        image_data = load_fits_image(fits_file)

        # Preprocess
        if log_callback: log_callback("Preprocessing image...")
        if progress_callback: progress_callback(20)
        processed_image = preprocess_image(image_data)

        # Detect streaks
        if log_callback: log_callback("Detecting streaks...")
        if progress_callback: progress_callback(30)
        lines = detect_streaks(processed_image)

        # Subtract streaks
        if lines is not None:
            if log_callback: log_callback(f"Detected {len(lines)} potential streaks:")
            for i, line in enumerate(lines):
                x1, y1, x2, y2 = line[0]
                if log_callback: log_callback(f"  Line {i+1}: ({x1}, {y1}) to ({x2}, {y2})")
            if log_callback: log_callback("Subtracting streaks from image...")
            if progress_callback: progress_callback(40)
            masked_image, streak_model = subtract_streaks(image_data, lines, progress_callback=progress_callback, log_callback=log_callback)
        else:
            if log_callback: log_callback("No streaks detected.")
            masked_image = image_data.copy()

        # Save cleaned image to FITS
        output_fits = fits_file.replace('.fits', '_cleaned.fits')
        hdu = fits.PrimaryHDU(masked_image.astype(np.float32))
        hdu.writeto(output_fits, overwrite=True)
        if log_callback: log_callback(f"Cleaned image saved to {output_fits}")
        if progress_callback: progress_callback(100)

        return image_data, processed_image, lines, streak_model, masked_image

    except Exception as e:
        print(f"Error: {e}")
        return None, None, None, None, None

import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QFileDialog, QLabel, QVBoxLayout, QWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PyQt6.QtWidgets import QProgressBar, QTextEdit

from PyQt6.QtCore import QObject, pyqtSignal, QThread

class Worker(QObject):
    progress = pyqtSignal(int)
    log = pyqtSignal(str)
    finished = pyqtSignal()
    plots = pyqtSignal(object, object, object, object, object)  # To send plot data back to main thread

    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path

    def run(self):
        # pass self.progress and self.log to functions
        original_image, processed_image, lines, streak_model, masked_image = detect_remove(
            self.file_path, progress_callback=self.progress.emit, log_callback=self.log.emit
        )
        self.plots.emit(original_image, processed_image, lines, streak_model, masked_image)
        self.finished.emit()
        

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Satellite Streak Detection and Removal")
        self.setGeometry(100, 100, 1400, 1200)

        # Central widget and layout
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        self.file_path = None

        self.button_file = QPushButton("Select File", self)
        self.button_file.clicked.connect(self.open_file_dialog)
        layout.addWidget(self.button_file)

        self.label = QLabel("No file selected", self)
        layout.addWidget(self.label)

        # Matplotlib Figure and Canvas
        self.figure = Figure(figsize=(16, 14))
        self.axes = self.figure.subplots(2,2)
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        self.progress_bar = QProgressBar(self)
        layout.addWidget(self.progress_bar)

        self.console = QTextEdit(self)
        self.console.setReadOnly(True)
        self.console.setFixedHeight(150)
        layout.addWidget(self.console)

        self.button_detect = QPushButton("Detect and Remove Streaks", self)
        self.button_detect.clicked.connect(self.detect_and_plot)
        layout.addWidget(self.button_detect)

    def plot_results(self, original_image, processed_image, lines, streak_model, masked_image):
        if original_image is None:
            self.console.append("No image to display.")
            return
        
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
        self.file_path, _ = QFileDialog.getOpenFileName(self, "Open File", "", "All Files (*)")
        if self.file_path:
            self.label.setText(f"Selected: {self.file_path}")
            image_data = load_fits_image(self.file_path)
            self.axes[0, 0].imshow(image_data, cmap='gray', origin='lower')
            self.canvas.draw()

        else:
            self.label.setText("No file selected")

    def detect_and_plot(self):
        if self.file_path:
            self.progress_bar.setValue(0)
            self.console.append("Starting processing...")

            self.thread = QThread()
            self.worker = Worker(self.file_path)
            self.worker.moveToThread(self.thread)

            self.worker.progress.connect(self.progress_bar.setValue)
            self.worker.log.connect(self.console.append)
            self.worker.plots.connect(self.plot_results)
            self.worker.finished.connect(self.thread.quit)
            self.worker.finished.connect(self.worker.deleteLater)
            self.thread.finished.connect(self.thread.deleteLater)

            def on_finished():
                self.console.append("Processing finished.")

            self.worker.finished.connect(on_finished)

            self.thread.started.connect(self.worker.run)
            self.thread.start()
        else:
            self.label.setText("No file selected")



if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())