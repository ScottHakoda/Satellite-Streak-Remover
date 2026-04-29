from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt

# 1. Load the images
orig_data = fits.getdata('synthetic_image.fits')
clean_data = fits.getdata('synthetic_image_cleaned.fits')
model_data = fits.getdata('synthetic_image_streakmodel.fits')

# 2. Pick a row (or column) that intersects the streak
# CHANGE THIS to a Y-coordinate where the streak is clearly visible
row_index = 345


# Extract the row from both images
orig_slice = orig_data[row_index, :]
clean_slice = clean_data[row_index, :]
model_slice = model_data[row_index, :]

# 3. Find the center of the streak in this slice
# We'll look for the brightest pixel in the original image's row
streak_center_x = np.argmax(orig_slice)

# 4. Zoom in on a 40-pixel window around the streak
window_start = max(0, streak_center_x - 20)
window_end = min(orig_data.shape[1], streak_center_x + 20)

orig_window = orig_slice[window_start:window_end]
clean_window = clean_slice[window_start:window_end]
model_window = model_slice[window_start:window_end]
pixels = np.arange(window_start, window_end)

# 5. Print the numerical values 
print(f"--- Pixel Values Across the Streak (Row {row_index}) ---")
print("Pixel_X | Original_Flux | Cleaned_Flux | Model_Flux | Difference")
print("-" * 60)
for p, o, c, m in zip(pixels, orig_window, clean_window, model_window):
    print(f"{p:7d} | {o:13.2f} | {c:12.2f} | {m:10.2f} | {o-c:10.2f}")


plt.figure(figsize=(10, 5))
plt.plot(pixels, orig_window, label='Original Image', marker='.')
plt.plot(pixels, clean_window, label='Cleaned Image', marker='.')
plt.plot(pixels, model_window, label='Model Image', marker='.')
plt.axhline(1000, color='gray', linestyle='--', label='Expected Sky Background') # Based on your FITS header
plt.title(f"Streak Cross-Section at Row {row_index}")
plt.xlabel("X Pixel Coordinate")
plt.ylabel("Flux (ADU)")
plt.legend()
plt.grid(True)
plt.savefig('streak_profile.png')
print("\nPlot saved as 'streak_profile.png'.")