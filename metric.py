import numpy as np
import argparse
from skimage.draw import line
from photutils.aperture import CircularAperture, aperture_photometry
# skimage and photutils not in requirements.txt
# using for evaluation so maybe don't need to add them as dependencies

def streak_mask(shape, start, end, width):
    """Create a mask for a streak (ellipse or rectangle along the line)."""
    mask = np.zeros(shape, dtype=bool)
    # Use a thick line to approximate the streak width
    rr, cc = line(start[0], start[1], end[0], end[1])
    for dy in range(-int(width), int(width)+1):
        for dx in range(-int(width), int(width)+1):
            y = rr + dy
            x = cc + dx
            valid = (y >= 0) & (y < shape[0]) & (x >= 0) & (x < shape[1])
            mask[y[valid], x[valid]] = True
    return mask

def evaluate_streak_removal(cleaned_image, original_image, metadata):
    streak_scores = []
    for streak in metadata['streaks']:
        mask = streak_mask(cleaned_image.shape, streak['start'], streak['end'], streak['width'])
        injected_flux = (
            streak['flux'] * streak['length'] * np.sqrt(2 * np.pi) * streak['width']
        )
        residual_flux = np.sum(cleaned_image[mask])
        effectiveness = 1 - (residual_flux / injected_flux)
        streak_scores.append(effectiveness)
    return np.mean(streak_scores)

def evaluate_star_recovery(cleaned_image, metadata, aperture=3, sigma=2.0, indices=None):
    # Use the same sigma as used for injection
    if indices is None:
        indices = range(len(metadata['stars']))
    positions = [(metadata['stars'][i][1], metadata['stars'][i][0]) for i in indices]  # (x, y)
    fluxes = [metadata['stars'][i][2] for i in indices]
    # Calculate total flux for each star
    total_fluxes = [f * 2 * np.pi * sigma**2 for f in fluxes]
    apertures = CircularAperture(positions, r=aperture)
    phot_table = aperture_photometry(cleaned_image, apertures)
    measured_fluxes = phot_table['aperture_sum']
    recovery = np.array(measured_fluxes) / np.array(total_fluxes)
    return np.median(recovery) if len(recovery) > 0 else float('nan')

def stars_near_streaks(metadata, threshold=10):
    """Return indices of stars within `threshold` pixels of any streak."""
    affected_indices = []
    for i, (y, x, flux) in enumerate(metadata['stars']):
        for streak in metadata['streaks']:
            y1, x1 = streak['start']
            y2, x2 = streak['end']
            px, py = x, y  # Note: x, y order for photutils
            dx, dy = x2 - x1, y2 - y1
            if dx == dy == 0:
                dist = np.hypot(px - x1, py - y1)
            else:
                t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / (dx*dx + dy*dy)))
                proj_x = x1 + t * dx
                proj_y = y1 + t * dy
                dist = np.hypot(px - proj_x, py - proj_y)
            if dist < threshold:
                affected_indices.append(i)
                break
    return affected_indices

import json
from astropy.io import fits

def main():
    parser = argparse.ArgumentParser(description="Evaluate streak removal performance")
    parser.add_argument("original_image", help="Path to the original image with injected streaks")
    parser.add_argument("cleaned_image", help="Path to the cleaned image")
    parser.add_argument("metadata", help="Path to the metadata JSON file")
    args = parser.parse_args()

    # Load images
    original_image = fits.getdata(args.original_image)
    cleaned_image = fits.getdata(args.cleaned_image)

    # Load metadata
    with open(args.metadata, "r") as f:
        metadata = json.load(f)

    streak_removal_score = evaluate_streak_removal(cleaned_image, original_image, metadata)
    star_recovery_score = evaluate_star_recovery(cleaned_image, metadata)

    affected_indices = stars_near_streaks(metadata)
    affected_star_recovery_score = evaluate_star_recovery(cleaned_image, metadata, indices=affected_indices)

    print(f"Streak Removal Score: {streak_removal_score}")
    print(f"Star Recovery Score: {star_recovery_score}")
    print(f"Number of affected stars: {len(affected_indices)} out of {len(metadata['stars'])}")
    print(f"Affected Star Recovery Score: {affected_star_recovery_score}")

    # Print individual recovery ratios for affected stars
    if affected_indices:
        print("\nIndividual recovery ratios for affected stars:")
        positions = [(metadata['stars'][i][1], metadata['stars'][i][0]) for i in affected_indices]
        fluxes = [metadata['stars'][i][2] for i in affected_indices]
        sigma = 2.0  # Or pass as argument if variable
        total_fluxes = [f * 2 * np.pi * sigma**2 for f in fluxes]
        apertures = CircularAperture(positions, r=3)
        phot_table = aperture_photometry(cleaned_image, apertures)
        measured_fluxes = phot_table['aperture_sum']
        for idx, i in enumerate(affected_indices):
            recovery = measured_fluxes[idx] / total_fluxes[idx]
            print(f"Star {i} {positions[idx]}: recovery = {recovery:.3f}")

    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 8))
    plt.imshow(cleaned_image, origin='lower', cmap='gray', vmin=np.percentile(cleaned_image, 5), vmax=np.percentile(cleaned_image, 99))
    # Plot all stars
    all_x = [star[1] for star in metadata['stars']]
    all_y = [star[0] for star in metadata['stars']]
    plt.scatter(all_x, all_y, s=20, edgecolor='blue', facecolor='none', label='All Stars')
    # Plot affected stars
    if affected_indices:
        affected_x = [metadata['stars'][i][1] for i in affected_indices]
        affected_y = [metadata['stars'][i][0] for i in affected_indices]
        plt.scatter(affected_x, affected_y, s=40, edgecolor='red', facecolor='none', label='Affected Stars')
    # Plot streaks
    for streak in metadata['streaks']:
        y1, x1 = streak['start']
        y2, x2 = streak['end']
        plt.plot([x1, x2], [y1, y2], color='yellow', linewidth=2, label='Streak' if 'Streak' not in plt.gca().get_legend_handles_labels()[1] else "")
    plt.legend()
    plt.title("Stars, Affected Stars, and Streaks")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()