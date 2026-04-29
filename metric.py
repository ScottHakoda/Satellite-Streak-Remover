import numpy as np
import argparse
import json
from astropy.io import fits
from skimage.draw import line
from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry

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

def evaluate_streak_removal(cleaned_image, original_image, truth_image, metadata):
    streak_scores = []
    # Get the sky level from metadata to subtract it
    sky_level = metadata.get('sky_level', 0)
    
    for streak in metadata['streaks']:
        mask = streak_mask(cleaned_image.shape, streak['start'], streak['end'], streak['width'])

        if truth_image is None:
            actual_injected = np.sum(original_image[mask] - sky_level)
            residual_flux = np.sum(cleaned_image[mask] - sky_level)
        else:
            actual_injected = np.sum((original_image - truth_image)[mask])
            residual_flux = np.sum((cleaned_image - truth_image)[mask])

        if actual_injected <= 0:
            continue

        effectiveness = 1 - (max(0, residual_flux) / actual_injected)
        streak_scores.append(effectiveness)

    return np.mean(streak_scores) if streak_scores else float('nan')

def star_true_fluxes(metadata, indices=None, sigma=2.0):
    if indices is None:
        indices = range(len(metadata['stars']))
    return np.array([metadata['stars'][i][2] * 2 * np.pi * sigma**2 for i in indices])

def measure_star_fluxes(image, metadata, indices=None, aperture=6, annulus_radii=(8, 12)):
    if indices is None:
        indices = range(len(metadata['stars']))
    positions = [(metadata['stars'][i][1], metadata['stars'][i][0]) for i in indices]
    apertures = CircularAperture(positions, r=aperture)
    annuli = CircularAnnulus(positions, r_in=annulus_radii[0], r_out=annulus_radii[1])

    aper_table = aperture_photometry(image, apertures)
    ann_table = aperture_photometry(image, annuli)

    bkg_mean = ann_table['aperture_sum'] / annuli.area
    net_flux = aper_table['aperture_sum'] - bkg_mean * apertures.area
    return np.array(net_flux)

def evaluate_star_recovery(cleaned_image, metadata, truth_image=None, aperture=6, annulus_radii=(8, 12),
                          sigma=2.0, indices=None):
    measured = measure_star_fluxes(cleaned_image, metadata, indices, aperture, annulus_radii)
    if truth_image is None:
        true_flux = star_true_fluxes(metadata, indices, sigma)
    else:
        true_flux = measure_star_fluxes(truth_image, metadata, indices, aperture, annulus_radii)
    recovery = measured / true_flux
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

def unaffected_star_indices(metadata, threshold=10):
    """Return indices of stars farther than `threshold` pixels from any streak."""
    affected = set(stars_near_streaks(metadata, threshold=threshold))
    return [i for i in range(len(metadata['stars'])) if i not in affected]

def validate_star_photometry(reference_image, cleaned_image, metadata, indices,
                             aperture=6, annulus_radii=(8, 12)):
    orig_flux = measure_star_fluxes(reference_image, metadata, indices, aperture, annulus_radii)
    clean_flux = measure_star_fluxes(cleaned_image, metadata, indices, aperture, annulus_radii)
    ratio = clean_flux / orig_flux
    print("Photometry consistency (cleaned/reference):")
    print("  median:", np.median(ratio))
    print("  mean:  ", np.mean(ratio))
    print("  min/max:", ratio.min(), ratio.max())
    return ratio

def main():
    parser = argparse.ArgumentParser(description="Evaluate streak removal performance")
    parser.add_argument("original_image", help="Path to the original image with injected streaks")
    parser.add_argument("cleaned_image", help="Path to the cleaned image")
    parser.add_argument("truth_image", help="Path to the truth image without streaks")
    parser.add_argument("metadata", help="Path to the metadata JSON file")
    parser.add_argument("--star-threshold", type=float, default=10.0,
                        help="Distance threshold for affected stars")
    parser.add_argument("--aperture-radius", type=float, default=6.0,
                        help="Aperture radius for star photometry")
    parser.add_argument("--annulus-inner", type=float, default=8.0,
                        help="Inner radius for background annulus")
    parser.add_argument("--annulus-outer", type=float, default=12.0,
                        help="Outer radius for background annulus")
    parser.add_argument("--star-sigma", type=float, default=2.0,
                        help="PSF sigma used for star injection")
    args = parser.parse_args()

    # Load images
    original_image = fits.getdata(args.original_image)
    cleaned_image = fits.getdata(args.cleaned_image)
    truth_image = fits.getdata(args.truth_image)

    # Load metadata
    with open(args.metadata, "r") as f:
        metadata = json.load(f)

    streak_removal_score = evaluate_streak_removal(cleaned_image, original_image, truth_image, metadata)

    affected_indices = stars_near_streaks(metadata, threshold=args.star_threshold)
    unaffected_indices = unaffected_star_indices(metadata, threshold=args.star_threshold)

    all_star_recovery = evaluate_star_recovery(cleaned_image, metadata,
                                               truth_image=truth_image,
                                               aperture=args.aperture_radius,
                                               annulus_radii=(args.annulus_inner, args.annulus_outer),
                                               sigma=args.star_sigma)
    affected_star_recovery = evaluate_star_recovery(cleaned_image, metadata,
                                                    truth_image=truth_image,
                                                    aperture=args.aperture_radius,
                                                    annulus_radii=(args.annulus_inner, args.annulus_outer),
                                                    sigma=args.star_sigma,
                                                    indices=affected_indices)
    unaffected_star_recovery = evaluate_star_recovery(cleaned_image, metadata,
                                                      truth_image=truth_image,
                                                      aperture=args.aperture_radius,
                                                      annulus_radii=(args.annulus_inner, args.annulus_outer),
                                                      sigma=args.star_sigma,
                                                      indices=unaffected_indices)

    print(f"Streak Removal Score: {streak_removal_score}")
    print(f"Star Recovery Score (all stars): {all_star_recovery}")
    print(f"Number of affected stars: {len(affected_indices)} out of {len(metadata['stars'])}")
    print(f"Number of unaffected stars: {len(unaffected_indices)} out of {len(metadata['stars'])}")
    print(f"Affected Star Recovery Score: {affected_star_recovery}")
    print(f"Unaffected Star Recovery Score: {unaffected_star_recovery}")

    validate_star_photometry(truth_image, cleaned_image, metadata, unaffected_indices,
                             aperture=args.aperture_radius,
                             annulus_radii=(args.annulus_inner, args.annulus_outer))

    # individual affected star ratios
    if affected_indices:
        measured = measure_star_fluxes(cleaned_image, metadata, affected_indices,
                                      aperture=args.aperture_radius,
                                      annulus_radii=(args.annulus_inner, args.annulus_outer))
        true_flux = star_true_fluxes(metadata, affected_indices, args.star_sigma)
        print("\nIndividual affected star recovery ratios:")
        for i, ratio in zip(affected_indices, measured / true_flux):
            y, x, _ = metadata['stars'][i]
            print(f"Star {i} ({y},{x}): recovery = {ratio:.3f}")

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