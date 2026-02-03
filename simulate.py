"""
Artificial astronomical image generation tool for satellite streak detection testing.

This module creates synthetic FITS images with realistic stars and satellite streaks
to practice and validate satellite streak detection algorithms.
"""

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
import matplotlib.pyplot as plt

import json  


def generate_background(shape, sky_level=1000, read_noise=10):
    """
    Generate a realistic astronomical image background with noise.
    
    Parameters
    ----------
    shape : tuple
        Image dimensions (height, width)
    sky_level : float
        Mean sky background level in counts (default: 1000)
    read_noise : float
        Gaussian read noise standard deviation in counts (default: 10)
    
    Returns
    -------
    background : ndarray
        2D array containing background with Poisson and Gaussian noise
    """
    # Create sky background with Poisson noise (shot noise)
    background = np.random.poisson(sky_level, shape).astype(float)
    
    # Add Gaussian read noise
    background += np.random.normal(0, read_noise, shape)
    
    return background


def gaussian_2d(shape, center, sigma, amplitude):
    """
    Generate a 2D Gaussian profile (PSF for stars).
    
    Parameters
    ----------
    shape : tuple
        Image dimensions (height, width)
    center : tuple
        (y, x) coordinates of the Gaussian center
    sigma : float
        Standard deviation of the Gaussian in pixels
    amplitude : float
        Peak amplitude (total flux will be amplitude * 2*pi*sigma^2)
    
    Returns
    -------
    gaussian : ndarray
        2D Gaussian profile
    """
    y, x = np.ogrid[0:shape[0], 0:shape[1]]
    cy, cx = center
    
    # 2D Gaussian formula
    r2 = (x - cx)**2 + (y - cy)**2
    gaussian = amplitude * np.exp(-r2 / (2 * sigma**2))
    
    return gaussian


def inject_stars(image, n_stars=100, sigma=2.0, min_flux=500, max_flux=10000, positions=None):
    """
    Inject point sources (stars) into an image with Gaussian PSF.
    
    Parameters
    ----------
    image : ndarray
        Base image to inject stars into (modified in-place)
    n_stars : int
        Number of stars to inject (default: 100)
    sigma : float
        Gaussian PSF width in pixels (default: 2.0)
    min_flux : float
        Minimum star peak flux (default: 500)
    max_flux : float
        Maximum star peak flux (default: 10000)
    positions : list of tuples, optional
        List of (y, x) positions for stars. If None, random positions are used.
    
    Returns
    -------
    positions : list of tuples
        List of (y, x, flux) for each injected star
    """
    height, width = image.shape
    star_list = []
    
    if positions is None:
        # Generate random positions with margins to avoid edge effects
        margin = int(5 * sigma)
        for _ in range(n_stars):
            y = np.random.randint(margin, height - margin)
            x = np.random.randint(margin, width - margin)
            flux = np.random.uniform(min_flux, max_flux)
            
            # Add Gaussian PSF
            image += gaussian_2d(image.shape, (y, x), sigma, flux)
            star_list.append((y, x, flux))
    else:
        # Use provided positions
        for pos in positions:
            if len(pos) == 2:
                y, x = pos
                flux = np.random.uniform(min_flux, max_flux)
            else:
                y, x, flux = pos
            
            # Add Gaussian PSF
            image += gaussian_2d(image.shape, (y, x), sigma, flux)
            star_list.append((y, x, flux))
    
    return star_list


def inject_streak(image, start, end, width=3.0, flux=5000):
    """
    Inject a linear satellite streak into an image.
    
    Parameters
    ----------
    image : ndarray
        Base image to inject streak into (modified in-place)
    start : tuple
        (y, x) starting coordinates of the streak
    end : tuple
        (y, x) ending coordinates of the streak
    width : float
        Gaussian width of the streak perpendicular to motion (default: 3.0)
    flux : float
        Total flux per pixel along the streak centerline (default: 5000)
    
    Returns
    -------
    streak_params : dict
        Dictionary containing streak parameters (start, end, width, flux, angle, length)
    """
    y_start, x_start = start
    y_end, x_end = end
    
    # Calculate streak parameters
    length = np.sqrt((y_end - y_start)**2 + (x_end - x_start)**2)
    angle = np.arctan2(y_end - y_start, x_end - x_start)
    
    # Create coordinate grids
    height, width_img = image.shape
    y_grid, x_grid = np.ogrid[0:height, 0:width_img]
    
    # Vector along the streak
    dx = x_end - x_start
    dy = y_end - y_start
    
    # Normalize direction vector
    if length > 0:
        dx_norm = dx / length
        dy_norm = dy / length
    else:
        return {'start': start, 'end': end, 'width': width, 'flux': flux, 
                'angle': 0, 'length': 0}
    
    # Calculate perpendicular distance from each pixel to the streak line
    # and parallel distance along the streak
    x_rel = x_grid - x_start
    y_rel = y_grid - y_start
    
    # Project onto streak direction (parallel component)
    parallel_dist = x_rel * dx_norm + y_rel * dy_norm
    
    # Perpendicular component
    perp_dist = x_rel * (-dy_norm) + y_rel * dx_norm
    
    # Create streak mask (only pixels along the streak length)
    along_streak = (parallel_dist >= 0) & (parallel_dist <= length)
    
    # Gaussian profile perpendicular to streak
    streak_profile = flux * np.exp(-perp_dist**2 / (2 * width**2))
    
    # Apply only where along the streak
    streak = np.where(along_streak, streak_profile, 0)
    
    # Add to image
    image += streak
    
    return {
        'start': start,
        'end': end,
        'width': width,
        'flux': flux,
        'angle': np.degrees(angle),
        'length': length
    }


def create_synthetic_image(shape=(512, 512), sky_level=1000, read_noise=10,
                          n_stars=100, star_sigma=2.0, star_flux_range=(500, 10000),
                          streaks=None, star_positions=None):
    """
    Create a complete synthetic astronomical image with background, stars, and streaks.
    
    Parameters
    ----------
    shape : tuple
        Image dimensions (height, width) (default: (512, 512))
    sky_level : float
        Mean sky background level in counts (default: 1000)
    read_noise : float
        Gaussian read noise standard deviation (default: 10)
    n_stars : int
        Number of stars to inject (default: 100)
    star_sigma : float
        Gaussian PSF width for stars in pixels (default: 2.0)
    star_flux_range : tuple
        (min, max) flux range for stars (default: (500, 10000))
    streaks : list of dicts, optional
        List of streak parameters. Each dict should contain 'start' and 'end' keys,
        and optionally 'width' and 'flux'. Example:
        [{'start': (100, 50), 'end': (400, 450), 'width': 3.0, 'flux': 5000}]
    star_positions : list of tuples, optional
        List of (y, x) or (y, x, flux) positions for stars
    
    Returns
    -------
    image : ndarray
        Synthetic image
    metadata : dict
        Dictionary containing information about injected objects
    """
    # Generate background
    image = generate_background(shape, sky_level, read_noise)
    
    # Inject stars
    stars = inject_stars(image, n_stars=n_stars, sigma=star_sigma,
                        min_flux=star_flux_range[0], max_flux=star_flux_range[1],
                        positions=star_positions)
    
    # Inject streaks
    streak_info = []
    if streaks is not None:
        for streak_params in streaks:
            start = streak_params['start']
            end = streak_params['end']
            width = streak_params.get('width', 3.0)
            flux = streak_params.get('flux', 5000)
            
            streak_data = inject_streak(image, start, end, width, flux)
            streak_info.append(streak_data)
    
    # Compile metadata
    metadata = {
        'shape': shape,
        'sky_level': sky_level,
        'read_noise': read_noise,
        'n_stars': len(stars),
        'stars': stars,
        'n_streaks': len(streak_info),
        'streaks': streak_info
    }
    
    return image, metadata


def save_to_fits(image, filename, metadata=None):
    """
    Save synthetic image to FITS file with appropriate header information.
    
    Parameters
    ----------
    image : ndarray
        Image data to save
    filename : str
        Output FITS filename
    metadata : dict, optional
        Metadata about the synthetic image to store in header and complementary file
    """
    # Create primary HDU
    hdu = fits.PrimaryHDU(image.astype(np.float32))
    
    # Add basic header information
    hdu.header['SIMPLE'] = True
    hdu.header['BITPIX'] = -32  # 32-bit floating point
    hdu.header['NAXIS'] = 2
    hdu.header['NAXIS1'] = image.shape[1]
    hdu.header['NAXIS2'] = image.shape[0]
    hdu.header['EXTEND'] = True
    hdu.header['BSCALE'] = 1.0
    hdu.header['BZERO'] = 0.0
    
    # Add custom metadata
    hdu.header['SIMULATED'] = (True, 'Synthetic image for testing')
    hdu.header['CREATOR'] = ('simulate.py', 'Image generation tool')
    
    if metadata is not None:
        hdu.header['SKYLEVEL'] = (metadata.get('sky_level', 0), 'Sky background level')
        hdu.header['RDNOISE'] = (metadata.get('read_noise', 0), 'Read noise std dev')
        hdu.header['NSTARS'] = (metadata.get('n_stars', 0), 'Number of injected stars')
        hdu.header['NSTREAKS'] = (metadata.get('n_streaks', 0), 'Number of satellite streaks')
        
        # Save full metadata to a complementary JSON file
        json_filename = filename.replace('.fits', '_metadata.json')
        with open(json_filename, 'w') as f:
            json.dump(metadata, f, indent=4)
        print(f"Saved metadata to {json_filename}")
    
    # Create HDU list and write to file
    hdul = fits.HDUList([hdu])
    hdul.writeto(filename, overwrite=True)
    print(f"Saved synthetic image to {filename}")


def visualize_image(image, metadata=None, show_annotations=True):
    """
    Visualize the synthetic image with optional annotations.
    
    Parameters
    ----------
    image : ndarray
        Image to display
    metadata : dict, optional
        Metadata containing star and streak information for annotations
    show_annotations : bool
        Whether to show annotations for stars and streaks (default: True)
    """
    plt.figure(figsize=(10, 10))
    
    # Display image with appropriate scaling
    vmin = np.percentile(image, 1)
    vmax = np.percentile(image, 99)
    plt.imshow(image, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
    plt.colorbar(label='Counts')
    
    # Add annotations if requested
    # if show_annotations and metadata is not None:
    #     # Mark stars
    #     if 'stars' in metadata:
    #         stars = metadata['stars']
    #         for y, x, flux in stars[:20]:  # Show only first 20 to avoid clutter
    #             circle = plt.Circle((x, y), radius=5, color='cyan', 
    #                               fill=False, linewidth=1, alpha=0.6)
    #             plt.gca().add_patch(circle)
        
    #     # Mark streaks
    #     if 'streaks' in metadata:
    #         for streak in metadata['streaks']:
    #             y_start, x_start = streak['start']
    #             y_end, x_end = streak['end']
    #             plt.plot([x_start, x_end], [y_start, y_end], 
    #                     'r-', linewidth=2, alpha=0.7, label='Streak')
    
    plt.title('Synthetic Astronomical Image')
    plt.xlabel('X (pixels)')
    plt.ylabel('Y (pixels)')
    plt.tight_layout()
    plt.show()


def main():
    """
    Example usage: Generate a synthetic image with stars and satellite streaks.
    """
    print("Generating synthetic astronomical image...")
    
    # Define image parameters
    image_shape = (512, 512)
    
    # Define satellite streaks
    streaks = [
        {
            'start': (100, 50),
            'end': (400, 450),
            'width': 2.5,
            'flux': 6000
        },
        {
            'start': (450, 100),
            'end': (300, 500),
            'width': 3.0,
            'flux': 4000
        }
    ]
    
    # Create synthetic image
    image, metadata = create_synthetic_image(
        shape=image_shape,
        sky_level=1000,
        read_noise=10,
        n_stars=150,
        star_sigma=2.0,
        star_flux_range=(500, 15000),
        streaks=streaks
    )
    
    # Print summary
    print(f"\nImage generated:")
    print(f"  Shape: {metadata['shape']}")
    print(f"  Sky level: {metadata['sky_level']} counts")
    print(f"  Read noise: {metadata['read_noise']} counts")
    print(f"  Number of stars: {metadata['n_stars']}")
    print(f"  Number of streaks: {metadata['n_streaks']}")
    
    if metadata['streaks']:
        print("\nStreak information:")
        for i, streak in enumerate(metadata['streaks'], 1):
            print(f"  Streak {i}:")
            print(f"    Start: {streak['start']}")
            print(f"    End: {streak['end']}")
            print(f"    Length: {streak['length']:.1f} pixels")
            print(f"    Angle: {streak['angle']:.1f} degrees")
            print(f"    Width: {streak['width']:.1f} pixels")
            print(f"    Flux: {streak['flux']} counts/pixel")
    
    # Save to FITS file
    output_filename = 'data/synthetic_image.fits'
    save_to_fits(image, output_filename, metadata)
    
    # Visualize
    print("\nDisplaying image...")
    visualize_image(image, metadata, show_annotations=True)


if __name__ == '__main__':
    main()
