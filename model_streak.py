import numpy as np
from scipy.optimize import curve_fit
from scipy.ndimage import median_filter

def estimate_streak_profile(image_data, line, profile_half_width=35, n_samples=50, exclude_mask=None):
    """
    Model a streak using a dynamic angle-correcting approach:
    1. sample cross-sections to establish global shape parameters.
    2. Fit a polynomial to correct angle errors and dynamic amplitude.
    3. render the streak using signed geometric distance
    """
    if isinstance(line, np.ndarray) and line.ndim > 1:
        x1, y1, x2, y2 = line[0]
    else:
        x1, y1, x2, y2 = line

    length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    if length == 0:
        return np.zeros_like(image_data, dtype=float)

    dx, dy = (x2 - x1) / length, (y2 - y1) / length
    perp_dx, perp_dy = -dy, dx

    h, w = image_data.shape
    streak_model = np.zeros_like(image_data, dtype=float)

    def moffat(x, amplitude, center, alpha, beta):
        return amplitude * (1 + ((x - center) / alpha)**2)**(-beta)
    
    def estimate_wing_baseline(values, offs, min_wing_frac=0.6):
        for frac in (min_wing_frac, 0.4):
            wing_mask = np.abs(offs) > profile_half_width * frac
            if np.sum(wing_mask) < 4: continue

            wing_vals = values[wing_mask]
            wing_offs = offs[wing_mask]

            for _ in range(3):
                med = np.median(wing_vals)
                std = np.std(wing_vals)
                if std == 0: break
                keep = np.abs(wing_vals - med) < 2.5 * std
                if np.sum(keep) < 3: break
                wing_vals = wing_vals[keep]
                wing_offs = wing_offs[keep]

            if np.sum(wing_vals) < 3: continue
            left = wing_vals[wing_offs < 0]
            right = wing_vals[wing_offs > 0]
            if left.size >= 2 and right.size >= 2:
                if np.abs(np.median(left) - np.median(right)) > 2.5 * np.std(np.concatenate([left, right])):
                    continue
            return np.median(wing_vals)
        return None

    # =========================================================================
    # Estimate parameters & track center drift
    # =========================================================================
    trial_alphas, trial_betas, trial_amps = [], [], []
    trial_centers, trial_t_dists = [], [] 

    for t in np.linspace(0.05, 0.95, min(n_samples, int(length))):
        cx = x1 + t * (x2 - x1)
        cy = y1 + t * (y2 - y1)

        offsets = np.arange(-profile_half_width, profile_half_width + 1)
        values, offs_list, excluded = [], [], []
        
        for off in offsets:
            px, py = int(round(cx + off * perp_dx)), int(round(cy + off * perp_dy))
            if 0 <= px < w and 0 <= py < h:
                values.append(image_data[py, px])
                offs_list.append(off)
                excluded.append(exclude_mask[py, px] if exclude_mask is not None else False)
                
        if len(values) < 10: continue

        values = np.array(values, dtype=float)
        offs_arr = np.array(offs_list, dtype=float)
        valid = ~np.array(excluded, dtype=bool)

        if np.sum(valid) < 10: continue
        values, offs_arr = values[valid], offs_arr[valid]

        wing_mask = np.abs(offs_arr) > profile_half_width * 0.6
        if np.sum(wing_mask) < 4: continue
        
        baseline = estimate_wing_baseline(values, offs_arr)
        if baseline is None: continue

        excess = values - baseline
        try:
            amp_guess = np.max(excess)
            if amp_guess <= 0: continue
            
            popt, _ = curve_fit(
                moffat, offs_arr, excess,
                p0=[amp_guess, 0.0, 3.0, 2.5],
                bounds=([0, -5.0, 0.5, 1.1], [np.inf, 5.0, profile_half_width/2, 8.0]),
                maxfev=2000
            )
            trial_amps.append(popt[0])
            trial_centers.append(popt[1]) 
            trial_t_dists.append(t * length) 
            trial_alphas.append(popt[2])
            trial_betas.append(popt[3])
        except (RuntimeError, ValueError):
            continue

    if len(trial_alphas) < 3:
        return np.zeros_like(image_data, dtype=float)

    # Global shape parameters for width/decay
    global_alpha = np.median(trial_alphas)
    global_beta = np.median(trial_betas)

    # --- DYNAMIC AMPLITUDE LOGIC ---
    trial_amps = np.array(trial_amps)
    trial_t_dists_arr = np.array(trial_t_dists)
    
    # 1. Erase stars using a rolling median filter
    smoothed_amps = median_filter(trial_amps, size=5)
    
    # 2. Fit a low-degree polynomial to the smoothed amplitudes.
    # Fallback to lower degrees if there are very few valid samples
    if len(smoothed_amps) >= 3:
        amp_poly = np.polyfit(trial_t_dists_arr, smoothed_amps, 2)
    elif len(smoothed_amps) == 2:
        amp_poly = np.polyfit(trial_t_dists_arr, smoothed_amps, 1)
    else:
        amp_poly = [0.0, 0.0, smoothed_amps[0]]
    # -----------------------------------

    # Fix Angle Errors
    trial_centers = np.array(trial_centers)
    trial_t_dists = np.array(trial_t_dists)
    
    med_center = np.median(trial_centers)
    mad_center = np.median(np.abs(trial_centers - med_center))
    valid_centers = np.abs(trial_centers - med_center) < (3.0 * 1.4826 * mad_center + 0.5)

    # Only use the middle 70% of the line to establish the angle
    # core_mask = (trial_t_dists > length * 0.15) & (trial_t_dists < length * 0.85)
    # valid_core = valid_centers & core_mask
    
    if np.sum(valid_centers) >= 2:
        center_poly = np.polyfit(trial_t_dists[valid_centers], trial_centers[valid_centers], 2)
    else:
        center_poly = [0.0, med_center]

    # =========================================================================
    # Vectorized Rendering 
    # =========================================================================
    x_min = max(0, int(min(x1, x2) - profile_half_width - 2))
    x_max = min(w, int(max(x1, x2) + profile_half_width + 2))
    y_min = max(0, int(min(y1, y2) - profile_half_width - 2))
    y_max = min(h, int(max(y1, y2) + profile_half_width + 2))

    yy, xx = np.mgrid[y_min:y_max, x_min:x_max]

    wx = xx - x1
    wy = yy - y1

    t_dist = (wx * dx) + (wy * dy)
    signed_d_perp = wy * dx - wx * dy
    d_perp_abs = np.abs(signed_d_perp)

    valid_mask = (t_dist >= 0.0) & (t_dist <= length) & (d_perp_abs <= profile_half_width)

    dynamic_center = np.polyval(center_poly, t_dist[valid_mask])
    
    # --- DYNAMIC AMPLITUDE RENDERING ---
    dynamic_amp = np.polyval(amp_poly, t_dist[valid_mask])
    dynamic_amp = np.clip(dynamic_amp, 0.0, None)
    
    fluxes = moffat(signed_d_perp[valid_mask], dynamic_amp, dynamic_center, global_alpha, global_beta)
    # ---------------------------------------

    # TAPER LOGIC: Smoothly fade out the flux over the last 3 pixels of the line.
    # t_valid = t_dist[valid_mask]
    # dist_to_ends = np.minimum(t_valid, length - t_valid)
    # taper = np.clip(dist_to_ends / 3.0, 0.0, 1.0)
    
    # fluxes *= taper

    streak_model[yy[valid_mask], xx[valid_mask]] += fluxes

    return streak_model