import numpy as np
from scipy.optimize import curve_fit

def estimate_streak_profile(image_data, line, profile_half_width=15, n_samples=100, exclude_mask=None):
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
    
    def estimate_wing_baseline(values, offs, min_wing_frac=0.6):
        for frac in (min_wing_frac, 0.4):
            wing_mask = np.abs(offs) > profile_half_width * frac
            if np.sum(wing_mask) < 4:
                continue

            wing_vals = values[wing_mask]
            wing_offs = offs[wing_mask]

            for _ in range(3):
                med = np.median(wing_vals)
                std = np.std(wing_vals)
                if std == 0:
                    break
                keep = np.abs(wing_vals - med) < 2.5 * std
                if np.sum(keep) < 3:
                    break
                wing_vals = wing_vals[keep]
                wing_offs = wing_offs[keep]

            if np.sum(wing_vals) < 3:
                continue
            left = wing_vals[wing_offs < 0]
            right = wing_vals[wing_offs > 0]
            if left.size >= 2 and right.size >= 2:
                if np.abs(np.median(left) - np.median(right)) > 2.5 * np.std(np.concatenate([left, right])):
                    continue

            return np.median(wing_vals)

        return None

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
        excluded = []
        for off in offsets:
            px = int(round(cx + off * perp_dx))
            py = int(round(cy + off * perp_dy))
            if 0 <= px < w and 0 <= py < h:
                values.append(image_data[py, px])
                offs_list.append(off)
                excluded.append(exclude_mask[py, px] if exclude_mask is not None else False)
        if len(values) < 10:
            continue

        values = np.array(values, dtype=float)
        offs_arr = np.array(offs_list, dtype=float)
        excluded = np.array(excluded, dtype=bool)

        valid = ~excluded
        if np.sum(valid) < 10:
            continue
        values = values[valid]
        offs_arr = offs_arr[valid]

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
        baseline = estimate_wing_baseline(values, offs_arr)
        if baseline is None:
            continue
        trial_baselines.append(baseline)

        global_baseline = np.median(trial_baselines) if trial_baselines else 0.0
        global_baseline_std = np.std(trial_baselines) if len(trial_baselines) > 1 else 0.0

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
        excluded = []

        for off in offsets:
            px = int(round(cx + off * perp_dx))
            py = int(round(cy + off * perp_dy))
            if 0 <= px < w and 0 <= py < h:
                coords.append((py, px, off))
                values.append(image_data[py, px])
                excluded.append(exclude_mask[py, px] if exclude_mask is not None else False)

        if len(values) < 2 * profile_half_width // 3:
            continue

        values = np.array(values, dtype=float)
        offs = np.array([c[2] for c in coords], dtype=float)
        excluded = np.array(excluded, dtype=bool)

        # --- Step 1: Estimate background from the wings ---
        wing_mask = (np.abs(offs) > profile_half_width * 0.6) & (~excluded)
        if np.sum(wing_mask) < 4:
            wing_mask = (np.abs(offs) > profile_half_width * 0.4) & (~excluded)
        # wing_vals = values[wing_mask]
        # for _ in range(3):
        #     med = np.median(wing_vals)
        #     std = np.std(wing_vals)
        #     if std == 0:
        #         break
        #     keep = np.abs(wing_vals - med) < 2.5 * std
        #     if np.sum(keep) < 3:
        #         break
        #     wing_vals = wing_vals[keep]
        baseline = estimate_wing_baseline(values[wing_mask], offs[wing_mask])
        if baseline is None:
            baseline = global_baseline
        elif global_baseline_std > 0 and np.abs(baseline - global_baseline) > 3.0 * global_baseline_std:
            baseline = global_baseline

        excess = values - baseline

        # --- Step 2: Pre-mask likely source pixels before fitting ---
        # Any pixel with excess > amp_cap is almost certainly a source.
        # Exclude those from the fit entirely.
        source_mask = excess > amp_cap
        fit_mask = (~source_mask) & (~excluded)
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

