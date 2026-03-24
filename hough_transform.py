import cv2
import numpy as np

def detect_streaks(image, min_line_length=50, max_line_gap=10):
    """Detect streaks using Hough line transform."""
    lines = cv2.HoughLinesP(image, 1, np.pi/180, threshold=50,
                            minLineLength=min_line_length, maxLineGap=max_line_gap)
    return lines


def merge_duplicate_lines(lines, angle_threshold=2.0, distance_threshold=10.0):
    """
    Merge duplicate/near-duplicate line detections into unique streaks.

    Lines that have similar angles and whose midpoints are close together
    (perpendicular to the line direction) are grouped and averaged.

    Parameters:
    - lines: Array of detected lines from HoughLinesP, shape (N, 1, 4)
    - angle_threshold: Maximum angle difference (degrees) to consider lines as duplicates
    - distance_threshold: Maximum perpendicular distance (pixels) between line midpoints

    Returns:
    - merged: List of unique lines as [[x1, y1, x2, y2], ...]
    """
    if lines is None or len(lines) == 0:
        return []

    # Extract line parameters
    line_params = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180  # Normalize to [0, 180)
        mid_x = (x1 + x2) / 2.0
        mid_y = (y1 + y2) / 2.0
        length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        line_params.append({
            'coords': [x1, y1, x2, y2],
            'angle': angle,
            'mid': (mid_x, mid_y),
            'length': length,
            'assigned': False
        })

    groups = []
    for i, lp in enumerate(line_params):
        if lp['assigned']:
            continue
        group = [i]
        lp['assigned'] = True

        # Direction unit vector for this line (used for perpendicular distance)
        angle_rad = np.radians(lp['angle'])
        dir_x, dir_y = np.cos(angle_rad), np.sin(angle_rad)
        # Perpendicular direction
        perp_x, perp_y = -dir_y, dir_x

        for j, other in enumerate(line_params):
            if other['assigned']:
                continue

            # Check angle similarity
            angle_diff = abs(lp['angle'] - other['angle'])
            angle_diff = min(angle_diff, 180 - angle_diff)  # Handle wrapping
            if angle_diff > angle_threshold:
                continue

            # Check perpendicular distance between midpoints
            dmx = other['mid'][0] - lp['mid'][0]
            dmy = other['mid'][1] - lp['mid'][1]
            perp_dist = abs(dmx * perp_x + dmy * perp_y)
            if perp_dist > distance_threshold:
                continue

            group.append(j)
            other['assigned'] = True

        groups.append(group)

    # For each group, compute the merged line by taking the extreme endpoints
    merged = []
    for group in groups:
        all_points = []
        angles = []
        for idx in group:
            x1, y1, x2, y2 = line_params[idx]['coords']
            all_points.append((x1, y1))
            all_points.append((x2, y2))
            angles.append(line_params[idx]['angle'])

        # Average angle for projection
        avg_angle = np.mean(angles)
        angle_rad = np.radians(avg_angle)
        dir_x, dir_y = np.cos(angle_rad), np.sin(angle_rad)

        # Project all points onto the average direction
        projections = []
        for px, py in all_points:
            proj = px * dir_x + py * dir_y
            projections.append((proj, px, py))

        projections.sort(key=lambda p: p[0])
        _, sx, sy = projections[0]
        _, ex, ey = projections[-1]

        merged.append(np.array([[int(round(sx)), int(round(sy)),
                                  int(round(ex)), int(round(ey))]]))

    print(f"  Merged {len(lines)} raw detections into {len(merged)} unique streaks")
    return merged