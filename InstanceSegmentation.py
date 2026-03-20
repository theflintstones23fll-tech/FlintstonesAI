import os
import cv2
import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
def hausdorff(A,B):
    D = cdist(A,B)
    return max(D.min(axis=1).max(), D.min(axis=0).max())

def segment_by_edges(image, min_area=500):
    """Segment using Canny edge detection and contours."""
    img = cv2.imread(image) if isinstance(image, str) else image
    if img is None:
        return None, None
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    edges = cv2.Canny(blurred, 50, 150)
    
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=2)
    
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    mask = np.zeros(gray.shape, np.uint8)
    
    valid_contours = []
    for cnt in contours:
        if cv2.contourArea(cnt) >= min_area:
            valid_contours.append(cnt)
            cv2.drawContours(mask, [cnt], -1, 255, -1)
    
    return valid_contours, mask


def get_polygon_from_contour(contour):
    """Convert OpenCV contour to polygon list."""
    if contour is None or len(contour) == 0:
        return []
    return contour.squeeze().tolist()


def classify_by_position(contours, img_shape):
    """
    Classify contours as artifact or meter using multi-feature scoring.
    The meter is a manufactured ruler: near-perfect rectangle with
    straight edges, high elongation, and small relative area.
    """
    if not contours:
        return {}

    meter, artifacts = find_meter_contour(contours, img_shape)
    return {'meter': meter, 'artifacts': artifacts}


def score_meter_likelihood(contour, img_shape):
    """
    Score how likely a contour is to be a meter/scale bar (0 to 1).
    Uses multiple geometric features that distinguish a manufactured ruler
    from a natural artifact:

    1. Rectangularity: contour area / bounding rect area
       Meter bars fill their bounding box almost perfectly (>0.85).
       Artifacts with curved/irregular edges score much lower.

    2. Elongation: max(w,h) / min(w,h) of bounding rect
       Meter bars are long and thin (ratio > 3).

    3. Edge straightness: polygon approximation vertex count
       A rectangle approximates to ~4 vertices. Artifacts with curves
       need many more vertices even at coarse approximation.

    4. Min-area-rect fit: contour area / min-area rotated rect area
       Meter bars fit tightly in their rotated bounding rect.
       This handles tilted meter bars that the axis-aligned bbox misses.

    5. Consistent width: the min-area-rect's short side should be
       much smaller than the long side, and the contour should have
       uniform width along its length.
    """
    if contour is None or len(contour) < 4:
        return 0.0

    area = cv2.contourArea(contour)
    if area < 100:
        return 0.0

    # --- Feature 1: Rectangularity (axis-aligned) ---
    x, y, bw, bh = cv2.boundingRect(contour)
    bbox_area = bw * bh
    if bbox_area == 0:
        return 0.0
    rectangularity = area / bbox_area  # 1.0 = perfect rectangle

    # --- Feature 2: Elongation ---
    aspect = max(bw, bh) / max(1, min(bw, bh))

    # --- Feature 3: Polygon approximation vertex count ---
    perimeter = cv2.arcLength(contour, True)
    # Use a coarse epsilon — rectangles stay at 4 vertices,
    # curved shapes need more
    epsilon = 0.03 * perimeter
    approx = cv2.approxPolyDP(contour, epsilon, True)
    n_vertices = len(approx)
    # Score: 4 vertices = 1.0, more vertices = lower score
    vertex_score = max(0, 1.0 - abs(n_vertices - 4) * 0.15)

    # --- Feature 4: Min-area rotated rect fit ---
    rect = cv2.minAreaRect(contour)
    rect_area = rect[1][0] * rect[1][1]
    if rect_area == 0:
        return 0.0
    rotated_rect_fit = area / rect_area

    # --- Feature 5: Rotated rect elongation ---
    short_side = min(rect[1][0], rect[1][1])
    long_side = max(rect[1][0], rect[1][1])
    rot_aspect = long_side / max(1, short_side)

    # --- Feature 6: Relative size in image ---
    # Meter bars are typically smaller objects, not the dominant one
    img_h, img_w = img_shape[:2]
    relative_area = area / (img_h * img_w)
    # Meter bars usually occupy < 5% of image area
    size_score = 1.0 if relative_area < 0.05 else max(0, 1.0 - (relative_area - 0.05) * 10)

    # --- Combine into final score ---
    # Require minimum elongation to even consider
    if rot_aspect < 2.5 and aspect < 2.5:
        return 0.0

    score = (
        0.30 * rectangularity +       # high = fills bounding box well
        0.15 * min(1.0, aspect / 8) +  # higher aspect = more meter-like
        0.20 * vertex_score +           # fewer vertices = straighter edges
        0.25 * rotated_rect_fit +       # high = fits rotated rect well
        0.10 * size_score               # smaller relative to image = more likely meter
    )

    return score


def find_meter_contour(valid_contours, img_shape, min_score=0.55):
    """
    Find the meter bar among a list of contours using multi-feature scoring.

    Returns (meter_contour, remaining_contours) or (None, all_contours).
    """
    if not valid_contours:
        return None, []

    scores = []
    for cnt in valid_contours:
        s = score_meter_likelihood(cnt, img_shape)
        scores.append(s)

    best_idx = int(np.argmax(scores))
    best_score = scores[best_idx]

    if best_score >= min_score:
        meter = valid_contours[best_idx]
        others = [c for i, c in enumerate(valid_contours) if i != best_idx]
        return meter, others
    else:
        return None, list(valid_contours)


def separate_objects(image, min_area=500):
    """
    Separate ONE artifact and ONE meter from background using edge detection.
    """
    img = cv2.imread(image) if isinstance(image, str) else image
    if img is None:
        return {'meter': None, 'artifact': None}

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blurred, 30, 100)
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    valid_contours = [c for c in contours if cv2.contourArea(c) >= min_area]

    if valid_contours:
        meter_contour, artifact_contours = find_meter_contour(valid_contours, img.shape)
        if meter_contour is not None:
            artifact_contour = max(artifact_contours, key=cv2.contourArea) if artifact_contours else None
            return {'meter': meter_contour, 'artifact': artifact_contour}

    otsu_thresh, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    t_val = int(otsu_thresh * 0.7)
    binary_inv = cv2.threshold(blurred, t_val, 255, cv2.THRESH_BINARY_INV)[1]
    contours_inv, _ = cv2.findContours(binary_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    binary = cv2.threshold(blurred, t_val, 255, cv2.THRESH_BINARY)[1]
    contours_raw, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    all_contours = list(contours_inv) + list(contours_raw)
    all_contours = sorted(all_contours, key=cv2.contourArea, reverse=True)
    seen = []
    for c in all_contours:
        is_dup = any(cv2.matchShapes(c, s, cv2.CONTOURS_MATCH_I3, 0) < 0.01 for s in seen)
        if not is_dup:
            seen.append(c)
    valid_contours = [c for c in seen if cv2.contourArea(c) >= min_area]

    if not valid_contours:
        return {'meter': None, 'artifact': None}

    meter_contour, artifact_contours = find_meter_contour(valid_contours, img.shape)

    artifact_contour = None
    if artifact_contours:
        artifact_contour = max(artifact_contours, key=cv2.contourArea)

    return {'meter': meter_contour, 'artifact': artifact_contour}


def separate_all_objects(image, min_area=500):
    """
    Separate the meter and ALL artifact contours from one image.
    Returns {'meter': contour, 'artifacts': [contour, ...]}.
    """
    img = cv2.imread(image) if isinstance(image, str) else image
    if img is None:
        return {'meter': None, 'artifacts': []}

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blurred, 30, 100)
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    valid_contours = [c for c in contours if cv2.contourArea(c) >= min_area]

    if valid_contours:
        meter_contour, artifact_contours = find_meter_contour(valid_contours, img.shape)
        if meter_contour is not None:
            return {'meter': meter_contour, 'artifacts': artifact_contours}

    otsu_thresh, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    t_val = int(otsu_thresh * 0.7)
    binary_inv = cv2.threshold(blurred, t_val, 255, cv2.THRESH_BINARY_INV)[1]
    contours_inv, _ = cv2.findContours(binary_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    binary = cv2.threshold(blurred, t_val, 255, cv2.THRESH_BINARY)[1]
    contours_raw, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    all_contours = list(contours_inv) + list(contours_raw)
    all_contours = sorted(all_contours, key=cv2.contourArea, reverse=True)
    seen = []
    for c in all_contours:
        is_dup = any(cv2.matchShapes(c, s, cv2.CONTOURS_MATCH_I3, 0) < 0.01 for s in seen)
        if not is_dup:
            seen.append(c)
    valid_contours = [c for c in seen if cv2.contourArea(c) >= min_area]

    if not valid_contours:
        return {'meter': None, 'artifacts': []}

    meter_contour, artifact_contours = find_meter_contour(valid_contours, img.shape)

    return {'meter': meter_contour, 'artifacts': artifact_contours}


def get_multi_artifact_data(image_path, meter_length_cm=8, min_area=500):
    """
    Extract ALL artifacts from a single image, each with polygon, dimensions,
    contour, and crop (BGRA).

    Args:
        image_path: Path to the image
        meter_length_cm: Known length of the meter bar in cm
        min_area: Minimum contour area to consider

    Returns:
        Dictionary with 'meter', 'artifacts' (list), and 'scale' info.
        Each artifact has: contour, polygon_px, polygon_cm, dimensions, crop.
    """
    img = cv2.imread(image_path)
    if img is None:
        return None

    result = separate_all_objects(image_path, min_area)

    meter_contour = result.get('meter')
    if meter_contour is None:
        return {'error': 'No meter detected.', 'meter': None, 'artifacts': [], 'scale': None}

    mx, my, mw, mh = cv2.boundingRect(meter_contour)
    meter_pixel_length = max(mw, mh)
    if meter_pixel_length == 0:
        meter_pixel_length = 1
    px_per_cm = meter_pixel_length / meter_length_cm

    artifacts = []
    for i, cnt in enumerate(result['artifacts']):
        poly = get_polygon_from_contour(cnt)
        if not poly or len(poly) < 3:
            continue
        poly_cm = [[pt[0] / px_per_cm, pt[1] / px_per_cm] for pt in poly]
        dims = calculate_dimensions(cnt, px_per_cm)
        crop = extract_artifact_image(image_path, cnt)

        artifacts.append({
            'id': i,
            'contour': cnt,
            'polygon_px': poly,
            'polygon_cm': poly_cm,
            'dimensions': dims,
            'crop': crop,
        })

    meter_dims = calculate_dimensions(meter_contour, px_per_cm)

    meter_poly_px = get_polygon_from_contour(meter_contour) if meter_contour is not None else None
    meter_poly_cm = [[pt[0] / px_per_cm, pt[1] / px_per_cm] for pt in meter_poly_px] if meter_poly_px else None

    return {
        'meter': {
            'contour': meter_contour,
            'dimensions': meter_dims,
            'polygon_px': meter_poly_px,
            'polygon_cm': meter_poly_cm,
        },
        'artifacts': artifacts,
        'scale': {
            'pixels_per_cm': px_per_cm,
            'meter_length_cm': meter_length_cm,
        }
    }


def calculate_dimensions(contour, px_per_cm):
    """Calculate dimensions of a contour in cm."""
    if contour is None:
        return None
    
    x, y, w, h = cv2.boundingRect(contour)
    
    perimeter_px = cv2.arcLength(contour, True)
    area_px = cv2.contourArea(contour)
    
    w_cm = w / px_per_cm
    h_cm = h / px_per_cm
    perimeter_cm = perimeter_px / px_per_cm
    area_cm2 = area_px / (px_per_cm ** 2)
    
    return {
        'width_cm': round(w_cm, 2),
        'height_cm': round(h_cm, 2),
        'perimeter_cm': round(perimeter_cm, 2),
        'area_cm2': round(area_cm2, 2),
        'bounding_box': {'x': x, 'y': y, 'width': w, 'height': h}
    }


def create_segmentation_visualization(image, classified):
    """Create visualization of segmentation results with dimensions."""
    img = cv2.imread(image) if isinstance(image, str) else image
    if img is None:
        return None
    
    output = img.copy()
    
    if classified.get('meter') is not None:
        cv2.drawContours(output, [classified['meter']], -1, (0, 255, 255), 3)
        x, y, w, h = cv2.boundingRect(classified['meter'])
        cv2.putText(output, "METER", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    if classified.get('artifact') is not None:
        cv2.drawContours(output, [classified['artifact']], -1, (255, 0, 0), 3)
        x, y, w, h = cv2.boundingRect(classified['artifact'])
        cv2.putText(output, "ARTIFACT", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    return output


def get_all_artifact_polygons(image_path, meter_length_cm=8, min_area=500):
    """
    Main function to extract ONE artifact polygon with dimensions in cm.
    
    Args:
        image_path: Path to the image
        meter_length_cm: Length of the meter in cm (default 8cm)
        min_area: Minimum contour area to consider
    
    Returns:
        Dictionary with meter info and single artifact with dimensions
    """
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    result = separate_objects(image_path, min_area)
    
    meter_contour = result.get('meter')
    if meter_contour is None:
        return {'error': 'No meter detected. Ensure image contains a rectangular meter.', 'meter': None, 'artifact': None}
    
    mx, my, mw, mh = cv2.boundingRect(meter_contour)
    
    if mw > mh:
        meter_pixel_length = mw
    else:
        meter_pixel_length = mh
    
    if meter_pixel_length == 0:
        meter_pixel_length = 1
    
    px_per_cm = meter_pixel_length / meter_length_cm
    
    artifact_contour = result.get('artifact')
    artifact_info = None
    
    if artifact_contour is not None:
        poly = get_polygon_from_contour(artifact_contour)
        if poly and len(poly) >= 3:
            poly_cm = [[pt[0] / px_per_cm, pt[1] / px_per_cm] for pt in poly]
            dims = calculate_dimensions(artifact_contour, px_per_cm)
            artifact_info = {
                'polygon': poly_cm,
                'polygon_px': poly,
                'dimensions': dims,
                'contour': artifact_contour
            }
    
    meter_dims = calculate_dimensions(meter_contour, px_per_cm)
    
    return {
        'meter': {
            'contour': meter_contour,
            'dimensions': meter_dims,
            'polygon_px': get_polygon_from_contour(meter_contour)
        },
        'artifact': artifact_info,
        'scale': {
            'pixels_per_cm': px_per_cm,
            'meter_length_cm': meter_length_cm
        }
    }
def rotate(points, theta):
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])
    return points @ R.T
def resample_polygon(points, n=500):
    points = np.array(points, dtype=float)
    if len(points) < 2:
        return points

    d = np.sqrt(((np.diff(points, axis=0))**2).sum(axis=1))
    d = np.insert(d, 0, 0)

    s = np.cumsum(d)
    total = s[-1]
    if total == 0:
        return np.tile(points[0], (n, 1))
    s /= total

    new_s = np.linspace(0, 1, n)
    new_points = []

    for t in new_s:
        i = np.searchsorted(s, t)

        if i == 0:
            new_points.append(points[0])
        elif i >= len(s):
            new_points.append(points[-1])
        else:
            denom = s[i] - s[i-1]
            if denom == 0:
                new_points.append(points[i])
            else:
                alpha = (t - s[i-1]) / denom
                p = points[i-1] + alpha * (points[i] - points[i-1])
                new_points.append(p)

    return np.array(new_points)
def normalize_polygon(points):
    """Center polygon at origin and scale so max distance from center = 1."""
    points = np.array(points, dtype=float)
    centroid = points.mean(axis=0)
    centered = points - centroid
    scale = np.max(np.sqrt((centered**2).sum(axis=1)))
    if scale == 0:
        scale = 1.0
    normalized = centered / scale
    return normalized, centroid, scale

def match_fragments(polyA, polyB):

    A = resample_polygon(polyA)
    B = resample_polygon(polyB)

    # Normalize both polygons: center at origin, scale to unit size
    # This makes the comparison position- and scale-independent
    A_norm, A_centroid, A_scale = normalize_polygon(A)
    B_norm, B_centroid, B_scale = normalize_polygon(B)

    best_score = float("inf")
    best_theta = None

    for theta in np.linspace(0, 2*np.pi, 360):

        B_rot = rotate(B_norm, theta)

        # Centroids are already at origin after normalization,
        # so no translation needed
        score = hausdorff(A_norm, B_rot)

        if score < best_score:
            best_score = score
            best_theta = theta

    return {
        "score": best_score,
        "rotation": best_theta,
        "scale_A": A_scale,
        "scale_B": B_scale
    }
def visualize_match(polyA, polyB, match_result):

    A = resample_polygon(polyA)
    B = resample_polygon(polyB)

    A_norm, _, _ = normalize_polygon(A)
    B_norm, _, _ = normalize_polygon(B)

    theta = match_result["rotation"]
    B_rot = rotate(B_norm, theta)

    plt.figure(figsize=(8,8))

    plt.plot(A_norm[:,0], A_norm[:,1], '-o', label="Fragment A", color="blue", markersize=2)
    plt.plot(B_rot[:,0], B_rot[:,1], '-o', label="Fragment B aligned", color="red", markersize=2)

    plt.legend()
    plt.title(f"Fragment Matching (normalized)\nHausdorff Score: {match_result['score']:.4f}")

    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True)

    plt.show()
def extract_artifact_image(image_path, contour):
    """
    Extract the artifact region from the source image using its contour as a mask.
    Returns the cropped BGRA image (with transparency outside the contour).
    """
    img = cv2.imread(image_path)
    if img is None or contour is None:
        return None

    # Create alpha mask from contour
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)

    # Convert to BGRA and apply mask as alpha channel
    bgra = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    bgra[:, :, 3] = mask

    # Crop to bounding rect
    x, y, w, h = cv2.boundingRect(contour)
    cropped = bgra[y:y+h, x:x+w]

    return cropped


def rotate_image(image, angle_rad):
    """
    Rotate a BGRA image by angle (radians) around its center,
    expanding the canvas so nothing is clipped.
    """
    h, w = image.shape[:2]
    angle_deg = np.degrees(angle_rad)
    cx, cy = w / 2.0, h / 2.0

    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)

    # Compute new bounding box size
    cos_a = abs(M[0, 0])
    sin_a = abs(M[0, 1])
    new_w = int(h * sin_a + w * cos_a)
    new_h = int(h * cos_a + w * sin_a)

    # Adjust the rotation matrix for the new canvas center
    M[0, 2] += (new_w / 2.0) - cx
    M[1, 2] += (new_h / 2.0) - cy

    rotated = cv2.warpAffine(image, M, (new_w, new_h),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT,
                             borderValue=(0, 0, 0, 0))
    return rotated


def extract_sub_contour(pts, start, length):
    """Extract a contiguous sub-contour of 'length' points starting at 'start', wrapping around."""
    n = len(pts)
    indices = [(start + j) % n for j in range(length)]
    return pts[indices]


def find_matching_fracture(A_pts, B_pts, seg_fraction=0.30, n_starts=60):
    """
    Find the pair of sub-contour segments (one from A, one from B) that
    are most similar in shape — these are the actual fracture lines.

    Instead of guessing independently which edge is the fracture on each
    fragment, we search all combinations of sub-contour pairs across
    both fragments at multiple rotations.

    For each candidate segment pair, B's segment is flipped (reversed)
    because the fracture line runs in opposite directions on the two
    fragments (they broke apart).

    Args:
        A_pts: resampled contour points of fragment A (centered at origin)
        B_pts: resampled contour points of fragment B (centered at origin)
        seg_fraction: fraction of contour to use as candidate fracture length
        n_starts: number of starting positions to try on each contour

    Returns:
        best result dict with rotation, translation, segment info
    """
    nA = len(A_pts)
    nB = len(B_pts)
    seg_lenA = max(10, int(nA * seg_fraction))
    seg_lenB = max(10, int(nB * seg_fraction))

    # Normalize segment length for comparison
    seg_n = 50  # resample both segments to this many points

    best_score = float("inf")
    best_result = None

    startsA = np.linspace(0, nA - 1, n_starts, dtype=int)
    startsB = np.linspace(0, nB - 1, n_starts, dtype=int)

    for sA in startsA:
        segA = extract_sub_contour(A_pts, sA, seg_lenA)
        # Resample segment A to fixed length
        segA_rs = resample_polygon(segA, n=seg_n)
        # Center segment A at origin
        segA_center = segA_rs.mean(axis=0)
        segA_c = segA_rs - segA_center

        for sB in startsB:
            segB = extract_sub_contour(B_pts, sB, seg_lenB)
            # Flip B's segment — fracture runs opposite direction on the mate
            segB_flip = segB[::-1]
            segB_rs = resample_polygon(segB_flip, n=seg_n)
            segB_center = segB_rs.mean(axis=0)
            segB_c = segB_rs - segB_center

            # Normalize both to unit scale for shape comparison
            scaleA = np.max(np.sqrt((segA_c ** 2).sum(axis=1)))
            scaleB = np.max(np.sqrt((segB_c ** 2).sum(axis=1)))
            if scaleA < 1e-6 or scaleB < 1e-6:
                continue
            segA_n = segA_c / scaleA
            segB_n = segB_c / scaleB

            # Try aligning: rotate B's segment to match A's segment
            # Use PCA to find principal direction of each
            covA = np.cov(segA_n.T)
            covB = np.cov(segB_n.T)
            _, eA = np.linalg.eigh(covA)
            _, eB = np.linalg.eigh(covB)
            dirA = eA[:, -1]
            dirB = eB[:, -1]

            angA = np.arctan2(dirA[1], dirA[0])
            angB = np.arctan2(dirB[1], dirB[0])

            # Two candidate rotations (0 and pi ambiguity in PCA)
            for flip in [0, np.pi]:
                theta = angA - angB + flip
                segB_rot = rotate(segB_n, theta)

                # Score: sum of point-to-point distances (same number of points,
                # corresponding order because both are resampled along the fracture)
                dist = np.sqrt(((segA_n - segB_rot) ** 2).sum(axis=1)).mean()

                if dist < best_score:
                    best_score = dist
                    best_result = {
                        "startA": sA,
                        "startB": sB,
                        "seg_lenA": seg_lenA,
                        "seg_lenB": seg_lenB,
                        "theta_seg": theta,
                        "score": dist,
                        "segA_center": segA_center,
                        "segB_center": segB_center,
                        "scaleA": scaleA,
                        "scaleB": scaleB,
                    }

    return best_result


def find_edge_alignment(contourA, contourB, px_per_cm_A, px_per_cm_B):
    """
    Find how to rotate and translate fragment B so its fracture edge
    connects with fragment A's fracture edge — pieces touching side by side,
    not overlapping.

    Strategy:
    1. Search all pairs of sub-contour segments to find the matching
       fracture lines (the edges that actually broke apart)
    2. Compute the rotation that aligns B's fracture to A's fracture
    3. Translate B so the fracture edges meet
    4. Nudge B outward so the fragment bodies don't overlap
    """
    polyA = np.array(get_polygon_from_contour(contourA), dtype=float)
    polyB = np.array(get_polygon_from_contour(contourB), dtype=float)

    # Scale B to A's px/cm
    scale_ratio = px_per_cm_A / px_per_cm_B
    polyB_scaled = polyB * scale_ratio

    # Center both at origin
    centA = polyA.mean(axis=0)
    centB = polyB_scaled.mean(axis=0)
    A_c = polyA - centA
    B_c = polyB_scaled - centB

    # Resample
    A_rs = resample_polygon(A_c, n=200)
    B_rs = resample_polygon(B_c, n=200)

    # Find the matching fracture pair
    match = find_matching_fracture(A_rs, B_rs)

    if match is None:
        # Fallback: just place side by side
        return {
            "rotation": 0.0,
            "translation": np.array([np.max(A_c[:, 0]) - np.min(B_c[:, 0]) + 20, 0.0]),
            "scale_ratio": scale_ratio,
            "centroid_A": centA,
            "centroid_B": centB,
        }

    # --- Compute full-contour rotation ---
    # match["theta_seg"] aligns normalized segments. We need the real rotation
    # for the actual (non-normalized) contour.
    #
    # The segment normalization scales are scaleA and scaleB.
    # The PCA-based theta_seg already accounts for the shape alignment.
    # We need to compute the rotation in real (non-normalized) coordinates.

    # Extract the actual matched segments in real coordinates
    segA_real = extract_sub_contour(A_rs, match["startA"], match["seg_lenA"])
    segB_raw = extract_sub_contour(B_rs, match["startB"], match["seg_lenB"])
    segB_real = segB_raw[::-1]  # flip to match fracture direction

    # Center both segments
    cA = segA_real.mean(axis=0)
    cB = segB_real.mean(axis=0)
    segA_c = segA_real - cA
    segB_c = segB_real - cB

    # Find the rotation that best aligns B's segment onto A's segment
    # in real (unscaled) coordinates using Procrustes
    segA_rs = resample_polygon(segA_c, n=50)
    segB_rs = resample_polygon(segB_c, n=50)

    best_theta = 0.0
    best_dist = float("inf")
    for theta in np.linspace(0, 2 * np.pi, 360):
        B_rot = rotate(segB_rs, theta)
        dist = np.sqrt(((segA_rs - B_rot) ** 2).sum(axis=1)).mean()
        if dist < best_dist:
            best_dist = dist
            best_theta = theta

    # Now compute translation:
    # Rotate B's full contour, then align the fracture segment centers
    B_full_rot = rotate(B_rs, best_theta)
    segB_center_rot = rotate(cB.reshape(1, 2), best_theta).flatten()

    # Translation = A's fracture center - B's rotated fracture center
    translation = cA - segB_center_rot

    # Nudge B outward so the bodies don't overlap:
    # Compute the normal to A's fracture segment pointing away from A's centroid
    covA_seg = np.cov(segA_c.T)
    _, eig_A = np.linalg.eigh(covA_seg)
    frac_dir = eig_A[:, -1]
    frac_normal = np.array([-frac_dir[1], frac_dir[0]])
    # Normal should point from A's centroid toward the fracture
    if np.dot(frac_normal, cA) < 0:
        frac_normal = -frac_normal

    # Check if B's body (centroid after rotation+translation) is on the
    # correct side (opposite to A's centroid). If not, nudge it.
    B_centroid_placed = np.array([0.0, 0.0]) + translation  # B was centered at origin
    a_side = np.dot(frac_normal, np.array([0.0, 0.0]) - cA)  # A centroid side
    b_side = np.dot(frac_normal, B_centroid_placed - cA)     # B centroid side

    if a_side * b_side > 0:
        # Both centroids on the same side — B needs to be on the opposite side
        # Flip B to the other side of the fracture line
        nudge = frac_normal * abs(b_side) * 2
        translation = translation + nudge

    return {
        "rotation": best_theta,
        "translation": translation,
        "scale_ratio": scale_ratio,
        "centroid_A": centA,
        "centroid_B": centB,
        "fracture_score": best_dist,
    }


def reconstruct_artifact(img_path1, img_path2, output_path=None):
    """
    Full reconstruction pipeline — places two fragment images side by side
    so their broken edges touch, with the fragment bodies facing outward.

    Steps:
    1. Segment both images to extract artifact + meter
    2. Extract masked artifact crops (BGRA with transparency)
    3. Scale fragment 2 to match fragment 1's px/cm
    4. Detect the broken edge on each fragment
    5. Rotate fragment B so its break line is parallel to A's,
       with normals pointing in opposite directions (bodies outward)
    6. Translate B so its break edge touches A's break edge
    7. Composite onto a canvas, trim, annotate, and save

    Args:
        img_path1: Path to first fragment image
        img_path2: Path to second fragment image
        output_path: Where to save the result (default: reconstruction.png)

    Returns:
        Path to the saved reconstruction image, or None on failure.
    """
    # --- Step 1: Segment both images ---
    r1 = get_all_artifact_polygons(img_path1)
    r2 = get_all_artifact_polygons(img_path2)

    if r1 is None or r1.get("artifact") is None:
        print("Reconstruction failed: artifact not detected in image 1")
        return None
    if r2 is None or r2.get("artifact") is None:
        print("Reconstruction failed: artifact not detected in image 2")
        return None

    contour1 = r1["artifact"]["contour"]
    contour2 = r2["artifact"]["contour"]
    px_per_cm_1 = r1["scale"]["pixels_per_cm"]
    px_per_cm_2 = r2["scale"]["pixels_per_cm"]

    # --- Step 2: Extract artifact crops (BGRA) ---
    crop1 = extract_artifact_image(img_path1, contour1)
    crop2 = extract_artifact_image(img_path2, contour2)

    if crop1 is None or crop2 is None:
        print("Reconstruction failed: could not extract artifact images")
        return None

    # --- Step 3: Scale fragment 2 to match fragment 1's px/cm ---
    scale_ratio = px_per_cm_1 / px_per_cm_2
    if abs(scale_ratio - 1.0) > 0.01:
        crop2_scaled_w = max(1, int(crop2.shape[1] * scale_ratio))
        crop2_scaled_h = max(1, int(crop2.shape[0] * scale_ratio))
        crop2 = cv2.resize(crop2, (crop2_scaled_w, crop2_scaled_h),
                           interpolation=cv2.INTER_LINEAR)

    # --- Step 4: Find break-edge alignment ---
    alignment = find_edge_alignment(contour1, contour2, px_per_cm_1, px_per_cm_2)
    theta = alignment["rotation"]
    tx, ty = alignment["translation"]

    # --- Step 5: Rotate crop2 ---
    # OpenCV rotates clockwise with positive angle, our math is CCW, so negate
    crop2_rot = rotate_image(crop2, -theta)

    # --- Step 6: Compute placement on canvas ---
    #
    # The translation vector from find_edge_alignment is in centered-contour
    # space (both contours were centered at origin). We need to map this
    # to pixel placement on the output canvas.
    #
    # Fragment A's contour was centered at centroid_A in the original image.
    # After extract_artifact_image, the crop's (0,0) is at the bounding box
    # top-left, so the contour centroid within the crop is:
    #   crop_centroid_A = centroid_A - bbox_A_topleft
    # The centered-contour origin corresponds to crop_centroid_A in crop space.
    #
    # Similarly for B (after scaling and rotation).
    #
    # We place A at a fixed position, then compute B's position from the
    # translation vector.

    h1, w1 = crop1.shape[:2]
    h2, w2 = crop2_rot.shape[:2]

    # The bounding box top-left of each contour in the original image
    bboxA = cv2.boundingRect(contour1)  # x, y, w, h
    bboxB = cv2.boundingRect(contour2)

    # Centroid of contour A relative to its crop
    centA = alignment["centroid_A"]
    crop_cent_A = np.array([centA[0] - bboxA[0], centA[1] - bboxA[1]])

    # For B, the centroid was scaled and we need to account for rotation.
    # After rotate_image, the center of the rotated image corresponds to
    # the center of the original crop. The contour centroid offset from
    # crop center rotates with the image.
    centB = alignment["centroid_B"]
    crop_cent_B_orig = np.array([centB[0] * scale_ratio - bboxB[0] * scale_ratio,
                                  centB[1] * scale_ratio - bboxB[1] * scale_ratio])
    # The crop center before rotation
    crop2_pre_h, crop2_pre_w = crop2.shape[:2]
    crop2_center = np.array([crop2_pre_w / 2.0, crop2_pre_h / 2.0])
    # Offset of contour centroid from crop center
    offset_from_center = crop_cent_B_orig - crop2_center
    # Rotate this offset
    offset_rotated = rotate(offset_from_center.reshape(1, 2), theta).flatten()
    # In the rotated crop, the contour centroid is at:
    crop_cent_B_rot = np.array([w2 / 2.0, h2 / 2.0]) + offset_rotated

    # Now place on canvas:
    # We want:  canvas_pos_A + crop_cent_A + translation = canvas_pos_B + crop_cent_B_rot
    # So:       canvas_pos_B = canvas_pos_A + crop_cent_A + translation - crop_cent_B_rot

    margin = 80
    # Place A with margin
    a_x = margin + max(0, w2)  # leave room in case B ends up to the left
    a_y = margin + max(0, h2)

    b_x = int(a_x + crop_cent_A[0] + tx - crop_cent_B_rot[0])
    b_y = int(a_y + crop_cent_A[1] + ty - crop_cent_B_rot[1])

    # Ensure all positions are non-negative; shift everything if needed
    min_x = min(a_x, b_x)
    min_y = min(a_y, b_y)
    if min_x < 0:
        a_x -= min_x
        b_x -= min_x
    if min_y < 0:
        a_y -= min_y
        b_y -= min_y

    # Canvas size
    canvas_w = max(a_x + w1, b_x + w2) + margin
    canvas_h = max(a_y + h1, b_y + h2) + margin

    canvas = np.zeros((canvas_h, canvas_w, 4), dtype=np.uint8)

    # --- Step 7: Composite fragments onto canvas ---
    paste_rgba(canvas, crop1, a_x, a_y)
    paste_rgba(canvas, crop2_rot, b_x, b_y)

    # --- Step 8: Trim empty space ---
    alpha_mask = canvas[:, :, 3]
    rows = np.any(alpha_mask > 0, axis=1)
    cols = np.any(alpha_mask > 0, axis=0)
    if rows.any() and cols.any():
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        pad = 50
        rmin = max(0, rmin - pad)
        rmax = min(canvas_h - 1, rmax + pad)
        cmin = max(0, cmin - pad)
        cmax = min(canvas_w - 1, cmax + pad)
        canvas = canvas[rmin:rmax+1, cmin:cmax+1]

    # --- Step 9: Convert to BGR with white background and annotate ---
    ch, cw = canvas.shape[:2]
    output = np.ones((ch, cw, 3), dtype=np.uint8) * 255

    alpha = canvas[:, :, 3:4].astype(float) / 255.0
    output = (canvas[:, :, :3].astype(float) * alpha +
              output.astype(float) * (1 - alpha)).astype(np.uint8)

    # Scale bar
    bar_length_cm = 4
    bar_length_px = int(bar_length_cm * px_per_cm_1)
    if bar_length_px > cw - 40:
        bar_length_cm = 2
        bar_length_px = int(bar_length_cm * px_per_cm_1)
    if bar_length_px > cw - 40:
        bar_length_cm = 1
        bar_length_px = int(bar_length_cm * px_per_cm_1)
    bar_x = 20
    bar_y = ch - 30
    cv2.line(output, (bar_x, bar_y), (bar_x + bar_length_px, bar_y), (0, 0, 0), 3)
    cv2.line(output, (bar_x, bar_y - 8), (bar_x, bar_y + 8), (0, 0, 0), 2)
    cv2.line(output, (bar_x + bar_length_px, bar_y - 8),
             (bar_x + bar_length_px, bar_y + 8), (0, 0, 0), 2)
    cv2.putText(output, f"{bar_length_cm} cm", (bar_x + 5, bar_y - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # Title
    cv2.putText(output, "Reconstructed Artifact", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    # Dimension info
    dims1 = r1["artifact"]["dimensions"]
    dims2 = r2["artifact"]["dimensions"]
    y_text = 55
    cv2.putText(output, f"Fragment 1: {dims1['width_cm']}x{dims1['height_cm']} cm",
                (20, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 0, 0), 1)
    cv2.putText(output, f"Fragment 2: {dims2['width_cm']}x{dims2['height_cm']} cm",
                (20, y_text + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 180), 1)

    # Match score
    poly1 = r1["artifact"]["polygon_px"]
    poly2 = r2["artifact"]["polygon_px"]
    match = match_fragments(poly1, poly2)
    cv2.putText(output, f"Match score: {match['score']:.4f}",
                (20, y_text + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 100, 0), 1)

    # --- Step 10: Save ---
    if output_path is None:
        output_dir = os.path.dirname(img_path1) or "."
        output_path = os.path.join(output_dir, "reconstruction.png")

    cv2.imwrite(output_path, output)
    print(f"\nReconstruction saved to: {output_path}")
    print(f"Canvas size: {cw} x {ch} px")
    print(f"Scale: {px_per_cm_1:.1f} px/cm")

    return output_path


def find_pairwise_alignment(contourA, contourB, px_per_cm):
    """
    Convenience wrapper around find_edge_alignment for same-image fragments
    (same px_per_cm for both).
    """
    return find_edge_alignment(contourA, contourB, px_per_cm, px_per_cm)


def paste_rgba(canvas, fragment, ox, oy):
    """Paste a BGRA fragment onto the canvas at (ox, oy) with alpha blending."""
    fh, fw = fragment.shape[:2]
    src_x1 = max(0, -ox)
    src_y1 = max(0, -oy)
    dst_x1 = max(0, ox)
    dst_y1 = max(0, oy)
    src_x2 = min(fw, canvas.shape[1] - ox)
    src_y2 = min(fh, canvas.shape[0] - oy)
    if src_x2 <= src_x1 or src_y2 <= src_y1:
        return
    dst_x2 = dst_x1 + (src_x2 - src_x1)
    dst_y2 = dst_y1 + (src_y2 - src_y1)

    frag_region = fragment[src_y1:src_y2, src_x1:src_x2]
    alpha = frag_region[:, :, 3:4].astype(float) / 255.0
    canvas_region = canvas[dst_y1:dst_y2, dst_x1:dst_x2]

    blended = (frag_region[:, :, :3].astype(float) * alpha +
               canvas_region[:, :, :3].astype(float) * (1 - alpha))
    canvas[dst_y1:dst_y2, dst_x1:dst_x2, :3] = blended.astype(np.uint8)
    new_alpha = np.maximum(canvas_region[:, :, 3:4], frag_region[:, :, 3:4])
    canvas[dst_y1:dst_y2, dst_x1:dst_x2, 3:4] = new_alpha


def compute_placement(alignment, contourA, contourB, cropA, crop_rot_B,
                      scale_ratio=1.0):
    """
    Given an alignment result and the two crops, compute canvas positions
    (a_x, a_y, b_x, b_y) for placing A and B on a shared canvas.

    Returns (a_x, a_y, b_x, b_y).
    """
    theta = alignment["rotation"]
    tx, ty = alignment["translation"]

    h1, w1 = cropA.shape[:2]
    h2, w2 = crop_rot_B.shape[:2]

    bboxA = cv2.boundingRect(contourA)
    bboxB = cv2.boundingRect(contourB)

    centA = alignment["centroid_A"]
    crop_cent_A = np.array([centA[0] - bboxA[0], centA[1] - bboxA[1]])

    centB = alignment["centroid_B"]
    crop_cent_B_orig = np.array([centB[0] * scale_ratio - bboxB[0] * scale_ratio,
                                  centB[1] * scale_ratio - bboxB[1] * scale_ratio])

    crop2_pre_h, crop2_pre_w = crop_rot_B.shape[:2]
    # Use the pre-rotation size to find the crop center
    # (rotate_image expands canvas, so post-rotation center = w2/2, h2/2)
    # We need the pre-rotation crop size. Since scale_ratio=1 for same-image,
    # and we didn't resize, the pre-rotation size is bboxB[2]*sr x bboxB[3]*sr
    pre_rot_w = int(bboxB[2] * scale_ratio)
    pre_rot_h = int(bboxB[3] * scale_ratio)
    crop2_center = np.array([pre_rot_w / 2.0, pre_rot_h / 2.0])

    offset_from_center = crop_cent_B_orig - crop2_center
    offset_rotated = rotate(offset_from_center.reshape(1, 2), theta).flatten()
    crop_cent_B_rot = np.array([w2 / 2.0, h2 / 2.0]) + offset_rotated

    margin = 80
    a_x = margin + max(0, w2)
    a_y = margin + max(0, h2)

    b_x = int(a_x + crop_cent_A[0] + tx - crop_cent_B_rot[0])
    b_y = int(a_y + crop_cent_A[1] + ty - crop_cent_B_rot[1])

    min_x = min(a_x, b_x)
    min_y = min(a_y, b_y)
    if min_x < 0:
        a_x -= min_x
        b_x -= min_x
    if min_y < 0:
        a_y -= min_y
        b_y -= min_y

    return a_x, a_y, b_x, b_y


def reconstruct_multi(image_path, output_path=None, meter_length_cm=8, min_area=500):
    """
    Reconstruct an artifact from N fragments found in a single image.

    Greedy assembly:
    1. Extract all fragments from the image
    2. Pick the largest fragment as the anchor
    3. Iteratively find the unplaced fragment with the best fracture match
       to any already-placed fragment, and attach it
    4. Composite all placed fragments onto a canvas and save

    Args:
        image_path: Path to the image containing N fragments + meter
        output_path: Where to save (default: reconstruction.png next to image)
        meter_length_cm: Known meter bar length in cm
        min_area: Minimum contour area

    Returns:
        Path to saved reconstruction, or None on failure.
    """
    data = get_multi_artifact_data(image_path, meter_length_cm, min_area)

    if data is None or data.get('scale') is None:
        print("Reconstruction failed: could not process image")
        return None

    artifacts = data['artifacts']
    n = len(artifacts)

    if n == 0:
        print("Reconstruction failed: no artifacts detected")
        return None

    if n == 1:
        print("Only 1 artifact detected — nothing to assemble")
        # Just save the single artifact crop
        if output_path is None:
            output_dir = os.path.dirname(image_path) or "."
            output_path = os.path.join(output_dir, "reconstruction.png")
        crop = artifacts[0]['crop']
        if crop is not None:
            # Convert BGRA to BGR with white background
            h, w = crop.shape[:2]
            out = np.ones((h, w, 3), dtype=np.uint8) * 255
            alpha = crop[:, :, 3:4].astype(float) / 255.0
            out = (crop[:, :, :3].astype(float) * alpha +
                   out.astype(float) * (1 - alpha)).astype(np.uint8)
            cv2.imwrite(output_path, out)
            print(f"Single artifact saved to: {output_path}")
        return output_path

    px_per_cm = data['scale']['pixels_per_cm']
    print(f"\nDetected {n} artifact fragments in image")
    print(f"Scale: {px_per_cm:.1f} px/cm")

    # --- Greedy assembly ---
    # Each placed fragment has: crop (BGRA), canvas position (x, y),
    # and its contour (for fracture matching with remaining fragments).

    # Start with the largest fragment as anchor
    artifacts_sorted = sorted(artifacts, key=lambda a: cv2.contourArea(a['contour']),
                              reverse=True)

    # Track placed and remaining fragments
    # For each placed fragment, store its crop, position on canvas, and contour
    placed = []
    remaining = list(range(n))

    # Place the anchor (largest) at the origin
    anchor_idx = 0  # index into artifacts_sorted
    anchor = artifacts_sorted[anchor_idx]
    anchor_crop = anchor['crop']

    if anchor_crop is None:
        print("Reconstruction failed: could not extract anchor artifact")
        return None

    # The anchor is placed at (0, 0) in our coordinate system
    # (we'll shift everything to be positive later)
    placed.append({
        'artifact': anchor,
        'crop': anchor_crop,
        'x': 0.0,
        'y': 0.0,
        'rotation': 0.0,
    })
    remaining.remove(anchor_idx)
    print(f"  Anchor: fragment {anchor['id']} "
          f"({anchor['dimensions']['width_cm']}x{anchor['dimensions']['height_cm']} cm)")

    # Iteratively attach best-matching fragment
    while remaining:
        best_score = float("inf")
        best_pair = None  # (placed_idx, remaining_idx, alignment)

        for pi, pf in enumerate(placed):
            for ri in remaining:
                rf = artifacts_sorted[ri]
                alignment = find_pairwise_alignment(
                    pf['artifact']['contour'],
                    rf['contour'],
                    px_per_cm
                )
                score = alignment.get('fracture_score', float("inf"))
                if score < best_score:
                    best_score = score
                    best_pair = (pi, ri, alignment)

        if best_pair is None:
            print(f"  Could not find match for remaining {len(remaining)} fragments")
            break

        pi, ri, alignment = best_pair
        pf = placed[pi]
        rf = artifacts_sorted[ri]

        print(f"  Attaching fragment {rf['id']} to fragment {pf['artifact']['id']} "
              f"(fracture score: {best_score:.4f})")

        theta = alignment["rotation"]
        tx, ty = alignment["translation"]

        # Rotate the new fragment's crop
        new_crop = rf['crop']
        if new_crop is None:
            remaining.remove(ri)
            continue
        new_crop_rot = rotate_image(new_crop, -theta)

        # Compute position relative to the placed fragment's contour centroid
        # The alignment translation is in centered-contour space.
        # The placed fragment's contour centroid in canvas space is at
        # (pf['x'] + crop_cent_pf_x, pf['y'] + crop_cent_pf_y)
        bboxP = cv2.boundingRect(pf['artifact']['contour'])
        centP = alignment["centroid_A"]
        crop_cent_P = np.array([centP[0] - bboxP[0], centP[1] - bboxP[1]])

        bboxR = cv2.boundingRect(rf['contour'])
        centR = alignment["centroid_B"]
        crop_cent_R_orig = np.array([centR[0] - bboxR[0], centR[1] - bboxR[1]])

        pre_rot_w = bboxR[2]
        pre_rot_h = bboxR[3]
        crop_center_R = np.array([pre_rot_w / 2.0, pre_rot_h / 2.0])
        offset = crop_cent_R_orig - crop_center_R
        offset_rot = rotate(offset.reshape(1, 2), theta).flatten()
        h_rot, w_rot = new_crop_rot.shape[:2]
        crop_cent_R_rot = np.array([w_rot / 2.0, h_rot / 2.0]) + offset_rot

        # New fragment's canvas position
        new_x = pf['x'] + crop_cent_P[0] + tx - crop_cent_R_rot[0]
        new_y = pf['y'] + crop_cent_P[1] + ty - crop_cent_R_rot[1]

        placed.append({
            'artifact': rf,
            'crop': new_crop_rot,
            'x': new_x,
            'y': new_y,
            'rotation': theta,
        })
        remaining.remove(ri)

    # --- Composite all placed fragments onto canvas ---
    if not placed:
        print("Reconstruction failed: no fragments placed")
        return None

    # Find bounding box of all placed fragments
    min_cx = min(p['x'] for p in placed)
    min_cy = min(p['y'] for p in placed)
    max_cx = max(p['x'] + p['crop'].shape[1] for p in placed)
    max_cy = max(p['y'] + p['crop'].shape[0] for p in placed)

    margin = 80
    # Shift so everything starts at margin
    shift_x = margin - min_cx
    shift_y = margin - min_cy

    canvas_w = int(max_cx - min_cx + margin * 2)
    canvas_h = int(max_cy - min_cy + margin * 2)
    canvas = np.zeros((canvas_h, canvas_w, 4), dtype=np.uint8)

    for p in placed:
        px_pos = int(p['x'] + shift_x)
        py_pos = int(p['y'] + shift_y)
        paste_rgba(canvas, p['crop'], px_pos, py_pos)

    # Trim empty space
    alpha_mask = canvas[:, :, 3]
    rows = np.any(alpha_mask > 0, axis=1)
    cols = np.any(alpha_mask > 0, axis=0)
    if rows.any() and cols.any():
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        pad = 50
        rmin = max(0, rmin - pad)
        rmax = min(canvas_h - 1, rmax + pad)
        cmin = max(0, cmin - pad)
        cmax = min(canvas_w - 1, cmax + pad)
        canvas = canvas[rmin:rmax+1, cmin:cmax+1]

    # Convert to BGR with white background
    ch, cw = canvas.shape[:2]
    output = np.ones((ch, cw, 3), dtype=np.uint8) * 255
    alpha = canvas[:, :, 3:4].astype(float) / 255.0
    output = (canvas[:, :, :3].astype(float) * alpha +
              output.astype(float) * (1 - alpha)).astype(np.uint8)

    # --- Annotations ---
    # Scale bar
    bar_length_cm = 4
    bar_length_px = int(bar_length_cm * px_per_cm)
    if bar_length_px > cw - 40:
        bar_length_cm = 2
        bar_length_px = int(bar_length_cm * px_per_cm)
    if bar_length_px > cw - 40:
        bar_length_cm = 1
        bar_length_px = int(bar_length_cm * px_per_cm)
    bar_x = 20
    bar_y = ch - 30
    cv2.line(output, (bar_x, bar_y), (bar_x + bar_length_px, bar_y), (0, 0, 0), 3)
    cv2.line(output, (bar_x, bar_y - 8), (bar_x, bar_y + 8), (0, 0, 0), 2)
    cv2.line(output, (bar_x + bar_length_px, bar_y - 8),
             (bar_x + bar_length_px, bar_y + 8), (0, 0, 0), 2)
    cv2.putText(output, f"{bar_length_cm} cm", (bar_x + 5, bar_y - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # Title
    cv2.putText(output, f"Reconstructed Artifact ({len(placed)} fragments)", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    # Fragment info
    colors = [(180, 0, 0), (0, 0, 180), (0, 140, 0), (140, 0, 140),
              (0, 140, 140), (140, 140, 0), (100, 100, 100)]
    y_text = 55
    for i, p in enumerate(placed):
        dims = p['artifact']['dimensions']
        color = colors[i % len(colors)]
        label = f"Fragment {p['artifact']['id']}: {dims['width_cm']}x{dims['height_cm']} cm"
        cv2.putText(output, label, (20, y_text + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # Save
    if output_path is None:
        output_dir = os.path.dirname(image_path) or "."
        output_path = os.path.join(output_dir, "reconstruction.png")

    cv2.imwrite(output_path, output)
    print(f"\nReconstruction saved to: {output_path}")
    print(f"Canvas size: {cw} x {ch} px")
    print(f"Fragments assembled: {len(placed)}/{n}")

    return output_path


def reconstruct_from_artifacts(artifacts, px_per_cm, output_path=None):
    """
    Reconstruct an artifact from N pre-extracted fragments (from separate images).

    Args:
        artifacts: List of dicts, each with:
            - contour: np.ndarray (N,1,2) polygon in pixels
            - crop: BGRA numpy array from extract_artifact_image
            - px_per_cm: float pixels per cm for this fragment
            - dimensions: dict with width_cm, height_cm, area_cm2
        px_per_cm: Reference scale (px/cm), used for annotations
        output_path: Where to save the result

    Returns:
        Path to saved image, or None on failure.
    """
    n = len(artifacts)
    if n == 0:
        return None

    if n == 1:
        crop = artifacts[0]['crop']
        if crop is None:
            return None
        h, w = crop.shape[:2]
        out = np.ones((h, w, 3), dtype=np.uint8) * 255
        alpha = crop[:, :, 3:4].astype(float) / 255.0
        out = (crop[:, :, :3].astype(float) * alpha +
               out.astype(float) * (1 - alpha)).astype(np.uint8)
        if output_path:
            cv2.imwrite(output_path, out)
        return output_path

    sorted_arts = sorted(artifacts, key=lambda a: cv2.contourArea(a['contour']), reverse=True)
    placed = []
    remaining = list(range(n))

    anchor = sorted_arts[0]
    if anchor['crop'] is None:
        return None

    placed.append({
        'crop': anchor['crop'],
        'x': 0.0, 'y': 0.0,
        'contour': anchor['contour'],
        'px_per_cm': anchor['px_per_cm'],
    })
    remaining.remove(0)

    while remaining:
        best_score = float("inf")
        best_pair = None

        for pi, pf in enumerate(placed):
            for ri in remaining:
                rf = sorted_arts[ri]
                alignment = find_pairwise_alignment(
                    pf['contour'], rf['contour'], pf['px_per_cm']
                )
                score = alignment.get('fracture_score', float("inf"))
                if score < best_score:
                    best_score = score
                    best_pair = (pi, ri, alignment)

        if best_pair is None:
            for ri in remaining:
                rf = sorted_arts[ri]
                placed.append({
                    'crop': rf['crop'],
                    'x': 100.0, 'y': 100.0,
                    'contour': rf['contour'],
                    'px_per_cm': rf['px_per_cm'],
                })
            remaining = []
            continue

        pi, ri, alignment = best_pair
        pf = placed[pi]
        rf = sorted_arts[ri]

        theta = alignment["rotation"]
        tx, ty = alignment["translation"]
        new_crop = rf['crop']
        if new_crop is None:
            remaining.remove(ri)
            continue
        new_crop_rot = rotate_image(new_crop, -theta)

        bboxP = cv2.boundingRect(pf['contour'])
        centP = alignment["centroid_A"]
        crop_cent_P = np.array([centP[0] - bboxP[0], centP[1] - bboxP[1]])

        bboxR = cv2.boundingRect(rf['contour'])
        centR = alignment["centroid_B"]
        crop_cent_R_orig = np.array([centR[0] - bboxR[0], centR[1] - bboxR[1]])

        pre_rot_w = bboxR[2]
        pre_rot_h = bboxR[3]
        crop_center_R = np.array([pre_rot_w / 2.0, pre_rot_h / 2.0])
        offset = crop_cent_R_orig - crop_center_R
        offset_rot = rotate(offset.reshape(1, 2), theta).flatten()
        h_rot, w_rot = new_crop_rot.shape[:2]
        crop_cent_R_rot = np.array([w_rot / 2.0, h_rot / 2.0]) + offset_rot

        new_x = pf['x'] + crop_cent_P[0] + tx - crop_cent_R_rot[0]
        new_y = pf['y'] + crop_cent_P[1] + ty - crop_cent_R_rot[1]

        placed.append({
            'crop': new_crop_rot,
            'x': new_x, 'y': new_y,
            'contour': rf['contour'],
            'px_per_cm': rf['px_per_cm'],
        })
        remaining.remove(ri)

    if not placed:
        return None

    min_cx = min(p['x'] for p in placed)
    min_cy = min(p['y'] for p in placed)
    max_cx = max(p['x'] + p['crop'].shape[1] for p in placed)
    max_cy = max(p['y'] + p['crop'].shape[0] for p in placed)

    margin = 80
    shift_x = margin - min_cx
    shift_y = margin - min_cy
    canvas_w = int(max_cx - min_cx + margin * 2)
    canvas_h = int(max_cy - min_cy + margin * 2)
    canvas = np.zeros((canvas_h, canvas_w, 4), dtype=np.uint8)

    for p in placed:
        px_pos = int(p['x'] + shift_x)
        py_pos = int(p['y'] + shift_y)
        paste_rgba(canvas, p['crop'], px_pos, py_pos)

    alpha_mask = canvas[:, :, 3]
    rows = np.any(alpha_mask > 0, axis=1)
    cols = np.any(alpha_mask > 0, axis=0)
    if rows.any() and cols.any():
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        pad = 50
        canvas = canvas[max(0, rmin - pad):min(canvas_h, rmax + pad + 1),
                          max(0, cmin - pad):min(canvas_w, cmax + pad + 1)]

    ch, cw = canvas.shape[:2]
    output = np.ones((ch, cw, 3), dtype=np.uint8) * 255
    alpha = canvas[:, :, 3:4].astype(float) / 255.0
    output = (canvas[:, :, :3].astype(float) * alpha +
              output.astype(float) * (1 - alpha)).astype(np.uint8)

    bar_length_cm = 4
    bar_length_px = int(bar_length_cm * px_per_cm)
    if bar_length_px > cw - 40:
        bar_length_cm = 2
        bar_length_px = int(bar_length_cm * px_per_cm)
    if bar_length_px > cw - 40:
        bar_length_cm = 1
        bar_length_px = int(bar_length_cm * px_per_cm)
    bar_x, bar_y = 20, ch - 30
    cv2.line(output, (bar_x, bar_y), (bar_x + bar_length_px, bar_y), (0, 0, 0), 3)
    cv2.line(output, (bar_x, bar_y - 8), (bar_x, bar_y + 8), (0, 0, 0), 2)
    cv2.line(output, (bar_x + bar_length_px, bar_y - 8), (bar_x + bar_length_px, bar_y + 8), (0, 0, 0), 2)
    cv2.putText(output, f"{bar_length_cm} cm", (bar_x + 5, bar_y - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(output, f"Reconstructed Artifact ({len(placed)} fragments)", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    if output_path:
        cv2.imwrite(output_path, output)
    return output_path


def reconstruct_multi_separated(image_paths_meters, output_path=None, artifact_names=None):
    """
    Reconstruct an artifact from N fragments found in N separate images,
    using the same greedy assembly algorithm as reconstruct_multi.

    Args:
        image_paths_meters: List of (image_path, meter_length_cm, meter_polygon_px, meter_polygon_cm, art_polygon_px) tuples
                            meter_polygon_px: [[x,y], ...] pixel coords of meter bar (optional)
                            meter_polygon_cm: [[x,y], ...] cm coords of meter bar (optional)
                            art_polygon_px: [[x,y], ...] pixel coords of artifact (optional)
        output_path: Where to save the result
        artifact_names: list of artifact names (one per image)

    Returns:
        Path to saved reconstruction, or None on failure.
    """
    if artifact_names is None:
        artifact_names = []
    artifacts = []
    ref_px_cm = 0

    for idx, (img_path, meter_cm, meter_poly_px, meter_poly_cm, art_poly_px) in enumerate(image_paths_meters):
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: could not read {img_path}")
            continue

        px_per_cm = None
        art_contour = None

        if meter_poly_px is not None and meter_poly_cm is not None and len(meter_poly_px) >= 2:
            meter_pts = np.array(meter_poly_px, dtype=np.float64)
            meter_len = max(
                meter_pts[:, 0].max() - meter_pts[:, 0].min(),
                meter_pts[:, 1].max() - meter_pts[:, 1].min()
            )
            if meter_len > 0:
                px_per_cm = meter_len / meter_cm

        if art_poly_px is not None and len(art_poly_px) >= 3:
            art_contour = np.array([[int(x), int(y)] for x, y in art_poly_px], dtype=np.int32)
            if px_per_cm is None:
                img_h, img_w = img.shape[:2]
                result = get_all_artifact_polygons(img_path, meter_cm, min_area=500)
                if result and result.get('artifact'):
                    scale = result.get('scale', {})
                    px_per_cm = scale.get('pixels_per_cm', 0)

        if art_contour is None:
            if px_per_cm is None:
                result = get_all_artifact_polygons(img_path, meter_cm, min_area=500)
                if result and result.get('artifact'):
                    art_data = result['artifact']
                    art_contour = art_data.get('contour')
                    scale = result.get('scale', {})
                    px_per_cm = scale.get('pixels_per_cm', 0)
                else:
                    print(f"Warning: could not detect artifact in {img_path}")
                    continue
            else:
                result_sep = separate_all_objects(img_path, min_area=500)
                art_contours = result_sep.get('artifacts') or []
                if art_contours:
                    art_contour = max(art_contours, key=cv2.contourArea)
                else:
                    print(f"Warning: no artifacts found in {img_path}")
                    continue

        if px_per_cm is None or px_per_cm <= 0:
            print(f"Warning: no scale info for {img_path}")
            continue

        crop = extract_artifact_image(img_path, art_contour)
        if crop is None:
            print(f"Warning: could not extract artifact from {img_path}")
            continue

        ar = cv2.boundingRect(art_contour)
        art_dims = {
            'width_cm': ar[2] / px_per_cm,
            'height_cm': ar[3] / px_per_cm,
            'area_cm2': cv2.contourArea(art_contour) / (px_per_cm ** 2),
        }

        artifacts.append({
            'id': len(artifacts),
            'name': artifact_names[idx] if idx < len(artifact_names) else f"Fragment {len(artifacts)}",
            'contour': art_contour,
            'crop': crop,
            'dimensions': art_dims,
            'px_per_cm': px_per_cm,
        })
        if px_per_cm > ref_px_cm:
            ref_px_cm = px_per_cm

    n = len(artifacts)
    total_images = len(image_paths_meters)
    if n == 0:
        print("Reconstruction failed: no artifacts detected")
        return None

    if n == 1:
        crop = artifacts[0]['crop']
        h, w = crop.shape[:2]
        out = np.ones((h, w, 3), dtype=np.uint8) * 255
        alpha = crop[:, :, 3:4].astype(float) / 255.0
        out = (crop[:, :, :3].astype(float) * alpha +
               out.astype(float) * (1 - alpha)).astype(np.uint8)
        if output_path:
            cv2.imwrite(output_path, out)
        return output_path

    print(f"\nProcessing {n} artifact fragments from separate images")
    print(f"Reference scale: {ref_px_cm:.1f} px/cm")

    artifacts_sorted = sorted(artifacts, key=lambda a: cv2.contourArea(a['contour']), reverse=True)

    def get_collision_ratio(new_x, new_y, new_crop, placed_frags):
        """Return ratio of overlap area to new fragment area. 0 = no overlap."""
        new_h, new_w = new_crop.shape[:2]
        new_area = new_h * new_w
        if new_area == 0:
            return 0.0
        overlap_area = 0.0
        for pf in placed_frags:
            ph, pw = pf['crop'].shape[:2]
            x1 = max(new_x, pf['x'])
            y1 = max(new_y, pf['y'])
            x2 = min(new_x + new_w, pf['x'] + pw)
            y2 = min(new_y + new_h, pf['y'] + ph)
            if x2 > x1 and y2 > y1:
                overlap_area += (x2 - x1) * (y2 - y1)
        return overlap_area / new_area

    def place_fragment(frag_artifact, placed_frags):
        """Try to place frag_artifact next to any placed fragment. Returns (placed_dict) or None."""
        best_score = float("inf")
        best_placement = None
        for pi, pf in enumerate(placed_frags):
            alignment = find_edge_alignment(
                pf['artifact']['contour'],
                frag_artifact['contour'],
                pf['artifact']['px_per_cm'],
                frag_artifact['px_per_cm']
            )
            score = alignment.get('fracture_score', float("inf"))
            if score >= best_score:
                continue

            theta = alignment["rotation"]
            tx, ty = alignment["translation"]
            crop = frag_artifact['crop']
            if crop is None:
                continue
            crop_rot = rotate_image(crop.copy(), -theta)

            bboxP = cv2.boundingRect(pf['artifact']['contour'])
            centP = alignment["centroid_A"]
            crop_cent_P = np.array([centP[0] - bboxP[0], centP[1] - bboxP[1]])

            bboxR = cv2.boundingRect(frag_artifact['contour'])
            centR = alignment["centroid_B"]
            crop_cent_R_orig = np.array([centR[0] - bboxR[0], centR[1] - bboxR[1]])

            pre_rot_w = bboxR[2]
            pre_rot_h = bboxR[3]
            crop_center_R = np.array([pre_rot_w / 2.0, pre_rot_h / 2.0])
            offset = crop_cent_R_orig - crop_center_R
            offset_rot = rotate(offset.reshape(1, 2), theta).flatten()
            h_rot, w_rot = crop_rot.shape[:2]
            crop_cent_R_rot = np.array([w_rot / 2.0, h_rot / 2.0]) + offset_rot

            new_x = pf['x'] + crop_cent_P[0] + tx - crop_cent_R_rot[0]
            new_y = pf['y'] + crop_cent_P[1] + ty - crop_cent_R_rot[1]

            best_score = score
            best_placement = {
                'artifact': frag_artifact,
                'crop': crop_rot,
                'x': new_x,
                'y': new_y,
                'rotation': theta,
                'score': score,
            }

        return best_placement

    COLLISION_THRESHOLD = 0.8

    pair_scores = []
    for i in range(n):
        for j in range(i + 1, n):
            alignment = find_edge_alignment(
                artifacts_sorted[i]['contour'],
                artifacts_sorted[j]['contour'],
                artifacts_sorted[i]['px_per_cm'],
                artifacts_sorted[j]['px_per_cm']
            )
            score = alignment.get('fracture_score', float("inf"))
            pair_scores.append((score, i, j, alignment))

    pair_scores.sort(key=lambda x: x[0])
    print(f"  Pair scores: {[(p[1], p[2], f'{p[0]:.4f}') for p in pair_scores]}")

    for best_score, i, j, init_align in pair_scores:
        frag_i = artifacts_sorted[i]
        frag_j = artifacts_sorted[j]
        crop_i = frag_i['crop']
        crop_j = frag_j['crop']
        if crop_i is None or crop_j is None:
            continue

        theta = init_align["rotation"]
        tx, ty = init_align["translation"]
        crop_j_rot = rotate_image(crop_j.copy(), -theta)

        bboxI = cv2.boundingRect(frag_i['contour'])
        centI = init_align["centroid_A"]
        crop_cent_I = np.array([centI[0] - bboxI[0], centI[1] - bboxI[1]])

        bboxJ = cv2.boundingRect(frag_j['contour'])
        centJ = init_align["centroid_B"]
        crop_cent_J_orig = np.array([centJ[0] - bboxJ[0], centJ[1] - bboxJ[1]])

        pre_rot_w = bboxJ[2]
        pre_rot_h = bboxJ[3]
        crop_center_J = np.array([pre_rot_w / 2.0, pre_rot_h / 2.0])
        offset = crop_cent_J_orig - crop_center_J
        offset_rot = rotate(offset.reshape(1, 2), theta).flatten()
        h_rot, w_rot = crop_j_rot.shape[:2]
        crop_cent_J_rot = np.array([w_rot / 2.0, h_rot / 2.0]) + offset_rot

        j_x = crop_cent_I[0] + tx - crop_cent_J_rot[0]
        j_y = crop_cent_I[1] + ty - crop_cent_J_rot[1]

        print(f"  Trying pair ({i},{j}) score={best_score:.4f}: frag {j} at ({j_x:.1f}, {j_y:.1f})")

        placed = [{
            'artifact': frag_i,
            'crop': crop_i,
            'x': 0.0,
            'y': 0.0,
            'rotation': 0.0,
        }, {
            'artifact': frag_j,
            'crop': crop_j_rot,
            'x': j_x,
            'y': j_y,
            'rotation': -theta,
        }]
        remaining = [k for k in range(n) if k != i and k != j]
        remaining.sort(key=lambda k: cv2.contourArea(artifacts_sorted[k]['contour']), reverse=True)

        print(f"  Pair ({i},{j}) accepted. Placing remaining: {remaining}")
        all_placed = True

        for ri in remaining:
            placement = place_fragment(artifacts_sorted[ri], placed)
            if placement is None:
                print(f"    Could not place fragment {ri}")
                all_placed = False
                break

            collision = get_collision_ratio(placement['x'], placement['y'], placement['crop'], placed)
            print(f"    Fragment {ri}: collision={collision:.2f}, score={placement['score']:.4f}")

            if collision > COLLISION_THRESHOLD:
                print(f"    -> Too much collision, skipping")
                all_placed = False
                break

            placed.append(placement)

        if all_placed:
            print(f"  Pair ({i},{j}) worked! Assembly complete.")
            break
    else:
        print("  All pairs caused too much collision. Placing fragments without constraints.")
        placed = []
        for k in range(n):
            crop = artifacts_sorted[k]['crop']
            if crop is not None:
                placed.append({
                    'artifact': artifacts_sorted[k],
                    'crop': crop,
                    'x': float(k * 150),
                    'y': float(k * 150),
                    'rotation': 0.0,
                })

    remaining = []

    if not placed:
        print("Reconstruction failed: no fragments placed")
        return None

    min_cx = min(p['x'] for p in placed)
    min_cy = min(p['y'] for p in placed)
    max_cx = max(p['x'] + p['crop'].shape[1] for p in placed)
    max_cy = max(p['y'] + p['crop'].shape[0] for p in placed)

    margin = 80
    shift_x = margin - min_cx
    shift_y = margin - min_cy
    canvas_w = int(max_cx - min_cx + margin * 2)
    canvas_h = int(max_cy - min_cy + margin * 2)
    canvas = np.zeros((canvas_h, canvas_w, 4), dtype=np.uint8)

    for p in placed:
        px_pos = int(p['x'] + shift_x)
        py_pos = int(p['y'] + shift_y)
        paste_rgba(canvas, p['crop'], px_pos, py_pos)

    alpha_mask = canvas[:, :, 3]
    rows = np.any(alpha_mask > 0, axis=1)
    cols = np.any(alpha_mask > 0, axis=0)
    if rows.any() and cols.any():
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        pad = 120
        canvas = canvas[max(0, rmin - pad):min(canvas_h, rmax + pad + 1),
                          max(0, cmin - pad):min(canvas_w, cmax + pad + 1)]

    ch, cw = canvas.shape[:2]
    min_side = max(800, cw, ch)
    canvas_h = max(canvas_h, min_side)
    canvas_w = max(canvas_w, min_side)

    output = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255
    if ch > 0 and cw > 0:
        alpha = canvas[:ch, :cw, 3:4].astype(float) / 255.0
        frag_rgb = canvas[:ch, :cw, :3].astype(float)
        white = np.full_like(frag_rgb, 255.0)
        blended = (frag_rgb * alpha + white * (1 - alpha)).clip(0, 255).astype(np.uint8)
        output[:ch, :cw] = blended

    bar_length_cm = 4
    bar_length_px = int(bar_length_cm * ref_px_cm)
    if bar_length_px > canvas_w - 40:
        bar_length_cm = 2
        bar_length_px = int(bar_length_cm * ref_px_cm)
    if bar_length_px > canvas_w - 40:
        bar_length_cm = 1
        bar_length_px = int(bar_length_cm * ref_px_cm)
    bar_x, bar_y = 20, canvas_h - 30
    cv2.line(output, (bar_x, bar_y), (bar_x + bar_length_px, bar_y), (0, 0, 0), 3)
    cv2.line(output, (bar_x, bar_y - 8), (bar_x, bar_y + 8), (0, 0, 0), 2)
    cv2.line(output, (bar_x + bar_length_px, bar_y - 8), (bar_x + bar_length_px, bar_y + 8), (0, 0, 0), 2)
    cv2.putText(output, f"{bar_length_cm} cm", (bar_x + 5, bar_y - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    colors = [(0, 0, 0), (50, 50, 200), (0, 130, 0), (180, 0, 140),
              (0, 140, 140), (140, 100, 0), (80, 80, 80)]
    for i, p in enumerate(placed):
        crop_h, crop_w = p['crop'].shape[:2]
        dims = p['artifact']['dimensions']
        frag_cm = dims.get('width_cm', 1) * dims.get('height_cm', 1)
        font_scale = max(0.15, min(0.4, frag_cm / 10))
        text_thick = max(1, int(font_scale * 2))
        cx = int(p['x'] + shift_x) + crop_w // 2
        cy = int(p['y'] + shift_y) + crop_h // 2
        name = p['artifact']['name']
        text_size = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thick)[0]
        tx = cx - text_size[0] // 2
        ty = cy + text_size[1] // 2
        pad = 2
        cv2.rectangle(output, (tx - pad, ty - text_size[1] - pad), (tx + text_size[0] + pad, ty + pad), (255, 255, 255), -1)
        cv2.rectangle(output, (tx - pad, ty - text_size[1] - pad), (tx + text_size[0] + pad, ty + pad), (0, 0, 0), 1)
        cv2.putText(output, name, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, colors[i % len(colors)], text_thick)

    if output_path:
        cv2.imwrite(output_path, output)
        print(f"\nReconstruction saved to: {output_path}")
    print(f"Canvas size: {canvas_w} x {canvas_h} px")
    print(f"Fragments assembled: {len(placed)}/{total_images} ({n} valid artifacts)")

    return output_path


def compare_artifacts(img1, img2):

    r1 = get_all_artifact_polygons(img1)
    r2 = get_all_artifact_polygons(img2)

    if r1 is None or r1.get("artifact") is None:
        print("Artifact not detected in image 1")
        return None
    if r2 is None or r2.get("artifact") is None:
        print("Artifact not detected in image 2")
        return None

    # Use pixel polygons for shape comparison — position/scale is
    # handled by normalize_polygon inside match_fragments
    poly1 = r1["artifact"]["polygon_px"]
    poly2 = r2["artifact"]["polygon_px"]

    result = match_fragments(poly1, poly2)

    print("\nMATCH RESULT")
    print("------------------")
    print(f"Hausdorff score (normalized): {result['score']:.4f}")
    print(f"Rotation (rad): {result['rotation']:.4f}")

    # After normalization, score is between 0 and ~2
    # (0 = identical shapes, 2 = completely different)
    if result["score"] < 0.15:
        print("Result: Fragments likely MATCH")
    elif result["score"] < 0.35:
        print("Result: Possible match — similar shapes")
    else:
        print("Result: Fragments DO NOT match")

    # Also report real-world size comparison
    dims1 = r1["artifact"]["dimensions"]
    dims2 = r2["artifact"]["dimensions"]
    print(f"\nArtifact 1: {dims1['width_cm']} x {dims1['height_cm']} cm "
          f"(area: {dims1['area_cm2']} cm²)")
    print(f"Artifact 2: {dims2['width_cm']} x {dims2['height_cm']} cm "
          f"(area: {dims2['area_cm2']} cm²)")

    visualize_match(poly1, poly2, result)

    # Generate reconstruction image
    recon_path = reconstruct_artifact(img1, img2)
    result["reconstruction"] = recon_path

    return result


if __name__ == "__main__":
    import sys

    def print_usage():
        print("Usage:")
        print("  Single image with N fragments:")
        print("    python InstanceSegmentation.py --multi <image> [output.png]")
        print("")
        print("  Two images (one fragment each), compare + reconstruct:")
        print("    python InstanceSegmentation.py <image1> <image2> [output.png]")

    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)

    if sys.argv[1] == "--multi":
        # Single-image multi-fragment mode
        if len(sys.argv) < 3:
            print_usage()
            sys.exit(1)
        image_path = sys.argv[2]
        output_path = sys.argv[3] if len(sys.argv) >= 4 else None
        reconstruct_multi(image_path, output_path)

    elif len(sys.argv) >= 3:
        # Two-image comparison mode (original behavior)
        image_path1 = sys.argv[1]
        image_path2 = sys.argv[2]
        compare_artifacts(image_path1, image_path2)

    else:
        print_usage()
        sys.exit(1)