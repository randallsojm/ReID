import json
import math
from collections import defaultdict
import shutil
import os

DATASET_PATH     = "car1_dataset_moving_clean/detections.json"
OUTPUT_PATH      = "car1_dataset_moving_clean/candidates.json"
PIXEL_ERROR      = 4.0
MIN_PITCH_ABS    = 45.0
THRESHOLD_M      = 3.0
PITCH_POWER      = 2.0
MAX_SPEED_MS     = 20.0   # m/s (72 km/h)
CAPTURE_INTERVAL = 2.0    # seconds between frames
MAX_TIME_DIFF_S  = 6.0    # only compare frames within this window
#                           0s = same frame, 2-6s = cross-frame

CAMERA_ANGLES = {
    "chase_top"         : {"pitch": -90, "yaw":    0},
    "angled_front"      : {"pitch": -50, "yaw":    0},
    "angled_front_left" : {"pitch": -60, "yaw":  -45},
    "angled_front_right": {"pitch": -60, "yaw":   45},
    "angled_left"       : {"pitch": -70, "yaw":  -90},
    "angled_right"      : {"pitch": -70, "yaw":   90},
    "angled_back"       : {"pitch": -50, "yaw":  180},
    "angled_back_left"  : {"pitch": -60, "yaw": -135},
    "angled_back_right" : {"pitch": -60, "yaw":  135},
    "chase_right"       : {"pitch": -70, "yaw": -135},
}

COMPARE_CAMERAS = [
    "chase_top", "angled_front", "angled_front_left", "angled_front_right",
    "angled_left", "angled_right", "angled_back", "angled_back_left",
    "angled_back_right", "chase_right",
]


# ── Math helpers ──────────────────────────────────────────────────────────────

def mat_vec(M, v):
    return [
        M[0][0]*v[0] + M[0][1]*v[1] + M[0][2]*v[2],
        M[1][0]*v[0] + M[1][1]*v[1] + M[1][2]*v[2],
        M[2][0]*v[0] + M[2][1]*v[1] + M[2][2]*v[2],
    ]

def quat_to_rotation_matrix(w, x, y, z):
    return [
        [1-2*(y*y+z*z),   2*(x*y-z*w),   2*(x*z+y*w)],
        [  2*(x*y+z*w), 1-2*(x*x+z*z),   2*(y*z-x*w)],
        [  2*(x*z-y*w),   2*(y*z+x*w), 1-2*(x*x+y*y)],
    ]

def ned_to_gps(origin, ned_x, ned_y):
    R     = 6378137.0
    d_lat = (ned_x / R) * (180.0 / math.pi)
    d_lon = (ned_y / (R * math.cos(math.radians(origin["lat"])))) * (180.0 / math.pi)
    return origin["lat"] + d_lat, origin["lon"] + d_lon

def haversine(lat1, lon1, lat2, lon2):
    R    = 6378137.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a    = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

def compute_margin(pitch_deg, cam_pose, car1_ned, fx):
    sin_p = abs(math.sin(math.radians(pitch_deg)))
    if sin_p < 1e-6:
        return 0.0
    h = abs(car1_ned["z"] - cam_pose["pos_z"])
    return round(PIXEL_ERROR * h / (fx * sin_p), 3)

def compute_threshold(pitch_a, pitch_b, margin_a, margin_b):
    def camera_thresh(p):
        s = abs(math.sin(math.radians(p)))
        return THRESHOLD_M / (s ** 2) if s > 1e-6 else THRESHOLD_M
    return round(camera_thresh(pitch_a) + camera_thresh(pitch_b) + margin_a + margin_b, 3)

def is_trusted(pitch_deg):
    return abs(pitch_deg) >= MIN_PITCH_ABS


# ── GPS projection ────────────────────────────────────────────────────────────

def project_gps(det, origin, fx, W, H):
    cam_pose = det["camera_pose"]
    cx, cy   = det["bbox"]["cx"], det["bbox"]["cy"]
    obj_alt  = det["detected_gps"]["alt"]
    ground_z = -(obj_alt - origin["alt"])
    ray_cam  = [1.0, (cx - W/2) / fx, (cy - H/2) / fx]
    R        = quat_to_rotation_matrix(
        cam_pose["ori_w"], cam_pose["ori_x"],
        cam_pose["ori_y"], cam_pose["ori_z"]
    )
    ray_world  = mat_vec(R, ray_cam)
    dz         = ray_world[2]
    if dz < 0.1:
        return None, None
    cam_height = ground_z - cam_pose["pos_z"]
    if cam_height <= 0:
        return None, None
    t = cam_height / dz
    if t > cam_height * 5:
        return None, None
    ned_x = cam_pose["pos_x"] + t * ray_world[0]
    ned_y = cam_pose["pos_y"] + t * ray_world[1]
    return ned_to_gps(origin, ned_x, ned_y)


# ── Threshold logic ───────────────────────────────────────────────────────────

# ── SIMULATION THRESHOLDS (active) ────────────────────────────────────────────
SAME_FRAME_THRESH_M = 5.0   # fixed 1.0m for same-frame pairs
CROSS_FRAME_FACTOR  = 1.5   # threshold = 2.0 × time_diff (seconds)
#                             Δt=2s → 2.0m, Δt=4s → 4.0m, Δt=6s → 6.0m
# ─────────────────────────────────────────────────────────────────────────────

def get_threshold(obs_i, obs_j, frame_i, frame_j):
    """
    Simplified simulation threshold — no GPS error buffer, no pitch formula.

    Same frame  (time_diff=0): fixed 1.0m
    Cross frame (time_diff>0): 1.0 × time_diff in seconds

    # ── ORIGINAL REAL-WORLD THRESHOLD (comment back in for real deployment) ──
    # time_diff  = abs(frame_j - frame_i) * CAPTURE_INTERVAL
    # err_buffer = obs_i['error_m'] + obs_j['error_m']
    # if time_diff == 0:
    #     thresh = compute_threshold(
    #         obs_i['pitch'], obs_j['pitch'],
    #         obs_i['margin'], obs_j['margin']
    #     )
    #     return round(thresh + err_buffer, 3), 'same_frame'
    # else:
    #     radius = MAX_SPEED_MS * time_diff + err_buffer
    #     return round(radius, 3), 'cross_frame'
    # ─────────────────────────────────────────────────────────────────────────
    """
    time_diff = abs(frame_j - frame_i) * CAPTURE_INTERVAL

    if time_diff == 0:
        return SAME_FRAME_THRESH_M, 'same_frame'
    else:
        return round(SAME_FRAME_THRESH_M + CROSS_FRAME_FACTOR * time_diff, 3), 'cross_frame'


# ── Index builder ─────────────────────────────────────────────────────────────

def build_index(all_frames_data, raw_frames):
    """
    Build lookup dicts for O(1) access during comparison.

    index[(frame_id, cam)]           → list of vehicle obs dicts
    frame_pairs                      → list of (frame_i, frame_j) to compare
                                       includes same-frame pairs (i==j)
    raw_lookup[(frame_id, cam, name)]→ raw detection dict
    """
    index      = defaultdict(list)
    raw_lookup = {}

    for frame_id, by_name in all_frames_data:
        for name, cams in by_name.items():
            for cam, obs in cams.items():
                index[(frame_id, cam)].append({
                    'name'    : name,
                    'pred_lat': obs['pred_lat'],
                    'pred_lon': obs['pred_lon'],
                    'error_m' : obs['error_m'],
                    'pitch'   : obs['pitch'],
                    'margin'  : obs['margin'],
                })

    for frame in raw_frames:
        for det in frame['detections']:
            raw_lookup[(frame['frame'], det['camera'], det['object'])] = det

    # build all valid frame pairs including same-frame (i==j)
    frame_ids  = sorted(set(fid for fid, _ in all_frames_data))
    frame_pairs = []
    for i, fi in enumerate(frame_ids):
        for fj in frame_ids[i:]:             # start at i not i+1 → includes fi==fj
            time_diff = abs(fj - fi) * CAPTURE_INTERVAL
            if time_diff > MAX_TIME_DIFF_S:
                break
            frame_pairs.append((fi, fj))

    return index, frame_pairs, raw_lookup


# ── Unified comparison ────────────────────────────────────────────────────────

def analyse_all(all_frames_data, raw_frames):
    """
    Single unified function that compares ALL vehicle pairs across:
      - same frame, any camera combination     (time_diff = 0)
      - cross frame within MAX_TIME_DIFF_S,
        any camera combination                 (time_diff = 2,4,6s)

    Skips (same frame, same camera) — comparing an image to itself.

    Every candidate has:
      frame_i, frame_j, cam_i, cam_j
      object_i, object_j
      dist_m, threshold_m, time_diff_s
      match_type: 'same_frame' or 'cross_frame'
      spatial_match: dist <= threshold
      same_object: ground truth
      view_i, view_j: image_file, bbox, pred_lat/lon, err_m
    """
    index, frame_pairs, raw_lookup = build_index(all_frames_data, raw_frames)

    candidates      = []   # spatial_match=True  → pass to CNN
    true_positives  = []   # spatial_match AND same_object
    false_positives = []   # spatial_match AND NOT same_object
    false_negatives = []   # NOT spatial_match AND same_object

    for frame_i, frame_j in frame_pairs:
        cams_i = [c for c in COMPARE_CAMERAS if index[(frame_i, c)]]
        cams_j = [c for c in COMPARE_CAMERAS if index[(frame_j, c)]]

        for cam_i in cams_i:
            for cam_j in cams_j:

                # skip same image — no point comparing to itself
                if frame_i == frame_j and cam_i == cam_j:
                    continue

                for di in index[(frame_i, cam_i)]:
                    for dj in index[(frame_j, cam_j)]:

                        dist = round(haversine(
                            di['pred_lat'], di['pred_lon'],
                            dj['pred_lat'], dj['pred_lon']
                        ), 2)

                        threshold, match_type = get_threshold(
                            di, dj, frame_i, frame_j
                        )

                        spatial_match = dist <= threshold
                        same_object   = (di['name'] == dj['name'])

                        # skip pairs neither matched nor should have matched
                        if not spatial_match and not same_object:
                            continue

                        det_i = raw_lookup.get((frame_i, cam_i, di['name']))
                        det_j = raw_lookup.get((frame_j, cam_j, dj['name']))

                        record = {
                            'frame_i'      : frame_i,
                            'frame_j'      : frame_j,
                            'cam_i'        : cam_i,
                            'cam_j'        : cam_j,
                            'object_i'     : di['name'],
                            'object_j'     : dj['name'],
                            'dist_m'       : dist,
                            'threshold_m'  : threshold,
                            'time_diff_s'  : abs(frame_j - frame_i) * CAPTURE_INTERVAL,
                            'match_type'   : match_type,
                            'spatial_match': spatial_match,
                            'same_object'  : same_object,
                            'err_i'        : di['error_m'],
                            'err_j'        : dj['error_m'],
                            'view_i'       : {
                                'image_file': det_i['image_file'] if det_i else None,
                                'array_file': det_i['array_file'] if det_i else None,
                                'bbox'      : det_i['bbox']       if det_i else None,
                                'pred_lat'  : round(di['pred_lat'], 6),
                                'pred_lon'  : round(di['pred_lon'], 6),
                                'err_m'     : di['error_m'],
                            },
                            'view_j'       : {
                                'image_file': det_j['image_file'] if det_j else None,
                                'array_file': det_j['array_file'] if det_j else None,
                                'bbox'      : det_j['bbox']       if det_j else None,
                                'pred_lat'  : round(dj['pred_lat'], 6),
                                'pred_lon'  : round(dj['pred_lon'], 6),
                                'err_m'     : dj['error_m'],
                            },
                        }

                        if spatial_match:
                            candidates.append(record)
                        if spatial_match and same_object:
                            true_positives.append(record)
                        elif spatial_match and not same_object:
                            false_positives.append(record)
                        elif not spatial_match and same_object:
                            false_negatives.append(record)

    return candidates, true_positives, false_positives, false_negatives


# ── Per-frame processing ──────────────────────────────────────────────────────

def process_frame(frame, origin, fx, W, H):
    frame_id = frame["frame"]
    by_name  = {}
    for det in frame["detections"]:
        name = det["object"]
        cam  = det["camera"]
        if cam not in COMPARE_CAMERAS:
            continue
        config_pitch = CAMERA_ANGLES[cam]["pitch"]
        if not is_trusted(config_pitch):
            continue
        pred_lat, pred_lon = project_gps(det, origin, fx, W, H)
        if pred_lat is None:
            continue
        margin = compute_margin(config_pitch, det["camera_pose"], det["car1_ned"], fx)
        gt_lat = det["detected_gps"]["lat"]
        gt_lon = det["detected_gps"]["lon"]
        err_m  = round(haversine(pred_lat, pred_lon, gt_lat, gt_lon), 2)
        if name not in by_name:
            by_name[name] = {}
        by_name[name][cam] = {
            "pred_lat": pred_lat,
            "pred_lon": pred_lon,
            "gt_lat"  : gt_lat,
            "gt_lon"  : gt_lon,
            "margin"  : margin,
            "error_m" : err_m,
            "pitch"   : config_pitch,
        }
    return frame_id, by_name


# ── Reporting ─────────────────────────────────────────────────────────────────

def print_report(candidates, true_positives, false_positives, false_negatives):
    same  = [r for r in candidates if r['match_type'] == 'same_frame']
    cross = [r for r in candidates if r['match_type'] == 'cross_frame']

    total = len(true_positives) + len(false_positives) + len(false_negatives)
    prec  = round(len(true_positives) / (len(true_positives) + len(false_positives)) * 100, 1) \
            if (true_positives or false_positives) else 0
    rec   = round(len(true_positives) / (len(true_positives) + len(false_negatives)) * 100, 1) \
            if (true_positives or false_negatives) else 0

    print(f"\n{'═'*70}")
    print(f" RESULTS SUMMARY")
    print(f"{'═'*70}")
    print(f"  Total pairs evaluated      : {total}")
    print(f"  Spatial matches (candidates): {len(candidates)}")
    print(f"    same-frame pairs          : {len(same)}")
    print(f"    cross-frame pairs         : {len(cross)}")
    print(f"  True positives             : {len(true_positives)}")
    print(f"  False positives            : {len(false_positives)}  ← CNN will filter")
    print(f"  False negatives (missed)   : {len(false_negatives)}")
    print(f"  Precision                  : {prec}%")
    print(f"  Recall                     : {rec}%")

    print(f"\n TRUE POSITIVES [{len(true_positives)}]")
    for r in true_positives[:20]:
        print(f"  f{r['frame_i']:04d}({r['cam_i']:12s}) ↔ "
              f"f{r['frame_j']:04d}({r['cam_j']:12s}) | "
              f"{r['object_i']:10s} | dist={r['dist_m']:.1f}m "
              f"thresh={r['threshold_m']:.1f}m Δt={r['time_diff_s']:.0f}s ✓")
    if len(true_positives) > 20:
        print(f"  ... and {len(true_positives)-20} more")

    print(f"\n FALSE POSITIVES [{len(false_positives)}]")
    for r in false_positives[:10]:
        print(f"  f{r['frame_i']:04d}({r['cam_i']:12s}) ↔ "
              f"f{r['frame_j']:04d}({r['cam_j']:12s}) | "
              f"{r['object_i']:10s} vs {r['object_j']:10s} | "
              f"dist={r['dist_m']:.1f}m thresh={r['threshold_m']:.1f}m ✗")
    if len(false_positives) > 10:
        print(f"  ... and {len(false_positives)-10} more")

    print(f"\n FALSE NEGATIVES (missed) [{len(false_negatives)}]")
    for r in false_negatives[:10]:
        print(f"  f{r['frame_i']:04d}({r['cam_i']:12s}) ↔ "
              f"f{r['frame_j']:04d}({r['cam_j']:12s}) | "
              f"{r['object_i']:10s} | dist={r['dist_m']:.1f}m > "
              f"thresh={r['threshold_m']:.1f}m  ← outside radius")
    if len(false_negatives) > 10:
        print(f"  ... and {len(false_negatives)-10} more")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    with open(DATASET_PATH) as f:
        data = json.load(f)

    meta   = data["metadata"]
    origin = meta["origin_gps"]
    W      = meta["image_width"]
    H      = meta["image_height"]
    FOV    = meta["fov_degrees"]
    fx     = (W / 2.0) / math.tan(math.radians(FOV / 2.0))

    print(f"[LOADED]  {meta['total_frames']} frames")
    print(f"[IMAGE]   {W}x{H}  FOV={FOV}  fx={fx:.2f}")
    print(f"[CAMERAS] {len(COMPARE_CAMERAS)} cameras")
    print(f"[THRESH]  base={THRESHOLD_M}m  pitch_power={PITCH_POWER}")
    print(f"[REACH]   max_speed={MAX_SPEED_MS}m/s  "
          f"interval={CAPTURE_INTERVAL}s  max_window={MAX_TIME_DIFF_S}s")

    # process all frames
    all_frames_data = []
    for frame in data["frames"]:
        frame_id, by_name = process_frame(frame, origin, fx, W, H)
        all_frames_data.append((frame_id, by_name))

    # single unified comparison
    candidates, tp, fp, fn = analyse_all(all_frames_data, data["frames"])

    # print report
    print_report(candidates, tp, fp, fn)

    # save single output file — clean JSON, no comment noise
    output = {
        "summary": {
            "total_pairs"    : len(tp) + len(fp) + len(fn),
            "candidates"     : len(candidates),
            "same_frame"     : sum(1 for r in candidates if r['match_type']=='same_frame'),
            "cross_frame"    : sum(1 for r in candidates if r['match_type']=='cross_frame'),
            "true_positives" : len(tp),
            "false_positives": len(fp),
            "false_negatives": len(fn),
            "precision"      : round(len(tp)/(len(tp)+len(fp))*100,1) if (tp or fp) else 0,
            "recall"         : round(len(tp)/(len(tp)+len(fn))*100,1) if (tp or fn) else 0,
        },
        "candidates": candidates,
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    # write plain-text readme alongside the JSON with real line breaks
    readme_path = OUTPUT_PATH.replace(".json", "_README.txt")
    with open(readme_path, "w") as f:
        f.write(f"""\
candidates.json — output of analysis.py
{'='*60}

WHAT THIS FILE IS
    Every pair of vehicle detections that passed the spatial
    filter. These are candidates to pass to the CNN ReID model
    for identity verification.

SETTINGS USED
    MAX_SPEED_MS     = {MAX_SPEED_MS} m/s  ({MAX_SPEED_MS*3.6:.0f} km/h)
    CAPTURE_INTERVAL = {CAPTURE_INTERVAL} s between frames
    MAX_TIME_DIFF_S  = {MAX_TIME_DIFF_S} s  (max window to compare)
    THRESHOLD_M      = {THRESHOLD_M} m  (base GPS threshold)
    PIXEL_ERROR      = {PIXEL_ERROR} px

FIELD DEFINITIONS
    frame_i / frame_j
        Frame numbers of the two detections being compared.

    cam_i / cam_j
        Camera names that captured each detection.

    object_i / object_j
        Vehicle name from ground truth labels.

    dist_m
        GPS distance in metres between the two projected
        positions.

    threshold_m
        Maximum allowed distance to pass the spatial filter.
        same_frame  -> pitch-based formula (tight)
        cross_frame -> reachability circle
                       = max_speed x time_diff + gps_error

    time_diff_s
        Seconds between the two frames. 0 = same frame.

    match_type
        same_frame  -- both detections from the same frame,
                       different cameras.
        cross_frame -- detections from different frames
                       within MAX_TIME_DIFF_S.

    spatial_match
        True if dist_m <= threshold_m.
        Always True for entries in the candidates list.

    same_object
        Ground truth. True if both detections are the same
        physical vehicle.

    err_i / err_j
        GPS projection error in metres for each detection.
        How far the predicted lat/lon is from the true GPS.
        Included as a buffer in threshold_m.

    view_i / view_j  (one block per detection)
        image_file  path to the full camera image
        array_file  path to the depth array (.npy)
        bbox        bounding box in pixels
                    x0,y0 = top-left corner
                    x1,y1 = bottom-right corner
                    cx,cy = centre (used for GPS projection)
        pred_lat    projected GPS latitude of the vehicle
        pred_lon    projected GPS longitude of the vehicle
        err_m       GPS projection error for this detection

HOW TO USE
    For each candidate:
    1. Crop view_i image to view_i bbox
    2. Crop view_j image to view_j bbox
    3. Extract ReID embedding for each crop
    4. Compute cosine similarity between embeddings
    5. score > 0.85  -> confirmed same vehicle
       score <= 0.85 -> different vehicle (false positive)

INTERPRETING same_object vs spatial_match
    spatial_match=True  same_object=True  -> true positive
    spatial_match=True  same_object=False -> false positive (CNN rejects)
    spatial_match=False same_object=True  -> false negative (not in file)
""")

    print(f"\n[SAVED]  -> {OUTPUT_PATH}  ({len(candidates)} candidates)")
    print(f"[SAVED]  -> {readme_path}")

        # ── Copy CNN candidate images to separate folder ───────────────────────────
    CNN_IMAGES_DIR = OUTPUT_PATH.replace("candidates.json", "cnn_images")
    os.makedirs(CNN_IMAGES_DIR, exist_ok=True)

    all_images = set()
    for c in candidates:
        if c['view_i']['image_file']:
            all_images.add(c['view_i']['image_file'])
        if c['view_j']['image_file']:
            all_images.add(c['view_j']['image_file'])

    source_dir = os.path.join(os.path.dirname(OUTPUT_PATH), "images")
    copied, missing = 0, 0
    for img_path in sorted(all_images):
        filename = os.path.basename(img_path)
        src = os.path.join(source_dir, filename)
        dst = os.path.join(CNN_IMAGES_DIR, filename)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            copied += 1
        else:
            missing += 1

    print(f"[SAVED]  -> {CNN_IMAGES_DIR}/  ({copied} images copied, {missing} missing)")
    # ──────────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    main()
