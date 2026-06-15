import json
import math

DATASET_PATH  = "car1_dataset/detections.json"
PIXEL_ERROR   = 4.0
MIN_PITCH_ABS = 45.0
THRESHOLD_M   = 3.0
PITCH_POWER   = 2.0

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

def get_intrinsic_matrix(width, height, fov_deg):
    fov_rad = math.radians(fov_deg)
    fx = (width / 2.0) / math.tan(fov_rad / 2.0)
    cx = width  / 2.0
    cy = height / 2.0
    return [[fx, 0, cx], [0, fx, cy], [0, 0, 1]]

def inv3x3(M):
    a,b,c = M[0]; d,e,f = M[1]; g,h,i = M[2]
    det = a*(e*i-f*h) - b*(d*i-f*g) + c*(d*h-e*g)
    if abs(det) < 1e-10:
        return None
    return [
        [(e*i-f*h)/det, -(b*i-c*h)/det,  (b*f-c*e)/det],
        [-(d*i-f*g)/det, (a*i-c*g)/det, -(a*f-c*d)/det],
        [(d*h-e*g)/det, -(a*h-b*g)/det,  (a*e-b*d)/det],
    ]

def mat_vec(M, v):
    return [
        M[0][0]*v[0] + M[0][1]*v[1] + M[0][2]*v[2],
        M[1][0]*v[0] + M[1][1]*v[1] + M[1][2]*v[2],
        M[2][0]*v[0] + M[2][1]*v[1] + M[2][2]*v[2],
    ]

def mm3x3(A, B):
    return [
        [sum(A[i][k]*B[k][j] for k in range(3)) for j in range(3)]
        for i in range(3)
    ]

def quat_to_rotation_matrix(w, x, y, z):
    """
    Convert quaternion to 3x3 rotation matrix.
    Uses world-space quaternion from simGetCameraInfo.
    """
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
    """
    Estimate pixel projection error in metres.
    Uses actual camera height above object ground level.
    """
    sin_p = abs(math.sin(math.radians(pitch_deg)))
    if sin_p < 1e-6:
        return 0.0
    h = abs(car1_ned["z"] - cam_pose["pos_z"])
    return round(PIXEL_ERROR * h / (fx * sin_p), 3)

def compute_threshold(pitch_a, pitch_b, margin_a, margin_b):
    """
    Simpler threshold now that projection errors are low (sub-metre to 4m).
    Base threshold accounts for projection error from both cameras.
    Margin adds pixel uncertainty based on height and pitch.
    """
    def camera_thresh(p):
        s = abs(math.sin(math.radians(p)))
        # Gentle exponent — pitch_power=2 instead of 24
        return THRESHOLD_M / (s ** 2) if s > 1e-6 else THRESHOLD_M
    return round(camera_thresh(pitch_a) + camera_thresh(pitch_b) + margin_a + margin_b, 3)

def is_trusted(pitch_deg):
    return abs(pitch_deg) >= MIN_PITCH_ABS


# ── Projection ────────────────────────────────────────────────────────────────
#
# AirSim camera coordinate system (confirmed by testing):
#   X = forward (out of lens)  ← NOT Z like standard OpenCV
#   Y = right
#   Z = down
#
# So a pixel (cx, cy) maps to camera-space ray:
#   forward (X) = 1.0
#   right   (Y) = (cx - W/2) / fx
#   down    (Z) = (cy - H/2) / fx
#
# This ray is then rotated by the world-space quaternion (from simGetCameraInfo)
# to get the NED world-space ray direction.
#
# Ground plane = object's own GPS altitude converted to NED z.
# This handles hills and slopes correctly.

def project_gps(det, origin, fx, W, H):
    cam_pose = det["camera_pose"]
    cx, cy   = det["bbox"]["cx"], det["bbox"]["cy"]

    # Object's ground plane in NED
    obj_alt  = det["detected_gps"]["alt"]
    ground_z = -(obj_alt - origin["alt"])

    # AirSim camera axes: X=forward, Y=right, Z=down
    ray_cam = [
        1.0,              # forward
        (cx - W/2) / fx,  # right
        (cy - H/2) / fx,  # down
    ]

    # Rotate to NED world space using quaternion
    R = quat_to_rotation_matrix(
        cam_pose["ori_w"], cam_pose["ori_x"],
        cam_pose["ori_y"], cam_pose["ori_z"]
    )
    ray_world = mat_vec(R, ray_cam)
    dz        = ray_world[2]  # positive = pointing down in NED

    # Reject rays not pointing meaningfully downward
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

        margin = compute_margin(
            config_pitch,
            det["camera_pose"], det["car1_ned"], fx
        )

        gt_lat = det["detected_gps"]["lat"]
        gt_lon = det["detected_gps"]["lon"]
        err_m  = round(haversine(pred_lat, pred_lon, gt_lat, gt_lon), 2)

        if name not in by_name:
            by_name[name] = {}

        by_name[name][cam] = {
            "pred_lat" : pred_lat,
            "pred_lon" : pred_lon,
            "gt_lat"   : gt_lat,
            "gt_lon"   : gt_lon,
            "margin"   : margin,
            "error_m"  : err_m,
            "pitch"    : config_pitch,
        }

    return frame_id, by_name


# ── Frame analysis ────────────────────────────────────────────────────────────

def analyse_frame(frame_id, by_name):
    correct        = []
    false_positive = []
    missed         = []

    top_dets   = {name: obs["chase_top"]   for name, obs in by_name.items() if "chase_top"   in obs}
    right_dets = {name: obs["chase_right"] for name, obs in by_name.items() if "chase_right" in obs}

    for name_t, top in top_dets.items():
        for name_r, right in right_dets.items():
            dist   = haversine(
                top["pred_lat"], top["pred_lon"],
                right["pred_lat"], right["pred_lon"]
            )
            thresh = compute_threshold(
                top["pitch"], right["pitch"],
                top["margin"], right["margin"]
            )
            estimated = dist <= thresh
            actual    = (name_t == name_r)

            if not estimated and not actual:
                continue

            record = {
                "frame"        : frame_id,
                "object_top"   : name_t,
                "object_right" : name_r,
                "dist_m"       : round(dist, 2),
                "threshold"    : thresh,
                "estimated"    : estimated,
                "actual"       : actual,
                "err_top"      : top["error_m"],
                "err_right"    : right["error_m"],
                "margin_top"   : top["margin"],
                "margin_right" : right["margin"],
                "pred_top"     : (round(top["pred_lat"], 6), round(top["pred_lon"], 6)),
                "pred_right"   : (round(right["pred_lat"], 6), round(right["pred_lon"], 6)),
            }

            if estimated and actual:
                correct.append(record)
            elif estimated and not actual:
                false_positive.append(record)
            elif not estimated and actual:
                missed.append(record)

    return correct, false_positive, missed


# ── Reporting ─────────────────────────────────────────────────────────────────

def fmt(r):
    return (
        f"  frame {r['frame']:04d} | "
        f"top={r['object_top']:15s} ↔ right={r['object_right']:15s} | "
        f"dist={r['dist_m']:6.2f}m | thresh={r['threshold']:.2f}m | "
        f"err_top={r['err_top']}m  err_right={r['err_right']}m"
    )

def print_per_frame_report(all_frames_data):
    print(f"\n{'═'*70}")
    print(f" PER-FRAME REPORT — chase_top vs chase_right")
    print(f" threshold = {THRESHOLD_M}m / sin(|pitch|)^{PITCH_POWER:.0f} + margin")
    print(f" -90° → {THRESHOLD_M:.1f}m base  |  -70° → ~30m base")
    print(f"{'═'*70}")

    for frame_id, by_name in all_frames_data:
        has_both = any(
            "chase_top" in cams and "chase_right" in cams
            for cams in by_name.values()
        )
        if not has_both:
            continue

        print(f"\nFrame {frame_id:04d}:")

        top_dets   = {n: obs["chase_top"]   for n, obs in by_name.items() if "chase_top"   in obs}
        right_dets = {n: obs["chase_right"] for n, obs in by_name.items() if "chase_right" in obs}

        print(f"  chase_top   detections : {list(top_dets.keys())}")
        print(f"  chase_right detections : {list(right_dets.keys())}")

        for name_t, top in sorted(top_dets.items()):
            for name_r, right in sorted(right_dets.items()):
                dist   = haversine(
                    top["pred_lat"], top["pred_lon"],
                    right["pred_lat"], right["pred_lon"]
                )
                thresh = compute_threshold(
                    top["pitch"], right["pitch"],
                    top["margin"], right["margin"]
                )
                estimated = dist <= thresh
                actual    = (name_t == name_r)

                if not estimated and not actual:
                    continue

                est_tag = "TRUE ✓" if estimated else "false  "
                act_tag = "TRUE ✓" if actual    else "false  "
                fp_tag  = "  ← FALSE POSITIVE" if (estimated and not actual) else ""
                ms_tag  = "  ← MISSED"         if (not estimated and actual) else ""

                print(f"  {name_t:15s}(top) ↔ {name_r:15s}(right) | "
                      f"dist={dist:.2f}m thresh={thresh:.2f}m | "
                      f"est={est_tag} act={act_tag}{fp_tag}{ms_tag}")


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
    print(f"[IMAGE]   {W}x{H}  FOV={FOV}°  fx={fx:.2f}")
    print(f"[CAMERAS] {COMPARE_CAMERAS}")
    print(f"[PROJ]    AirSim axes (X=forward) + world-space quaternion")
    print(f"[GROUND]  per-object GPS altitude → NED z")
    print(f"[THRESH]  base={THRESHOLD_M}m  pitch_power={PITCH_POWER}")
    print(f"[MARGIN]  pixel_error=±{PIXEL_ERROR}px")

    all_correct        = []
    all_false_positive = []
    all_missed         = []
    all_frames_data    = []

    for frame in data["frames"]:
        frame_id, by_name = process_frame(frame, origin, fx, W, H)
        c, fp, m          = analyse_frame(frame_id, by_name)
        all_correct        += c
        all_false_positive += fp
        all_missed         += m
        all_frames_data.append((frame_id, by_name))

    # ── Per-frame report ───────────────────────────────────────────────────────
    print_per_frame_report(all_frames_data)

    # ── Summary sections ───────────────────────────────────────────────────────
    print(f"\n{'═'*70}")
    print(f" CORRECTLY PREDICTED  (estimated=True AND actual=True)")
    print(f"{'═'*70}")
    if not all_correct:
        print("  none")
    for r in all_correct:
        print(fmt(r))

    print(f"\n{'═'*70}")
    print(f" FALSE POSITIVES  (estimated=True AND actual=False)")
    print(f"{'═'*70}")
    if not all_false_positive:
        print("  none")
    for r in all_false_positive:
        print(fmt(r))

    print(f"\n{'═'*70}")
    print(f" MISSED  (actual=True AND estimated=False)")
    print(f"{'═'*70}")
    if not all_missed:
        print("  none")
    for r in all_missed:
        print(fmt(r))

    # ── Summary stats ──────────────────────────────────────────────────────────
    total = len(all_correct) + len(all_false_positive) + len(all_missed)
    prec  = round(len(all_correct) / (len(all_correct) + len(all_false_positive)) * 100, 1) \
            if (all_correct or all_false_positive) else 0
    rec   = round(len(all_correct) / (len(all_correct) + len(all_missed)) * 100, 1) \
            if (all_correct or all_missed) else 0

    print(f"\n{'═'*70}")
    print(f" SUMMARY")
    print(f"{'═'*70}")
    print(f"  Total pairs evaluated            : {total}")
    print(f"  Correctly predicted              : {len(all_correct)}")
    print(f"  False positives                  : {len(all_false_positive)}")
    print(f"  Missed                           : {len(all_missed)}")
    print(f"  Precision                        : {prec}%")
    print(f"  Recall                           : {rec}%")

    # ── CNN candidates ─────────────────────────────────────────────────────────
    estimated_true = [r for r in all_correct + all_false_positive if r["estimated"]]

    print(f"\n{'═'*70}")
    print(f" CNN CANDIDATES  (estimated=True — pass to CNN for verification)")
    print(f"{'═'*70}")
    if not estimated_true:
        print("  none")
    for r in estimated_true:
        act_tag = "actual=TRUE ✓" if r["actual"] else "actual=false ← needs CNN"
        print(f"  frame {r['frame']:04d} | "
              f"top={r['object_top']:15s} ↔ right={r['object_right']:15s} | "
              f"dist={r['dist_m']:6.2f}m | {act_tag}")

    print(f"\n  Total CNN candidates : {len(estimated_true)}")
    print(f"  Confirmed (actual=T) : {sum(1 for r in estimated_true if r['actual'])}")
    print(f"  Needs CNN verify     : {sum(1 for r in estimated_true if not r['actual'])}")

    # ── Build and save CNN candidates JSON ─────────────────────────────────────
    cnn_candidates = []
    for r in estimated_true:
        frame_id   = r["frame"]
        name_top   = r["object_top"]
        name_right = r["object_right"]

        frame_data = next((f for f in data["frames"] if f["frame"] == frame_id), None)
        if frame_data is None:
            continue

        top_det   = None
        right_det = None
        for det in frame_data["detections"]:
            if det["camera"] == "chase_top"   and det["object"] == name_top:
                top_det = det
            if det["camera"] == "chase_right" and det["object"] == name_right:
                right_det = det

        cnn_candidates.append({
            "frame"        : frame_id,
            "object_top"   : name_top,
            "object_right" : name_right,
            "same_object"  : r["actual"],
            "dist_m"       : r["dist_m"],
            "threshold"    : r["threshold"],
            "chase_top"    : {
                "image_file" : top_det["image_file"]  if top_det else None,
                "array_file" : top_det["array_file"]  if top_det else None,
                "bbox"       : top_det["bbox"]         if top_det else None,
                "pred_lat"   : r["pred_top"][0],
                "pred_lon"   : r["pred_top"][1],
                "err_m"      : r["err_top"],
            },
            "chase_right"  : {
                "image_file" : right_det["image_file"] if right_det else None,
                "array_file" : right_det["array_file"] if right_det else None,
                "bbox"       : right_det["bbox"]        if right_det else None,
                "pred_lat"   : r["pred_right"][0],
                "pred_lon"   : r["pred_right"][1],
                "err_m"      : r["err_right"],
            },
        })

    # ── Save all files ─────────────────────────────────────────────────────────
    with open("car1_dataset/correct.json", "w") as f:
        json.dump(all_correct, f, indent=2)
    with open("car1_dataset/false_positives.json", "w") as f:
        json.dump(all_false_positive, f, indent=2)
    with open("car1_dataset/missed.json", "w") as f:
        json.dump(all_missed, f, indent=2)
    with open("car1_dataset/cnn_candidates.json", "w") as f:
        json.dump(cnn_candidates, f, indent=2)

    print(f"\n[SAVED]  → correct.json          ({len(all_correct)})")
    print(f"[SAVED]  → false_positives.json   ({len(all_false_positive)})")
    print(f"[SAVED]  → missed.json            ({len(all_missed)})")
    print(f"[SAVED]  → cnn_candidates.json    ({len(cnn_candidates)})")


if __name__ == "__main__":
    main()