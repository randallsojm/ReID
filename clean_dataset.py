import json
import os
import shutil

DATASET_PATH  = "car1_dataset"
OUTPUT_PATH   = "car1_dataset_moving_clean"
MIN_CARS      = 2   # minimum unique parked vehicles per frame (across all cameras)

def clean_dataset():
    src_json = os.path.join(DATASET_PATH, "detections.json")
    with open(src_json) as f:
        data = json.load(f)

    meta   = data["metadata"]
    frames = data["frames"]

    out_images = os.path.join(OUTPUT_PATH, "images")
    out_arrays = os.path.join(OUTPUT_PATH, "arrays")
    os.makedirs(out_images, exist_ok=True)
    os.makedirs(out_arrays, exist_ok=True)

    kept_frames   = []
    skipped       = 0
    copied_images = set()

    for frame in frames:
        frame_id   = frame["frame"]
        detections = frame["detections"]

        # Validate detections — reject invalid bboxes
        def is_valid_bbox(det, W, H,
                          min_size=10,       # minimum bbox width AND height in pixels
                          edge_margin=10,    # pixels from edge to be considered "clipped"
                          min_visible=0.5):  # at least 50% of bbox must be inside frame
            """
            Reject detections that are:
            - Too small (likely noise or very distant)
            - Clipped at image edge (only partial vehicle visible)
            - Mostly outside the frame
            """
            x0 = det["bbox"]["x0"]
            y0 = det["bbox"]["y0"]
            x1 = det["bbox"]["x1"]
            y1 = det["bbox"]["y1"]
            w  = x1 - x0
            h  = y1 - y0

            # Reject zero or tiny bboxes
            if w < min_size or h < min_size:
                return False

            # Reject if bbox centre is too close to any edge
            cx = (x0 + x1) / 2.0
            cy = (y0 + y1) / 2.0
            if cx < edge_margin or cx > W - edge_margin:
                return False
            if cy < edge_margin or cy > H - edge_margin:
                return False

            # Reject if bbox is clipped — any edge of bbox touches image border
            if x0 <= edge_margin:              # clipped on left
                return False
            if x1 >= W - edge_margin:          # clipped on right
                return False
            if y0 <= edge_margin:              # clipped on top
                return False
            if y1 >= H - edge_margin:          # clipped on bottom
                return False

            # Reject if too little of the vehicle is visible
            # (bbox visible area vs total bbox area)
            visible_x0 = max(x0, 0)
            visible_y0 = max(y0, 0)
            visible_x1 = min(x1, W)
            visible_y1 = min(y1, H)
            visible_area = max(0, visible_x1 - visible_x0) * max(0, visible_y1 - visible_y0)
            total_area   = w * h
            if total_area > 0 and (visible_area / total_area) < min_visible:
                return False

            return True

        W = meta["image_width"]
        H = meta["image_height"]

        # Only keep known parked vehicles — excludes Car1, buildings, roads etc.
        PARKED_NAMES = {
            "Parked_01", "Parked_01B", "Parked_02", "Parked_02B",
            "Parked_03", "Parked_03B", "Parked_04", "Parked_04B",
            "Parked_05", "Parked_05B", "Parked_06", "Parked_06B",
            "Parked_07", "Parked_07B", "Parked_08", "Parked_08B",
            "Parked_09", "Parked_09B", "Parked_10", "Parked_10B",
            "Parked_11", "Parked_11B", "Parked_12", "Parked_12B",
            "Parked_13", "Parked_13B", "Parked_14", "Parked_14B",
            "Parked_15", "Parked_15B", "Parked_16", "Parked_16B",
        }
        parked_dets = [
            d for d in detections
            if d["object"] in PARKED_NAMES
            and is_valid_bbox(d, W, H)
        ]

        # Count unique parked vehicles across all cameras
        unique_vehicles = set(d["object"] for d in parked_dets)

        if len(unique_vehicles) < MIN_CARS:
            skipped += 1
            print(f"  [SKIP] frame {frame_id:04d} — only {len(unique_vehicles)} parked vehicle(s)")
            continue

        # Only copy images for cameras that have at least one parked car detection
        cameras_with_detections = set(d["camera"] for d in parked_dets)

        for det in parked_dets:
            img_file = det["image_file"]
            arr_file = det["array_file"]

            # Only copy if this camera actually detected something
            if det["camera"] not in cameras_with_detections:
                continue

            if img_file not in copied_images:
                src_img = os.path.join(DATASET_PATH, img_file)
                src_arr = os.path.join(DATASET_PATH, arr_file)
                dst_img = os.path.join(OUTPUT_PATH,  img_file)
                dst_arr = os.path.join(OUTPUT_PATH,  arr_file)

                if os.path.exists(src_img):
                    shutil.copy2(src_img, dst_img)
                else:
                    print(f"  [WARN] missing image: {src_img}")

                if os.path.exists(src_arr):
                    shutil.copy2(src_arr, dst_arr)
                else:
                    print(f"  [WARN] missing array: {src_arr}")

                copied_images.add(img_file)

        # Save frame with only parked car detections
        # and only cameras that had detections
        clean_dets = [
            d for d in parked_dets
            if d["camera"] in cameras_with_detections
        ]

        clean_frame = {
            **frame,
            "detections"               : clean_dets,
            "cameras_with_detections"  : sorted(cameras_with_detections),
        }
        kept_frames.append(clean_frame)

        print(f"  [KEEP] frame {frame_id:04d} — "
              f"{len(unique_vehicles)} vehicles: {sorted(unique_vehicles)}  "
              f"cameras: {sorted(cameras_with_detections)}")

    # Save cleaned detections.json
    clean_data = {
        "metadata": {
            **meta,
            "total_frames"     : len(kept_frames),
            "original_frames"  : meta["total_frames"],
            "min_cars_filter"  : MIN_CARS,
            "car1_removed"     : True,
        },
        "frames": kept_frames,
    }

    out_json = os.path.join(OUTPUT_PATH, "detections.json")
    with open(out_json, "w") as f:
        json.dump(clean_data, f, indent=2)

    print(f"\n{'═'*55}")
    print(f" CLEAN DATASET SUMMARY")
    print(f"{'═'*55}")
    print(f"  Original frames  : {meta['total_frames']}")
    print(f"  Kept frames      : {len(kept_frames)}")
    print(f"  Skipped frames   : {skipped}")
    print(f"  Images copied    : {len(copied_images)}")
    print(f"  Output folder    : {OUTPUT_PATH}/")


if __name__ == "__main__":
    clean_dataset()