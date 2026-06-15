import airsim
import os
import time
import numpy as np
import json
import cv2

CAMERAS = [
    "chase_top",
    "angled_front",
    "angled_front_left",
    "angled_front_right",
    "angled_left",
    "angled_right",
    "angled_back",
    "angled_back_left",
    "angled_back_right",
    "chase_right",
]

ALLOWED_NAMES = {
    "Car1",
    "Parked_01",  "Parked_01B",
    "Parked_02",  "Parked_02B",
    "Parked_03",  "Parked_03B",
    "Parked_04",  "Parked_04B",
    "Parked_05",  "Parked_05B",
    "Parked_06",  "Parked_06B",
    "Parked_07",  "Parked_07B",
    "Parked_08",  "Parked_08B",
    "Parked_09",  "Parked_09B",
    "Parked_10",  "Parked_10B",
    "Parked_11",  "Parked_11B",
    "Parked_12",  "Parked_12B",
    "Parked_13",  "Parked_13B",
    "Parked_14",  "Parked_14B",
    "Parked_15",  "Parked_15B",
    "Parked_16",  "Parked_16B",
}

CAPTURE_INTERVAL = 2.0   # seconds between frames


def run_camera_observer():
    output_dir = "car1_dataset"
    images_dir = os.path.join(output_dir, "images")
    arrays_dir = os.path.join(output_dir, "arrays")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(arrays_dir, exist_ok=True)

    client = airsim.CarClient()
    client.confirmConnection()
    print("[CONNECTED] AirSim client connected")

    # ── Detection filter — enable for all cameras ──────────────────────────────
    for cam in CAMERAS:
        client.simClearDetectionMeshNames(cam, airsim.ImageType.Scene, vehicle_name="Car1")
        client.simSetDetectionFilterRadius(cam, airsim.ImageType.Scene, radius_cm=100000, vehicle_name="Car1")
        client.simAddDetectionFilterMeshName(cam, airsim.ImageType.Scene, mesh_name="*", vehicle_name="Car1")
    print(f"[INIT] Detection filter set for {len(CAMERAS)} cameras")

    # ── Auto-detect resolution from first camera ───────────────────────────────
    test     = client.simGetImages([
        airsim.ImageRequest(CAMERAS[0], airsim.ImageType.Scene, False, False)
    ], vehicle_name="Car1")[0]
    actual_h = test.height
    actual_w = test.width
    actual_c = len(test.image_data_uint8) // (actual_h * actual_w)
    print(f"[INIT] Resolution: {actual_w}x{actual_h}  channels: {actual_c}")

    # ── Origin GPS ────────────────────────────────────────────────────────────
    origin_gps = client.getHomeGeoPoint()
    print(f"[INIT] Origin GPS: lat={origin_gps.latitude:.6f}  lon={origin_gps.longitude:.6f}  alt={origin_gps.altitude:.2f}")

    consolidated = {
        "metadata": {
            "total_frames"   : 0,
            "cameras"        : CAMERAS,
            "image_width"    : actual_w,
            "image_height"   : actual_h,
            "fov_degrees"    : 90.0,
            "capture_interval_s": CAPTURE_INTERVAL,
            "origin_gps"     : {
                "lat": origin_gps.latitude,
                "lon": origin_gps.longitude,
                "alt": origin_gps.altitude,
            },
        },
        "frames": []
    }

    print(f"[READY] Capturing every {CAPTURE_INTERVAL}s → '{output_dir}/'\n")

    frame_count = 0
    try:
        while True:
            t_start = time.time()

            frame_entry = {
                "frame"      : frame_count,
                "detections" : []
            }

            # ── 1. Car1 state ──────────────────────────────────────────────────
            car1_gps = client.getGpsData(gps_name="", vehicle_name="Car1")
            car1_ned = client.simGetVehiclePose(vehicle_name="Car1").position

            # ── 2. Grab all images in one batch call ───────────────────────────
            responses = client.simGetImages([
                airsim.ImageRequest(cam, airsim.ImageType.Scene, False, False)
                for cam in CAMERAS
            ], vehicle_name="Car1")

            # ── 3. Process each camera ─────────────────────────────────────────
            for resp, cam in zip(responses, CAMERAS):

                img_name = f"frame_{frame_count:04d}_{cam}"
                img_file = f"images/{img_name}.png"
                arr_file = f"arrays/{img_name}.npy"

                # Save image — flipud corrects AirSim's upside-down orientation
                if len(resp.image_data_uint8) > 0:
                    img_1d   = np.frombuffer(resp.image_data_uint8, dtype=np.uint8)
                    img_bgra = img_1d.reshape(resp.height, resp.width, actual_c)
                    img_bgr  = img_bgra[:, :, :3]          # drop alpha channel
                    img_bgr = np.flip(img_bgr, axis=(0, 1))          # correct orientation
                    np.save(os.path.join(output_dir, arr_file), img_bgr)
                    cv2.imwrite(os.path.join(output_dir, img_file), img_bgr)
                else:
                    print(f"  [{cam}] WARNING: empty image response")
                    continue

                # Camera world-space pose from simGetCameraInfo
                # (NOT resp.camera_orientation — that is body-frame, not world-frame)
                cam_info = client.simGetCameraInfo(cam, vehicle_name="Car1")
                cam_pos  = cam_info.pose.position
                cam_ori  = cam_info.pose.orientation

                # Detections for this camera
                detections = client.simGetDetections(cam, airsim.ImageType.Scene, vehicle_name="Car1")
                detections = [d for d in detections if d.name in ALLOWED_NAMES]

                for det in detections:
                    x0 = det.box2D.min.x_val
                    y0 = det.box2D.min.y_val
                    x1 = det.box2D.max.x_val
                    y1 = det.box2D.max.y_val


                    frame_entry["detections"].append({
                        "image_file"   : img_file,
                        "array_file"   : arr_file,
                        "camera"       : cam,
                        "object"       : det.name,

                        "bbox": {
                            "x0": round(x0, 1),
                            "y0": round(y0, 1),
                            "x1": round(x1, 1),
                            "y1": round(y1, 1),
                            "cx": round((x0 + x1) / 2.0, 1),
                            "cy": round((y0 + y1) / 2.0, 1),
                        },

                        "detected_gps": {
                            "lat": det.geo_point.latitude,
                            "lon": det.geo_point.longitude,
                            "alt": det.geo_point.altitude,
                        },

                        # World-space camera pose (from simGetCameraInfo)
                        "camera_pose": {
                            "pos_x": cam_pos.x_val,
                            "pos_y": cam_pos.y_val,
                            "pos_z": cam_pos.z_val,
                            "ori_w": cam_ori.w_val,
                            "ori_x": cam_ori.x_val,
                            "ori_y": cam_ori.y_val,
                            "ori_z": cam_ori.z_val,
                        },

                        "car1_gps": {
                            "lat": car1_gps.gnss.geo_point.latitude,
                            "lon": car1_gps.gnss.geo_point.longitude,
                            "alt": car1_gps.gnss.geo_point.altitude,
                        },
                        "car1_ned": {
                            "x": car1_ned.x_val,
                            "y": car1_ned.y_val,
                            "z": car1_ned.z_val,
                        },
                    })

                    print(f"  [{cam}] {det.name:12s} bbox=({x0:.0f},{y0:.0f},{x1:.0f},{y1:.0f})"
                          f"  gps=({det.geo_point.latitude:.5f},{det.geo_point.longitude:.5f})")

            # ── 4. Save JSON ───────────────────────────────────────────────────
            consolidated["frames"].append(frame_entry)
            consolidated["metadata"]["total_frames"] = frame_count + 1

            with open(os.path.join(output_dir, "detections.json"), "w") as f:
                json.dump(consolidated, f, indent=2)

            elapsed = time.time() - t_start
            n_det   = len(frame_entry["detections"])
            print(f"[FRAME {frame_count:04d}] {n_det} detection(s) across {len(CAMERAS)} cameras  ({elapsed:.2f}s)")

            # Sleep remaining time to hit CAPTURE_INTERVAL
            sleep_time = max(0.0, CAPTURE_INTERVAL - elapsed)
            time.sleep(sleep_time)

            frame_count += 1

    except KeyboardInterrupt:
        with open(os.path.join(output_dir, "detections.json"), "w") as f:
            json.dump(consolidated, f, indent=2)
        print(f"\n[STOPPED] {frame_count} frames saved to '{output_dir}/'")


if __name__ == "__main__":
    run_camera_observer()