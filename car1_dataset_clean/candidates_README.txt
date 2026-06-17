candidates.json — output of analysis.py
============================================================

WHAT THIS FILE IS
    Every pair of vehicle detections that passed the spatial
    filter. These are candidates to pass to the CNN ReID model
    for identity verification.

SETTINGS USED
    MAX_SPEED_MS     = 1.0 m/s  (4 km/h)
    CAPTURE_INTERVAL = 2.0 s between frames
    MAX_TIME_DIFF_S  = 16.0 s  (max window to compare)
    THRESHOLD_M      = 1.0 m  (base GPS threshold)
    PIXEL_ERROR      = 1.0 px, caused by imprecise bounding box detection

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
