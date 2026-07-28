"""
reid_model_crops.py
-------------------
Vehicle ReID evaluation script for the orbit crops dataset.
Vehicle identity = vehicle_label from crops.json (SUV, SportsCar, etc.)

Mirrors reid_model_clean.py structure — imports load_model and
extract_embedding from inference_fixed.py, same embedding cache format.

Cross-camera filter
-------------------
When camera information is present in crops.json AND same-camera pairs exist
(e.g. VeRi-776), all pair-level evaluations (threshold search, F1, PR curve,
per-vehicle breakdown) automatically skip same-camera pairs, matching VeRi's
standard evaluation protocol.
On orbit datasets where all cameras are different, this is a no-op.

Usage
-----
# Extract embeddings (first run)
python reid_model_crops.py \
    --weights logs/Veri776/MBR_4B/1/best_mAP.pt \
    --images  ReID/crops \
    --crops_json ReID/crops/crops.json \
    --save_emb ReID/crops/embeddings.pt

# Evaluate (subsequent runs use cache)
python reid_model_crops.py \
    --save_emb   ReID/crops/embeddings.pt \
    --crops_json ReID/crops/crops.json

# All flags
python reid_model_crops.py \
    --save_emb    ReID/crops/embeddings.pt \
    --crops_json  ReID/crops/crops.json \
    --breakdown \
    --camera_strat \
    --worst_pairs 20 \
    --pr_curve \
    --pr_csv ReID/crops/pr_curve.csv
"""

import os
import sys
import argparse
import json
from collections import defaultdict

import torch
import torch.nn.functional as F

from inference_fixed import load_model, extract_embedding


# ── Crops metadata ────────────────────────────────────────────────────────────

def load_crops_meta(crops_json_path):
    with open(crops_json_path) as f:
        crops = json.load(f)
    return {c["crop_filename"]: c for c in crops}


def get_vehicle_id(filename, meta):
    return meta[filename]["vehicle_label"]

def get_camera(filename, meta):
    return meta[filename]["camera"]

def get_azimuth(filename, meta):
    return meta[filename]["azimuth"]


# ── Embedding extraction ──────────────────────────────────────────────────────

def extract_all_embeddings(image_dir, model, device, meta,
                           save_path=None,
                           extensions=('.png', '.jpg', '.jpeg')):
    if save_path and os.path.exists(save_path):
        print(f"Loading cached embeddings from {save_path} ...")
        data = torch.load(save_path, map_location=torch.device('cpu'))
        print(f"  {data['embeddings'].shape[0]} embeddings loaded.")
        return data['embeddings'], data['paths'], data['vehicle_ids']

    files = sorted([
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if f.lower().endswith(extensions)
        and os.path.isfile(os.path.join(image_dir, f))
        and f in meta
    ])
    if not files:
        raise FileNotFoundError(
            f"No matching image files found in {image_dir}. "
            "Check --images and --crops_json point to the same dataset.")

    print(f"Extracting embeddings from {len(files)} images ...")
    embeddings, paths, vehicle_ids, failed = [], [], [], []

    for i, path in enumerate(files):
        fname = os.path.basename(path)
        try:
            emb = extract_embedding(path, model, device)
            embeddings.append(emb)
            paths.append(path)
            vehicle_ids.append(get_vehicle_id(fname, meta))
        except Exception as exc:
            print(f"  WARNING skip {fname}: {exc}")
            failed.append(path)

        if (i + 1) % 1 == 0 or (i + 1) == len(files):
            print(f"  {i+1}/{len(files)} done ...")

    if not embeddings:
        raise RuntimeError("No embeddings extracted — check images and model.")

    all_embs = torch.cat(embeddings, dim=0)
    all_embs = F.normalize(all_embs, dim=1)
    print(f"\nEmbedding matrix: {all_embs.shape}  (failed: {len(failed)})")

    if save_path:
        torch.save({'embeddings':  all_embs,
                    'paths':       paths,
                    'vehicle_ids': vehicle_ids}, save_path)
        print(f"Saved embeddings to {save_path}")

    return all_embs, paths, vehicle_ids


# ── Camera list extraction ────────────────────────────────────────────────────

def get_cameras(paths, meta):
    """
    Return a list of camera IDs parallel to paths/vehicle_ids.
    Returns None if camera info is absent or all cameras are unique
    (no same-camera filtering needed).
    """
    filenames = [os.path.basename(p) for p in paths]
    cameras   = [meta.get(f, {}).get("camera", None) for f in filenames]

    if any(c is None for c in cameras):
        return None   # camera info missing — can't filter

    # Check whether any same-camera pairs actually exist
    from collections import Counter
    cam_counts = Counter(cameras)
    if max(cam_counts.values()) == 1:
        return None   # all cameras unique — filter would be a no-op, skip it

    return cameras


def same_camera(cameras, i, j):
    """True if both images share the same camera (and filtering is active)."""
    if cameras is None:
        return False
    return cameras[i] == cameras[j]


# ── Average Precision ─────────────────────────────────────────────────────────

def compute_ap(correct_flags):
    num_correct = sum(correct_flags)
    if num_correct == 0:
        return 0.0
    hits = precision_sum = 0.0
    for rank, flag in enumerate(correct_flags, start=1):
        if flag:
            hits += 1
            precision_sum += hits / rank
    return precision_sum / num_correct


# ── Leave-one-out evaluation ──────────────────────────────────────────────────

def find_optimal_threshold(sim_matrix, vehicle_ids, cameras=None, steps=100):
    """
    Find threshold maximising F1 across all valid pairs.
    Same-camera pairs are excluded when camera info is available (VeRi protocol).
    """
    N = len(vehicle_ids)
    all_scores, all_labels = [], []

    skipped_same_cam = 0
    for i in range(N):
        for j in range(i + 1, N):
            if same_camera(cameras, i, j):
                skipped_same_cam += 1
                continue
            all_scores.append(sim_matrix[i, j].item())
            all_labels.append(int(vehicle_ids[i] == vehicle_ids[j]))

    if skipped_same_cam:
        print(f"  [cross-cam filter] Skipped {skipped_same_cam:,} same-camera pairs "
              f"({len(all_scores):,} pairs remaining)")

    all_scores  = torch.tensor(all_scores)
    all_labels  = torch.tensor(all_labels)
    total_pos   = all_labels.sum().item()

    best_f1, best_thresh = 0.0, 0.5
    for t in [i / steps for i in range(0, steps + 1)]:
        pred = (all_scores >= t).int()
        tp   = (pred * all_labels).sum().item()
        fp   = (pred * (1 - all_labels)).sum().item()
        fn   = total_pos - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        if f1 > best_f1:
            best_f1, best_thresh = f1, t

    return best_thresh, best_f1


def evaluate(embeddings, vehicle_ids, paths, meta):
    """
    Precision/Recall at optimal F1 threshold (instance-level, cross-camera pairs only).
    Also reports CMC and overall mAP for ranking context.
    """
    N          = embeddings.shape[0]
    sim_matrix = torch.mm(embeddings, embeddings.t())

    # Extract camera list for cross-camera filtering
    cameras = get_cameras(paths, meta)
    if cameras is not None:
        print(f"\n[cross-cam filter] Active — same-camera pairs excluded from F1/threshold search.")
    else:
        print(f"\n[cross-cam filter] Inactive — all cameras unique or camera info absent.")

    # ── Find optimal threshold ────────────────────────────────────────────────
    opt_thresh, opt_f1 = find_optimal_threshold(sim_matrix, vehicle_ids, cameras)

    # ── Precision / Recall at optimal threshold ───────────────────────────────
    tp = fp = fn = 0
    for i in range(N):
        for j in range(i + 1, N):
            if same_camera(cameras, i, j):
                continue
            score     = sim_matrix[i, j].item()
            predicted = score >= opt_thresh
            actually  = vehicle_ids[i] == vehicle_ids[j]
            if predicted and actually:       tp += 1
            elif predicted and not actually: fp += 1
            elif not predicted and actually: fn += 1

    prec = 100.0 * tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = 100.0 * tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    # ── CMC (ranking context) + overall mAP ───────────────────────────────────
    cmc_hits = {1: 0, 5: 0, 10: 0}
    no_match = 0
    ap_scores = []

    for q_idx in range(N):
        q_vid   = vehicle_ids[q_idx]
        # Exclude same-camera gallery entries (both CMC and mAP)
        gallery = [g for g in range(N)
                   if g != q_idx and not same_camera(cameras, q_idx, g)]
        scores  = sim_matrix[q_idx, gallery]
        order   = scores.argsort(descending=True).tolist()
        ranked  = [gallery[i] for i in order]
        correct = [1 if vehicle_ids[g] == q_vid else 0 for g in ranked]

        if sum(correct) == 0:
            no_match += 1
            continue
        # compute_ap was previously defined but never called — this is the
        # standard per-query Average Precision, over the FULL cross-camera
        # gallery (not restricted to one azimuth bin), matching the same
        # definition eval_func() uses during training validation.
        ap_scores.append(compute_ap(correct))
        for k in (1, 5, 10):
            if any(correct[:k]):
                cmc_hits[k] += 1

    num_queries = N - no_match
    cmc1  = 100.0 * cmc_hits[1]  / num_queries if num_queries else 0.0
    cmc5  = 100.0 * cmc_hits[5]  / num_queries if num_queries else 0.0
    cmc10 = 100.0 * cmc_hits[10] / num_queries if num_queries else 0.0
    mAP   = 100.0 * sum(ap_scores) / len(ap_scores) if ap_scores else 0.0

    return dict(precision=round(prec, 2), recall=round(rec, 2),
                f1=round(f1, 2), threshold=round(opt_thresh, 2),
                CMC_at_1=round(cmc1, 2), CMC_at_5=round(cmc5, 2),
                CMC_at_10=round(cmc10, 2),
                mAP=round(mAP, 2),
                tp=tp, fp=fp, fn=fn,
                num_queries=num_queries, no_match=no_match, skipped=0)


# ── Precision-Recall curve ────────────────────────────────────────────────────

def precision_recall_curve(embeddings, vehicle_ids, paths, steps=50, save_csv=None, meta=None):
    N          = len(paths)
    sim_matrix = torch.mm(embeddings, embeddings.t())
    cameras    = get_cameras(paths, meta) if meta else None
    print("\nBuilding similarity pairs for PR curve ...")

    scores_same, scores_diff = [], []
    for i in range(N):
        for j in range(i + 1, N):
            if same_camera(cameras, i, j):
                continue
            score = sim_matrix[i, j].item()
            if vehicle_ids[i] == vehicle_ids[j]:
                scores_same.append(score)
            else:
                scores_diff.append(score)

    print(f"  Same-vehicle pairs : {len(scores_same):,}")
    print(f"  Diff-vehicle pairs : {len(scores_diff):,}")

    total_pos  = len(scores_same)
    thresholds = [i / steps for i in range(0, steps + 1)]
    rows = []
    for thresh in thresholds:
        tp   = sum(1 for s in scores_same if s >= thresh)
        fp   = sum(1 for s in scores_diff if s >= thresh)
        fn   = total_pos - tp
        prec = 100.0 * tp / (tp + fp) if (tp + fp) > 0 else 100.0
        rec  = 100.0 * tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        rows.append((thresh, prec, rec, f1, tp, fp))

    best = max(rows, key=lambda x: x[3])
    print(f"\nPrecision-Recall curve")
    print(f"{'Threshold':>10} {'Precision':>10} {'Recall':>10} {'F1':>8}")
    print("-" * 44)
    for thresh, prec, rec, f1, tp, fp in rows:
        marker = " ◄ best F1" if thresh == best[0] else ""
        print(f"{thresh:>10.2f} {prec:>9.1f}% {rec:>9.1f}% {f1:>7.1f}%{marker}")
    print("-" * 44)
    print(f"\nOptimal threshold : {best[0]:.2f}")
    print(f"  Precision       : {best[1]:.1f}%")
    print(f"  Recall          : {best[2]:.1f}%")
    print(f"  F1 score        : {best[3]:.1f}%")

    if save_csv:
        with open(save_csv, 'w') as f:
            f.write("threshold,precision,recall,f1,tp,fp\n")
            for thresh, prec, rec, f1, tp, fp in rows:
                f.write(f"{thresh:.2f},{prec:.2f},{rec:.2f},{f1:.2f},{tp},{fp}\n")
        print(f"Curve saved to {save_csv}")

def precision_recall_curve_by_gap(embeddings, vehicle_ids, paths, meta, steps=50):
    """
    Runs the full precision/recall/F1 threshold sweep SEPARATELY within each
    azimuth-gap bucket, so you can see the whole curve shape per bucket and
    identify the true optimal threshold for each — not just a single F1
    snapshot at one global threshold.
    """
    filenames  = [os.path.basename(p) for p in paths]
    sim_matrix = torch.mm(embeddings, embeddings.t())
    cameras    = get_cameras(paths, meta)
    N          = len(paths)

    gap_pairs = defaultdict(lambda: {"same": [], "diff": []})

    for i in range(N):
        for j in range(i + 1, N):
            if same_camera(cameras, i, j):
                continue
            az_i = get_azimuth(filenames[i], meta) if filenames[i] in meta else None
            az_j = get_azimuth(filenames[j], meta) if filenames[j] in meta else None
            if az_i is None or az_j is None:
                continue
            gap = abs(az_i - az_j)
            gap = min(gap, 360 - gap)

            score = sim_matrix[i, j].item()
            if vehicle_ids[i] == vehicle_ids[j]:
                gap_pairs[gap]["same"].append(score)
            else:
                gap_pairs[gap]["diff"].append(score)

    results_by_gap = {}
    for gap in sorted(gap_pairs):
        scores_same = gap_pairs[gap]["same"]
        scores_diff = gap_pairs[gap]["diff"]
        total_pos = len(scores_same)
        n_samples = total_pos + len(scores_diff)

        print(f"\n── PR curve for gap={gap}° (n_same={len(scores_same)}, n_diff={len(scores_diff)}) ──")
        rows = []
        for t in [i / steps for i in range(0, steps + 1)]:
            tp = sum(1 for s in scores_same if s >= t)
            fp = sum(1 for s in scores_diff if s >= t)
            fn = total_pos - tp
            prec = 100.0 * tp / (tp + fp) if (tp + fp) > 0 else 100.0
            rec  = 100.0 * tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            rows.append((t, prec, rec, f1))

        best = max(rows, key=lambda x: x[3])
        print(f"  Best threshold: {best[0]:.2f}  Prec={best[1]:.1f}%  Rec={best[2]:.1f}%  F1={best[3]:.1f}%")
        results_by_gap[gap] = {"best_threshold": best[0], "f1": best[3], "n_samples": n_samples}

    print(f"\n── Summary: optimal threshold per gap bucket ──")
    print(f"  {'Gap':>6}  {'Threshold':>10}  {'F1':>7}  {'Samples':>8}")
    for gap, r in results_by_gap.items():
        print(f"  {gap:>5}°  {r['best_threshold']:>10.2f}  {r['f1']:>6.1f}%  {r['n_samples']:>8}")

    return results_by_gap


def f1_by_gap_comparison(embeddings, vehicle_ids, paths, meta, global_thresh, steps=50):
    """
    For each azimuth-gap bucket, reports F1 at TWO thresholds side by side:
      - the current fixed global threshold (same one used everywhere else,
        e.g. the 0.72 optimum found by find_optimal_threshold)
      - that bucket's OWN optimal threshold (independently swept)

    This directly answers "is a single global threshold structurally unfair
    to some azimuth gaps" — the gap between the two columns IS the answer,
    not just each column on its own. Reuses the same score-collection logic
    as precision_recall_curve_by_gap (raw same/diff scores per gap bucket),
    so results are directly comparable to that function's per-bucket optima.
    """
    filenames  = [os.path.basename(p) for p in paths]
    sim_matrix = torch.mm(embeddings, embeddings.t())
    cameras    = get_cameras(paths, meta)
    N          = len(paths)

    gap_pairs = defaultdict(lambda: {"same": [], "diff": []})

    for i in range(N):
        for j in range(i + 1, N):
            if same_camera(cameras, i, j):
                continue
            az_i = get_azimuth(filenames[i], meta) if filenames[i] in meta else None
            az_j = get_azimuth(filenames[j], meta) if filenames[j] in meta else None
            if az_i is None or az_j is None:
                continue
            gap = abs(az_i - az_j)
            gap = min(gap, 360 - gap)

            score = sim_matrix[i, j].item()
            if vehicle_ids[i] == vehicle_ids[j]:
                gap_pairs[gap]["same"].append(score)
            else:
                gap_pairs[gap]["diff"].append(score)

    def f1_at(scores_same, scores_diff, thresh, total_pos):
        tp   = sum(1 for s in scores_same if s >= thresh)
        fp   = sum(1 for s in scores_diff if s >= thresh)
        fn   = total_pos - tp
        prec = 100.0 * tp / (tp + fp) if (tp + fp) > 0 else 100.0
        rec  = 100.0 * tp / (tp + fn) if (tp + fn) > 0 else 0.0
        return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    rows = []
    for gap in sorted(gap_pairs):
        scores_same = gap_pairs[gap]["same"]
        scores_diff = gap_pairs[gap]["diff"]
        total_pos   = len(scores_same)
        n_samples   = total_pos + len(scores_diff)
        if n_samples == 0:
            continue

        f1_global = f1_at(scores_same, scores_diff, global_thresh, total_pos)

        best_f1, best_thresh = 0.0, 0.5
        for t in [i / steps for i in range(0, steps + 1)]:
            f1_t = f1_at(scores_same, scores_diff, t, total_pos)
            if f1_t > best_f1:
                best_f1, best_thresh = f1_t, t

        rows.append({
            "gap": gap,
            "f1_global": round(f1_global, 1),
            "f1_optimal": round(best_f1, 1),
            "optimal_thresh": round(best_thresh, 2),
            "gap_pp": round(best_f1 - f1_global, 1),
            "n_samples": n_samples,
        })

    print(f"\n── F1 by azimuth gap: global threshold ({global_thresh:.2f}) vs. per-bucket optimal ──")
    print(f"  {'Gap':>6}  {'F1 @ global':>12}  {'F1 @ optimal':>13}  {'Gap (pp)':>9}  {'Opt.thresh':>10}  {'N':>6}")
    print("  " + "-" * 66)
    for r in rows:
        print(f"  {r['gap']:>5}°  {r['f1_global']:>11.1f}%  {r['f1_optimal']:>12.1f}%  "
              f"{r['gap_pp']:>8.1f}  {r['optimal_thresh']:>10.2f}  {r['n_samples']:>6}")

    if rows:
        worst = max(rows, key=lambda x: x['gap_pp'])
        print(f"\n  Most underserved by the global threshold: {worst['gap']}° gap "
              f"(losing {worst['gap_pp']:.1f}pp F1 vs. its own optimum)")

    return rows

# ── Angle-pair precision/recall breakdown ─────────────────────────────────────

def angle_pair_breakdown(embeddings, vehicle_ids, paths, meta, opt_thresh):
    filenames  = [os.path.basename(p) for p in paths]
    azimuths   = sorted({meta[f]["azimuth"] for f in filenames if f in meta})
    N          = len(paths)
    sim_matrix = torch.mm(embeddings, embeddings.t())
    cameras    = get_cameras(paths, meta)

    az_of = {i: meta.get(filenames[i], {}).get("azimuth") for i in range(N)}

    rows = []
    for q_az in azimuths:
        for g_az in azimuths:
            if q_az == g_az:
                continue

            tp = fp = fn = 0
            for i in range(N):
                if az_of[i] != q_az:
                    continue
                for j in range(N):
                    if i == j or az_of[j] != g_az:
                        continue
                    if same_camera(cameras, i, j):
                        continue
                    score     = sim_matrix[i, j].item()
                    predicted = score >= opt_thresh
                    actually  = vehicle_ids[i] == vehicle_ids[j]
                    if predicted and actually:       tp += 1
                    elif predicted and not actually: fp += 1
                    elif not predicted and actually: fn += 1

            if (tp + fp + fn) == 0:
                continue

            prec = 100.0 * tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec  = 100.0 * tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            az_gap = abs(q_az - g_az)
            az_gap = min(az_gap, 360 - az_gap)
            rows.append((f1, prec, rec, q_az, g_az, az_gap, tp, fp, fn))

    rows.sort(key=lambda x: x[0])

    print(f"\n── Angle-pair Precision/Recall (threshold={opt_thresh:.2f}) ──────────────")
    print(f"  {'Q-az':>6} {'G-az':>6} {'Gap':>6}  {'Prec':>7}  {'Rec':>7}  {'F1':>7}  TP/FP/FN")
    print(f"  {'─'*6} {'─'*6} {'─'*6}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*8}")
    for f1, prec, rec, q_az, g_az, gap, tp, fp, fn in rows:
        print(f"  {q_az:>5}° {g_az:>5}° {gap:>5}°  "
              f"{prec:>6.1f}%  {rec:>6.1f}%  {f1:>6.1f}%  "
              f"{tp}/{fp}/{fn}")

    print(f"\n── F1 by azimuth gap ─────────────────────────────────────────────────")
    from collections import defaultdict
    by_gap = defaultdict(list)
    for f1, prec, rec, q_az, g_az, gap, tp, fp, fn in rows:
        by_gap[gap].append(f1)
    print(f"  {'Gap':>6}  {'Mean F1':>8}  {'Min F1':>8}  Pairs")
    for gap in sorted(by_gap):
        vals = by_gap[gap]
        print(f"  {gap:>5}°  {sum(vals)/len(vals):>7.1f}%  {min(vals):>7.1f}%  {len(vals)}")

# ── mAP by azimuth gap ─────────────────────────────────────────────────────────

def angle_gap_map_breakdown(results, azimuths):
    """
    Re-aggregates the (q_az, g_az) -> (mAP, n) grid from camera_stratified_eval
    into mAP grouped by azimuth GAP (0°, 30°, 60°, ... up to 180°), rather than
    by absolute angle pair. Weighted by sample count n, not a flat average of
    per-pair means — so a gap bucket built from more query images counts more.
    """
    gap_data = defaultdict(lambda: [0.0, 0])  # gap -> [sum(mAP*n), sum(n)]

    for (q_az, g_az), (mAP, n) in results.items():
        if n == 0 or mAP != mAP:  # skip empty / NaN cells
            continue
        gap = abs(q_az - g_az)
        gap = min(gap, 360 - gap)
        gap_data[gap][0] += mAP * n
        gap_data[gap][1] += n

    print("\n── mAP by azimuth gap ──────────────────────────────────────────")
    print(f"  {'Gap':>6}  {'Weighted mAP':>13}  {'Samples':>8}")
    print(f"  {'─'*6}  {'─'*13}  {'─'*8}")

    rows = []
    for gap in sorted(gap_data):
        total_weighted, total_n = gap_data[gap]
        if total_n == 0:
            continue
        mean_mAP = total_weighted / total_n
        rows.append((gap, mean_mAP, total_n))
        print(f"  {gap:>5}°  {mean_mAP:>12.1f}%  {total_n:>8}")

    if rows:
        best = max(rows, key=lambda x: x[1])
        worst = min(rows, key=lambda x: x[1])
        print(f"\n  Best gap  : {best[0]}°  (mAP={best[1]:.1f}%, n={best[2]})")
        print(f"  Worst gap : {worst[0]}°  (mAP={worst[1]:.1f}%, n={worst[2]})")
        print(f"  Spread    : {best[1]-worst[1]:.1f}pp")

    return rows
# ── Camera-stratified evaluation (by azimuth) ─────────────────────────────────

def camera_stratified_eval(embeddings, vehicle_ids, paths, meta):
    filenames  = [os.path.basename(p) for p in paths]
    azimuths   = sorted({get_azimuth(f, meta) for f in filenames if f in meta})
    N          = len(paths)
    sim_matrix = torch.mm(embeddings, embeddings.t())
    cameras    = get_cameras(paths, meta)

    az_indices = {az: [i for i, f in enumerate(filenames)
                        if f in meta and get_azimuth(f, meta) == az]
                  for az in azimuths}

    results = {}
    for q_az in azimuths:
        for g_az in azimuths:
            q_idxs = az_indices[q_az]
            g_idxs = set(az_indices[g_az])
            ap_scores = []
            for q_idx in q_idxs:
                q_vid   = vehicle_ids[q_idx]
                gallery = [i for i in g_idxs
                           if not (i == q_idx and q_az == g_az)
                           and not same_camera(cameras, q_idx, i)]
                if not gallery:
                    continue
                if not any(vehicle_ids[i] == q_vid for i in gallery):
                    continue
                scores  = sim_matrix[q_idx, gallery]
                order   = scores.argsort(descending=True).tolist()
                ranked  = [gallery[k] for k in order]
                correct = [1 if vehicle_ids[i] == q_vid else 0 for i in ranked]
                ap_scores.append(compute_ap(correct))
            mAP = 100.0 * sum(ap_scores) / len(ap_scores) if ap_scores else float('nan')
            results[(q_az, g_az)] = (round(mAP, 1), len(ap_scores))

    col_w  = 8
    q_g    = "Q\\G"
    header = f"  {q_g:>6}" + "".join(f"  {az:>{col_w}}" for az in azimuths)
    print("\nmAP (%) by query azimuth → gallery azimuth")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    same_aps, cross_aps = [], []
    for q_az in azimuths:
        row = f"  {q_az:>6}"
        for g_az in azimuths:
            val, n = results.get((q_az, g_az), (float('nan'), 0))
            row += f"  {val:>{col_w}.1f}" if n > 0 else f"  {'—':>{col_w}}"
            if val == val:
                (same_aps if q_az == g_az else cross_aps).append(val)
        print(row)
    print("=" * len(header))

    same_mean  = sum(same_aps)  / len(same_aps)  if same_aps  else 0
    cross_mean = sum(cross_aps) / len(cross_aps) if cross_aps else 0
    print(f"\nSame-azimuth  mAP : {same_mean:.1f}%")
    print(f"Cross-azimuth mAP : {cross_mean:.1f}%")
    print(f"Viewpoint penalty : {same_mean - cross_mean:.1f}pp")

    cross_pairs = [(mAP, n, q, g) for (q,g),(mAP,n) in results.items()
                   if q != g and n > 0 and mAP == mAP]
    cross_pairs.sort(key=lambda x: x[0])
    print("\nWorst 5 cross-azimuth pairs:")
    for mAP, n, q, g in cross_pairs[:5]:
        print(f"  az{q:>3}° → az{g:>3}°   mAP={mAP:.1f}%  (n={n})")
    print("\nBest 5 cross-azimuth pairs:")
    for mAP, n, q, g in reversed(cross_pairs[-5:]):
        print(f"  az{q:>3}° → az{g:>3}°   mAP={mAP:.1f}%  (n={n})")

    return results, azimuths


# ── Heatmap plot ──────────────────────────────────────────────────────────────

def plot_heatmap(results, azimuths, save_path="heatmap.png"):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n   = len(azimuths)
    mat = np.full((n, n), float('nan'))
    for i, q_az in enumerate(azimuths):
        for j, g_az in enumerate(azimuths):
            val, cnt = results.get((q_az, g_az), (float('nan'), 0))
            if cnt > 0:
                mat[i, j] = val

    fig, ax = plt.subplots(figsize=(max(6, n), max(5, n - 1)))
    im = ax.imshow(mat, vmin=0, vmax=100, cmap="RdYlGn", aspect="auto")

    for i in range(n):
        for j in range(n):
            val = mat[i, j]
            if not np.isnan(val):
                colour = "black" if 20 < val < 80 else "white"
                ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                        fontsize=8, color=colour)

    labels = [f"{az}°" for az in azimuths]
    ax.set_xticks(range(n));  ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(n));  ax.set_yticklabels(labels)
    ax.set_xlabel("Gallery azimuth")
    ax.set_ylabel("Query azimuth")
    ax.set_title("mAP (%) by Query → Gallery Azimuth")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("mAP (%)")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Heatmap saved to {save_path}")


# ── Vehicle similarity heatmap ────────────────────────────────────────────────

def plot_vehicle_heatmap(embeddings, vehicle_ids, save_path="vehicle_heatmap.png"):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sim_matrix = torch.mm(embeddings, embeddings.t()).numpy()
    vehicles   = sorted(set(vehicle_ids), key=lambda x: int(x) if str(x).isdigit() else x)
    V          = len(vehicles)

    clusters = {v: [i for i, vid in enumerate(vehicle_ids) if vid == v]
                for v in vehicles}

    mean_mat  = np.full((V, V), float('nan'))
    count_mat = np.zeros((V, V), dtype=int)

    for i, v_q in enumerate(vehicles):
        for j, v_g in enumerate(vehicles):
            idxs_q = clusters[v_q]
            idxs_g = clusters[v_g]
            if i == j:
                sims = [sim_matrix[a, b]
                        for a in idxs_q for b in idxs_g if a != b]
            else:
                sims = [sim_matrix[a, b]
                        for a in idxs_q for b in idxs_g]
            if sims:
                mean_mat[i, j]  = float(np.mean(sims))
                count_mat[i, j] = len(sims)

    diag    = [mean_mat[i, i] for i in range(V) if not np.isnan(mean_mat[i, i])]
    offdiag = [mean_mat[i, j] for i in range(V) for j in range(V)
               if i != j and not np.isnan(mean_mat[i, j])]
    print(f"\nVehicle similarity matrix ({V}×{V}):")
    print(f"  Intra-vehicle mean  : {np.mean(diag):.4f}")
    print(f"  Inter-vehicle mean  : {np.mean(offdiag):.4f}")
    print(f"  Separation score    : {np.mean(diag) - np.mean(offdiag):.4f}")

    vmin = max(0, np.nanmin(mean_mat) - 0.05)
    vmax = min(1, np.nanmax(mean_mat) + 0.05)

    fig, ax = plt.subplots(figsize=(max(6, V), max(5, V - 1)))
    im = ax.imshow(mean_mat, vmin=vmin, vmax=vmax, cmap="RdYlGn", aspect="auto")

    for i in range(V):
        for j in range(V):
            val = mean_mat[i, j]
            if not np.isnan(val):
                colour = "black" if (vmin + vmax) / 2 * 0.8 < val < (vmin + vmax) / 2 * 1.2 else "white"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=7, color=colour)

    ax.set_xticks(range(V)); ax.set_xticklabels(vehicles, rotation=45, ha="right")
    ax.set_yticks(range(V)); ax.set_yticklabels(vehicles)
    ax.set_xlabel("Gallery vehicle")
    ax.set_ylabel("Query vehicle")
    ax.set_title("Mean Cosine Similarity by Vehicle Type\n(diagonal = intra, off-diagonal = inter)")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Mean cosine similarity")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Vehicle heatmap saved to {save_path}")


# ── Per-vehicle breakdown ─────────────────────────────────────────────────────

def per_vehicle_breakdown(embeddings, vehicle_ids, paths, meta):
    sim_matrix = torch.mm(embeddings, embeddings.t())
    cameras    = get_cameras(paths, meta)
    clusters   = defaultdict(list)
    for i, vid in enumerate(vehicle_ids):
        clusters[vid].append(i)

    opt_thresh, _ = find_optimal_threshold(sim_matrix, vehicle_ids, cameras)

    print(f"\nPer-vehicle Precision / Recall  (threshold={opt_thresh:.2f})")
    print(f"  {'Vehicle':<20} {'N':>4}  {'Prec':>7}  {'Rec':>7}  {'F1':>7}  {'MeanSim':>8}  {'MinSim':>8}  TP/FP/FN")
    print("  " + "-" * 78)

    for vid in sorted(clusters):
        idxs = clusters[vid]
        if len(idxs) < 2:
            continue

        sims = [sim_matrix[i, j].item() for i in idxs for j in idxs if i != j]
        mean_sim = sum(sims) / len(sims)
        min_sim  = min(sims)

        tp = fp = fn = 0
        for i in idxs:
            for j in range(len(vehicle_ids)):
                if i == j:
                    continue
                if same_camera(cameras, i, j):
                    continue
                score     = sim_matrix[i, j].item()
                predicted = score >= opt_thresh
                actually  = vehicle_ids[j] == vid
                if predicted and actually:       tp += 1
                elif predicted and not actually: fp += 1
                elif not predicted and actually: fn += 1

        prec = 100.0 * tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = 100.0 * tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        print(f"  {str(vid):<20} {len(idxs):>4}  "
              f"{prec:>6.1f}%  {rec:>6.1f}%  {f1:>6.1f}%  "
              f"{mean_sim:>8.4f}  {min_sim:>8.4f}  {tp}/{fp}/{fn}")


def coverage_breakdown(embeddings, vehicle_ids, paths, meta, bins=None, crop_size=224):
    if bins is None:
        bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0, 5.0]

    filenames = [os.path.basename(p) for p in paths]
    cameras   = get_cameras(paths, meta)

    crop_coverage_of = {}
    for i in range(len(paths)):
        entry  = meta.get(filenames[i], {})
        area   = entry.get("quality", {}).get("area", None)
        aspect = entry.get("quality", {}).get("aspect", None)
        is_resized = entry.get("resized_down", False)

        if is_resized and aspect is not None:
            ratio = min(aspect, 1/aspect) if aspect > 0 else None
            crop_coverage_of[i] = ratio
        else:
            crop_coverage_of[i] = (area / (crop_size * crop_size)) if area is not None else None

    sim_matrix = torch.mm(embeddings, embeddings.t())
    opt_thresh, _ = find_optimal_threshold(sim_matrix, vehicle_ids, cameras)

    print(f"\n── F1 by crop coverage bucket (bbox_area / {crop_size}x{crop_size}, threshold={opt_thresh:.2f}) ──")
    print(f"  {'Coverage range':<18} {'N':>4}  {'Prec':>7}  {'Rec':>7}  {'F1':>7}")

    for lo, hi in zip(bins[:-1], bins[1:]):
        idxs = [i for i in range(len(paths))
                if crop_coverage_of[i] is not None and lo <= crop_coverage_of[i] < hi]
        if len(idxs) < 2:
            continue

        tp = fp = fn = 0
        for i in idxs:
            for j in range(len(paths)):
                if i == j:
                    continue
                if same_camera(cameras, i, j):
                    continue
                score     = sim_matrix[i, j].item()
                predicted = score >= opt_thresh
                actually  = vehicle_ids[j] == vehicle_ids[i]
                if predicted and actually:       tp += 1
                elif predicted and not actually: fp += 1
                elif not predicted and actually: fn += 1

        prec = 100.0 * tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = 100.0 * tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        print(f"  {lo:.0%}-{hi:.0%}".ljust(18) +
              f" {len(idxs):>4}  {prec:>6.1f}%  {rec:>6.1f}%  {f1:>6.1f}%")


# ── Worst same-vehicle pairs ──────────────────────────────────────────────────

def show_worst_pairs(embeddings, vehicle_ids, paths, meta, n=10):
    filenames  = [os.path.basename(p) for p in paths]
    sim_matrix = torch.mm(embeddings, embeddings.t())
    cameras    = get_cameras(paths, meta)
    N          = len(paths)
    worst      = []

    for i in range(N):
        for j in range(i + 1, N):
            if vehicle_ids[i] != vehicle_ids[j]:
                continue
            if same_camera(cameras, i, j):
                continue
            score = sim_matrix[i, j].item()
            az_i  = get_azimuth(filenames[i], meta) if filenames[i] in meta else '?'
            az_j  = get_azimuth(filenames[j], meta) if filenames[j] in meta else '?'
            worst.append((score, paths[i], paths[j], vehicle_ids[i], az_i, az_j))

    worst.sort(key=lambda x: x[0])
    print(f"\nWorst {n} same-vehicle pairs by cosine similarity:")
    print("=" * 80)
    for rank, (score, pi, pj, vid, az_i, az_j) in enumerate(worst[:n], 1):
        print(f"\n#{rank}  vehicle={vid}  score={score:.4f}")
        print(f"  A [az={az_i:>3}°]  {os.path.basename(pi)}")
        print(f"  B [az={az_j:>3}°]  {os.path.basename(pj)}")
    print("=" * 80)


def show_best_pairs(embeddings, vehicle_ids, paths, meta, n=10):
    filenames  = [os.path.basename(p) for p in paths]
    sim_matrix = torch.mm(embeddings, embeddings.t())
    cameras    = get_cameras(paths, meta)
    N          = len(paths)
    best       = []

    for i in range(N):
        for j in range(i + 1, N):
            if vehicle_ids[i] != vehicle_ids[j]:
                continue
            if same_camera(cameras, i, j):
                continue
            score = sim_matrix[i, j].item()
            az_i  = get_azimuth(filenames[i], meta) if filenames[i] in meta else '?'
            az_j  = get_azimuth(filenames[j], meta) if filenames[j] in meta else '?'
            best.append((score, paths[i], paths[j], vehicle_ids[i], az_i, az_j))

    best.sort(key=lambda x: x[0], reverse=True)
    print(f"\nBest {n} same-vehicle pairs by cosine similarity:")
    print("=" * 80)
    for rank, (score, pi, pj, vid, az_i, az_j) in enumerate(best[:n], 1):
        print(f"\n#{rank}  vehicle={vid}  score={score:.4f}")
        print(f"  A [az={az_i:>3}°]  {os.path.basename(pi)}")
        print(f"  B [az={az_j:>3}°]  {os.path.basename(pj)}")
    print("=" * 80)


# ── Remove embeddings from cache ──────────────────────────────────────────────

def remove_embeddings(save_path, filenames):
    data      = torch.load(save_path, map_location=torch.device('cpu'))
    paths     = data['paths']
    vids      = data['vehicle_ids']
    embs      = data['embeddings']
    remove_set = set(filenames)
    keep      = [i for i, p in enumerate(paths)
                 if os.path.basename(p) not in remove_set]
    removed   = len(paths) - len(keep)
    if removed == 0:
        print("No matching filenames found in cache.")
        return
    torch.save({'embeddings':  embs[keep],
                'paths':       [paths[i] for i in keep],
                'vehicle_ids': [vids[i]  for i in keep]}, save_path)
    print(f"Removed {removed} embedding(s). Cache: {len(keep)} remaining.")


# ── Print results ─────────────────────────────────────────────────────────────

def print_results(results):
    width = 46
    print("\n" + "=" * width)
    print("  ReID Evaluation Results")
    print("=" * width)
    print(f"  Queries evaluated : {results['num_queries']}")
    if results.get('no_match'):
        print(f"  No GT match       : {results['no_match']}")
    print(f"  Optimal threshold : {results['threshold']}")
    print("-" * width)
    print(f"  Precision         : {results['precision']:>6.2f} %")
    print(f"  Recall            : {results['recall']:>6.2f} %")
    print(f"  F1 score          : {results['f1']:>6.2f} %")
    print(f"  TP / FP / FN      : {results['tp']} / {results['fp']} / {results['fn']}")
    print("-" * width)
    print(f"  mAP               : {results['mAP']:>6.2f} %")
    print(f"  CMC @ 1           : {results['CMC_at_1']:>6.2f} %")
    print(f"  CMC @ 5           : {results['CMC_at_5']:>6.2f} %")
    print(f"  CMC @ 10          : {results['CMC_at_10']:>6.2f} %")
    print("=" * width)


# ── Worst pairs image grid ────────────────────────────────────────────────────

def plot_worst_pairs_images(embeddings, vehicle_ids, paths, meta,
                            n=10, save_path="worst_pairs.png",
                            images_dir=None):
    args_images_dir = images_dir or os.path.dirname(save_path)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image as PILImage

    sim_matrix = torch.mm(embeddings, embeddings.t()).numpy()
    cameras    = get_cameras(paths, meta)
    N = len(paths)

    pairs = []
    for i in range(N):
        for j in range(i + 1, N):
            if vehicle_ids[i] == vehicle_ids[j] and not same_camera(cameras, i, j):
                fname_a = os.path.basename(paths[i])
                fname_b = os.path.basename(paths[j])
                pairs.append({
                    "score":   float(sim_matrix[i, j]),
                    "path_a":  paths[i], "path_b":  paths[j],
                    "fname_a": fname_a,  "fname_b": fname_b,
                    "vehicle": vehicle_ids[i],
                    "az_a":    meta.get(fname_a, {}).get("azimuth", "?"),
                    "az_b":    meta.get(fname_b, {}).get("azimuth", "?"),
                })
    pairs.sort(key=lambda x: x["score"])
    pairs = pairs[:n]

    def load_img(path):
        if os.path.exists(path):
            return PILImage.open(path).convert("RGB")
        alt = os.path.join(args_images_dir, os.path.basename(path))
        if os.path.exists(alt):
            return PILImage.open(alt).convert("RGB")
        alt2 = os.path.join(os.path.dirname(save_path), os.path.basename(path))
        if os.path.exists(alt2):
            return PILImage.open(alt2).convert("RGB")
        return None

    loaded = [(load_img(p["path_a"]), load_img(p["path_b"])) for p in pairs]

    col_w_in = 4.0
    row_heights = []
    for img_a, img_b in loaded:
        h, w = (img_a.size[1], img_a.size[0]) if img_a else \
               (img_b.size[1], img_b.size[0]) if img_b else (256, 256)
        row_heights.append(col_w_in * h / w + 0.6)

    fig, axes = plt.subplots(len(pairs), 2,
                             figsize=(col_w_in * 2, sum(row_heights)),
                             gridspec_kw={"height_ratios": row_heights})
    if len(pairs) == 1:
        axes = [axes]

    for rank, (ax_row, pair, (img_a, img_b)) in enumerate(zip(axes, pairs, loaded), 1):
        ax_a, ax_b = ax_row
        if img_a: ax_a.imshow(img_a)
        else: ax_a.text(0.5, 0.5, "Not found", ha="center", va="center",
                        transform=ax_a.transAxes, color="red")
        if img_b: ax_b.imshow(img_b)
        else: ax_b.text(0.5, 0.5, "Not found", ha="center", va="center",
                        transform=ax_b.transAxes, color="red")

        ax_a.set_title(f"#{rank} {pair['vehicle']}  az={pair['az_a']}°\n{pair['fname_a']}", fontsize=8)
        ax_b.set_title(f"#{rank} {pair['vehicle']}  az={pair['az_b']}°\n{pair['fname_b']}", fontsize=8)

        score  = pair["score"]
        colour = "red" if score < 0.3 else "orange" if score < 0.5 else "gold"
        for ax in (ax_a, ax_b):
            ax.axis("off")
            for spine in ax.spines.values():
                spine.set_edgecolor(colour); spine.set_linewidth(3); spine.set_visible(True)
        ax_a.text(0.5, 0.02, f"sim={score:.4f}", ha="center", va="bottom", fontsize=8,
                  color="white", fontweight="bold", transform=ax_a.transAxes,
                  bbox=dict(boxstyle="round,pad=0.2", fc=colour, ec=colour, lw=1))

    fig.suptitle(f"Top {len(pairs)} Worst Same-Vehicle Pairs (lowest cosine similarity)",
                 fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Worst pairs image saved to {save_path}")


def plot_best_pairs_images(embeddings, vehicle_ids, paths, meta,
                           n=10, save_path="best_pairs.png", images_dir=None):
    args_images_dir = images_dir or os.path.dirname(save_path)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image as PILImage

    sim_matrix = torch.mm(embeddings, embeddings.t()).numpy()
    cameras    = get_cameras(paths, meta)
    N = len(paths)

    pairs = []
    for i in range(N):
        for j in range(i + 1, N):
            if vehicle_ids[i] == vehicle_ids[j] and not same_camera(cameras, i, j):
                fname_a = os.path.basename(paths[i])
                fname_b = os.path.basename(paths[j])
                pairs.append({
                    "score":   float(sim_matrix[i, j]),
                    "path_a":  paths[i], "path_b":  paths[j],
                    "fname_a": fname_a,  "fname_b": fname_b,
                    "vehicle": vehicle_ids[i],
                    "az_a":    meta.get(fname_a, {}).get("azimuth", "?"),
                    "az_b":    meta.get(fname_b, {}).get("azimuth", "?"),
                })
    pairs.sort(key=lambda x: x["score"], reverse=True)
    pairs = pairs[:n]

    def load_img(path):
        if os.path.exists(path): return PILImage.open(path).convert("RGB")
        alt = os.path.join(args_images_dir, os.path.basename(path))
        if os.path.exists(alt): return PILImage.open(alt).convert("RGB")
        alt2 = os.path.join(os.path.dirname(save_path), os.path.basename(path))
        if os.path.exists(alt2): return PILImage.open(alt2).convert("RGB")
        return None

    loaded = [(load_img(p["path_a"]), load_img(p["path_b"])) for p in pairs]
    col_w_in = 4.0
    row_heights = []
    for img_a, img_b in loaded:
        h, w = (img_a.size[1], img_a.size[0]) if img_a else \
               (img_b.size[1], img_b.size[0]) if img_b else (256, 256)
        row_heights.append(col_w_in * h / w + 0.6)

    fig, axes = plt.subplots(len(pairs), 2,
                             figsize=(col_w_in * 2, sum(row_heights)),
                             gridspec_kw={"height_ratios": row_heights})
    if len(pairs) == 1: axes = [axes]

    for rank, (ax_row, pair, (img_a, img_b)) in enumerate(zip(axes, pairs, loaded), 1):
        ax_a, ax_b = ax_row
        if img_a: ax_a.imshow(img_a)
        else: ax_a.text(0.5, 0.5, "Not found", ha="center", va="center",
                        transform=ax_a.transAxes, color="red")
        if img_b: ax_b.imshow(img_b)
        else: ax_b.text(0.5, 0.5, "Not found", ha="center", va="center",
                        transform=ax_b.transAxes, color="red")

        ax_a.set_title(f"#{rank} {pair['vehicle']}  az={pair['az_a']}°\n{pair['fname_a']}", fontsize=8)
        ax_b.set_title(f"#{rank} {pair['vehicle']}  az={pair['az_b']}°\n{pair['fname_b']}", fontsize=8)

        score  = pair["score"]
        colour = "green" if score > 0.7 else "limegreen" if score > 0.5 else "gold"
        for ax in (ax_a, ax_b):
            ax.axis("off")
            for spine in ax.spines.values():
                spine.set_edgecolor(colour); spine.set_linewidth(3); spine.set_visible(True)
        ax_a.text(0.5, 0.02, f"sim={score:.4f}", ha="center", va="bottom", fontsize=8,
                  color="white", fontweight="bold", transform=ax_a.transAxes,
                  bbox=dict(boxstyle="round,pad=0.2", fc=colour, ec=colour, lw=1))

    fig.suptitle(f"Top {len(pairs)} Best Same-Vehicle Pairs (highest cosine similarity)",
                 fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Best pairs image saved to {save_path}")

def show_worst_fp_pairs(embeddings, vehicle_ids, paths, meta, opt_thresh, n=20):
    """Print top false-positive pairs (different vehicle, score >= threshold),
    sorted by score descending (most confidently wrong first)."""
    filenames  = [os.path.basename(p) for p in paths]
    sim_matrix = torch.mm(embeddings, embeddings.t())
    cameras    = get_cameras(paths, meta)
    N          = len(paths)
    fp_pairs   = []

    for i in range(N):
        for j in range(i + 1, N):
            if vehicle_ids[i] == vehicle_ids[j]:
                continue
            if same_camera(cameras, i, j):
                continue
            score = sim_matrix[i, j].item()
            if score >= opt_thresh:
                az_i = get_azimuth(filenames[i], meta) if filenames[i] in meta else '?'
                az_j = get_azimuth(filenames[j], meta) if filenames[j] in meta else '?'
                fp_pairs.append((score, paths[i], paths[j], vehicle_ids[i], vehicle_ids[j], az_i, az_j))

    fp_pairs.sort(key=lambda x: -x[0])
    print(f"\nTop {min(n, len(fp_pairs))} false-positive pairs (threshold={opt_thresh:.2f}):")
    print("=" * 80)
    for rank, (score, pi, pj, vid_a, vid_b, az_i, az_j) in enumerate(fp_pairs[:n], 1):
        print(f"\n#{rank}  vehicle_A={vid_a}  vehicle_B={vid_b}  score={score:.4f}")
        print(f"  A [az={az_i:>3}°]  {os.path.basename(pi)}")
        print(f"  B [az={az_j:>3}°]  {os.path.basename(pj)}")
    print("=" * 80)
# ── FP / FN image grid ────────────────────────────────────────────────────────

def plot_fp_fn_images(embeddings, vehicle_ids, paths, meta,
                      opt_thresh, save_dir=".", images_dir=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image as PILImage

    filenames  = [os.path.basename(p) for p in paths]
    sim_matrix = torch.mm(embeddings, embeddings.t())
    cameras    = get_cameras(paths, meta)
    N          = len(paths)
    src_dir    = images_dir or save_dir

    def load_img(path):
        if os.path.exists(path): return PILImage.open(path).convert("RGB")
        alt = os.path.join(src_dir, os.path.basename(path))
        return PILImage.open(alt).convert("RGB") if os.path.exists(alt) else None

    fn_pairs = []
    fp_pairs = []

    for i in range(N):
        for j in range(i + 1, N):
            if same_camera(cameras, i, j):
                continue
            score   = sim_matrix[i, j].item()
            same    = vehicle_ids[i] == vehicle_ids[j]
            fname_a = filenames[i]
            fname_b = filenames[j]
            entry   = {
                "score":   score,
                "path_a":  paths[i], "path_b": paths[j],
                "fname_a": fname_a,  "fname_b": fname_b,
                "vid_a":   vehicle_ids[i], "vid_b": vehicle_ids[j],
                "az_a":    meta.get(fname_a, {}).get("azimuth", "?"),
                "az_b":    meta.get(fname_b, {}).get("azimuth", "?"),
            }
            if same and score < opt_thresh:
                fn_pairs.append(entry)
            elif not same and score >= opt_thresh:
                fp_pairs.append(entry)

    fn_pairs.sort(key=lambda x: x["score"])
    fp_pairs.sort(key=lambda x: -x["score"])

    print(f"  False Negatives : {len(fn_pairs)}  (same vehicle, score < {opt_thresh:.2f})")
    print(f"  False Positives : {len(fp_pairs)}  (diff vehicle, score >= {opt_thresh:.2f})")

    def save_grid(pairs, title, filename, colour):
        if not pairs:
            print(f"  No {title} to display.")
            return
        n = len(pairs)
        col_w_in = 4.0
        row_heights = []
        loaded = []
        for p in pairs:
            ia = load_img(p["path_a"])
            ib = load_img(p["path_b"])
            loaded.append((ia, ib))
            h = max((ia.size[1] if ia else 256), (ib.size[1] if ib else 256))
            w = max((ia.size[0] if ia else 256), (ib.size[0] if ib else 256))
            row_heights.append(col_w_in * h / w + 0.7)

        fig, axes = plt.subplots(n, 2, figsize=(col_w_in * 2, sum(row_heights)),
                                 gridspec_kw={"height_ratios": row_heights})
        if n == 1: axes = [axes]

        for rank, (ax_row, pair, (img_a, img_b)) in enumerate(zip(axes, pairs, loaded), 1):
            ax_a, ax_b = ax_row
            if img_a: ax_a.imshow(img_a)
            else: ax_a.text(0.5, 0.5, "Not found", ha="center", va="center",
                            transform=ax_a.transAxes, color="red")
            if img_b: ax_b.imshow(img_b)
            else: ax_b.text(0.5, 0.5, "Not found", ha="center", va="center",
                            transform=ax_b.transAxes, color="red")

            ax_a.set_title(f"#{rank} vid={pair['vid_a']}  az={pair['az_a']}°\n{pair['fname_a']}", fontsize=7)
            ax_b.set_title(f"#{rank} vid={pair['vid_b']}  az={pair['az_b']}°\n{pair['fname_b']}", fontsize=7)

            for ax in (ax_a, ax_b):
                ax.axis("off")
                for spine in ax.spines.values():
                    spine.set_edgecolor(colour); spine.set_linewidth(3); spine.set_visible(True)
            ax_a.text(0.5, 0.02, f"sim={pair['score']:.4f}", ha="center", va="bottom", fontsize=8,
                      color="white", fontweight="bold", transform=ax_a.transAxes,
                      bbox=dict(boxstyle="round,pad=0.2", fc=colour, ec=colour, lw=1))

        fig.suptitle(f"{title} ({n} pairs)  threshold={opt_thresh:.2f}",
                     fontsize=11, fontweight="bold", y=1.005)
        plt.tight_layout()
        out = os.path.join(save_dir, filename)
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {out}")

    save_grid(fn_pairs[:50], "False Negatives (missed same-vehicle)", "false_negatives.png", colour="red")
    save_grid(fp_pairs[:50], "False Positives (wrong match)", "false_positives.png", colour="orange")
    if len(fn_pairs) > 50: print(f"  [NOTE] Showing 50 of {len(fn_pairs)} FN pairs")
    if len(fp_pairs) > 50: print(f"  [NOTE] Showing 50 of {len(fp_pairs)} FP pairs")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate MBR_4B ReID on orbit crops dataset.")
    parser.add_argument('--weights',          default='logs/Veri776/MBR_4B/1/best_mAP.pt')
    parser.add_argument('--images',           default='ReID/crops')
    parser.add_argument('--crops_json',       default='ReID/crops/crops.json')
    parser.add_argument('--save_emb',         default=None)
    parser.add_argument('--breakdown',        action='store_true')
    parser.add_argument('--camera_strat',     action='store_true')
    parser.add_argument('--worst_pairs',      type=int, default=0, metavar='N')
    parser.add_argument('--pr_curve',         action='store_true')
    parser.add_argument('--pr_csv',           default=None, metavar='PATH')
    parser.add_argument('--remove',           nargs='+', default=None, metavar='FILENAME')
    parser.add_argument('--heatmap',          default=None, metavar='PATH')
    parser.add_argument('--vehicle_heatmap',  default=None, metavar='PATH')
    parser.add_argument('--angle_breakdown',  action='store_true')
    parser.add_argument('--worst_pairs_img',  default=None, metavar='PATH')
    parser.add_argument('--fp_fn_imgs',       default=None, metavar='DIR')
    parser.add_argument('--best_pairs',       type=int, default=0, metavar='N')
    parser.add_argument('--best_pairs_img',   default=None, metavar='PATH')
    parser.add_argument('--coverage_breakdown', action='store_true')
    parser.add_argument('--angle_gap_map', action='store_true',
                     help='Show mAP broken down by azimuth gap (not just F1)')
    parser.add_argument('--pr_curve_by_gap', action='store_true')
    parser.add_argument('--f1_by_gap', action='store_true',
                     help='Show F1 by azimuth gap, comparing the global threshold vs. each bucket\'s own optimum')
    parser.add_argument('--worst_fp', type=int, default=0, metavar='N')
    args = parser.parse_args()

    if args.remove:
        if not args.save_emb:
            print("ERROR: --remove requires --save_emb")
            return
        remove_embeddings(args.save_emb, args.remove)
        return

    meta = load_crops_meta(args.crops_json)
    print(f"[META] Loaded {len(meta)} entries from {args.crops_json}")
    vehicle_labels = sorted({v["vehicle_label"] for v in meta.values()})
    print(f"[META] Vehicles: {vehicle_labels}")

    model, device = load_model(args.weights)

    embeddings, paths, vehicle_ids = extract_all_embeddings(
        image_dir=args.images, model=model, device=device,
        meta=meta, save_path=args.save_emb)

    counts = defaultdict(int)
    for vid in vehicle_ids:
        counts[vid] += 1
    print(f"\nVehicle classes   : {len(counts)}")
    print(f"Images per vehicle: min={min(counts.values())}  "
          f"max={max(counts.values())}  "
          f"mean={sum(counts.values())/len(counts):.1f}")

    results = evaluate(embeddings, vehicle_ids, paths, meta)
    print_results(results)

    if args.angle_breakdown:
        angle_pair_breakdown(embeddings, vehicle_ids, paths, meta,
                             opt_thresh=results['threshold'])
    if args.breakdown:
        per_vehicle_breakdown(embeddings, vehicle_ids, paths, meta)
    if args.worst_fp > 0:
        show_worst_fp_pairs(embeddings, vehicle_ids, paths, meta, opt_thresh=results['threshold'], n=args.worst_fp)
    if args.camera_strat or args.heatmap:
        strat_results, azimuths = camera_stratified_eval(embeddings, vehicle_ids, paths, meta)
        if args.heatmap:
            plot_heatmap(strat_results, azimuths, save_path=args.heatmap)
    if args.coverage_breakdown:
        coverage_breakdown(embeddings, vehicle_ids, paths, meta)
    if args.worst_pairs > 0:
        show_worst_pairs(embeddings, vehicle_ids, paths, meta, n=args.worst_pairs)
    if args.worst_pairs_img:
        n = args.worst_pairs if args.worst_pairs > 0 else 10
        plot_worst_pairs_images(embeddings, vehicle_ids, paths, meta,
                                n=n, save_path=args.worst_pairs_img,
                                images_dir=args.images)
    if args.camera_strat or args.heatmap or args.angle_gap_map:
        strat_results, azimuths = camera_stratified_eval(embeddings, vehicle_ids, paths, meta)
        if args.heatmap:
            plot_heatmap(strat_results, azimuths, save_path=args.heatmap)
        if args.angle_gap_map:
            angle_gap_map_breakdown(strat_results, azimuths)
    if args.pr_curve_by_gap:
        precision_recall_curve_by_gap(embeddings, vehicle_ids, paths, meta)
    if args.f1_by_gap:
        f1_by_gap_comparison(embeddings, vehicle_ids, paths, meta,
                             global_thresh=results['threshold'])
    if args.best_pairs > 0:
        show_best_pairs(embeddings, vehicle_ids, paths, meta, n=args.best_pairs)
    if args.best_pairs_img:
        n = args.best_pairs if args.best_pairs > 0 else 10
        plot_best_pairs_images(embeddings, vehicle_ids, paths, meta,
                               n=n, save_path=args.best_pairs_img,
                               images_dir=args.images)
    if args.fp_fn_imgs:
        os.makedirs(args.fp_fn_imgs, exist_ok=True)
        plot_fp_fn_images(embeddings, vehicle_ids, paths, meta,
                          opt_thresh=results['threshold'],
                          save_dir=args.fp_fn_imgs, images_dir=args.images)
    if args.vehicle_heatmap:
        plot_vehicle_heatmap(embeddings, vehicle_ids, save_path=args.vehicle_heatmap)
    if args.pr_curve or args.pr_csv:
        precision_recall_curve(embeddings, vehicle_ids, paths,
                               save_csv=args.pr_csv, meta=meta)


if __name__ == '__main__':
    main()