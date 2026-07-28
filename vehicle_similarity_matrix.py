"""
vehicle_similarity_matrix.py
-----------------------------
Computes and displays a matrix of average cosine similarity scores
between every pair of vehicle IDs.

- Diagonal: intra-vehicle (same vehicle, different images) — should be HIGH
- Off-diagonal: inter-vehicle (different vehicles) — should be LOW

A good ReID model shows a clear separation between diagonal and off-diagonal.

Usage:
    python vehicle_similarity_matrix.py --save_emb simulation/embeddings_clean.pt
    python vehicle_similarity_matrix.py --save_emb simulation/embeddings_clean.pt --cam_filter
    python vehicle_similarity_matrix.py --save_emb simulation/embeddings_clean.pt --cam_filter --save_csv simulation/sim_matrix.csv
"""

import os
import argparse
import torch
import numpy as np
from collections import defaultdict


def parse_camera(filepath):
    base  = os.path.splitext(os.path.basename(filepath))[0]
    parts = base.split('_')
    return '_'.join(parts[2:-3])


CAM_GROUPS = {
    'front': {'angled_front', 'angled_front_left', 'angled_front_right', 'angled_left'},
    'back':  {'angled_back',  'angled_back_left',  'angled_back_right',  'angled_right'},
}

ALLOWED_PAIRS = {
    (q, g)
    for group_cams in CAM_GROUPS.values()
    for q in group_cams for g in group_cams
}


def main():
    parser = argparse.ArgumentParser(
        description='Vehicle-to-vehicle average cosine similarity matrix.')
    parser.add_argument('--save_emb',  required=True,
                        help='Path to embedding cache .pt file')
    parser.add_argument('--cam_filter', action='store_true',
                        help='Restrict to front/back camera groups only')
    parser.add_argument('--save_csv',  default=None,
                        help='Save matrix to CSV file')
    parser.add_argument('--save_img',  default='vehicle_sim_matrix.png',
                        help='Save heatmap image (default: vehicle_sim_matrix.png)')
    args = parser.parse_args()

    # ── Load embeddings ───────────────────────────────────────────────────────
    data    = torch.load(args.save_emb, map_location='cpu')
    embs    = data['embeddings']
    paths   = data['paths']
    vids    = data['vehicle_ids']
    cameras = [parse_camera(p) for p in paths]
    N       = len(paths)

    print(f"Loaded {N} embeddings")

    # ── Full similarity matrix ────────────────────────────────────────────────
    print("Computing similarity matrix ...")
    sim_matrix = torch.mm(embs, embs.t())

    # ── Build vehicle index groups ────────────────────────────────────────────
    vehicle_indices = defaultdict(list)
    for i, vid in enumerate(vids):
        if args.cam_filter:
            if cameras[i] not in {c for g in CAM_GROUPS.values() for c in g}:
                continue
        vehicle_indices[vid].append(i)

    vehicle_ids_sorted = sorted(vehicle_indices.keys())
    V = len(vehicle_ids_sorted)
    print(f"Vehicles in matrix: {V}")

    # ── Compute average pairwise similarity between every vehicle pair ────────
    avg_matrix  = np.zeros((V, V))
    count_matrix = np.zeros((V, V))

    for vi, vid_a in enumerate(vehicle_ids_sorted):
        for vj, vid_b in enumerate(vehicle_ids_sorted):
            idxs_a = vehicle_indices[vid_a]
            idxs_b = vehicle_indices[vid_b]

            scores = []
            for i in idxs_a:
                for j in idxs_b:
                    if i == j:
                        continue
                    if args.cam_filter:
                        ci, cj = cameras[i], cameras[j]
                        if (ci, cj) not in ALLOWED_PAIRS and (cj, ci) not in ALLOWED_PAIRS:
                            continue
                    scores.append(sim_matrix[i, j].item())

            if scores:
                avg_matrix[vi, vj]   = sum(scores) / len(scores)
                count_matrix[vi, vj] = len(scores)

    # ── Print text table ──────────────────────────────────────────────────────
    col_w = 10
    abbrev = {v: v.replace('Parked_', 'P') for v in vehicle_ids_sorted}

    header = f"{'':>12}" + ''.join(f"{abbrev[v]:>{col_w}}" for v in vehicle_ids_sorted)
    print(f"\nAverage cosine similarity matrix"
          f"{' (cam_filter)' if args.cam_filter else ''}:")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for vi, vid_a in enumerate(vehicle_ids_sorted):
        row = f"{abbrev[vid_a]:>12}"
        for vj, vid_b in enumerate(vehicle_ids_sorted):
            val = avg_matrix[vi, vj]
            row += f"{val:>{col_w}.3f}"
        print(row)
    print("=" * len(header))

    # ── Summary stats ─────────────────────────────────────────────────────────
    diag_vals  = [avg_matrix[i, i] for i in range(V)]
    off_diag   = [avg_matrix[i, j] for i in range(V)
                  for j in range(V) if i != j and avg_matrix[i,j] != 0]

    print(f"\nDiagonal (intra-vehicle):")
    print(f"  Mean : {np.mean(diag_vals):.4f}")
    print(f"  Min  : {np.min(diag_vals):.4f}  ({vehicle_ids_sorted[np.argmin(diag_vals)]})")
    print(f"  Max  : {np.max(diag_vals):.4f}  ({vehicle_ids_sorted[np.argmax(diag_vals)]})")

    print(f"\nOff-diagonal (inter-vehicle):")
    print(f"  Mean : {np.mean(off_diag):.4f}")
    print(f"  Max  : {np.max(off_diag):.4f}")
    print(f"  Min  : {np.min(off_diag):.4f}")

    print(f"\nSeparation (intra mean - inter mean): "
          f"{np.mean(diag_vals) - np.mean(off_diag):.4f}")

    # ── Hardest confusions (highest off-diagonal) ─────────────────────────────
    print(f"\nTop 10 most confused vehicle pairs (highest inter-vehicle similarity):")
    off_diag_pairs = [
        (avg_matrix[vi, vj], vehicle_ids_sorted[vi], vehicle_ids_sorted[vj])
        for vi in range(V) for vj in range(vi + 1, V)
        if avg_matrix[vi, vj] > 0
    ]
    off_diag_pairs.sort(reverse=True)
    for score, va, vb in off_diag_pairs[:10]:
        print(f"  {va:<15} ↔ {vb:<15}  avg_sim={score:.4f}")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    if args.save_csv:
        import csv
        with open(args.save_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([''] + vehicle_ids_sorted)
            for vi, vid_a in enumerate(vehicle_ids_sorted):
                writer.writerow([vid_a] + [f"{avg_matrix[vi,vj]:.4f}"
                                           for vj in range(V)])
        print(f"\nMatrix saved to {args.save_csv}")

    # ── Heatmap ───────────────────────────────────────────────────────────────
    try:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors

        fig, ax = plt.subplots(figsize=(max(10, V * 0.6), max(8, V * 0.55)))

        labels = [abbrev[v] for v in vehicle_ids_sorted]

        # Use a diverging colourmap centred at 0.5
        im = ax.imshow(avg_matrix, cmap='RdYlGn', vmin=0.0, vmax=1.0,
                       aspect='auto')

        ax.set_xticks(range(V))
        ax.set_yticks(range(V))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)

        # Annotate cells
        for vi in range(V):
            for vj in range(V):
                val = avg_matrix[vi, vj]
                if val != 0:
                    colour = 'black' if 0.3 < val < 0.75 else 'white'
                    ax.text(vj, vi, f"{val:.2f}", ha='center', va='center',
                            fontsize=6, color=colour)

        plt.colorbar(im, ax=ax, label='Average Cosine Similarity')
        title = f"Vehicle Similarity Matrix{' (cam_filter)' if args.cam_filter else ''}"
        ax.set_title(title, fontsize=12, pad=15)
        ax.set_xlabel("Gallery Vehicle")
        ax.set_ylabel("Query Vehicle")

        # Highlight diagonal
        for i in range(V):
            ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1,
                                       fill=False, edgecolor='blue',
                                       linewidth=1.5))

        plt.tight_layout()
        plt.savefig(args.save_img, dpi=150, bbox_inches='tight')
        print(f"Heatmap saved to {args.save_img}")
        plt.show()

    except ImportError:
        print("matplotlib not available — skipping heatmap")


if __name__ == '__main__':
    main()