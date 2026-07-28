# ReID Model — Usage

Commands for training, evaluation, and diagnostics using the files in this repo. Run from the repo root.

## Folders

| Folder | Contents |
|---|---|
| `config/` | YAML configs — one per dataset/architecture combination |
| `data/` | Dataset loading classes (`triplet_sampler.py`) |
| `dataset/` | Raw dataset files (VeRi-776 / VehicleX / CityFlow etc. — not included, obtain separately) |
| `crops_final/` | AirSim orbit crops + `crops.json` metadata |
| `logs/` | Training output — checkpoints (`best_mAP.pt`) and saved configs, one subfolder per run |
| `loss/`, `lr_scheduler/`, `metrics/`, `models/` | Model architecture, loss functions, LR scheduling, mAP/CMC evaluation — imported by the scripts below, not run directly |
| `airsim_eval/` | Analysis/reporting package used by `teste_airsim.py` |

---

## 1. Config files

Configs live in `config/`. Naming convention: `config.yaml` is the default (Veri776, MBR_4B). Other files are `config_<Dataset>.yaml` for the default architecture, or `config_BoT_<Dataset>.yaml` for the BoT_baseline architecture:

| File | Dataset | Architecture |
|---|---|---|
| `config.yaml` | Veri776 | MBR_4B (default) |
| `config_VehicleID.yaml` | VehicleID | MBR_4B (default) |
| `config_VERIWILD.yaml` | VERIWILD | MBR_4B (default) |
| `config_BoT_Veri776.yaml` | Veri776 | BoT_baseline |
| `config_BoT_vehiclex.yaml` | VehicleX | BoT_baseline |
| `config_BoT_VehicleID.yaml` | VehicleID | BoT_baseline |
| `config_BoT_VERIWILD.yaml` | VERIWILD | BoT_baseline |

Every command below takes `--config path/to/config.yaml`; most fields can also be overridden via CLI flags (see `python main.py --help` / `python teste.py --help`).

---

## 2. Training — `main.py`

```bash
python main.py \
    --config config/config.yaml \
    --dataset Veri776 \
    --model_arch MBR_4B
```

Notable options:
- `--finetune_from <checkpoint>` — warm-start from an existing checkpoint instead of training from scratch
- `--eval_only <checkpoint>` — skip training, run one evaluation pass against the config's dataset and exit
- `LAI: true/false` in the config — toggles viewpoint-aware training (requires `train_keypoint`/`test_keypoint` annotation files to be set correctly when `true`)

Uses `processor.py` (`get_model`, `train_epoch`, `test_epoch`), `data/triplet_sampler.py`, `loss/`, `lr_scheduler/`, and `tensorboard_log.py` (`Logger`). Checkpoints and logs are written under `logs/<dataset>/<model_arch>/<run>/`, including `best_mAP.pt` and a saved `config.yaml`.

---

## 3. Evaluation — `teste.py`

```bash
python teste.py \
    --path_weights logs/Veri776/MBR_4B/1/ \
    --dataset Veri776
```

`--dataset` accepts `Veri776`, `VERIWILD`, `CityFlow`, `CityFlowVal`, `VehicleID`, or `VehicleX`.

**Cross-domain evaluation** (e.g. evaluating a VeRi-776-trained checkpoint on VehicleX) requires overriding the annotation/image paths explicitly, since the checkpoint's saved config only points at its own training dataset:
```bash
python teste.py \
    --path_weights logs/Veri776/MBR_4B/1/ \
    --dataset VehicleX \
    --train_list_file path/to/VehicleX/train_label.xml \
    --train_dir path/to/VehicleX/image_train/
```

**Re-ranking** (implemented in `utils.py`):
```bash
--re_rank                                   # plain k-reciprocal re-ranking
--re_rank --azimuth_rerank                  # + azimuth-gap neighbor exclusion (Veri776 only —
                                             #   the only dataset with real per-image orientation labels)
--k1 <int> --k2 <int>                       # neighbor-set sizes (defaults 80/16 — lower for small galleries)
--min_azimuth_gap_bins <int>                # min orientation-bin gap for azimuth_rerank (default 1, ~45°)
```

---

## 4. AirSim evaluation — `teste_airsim.py`

```bash
python teste_airsim.py \
    --path_weights logs/Veri776/MBR_4B/1/ \
    --airsim_images crops_final \
    --crops_json crops_final/crops.json
```

Prints `mAP`, `CMC@1`, `CMC@5`, and saves a full analysis report (via `airsim_eval/runner.py`) to `--results_dir` (default `results/`), timestamped per run.

Key flags:
- `--pass_metadata_ids_to_model` — feeds real camera/viewpoint IDs into the model instead of placeholders. Only safe if the checkpoint's camera-embedding table (`n_cams` in its config) covers the camera IDs actually present in your data — otherwise this crashes with an out-of-bounds embedding error.
- `--re_rank`, `--azimuth_rerank`, `--k1`, `--k2` — same meaning as `teste.py` above.
- `--breakdown` — per-vehicle metrics, saved to `vehicle_breakdown.csv`
- `--anglebymap` — mAP by query/gallery azimuth and by azimuth gap, saved to `angle_map.csv`/`angle_gap_map.csv`
- `--coveragebreakdown` — metrics by crop-coverage bucket, saved to `coverage.csv`
- `--bestpairs N` / `--bestpairsimg` — top-N best-scoring correct retrieval pairs, as CSV / image grid
- `--fp_fnimgs` — Top-K retrieval errors (wrong gallery matches and queries with no correct match), CSV + image grids; `--fpfn_topk N` sets K (default 1)
- `--vehicleheatmap` — per-vehicle distance heatmap, saved to `vehicle_heatmap.png`
- `--heatmap` — query/gallery azimuth mAP heatmap, saved to `angle_heatmap.png`
- `--image_limit N` — caps images per plotted grid (default 50)

Every run also writes `metrics.json` and `per_query.csv` regardless of which optional flags are set.

---

## 5. Other files

- **`inference_fixed.py`** — provides `load_model`/`extract_embedding`, imported by `reid_crops.py`. Not run standalone in any confirmed usage.
- **`vehicle_similarity_matrix.py`** — usage not documented here; refer to its own `--help` output or docstring.
- **`utils.py`** — `re_ranking`/`re_ranking_azimuth`, used internally by `teste.py`/`teste_airsim.py` (Sections 3–4); not run standalone.
- **`tensorboard_log.py`** — provides the `Logger` class used internally by `main.py`/`teste_airsim.py`; not run standalone. View training curves with `tensorboard --logdir logs/`.

---

## Notes
- Dataset downloads (VeRi-776, VehicleX, CityFlow) are not included and are out of scope for this README — obtain them separately and update the relevant config paths.