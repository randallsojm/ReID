"""Evaluate the existing vehicle-ReID model on an AirSim crop dataset.

This is an additive entry point: it does not import or modify the executable
body of the original ``teste.py``.  It reuses the repository's established
``processor.get_model``, ``utils.re_ranking`` and ``metrics.eval_reid`` APIs.
Run it from the original repository root after copying this file and the
``airsim_eval`` directory there.
"""

from __future__ import annotations

import argparse
import random
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from airsim_eval.context import EvaluationContext
from airsim_eval.dataset import azimuth_to_bin, build_airsim_datasets
from airsim_eval.runner import AnalysisOptions, run_analyses
from metrics.eval_reid import eval_func
from processor import get_model
from utils import re_ranking, re_ranking_azimuth


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_model_files(path_weights: str) -> tuple[Path, Path]:
    supplied = Path(path_weights).expanduser()
    if supplied.is_file():
        weights = supplied
        config = supplied.parent / "config.yaml"
    else:
        weights = supplied / "best_mAP.pt"
        config = supplied / "config.yaml"
    if not weights.is_file():
        raise FileNotFoundError(f"Model weights not found: {weights}")
    if not config.is_file():
        raise FileNotFoundError(f"Model config not found: {config}")
    return weights.resolve(), config.resolve()


def load_checkpoint(model: torch.nn.Module, weights_path: Path) -> list[str]:
    value = torch.load(weights_path, map_location="cpu")
    if isinstance(value, dict) and "state_dict" in value:
        value = value["state_dict"]
    if not isinstance(value, dict):
        raise TypeError(f"Unsupported checkpoint object: {type(value).__name__}")
    cleaned = OrderedDict(
        (
            key[len("module.") :] if key.startswith("module.") else key,
            tensor,
        )
        for key, tensor in value.items()
    )
    current = model.state_dict()
    compatible = {
        key: tensor
        for key, tensor in cleaned.items()
        if key in current and getattr(tensor, "shape", None) == current[key].shape
    }
    skipped = sorted(set(cleaned) - set(compatible))
    missing, unexpected = model.load_state_dict(compatible, strict=False)
    if unexpected:
        skipped.extend(unexpected)
    material_missing = [key for key in missing if not _looks_like_classifier(key)]
    if material_missing:
        print("WARNING: non-classifier checkpoint keys were missing:")
        for key in material_missing:
            print(f"  {key}")
    return sorted(set(skipped))


def _looks_like_classifier(key: str) -> bool:
    lower = key.lower()
    return any(token in lower for token in ("classifier", "head", "fc", "logits"))


def extract_features(model, device, dataloader, description: str, half_precision: bool):
    features: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for images, _pids, camids, viewids in tqdm(dataloader, desc=description):
            images = images.to(device, non_blocking=True)
            camids = camids.to(device, non_blocking=True)
            viewids = viewids.to(device, non_blocking=True)
            autocast_enabled = half_precision and device.type == "cuda"
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=autocast_enabled,
            ):
                output = model(images, camids, viewids)
                if not isinstance(output, (tuple, list)) or len(output) < 3:
                    raise TypeError(
                        "Expected model(image, cam_id, view_id) to return a tuple "
                        "whose third item is the feature list"
                    )
                feature_parts = output[2]
                if torch.is_tensor(feature_parts):
                    feature_parts = [feature_parts]
                batch = torch.cat([F.normalize(item, dim=1) for item in feature_parts], dim=1)
            features.append(batch.detach().cpu())
    if not features:
        raise RuntimeError(f"No features were extracted from {description}")
    return torch.cat(features, dim=0)


def calculate_distance_matrix(
    query_features, gallery_features, *,
    re_rank, azimuth_rerank, query_azimuths, gallery_azimuths,
    azimuth_bins, min_azimuth_gap_bins, k1=80, k2=16,   # NEW
):
    if re_rank and azimuth_rerank:
        q_view = torch.tensor(
            [azimuth_to_bin(value, azimuth_bins) for value in query_azimuths],
            dtype=torch.long,
        )
        g_view = torch.tensor(
            [azimuth_to_bin(value, azimuth_bins) for value in gallery_azimuths],
            dtype=torch.long,
        )
        matrix = re_ranking_azimuth(
            query_features,
            gallery_features,
            k1=k1,
            k2=k2,
            lambda_value=0.3,
            q_view=q_view,
            g_view=g_view,
            min_azimuth_gap_bins=min_azimuth_gap_bins,
        )
        return np.asarray(matrix), "k-reciprocal-azimuth"
    if re_rank:
        matrix = re_ranking(
            query_features,
            gallery_features,
            k1=k1,
            k2=k2,
            lambda_value=0.3,
        )
        return np.asarray(matrix), "k-reciprocal"
    return torch.cdist(query_features, gallery_features, p=2).numpy(), "euclidean"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate ReID on AirSim orbit crops")
    parser.add_argument(
        "--path_weights",
        required=True,
        help="Checkpoint file, or directory containing best_mAP.pt and config.yaml",
    )
    parser.add_argument("--airsim_images", "--images", required=True, dest="airsim_images")
    parser.add_argument("--crops_json", required=True)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--model_arch", default=None)
    parser.add_argument("--re_rank", action="store_true")
    parser.add_argument("--azimuth_rerank", action="store_true")
    parser.add_argument("--azimuth_bins", type=int, default=8)
    parser.add_argument("--min_azimuth_gap_bins", type=int, default=1)
    parser.add_argument(
        "--pass_metadata_ids_to_model",
        action="store_true",
        help="Pass AirSim camera/view IDs into the model; unsafe unless its embedding tables support them",
    )
    parser.add_argument("--k1", type=int, default=80)
    parser.add_argument("--k2", type=int, default=16)
    parser.add_argument(
        "--skip_invalid_records",
        action="store_true",
        help="Warn and skip invalid JSON records instead of failing",
    )
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--anglebymap", "--angle_by_map", action="store_true", dest="angle_by_map")
    parser.add_argument(
        "--coveragebreakdown",
        "--coverage_breakdown",
        action="store_true",
        dest="coverage_breakdown",
    )
    parser.add_argument(
        "--bestpairs", "--best_pairs", type=int, default=0, metavar="N", dest="best_pairs"
    )
    parser.add_argument(
        "--bestpairsimg", "--best_pairs_img", action="store_true", dest="best_pairs_img"
    )
    parser.add_argument("--fp_fnimgs", "--fp_fn_imgs", action="store_true", dest="fp_fn_imgs")
    parser.add_argument(
        "--vehicleheatmap", "--vehicle_heatmap", action="store_true", dest="vehicle_heatmap"
    )
    parser.add_argument("--heatmap", action="store_true")
    parser.add_argument("--breakdown", action="store_true")
    parser.add_argument("--fpfn_topk", type=int, default=1)
    parser.add_argument("--image_limit", type=int, default=50)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.azimuth_rerank and not args.re_rank:
        raise ValueError("--azimuth_rerank requires --re_rank")
    if args.fpfn_topk < 1:
        raise ValueError("--fpfn_topk must be at least 1")
    set_seed(0)

    weights_path, config_path = resolve_model_files(args.path_weights)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["dataset"] = "AirSim"
    if args.batch_size is not None:
        config["BATCH_SIZE"] = args.batch_size
    if args.model_arch is not None:
        config["model_arch"] = args.model_arch

    transform = transforms.Compose(
        [
            transforms.Resize((config["y_length"], config["x_length"]), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(config["n_mean"], config["n_std"]),
        ]
    )
    query_dataset, gallery_dataset, pid_map, camera_map = build_airsim_datasets(
        args.airsim_images,
        args.crops_json,
        transform,
        strict=not args.skip_invalid_records,
        pass_metadata_ids_to_model=args.pass_metadata_ids_to_model,
        azimuth_bins=args.azimuth_bins,
    )
    workers = args.num_workers if args.num_workers is not None else config.get("num_workers_teste", 0)
    loader_kwargs = dict(
        batch_size=config["BATCH_SIZE"],
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )
    query_loader = DataLoader(query_dataset, **loader_kwargs)
    gallery_loader = DataLoader(gallery_dataset, **loader_kwargs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Selected device: {device}")
    print(
        f"AirSim split: {len(query_dataset)} queries, {len(gallery_dataset)} gallery, "
        f"{len(pid_map)} vehicles, {len(camera_map)} cameras"
    )
    model = get_model(config, torch.device("cpu"))
    skipped = load_checkpoint(model, weights_path)
    if skipped:
        print(f"Skipped {len(skipped)} absent or shape-mismatched checkpoint tensors")
    model = model.to(device).eval()

    half_precision = bool(config.get("half_precision", False))
    qf = extract_features(model, device, query_loader, "Query infer", half_precision)
    gf = extract_features(model, device, gallery_loader, "Gallery infer", half_precision)
    distance_matrix, distance_name = calculate_distance_matrix(
        qf, gf,
        re_rank=args.re_rank,
        azimuth_rerank=args.azimuth_rerank,
        query_azimuths=query_dataset.azimuths,
        gallery_azimuths=gallery_dataset.azimuths,
        azimuth_bins=args.azimuth_bins,
        min_azimuth_gap_bins=args.min_azimuth_gap_bins,
        k1=args.k1, k2=args.k2,   # NEW
    )

    cmc, mean_ap = eval_func(
        distance_matrix,
        np.asarray(query_dataset.pids),
        np.asarray(gallery_dataset.pids),
        np.asarray(query_dataset.camids),
        np.asarray(gallery_dataset.camids),
        remove_junk=True,
    )
    cmc1 = cmc[0] if len(cmc) else float("nan")
    cmc5 = cmc[4] if len(cmc) > 4 else float("nan")
    print(f"mAP={mean_ap:.6f}, CMC@1={cmc1:.6f}, CMC@5={cmc5:.6f}")

    context = EvaluationContext(
        distance_matrix=distance_matrix,
        query_features=qf,
        gallery_features=gf,
        query_records=query_dataset.records,
        gallery_records=gallery_dataset.records,
        reranked=args.re_rank,
        distance_name=distance_name,
    )
    options = AnalysisOptions(
        results_dir=args.results_dir,
        angle_by_map=args.angle_by_map,
        coverage_breakdown=args.coverage_breakdown,
        best_pairs=args.best_pairs,
        best_pairs_img=args.best_pairs_img,
        fp_fn_imgs=args.fp_fn_imgs,
        vehicle_heatmap=args.vehicle_heatmap,
        heatmap=args.heatmap,
        breakdown=args.breakdown,
        fpfn_topk=args.fpfn_topk,
        image_limit=args.image_limit,
    )
    output = run_analyses(context, options)
    print(f"Results saved to: {output.resolve()}")


if __name__ == "__main__":
    main()
