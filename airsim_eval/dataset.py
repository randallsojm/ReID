"""AirSim orbit-crop dataset loader.

The loader follows the contract inferred from ``reid_crops.py``:

* ``crops.json`` is a top-level list of objects.
* crop images live directly inside one flat directory.
* records and images are joined by the exact ``crop_filename`` basename.
* the first valid JSON record for each vehicle is the deterministic query.
* remaining records for that vehicle form the gallery.

Actual camera and azimuth metadata are retained on ``AirSimRecord``.  By
default the dataset returns zero camera/view IDs to the model to avoid going
outside camera/view embedding tables learned on another dataset.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import torch
from PIL import Image
from torch.utils.data import Dataset


REQUIRED_FIELDS = ("crop_filename", "vehicle_label", "camera", "azimuth")
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg"}


@dataclass(frozen=True)
class AirSimRecord:
    """One validated crop record with numeric evaluation IDs."""

    path: Path
    crop_filename: str
    vehicle_label: str
    pid: int
    camera: str
    camid: int
    azimuth: float
    raw: Mapping[str, Any]

    @property
    def elevation(self) -> float | None:
        value = self.raw.get("elevation")
        return float(value) if value is not None else None

    @property
    def crop_coverage(self) -> float | None:
        """Return the coverage definition used by the original script."""

        quality = self.raw.get("quality") or {}
        area = quality.get("area")
        aspect = quality.get("aspect")
        if self.raw.get("resized_down", False) and aspect is not None:
            aspect = float(aspect)
            return min(aspect, 1.0 / aspect) if aspect > 0 else None
        return float(area) / (224.0 * 224.0) if area is not None else None

    def as_dict(self) -> dict[str, Any]:
        result = dict(self.raw)
        result.update(
            path=str(self.path),
            crop_filename=self.crop_filename,
            vehicle_label=self.vehicle_label,
            pid=self.pid,
            camera=self.camera,
            camid=self.camid,
            azimuth=self.azimuth,
        )
        return result


class AirSimDataset(Dataset):
    """PyTorch dataset returning the four values expected by ``teste.py``."""

    def __init__(
        self,
        records: Sequence[AirSimRecord],
        transform: Callable[[Image.Image], torch.Tensor] | None = None,
        *,
        pass_metadata_ids_to_model: bool = False,
        azimuth_bins: int = 8,
    ) -> None:
        self.records = list(records)
        self.transform = transform
        self.pass_metadata_ids_to_model = pass_metadata_ids_to_model
        self.azimuth_bins = azimuth_bins
        if azimuth_bins < 1:
            raise ValueError("azimuth_bins must be at least 1")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        with Image.open(record.path) as source:
            image = source.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        if self.pass_metadata_ids_to_model:
            model_camid = record.camid
            model_viewid = azimuth_to_bin(record.azimuth, self.azimuth_bins)
        else:
            model_camid = 0
            model_viewid = 0
        return image, record.pid, model_camid, model_viewid

    @property
    def pids(self) -> list[int]:
        return [record.pid for record in self.records]

    @property
    def camids(self) -> list[int]:
        return [record.camid for record in self.records]

    @property
    def azimuths(self) -> list[float]:
        return [record.azimuth for record in self.records]

    @property
    def paths(self) -> list[str]:
        return [str(record.path) for record in self.records]


def azimuth_to_bin(azimuth: float, bins: int = 8) -> int:
    """Map degrees to the nearest circular orientation bin."""

    width = 360.0 / bins
    return int((float(azimuth) % 360.0 + width / 2.0) // width) % bins


def _load_json(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, list):
        raise ValueError(f"{path} must contain a top-level JSON list")
    if not value:
        raise ValueError(f"{path} contains no crop records")
    if not all(isinstance(item, dict) for item in value):
        raise ValueError(f"Every item in {path} must be a JSON object")
    return value


def load_airsim_records(
    images_dir: str | Path,
    crops_json: str | Path,
    *,
    strict: bool = True,
) -> tuple[list[AirSimRecord], dict[str, int], dict[str, int]]:
    """Validate metadata and resolve records in original JSON order."""

    image_root = Path(images_dir).expanduser().resolve()
    json_path = Path(crops_json).expanduser().resolve()
    if not image_root.is_dir():
        raise FileNotFoundError(f"AirSim image directory does not exist: {image_root}")
    if not json_path.is_file():
        raise FileNotFoundError(f"AirSim crops JSON does not exist: {json_path}")

    raw_records = _load_json(json_path)
    vehicle_labels: list[str] = []
    camera_names: list[str] = []
    valid_raw: list[tuple[dict[str, Any], Path]] = []
    seen_filenames: set[str] = set()
    problems: list[str] = []

    for index, item in enumerate(raw_records):
        missing = [field for field in REQUIRED_FIELDS if field not in item]
        if missing:
            problems.append(f"record {index}: missing {', '.join(missing)}")
            continue
        filename = str(item["crop_filename"])
        if Path(filename).name != filename:
            problems.append(
                f"record {index}: crop_filename must be a basename, got {filename!r}"
            )
            continue
        if filename in seen_filenames:
            problems.append(f"record {index}: duplicate crop_filename {filename!r}")
            continue
        seen_filenames.add(filename)
        path = image_root / filename
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            problems.append(f"record {index}: unsupported image extension for {filename!r}")
            continue
        if not path.is_file():
            problems.append(f"record {index}: missing image {path}")
            continue
        try:
            float(item["azimuth"])
        except (TypeError, ValueError):
            problems.append(f"record {index}: azimuth must be numeric")
            continue
        vehicle = str(item["vehicle_label"])
        camera = str(item["camera"])
        if vehicle not in vehicle_labels:
            vehicle_labels.append(vehicle)
        if camera not in camera_names:
            camera_names.append(camera)
        valid_raw.append((item, path))

    if problems:
        message = "Invalid AirSim crop metadata:\n  - " + "\n  - ".join(problems)
        if strict:
            raise ValueError(message)
        warnings.warn(message, stacklevel=2)
    if not valid_raw:
        raise ValueError("No valid AirSim crop records remain after validation")

    pid_map = {label: index for index, label in enumerate(vehicle_labels)}
    camera_map = {name: index for index, name in enumerate(camera_names)}
    records = [
        AirSimRecord(
            path=path,
            crop_filename=str(item["crop_filename"]),
            vehicle_label=str(item["vehicle_label"]),
            pid=pid_map[str(item["vehicle_label"])],
            camera=str(item["camera"]),
            camid=camera_map[str(item["camera"])],
            azimuth=float(item["azimuth"]),
            raw=item,
        )
        for item, path in valid_raw
    ]
    return records, pid_map, camera_map


def deterministic_query_gallery_split(
    records: Iterable[AirSimRecord],
) -> tuple[list[AirSimRecord], list[AirSimRecord]]:
    """Use the first valid JSON crop per vehicle as query."""

    grouped: dict[int, list[AirSimRecord]] = {}
    for record in records:
        grouped.setdefault(record.pid, []).append(record)

    query: list[AirSimRecord] = []
    gallery: list[AirSimRecord] = []
    singletons: list[str] = []
    for items in grouped.values():
        if len(items) < 2:
            singletons.append(items[0].vehicle_label)
            continue
        query.append(items[0])
        gallery.extend(items[1:])
    if singletons:
        warnings.warn(
            "Vehicles with fewer than two valid crops were excluded: "
            + ", ".join(singletons),
            stacklevel=2,
        )
    if not query or not gallery:
        raise ValueError("The AirSim split needs at least one vehicle with two valid crops")
    return query, gallery


def build_airsim_datasets(
    images_dir: str | Path,
    crops_json: str | Path,
    transform: Callable[[Image.Image], torch.Tensor] | None,
    *,
    strict: bool = True,
    pass_metadata_ids_to_model: bool = False,
    azimuth_bins: int = 8,
) -> tuple[AirSimDataset, AirSimDataset, dict[str, int], dict[str, int]]:
    records, pid_map, camera_map = load_airsim_records(
        images_dir, crops_json, strict=strict
    )
    query_records, gallery_records = deterministic_query_gallery_split(records)
    kwargs = dict(
        transform=transform,
        pass_metadata_ids_to_model=pass_metadata_ids_to_model,
        azimuth_bins=azimuth_bins,
    )
    return (
        AirSimDataset(query_records, **kwargs),
        AirSimDataset(gallery_records, **kwargs),
        pid_map,
        camera_map,
    )
