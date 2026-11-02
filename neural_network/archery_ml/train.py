"""Config-driven, manifest-bound Stage 4 training and independent evaluation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import ImageDraw, Image
from torch.utils.data import DataLoader

from neural_network.archery_ml.contracts import ModelInputSpec, ModelMetadata, ModelOutputSpec, ModelPostprocessSpec
from neural_network.archery_ml.data.snapshot import load_snapshot
from neural_network.archery_ml.data.view import DatasetView
from neural_network.archery_ml.evaluate import evaluate_faces
from neural_network.archery_ml.impact_dataset import CanonicalImpactDataset, collate_faces
from neural_network.archery_ml.inference import load_detector
from neural_network.archery_ml.heatmap import impact_loss
from neural_network.archery_ml.models import ImpactPointNet
from neural_network.archery_ml.postprocess import decode_impacts


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_configuration(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    model = config["model"]
    training = config["training"]
    dataset = config["dataset"]
    if model["input_size"] < 64 or model["input_size"] % 32 or model["output_stride"] != 2:
        raise ValueError("model must use input multiple of 32 and output stride two")
    if any(training[name] <= 0 for name in ("epochs", "batch_size", "learning_rate")):
        raise ValueError("training parameters must be positive")
    if training["weight_decay"] < 0 or model["fpn_channels"] <= 0:
        raise ValueError("invalid model or optimizer settings")
    if model.get("pretrained", False):
        # V2 ImageNet weights were trained with this normalization. Failing
        # early is safer than silent data distribution mismatch at inference.
        if (list(model.get("input_mean", ())) != [0.485, 0.456, 0.406]
                or list(model.get("input_std", ())) != [0.229, 0.224, 0.225]):
            raise ValueError("pretrained MobileNetV3 requires ImageNet V2 RGB normalization")
    if not dataset.get("snapshot") or not dataset.get("manifest"):
        raise ValueError("manifest and snapshot are required")
    return config


def _prepare(config: dict, split: str) -> CanonicalImpactDataset:
    dataset_config = config["dataset"]
    manifest = Path(dataset_config["manifest"])
    view = DatasetView.load(manifest).select(splits=[split])
    return CanonicalImpactDataset(view, manifest_dir=manifest.parent,
                                  input_mean=tuple(config["model"].get("input_mean", (0, 0, 0))),
                                  input_std=tuple(config["model"].get("input_std", (1, 1, 1))),
                                  input_size=config["model"]["input_size"],
                                  augment=split == "train",
                                  sigma=float(config["training"].get("gaussian_sigma", 1.5)),
                                  horizontal_flip_prob=float(config["training"].get("horizontal_flip_prob", 0.5)),
                                  brightness=float(config["training"].get("brightness", 0.15)))


def _verify_dataset(config: dict) -> str:
    manifest = Path(config["dataset"]["manifest"])
    snapshot = load_snapshot(config["dataset"]["snapshot"])
    digest = _sha256(manifest)
    if snapshot.manifest_sha256 != digest:
        raise ValueError("manifest SHA256 differs from immutable dataset snapshot")
    view = DatasetView.load(manifest)
    groups: dict[tuple[str | None, str], str] = {}
    sample_ids: set[str] = set()
    image_splits: dict[Path, str] = {}
    image_hash_splits: dict[str, str] = {}
    for sample in view:
        if sample.id in sample_ids:
            raise ValueError(f"duplicate manifest sample ID: {sample.id}")
        sample_ids.add(sample.id)
        if (sample.capture_metadata or {}).get("canonical_faces") is not None:
            if not sample.group_id or not sample.source_id:
                raise ValueError(f"canonical face requires source_id and group_id: {sample.id}")
        if sample.split not in ("train", "val", "test", "mobile_dev", "mobile_real_test"):
            raise ValueError("unknown split")
        if sample.group_id is not None:
            key = (sample.source_id, sample.group_id)
            previous = groups.setdefault(key, sample.split)
            if previous != sample.split:
                raise ValueError(f"group leakage between {previous} and {sample.split}: {key}")
        if (sample.capture_metadata or {}).get("canonical_faces") is not None:
            image_path = sample.image_path.resolve()
            previous_image_split = image_splits.setdefault(image_path, sample.split)
            if previous_image_split != sample.split:
                raise ValueError(f"raw image leakage between {previous_image_split} and {sample.split}")
            if not image_path.is_file():
                raise ValueError(f"missing original source image: {sample.id}")
            original_sha = _sha256(image_path)
            previous_hash_split = image_hash_splits.setdefault(original_sha, sample.split)
            if previous_hash_split != sample.split:
                raise ValueError(f"original image SHA256 leakage between {previous_hash_split} and {sample.split}")
    return digest


def _seed(seed: int) -> None:
    # Keep reproducible CPU reductions regardless of OMP/MKL environment defaults.
    # Set before any model execution; deterministic multilayer CPU kernels can
    # otherwise exhibit cross-run numerical drift under parallel reductions.
    torch.set_num_threads(1)
    # Deterministic CUDA GEMM on supported PyTorch/CUDA versions; ignored on CPU.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def _loader(dataset: CanonicalImpactDataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, collate_fn=collate_faces)


def _validation(model: ImpactPointNet, loader: DataLoader, device: torch.device,
                config: dict) -> tuple[float, dict, tuple | None]:
    model.eval()
    losses = []
    faces = []
    first_visualization = None
    threshold = float(config["postprocess"]["confidence_threshold"])
    radius = int(config["postprocess"]["nms_radius"])
    with torch.inference_mode():
        for batch in loader:
            outputs = model(batch["image"].to(device))
            truth = (batch["heatmap"].to(device), batch["offsets"].to(device),
                     batch["positive"].to(device))
            loss, _ = impact_loss(outputs, truth)
            losses.append(float(loss))
            for i, record in enumerate(batch["records"]):
                preds = decode_impacts(outputs[0][i].cpu(), outputs[1][i].cpu(),
                                       confidence_threshold=threshold, nms_radius=radius,
                                       face_index=record["face_index"])
                faces.append((preds, record["points"], record["target_metadata"]))
                if first_visualization is None:
                    first_visualization = (batch["image"][i].clone(), preds, record["points"])
    results = evaluate_faces(faces).to_dict()
    return sum(losses) / len(losses), results, first_visualization


def _visualize(visualization: tuple, output: Path, config: dict) -> None:
    tensor, predictions, truths = visualization
    mean = torch.tensor(config["model"].get("input_mean", (0, 0, 0))).view(3, 1, 1)
    std = torch.tensor(config["model"].get("input_std", (1, 1, 1))).view(3, 1, 1)
    pixels = ((tensor * std + mean).permute(1, 2, 0).numpy() * 255).clip(0, 255)
    image = Image.fromarray(pixels.astype("uint8"))
    draw = ImageDraw.Draw(image)
    size = image.width
    for x, y in truths:
        cx, cy = (x + 1) * (size - 1) / 2, (y + 1) * (size - 1) / 2
        draw.ellipse((cx - 4, cy - 4, cx + 4, cy + 4), outline="lime", width=2)
    for prediction in predictions:
        cx = (prediction.x_norm + 1) * (size - 1) / 2
        cy = (prediction.y_norm + 1) * (size - 1) / 2
        draw.line((cx - 4, cy, cx + 4, cy), fill="red", width=2)
        draw.line((cx, cy - 4, cx, cy + 4), fill="red", width=2)
    image.save(output)


def _checkpoint(model: ImpactPointNet, config: dict, epoch: int) -> dict:
    return {"model_state": model.state_dict(), "model_config": {
        "fpn_channels": config["model"]["fpn_channels"]},
        "model_version": config["model"]["version"], "epoch": epoch}


def train(config_path: str | Path, *, output_root: str | Path = "runs", device: str = "cpu") -> Path:
    config_path = Path(config_path)
    config = _load_configuration(config_path)
    dataset_hash = _verify_dataset(config)
    _seed(int(config["seed"]))
    train_set, val_set = _prepare(config, "train"), _prepare(config, "val")
    for name, dataset in (("train", train_set), ("val", val_set)):
        if not any(record.points for record in dataset.records):
            raise ValueError(f"{name} has no verified positive impact annotations")
    batch_size = int(config["training"]["batch_size"])
    loader_train, loader_val = _loader(train_set, batch_size, True), _loader(val_set, batch_size, False)
    model = ImpactPointNet(fpn_channels=int(config["model"]["fpn_channels"]),
                           pretrained=bool(config["model"].get("pretrained", False))).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"],
                                   weight_decay=config["training"]["weight_decay"])
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    root = Path(output_root) / f"impact-{run_id}"
    root.mkdir(parents=True, exist_ok=False)
    (root / "visualizations").mkdir()
    (root / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                         stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    info = {"manifest_sha256": dataset_hash,
            "snapshot_sha256": _sha256(Path(config["dataset"]["snapshot"])),
            "model_version": config["model"]["version"], "git_commit": commit,
            "backbone_weights": "IMAGENET1K_V2" if config["model"].get("pretrained") else None,
            "runtime": {"python": platform.python_version(),
                        "torch": torch.__version__, "torchvision": torchvision.__version__},
            "seed": config["seed"], "train_faces": len(train_set), "val_faces": len(val_set),
            "excluded_unprepared_train_samples": train_set.unprepared_samples,
            "excluded_unprepared_val_samples": val_set.unprepared_samples,
            "train_positive_impacts": sum(len(item.points) for item in train_set.records),
            "val_positive_impacts": sum(len(item.points) for item in val_set.records),
            "train_verified_empty_faces": sum(not item.points for item in train_set.records),
            "val_verified_empty_faces": sum(not item.points for item in val_set.records)}
    (root / "dataset_info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    metadata = ModelMetadata(
        contract_version=2, model_version=config["model"]["version"],
        input=ModelInputSpec(width=config["model"]["input_size"], height=config["model"]["input_size"],
                             mean=tuple(config["model"].get("input_mean", (0, 0, 0))),
                             std=tuple(config["model"].get("input_std", (1, 1, 1)))),
        output=ModelOutputSpec(type="impact_heatmap_offset", stride=2),
        postprocess=ModelPostprocessSpec(**config["postprocess"]),
    )
    (root / "model_metadata.json").write_text(json.dumps({**metadata.to_dict(),
        "training_dataset_sha256": dataset_hash, "git_commit": commit}, indent=2) + "\n", encoding="utf-8")
    best_loss = float("inf")
    history = []
    with (root / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["epoch", "train_loss", "val_loss", "precision", "recall", "f1"])
        writer.writeheader()
        for epoch in range(1, int(config["training"]["epochs"]) + 1):
            model.train()
            total = 0.0
            for batch in loader_train:
                optimizer.zero_grad(set_to_none=True)
                prediction = model(batch["image"].to(device))
                target = tuple(batch[key].to(device) for key in ("heatmap", "offsets", "positive"))
                loss, _ = impact_loss(prediction, target)
                loss.backward()
                optimizer.step()
                total += float(loss.detach())
            train_loss = total / len(loader_train)
            val_loss, metrics, visualization = _validation(model, loader_val, torch.device(device), config)
            record = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, **metrics}
            history.append(record)
            writer.writerow({key: record[key] for key in writer.fieldnames})
            stream.flush()
            torch.save(_checkpoint(model, config, epoch), root / "last_model.pt")
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(_checkpoint(model, config, epoch), root / "best_model.pt")
            if visualization is not None:
                _visualize(visualization, root / "visualizations" / f"epoch_{epoch:03d}.png", config)
            (root / "metrics.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    return root


def evaluate_checkpoint(config_path: str | Path, checkpoint: str | Path, metadata: str | Path,
                        *, split: str, device: str = "cpu") -> dict:
    if split not in ("val", "test", "mobile_real_test"):
        raise ValueError("independent evaluation requires val, test or mobile_real_test")
    config = _load_configuration(Path(config_path))
    _verify_dataset(config)
    dataset = _prepare(config, split)
    loader = _loader(dataset, int(config["training"]["batch_size"]), False)
    model, model_metadata = load_detector(Path(checkpoint), Path(metadata), device=device)
    if (model_metadata.input.width != config["model"]["input_size"]
            or tuple(model_metadata.input.mean) != tuple(config["model"].get("input_mean", (0, 0, 0)))
            or tuple(model_metadata.input.std) != tuple(config["model"].get("input_std", (1, 1, 1)))):
        raise ValueError("evaluation input config differs from metadata")
    # Evaluation uses the checkpoint's published postprocessing, not a tuned CLI threshold.
    override = {**config, "postprocess": model_metadata.postprocess.to_dict()}
    _, metrics, _ = _validation(model, loader, torch.device(device), override)
    result = {"split": split, "faces": len(dataset), "metrics": metrics}
    if split == "mobile_real_test":
        result["quality_gate"] = mobile_quality_gate(
            metrics, unique_images=len({item.source_image_sha256 for item in dataset.records})
        )
    return result


def mobile_quality_gate(metrics: dict, *, unique_images: int, min_images: int = 300) -> dict:
    """Fail closed when independent mobile evidence is absent or insufficient."""
    blocked = []
    if unique_images < min_images:
        blocked.append(f"verified mobile images {unique_images} < {min_images}")
    if metrics["true_positives"] + metrics["false_negatives"] == 0:
        blocked.append("no annotated real impact points")
    if metrics["empty_faces"] == 0:
        blocked.append("no verified empty target faces")
    if metrics["score_eligible_count"] == 0:
        blocked.append("no score-eligible, non-line-call points")
    for name in ("localization_median_ring", "localization_p95_ring",
                 "exact_score_accuracy_non_line_call", "false_positives_per_empty_face"):
        if metrics[name] is None:
            blocked.append(f"missing {name}")
    if blocked:
        return {"status": "BLOCKED", "reasons": blocked}
    thresholds = {
        "precision": (0.97, ">="), "recall": (0.97, ">="),
        "localization_median_ring": (0.15, "<="),
        "localization_p95_ring": (0.35, "<="),
        "exact_score_accuracy_non_line_call": (0.98, ">="),
    }
    failures = [
        f"{name}={metrics[name]} does not satisfy {op} {value}"
        for name, (value, op) in thresholds.items()
        if (metrics[name] < value if op == ">=" else metrics[name] > value)
    ]
    # On verified empty targets, even one detected arrow is a false accept.
    if metrics["false_positives_per_empty_face"] != 0:
        failures.append("false positives on verified empty targets")
    return {"status": "PASS" if not failures else "FAIL", "reasons": failures}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("runs"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--evaluate", type=Path, help="Existing checkpoint, never train")
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--split", choices=["val", "test", "mobile_real_test"], default="mobile_real_test")
    args = parser.parse_args()
    if args.evaluate:
        if args.metadata is None:
            parser.error("--metadata is required with --evaluate")
        result = evaluate_checkpoint(args.config, args.evaluate, args.metadata,
                                     split=args.split, device=args.device)
        print(json.dumps(result, indent=2))
        if args.split == "mobile_real_test":
            # A completed evaluation process is not a passed acceptance gate.
            # Distinct exit codes prevent CI from silently accepting missing GT.
            status = result["quality_gate"]["status"]
            if status not in ("PASS", "FAIL", "BLOCKED"):
                raise ValueError(f"invalid mobile quality gate status: {status}")
            return {"PASS": 0, "FAIL": 1, "BLOCKED": 2}[status]
    else:
        if args.metadata is not None:
            parser.error("--metadata is only used with --evaluate")
        print(train(args.config, output_root=args.output_root, device=args.device))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
