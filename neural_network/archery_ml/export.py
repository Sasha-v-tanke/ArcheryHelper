"""Fail-closed Stage 5 ImpactPointNet export to ExecuTorch/XNNPACK."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from neural_network.archery_ml.contracts import ModelMetadata
from neural_network.archery_ml.inference import load_detector
from neural_network.archery_ml.stage5_parity import (
    LOGIT_MAX_ABS, OFFSET_MAX_ABS, POINT_CONFIDENCE_ABS, POINT_NORM_DISTANCE,
    canonical_tensor, compare_outputs, read_golden_manifest, sha256_file,
)

PTE_LIMIT_BYTES = 30_000_000


def _compile_xnnpack(model: torch.nn.Module, example: torch.Tensor) -> bytes:
    # No synthetic fallback: absence of the real compiler is a hard error.
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
    from executorch.exir import to_edge_transform_and_lower

    with torch.inference_mode():
        exported = torch.export.export(model.eval(), (example,))
        program = to_edge_transform_and_lower(
            exported, partitioner=[XnnpackPartitioner()]
        ).to_executorch()
    pte = bytes(program.buffer)
    if not pte or b"XnnpackBackend" not in pte:
        raise ValueError("export did not contain an XNNPACK delegate")
    if len(pte) > PTE_LIMIT_BYTES:
        raise ValueError("FP32 PTE exceeds the 30 MB production asset limit")
    return pte


def _runtime_forward(pte: Path):
    from executorch.runtime import Runtime

    runtime = Runtime.get()
    program = runtime.load_program(pte)
    if "forward" not in program.method_names:
        raise ValueError("ExecuTorch PTE lacks the forward method")
    method = program.load_method("forward")

    def run(input_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = method.execute((input_tensor,))
        if not isinstance(outputs, (list, tuple)) or len(outputs) != 2:
            raise ValueError("ExecuTorch must return exactly logits and offsets")
        return outputs[0], outputs[1]

    return run


def _write_json(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n",
                    encoding="utf-8")


def export_model(
    checkpoint: Path,
    metadata_path: Path,
    golden_manifest: Path,
    output_directory: Path,
) -> Path:
    """Create the release directory only if XNNPACK compilation and parity pass.

    Golden manifest images are hash-bound, reviewed Stage 4 validation/test
    crops, never training images or the mobile acceptance holdout.
    """
    checkpoint = Path(checkpoint)
    metadata_path = Path(metadata_path)
    golden_manifest = Path(golden_manifest)
    output_directory = Path(output_directory).absolute()
    if output_directory.exists():
        raise FileExistsError(f"release output already exists: {output_directory}")
    if not checkpoint.is_file() or not metadata_path.is_file():
        raise FileNotFoundError("trained checkpoint and Stage 4 metadata are required")

    model, metadata = load_detector(checkpoint, metadata_path, device="cpu")
    if (metadata.contract_version != 2 or metadata.input.color_space != "RGB"
            or metadata.input.width != 512 or metadata.input.height != 512
            or metadata.output.type != "impact_heatmap_offset" or metadata.output.stride != 2):
        raise ValueError("unsupported Stage 4 export input/output contract")
    original_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if "executorch" in original_metadata:
        raise ValueError("source metadata must be the unmodified Stage 4 training metadata")
    samples = read_golden_manifest(golden_manifest, metadata)
    golden_document = json.loads(golden_manifest.read_text(encoding="utf-8"))
    declared_checkpoint = golden_document.get("source_checkpoint_sha256")
    declared_manifest = golden_document.get("source_manifest_sha256")
    if declared_checkpoint is not None and declared_checkpoint != sha256_file(checkpoint):
        raise ValueError("golden set was prepared for a different checkpoint")
    if (declared_manifest is not None
            and declared_manifest != original_metadata.get("training_dataset_sha256")):
        raise ValueError("golden dataset does not match the Stage 4 checkpoint")
    if any(sample.image == output_directory or output_directory in sample.image.parents for sample in samples):
        raise ValueError("golden inputs must not be inside release output")

    output_directory.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".stage5-", dir=output_directory.parent))
    try:
        with Image.open(samples[0].image) as image:
            example = canonical_tensor(image, metadata)
        pte = _compile_xnnpack(model, example)
        pte_path = staging / "impact_model.pte"
        pte_path.write_bytes(pte)
        run = _runtime_forward(pte_path)

        golden_records = []
        measured = []
        with torch.inference_mode():
            for sample in samples:
                # Recheck source integrity during export, not only at manifest parsing.
                if sha256_file(sample.image) != sample.crop_sha256:
                    raise ValueError(f"golden image changed during export: {sample.sample_id}")
                with Image.open(sample.image) as image:
                    tensor = canonical_tensor(image, metadata)
                eager_outputs = tuple(value.detach().cpu().contiguous() for value in model(tensor))
                runtime_outputs = run(tensor.clone())
                outcome = compare_outputs(eager_outputs, runtime_outputs, metadata, sample.face_index)
                measured.append(outcome)

                image_rel = Path("golden/images") / (sample.sample_id + ".png")
                expected_rel = Path("golden/expected") / (sample.sample_id + ".npz")
                (staging / image_rel).parent.mkdir(parents=True, exist_ok=True)
                (staging / expected_rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(sample.image, staging / image_rel)
                np.savez_compressed(staging / expected_rel,
                                    heatmap_logits=eager_outputs[0].numpy(),
                                    offsets=eager_outputs[1].numpy())
                if sha256_file(staging / image_rel) != sample.crop_sha256:
                    raise ValueError("copied golden image checksum mismatch")
                golden_records.append({
                    "sample_id": sample.sample_id,
                    "split": sample.split,
                    "original_sha256": sample.original_sha256,
                    "face_index": sample.face_index,
                    "image_path": image_rel.as_posix(),
                    "image_sha256": sample.crop_sha256,
                    "expected_tensors_path": expected_rel.as_posix(),
                    "expected_tensors_sha256": sha256_file(staging / expected_rel),
                    "expected_points": outcome["expected_points"],
                })

        maxima = {
            metric: max(result[metric] for result in measured)
            for metric in ("logits_max_abs", "offsets_max_abs",
                           "points_xy_max_norm", "points_confidence_max_abs")
        }
        _write_json(staging / "golden/index.json", {
            "schema_version": 1, "samples": golden_records,
        })
        parity = {
            "status": "PASS",
            "backend": "XNNPACK",
            "golden_count": len(samples),
            "total_expected_points": sum(result["point_count"] for result in measured),
            "max_errors": maxima,
            "limits": {
                "logits_max_abs": LOGIT_MAX_ABS,
                "offsets_max_abs": OFFSET_MAX_ABS,
                "points_xy_max_norm": POINT_NORM_DISTANCE,
                "points_confidence_max_abs": POINT_CONFIDENCE_ABS,
            },
            "golden_index_sha256": sha256_file(staging / "golden/index.json"),
            "golden_source_manifest_sha256": declared_manifest,
            "golden_source_snapshot_sha256": golden_document.get("source_snapshot_sha256"),
            "checkpoint_sha256": sha256_file(checkpoint),
            "pte_sha256": sha256_file(pte_path),
            "torch_version": torch.__version__,
            "executorch_version": importlib.metadata.version("executorch"),
        }
        _write_json(staging / "parity.json", parity)
        release_metadata = {**original_metadata, "executorch": {
            "format": "pte",
            "backend": "XNNPACK",
            "artifact": "impact_model.pte",
            "sha256": parity["pte_sha256"],
            "checkpoint_sha256": parity["checkpoint_sha256"],
            "golden_index_sha256": parity["golden_index_sha256"],
            "parity_report_sha256": sha256_file(staging / "parity.json"),
            "parity_status": "PASS",
            "torch_version": str(torch.__version__),
            "executorch_version": parity["executorch_version"],
        }}
        _write_json(staging / "model_metadata.json", release_metadata)
        listed = sorted(path for path in staging.rglob("*") if path.is_file())
        hashes = [
            f"{sha256_file(file)}  {file.relative_to(staging).as_posix()}"
            for file in listed
        ]
        (staging / "sha256.txt").write_text("\n".join(hashes) + "\n", encoding="utf-8")

        # Same-filesystem rename exposes no partial release; never overwrite.
        if output_directory.exists():
            raise FileExistsError("release output appeared while exporting")
        staging.rename(output_directory)
        return output_directory
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def verify_release(directory: Path) -> dict:
    """Verify every package byte and all cross-file SHA256 references."""
    directory = Path(directory).resolve(strict=True)
    entries = (directory / "sha256.txt").read_text(encoding="utf-8").splitlines()
    actual_files = {p.relative_to(directory).as_posix() for p in directory.rglob("*") if p.is_file()}
    if not entries:
        raise ValueError("empty release checksum manifest")
    indexed = set()
    for entry in entries:
        if len(entry) < 67 or entry[64:66] != "  ":
            raise ValueError("invalid sha256.txt entry")
        digest, relative = entry[:64], entry[66:]
        if (not relative or relative.startswith("/") or ".." in Path(relative).parts
                or relative == "sha256.txt" or relative in indexed):
            raise ValueError("unsafe or duplicate sha256.txt path")
        indexed.add(relative)
        path = (directory / relative).resolve(strict=True)
        if not path.is_relative_to(directory) or sha256_file(path) != digest:
            raise ValueError(f"release file SHA256 mismatch: {relative}")
    if indexed != actual_files - {"sha256.txt"}:
        raise ValueError("release files missing from or added to sha256.txt")
    meta = json.loads((directory / "model_metadata.json").read_text(encoding="utf-8"))
    parity = json.loads((directory / "parity.json").read_text(encoding="utf-8"))
    artifact = meta["executorch"]
    if (artifact["parity_status"] != "PASS" or parity["status"] != "PASS"
            or artifact["backend"] != "XNNPACK" or artifact["artifact"] != "impact_model.pte"
            or artifact["sha256"] != sha256_file(directory / "impact_model.pte")
            or artifact["golden_index_sha256"] != sha256_file(directory / "golden/index.json")
            or artifact["parity_report_sha256"] != sha256_file(directory / "parity.json")
            or artifact["sha256"] != parity["pte_sha256"]
            or artifact["checkpoint_sha256"] != parity["checkpoint_sha256"]):
        raise ValueError("inconsistent Stage 5 metadata or parity references")
    return parity


def replay_release(directory: Path) -> dict:
    """Re-run the *shipped* PTE on the shipped golden images and reference tensors.

    This is independent of the training checkpoint and prevents a package from
    being called parity-verified only because its metadata says 'PASS'.
    """
    directory = Path(directory).resolve(strict=True)
    original = verify_release(directory)
    metadata = ModelMetadata.from_dict(
        json.loads((directory / "model_metadata.json").read_text(encoding="utf-8"))
    )
    golden = json.loads((directory / "golden/index.json").read_text(encoding="utf-8"))
    entries = golden.get("samples")
    if (golden.get("schema_version") != 1 or not isinstance(entries, list)
            or len(entries) != original["golden_count"]):
        raise ValueError("golden index sample count differs from parity report")
    forward = _runtime_forward(directory / "impact_model.pte")
    maxima = {"logits_max_abs": 0., "offsets_max_abs": 0.,
              "points_xy_max_norm": 0., "points_confidence_max_abs": 0.}
    total_points = 0
    with torch.inference_mode():
        for entry in entries:
            image = (directory / entry["image_path"]).resolve(strict=True)
            expected_file = (directory / entry["expected_tensors_path"]).resolve(strict=True)
            if not image.is_relative_to(directory) or not expected_file.is_relative_to(directory):
                raise ValueError("golden replay path escapes release directory")
            if (sha256_file(image) != entry["image_sha256"]
                    or sha256_file(expected_file) != entry["expected_tensors_sha256"]):
                raise ValueError("golden replay file SHA256 mismatch")
            with Image.open(image) as pixels:
                input_tensor = canonical_tensor(pixels, metadata)
            with np.load(expected_file, allow_pickle=False) as array:
                expected = (
                    torch.from_numpy(array["heatmap_logits"].copy()),
                    torch.from_numpy(array["offsets"].copy()),
                )
            compared = compare_outputs(expected, forward(input_tensor), metadata, entry["face_index"])
            if compared["expected_points"] != entry["expected_points"]:
                raise ValueError("golden expected points do not match saved reference tensors")
            total_points += compared["point_count"]
            for key in maxima:
                maxima[key] = max(maxima[key], compared[key])
    if total_points != original["total_expected_points"]:
        raise ValueError("golden reference point counts differ from recorded parity")
    return {"status": "PASS", "golden_count": len(entries),
            "total_expected_points": total_points, "max_errors": maxima,
            "pte_sha256": sha256_file(directory / "impact_model.pte")}


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 5 XNNPACK export and independent Python runtime parity")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--golden-manifest", type=Path)
    parser.add_argument("--output-directory", type=Path)
    checks = parser.add_mutually_exclusive_group()
    checks.add_argument("--verify-release", type=Path, help="Verify release byte integrity only")
    checks.add_argument("--replay-release", type=Path, help="Run shipped PTE vs shipped golden tensors")
    arguments = parser.parse_args()
    if arguments.verify_release or arguments.replay_release:
        if any((arguments.checkpoint, arguments.metadata, arguments.golden_manifest, arguments.output_directory)):
            parser.error("verification/replay cannot be combined with export inputs")
        result = (verify_release(arguments.verify_release) if arguments.verify_release
                  else replay_release(arguments.replay_release))
        print(json.dumps(result, indent=2))
    else:
        if not all((arguments.checkpoint, arguments.metadata,
                    arguments.golden_manifest, arguments.output_directory)):
            parser.error("export requires --checkpoint --metadata --golden-manifest --output-directory")
        path = export_model(arguments.checkpoint, arguments.metadata,
                            arguments.golden_manifest, arguments.output_directory)
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
