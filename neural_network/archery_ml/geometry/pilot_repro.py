"""Versioned, content-addressed inventory of Stage 3 pilot output artifacts.

Only the mutable workflow state and the manifest itself are excluded. A saved
geometry timing field is normalized; all other data and image bytes are sealed.
"""
import hashlib
import json
from pathlib import Path


SCHEMA_VERSION = 1
EXCLUDED = frozenset({"run-manifest.json", "reproducibility.json"})
VOLATILE_GEOMETRY_FIELDS = frozenset({"processing_seconds"})


def _stable_geometry(value):
    if isinstance(value, dict):
        return {key: _stable_geometry(item) for key, item in value.items()
                if key not in VOLATILE_GEOMETRY_FIELDS}
    if isinstance(value, list):
        return [_stable_geometry(item) for item in value]
    return value


def _content(path, relative):
    if relative == "geometry-results.json":
        value = _stable_geometry(json.loads(path.read_text()))
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, allow_nan=False).encode("utf-8")
    return path.read_bytes()


def _kind(path):
    return {".json": "json", ".png": "image/png", ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg", ".html": "html", ".md": "markdown"}.get(
                path.suffix.lower(), "file")


def artifact_inventory(output):
    """Return a portable inventory. Reject symlinks rather than following them."""
    output = Path(output)
    files = []
    for path in sorted(output.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"symlink in pilot output: {path.relative_to(output)}")
        if not path.is_file():
            continue
        relative = path.relative_to(output).as_posix()
        if relative in EXCLUDED:
            continue
        files.append({"path": relative, "kind": _kind(path),
                      "sha256": hashlib.sha256(_content(path, relative)).hexdigest()})
    return {"version": SCHEMA_VERSION, "files": files}


def require_manifest(record):
    """Legacy or malformed manifests cannot authorize VERIFIED."""
    manifest = record.get("artifact_manifest")
    if not isinstance(manifest, dict) or manifest.get("version") != SCHEMA_VERSION:
        raise ValueError("legacy or unsupported artifact manifest; rerun normal batch before replay")
    rows = manifest.get("files")
    if not isinstance(rows, list) or any(
        not isinstance(row, dict)
        or not isinstance(row.get("path"), str)
        or not isinstance(row.get("kind"), str)
        or not isinstance(row.get("sha256"), str)
        or not row["path"] or Path(row["path"]).is_absolute()
        or ".." in Path(row["path"]).parts
        for row in rows
    ):
        raise ValueError("invalid artifact manifest; rerun normal batch before replay")
    paths = [row["path"] for row in rows]
    if len(paths) != len(set(paths)) or any(path in EXCLUDED for path in paths):
        raise ValueError("invalid artifact manifest paths; rerun normal batch before replay")
    return manifest


def artifact_differences(expected, actual, scope):
    previous = {item["path"]: item for item in expected["files"]}
    current = {item["path"]: item for item in actual["files"]}
    differences = []
    for path in sorted(previous.keys() | current.keys()):
        if path not in current:
            reason = "missing"
        elif path not in previous:
            reason = "unexpected"
        elif previous[path]["kind"] != current[path]["kind"]:
            reason = "type_changed"
        elif previous[path]["sha256"] != current[path]["sha256"]:
            reason = "content_changed"
        else:
            continue
        differences.append({"scope": scope, "path": path, "difference": reason})
    return differences
