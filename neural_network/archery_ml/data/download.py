from __future__ import annotations

import hashlib
import shutil
import stat
import tarfile
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

from neural_network.archery_ml.data.registry import DatasetSource


def materialize_source(source: DatasetSource, workspace: str | Path) -> Path:
    local = _local_path(source.url)
    if local is not None and local.is_dir():
        return local.resolve()

    workspace_path = Path(workspace)
    cache_dir = workspace_path / ".cache"
    source_dir = workspace_path / "sources" / source.id / source.version
    cache_dir.mkdir(parents=True, exist_ok=True)

    if local is not None:
        archive_path = local.resolve()
        if not archive_path.is_file():
            raise FileNotFoundError(archive_path)
    else:
        archive_path = cache_dir / _download_name(source)
        if not archive_path.exists():
            _download(source.url, archive_path)

    _verify_checksum(archive_path, source.checksum_sha256)

    marker = source_dir / ".source.sha256"
    digest = _sha256(archive_path)
    if marker.exists() and marker.read_text(encoding="utf-8").strip() == digest:
        return source_dir.resolve()

    if source_dir.exists():
        shutil.rmtree(source_dir)
    source_dir.mkdir(parents=True)
    _extract_or_copy(archive_path, source_dir)
    marker.write_text(digest + "\n", encoding="utf-8")
    return source_dir.resolve()


def _local_path(url: str) -> Path | None:
    if url.startswith("file:"):
        parsed = urllib.parse.urlparse(url)
        raw = urllib.parse.unquote(parsed.path or parsed.netloc)
        return Path(raw)
    if "://" not in url:
        return Path(url)
    return None


def _download_name(source: DatasetSource) -> str:
    parsed = urllib.parse.urlparse(source.url)
    name = Path(parsed.path).name
    return name or f"{source.id}-{source.version}.archive"


def _download(url: str, output: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "ArcheryHelper-dataset-pipeline/1"})
    temp = output.with_suffix(output.suffix + ".part")
    with urllib.request.urlopen(request, timeout=60) as response, temp.open("wb") as stream:
        shutil.copyfileobj(response, stream)
    temp.replace(output)


def _verify_checksum(path: Path, expected: str | None) -> None:
    if expected is None:
        return
    actual = _sha256(path)
    if actual.lower() != expected.lower():
        raise ValueError(f"checksum mismatch for {path}: expected {expected}, got {actual}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _extract_or_copy(archive: Path, destination: Path) -> None:
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as handle:
            for info in handle.infolist():
                _validate_member(destination, info.filename)
                mode = info.external_attr >> 16
                if stat.S_ISLNK(mode):
                    raise ValueError(f"archive contains symlink: {info.filename}")
            handle.extractall(destination)
        return

    if tarfile.is_tarfile(archive):
        with tarfile.open(archive) as handle:
            members = handle.getmembers()
            for member in members:
                _validate_member(destination, member.name)
                if not (member.isfile() or member.isdir()):
                    raise ValueError(f"archive contains unsupported member: {member.name}")
            handle.extractall(destination, members=members, filter="data")
        return

    shutil.copy2(archive, destination / archive.name)


def _validate_member(destination: Path, member_name: str) -> None:
    target = (destination / member_name).resolve()
    destination_resolved = destination.resolve()
    if target != destination_resolved and destination_resolved not in target.parents:
        raise ValueError(f"archive path escapes destination: {member_name}")
