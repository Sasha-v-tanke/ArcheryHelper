import hashlib
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from neural_network.archery_ml.data.download import materialize_source
from neural_network.archery_ml.data.registry import DatasetRegistry, DatasetSource


class DatasetRegistryTest(unittest.TestCase):
    def test_registry_loads_complete_source_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "registry.json"
            path.write_text(
                json.dumps(
                    {
                        "registry_version": 1,
                        "sources": [
                            {
                                "id": "local",
                                "importer": "legacy",
                                "url": "data/local",
                                "version": "1",
                                "license": "CC-BY-4.0",
                                "author": "author",
                                "allowed_tasks": ["impact_detection"],
                                "class_mapping": {},
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            registry = DatasetRegistry.load(path)

            self.assertEqual("local", registry.get("local").id)
            self.assertEqual("CC-BY-4.0", registry.get("local").license)

    def test_remote_source_requires_checksum(self):
        source = {
            "id": "remote",
            "importer": "coco",
            "url": "https://example.com/dataset.zip",
            "version": "1",
            "license": "CC-BY-4.0",
            "author": "author",
            "allowed_tasks": ["impact_detection"],
        }

        with self.assertRaisesRegex(ValueError, "checksum_sha256"):
            DatasetSource.from_dict(source)

    def test_local_archive_is_verified_and_materialized(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "dataset.zip"
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr("images/sample.txt", "content")
            checksum = hashlib.sha256(archive.read_bytes()).hexdigest()
            source = DatasetSource.from_dict(
                {
                    "id": "archive",
                    "importer": "legacy",
                    "url": str(archive),
                    "version": "1",
                    "license": "CC-BY-4.0",
                    "author": "author",
                    "checksum_sha256": checksum,
                    "allowed_tasks": ["impact_detection"],
                }
            )

            materialized = materialize_source(source, root / "workspace")

            self.assertEqual("content", (materialized / "images/sample.txt").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
