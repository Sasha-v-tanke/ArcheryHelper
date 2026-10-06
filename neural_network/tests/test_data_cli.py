import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from neural_network.archery_ml.contracts import load_manifest
from neural_network.archery_ml.data.cli import main


class DatasetCliTest(unittest.TestCase):
    def test_full_local_pipeline(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dir = root / "source"
            source_dir.mkdir()
            image = source_dir / "sample.jpg"
            Image.new("RGB", (32, 32), (128, 128, 128)).save(image)
            image.with_suffix(".json").write_text(
                json.dumps({"shots": [{"r_norm": 0.25, "theta_deg": 0.0}]}),
                encoding="utf-8",
            )
            registry = root / "registry.json"
            registry.write_text(
                json.dumps(
                    {
                        "registry_version": 1,
                        "sources": [
                            {
                                "id": "local",
                                "importer": "legacy",
                                "url": str(source_dir),
                                "version": "1",
                                "license": "USER_OWNED",
                                "author": "local",
                                "allowed_tasks": ["impact_detection"],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            manifest = root / "manifest.json"
            workspace = root / "workspace"

            self.assertEqual(
                0,
                main(
                    [
                        "import",
                        "--source",
                        "local",
                        "--registry",
                        str(registry),
                        "--workspace",
                        str(workspace),
                        "--manifest",
                        str(manifest),
                    ]
                ),
            )
            self.assertEqual(
                0,
                main(["validate", "--registry", str(registry), "--manifest", str(manifest)]),
            )
            self.assertEqual(0, main(["deduplicate", "--manifest", str(manifest)]))
            self.assertEqual(0, main(["split", "--manifest", str(manifest), "--seed", "7"]))
            self.assertEqual(0, main(["report", "--manifest", str(manifest)]))

            samples = load_manifest(manifest)
            self.assertEqual(1, len(samples))
            self.assertEqual("local", samples[0].source_id)
            self.assertEqual(1, len(samples[0].annotations))


if __name__ == "__main__":
    unittest.main()
