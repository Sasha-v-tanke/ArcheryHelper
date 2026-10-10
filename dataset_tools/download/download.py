from pathlib import Path

from roboflow import Roboflow

from dataset_tools.download.config import *
from dataset_tools.download.secrets import ROBOFLOW_TOKEN

DATASET_ROOT = Path(__file__).resolve().parents[2] / "data" / "sources"


def download_archery_scoring(version=4):
    destination = DATASET_ROOT / "archery_scoring_keypoints"

    rf = Roboflow(api_key=ROBOFLOW_TOKEN)
    project = rf.workspace("archery-scoring").project("archery-scoring-keypoints")
    dataset_version = project.version(version)
    dataset = dataset_version.download("coco",
                                       location=str(destination / f"v{version}"),
                                       overwrite=False)


def download_target_10_ring(version=9):
    destination = DATASET_ROOT / "archery_aiscore_ounar"

    rf = Roboflow(api_key=ROBOFLOW_TOKEN)
    project = rf.workspace("aiscore-ounar").project("target-10-ring-instance")
    dataset_version = project.version(version)
    dataset = dataset_version.download("coco",
                                       location=str(destination / f"v{version}"),
                                       overwrite=False)


def download_all():
    if DOWNLOAD_ARCHERY_SCORING:
        download_archery_scoring()
    if DOWNLOAD_TARGET_10_RING:
        download_target_10_ring()


if __name__ == "__main__":
    download_all()
