"""Desktop latency and process peak RSS at fixed still-capture resolutions."""

import argparse
import json
import platform
import resource
import subprocess
import sys
import time

import cv2

from neural_network.archery_ml.geometry.benchmark import summary
from neural_network.archery_ml.geometry.detector import TargetGeometryDetector
from neural_network.archery_ml.geometry.synthetic import render_scene


def measure(width, height, method, repetitions=5):
    # Generation is outside timed detection. Peak RSS explicitly includes it.
    scene = render_scene("arrows", size=800, perspective=0.15)
    image = cv2.resize(scene.image, (width, height), interpolation=cv2.INTER_LINEAR)
    detector = TargetGeometryDetector(method=method)
    detector.detect(image, scene.config)
    elapsed = []
    for _ in range(repetitions):
        start = time.perf_counter()
        geometry = detector.detect(image, scene.config)
        elapsed.append((time.perf_counter() - start) * 1000)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {
        "width": width,
        "height": height,
        "method": method,
        "warmup": 1,
        "repetitions": repetitions,
        "latency_ms": summary(elapsed),
        "peak_process_rss_bytes": peak * (1 if platform.system() == "Darwin" else 1024),
        "faces": len(geometry.faces),
        "scope": "desktop detection; RSS includes interpreter, fixture and image; no Android measurement",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", nargs=3)
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    if args.worker:
        width, height, method = args.worker
        print(json.dumps(measure(int(width), int(height), method)))
        return
    results = []
    for width, height in ((800, 800), (1600, 1200), (3000, 4000)):
        for method in ("legacy", "robust"):
            data = subprocess.check_output(
                [
                    sys.executable,
                    "-m",
                    "neural_network.archery_ml.geometry.performance",
                    "--worker",
                    str(width),
                    str(height),
                    method,
                ],
                text=True,
            )
            results.append(json.loads(data))
    payload = {"platform": platform.platform(), "results": results}
    text = json.dumps(payload, indent=2)
    if args.output:
        from pathlib import Path

        Path(args.output).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
