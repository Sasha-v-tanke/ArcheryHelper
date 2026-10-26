"""Run a policy matrix once, preserve reports and produce an explicit closure status."""
import argparse
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path

from neural_network.archery_ml.data.snapshot import load_snapshot
from neural_network.archery_ml.geometry.cli import main as evaluate_main, _git_state
from neural_network.archery_ml.geometry.validation import aggregate_reports, configuration_key


def run(manifest, policy_path, output, *, snapshot=None, final=False, board_map=None, selections=None):
    output.mkdir(parents=True, exist_ok=False)
    policy = json.loads(policy_path.read_text())
    algorithm, dirty = _git_state()
    blocked = []
    if not manifest.is_file():
        blocked.append("EXTERNAL CAPTURE REQUIRED: independent mobile data unavailable")
    if final:
        if policy.get("status") != "frozen":
            blocked.append("owner-approved frozen policy and development evidence required")
        if not policy.get("require_physical_orientation"):
            blocked.append("Stage 3 closure requires physical orientation")
        if dirty or not algorithm:
            blocked.append("clean frozen algorithm commit required")
        if snapshot is None or not snapshot.is_file():
            blocked.append("immutable GT dataset snapshot required")
        elif manifest.is_file() and load_snapshot(snapshot).manifest_sha256 != hashlib.sha256(manifest.read_bytes()).hexdigest():
            blocked.append("snapshot/manifest checksum mismatch")
    if blocked:
        status = {"status":"BLOCKED", "reasons":blocked, "algorithm_sha":algorithm, "final_holdout_executed":False}
        (output / "status.json").write_text(json.dumps(status,indent=2)+"\n")
        return status
    reports, errors = [], []
    for config in policy["required_configurations"]:
        key = configuration_key(config)
        default_name = "single" if config["format"] == "SINGLE" else config["triple_layout"].lower()
        board = Path((board_map or {}).get(key,f"dataset_tools/boards/{default_name}-v1.json"))
        payload = json.loads(board.read_text())
        diameter = 2 * payload["faces"][0]["radius_mm"]
        path = output / (key.replace(":","-") + ".json")
        args = ["--manifest",str(manifest),"--split","mobile_real_test" if final else "mobile_dev",
            "--format",config["format"],"--minimum-scoring-zone",str(config["minimum_scoring_zone"]),
            "--ten-ring-mode",config["ten_ring_mode"],"--face-diameter-mm",str(int(diameter)),
            "--board-template",str(board),"--policy",str(policy_path),"--output",str(path)]
        args += ["--selection-mode","selected"]
        if selections is not None:
            args += ["--selections",str(selections)]
        if config["triple_layout"]:
            args += ["--triple-layout",config["triple_layout"]]
        try:
            # Partial reports cannot approve the full matrix individually. Only the
            # consolidated decision below is authoritative; preserve their FAILs.
            with redirect_stdout(io.StringIO()):
                evaluate_main(args)
            reports.append(json.loads(path.read_text()))
        except (ValueError, OSError) as error:
            errors.append({"configuration":key,"error":str(error)})
    result = aggregate_reports(reports,policy)
    if errors:
        result["validation"]["status"] = "FAIL"
        result["validation"]["reasons"].extend(f"{e['configuration']}: {e['error']}" for e in errors)
    result["stage3_status"] = result["validation"]["status"] if final else "BLOCKED"
    result["closure_reasons"] = [] if final else ["development evidence only; independent real holdout not executed"]
    result["final_holdout_executed"] = final and bool(reports)
    (output / "consolidated.json").write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n")
    status = {"status":result["stage3_status"], "validation_status":result["validation"]["status"],
              "algorithm_sha":algorithm,"final_holdout_executed":result["final_holdout_executed"],
              "reasons":result["closure_reasons"] + result["validation"]["reasons"],
              "configuration_reports":len(reports),"missing_or_invalid_configurations":errors}
    (output / "status.json").write_text(json.dumps(status,indent=2,sort_keys=True)+"\n")
    return status


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest",type=Path,required=True)
    parser.add_argument("--policy",type=Path,default=Path("dataset_tools/stage3_geometry_policy.json"))
    parser.add_argument("--snapshot",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--board-map",type=Path)
    parser.add_argument("--selections",type=Path)
    parser.add_argument("--final",action="store_true")
    args = parser.parse_args()
    status = run(args.manifest,args.policy,args.output,snapshot=args.snapshot,final=args.final,
                 board_map=None if args.board_map is None else json.loads(args.board_map.read_text()),selections=args.selections)
    print(json.dumps(status,indent=2,sort_keys=True))
    raise SystemExit(0 if status["status"] == "PASS" else 1)
