"""Local human center review. Photos and detector results remain read-only."""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import math
from pathlib import Path
from urllib.parse import urlsplit

from neural_network.archery_ml.geometry.calibration import project
from neural_network.archery_ml.geometry.pilot_inventory import save_json


class ConflictError(ValueError):
    pass


class CenterReviewStore:
    def __init__(self, pilot, output):
        self.pilot, self.output = Path(pilot).resolve(), Path(output).resolve()
        if self.output == self.pilot or self.output.is_relative_to(self.pilot):
            raise ValueError("center reviews must be separate from pilot evidence")
        inventory = json.loads((self.pilot/"inventory.json").read_text())
        detections = json.loads((self.pilot/"detections.json").read_text())
        results = json.loads((self.pilot/"geometry-results.json").read_text())
        self.images, self.paths = {}, {}
        for row in inventory["images"]:
            key = row.get("original_id")
            if row["read_status"] != "OK" or key in self.images:
                continue
            path = (self.pilot/row["rgb_path"]).resolve()
            if not path.is_relative_to(self.pilot):
                raise ValueError("image outside pilot directory")
            geometry = results.get(key,{})
            candidates = {c["id"]:c for c in detections[key]["candidates"]}
            points = []
            fitted = {f["face_index"]:f["center_rgb"] for f in geometry.get("faces",[])}
            for index,cid in enumerate(geometry.get("candidate_ids",[])):
                if cid in candidates:
                    x,y = fitted.get(index,candidates[cid]["center"])
                    points.append({"id":cid,"x":x,"y":y})
            binding = {"original_sha256":key,"image_sha256":row["rgb_sha256"],
                "image_size":row["rgb_size"],"rgb_to_raw":row["rgb_to_raw"],
                "suggested_centers":points}
            stamp = hashlib.sha256(json.dumps(binding,sort_keys=True).encode()).hexdigest()
            self.images[key] = {"id":key,"number":row["number"],"filename":row["filename"],
                "width":row["rgb_size"][0],"height":row["rgb_size"][1],"binding":stamp,
                "image_sha256":row["rgb_sha256"],"rgb_to_raw":row["rgb_to_raw"],
                "orientation_verified":row.get("exif_orientation_valid",False),
                "ownership_confirmed":geometry.get("ownership_confirmed",False),
                "suggested_centers":points,"image_url":f"/image/{key}"}
            self.paths[key] = path
        self.output.mkdir(parents=True,exist_ok=True)
        self.file = self.output/"center-reviews.json"

    def read(self):
        if not self.file.exists():
            return {"schema_version":1,"coordinate_space":"upright_rgb_px",
                "dataset_role":"development_center_review","independent_gt":False,"images":{}}
        return json.loads(self.file.read_text())

    def state(self):
        reviews = self.read()["images"]
        proposal_file = self.output/"center-proposals.json"
        proposals = json.loads(proposal_file.read_text()) if proposal_file.exists() else {}
        images = []
        for key,row in self.images.items():
            value = {**row,"review":reviews.get(key),
                "stale":key in reviews and reviews[key].get("binding")!=row["binding"]}
            # A human draft or confirmation always takes precedence. Keep the
            # original image binding so new proposals cannot invalidate labels.
            proposal = proposals.get("images",{}).get(key,{})
            points = proposal.get("points",[])
            if (key not in reviews and proposal.get("image_sha256")==row["image_sha256"]
                    and proposal.get("original_sha256")==key and proposals.get("revision")
                    and isinstance(points,list) and points
                    and all(isinstance(p,dict) and isinstance(p.get("id"),str)
                        and type(p.get("x")) in (int,float) and type(p.get("y")) in (int,float)
                        and math.isfinite(p["x"]) and math.isfinite(p["y"])
                        and 0<=p["x"]<row["width"] and 0<=p["y"]<row["height"] for p in points)
                    and len({p["id"] for p in points})==len(points)
                    and {p["id"] for p in points}=={p["id"] for p in row["suggested_centers"]}):
                value.update(suggested_centers=points,proposal_revision=proposals["revision"],
                    proposal_quality_status=proposal.get("quality_status"),automatic_proposal=True)
            images.append(value)
        return {"images":images,"save_path":str(self.file)}

    def image_bytes(self, key):
        if key not in self.images:
            raise ValueError("unknown photograph")
        content = self.paths[key].read_bytes()
        if hashlib.sha256(content).hexdigest()!=self.images[key]["image_sha256"]:
            raise ValueError("display image changed since inventory")
        return content

    def save(self, payload):
        key = payload.get("original_id")
        if key not in self.images:
            raise ValueError("unknown photograph")
        row = self.images[key]
        if payload.get("binding") != row["binding"]:
            raise ConflictError("Изображение изменилось. Обновите страницу.")
        data = self.read()
        previous = data["images"].get(key)
        revision = 0 if previous is None else previous["revision"]
        if type(payload.get("revision")) is not int or payload["revision"]!=revision:
            raise ConflictError("Есть более новая правка. Обновите страницу.")
        status, points = payload.get("status"), payload.get("points")
        if status not in {"DRAFT","CONFIRMED","UNCERTAIN"}:
            raise ValueError("invalid review status")
        if not isinstance(points,list) or len(points)>100 or status=="CONFIRMED" and not points:
            raise ValueError("Добавьте хотя бы один центр перед подтверждением.")
        clean, seen = [], set()
        for point in points:
            if not isinstance(point,dict):
                raise ValueError("invalid center")
            identity, x, y = point.get("id"), point.get("x"), point.get("y")
            if (not isinstance(identity,str) or not identity or len(identity)>80 or identity in seen
                    or type(x) not in (int,float) or type(y) not in (int,float)
                    or not math.isfinite(x) or not math.isfinite(y)
                    or not 0<=x<row["width"] or not 0<=y<row["height"]):
                raise ValueError("Центр должен находиться внутри фотографии; ID должны различаться.")
            seen.add(identity)
            item = {"id":identity,"x":float(x),"y":float(y)}
            if row["orientation_verified"]:
                item["original_raw_px"] = project([[x,y]],row["rgb_to_raw"])[0].tolist()
            clean.append(item)
        self.image_bytes(key)
        record = {"original_sha256":key,"image_sha256":row["image_sha256"],"binding":row["binding"],
            "revision":revision+1,"status":status,"points":clean,
            "coordinate_space":"upright_rgb_px","image_size":[row["width"],row["height"]],
            "method":"manual_from_blank" if payload.get("from_blank") is True else "reviewed_detector_proposal",
            "independent_gt":False,"saved_at":datetime.now(timezone.utc).isoformat()}
        suggestion_revision = payload.get("proposal_revision","baseline")
        if not isinstance(suggestion_revision,str) or len(suggestion_revision)>80:
            raise ValueError("invalid proposal revision")
        record["source_proposal_revision"] = suggestion_revision
        if previous is not None:
            with (self.output/"center-review-history.jsonl").open("a",encoding="utf-8") as stream:
                stream.write(json.dumps(previous,ensure_ascii=False,allow_nan=False)+"\n")
        data["images"][key] = record
        save_json(self.file,data)
        return copy.deepcopy(record)


def serve(pilot, output, port=8766):
    store = CenterReviewStore(pilot,output)
    html = Path(__file__).with_suffix(".html").read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def reply(self, status, body, content_type="application/json; charset=utf-8"):
            if not isinstance(body,bytes):
                body = json.dumps(body,ensure_ascii=False,allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type",content_type)
            self.send_header("Content-Length",str(len(body)))
            self.send_header("Cache-Control","no-store")
            self.send_header("X-Content-Type-Options","nosniff")
            self.end_headers()
            self.wfile.write(body)

        def check_origin(self):
            origins = {f"http://127.0.0.1:{self.server.server_port}",f"http://localhost:{self.server.server_port}"}
            if "http://"+self.headers.get("Host","") not in origins:
                raise ValueError("loopback host required")
            if self.headers.get("Origin") is not None and self.headers["Origin"] not in origins:
                raise ValueError("same-origin request required")

        def do_GET(self):
            try:
                self.check_origin()
                path = urlsplit(self.path).path
                if path in {"/","/centers.html"}:
                    self.reply(200,html,"text/html; charset=utf-8")
                elif path=="/api/state":
                    self.reply(200,store.state())
                elif path=="/api/export":
                    self.reply(200,store.read())
                elif path.startswith("/image/"):
                    self.reply(200,store.image_bytes(path.removeprefix("/image/")),"image/png")
                else:
                    self.reply(404,{"error":"not found"})
            except (ValueError,OSError) as error:
                self.reply(400,{"error":str(error)})

        def do_POST(self):
            try:
                self.check_origin()
                length = int(self.headers.get("Content-Length",0))
                if self.path!="/api/centers" or self.headers.get_content_type()!="application/json" or not 0<length<100000:
                    raise ValueError("bounded center-review JSON required")
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload,dict):
                    raise ValueError("expected a JSON object")
                self.reply(200,{"review":store.save(payload)})
            except ConflictError as error:
                self.reply(409,{"error":str(error)})
            except (ValueError,TypeError,KeyError,OSError) as error:
                self.reply(400,{"error":str(error)})

    server = HTTPServer(("127.0.0.1",port),Handler)
    print(f"Center review: http://127.0.0.1:{server.server_port}/ — {len(store.images)} photos",flush=True)
    server.serve_forever()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot",type=Path,default=Path("data/reports/stage3-finalization/pilot"))
    parser.add_argument("--output",type=Path,default=Path("data/reports/center-review-58"))
    parser.add_argument("--port",type=int,default=8766)
    args = parser.parse_args(argv)
    serve(args.pilot,args.output,args.port)


if __name__ == "__main__":
    main()
