"""Local, read-only review desk for Stage 3 to Stage 4 impact ownership."""

from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from PIL import Image

from neural_network.archery_ml.contracts import load_manifest


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_queue(manifest: Path, geometry_results: Path, data_root: Path) -> tuple[list[dict], dict[str, Path]]:
    """Build visual suggestions; none of the records is approved by this step."""
    root = data_root.resolve()
    manifest, geometry_results = manifest.resolve(), geometry_results.resolve()
    if not all(path.is_file() and path.is_relative_to(root) for path in (manifest, geometry_results)):
        raise ValueError("manifest and geometry results must be files inside data root")
    geometries = json.loads(geometry_results.read_text(encoding="utf-8"))
    if not isinstance(geometries, dict):
        raise ValueError("Stage 3 geometry results must be keyed by original SHA256")
    geometry_hash = _sha256(geometry_results)
    assets: dict[str, Path] = {}
    queue: list[dict] = []
    for sample in load_manifest(manifest):
        original = (sample.image_path if sample.image_path.is_absolute()
                    else root.parent / sample.image_path).resolve()
        if not original.is_file() and not sample.image_path.is_absolute():
            original = (manifest.parent / sample.image_path).resolve()
        if not original.is_file() or not original.is_relative_to(root):
            raise ValueError(f"original image unavailable or outside data root: {sample.id}")
        original_hash = _sha256(original)
        with Image.open(original) as image:
            orientation = image.getexif().get(274, 1)
        asset_id = f"image-{len(queue)}"
        assets[asset_id] = original
        geometry = geometries.get(original_hash)
        reasons = []
        if orientation != 1:
            reasons.append("EXIF orientation is not 1; register a coordinate-preserving derivative first")
        if geometry is None:
            reasons.append("No selected Stage 3 geometry for these exact original bytes")
        elif (geometry.get("original_id") != original_hash
              or geometry.get("status") != "READY"
              or geometry.get("quality_status") != "READY"
              or geometry.get("selection_status") != "USER_CONFIRMED"
              or geometry.get("ownership_confirmed") is not True):
            reasons.append("Stage 3 geometry is not confirmed READY for this original")
        faces = []
        if geometry is not None and not reasons:
            face_records = geometry.get("faces", [])
            count = 3 if sample.target_metadata and sample.target_metadata.format == "TRIPLE" else 1
            if (len(face_records) != count
                    or {face.get("face_index") for face in face_records} != set(range(count))
                    or (count == 3 and geometry.get("physical_face_order_verified") is not True)):
                reasons.append("Face identities/order do not match the source target")
            else:
                for face in sorted(face_records, key=lambda item: item["face_index"]):
                    if not isinstance(face.get("crop_path"), str):
                        reasons.append(f"Face {face['face_index']} has no crop path")
                        break
                    crop = (geometry_results.parent / face["crop_path"]).resolve()
                    if not crop.is_file() or not crop.is_relative_to(root):
                        reasons.append(f"Face {face['face_index']} crop is missing or outside data root")
                        break
                    with Image.open(crop) as image:
                        if image.mode != "RGB" or image.size != (512, 512):
                            reasons.append(f"Face {face['face_index']} is not a canonical RGB crop")
                            break
                    crop_id = f"crop-{len(queue)}-{face['face_index']}"
                    assets[crop_id] = crop
                    faces.append({"index": face["face_index"], "url": f"/asset/{crop_id}"})
        raw_labels = [item.to_dict() for item in sample.raw_annotations]
        labels_hash = hashlib.sha256(json.dumps(
            raw_labels, sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode()).hexdigest()
        queue.append({
            "id": sample.id, "split": sample.split, "original_url": f"/asset/{asset_id}",
            "original_sha256": original_hash, "raw_annotations_sha256": labels_hash,
            "raw": raw_labels, "faces": faces, "reasons": reasons,
            "geometry_path": str(geometry_results), "geometry_sha256": geometry_hash,
            "selected_candidate_ids": geometry.get("candidate_ids", []) if geometry else [],
        })
    return queue, assets


PAGE = r'''<!doctype html><html lang="ru"><meta charset="utf-8">
<title>Stage 4 — проверка попаданий</title>
<style>
body{font:16px system-ui;margin:20px auto;max-width:1250px;color:#202531}button,select,input{font:inherit;padding:6px;margin:3px}
.layout{display:grid;grid-template-columns:minmax(0,1fr) 300px;gap:18px}.photo{position:relative;width:100%}
.photo img{display:block;width:100%;height:auto}.pin{position:absolute;transform:translate(-50%,-50%);border:2px solid white;
background:#bf253a;color:white;border-radius:50%;width:27px;height:27px;text-align:center;font:bold 15px/27px system-ui;box-shadow:0 0 3px black}
.faces{display:flex;gap:10px;flex-wrap:wrap}.faces img{width:210px;max-width:100%}.card{border:1px solid #aaa;padding:8px}
.warning{background:#ffe5c6;padding:10px}.ok{background:#d8f4e3;padding:10px}textarea{width:100%;height:90px}
</style>
<h1>Stage 4 — проверка исходных попаданий</h1>
<p>Сверьте точки на исходном фото с выбранными Stage 3 гранями. Пустую грань подтверждайте только после осмотра. Состояние хранится в этом браузере; скачайте draft для передачи независимому проверяющему. Предсказания модели не показываются.</p>
<nav><button id="prev">←</button><select id="samples"></select><button id="next">→</button> <span id="progress"></span></nav>
<div id="status"></div><h2 id="title"></h2><div class="layout"><div><div class="photo" id="photo"></div>
<h3>Canonical faces</h3><div id="faces" class="faces"></div></div><aside>
<label>Разметчик <input id="annotator" autocomplete="off"></label><p><label><input id="owner" type="checkbox"> Подтверждаю принадлежность выбранных граней исходному фото</label></p>
<div id="assignments"></div><div id="empties"></div><hr>
<label>Независимый проверяющий <input id="reviewer" autocomplete="off"></label>
<p><label><input id="checked" type="checkbox"> Проверил исходник, каждую точку, принадлежность и отсутствие попаданий на отмеченных пустых гранях</label></p>
<label>Замечания <textarea id="notes"></textarea></label></aside></div>
<p><button id="draft">Скачать рабочий draft JSON</button><button id="import">Загрузить draft JSON</button><input id="file" type="file" accept="application/json" hidden>
<button id="export">Скачать карту проверенных записей</button></p>
<p>Карта применяется только через <code>python -m archery_ml.prepare --review-map ...</code>: там повторно проверяются байты, хеши, геометрия и назначения. Файл с именами сам по себе не доказывает независимую проверку.</p>
<script>
const queue=__QUEUE__, key='stage4-review:'+__MANIFEST_SHA__, $=id=>document.getElementById(id);
let state={};try{state=JSON.parse(localStorage.getItem(key)||'{}')}catch(_){}let current=0;
function entry(q){return state[q.id]??={annotator:'',reviewer:'',owner:false,checked:false,notes:'',assignments:q.raw.map(()=>''),empties:[]}}
function persist(){localStorage.setItem(key,JSON.stringify(state))}
function save(){let q=queue[current],s=entry(q);s.annotator=$('annotator').value.trim();s.reviewer=$('reviewer').value.trim();s.owner=$('owner').checked;
s.checked=$('checked').checked;s.notes=$('notes').value;s.assignments=[...document.querySelectorAll('.assignment')].map(x=>x.value);
s.empties=[...document.querySelectorAll('.empty')].filter(x=>x.checked).map(x=>Number(x.value));persist();progress()}
function progress(){let ready=queue.filter(q=>valid(q,entry(q))).length;$('progress').textContent=`${ready}/${queue.length} проверено`}
function valid(q,s){if(q.reasons.length||!q.faces.length||!s.owner||!s.checked||!s.annotator||!s.reviewer||s.annotator===s.reviewer)return false;
if(s.assignments.length!==q.raw.length||s.assignments.includes(''))return false;
let counts=new Map(q.faces.map(f=>[f.index,0]));for(const a of s.assignments){if(a==='foreign')continue;let n=Number(a);if(!counts.has(n))return false;counts.set(n,counts.get(n)+1)}
return q.faces.every(f=>s.empties.includes(f.index)===(counts.get(f.index)===0))}
function render(){let q=queue[current],s=entry(q);$('samples').value=String(current);$('title').textContent=`${current+1}. ${q.id} (${q.split})`;
$('status').className=q.reasons.length?'warning':'ok';$('status').textContent=q.reasons.length?q.reasons.join(' • '):'Геометрия и crops доступны для ручной проверки; решение ещё не подтверждено';
let photo=$('photo');photo.replaceChildren();let img=new Image();img.src=q.original_url;img.alt='Исходное фото';photo.append(img);
q.raw.forEach((p,i)=>{let pin=document.createElement('span');pin.className='pin';pin.textContent=String(i+1);pin.style.left=(100*p.x_fraction)+'%';pin.style.top=(100*p.y_fraction)+'%';photo.append(pin)});
$('faces').replaceChildren();q.faces.forEach(f=>{let card=document.createElement('div');card.className='card';let label=document.createElement('div');label.textContent='Грань '+f.index;
let image=new Image();image.src=f.url;image.alt='Canonical face '+f.index;card.append(label,image);$('faces').append(card)});
$('assignments').replaceChildren();q.raw.forEach((p,i)=>{let row=document.createElement('div'),label=document.createElement('label');label.textContent=`Точка ${i+1} → `;
let select=document.createElement('select');select.className='assignment';for(const [value,title] of [['','Не проверена'],['foreign','Чужая мишень'],...q.faces.map(f=>[String(f.index),'Грань '+f.index])]){
let option=new Option(title,value);select.add(option)}select.value=s.assignments[i]??'';select.onchange=save;label.append(select);row.append(label);$('assignments').append(row)});
$('empties').replaceChildren();q.faces.forEach(f=>{let label=document.createElement('label'),box=document.createElement('input');box.type='checkbox';box.className='empty';box.value=f.index;
box.checked=s.empties.includes(f.index);box.onchange=save;label.append(box,document.createTextNode(` Грань ${f.index} проверена как пустая`));$('empties').append(label,document.createElement('br'))});
for(const id of ['annotator','reviewer','notes'])$(id).value=s[id]||'';$('owner').checked=!!s.owner;$('checked').checked=!!s.checked;progress()}
function download(name,data){let url=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)+'\n'],{type:'application/json'}));let a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)}
$('samples').replaceChildren();queue.forEach((q,i)=>$('samples').add(new Option(`${i+1}: ${q.id.slice(0,50)}`,String(i))));
$('samples').onchange=()=>{save();current=Number($('samples').value);render()};$('prev').onclick=()=>{save();current=Math.max(0,current-1);render()};
$('next').onclick=()=>{save();current=Math.min(queue.length-1,current+1);render()};for(const id of ['annotator','reviewer','notes','owner','checked'])$(id).onchange=save;
$('draft').onclick=()=>{save();download('stage4-review-draft.json',{manifest_sha256:__MANIFEST_SHA__,state})};
$('import').onclick=()=>$('file').click();$('file').onchange=async()=>{try{let d=JSON.parse(await $('file').files[0].text());if(d.manifest_sha256!==__MANIFEST_SHA__||!d.state||typeof d.state!=='object')throw Error('Другой manifest или неверный draft');
state=d.state;persist();render()}catch(e){alert(e.message)}$('file').value=''};
$('export').onclick=()=>{save();let map={};for(const q of queue){let s=entry(q);if(!valid(q,s))continue;map[q.id]={annotator:s.annotator,reviewer:s.reviewer,face_ownership_reviewed:true,
original_sha256:q.original_sha256,raw_annotations_sha256:q.raw_annotations_sha256,geometry_path:q.geometry_path,geometry_sha256:q.geometry_sha256,
selected_candidate_ids:q.selected_candidate_ids,impact_face_indices:s.assignments.map(a=>a==='foreign'?null:Number(a)),
verified_empty_faces:s.empties.slice().sort((a,b)=>a-b),review_notes:s.notes}}
if(!Object.keys(map).length){alert('Нет полностью проверенных записей');return}download('stage4-review.json',map)};render();
</script></html>'''


def serve(manifest: Path, geometry_results: Path, data_root: Path, port: int = 0,
          open_browser: bool = False) -> None:
    queue, assets = build_queue(manifest, geometry_results, data_root)
    digest = _sha256(manifest)
    page = PAGE.replace("__QUEUE__", json.dumps(queue, ensure_ascii=False).replace("<", "\\u003c"))
    page = page.replace("__MANIFEST_SHA__", json.dumps(digest)).encode("utf-8")

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.headers.get("Host") != f"127.0.0.1:{self.server.server_port}":
                self.send_error(403)
                return
            path = urlsplit(self.path).path
            if path == "/":
                content, mime = page, "text/html; charset=utf-8"
            elif path.startswith("/asset/") and path[7:] in assets:
                source = assets[path[7:]]
                content = source.read_bytes()
                mime = mimetypes.guess_type(source.name)[0] or "application/octet-stream"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'none'; img-src 'self'; script-src 'unsafe-inline'; style-src 'unsafe-inline'")
            self.end_headers()
            self.wfile.write(content)

    with ThreadingHTTPServer(("127.0.0.1", port), Handler) as server:
        url = f"http://127.0.0.1:{server.server_port}/"
        print(f"Review {len(queue)} samples at {url}", flush=True)
        if open_browser:
            webbrowser.open(url)
        server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description="Local Stage 4 independent impact review desk")
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/dataset_manifest.json"))
    parser.add_argument("--geometry-results", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--open-browser", action="store_true")
    args = parser.parse_args()
    serve(args.manifest, args.geometry_results, args.data_root, args.port, args.open_browser)


if __name__ == "__main__":
    main()
