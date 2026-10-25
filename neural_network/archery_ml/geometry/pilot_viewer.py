"""Offline review UI and optional loopback-only JSON persistence."""
from functools import partial
from http.server import SimpleHTTPRequestHandler,HTTPServer
import json
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from PIL import Image

from neural_network.archery_ml.geometry.pilot_inventory import save_json
from neural_network.archery_ml.geometry.selection import fingerprint


def write_viewer(output,inventory,detections,selections,results,reviews):
    data = {"inventory":inventory,"detections":detections,"selections":selections,"results":results,"reviews":reviews}
    payload = json.dumps(data,ensure_ascii=False,allow_nan=False).replace("<","\\u003c")
    document = '''<!doctype html><html lang="ru"><meta charset="utf-8"><title>Mobile target pilot</title>
<style>body{font:15px system-ui;margin:18px;background:#f4f5f6;color:#162331}button,select,input{padding:8px;margin:3px}button{cursor:pointer}button.active{background:#c9edcf}.top{position:sticky;top:0;background:#f4f5f6;z-index:2;padding:6px}.grid{display:grid;grid-template-columns:minmax(300px,1fr) minmax(280px,1fr);gap:18px}.panel{background:white;padding:12px;border-radius:10px}svg{max-height:75vh;width:100%;background:#ddd}ellipse{cursor:pointer}img.crop{width:31%;max-width:240px}pre{white-space:pre-wrap}#message{color:#254b86}label{display:inline-block}#groups{max-height:24vh;overflow:auto}@media(max-width:850px){.grid{grid-template-columns:1fr}}</style>
<div class="top"><b>Legacy exploratory / mobile_dev — НЕ independent holdout</b><br>
<button id="prev">←</button><select id="photo"></select><button id="next">→</button>
<button id="exportSelections">Save selections JSON</button><button id="exportReviews">Save ratings JSON</button>
<button id="connect">Connect output directory</button><a href="contact-sheet.html">Все фото</a>
<span id="message"></span></div>
<h2 id="heading"></h2><p id="status"></p>
<div class="grid"><section class="panel"><svg id="image"></svg><p>Клик по face: ручной набор 1 или 3 face. Порядок кликов задаёт индексы 0,1,2; физическая идентичность требует отдельного подтверждения.</p><div id="groups"></div></section>
<section class="panel"><h3>Выбор своей мишени</h3><p>Автоматическая группа — предложение, а не подтверждение владельца. Количество стрел не используется.</p>
<label>Format <select id="format"><option>TRIPLE</option><option>SINGLE</option></select></label>
<label>Layout <select id="layout"><option>VERTICAL</option><option>TRIANGULAR</option></select></label>
<label>Зоны <select id="zone"><option value="auto">inferred / confirm</option><option>1</option><option>5</option><option>6</option></select></label><br>
<label><input id="physicalOrder" type="checkbox">Физические IDs 0/1/2 проверены мной</label>
<label><input id="remember" type="checkbox" checked>Предлагать выбор для визуально той же съёмки</label>
<p id="selected"></p><button id="confirm">Confirm</button><button id="uncertain">Uncertain</button><button id="noTarget">No target</button>
<h3>Геометрия и canonical crops</h3><p id="geometry"></p><div id="crops"></div>
<h3>Быстрая visual review</h3><p>Внешний вид не заменяет independent GT. Physical orientation остаётся UNANCHORED.</p><div id="ratings"></div>
<pre id="details"></pre></section></div>
<script id="payload" type="application/json">__PAYLOAD__</script>
<script>
const data=JSON.parse(document.getElementById('payload').textContent), $=id=>document.getElementById(id);
const rows=data.inventory.images;let current=0,chosen=[],highlight=null,directory=null;
const storage='archery-pilot-'+rows.map(r=>r.original_id||r.filename).join('').slice(0,80);
try{const saved=JSON.parse(localStorage.getItem(storage)||'null');if(location.protocol==='file:'&&saved){Object.assign(data.selections,saved.selections);Object.assign(data.reviews,saved.reviews);}}catch{}
function esc(value){return String(value).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}
function key(){return rows[current].original_id||'unreadable-'+rows[current].number;}
function group(){return data.detections[key()]?.groups.find(g=>JSON.stringify(g.candidate_ids)===JSON.stringify(chosen));}
function download(name,value){const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(value,null,2)+'\\n'],{type:'application/json'}));a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);}
async function persist(kind,value){
 if(location.protocol!=='file:'){
  const response=await fetch('/api/'+kind,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({original_id:key(),value})});
  const result=await response.json();if(!response.ok)throw Error(result.error);Object.assign(data,result);$('message').textContent='Saved to report JSON';
 }else if(directory){const filename=kind==='selection'?'target-selections.json':'visual-reviews.json';const h=await directory.getFileHandle(filename,{create:true});const w=await h.createWritable();await w.write(JSON.stringify(kind==='selection'?data.selections:data.reviews,null,2));await w.close();$('message').textContent='Saved JSON; rerun batch for new crops';}
 else $('message').textContent='Saved in browser. Export JSON / Connect directory, or use --serve-only for automatic processing.';
 localStorage.setItem(storage,JSON.stringify({selections:data.selections,reviews:data.reviews}));
}
function render(reset=true){
 const row=rows[current],k=key(),det=data.detections[k],sel=data.selections[k],res=data.results[k];
 if(reset){chosen=[...(sel?.candidate_ids||[])];highlight=sel?.group_id||null;$('physicalOrder').checked=!!sel?.physical_face_order_confirmed;$('zone').value=sel?.minimum_scoring_zone||'auto';$('format').value=sel?.format||'TRIPLE';$('layout').value=sel?.layout||'VERTICAL';}
 $('photo').value=current;$('heading').textContent='#'+row.number+' '+row.filename;
 $('status').textContent=(sel?.status||'NO_GROUP')+' | ownership '+(sel?.status==='USER_CONFIRMED'?'USER CONFIRMED':'UNVERIFIED')+' | orientation UNANCHORED';
 const [w,h]=row.rgb_size||[1,1];$('image').setAttribute('viewBox',`0 0 ${w} ${h}`);
 $('image').innerHTML=`<image href="${esc(row.preview_path||'')}" width="${w}" height="${h}"/>`+(det?.candidates||[]).map(c=>`<g><ellipse data-face="${esc(c.id)}" cx="${c.ellipse.center_x}" cy="${c.ellipse.center_y}" rx="${c.ellipse.diameter_x/2}" ry="${c.ellipse.diameter_y/2}" transform="rotate(${c.ellipse.angle_deg} ${c.ellipse.center_x} ${c.ellipse.center_y})" fill="${chosen.includes(c.id)?'#00dd7744':'#ff990011'}" stroke="${chosen.includes(c.id)?'#00bb55':'#ff9900'}" stroke-width="${w/250}"/><text x="${c.center[0]}" y="${c.center[1]}" font-size="${w/40}" fill="#e500ff" pointer-events="none">${esc(c.id)}</text></g>`).join('');
 $('image').querySelectorAll('[data-face]').forEach(el=>el.onclick=()=>{const id=el.dataset.face;chosen=chosen.includes(id)?chosen.filter(c=>c!==id):[...chosen,id];if(chosen.length>3)chosen.shift();highlight=null;render(false);});
 $('selected').textContent='Face order: '+chosen.join(', ');
 $('groups').innerHTML='';for(const g of det?.groups||[]){if(g.format==='SINGLE'&&(det?.candidates.length||0)>1)continue;const b=document.createElement('button');b.textContent=`Select ${g.id} ${g.format} ${g.layout||''}: ${g.candidate_ids.join(', ')} (${g.confidence.toFixed(2)})`;b.className=highlight===g.id?'active':'';b.onclick=()=>{chosen=[...g.candidate_ids];highlight=g.id;$('format').value=g.format;$('layout').value=g.layout||'VERTICAL';render(false);};$('groups').appendChild(b);}
 const g=group(),preview=g?res?.group_results?.[g.id]:null;
 const actual=JSON.stringify(chosen)===JSON.stringify(res?.candidate_ids)?res:preview;
 if(actual?.faces){const P=(h,x,y)=>[(h[0][0]*x+h[0][1]*y+h[0][2])/(h[2][0]*x+h[2][1]*y+h[2][2]),(h[1][0]*x+h[1][1]*y+h[1][2])/(h[2][0]*x+h[2][1]*y+h[2][2])];for(const f of actual.faces){for(const r of [.2,.4,Math.min(.6,(11-actual.processing_config.minimum_scoring_zone)*.1)]){const points=[];for(let i=0;i<=64;i++){const a=2*Math.PI*i/64;points.push(P(f.canonical_to_rgb,255.5+256*r*Math.cos(a),255.5+256*r*Math.sin(a)).join(','));}const line=document.createElementNS('http://www.w3.org/2000/svg','polyline');line.setAttribute('points',points.join(' '));line.setAttribute('fill','none');line.setAttribute('stroke',actual.status==='READY'?'#00ffff':'#ff4466');line.setAttribute('stroke-width',w/600);line.setAttribute('pointer-events','none');$('image').appendChild(line);}}}
 $('geometry').textContent=actual?`${actual.status} / ${actual.quality_status}; fit ${actual.fit_error??'unknown'}; confidence ${actual.geometry_confidence??'unknown'}; OWNERSHIP ${sel?.status==='USER_CONFIRMED'?'confirmed':'unverified'}`:'Selection needs processing; use loopback server or rerun batch with exported selections';
 $('crops').innerHTML=(actual?.faces||[]).filter(f=>f.crop_path).map(f=>`<a href="${esc(f.crop_path)}?v=${esc(actual.artifact_revision||'initial')}" target="_blank"><img class="crop" src="${esc(f.display_crop_path||f.crop_path)}?v=${esc(actual.artifact_revision||'initial')}" alt="face ${f.face_index}" title="face ${f.face_index}; click for full unmasked canonical crop"></a>`).join('');
 $('details').textContent=JSON.stringify({selected:sel,quality:actual?.quality_full_frame,diagnostics:actual?.diagnostics,visual_review:data.reviews[k],config:actual?.processing_config,config_assumptions:actual?.config_assumptions},null,2);
}
for(let i=0;i<rows.length;i++){const o=document.createElement('option');o.value=i;o.textContent='#'+rows[i].number+' '+rows[i].filename;$('photo').appendChild(o);}
$('photo').onchange=()=>{current=Number($('photo').value);render();};$('prev').onclick=()=>{current=Math.max(0,current-1);render();};$('next').onclick=()=>{current=Math.min(rows.length-1,current+1);render();};
async function select(status){try{const det=data.detections[key()],g=group();if(status==='USER_CONFIRMED'&&![1,3].includes(chosen.length))throw Error('Select exactly one or three face');const value={status,source:'USER',candidate_ids:status==='USER_CONFIRMED'?chosen:[],group_id:g?.id||'manual',format:chosen.length===1?'SINGLE':'TRIPLE',layout:chosen.length===1?null:(g?.layout||$('layout').value),detection_fingerprint:det.fingerprint,physical_face_order_confirmed:$('physicalOrder').checked,ownership_confirmed:status==='USER_CONFIRMED',remember:$('remember').checked};if($('zone').value!=='auto')value.minimum_scoring_zone=Number($('zone').value);else if(g)value.minimum_scoring_zone=g.minimum_zone_proposal;data.selections[key()]=value;await persist('selection',value);render();}catch(e){$('message').textContent=e.message;}}
$('confirm').onclick=()=>select('USER_CONFIRMED');$('uncertain').onclick=()=>select('AMBIGUOUS');$('noTarget').onclick=()=>select('NO_GROUP');
for(const rating of ['Correct','Wrong group','Wrong geometry','Wrong face order','Uncertain']){const b=document.createElement('button');b.textContent=rating;b.onclick=async()=>{try{const value={rating,source:'USER_VISUAL_REVIEW',candidate_ids:[...chosen],quantitative_gt:false};data.reviews[key()]=value;await persist('review',value);render(false);}catch(e){$('message').textContent=e.message;}};$('ratings').appendChild(b);}
$('exportSelections').onclick=()=>download('target-selections.json',data.selections);$('exportReviews').onclick=()=>download('visual-reviews.json',data.reviews);
$('connect').onclick=async()=>{try{directory=await window.showDirectoryPicker({mode:'readwrite'});$('message').textContent='Directory connected. JSON will be saved automatically; rerun batch for updated crops.';}catch(e){$('message').textContent='Use Save JSON or loopback server: '+e.message;}};
render();
</script></html>'''.replace("__PAYLOAD__",payload)
    (output/"review.html").write_text(document)


def update(output,original_id,kind,value):
    from neural_network.archery_ml.geometry.pilot import read_json,process_frame,reports
    inventory = read_json(output/"inventory.json",{})
    if not isinstance(original_id,str) or not isinstance(value,dict):
        raise ValueError("expected an original ID and a JSON object")
    row = next((r for r in inventory["images"] if r.get("original_id",f"unreadable-{r['number']}")==original_id),None)
    if row is None:
        raise ValueError("unknown original ID")
    detections = read_json(output/"detections.json",{})
    selections = read_json(output/"target-selections.json",{})
    results = read_json(output/"geometry-results.json",{})
    if kind == "selection":
        if value.get("status") not in {"USER_CONFIRMED","AMBIGUOUS","NO_GROUP"}:
            raise ValueError("invalid user selection status")
        det = detections[original_id]
        if value.get("detection_fingerprint") != det["fingerprint"]:
            raise ValueError("stale detections; reload viewer")
        ids = value.get("candidate_ids",[])
        valid = {c["id"] for c in det["candidates"]}
        if len(ids)!=len(set(ids)) or any(cid not in valid for cid in ids):
            raise ValueError("invalid candidate identities")
        if value["status"] == "USER_CONFIRMED" and len(ids) not in {1,3}:
            raise ValueError("select one or three faces")
        value.update(source="USER",ownership_confirmed=value["status"]=="USER_CONFIRMED",saved_at=datetime.now(timezone.utc).isoformat())
        value["format"] = "SINGLE" if len(ids)==1 else "TRIPLE"
        if len(ids)==1:
            value["layout"] = None
        elif value.get("layout") not in {"VERTICAL","TRIANGULAR"}:
            value["layout"] = "VERTICAL"
        if value.get("minimum_scoring_zone") not in {None,1,5,6}:
            raise ValueError("unsupported scoring-zone proposal")
        selections[original_id] = value
        previews = results[original_id].get("group_results",{})
        results[original_id] = process_frame(row,det,value,output,previews=False)
        results[original_id]["group_results"] = previews
        if value["status"] == "USER_CONFIRMED" and value.get("remember"):
            from neural_network.archery_ml.geometry.selection import track_confirmed_group
            previous_preview=np.asarray(Image.open(output/row["preview_path"]).convert("RGB"))
            for next_row in inventory["images"]:
                other=next_row.get("original_id")
                if other==original_id or next_row["read_status"]!="OK" or selections[other].get("source")=="USER":
                    continue
                suggestion=track_confirmed_group(row,next_row,det,detections[other],value,previous_preview,
                    np.asarray(Image.open(output/next_row["preview_path"]).convert("RGB")))
                if suggestion:
                    selections[other]=suggestion
                    cached=results[other].get("group_results",{})
                    results[other]=process_frame(next_row,detections[other],suggestion,output,previews=False)
                    results[other]["group_results"]=cached
    elif kind == "review":
        if value.get("rating") not in {"Correct","Wrong group","Wrong geometry","Wrong face order","Uncertain"}:
            raise ValueError("invalid visual review rating")
        ids = value.get("candidate_ids",[])
        valid = {c["id"] for c in detections[original_id]["candidates"]}
        if not isinstance(ids,list) or any(not isinstance(cid,str) or cid not in valid for cid in ids) or len(ids)!=len(set(ids)):
            raise ValueError("visual review must refer to actual candidate identities")
        reviews = read_json(output/"visual-reviews.json",{})
        value.update(source="USER_VISUAL_REVIEW",quantitative_gt=False,detection_fingerprint=detections[original_id]["fingerprint"],
                     stale=False,saved_at=datetime.now(timezone.utc).isoformat())
        reviews[original_id] = value
        save_json(output/"visual-reviews.json",reviews)
    else:
        raise ValueError("unknown review operation")
    reports(output,inventory,detections,selections,results)
    return {"selections":selections,"results":results,"reviews":read_json(output/"visual-reviews.json",{})}


def serve(output,port=8765):
    output = Path(output).resolve()
    class Handler(SimpleHTTPRequestHandler):
        def do_POST(self):
            try:
                host = self.headers.get("Host","")
                if host not in {f"127.0.0.1:{port}",f"localhost:{port}"}:
                    raise ValueError("loopback host required")
                origin = self.headers.get("Origin")
                if origin is not None and origin not in {f"http://127.0.0.1:{port}",f"http://localhost:{port}"}:
                    raise ValueError("same-origin request required")
                length = int(self.headers.get("Content-Length",0))
                if self.headers.get_content_type()!="application/json" or not 0<length<100000:
                    raise ValueError("bounded JSON request required")
                payload = json.loads(self.rfile.read(length))
                value = update(output,payload["original_id"],self.path.removeprefix("/api/"),payload["value"])
                body,status = json.dumps(value,ensure_ascii=False,allow_nan=False).encode(),200
            except (ValueError,KeyError,TypeError) as error:
                body,status = json.dumps({"error":str(error)}).encode(),400
            self.send_response(status);self.send_header("Content-Type","application/json");self.send_header("Content-Length",str(len(body)));self.end_headers();self.wfile.write(body)
    server = HTTPServer(("127.0.0.1",port),partial(Handler,directory=str(output)))
    print(f"Local private viewer: http://127.0.0.1:{port}/review.html",flush=True)
    server.serve_forever()
