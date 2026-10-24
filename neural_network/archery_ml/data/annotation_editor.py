"""Offline point entry; canonical values are entered by a human, never predicted."""
import base64
import html
import io
import json

from PIL import Image

from neural_network.archery_ml.contracts import DatasetSample


def write_editor(root, record, output):
    payload = json.loads(record.read_text())
    sample = DatasetSample.from_dict(payload)
    if (sample.capture_metadata or {}).get("annotation_state") != "draft":
        raise ValueError("only draft records may be edited; create a new revision for reviewed GT")
    image_path = (root / sample.image_path).resolve()
    if not image_path.is_relative_to(root.resolve()):
        raise ValueError("annotation image outside dataset directory")
    with Image.open(image_path) as raw:
        if raw.getexif().get(274,1) != 1:
            raise ValueError("register an explicit EXIF derivative first")
        image = raw.convert("RGB")
        buffer = io.BytesIO()
        image.save(buffer,format="PNG")
    image_url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
    document = '''<!doctype html><html lang="ru"><meta charset="utf-8">
<title>Stage 3 — независимая разметка</title>
<style>body{font:16px system-ui;margin:24px;max-width:1100px}label{display:inline-block;margin:8px}input{width:110px}canvas{max-width:100%;height:auto;outline:1px solid #999}textarea{width:100%;height:200px}button{padding:10px;margin:8px}</style>
<h1>Независимая разметка GT</h1>
<p>Кликните центр или landmark на фото. Canonical coordinates и uncertainty введите по независимому физическому измерению. Предсказания детектора здесь не используются. Страница работает локально, без отправки данных.</p>
<label>Face ID <input id="face" type="number" min="0" value="0"></label>
<label>Точка <select id="kind"><option value="landmark">Landmark</option><option value="target_center">Физический центр</option></select></label>
<label>Landmark ID <input id="pointid" value="f0-p1"></label>
<label>Canonical X <input id="cx" type="number" step="any"></label>
<label>Canonical Y <input id="cy" type="number" step="any"></label>
<label>Uncertainty <input id="uncertainty" type="number" min="0" step="any"></label>
<label>Пригодность <select id="suitability"><option>unknown</option><option>usable</option><option>unusable</option></select></label>
<label>GT orientation <select id="orientation"><option>unknown</option><option>physical_marker</option><option>board</option><option>manual_calibration</option></select></label>
<label>Независимый метод GT <select id="method"><option value="">Выберите</option><option>physical_measurement</option><option>surveyed_print_template</option><option>independent_calibration</option><option>independent_suitability_review</option></select></label>
<p id="message" role="status"></p><canvas id="canvas"></canvas>
<p><button id="undo">Удалить последнюю точку</button><button id="save">Скачать draft JSON</button></p>
<p>Полный draft доступен ниже. После скачивания замените файл drafts/ с тем же именем, откройте preview и передайте независимому reviewer. Reviewed records не редактируются этим инструментом.</p>
<textarea id="record">__PAYLOAD__</textarea>
<script>
const $ = id => document.getElementById(id);
const sample = JSON.parse($('record').value);
sample.geometry_annotations ??= [];
const meta=sample.capture_metadata;
const count=sample.target_metadata.format==='SINGLE'?1:3;
$('face').max=count-1;
for(const [id,key] of [['suitability','suitability'],['orientation','orientation_reference'],['method','independent_gt_method']]) $(id).value=meta[key]|| (id==='method'?'':'unknown');
const image = new Image(), canvas=$('canvas'), ctx=canvas.getContext('2d');
function sync(){ $('record').value=JSON.stringify(sample,null,2); }
function draw(){ctx.drawImage(image,0,0);for(const a of sample.geometry_annotations){for(const p of a.points){const x=p[0]*image.width,y=p[1]*image.height;ctx.strokeStyle='lime';ctx.beginPath();ctx.arc(x,y,5,0,Math.PI*2);ctx.stroke();ctx.fillStyle='lime';ctx.fillText(`face ${a.face_index} ${a.source_label||a.kind}`,x+7,y);}}sync();}
$('record').onchange=()=>{try{const edited=JSON.parse($('record').value);if(edited.id!==sample.id||edited.image_path!==sample.image_path)throw Error('Нельзя менять identity или изображение в редакторе');Object.assign(sample,edited);for(const [id,key] of [['suitability','suitability'],['orientation','orientation_reference'],['method','independent_gt_method']])$(id).value=sample.capture_metadata[key]||(id==='method'?'':'unknown');draw();}catch(error){$('message').textContent=error.message;}};
image.onload=()=>{canvas.width=image.width;canvas.height=image.height;draw();};
image.src='__IMAGE__';
canvas.onclick=e=>{try{
 const face=Number($('face').value), kind=$('kind').value, uncertainty=Number($('uncertainty').value);
 if(!Number.isInteger(face)||face<0||face>=count) throw Error('Неверный face ID');
 if($('uncertainty').value===''||!Number.isFinite(uncertainty)||uncertainty<0) throw Error('Укажите обоснованную uncertainty');
 const rect=canvas.getBoundingClientRect(), point=[(e.clientX-rect.left)/rect.width,(e.clientY-rect.top)/rect.height];
 if(point.some(v=>!Number.isFinite(v)||v<0||v>1)) throw Error('Точка вне изображения');
 const annotation={kind,points:[point],face_index:face,uncertainty_norm:uncertainty,annotation_quality:'draft'};
 if(kind==='landmark'){
  const x=Number($('cx').value),y=Number($('cy').value),id=$('pointid').value.trim();
  if($('cx').value===''||$('cy').value===''||!Number.isFinite(x)||!Number.isFinite(y)||!id) throw Error('Введите независимые canonical X/Y и landmark ID');
  if(sample.geometry_annotations.some(a=>a.source_label===id)) throw Error('Landmark ID уже используется');
  annotation.canonical_points=[[x,y]];annotation.source_label=id;
 }else{sample.geometry_annotations=sample.geometry_annotations.filter(a=>!(a.kind==='target_center'&&a.face_index===face));}
 sample.geometry_annotations.push(annotation);$('message').textContent='Draft point добавлена; независимый review ещё не выполнен';draw();
}catch(error){$('message').textContent=error.message;}};
$('undo').onclick=()=>{sample.geometry_annotations.pop();draw();};
$('save').onclick=()=>{try{
 const edited=JSON.parse($('record').value);edited.capture_metadata.suitability=$('suitability').value;
 edited.capture_metadata.orientation_reference=$('orientation').value;edited.capture_metadata.independent_gt_method=$('method').value;
 edited.capture_metadata.evaluation_landmark_ids=(edited.geometry_annotations||[]).filter(a=>a.canonical_points?.length).map(a=>a.source_label).filter(Boolean);
 edited.capture_metadata.annotation_state='draft';edited.capture_metadata.reviewer=null;
 delete edited.capture_metadata.gt_annotations_sha256;delete edited.capture_metadata.gt_manifest_sha256;
 const link=document.createElement('a');link.href=URL.createObjectURL(new Blob([JSON.stringify(edited,null,2)+'\\n'],{type:'application/json'}));
 link.download='__FILENAME__';link.style.display='none';document.body.appendChild(link);link.click();setTimeout(()=>{URL.revokeObjectURL(link.href);link.remove();},1000);
}catch(error){$('message').textContent=error.message;}};
</script></html>'''
    document = document.replace("__PAYLOAD__",html.escape(json.dumps(payload,indent=2))).replace("__IMAGE__",image_url)
    document = document.replace("__FILENAME__",sample.id + ".json")
    # IDs from registration are SHA-256; reject untrusted JS string content.
    if len(sample.id) != 64 or any(c not in "0123456789abcdef" for c in sample.id):
        raise ValueError("editor requires registered SHA-256 original ID")
    with output.open("x",encoding="utf-8") as stream:
        stream.write(document)
