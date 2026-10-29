"""Reproducible geometry diagnostics; human center feedback is never holdout GT."""
import argparse
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import resource
import subprocess
import time

import cv2
import numpy as np
from PIL import Image, ImageOps

from neural_network.archery_ml.geometry.detector import TargetGeometryDetector
from neural_network.archery_ml.geometry.quality import ImageQualityGate, QualityStatus
from neural_network.archery_ml.geometry.synthetic import project, scenarios
from neural_network.archery_ml.targets import TargetConfig, TargetFormat, TenRingMode, TripleLayout


def summary(values):
    return {"count":len(values), **dict(zip(("median","p95","max"),
        map(float,np.percentile(values,[50,95,100]))))} if values else {"count":0}


def json_value(value):
    if isinstance(value,np.ndarray):
        return value.tolist()
    if isinstance(value,np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def visualize(image, geometry, truth_centers, path, title, origin=(0,0)):
    canvas = image.copy()
    origin = np.asarray(origin)
    angles = np.linspace(0,2*np.pi,180)
    for face in geometry.faces:
        inverse = np.asarray(face.inverse_homography)
        for radius in (.1,.2,.3,.4,.5,.6,.8,1.):
            points = project(255.5+256*radius*np.column_stack((np.cos(angles),np.sin(angles))),inverse)+origin
            if np.isfinite(points).all() and np.max(np.abs(points))<1e6:
                cv2.polylines(canvas,[points.round().astype(np.int32)],True,(40,255,130),2)
        center = tuple(np.round(np.asarray(face.center)+origin).astype(int))
        cv2.drawMarker(canvas,center,(0,255,0),cv2.MARKER_CROSS,24,3)
        for arc in getattr(face,"observed_arcs",()):
            pts = np.asarray(arc["points"])+origin
            cv2.polylines(canvas,[pts.round().astype(np.int32)],False,(255,110,20),2)
        for point in getattr(face,"excluded_points",()):
            cv2.circle(canvas,tuple(np.round(np.asarray(point)+origin).astype(int)),3,(255,0,255),-1)
    for center in truth_centers:
        cv2.drawMarker(canvas,tuple(np.round(center).astype(int)),(255,40,40),cv2.MARKER_TILTED_CROSS,22,3)
    factor = min(1.,1280/max(canvas.shape[:2]))
    canvas = cv2.resize(canvas,None,fx=factor,fy=factor,interpolation=cv2.INTER_AREA)
    cv2.putText(canvas,title[:130],(12,25),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),2)
    Image.fromarray(canvas).save(path)


def timed_detect(detector,image,config):
    start = time.perf_counter()
    geometry = detector.detect(image,config)
    return geometry,(time.perf_counter()-start)*1000


def run(output, method="legacy", pilot=None, reviews=None, image_paths=(), config=None):
    output = Path(output)
    output.mkdir(parents=True,exist_ok=False)
    (output/'previews').mkdir()
    try:
        detector = TargetGeometryDetector(method=method)
    except TypeError:
        if method != "legacy":
            raise
        detector = TargetGeometryDetector()
    gate = ImageQualityGate()
    records=[]
    source = Path(__file__).parent
    provenance={"code_sha":subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        "source_hashes":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(source.glob('*.py'))},
        "python":platform.python_version(),"numpy":np.__version__,"opencv":cv2.__version__,
        "method":method,"independent_real_holdout":"NOT_VALIDATED",
        "runtime_scope":"desktop detector only; decode and visualization excluded; no Android claim"}
    for index, scene in enumerate(scenarios()):
        geometry,elapsed=timed_detect(detector,scene.image,scene.config)
        quality=gate.evaluate(scene.image,geometry)
        center_errors=[];point_errors=[];boundary_errors=[]
        for face,matrix in zip(geometry.faces,scene.norm_to_image):
            true_center=project([[0.,0.]],matrix)[0]
            center_errors.append(float(np.linalg.norm(project([face.center],np.linalg.inv(matrix))[0])))
            angles=np.linspace(0,2*np.pi,32,endpoint=False)
            probes=.45*np.column_stack((np.cos(angles),np.sin(angles)))
            predicted=(project(project(probes,matrix),np.asarray(face.homography))-255.5)/256
            # Physical orientation stays unanchored. Rotation-free radial errors
            # are separate from 2D errors in the declared synthetic image axes.
            point_errors.extend(np.linalg.norm(predicted-probes,axis=1).tolist())
            for radius in (.2,.4,min(.6,(11-scene.config.minimum_scoring_zone)*.1)):
                measured=(project(project(radius*np.column_stack((np.cos(angles),np.sin(angles))),matrix),
                                  np.asarray(face.homography))-255.5)/256
                boundary_errors.extend(np.abs(np.linalg.norm(measured,axis=1)-radius).tolist())
        name=f'synthetic-{index:02}-{scene.scenario}'
        truth=[project([[0.,0.]],m)[0].tolist() for m in scene.norm_to_image]
        record={"id":name,"dataset":"independent_synthetic","scenario":scene.scenario,"seed":scene.seed,
            "zone":scene.config.minimum_scoring_zone,"layout":str(scene.config.triple_layout),
            "status":quality.status.value,"latency_ms":elapsed,"center_errors_norm":center_errors,
            "point_errors_2d":point_errors,"ring_boundary_errors":boundary_errors,
            "expected_reject":scene.expected_reject,"expected_faces":len(scene.norm_to_image),
            "faces":asdict(geometry)["faces"],"input_sha256":hashlib.sha256(scene.image.tobytes()).hexdigest(),
            "preview":f'previews/{name}.jpg'}
        records.append(record)
        visualize(scene.image,geometry,truth,output/record['preview'],name+' '+quality.status.value)
        _canonical_previews(scene.image,geometry,output,name,quality.status.value)
    if pilot:
        pilot=Path(pilot)
        inventory=json.loads((pilot/'inventory.json').read_text())['images']
        detections=json.loads((pilot/'detections.json').read_text())
        choices=json.loads((pilot/'target-selections.json').read_text())
        raw=Path(reviews).read_bytes() if reviews else b'{"images":{}}'
        reference=json.loads(raw)['images']
        provenance['review_sha256']=hashlib.sha256(raw).hexdigest()
        (output/'input-reviews.json').write_bytes(raw)
        for row in inventory:
            key=row['original_id'];choice=choices.get(key,{})
            if row['read_status']!='OK' or not choice.get('candidate_ids'):
                records.append({'id':key,'dataset':'development_real','status':'INPUT_REJECTED','reason':'missing group or image'})
                continue
            path=pilot/row['rgb_path']
            if hashlib.sha256(path.read_bytes()).hexdigest()!=row['rgb_sha256']:
                raise ValueError('image binding changed')
            rgb=np.asarray(Image.open(path).convert('RGB'))
            selected=[next(c for c in detections[key]['candidates'] if c['id']==cid) for cid in choice['candidate_ids']]
            zone=choice.get('minimum_scoring_zone',choice.get('minimum_zone_proposal',5))
            target=TargetConfig(TargetFormat(choice['format']),zone,TenRingMode.RECURVE,1.,
                None if choice['layout'] is None else TripleLayout(choice['layout']))
            bounds=np.asarray([c['bbox'] for c in selected]);margin=max(c['diameter'] for c in selected)*.1
            x0,y0=np.maximum(0,np.floor(bounds[:,:2].min(axis=0)-margin)).astype(int)
            x1,y1=np.minimum(rgb.shape[1::-1],np.ceil(bounds[:,2:].max(axis=0)+margin)).astype(int)
            isolated=rgb.copy()
            for c in detections[key]['candidates']:
                if c['id'] not in choice['candidate_ids']:
                    e=c['ellipse'];cv2.ellipse(isolated,((e['center_x'],e['center_y']),
                        (e['diameter_x']*1.1,e['diameter_y']*1.1),e['angle_deg']),(118,118,118),-1)
            roi=isolated[y0:y1,x0:x1]
            geometry,elapsed=timed_detect(detector,roi,target)
            quality=gate.evaluate(roi,geometry)
            truth={v['id']:np.asarray([v['x'],v['y']]) for v in reference.get(key,{}).get('points',[])}
            ref=reference.get(key,{})
            if ref and (ref['image_sha256']!=row['rgb_sha256'] or ref['coordinate_space']!='upright_rgb_px'):
                raise ValueError('reference binding mismatch')
            pairs=[];assigned=set()
            for face in geometry.faces:
                center=np.asarray(face.center)+[x0,y0]
                nearest=min(selected,key=lambda c:np.linalg.norm(center-c['center']))
                cid=nearest['id']
                if cid in assigned or np.linalg.norm(center-nearest['center'])>.4*nearest['diameter']:
                    continue
                assigned.add(cid)
                if cid in truth:
                    pairs.append({'id':cid,'error_px':float(np.linalg.norm(center-truth[cid])),
                                  'center':center.tolist(),'review':truth[cid].tolist()})
            name=f'real-{row["number"]:02}'
            record={'id':key,'number':row['number'],'dataset':'development_real','status':quality.status.value,
                'latency_ms':elapsed,'pairs':pairs,'expected_review_points':len(truth),'roi':[int(x0),int(y0),int(x1),int(y1)],
                'selection_provenance':'cached user-selected group; foreign candidates masked identically for both methods',
                'independent_gt':False,'faces':asdict(geometry)['faces'],'preview':f'previews/{name}.jpg'}
            records.append(record)
            visualize(rgb,geometry,list(truth.values()),output/record['preview'],name+' '+quality.status.value,(x0,y0))
            _canonical_previews(roi,geometry,output,name,quality.status.value)
            print(name,quality.status.value,flush=True)
    for path in image_paths:
        rgb=np.asarray(ImageOps.exif_transpose(Image.open(path)).convert('RGB'))
        geometry,elapsed=timed_detect(detector,rgb,config)
        status=gate.evaluate(rgb,geometry).status.value
        name=hashlib.sha256(Path(path).read_bytes()).hexdigest()
        records.append({'id':name,'dataset':'unlabeled','status':status,'latency_ms':elapsed,
                        'faces':asdict(geometry)['faces'],'preview':f'previews/{name}.jpg'})
        visualize(rgb,geometry,[],output/f'previews/{name}.jpg',status)
    synthetic=[r for r in records if r['dataset']=='independent_synthetic']
    ready=[r for r in synthetic if r['status']=='READY']
    real=[r for r in records if r['dataset']=='development_real']
    report={'schema_version':1,'provenance':provenance,'records':records,
        'synthetic':{'samples':len(synthetic),'ready':len(ready),
            'center_error_norm':summary([e for r in synthetic for e in r['center_errors_norm']]),
            'point_error_2d':summary([e for r in synthetic for e in r['point_errors_2d']]),
            'ring_boundary_error':summary([e for r in synthetic for e in r['ring_boundary_errors']]),
            'false_accept_fraction':sum(r['expected_reject'] or not r['center_errors_norm'] or
                max(r['center_errors_norm'])>.035 or len(r['center_errors_norm'])!=r['expected_faces'] for r in ready)/max(1,len(ready)),
            'false_reject_fraction':sum(r['status']!='READY' and not r['expected_reject'] for r in synthetic)/max(1,sum(not r['expected_reject'] for r in synthetic))},
        'development_real':{'samples':len(real),'ready':sum(r['status']=='READY' for r in real),
            'expected_review_points':sum(r.get('expected_review_points',0) for r in real),
            'measured_points':sum(len(r.get('pairs',[])) for r in real),
            'center_error_px':summary([p['error_px'] for r in real for p in r.get('pairs',[])])},
        'latency_ms':{name:summary([r['latency_ms'] for r in records if r['dataset']==name and 'latency_ms' in r])
                      for name in ('independent_synthetic','development_real','unlabeled')},
        'process_peak_memory_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if platform.system()=='Darwin' else 1024)}
    (output/'report.json').write_text(json.dumps(report,indent=2,default=json_value,allow_nan=False))
    with (output/'metrics.csv').open('w') as stream:
        writer=csv.writer(stream);writer.writerow(['dataset','id','status','latency_ms','measured_centers','reasons'])
        for r in records:
            writer.writerow([r['dataset'],r['id'],r['status'],r.get('latency_ms'),len(r.get('pairs',r.get('center_errors_norm',[]))),
                ';'.join(reason for f in r.get('faces',[]) for reason in f.get('rejection_reasons',[]))])
    thumbs=[]
    for r in records:
        if 'preview' in r:
            im=Image.open(output/r['preview']);im.thumbnail((256,256));thumbs.append(im.copy())
    sheet=Image.new('RGB',(256*6,280*((len(thumbs)+5)//6)),(30,30,30))
    for i,im in enumerate(thumbs):sheet.paste(im,((i%6)*256,(i//6)*280))
    sheet.save(output/'contact-sheet.jpg')
    print(json.dumps({k:v for k,v in report.items() if k not in ('records','provenance')},indent=2))
    return report


def _canonical_previews(image,geometry,output,name,status):
    for face in geometry.faces:
        crop=cv2.warpPerspective(image,np.asarray(face.homography),(512,512))
        for coordinate in range(0,512,64):
            cv2.line(crop,(coordinate,0),(coordinate,511),(120,255,120),1)
            cv2.line(crop,(0,coordinate),(511,coordinate),(120,255,120),1)
        cv2.putText(crop,status,(8,24),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),2)
        Image.fromarray(crop).save(output/'previews'/f'{name}-canonical-{face.face_index}.jpg')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--method',choices=('legacy','robust'),default='robust')
    parser.add_argument('--pilot',type=Path)
    parser.add_argument('--reviews',type=Path)
    parser.add_argument('--image',type=Path,action='append',default=[])
    parser.add_argument('--folder',type=Path)
    parser.add_argument('--zone',type=int,choices=(1,5,6),default=5)
    parser.add_argument('--layout',choices=('VERTICAL','TRIANGULAR'))
    args=parser.parse_args(argv)
    paths=args.image+(sorted(p for p in args.folder.iterdir() if p.suffix.lower() in ('.png','.jpg','.jpeg')) if args.folder else [])
    config=TargetConfig(TargetFormat.TRIPLE if args.layout else TargetFormat.SINGLE,args.zone,TenRingMode.RECURVE,40.,
                        TripleLayout(args.layout) if args.layout else None)
    run(args.output,args.method,args.pilot,args.reviews,paths,config)


if __name__=='__main__':
    main()
