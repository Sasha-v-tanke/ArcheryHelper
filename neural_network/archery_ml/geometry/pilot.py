"""Historical real-photo pilot. Selection, visual review and independent GT stay separate."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import platform
import time
import shutil
import tempfile

import cv2
import numpy as np
import PIL
from PIL import Image

from neural_network.archery_ml.geometry.pilot_inventory import inventory,save_json
from neural_network.archery_ml.geometry.selection import TargetCandidateDetector,TargetGroupBuilder,TargetGroupSelector,fingerprint
from neural_network.archery_ml.geometry.selected_geometry import SelectedTargetGeometryPipeline,draw_ellipse
from neural_network.archery_ml.geometry.calibration import project
from neural_network.archery_ml.geometry.cli import _git_state
from neural_network.archery_ml.geometry.evaluate import _metrics,_equal_group_metrics


def read_json(path, default):
    return json.loads(path.read_text()) if path.exists() else default


def overlay(image, detection, result=None, selected_ids=()):
    preview = image.copy()
    line = max(2,round(min(image.shape[:2])/700))
    for candidate in detection["candidates"]:
        chosen = candidate["id"] in selected_ids
        color = (30,230,60) if chosen else (255,170,20)
        draw_ellipse(preview,candidate,color,line)
        x,y = np.rint(candidate["center"]).astype(int)
        cv2.drawMarker(preview,(x,y),color,cv2.MARKER_CROSS,line*8,line)
        cv2.putText(preview,candidate["id"],(x,y),cv2.FONT_HERSHEY_SIMPLEX,line*.45,color,line)
    if result:
        radius = 256.
        for face in result["faces"]:
            h = face["canonical_to_rgb"]
            for r in (.2,.4,min(.6,(11-result["processing_config"]["minimum_scoring_zone"])*.1)):
                angles = np.linspace(0,2*np.pi,100)
                points = project(np.column_stack((255.5+radius*r*np.cos(angles),255.5+radius*r*np.sin(angles))),h)
                cv2.polylines(preview,[np.rint(points).astype(np.int32)],True,(0,220,255),line)
            center = tuple(np.rint(face["center_rgb"]).astype(int))
            cv2.putText(preview,f'face {face["face_index"]}',center,cv2.FONT_HERSHEY_SIMPLEX,line*.45,(240,50,240),line)
    return preview


def relative_result(result, output):
    for face in result["faces"]:
        for key in ("crop_path","display_crop_path"):
            if face.get(key):
                face[key] = Path(face[key]).resolve().relative_to(output.resolve()).as_posix()
    return result


def reviewed_geometry(result, candidate_ids):
    if candidate_ids == result.get("candidate_ids"):
        return result
    return next((g for g in result.get("group_results",{}).values()
                 if candidate_ids == g.get("candidate_ids")),None)


def process_frame(row,detection,selection,output,previews=True,artifact_prefix=""):
    key = row.get("original_id",f"unreadable-{row['number']}")
    if row["read_status"] != "OK":
        return {"original_id":key,"number":row["number"],"status":"REJECTED","quality_status":"INPUT_ERROR","diagnostics":[row.get("error")],"faces":[],"group_results":{},"orientation_status":"UNANCHORED"}
    image = np.asarray(Image.open(output/row["rgb_path"]).convert("RGB"))
    calibrator = None
    board_path = output/"board-template.json"
    if board_path.exists():
        from neural_network.archery_ml.geometry.calibration import BoardTemplate, TargetOrientationCalibrator
        calibrator = TargetOrientationCalibrator(BoardTemplate.load(board_path))
    pipeline = SelectedTargetGeometryPipeline(calibrator)
    group_results = {}
    for group in detection["groups"]:
        if not previews or group["format"]=="SINGLE" and len(detection["candidates"])>1:
            continue
        try:
            value = pipeline.run(image,detection["candidates"],group,row["raw_to_rgb"],output/(artifact_prefix+"crops")/key/group["id"])
            group_results[group["id"]] = relative_result(value,output)
        except (OSError,ValueError,cv2.error) as error:
            group_results[group["id"]] = {"status":"REJECTED","quality_status":"PROCESSING_ERROR","diagnostics":[str(error)],"faces":[]}
    group = None
    if selection.get("candidate_ids") and selection["status"] in {"AUTO_CONFIDENT","USER_CONFIRMED"}:
        group = next((g for g in detection["groups"] if g["candidate_ids"]==selection["candidate_ids"]),None)
        if group is None:
            zones=[c["minimum_zone_proposal"] for c in detection["candidates"] if c["id"] in selection["candidate_ids"]]
            group = {"id":"manual","format":selection["format"],"layout":selection["layout"],"candidate_ids":selection["candidate_ids"],
                     "minimum_zone_proposal":max(set(zones),key=zones.count)}
        group = {**group,**{k:v for k,v in selection.items() if k in {"format","layout","minimum_scoring_zone","physical_face_order_confirmed"}}}
        result = relative_result(pipeline.run(image,detection["candidates"],group,row["raw_to_rgb"],output/(artifact_prefix+"crops")/key/"selected"),output)
    else:
        result = {"status":"REJECTED","quality_status":"GROUP_SELECTION_REQUIRED" if selection["status"]=="AMBIGUOUS" else "NO_GROUP",
            "faces":[],"diagnostics":[selection.get("reason","user target not selected")],"orientation_status":"UNANCHORED"}
    result.update(original_id=key,number=row["number"],selection_status=selection["status"],
        selection_source=selection["source"],ownership_confirmed=selection["status"]=="USER_CONFIRMED",
        group_results=group_results,coordinate_spaces={"raw_size":row["original_size"],"upright_rgb_size":row["rgb_size"],"raw_to_rgb":row["raw_to_rgb"]})
    annotated = overlay(image,detection,result if group else None,selection.get("candidate_ids",[]))
    annotated = Image.fromarray(annotated)
    annotated.thumbnail((1300,1300))
    path = output/(artifact_prefix+"overlays")/f"{key}.jpg"
    path.parent.mkdir(exist_ok=True)
    annotated.save(path,quality=92)
    result["overlay_path"] = path.relative_to(output).as_posix()
    return result


def reports(output,inv,detections,selections,results,before=None):
    rows = inv["images"]
    reviews = read_json(output/"visual-reviews.json",{})
    for key,review in reviews.items():
        if review.get("detection_fingerprint") and review["detection_fingerprint"] != detections.get(key,{}).get("fingerprint"):
            review["stale"] = True
        actual=reviewed_geometry(results.get(key,{}),review.get("candidate_ids"))
        if (review.get("source")=="USER_VISUAL_REVIEW" and selections.get(key,{}).get("status")=="USER_CONFIRMED"
                and review.get("candidate_ids") != selections[key].get("candidate_ids")):
            review["stale"] = True
            review["stale_reason"] = "selected group changed since this review; original rating retained"
        if review.get("geometry_artifact_revision"):
            if actual is None or review["geometry_artifact_revision"] != actual.get("artifact_revision"):
                review["stale"] = True
                review["stale_reason"] = "geometry changed since this visual review; original rating retained"
    failures = []
    for row in rows:
        key = row.get("original_id",f"unreadable-{row['number']}")
        result = results[key]
        selection = selections[key]
        reason = "selection" if selection["status"] in {"AMBIGUOUS","NO_GROUP"} else "geometry" if result["status"]!="READY" else "orientation"
        preview_reasons = Counter(g["quality_status"] for g in result.get("group_results",{}).values())
        severity = (100 if reason=="selection" else 60 if reason=="geometry" else 0)
        severity += max(0,1-result.get("geometry_confidence",0))*10 + min(20,result.get("fit_error",0)*100)
        failures.append({"number":row["number"],"original_id":key,"category":reason,"quality_status":result["quality_status"],
            "selection_status":selection["status"],"severity":severity,"diagnostics":result.get("diagnostics",[]),
            "group_preview_quality":dict(preview_reasons),"physical_orientation":"UNANCHORED",
            "geometry_evidence_scope":"selected_group" if result.get("candidate_ids") else "unselected_group_previews",
            "geometry_issues":[{"group_id":gid,"status":g["quality_status"],"confidence":g.get("geometry_confidence"),
                "fit_error":g.get("fit_error"),"diagnostics":g.get("diagnostics",[])} for gid,g in result.get("group_results",{}).items() if g["status"]!="READY"]})
    failures.sort(key=lambda f:(-f["severity"],f["number"]))
    representative = []
    strata = []
    def take(key,reason):
        if key not in representative:
            representative.append(key);strata.append({"original_id":key,"reason":reason})
    single = next((key for key,d in detections.items() if len(d["candidates"])==1),None)
    if single: take(single,"single / full-face target")
    if detections:
        take(max(detections,key=lambda k:len(detections[k]["candidates"])),"most visible neighbouring face")
    preview_values=[(k,g) for k,r in results.items() for g in r.get("group_results",{}).values() if g.get("quality_full_frame")]
    for key,reason in ([(min(preview_values,key=lambda pair:pair[1]["quality_full_frame"]["mean_luma"])[0],"darkest measured target"),
                       (min(preview_values,key=lambda pair:pair[1]["quality_full_frame"]["minimum_perspective_ratio"])[0],"strongest measured perspective"),
                       (max(preview_values,key=lambda pair:pair[1].get("fit_error",0))[0],"highest fitting residual")] if preview_values else []):
        take(key,reason)
    ready_cases=[(k,g) for k,g in preview_values if g["status"]=="READY"]
    if ready_cases: take(max(ready_cases,key=lambda pair:pair[1].get("geometry_confidence",0))[0],"good geometry proposal")
    for item in failures[:6]+list(reversed(failures))[:6]+([failures[-1]] if failures else []):
        if item["original_id"] not in representative:
            take(item["original_id"],"failure / acceptance coverage")
    # Include the single-face image even if its failure score is unremarkable.
    for key,det in detections.items():
        if len(det["candidates"])==1 and key not in representative:
            representative.append(key)
            break
    pseudo = []
    for key in representative[:15]:
        result = results[key]
        selected = result if result.get("faces") else next((r for r in result.get("group_results",{}).values() if r.get("faces")),None)
        annotations = []
        if selected:
            for face in selected["faces"]:
                radius = min(.6,(11-selected["processing_config"]["minimum_scoring_zone"])*.1)
                canonical = np.array([[0,0],[radius,0],[0,radius],[-radius,0],[0,-radius]])
                points = project(canonical*256+255.5,face["canonical_to_rgb"])
                annotations.append({"face_index":face["face_index"],"image_points_rgb_px":points.tolist(),"canonical_points":canonical.tolist()})
        pseudo.append({"original_id":key,"kind":"pseudo_gt","independent":False,"method":"evaluated detector output; must not be used as accuracy ground truth",
                       "annotations":annotations,"visual_review":reviews.get(key,{"rating":"Uncertain","source":"not_reviewed"})})
    statistics = {"files":inv["files"],"read":sum(r["read_status"]=="OK" for r in rows),"unique_originals":inv["unique_originals"],
        "exact_duplicates":inv["exact_duplicates"],"images_with_multiple_triple_groups":sum(sum(g["format"]=="TRIPLE" for g in d["groups"])>1 for d in detections.values()),
        "detected_face_counts":dict(Counter(len(d["candidates"]) for d in detections.values())),
        "images_with_more_than_three_face":sum(len(d["candidates"])>3 for d in detections.values()),
        "selection_statuses":dict(Counter(s["status"] for s in selections.values())),
        "user_confirmed_groups":sum(s["status"]=="USER_CONFIRMED" for s in selections.values()),
        "correct_user_group_count":sum(s["status"]=="USER_CONFIRMED" for s in selections.values()),
        "ownership_pending_original_numbers":[r["number"] for r in rows if selections[r.get("original_id",f"unreadable-{r['number']}")]["status"]!="USER_CONFIRMED"],
        "ready":sum(r["status"]=="READY" for r in results.values()),
        "quality_statuses":dict(Counter(r["quality_status"] for r in results.values())),
        "images_with_ready_group_preview":sum(any(g["status"]=="READY" for g in r.get("group_results",{}).values()) for r in results.values()),
        "visually_correct_geometry":sum(r.get("rating")=="Correct" and not r.get("stale") for r in reviews.values()),
        "visual_review_statuses":dict(Counter(r.get("rating","Uncertain") for r in reviews.values() if not r.get("stale"))),
        "visual_review_sources":dict(Counter(r.get("source","unknown") for r in reviews.values() if not r.get("stale"))),
        "historical_user_ratings":dict(Counter(r.get("rating","Uncertain") for r in reviews.values() if r.get("source")=="USER_VISUAL_REVIEW")),
        "user_ratings_requiring_recheck":sum(r.get("source")=="USER_VISUAL_REVIEW" and bool(r.get("stale")) for r in reviews.values()),
        "visual_layout_counts":dict(Counter(r["layout_visual"] for r in reviews.values() if r.get("layout_visual") and not r.get("stale"))),
        "layouts_proposed":dict(Counter(g["layout"] or "SINGLE" for d in detections.values() for g in d["groups"])),
        "physical_orientation":dict(Counter(r["orientation_status"] for r in results.values())),
        "display_boundary_statuses":dict(Counter(f.get("display_boundary",{}).get("status","not_recorded") for r in results.values() if r["status"]=="READY" for f in r.get("faces",[]))),
        "display_boundary_uncertain_photo_numbers":[r["number"] for r in results.values() if any(f.get("display_boundary",{}).get("status")=="BOUNDARY_UNCERTAIN" for f in r.get("faces",[]))],
        "quantitative":{"status":"QUALITATIVE ONLY / QUANTITATIVE NOT VERIFIED","independent_gt_images":0,"median_2d":None,"p95_2d":None},
        "stage3_status":"BLOCKED","provenance":{"algorithm_sha":_git_state()[0],"working_tree_dirty":_git_state()[1],
            "python":platform.python_version(),"opencv":cv2.__version__,"selection_policy":"no ownership inference from arrow counts", "capture_plan":"not_pre_registered"}}
    save_json(output/"failure-cases.json",failures)
    save_json(output/"problem-ranking.json",{
        "selection":[f for f in failures if f["category"]=="selection"][:15],
        "geometry":sorted([f for f in failures if f["geometry_issues"]],key=lambda f:max(g.get("fit_error") or 0 for g in f["geometry_issues"]),reverse=True)[:15],
        "orientation":{"statuses":statistics["physical_orientation"],"originals":len(results),"note":"separate limitation, not a geometry or ownership error"}})
    save_json(output/"representative-sampling.json",strata[:15])
    save_json(output/"pseudo-gt-proposals.json",pseudo)
    save_json(output/"metrics.json",statistics)
    save_json(output/"target-selections.json",selections)
    save_json(output/"geometry-results.json",results)
    save_json(output/"visual-reviews.json",reviews)
    from neural_network.archery_ml.geometry.pilot_metrics import independent_metrics
    statistics["quantitative"] = independent_metrics(output/"independent-gt.json",results)
    save_json(output/"metrics.json",statistics)
    baseline = read_json(output/"baseline-metrics.json",{})
    save_json(output/"comparison.json",{"before":baseline,"after":statistics,"note":"Initial pilot implementation versus revised pilot on identical originals; includes candidate/group layer changes, not an isolated detector benchmark. READY is geometry only; ownership and independent accuracy remain unverified"})
    write_report(output,inv,statistics,failures,baseline)
    from neural_network.archery_ml.geometry.pilot_viewer import write_viewer
    write_viewer(output,inv,detections,selections,results,reviews)
    return statistics


def write_report(output,inv,metrics,failures,baseline):
    lines=["# Mobile pilot — реальные исторические фотографии", "", "**Stage 3: BLOCKED. Данные — legacy_exploratory/mobile_dev, не независимый holdout.**", "",
        f"Прочитано {metrics['read']}/{metrics['files']}; уникальных оригиналов {metrics['unique_originals']}; точных дубликатов {metrics['exact_duplicates']}.",
        "Устройство и datetime берутся только из EXIF. Capture sessions и pre-capture plan неизвестны. Orientation=0 отмечена как некорректная; оригинальные storage axes сохранены.", "",
        "| Вопрос | Фактический ответ |", "|---|---|",
        f"| Несколько предложенных triple groups | {metrics['images_with_multiple_triple_groups']} |",
        f"| Пользователь подтвердил свою группу | {metrics['user_confirmed_groups']} |",
        f"| Выбор владельца подтверждён / ожидает подтверждения | {metrics['correct_user_group_count']} / {metrics['ownership_pending_original_numbers']}; AUTO_CONFIDENT не является USER_CONFIRMED |",
        f"| Статусы selection | {metrics['selection_statuses']} |",
        f"| READY для выбранной/автоматически предложенной группы | {metrics['ready']} |",
        f"| Хотя бы один READY group preview | {metrics['images_with_ready_group_preview']} — не утверждение принадлежности |",
        f"| Quality / selection отказы | {metrics['quality_statuses']} |",
        f"| Visual review | {metrics['visual_review_statuses']}; не independent GT |",
        f"| Источники visual review | {metrics['visual_review_sources']}; agent review не подтверждает владельца |",
        f"| Сохранённые оценки владельца прежней версии | {metrics['historical_user_ratings']}; изменённая geometry требует перепроверки: {metrics['user_ratings_requiring_recheck']} |",
        f"| Layout по visual review | {metrics['visual_layout_counts'] or 'не проверено'} |",
        f"| Independent quantitative GT | {metrics['quantitative']['status']} |",
        f"| Physical orientation | {metrics['physical_orientation']} |", "",
        f"| Граница display crop | {metrics['display_boundary_statuses']}; требуют проверки: {metrics['display_boundary_uncertain_photo_numbers']} |", "",
        "Display mask использует наблюдаемый переход синего в бумагу при достаточном угловом покрытии; короткие закрытые дуги интерполируются. При недостатке данных BOUNDARY_UNCERTAIN явно сохраняет номинальную маску. Белые scoring zones не обрезаются по синей границе. Это сегментация для отображения: исходный canonical crop, homography, READY и численная точность координат не изменяются; граница не является independent GT.", "",
        "## Сравнение до / после", "", "| Проверка | Baseline | После |", "|---|---:|---:|",
        f"| READY selected/provisional | {baseline.get('ready','not recorded')} | {metrics['ready']} |",
        f"| Кадры с READY group preview | {baseline.get('images_with_ready_group_preview','not recorded')} | {metrics['images_with_ready_group_preview']} |",
        f"| Не найден ни один candidate | {baseline.get('detected_face_counts',{}).get('0','not recorded')} | {metrics['detected_face_counts'].get(0,0)} |", "",
        "Качество geometry не равно правильности выбора владельца. Рост READY не доказывает 2D accuracy без independent GT. Численные quality thresholds и frozen policy не изменялись.", "",
        "Baseline — первый фактический запуск pilot с исходным detector; сравнение включает изменения поиска кандидатов и группировки. Это development before/after на одних оригиналах, не независимый accuracy benchmark.", "",
        "## Доказанные причины сбоев и исправления", "",
        "- Синий фон/соединение с синей областью поглощало face: concentric red/yellow evidence проверяется до отсечения кандидатов по площади; граница восстанавливается по реально наблюдаемым blue pixels.",
        "- Разорванное синее кольцо давало более плотный внутренний контур вместо внешнего: для blue candidates используется внешний angular envelope; допуски quality gate сохранены.",
        "- Неполная дуга искажала начальный масштаб homography: начальная опора берётся из fitted ellipse вдоль layout axes; физический верх этим не устанавливается.",
        "- Комбинации face соседних листов создавали ложные TRIANGULAR proposals: добавлены layout/continuity evidence и явный выбор владельца. На настоящих TRIANGULAR этот набор не проверен.",
        "- Phantom evaluation_landmark_ids теперь сверяются с настоящими ID аннотаций. READY false positive без landmarks уже корректно учитывался в false acceptance; добавлен regression test без выдуманного исправления.", "",
        "## Как быстро выбрать свои мишени", "", "Откройте review.html через localhost: кнопка Select выбирает предложение, Confirm фиксирует USER_CONFIRMED; можно вручную кликнуть 3 face. Uncertain откладывает, No target отмечает отсутствие своей группы. JSON сохраняется автоматически сервером. Offline-вариант хранит JSON в browser storage и позволяет Save JSON / Connect output directory; после offline выбора повторите batch для пересчёта crops.", "",
        "Для переносов выбора используются только подтверждённый anchor, EXIF device/time и совпадения фона ORB/RANSAC. Это визуальная continuity, а не доказанная capture session. Перенесённая группа получает AUTO_CONFIDENT и не выдаётся за новое пользовательское подтверждение. Количество стрел не используется.", "",
        "Индексы face по умолчанию PROVISIONAL_IMAGE_LAYOUT_ORDER. Для физических IDs выберите face в порядке 0/1/2 и подтвердите соответствующий checkbox. Без независимого ориентира угловая ориентация всё равно UNANCHORED.", "",
        "[Все исходные фотографии](contact-sheet.html) · [Интерактивный review](review.html) · [12–15 репрезентативных кадров](representative-sampling.json) · [Pseudo-GT предложения](pseudo-gt-proposals.json)", "",
        "[Визуальный аудит и примеры до/после](visual-audit.html). Agent review относится только к явно указанным preview candidate IDs и общему виду колец/центров на montage scale. Это не проверка ownership, физических face IDs, пиксельной точности или независимая quantitative GT.", "",
        "Pseudo-GT построен проверяемым детектором и не является измерением ошибки. Visual review подтверждает только внешний вид. Настоящие partial 2D errors можно считать через --independent-gt, где указаны независимый метод, reference/revision, разные annotator/reviewer, SHA оригинала, group IDs и ≥4 неколлинеарных matched landmarks каждого измеряемого face.", "",
        "## Топ проблем", "", "| № | Категория | Статус | Причины |", "|---:|---|---|---|"]
    for issue in failures[:15]:
        lines.append(f"| {issue['number']} | {issue['category']} | {issue['quality_status']} | {', '.join(str(v) for v in issue['diagnostics'])} |")
    lines += ["", "## Все фотографии", "", "| № | Файл | EXIF | Устройство | Capture time | Session |", "|---:|---|---:|---|---|---|"]
    for row in inv["images"]:
        lines.append(f"| {row['number']} | {row['filename']} | {row.get('exif_orientation','unknown')} | {row.get('device_model','unknown')} | {row.get('capture_datetime','unknown')} | unknown |")
    lines += ["", "Для окончательного Stage 3 нужны независимый mobile_real_test, реальный GT audit, проверенная физическая калибровка и утверждённая frozen policy. Этот исторический набор не закрывает эти условия.", "",
              "Ни детекция impacts, ни обучение нейросети здесь не выполнялись. Частные фотографии и их производные находятся только в gitignored output.", ""]
    (output/"REPORT.md").write_text("\n".join(lines))


def _source_digest():
    root = Path(__file__).resolve().parents[3]
    paths = sorted((root/"neural_network/archery_ml").rglob("*.py"))
    paths += sorted((root/"dataset_tools/boards").glob("*.json"))
    paths += [root/"dataset_tools/stage3_geometry_policy.json",root/"requirements.txt",root/".python-version"]
    return hashlib.sha256(b"".join(p.relative_to(root).as_posix().encode()+b"\0"+p.read_bytes()+b"\0" for p in paths)).hexdigest()


def _stable(value):
    if isinstance(value,dict):
        return {key:_stable(item) for key,item in value.items() if key!="processing_seconds"}
    if isinstance(value,list):
        return [_stable(item) for item in value]
    return value


def _digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def batch(input_dir,output,selections_path=None,inventory_only=False,baseline=False,verify_replay=False, *, replay_inputs=None):
    output = Path(output).resolve()
    previous_run = read_json(output/"run-manifest.json",{})
    if verify_replay and not previous_run:
        raise ValueError("verify-replay requires an existing completed run")
    if verify_replay:
        if inventory_only or baseline:
            raise ValueError("verify-replay cannot be combined with inventory-only or baseline")
        # Recompute into a sibling directory. Never mix a failed replay with the
        # original evidence, even when loading or processing raises an exception.
        with tempfile.TemporaryDirectory(prefix=f".{output.name}-replay-",dir=output.parent) as tmp:
            scratch = Path(tmp)
            for name in ("target-selections.json","visual-reviews.json","visual-review-history.json",
                    "board-template.json","independent-gt.json","baseline-metrics.json"):
                if (output/name).is_file():
                    shutil.copy2(output/name,scratch/name)
            for name,content in (replay_inputs or {}).items():
                (scratch/name).write_bytes(content)
            statistics = batch(input_dir,scratch,selections_path)
            record = read_json(scratch/"run-manifest.json",{})
            keys = ("algorithm_content_sha256","inputs_sha256","results_sha256","dependencies")
            differences = [key for key in keys if record.get(key)!=previous_run.get(key)]
            evidence = {"status":"MISMATCH" if differences else "VERIFIED","differences":differences,
                "previous":previous_run,"current":record,"excluded_fields":["processing_seconds"]}
            if differences:
                save_json(output.parent/(output.name+"-replay-mismatch.json"),evidence)
                raise ValueError("pilot replay differs: "+", ".join(differences))
            save_json(output/"reproducibility.json",evidence)
            return statistics
    source_digest = _source_digest()
    algorithm_sha, dirty = _git_state()
    if baseline and (output/"baseline-metrics.json").exists():
        raise ValueError("baseline already exists; use a new output revision")
    inv = inventory(input_dir,output)
    if inventory_only:
        return inv
    previous = read_json(Path(selections_path) if selections_path else output/"target-selections.json",{})
    detector,builder,selector = TargetCandidateDetector(),TargetGroupBuilder(),TargetGroupSelector()
    detections,selections,results = {},{},{}
    for row in inv["images"]:
        key = row.get("original_id",f"unreadable-{row['number']}")
        if key in detections:
            continue
        start = time.monotonic()
        if row["read_status"] == "OK":
            image = np.asarray(Image.open(output/row["rgb_path"]).convert("RGB"))
            candidates = detector.detect(image)
            for candidate in candidates:
                x0,y0,x1,y1=candidate["bbox"]
                corners=project([[x0,y0],[x1,y0],[x1,y1],[x0,y1]],row["rgb_to_raw"])
                candidate["bbox_original_raw"]=np.r_[corners.min(axis=0),corners.max(axis=0)].tolist()
                candidate["center_original_raw"]=project([candidate["center"]],row["rgb_to_raw"])[0].tolist()
            groups = builder.build(candidates,image)
            detections[key] = {"number":row["number"],"candidates":candidates,"groups":groups,"fingerprint":fingerprint(candidates)}
            selections[key] = selector.select(candidates,groups,previous.get(key))
            results[key] = process_frame(row,detections[key],selections[key],output,artifact_prefix="baseline-" if baseline else "")
        else:
            detections[key] = {"number":row["number"],"candidates":[],"groups":[],"fingerprint":fingerprint([])}
            selections[key] = {"status":"NO_GROUP","source":"INPUT_ERROR","candidate_ids":[]}
            results[key] = {"original_id":key,"number":row["number"],"status":"REJECTED","quality_status":"INPUT_ERROR","faces":[],"diagnostics":[row.get("error")],"orientation_status":"UNANCHORED"}
        results[key]["processing_seconds"] = time.monotonic()-start
        print(f'#{row["number"]:02} faces={len(detections[key]["candidates"])} selection={selections[key]["status"]} geometry={results[key]["quality_status"]}',flush=True)
    save_json(output/"detections.json",detections)
    statistics = reports(output,inv,detections,selections,results)
    if baseline:
        save_json(output/"baseline-geometry-results.json",results)
        save_json(output/"baseline-metrics.json",statistics)
        save_json(output/"baseline-detections.json",detections)
    if _source_digest()!=source_digest or _git_state()!=(algorithm_sha,dirty):
        raise ValueError("algorithm changed during pilot run; rerun on a frozen revision")
    binding = {"originals":[r.get("original_sha256") for r in inv["images"]],"selections":selections,
        "external_files":{name:hashlib.sha256((output/name).read_bytes()).hexdigest()
            for name in ("board-template.json","independent-gt.json") if (output/name).exists()}}
    record = {"algorithm_sha":algorithm_sha,"working_tree_dirty":dirty,"algorithm_content_sha256":source_digest,
        "inputs_sha256":_digest(binding),"results_sha256":_digest(_stable({"detections":detections,"results":results})),
        "dependencies":{"python":platform.python_version(),"numpy":np.__version__,"opencv":cv2.__version__,"pillow":PIL.__version__}}
    save_json(output/"reproducibility.json",{"status":"NOT_CHECKED",
        "differences":[],"previous":previous_run,"current":record,"excluded_fields":["processing_seconds"]})
    save_json(output/"run-manifest.json",record)
    return statistics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",type=Path,required=True)
    parser.add_argument("--output",type=Path,default=Path("data/reports/mobile-pilot-58"))
    parser.add_argument("--selection-mode",choices=["interactive","automatic"],default="interactive")
    parser.add_argument("--selections",type=Path)
    parser.add_argument("--inventory-only",action="store_true")
    parser.add_argument("--baseline",action="store_true")
    parser.add_argument("--verify-replay",action="store_true")
    parser.add_argument("--independent-gt",type=Path)
    parser.add_argument("--board-template",type=Path)
    parser.add_argument("--serve",action="store_true")
    parser.add_argument("--serve-only",action="store_true")
    parser.add_argument("--port",type=int,default=8765)
    args = parser.parse_args(argv)
    replay_inputs = {}
    if args.board_template:
        from neural_network.archery_ml.geometry.calibration import BoardTemplate
        board = BoardTemplate.load(args.board_template)
        destination = args.output/"board-template.json"
        content = args.board_template.read_bytes()
        if args.verify_replay:
            replay_inputs[destination.name] = content
        elif destination.exists() and destination.read_bytes()!=content:
            parser.error("do not overwrite board template; use a new output revision")
        else:
            args.output.mkdir(parents=True,exist_ok=True)
            destination.write_bytes(content)
    if args.independent_gt:
        content=args.independent_gt.read_bytes()
        destination=args.output/"independent-gt.json"
        if args.verify_replay:
            replay_inputs[destination.name] = content
        elif destination.exists() and destination.read_bytes()!=content:
            parser.error("do not overwrite independent GT; use a new output revision")
        else:
            args.output.mkdir(parents=True,exist_ok=True)
            destination.write_bytes(content)
    if args.serve_only:
        from neural_network.archery_ml.geometry.pilot_viewer import serve
        serve(args.output,args.port)
        return 0
    result = batch(args.input,args.output,args.selections,args.inventory_only,args.baseline,args.verify_replay,replay_inputs=replay_inputs)
    print(json.dumps({k:v for k,v in result.items() if k!="images"},indent=2,ensure_ascii=False))
    if args.serve and not args.inventory_only:
        from neural_network.archery_ml.geometry.pilot_viewer import serve
        serve(args.output,args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
