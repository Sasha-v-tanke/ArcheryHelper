"""PR #10 counterexamples: selected evaluation, tracking resume and isolated replay."""
from contextlib import redirect_stdout
from dataclasses import replace
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from neural_network.archery_ml.contracts import DatasetSample, ImageGeometryAnnotation, save_manifest
from neural_network.archery_ml.data.view import DatasetView
from neural_network.archery_ml.geometry.cli import main as evaluate_main
from neural_network.archery_ml.geometry.evaluate import GeometryEvaluator
from neural_network.archery_ml.geometry.pilot import batch
from neural_network.archery_ml.geometry.pilot_viewer import update
from neural_network.archery_ml.geometry.selection import TargetCandidateDetector, TargetGroupBuilder, TargetGroupSelector, fingerprint
from neural_network.archery_ml.targets import TargetConfig, TargetFormat, TenRingMode, TripleLayout
from neural_network.tests.test_stage3_pilot import scene


def snapshot(root):
    return {p.relative_to(root).as_posix():p.read_bytes() for p in root.rglob('*') if p.is_file()}


def confirmed(group, candidates):
    return {**group,'status':'USER_CONFIRMED','source':'USER','ownership_confirmed':True,
        'detection_fingerprint':fingerprint(candidates),'minimum_scoring_zone':6,'remember':True}


class ReviewFixesTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)
        self.config = TargetConfig(TargetFormat.TRIPLE,6,TenRingMode.RECURVE,1,TripleLayout.VERTICAL)

    def sample(self):
        image = scene()
        path = self.root/'targets.png'
        Image.fromarray(image).save(path)
        candidates = TargetCandidateDetector().detect(image)
        groups = [g for g in TargetGroupBuilder().build(candidates,image) if g['format']=='TRIPLE']
        middle = next(g for g in groups if g['candidate_ids']==['f1','f4','f7'])
        annotations = []
        canonical = ((-.2,-.2),(.2,-.2),(.2,.2),(-.2,.2))
        for index,y in enumerate((130,325,520)):
            annotations.append(ImageGeometryAnnotation('landmark',
                tuple(((300+x*140)/600,(y+dy*140)/650) for x,dy in canonical),
                face_index=index,canonical_points=canonical,annotation_quality='reviewed',uncertainty_norm=.002))
        sample = DatasetSample('selected',path,None,split='mobile_dev',geometry_annotations=tuple(annotations))
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        return sample,image,candidates,groups,sha,confirmed(middle,candidates)

    def test_evaluation_measures_confirmed_middle_column_through_pilot_pipeline(self):
        sample,image,candidates,groups,sha,choice = self.sample()
        report = GeometryEvaluator(selected_targets={sha:choice}).evaluate(DatasetView.from_samples([sample]),self.config).to_dict()
        result = report['sample_results'][0]
        self.assertEqual('READY',result['status'])
        self.assertEqual({'USER_CONFIRMED':1},report['selection_coverage'])
        pipeline = result['selection']['pipeline_result']
        self.assertEqual(choice['candidate_ids'],pipeline['candidate_ids'])
        self.assertEqual(6,len(pipeline['foreign_candidates_masked']))
        self.assertTrue(all(abs(f['center_rgb'][0]-300)<2 for f in pipeline['faces']))
        self.assertLess(result['point_error_2d']['max'],.03)
        # Changing GT cannot change which group the geometry path processes.
        changed = replace(sample,geometry_annotations=tuple(replace(a,
            canonical_points=tuple((x+.1,y) for x,y in a.canonical_points)) for a in sample.geometry_annotations))
        again = GeometryEvaluator(selected_targets={sha:choice}).evaluate(DatasetView.from_samples([changed]),self.config).to_dict()
        self.assertEqual(pipeline,again['sample_results'][0]['selection']['pipeline_result'])
        right = next(g for g in groups if g['candidate_ids']==['f2','f5','f8'])
        wrong = GeometryEvaluator(selected_targets={sha:confirmed(right,candidates)}).evaluate(DatasetView.from_samples([sample]),self.config).to_dict()
        self.assertGreater(wrong['sample_results'][0]['point_error_2d']['max'],1.)

    def test_missing_stale_and_invalid_selection_remain_in_denominator(self):
        sample,image,candidates,groups,sha,choice = self.sample()
        for changes in (None,{'detection_fingerprint':'old'},{'candidate_ids':['invented']},
                {'status':'AUTO_CONFIDENT','source':'UNIQUE_GEOMETRIC_GROUP'},{'layout':'TRIANGULAR'}):
            selections = {} if changes is None else {sha:{**choice,**changes}}
            report = GeometryEvaluator(selected_targets=selections).evaluate(DatasetView.from_samples([sample]),self.config).to_dict()
            self.assertEqual(1,report['samples'])
            self.assertEqual(0,report['ready_samples'])
            self.assertEqual('GROUP_SELECTION_REQUIRED',report['sample_results'][0]['status'])
            self.assertEqual(0,report['sample_results'][0]['detected_face_count'])

    def test_cli_accepts_and_pins_selection_sidecar(self):
        sample,image,candidates,groups,sha,choice = self.sample()
        manifest = self.root/'manifest.json'
        save_manifest(manifest,[sample])
        selections = self.root/'selections.json'
        selections.write_text(json.dumps({sha:choice}))
        output = self.root/'evaluation.json'
        with redirect_stdout(io.StringIO()):
            code = evaluate_main(['--manifest',str(manifest),'--selections',str(selections),
                '--format','TRIPLE','--triple-layout','VERTICAL','--minimum-scoring-zone','6','--output',str(output)])
        self.assertEqual(0,code)
        report = json.loads(output.read_text())
        self.assertEqual('selected-target-v1',report['provenance']['evaluation_pipeline'])
        self.assertEqual(hashlib.sha256(selections.read_bytes()).hexdigest(),report['provenance']['selections_sha256'])
        self.assertEqual(1,report['ready_samples'])

    def test_matrix_runner_forwards_the_same_selection_file_to_every_configuration(self):
        from dataset_tools.mobile.validate_all import run
        sample,image,candidates,groups,sha,choice = self.sample()
        manifest = self.root/'manifest.json'
        save_manifest(manifest,[sample])
        selections = self.root/'selections.json'
        selections.write_text(json.dumps({sha:choice}))
        calls = []
        def evaluate(args):
            calls.append(args)
            raise ValueError('fixture: stop after verifying arguments')
        with patch('dataset_tools.mobile.validate_all.evaluate_main',side_effect=evaluate):
            run(manifest,Path('dataset_tools/stage3_geometry_policy.json'),self.root/'matrix',selections=selections)
        self.assertEqual(18,len(calls))
        for args in calls:
            self.assertEqual('selected',args[args.index('--selection-mode')+1])
            self.assertEqual(str(selections),args[args.index('--selections')+1])

    def test_confirm_track_save_and_batch_restore_do_not_confirm_ownership(self):
        source = self.root/'input'
        source.mkdir()
        image = np.random.default_rng(42).integers(60,130,(1000,1000,3),dtype=np.uint8)
        image[175:825,200:800] = scene()
        for index in (1,2):
            exif = Image.Exif()
            exif[272] = 'fixture-camera'
            exif[306] = f'2026:10:26 12:00:0{index}'
            Image.fromarray(image).save(source/f'{index}.png',exif=exif)
        output = self.root/'pilot'
        with redirect_stdout(io.StringIO()):
            batch(source,output)
            inventory = json.loads((output/'inventory.json').read_text())
            first,second = [row['original_id'] for row in inventory['images']]
            detections = json.loads((output/'detections.json').read_text())
            det = detections[first]
            middle = next(g for g in det['groups'] if g['candidate_ids']==['f1','f4','f7'])
            result = update(output,first,'selection',confirmed(middle,det['candidates']))
            tracked = result['selections'][second]
            self.assertEqual('VISUAL_TRACKING_FROM_USER',tracked['source'])
            self.assertEqual('AUTO_CONFIDENT',tracked['status'])
            self.assertFalse(tracked['ownership_confirmed'])
            batch(source,output)
        stored = json.loads((output/'target-selections.json').read_text())[second]
        self.assertEqual(tracked,stored)
        self.assertFalse(stored['ownership_confirmed'])
        for changes in ({'candidate_ids':['invented']},{'format':'SINGLE'},{'layout':None}):
            with self.assertRaises(ValueError):
                TargetGroupSelector().select(detections[second]['candidates'],detections[second]['groups'],{**stored,**changes})
        stale = TargetGroupSelector().select(detections[second]['candidates'],detections[second]['groups'],
            {**stored,'detection_fingerprint':'old'})
        self.assertEqual('AMBIGUOUS',stale['status'])

    def test_failed_replay_preserves_every_original_artifact(self):
        source = self.root/'input'
        source.mkdir()
        Image.fromarray(scene(2)).save(source/'targets.png')
        output = self.root/'pilot'
        with redirect_stdout(io.StringIO()):
            batch(source,output)
        before = snapshot(output)
        detections = json.loads((output/'detections.json').read_text())
        sha,det = next(iter(detections.items()))
        group = next(g for g in det['groups'] if g['format']=='TRIPLE')
        changed = self.root/'changed-selections.json'
        changed.write_text(json.dumps({sha:confirmed(group,det['candidates'])}))
        with redirect_stdout(io.StringIO()),self.assertRaisesRegex(ValueError,'replay differs'):
            batch(source,output,changed,verify_replay=True)
        self.assertEqual(before,snapshot(output))
        mismatch = json.loads((self.root/'pilot-replay-mismatch.json').read_text())
        self.assertEqual('MISMATCH',mismatch['status'])
        self.assertIn('inputs_sha256',mismatch['differences'])
        self.assertFalse(list(self.root.glob('.pilot-replay-*')))
        # An exception before comparison also cannot leave mixed evidence.
        with patch('neural_network.archery_ml.geometry.pilot.process_frame',side_effect=ValueError('injected')):
            with self.assertRaisesRegex(ValueError,'injected'):
                batch(source,output,verify_replay=True)
        self.assertEqual(before,snapshot(output))
