"""Adversarial finalization regressions; these fixtures do not attest real accuracy."""
import copy
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from neural_network.archery_ml.contracts import DatasetSample, ImageGeometryAnnotation
from neural_network.archery_ml.data.view import DatasetView
from neural_network.archery_ml.geometry.audit import require_holdout_isolation
from neural_network.archery_ml.geometry.evaluate import GeometryEvaluator, _landmark_decomposition
from neural_network.archery_ml.geometry.pilot_metrics import independent_metrics
from neural_network.archery_ml.geometry.selected_geometry import SelectedTargetGeometryPipeline
from neural_network.archery_ml.geometry.selection import TargetCandidateDetector, TargetGroupBuilder, TargetGroupSelector, fingerprint
from neural_network.tests.test_stage3_calibration import fixture
from neural_network.archery_ml.geometry.calibration import TargetOrientationCalibrator
from neural_network.tests import test_geometry_evaluation as evaluation_fixtures
from neural_network.tests.test_stage3_pilot import scene


class FinalizationTest(unittest.TestCase):
    def test_error_components_separate_scale_center_rotation_without_correcting_predictions(self):
        expected = np.array([[-.4,-.4],[.4,-.4],[.4,.4],[-.4,.4]])
        angle = np.deg2rad(17.)
        rotation = np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
        predicted = expected @ rotation.T * 1.08 + [.03,.04]
        before = predicted.copy()
        result = _landmark_decomposition(predicted,expected)
        self.assertAlmostEqual(.08,result['scale_bias_fraction'])
        self.assertAlmostEqual(17.,result['orientation_error_deg'])
        self.assertAlmostEqual(.05,result['affine_center_offset_norm'])
        self.assertAlmostEqual(1.,result['affine_axis_ratio'])
        self.assertLess(result['affine_residual_p95_norm'],1e-12)
        np.testing.assert_array_equal(before,predicted)

    def test_downstream_normalization_failure_cannot_leave_ready_quality(self):
        image = scene(1)
        candidates = TargetCandidateDetector().detect(image)
        group = next(g for g in TargetGroupBuilder().build(candidates,image) if g['format']=='TRIPLE')
        with patch('neural_network.archery_ml.geometry.selected_geometry.TargetNormalizer.normalize',
                side_effect=ValueError('normalization quality rejected')):
            result = SelectedTargetGeometryPipeline().run(image,candidates,group)
        self.assertEqual('REJECTED',result['status'])
        self.assertEqual('GEOMETRY_UNCERTAIN',result['quality_status'])
        self.assertIn('normalization quality rejected',result['diagnostics'])

    def test_batch_replay_binds_originals_selections_algorithm_and_artifacts(self):
        from neural_network.archery_ml.geometry.pilot import batch
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root/'input'
            source.mkdir()
            Image.fromarray(scene(1)).save(source/'target.png')
            output = root/'output'
            batch(source,output)
            batch(source,output,verify_replay=True)
            evidence = json.loads((output/'reproducibility.json').read_text())
            self.assertEqual('VERIFIED',evidence['status'])
            run = json.loads((output/'run-manifest.json').read_text())
            run['results_sha256'] = 'tampered'
            (output/'run-manifest.json').write_text(json.dumps(run))
            with self.assertRaisesRegex(ValueError,'replay differs'):
                batch(source,output,verify_replay=True)

    def test_saved_selection_cannot_repeat_or_invent_candidate_ids(self):
        image = scene(1)
        candidates = TargetCandidateDetector().detect(image)
        groups = TargetGroupBuilder().build(candidates, image)
        confirmed = {**next(g for g in groups if g['format'] == 'TRIPLE'),
            'status':'USER_CONFIRMED', 'source':'USER', 'detection_fingerprint':fingerprint(candidates)}
        for ids in (['f0', 'f0', 'f0'], ['missing'], ['f0', 'f1']):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                TargetGroupSelector().select(candidates, groups, {**confirmed, 'candidate_ids':ids})
        self.assertEqual(confirmed, TargetGroupSelector().select(candidates, groups, confirmed))

    def test_saved_selection_rejects_inconsistent_config_and_stale_fingerprint(self):
        image = scene(1)
        candidates = TargetCandidateDetector().detect(image)
        groups = TargetGroupBuilder().build(candidates, image)
        confirmed = {**next(g for g in groups if g['format'] == 'TRIPLE'),
            'status':'USER_CONFIRMED', 'source':'USER', 'detection_fingerprint':fingerprint(candidates)}
        for change in ({'format':'SINGLE'}, {'layout':None}, {'minimum_scoring_zone':2}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                TargetGroupSelector().select(candidates, groups, {**confirmed, **change})
        result = TargetGroupSelector().select(candidates, groups, {**confirmed, 'detection_fingerprint':'old'})
        self.assertEqual('AMBIGUOUS', result['status'])
        self.assertEqual([], result['candidate_ids'])

    def test_partial_gt_rejects_duplicate_originals_and_partial_point_duplication(self):
        points = [[-.2,-.2],[.2,-.2],[.2,.2],[-.2,.2]]
        face = {'face_index':0, 'image_points_px':(np.array(points)*256+255.5).tolist(), 'canonical_points':points}
        item = {'original_sha256':'sample', 'candidate_ids':['f0'], 'coordinate_space':'original_raw_px', 'faces':[face]}
        payload = {'kind':'independent_gt', 'method':'physical_measurement', 'revision':'r1', 'reference':'survey',
            'annotator':'one', 'reviewer':'two', 'images':[item]}
        results = {'sample':{'status':'READY', 'candidate_ids':['f0'], 'orientation_status':'UNANCHORED',
            'faces':[{'face_index':0, 'original_to_canonical':np.eye(3).tolist()}]}}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'gt.json'
            path.write_text(json.dumps(payload))
            self.assertEqual(1, independent_metrics(path, results)['independent_gt_images'])
            duplicated = copy.deepcopy(payload)
            duplicated['images'].append(copy.deepcopy(item))
            path.write_text(json.dumps(duplicated))
            with self.assertRaisesRegex(ValueError, 'duplicate.*original'):
                independent_metrics(path, results)
            duplicated = copy.deepcopy(payload)
            for key in ('image_points_px', 'canonical_points'):
                duplicated['images'][0]['faces'][0][key].extend([face[key][0]]*100)
            path.write_text(json.dumps(duplicated))
            with self.assertRaisesRegex(ValueError, 'unique'):
                independent_metrics(path, results)

    def test_evaluator_rejects_repeated_good_landmark_with_new_annotation_identity(self):
        example = evaluation_fixtures.GeometryEvaluationTest()
        image, _ = example._fixture()
        points = ((.4,.4),(.6,.4),(.6,.6),(.4,.6),(.4,.4))
        expected = ((-.2,-.2),(.2,-.2),(.2,.2),(-.2,.2),(-.2,-.2))
        annotation = ImageGeometryAnnotation('landmark', points, face_index=0, canonical_points=expected)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'target.png'
            Image.fromarray(image).save(path)
            sample = DatasetSample('duplicate', path, None, geometry_annotations=(annotation,))
            with self.assertRaisesRegex(ValueError, 'duplicate evaluation landmark'):
                GeometryEvaluator().evaluate(DatasetView.from_samples([sample]), example._config())

    def test_full_manifest_isolation_catches_cross_source_session_and_original(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = [root/'one.png', root/'two.png']
            for i, path in enumerate(paths):
                Image.fromarray(np.random.default_rng(i).integers(0,256,(64,64,3),dtype=np.uint8)).save(path)
            samples = [DatasetSample(str(i), path, None, split=split, source_id=str(i),
                group_id=str(i), capture_metadata={'capture_session_id':'shared', 'original_sha256':str(i)})
                for i, (path, split) in enumerate(zip(paths, ('train', 'mobile_real_test')))]
            with self.assertRaisesRegex(ValueError, 'cross-split leakage'):
                require_holdout_isolation(DatasetView.from_samples(samples))
            samples[0] = replace(samples[0], capture_metadata={'capture_session_id':'other', 'original_sha256':'1'})
            with self.assertRaisesRegex(ValueError, 'cross-split leakage'):
                require_holdout_isolation(DatasetView.from_samples(samples))

    def test_selected_pipeline_physically_anchors_only_with_observed_board(self):
        board, image, _, _ = fixture('vertical', scale=1.4)
        candidates = TargetCandidateDetector().detect(image)
        group = next(g for g in TargetGroupBuilder().build(candidates,image) if g['format']=='TRIPLE')
        pipeline = SelectedTargetGeometryPipeline(TargetOrientationCalibrator(board))
        result = pipeline.run(image,candidates,group)
        self.assertEqual('READY', result['status'], result['diagnostics'])
        self.assertEqual('ANCHORED', result['orientation_status'], result.get('calibration'))
        self.assertEqual('PHYSICAL_BOARD_MARKERS', result['face_index_reference'])
        plain = pipeline.run(scene(1), TargetCandidateDetector().detect(scene(1)), group)
        self.assertEqual('UNANCHORED', plain['orientation_status'])

