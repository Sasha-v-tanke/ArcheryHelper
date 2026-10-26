import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from neural_network.archery_ml.geometry.center_review import CenterReviewStore, ConflictError


class CenterReviewTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.pilot = self.root/'pilot'
        self.pilot.mkdir()
        self.image = self.pilot/'image.png'
        Image.new('RGB',(30,20),'white').save(self.image)
        self.sha = 'a'*64
        row = {'original_id':self.sha,'read_status':'OK','number':1,'filename':'fixture.jpg',
            'rgb_path':'image.png','rgb_sha256':hashlib.sha256(self.image.read_bytes()).hexdigest(),
            'rgb_size':[30,20],'rgb_to_raw':[[0,1,0],[-1,0,29],[0,0,1]],'exif_orientation_valid':True}
        (self.pilot/'inventory.json').write_text(json.dumps({'images':[row]}))
        (self.pilot/'detections.json').write_text(json.dumps({self.sha:{'candidates':[{'id':'f0','center':[10,10]}]}}))
        (self.pilot/'geometry-results.json').write_text(json.dumps({self.sha:{'candidate_ids':['f0'],
            'faces':[{'face_index':0,'center_rgb':[11,9]}],'ownership_confirmed':True}}))
        self.store = CenterReviewStore(self.pilot,self.root/'review')

    def payload(self):
        row = self.store.state()['images'][0]
        return {'original_id':self.sha,'binding':row['binding'],'revision':0,
            'status':'CONFIRMED','points':[{'id':'f0','x':10,'y':5}]}

    def test_confirm_edit_resume_history_and_exif_without_changing_source(self):
        before = {p.name:p.read_bytes() for p in self.pilot.iterdir()}
        self.assertFalse(self.store.file.exists())
        self.assertEqual([{'id':'f0','x':11,'y':9}],self.store.state()['images'][0]['suggested_centers'])
        record = self.store.save(self.payload())
        self.assertEqual([5.,19.],record['points'][0]['original_raw_px'])
        self.assertFalse(record['independent_gt'])
        again = CenterReviewStore(self.pilot,self.root/'review')
        self.assertEqual(record,again.state()['images'][0]['review'])
        changed = self.payload()
        changed.update(revision=1,status='DRAFT',from_blank=True)
        changed['points'][0]['x'] = 14
        current = again.save(changed)
        self.assertEqual(2,current['revision'])
        self.assertEqual('manual_from_blank',current['method'])
        history = json.loads((self.root/'review/center-review-history.jsonl').read_text())
        self.assertEqual(record,history)
        self.assertEqual(before,{p.name:p.read_bytes() for p in self.pilot.iterdir()})

    def test_stale_revision_and_source_binding_cannot_overwrite_labels(self):
        self.store.save(self.payload())
        before = self.store.file.read_bytes()
        with self.assertRaises(ConflictError):
            self.store.save(self.payload())
        changed = self.payload()
        changed.update(revision=1,binding='wrong')
        with self.assertRaises(ConflictError):
            self.store.save(changed)
        self.assertEqual(before,self.store.file.read_bytes())

    def test_invalid_centers_and_empty_confirmation_do_not_write(self):
        for points in ([],[{'id':'f0','x':float('nan'),'y':5}],
                [{'id':'f0','x':30,'y':5}],[{'id':'f0','x':True,'y':5}],
                [{'id':'f0','x':1,'y':2},{'id':'f0','x':2,'y':3}]):
            payload = self.payload()
            payload['points'] = points
            with self.assertRaises(ValueError):
                self.store.save(payload)
        self.assertFalse(self.store.file.exists())

    def test_modified_image_and_output_inside_pilot_are_rejected(self):
        Image.new('RGB',(30,20),'red').save(self.image)
        with self.assertRaisesRegex(ValueError,'image changed'):
            self.store.save(self.payload())
        self.assertFalse(self.store.file.exists())
        with self.assertRaisesRegex(ValueError,'separate'):
            CenterReviewStore(self.pilot,self.pilot/'labels')

    def test_proposals_update_unreviewed_centers_without_changing_binding_or_human_labels(self):
        before = self.store.state()['images'][0]
        proposals = {'revision':'test-refinement','images':{self.sha:{'original_sha256':self.sha,
            'image_sha256':before['image_sha256'],'points':[{'id':'f0','x':13,'y':8}]}}}
        (self.store.output/'center-proposals.json').write_text(json.dumps(proposals))
        proposed = self.store.state()['images'][0]
        self.assertTrue(proposed['automatic_proposal'])
        self.assertEqual(before['binding'],proposed['binding'])
        self.assertEqual(13,proposed['suggested_centers'][0]['x'])
        self.assertIsNone(proposed['review'])
        record = self.store.save(self.payload())
        saved = self.store.file.read_bytes()
        proposals['images'][self.sha]['points'][0]['x'] = 19
        (self.store.output/'center-proposals.json').write_text(json.dumps(proposals))
        current = self.store.state()['images'][0]
        self.assertEqual(record,current['review'])
        self.assertFalse(current.get('automatic_proposal',False))
        self.assertFalse(current['stale'])
        self.assertEqual(saved,self.store.file.read_bytes())
