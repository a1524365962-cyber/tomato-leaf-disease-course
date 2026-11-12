"""Standard-library unit tests of the actual frozen restore function.

The image decoder and HTTP transport are test doubles; no download is performed.
Real-image decoding is covered by the full original-data audit, not these tests.
"""
import ast
import concurrent.futures as cf
import hashlib
import io
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import Mock
import urllib.parse

SOURCE=Path(__file__).with_name('prepare_experiment.py')
COMMIT='7f7ecc7e1eaca78107e3affe7cb5abd9427e139a'


class FakeImage:
    width=2
    height=2
    def __enter__(self): return self
    def __exit__(self,*args): return False
    def convert(self,*args): return self
    def load(self): return None
    def tobytes(self): return b'pixels'


class FrozenRestoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.dest=self.root/'data/class/a.jpg';self.dest.parent.mkdir(parents=True)
        self.content=b'original-file'
        self.row={'path':'data/class/a.jpg','source_path':'raw/color/class/a.jpg',
                  'sha256':hashlib.sha256(self.content).hexdigest(),
                  'pixel_sha256':hashlib.sha256(b'pixels').hexdigest(),'width':2,'height':2}
        self.manifest=self.root/'manifest.json'
        self.http=Mock(return_value=io.BytesIO(self.content))
        node=next(n for n in ast.parse(SOURCE.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='restore_frozen_manifest')
        module=ast.Module(body=[node],type_ignores=[])
        namespace={'json':json,'ROOT':self.root,'hashlib':hashlib,'io':io,'cf':cf,
                   'Image':types.SimpleNamespace(open=lambda _:FakeImage()),
                   'urllib':types.SimpleNamespace(parse=urllib.parse,request=types.SimpleNamespace(urlopen=self.http)),
                   'time':types.SimpleNamespace(sleep=lambda _:None),'print':lambda *a,**k:None}
        exec(compile(module,str(SOURCE),'exec'),namespace)
        self.restore=namespace['restore_frozen_manifest']

    def write_manifest(self,commit=COMMIT):
        self.manifest.write_text(json.dumps({'commit':commit,'rows':[self.row]},indent=2))
        return self.manifest.read_bytes()

    def test_valid_cache_preserves_manifest_bytes(self):
        original=self.write_manifest();self.dest.write_bytes(self.content)
        self.restore(self.manifest)
        self.assertEqual(self.manifest.read_bytes(),original);self.http.assert_not_called()

    def test_corrupt_cache_is_not_overwritten(self):
        original=self.write_manifest();self.dest.write_bytes(b'corrupt')
        with self.assertRaises(ValueError): self.restore(self.manifest)
        self.assertEqual(self.dest.read_bytes(),b'corrupt')
        self.assertEqual(self.manifest.read_bytes(),original);self.http.assert_not_called()

    def test_missing_file_is_verified_then_restored(self):
        self.write_manifest();self.restore(self.manifest)
        self.assertEqual(self.dest.read_bytes(),self.content)

    def test_bad_download_is_not_written(self):
        self.write_manifest();self.http.return_value=io.BytesIO(b'wrong')
        with self.assertRaises(ValueError): self.restore(self.manifest)
        self.assertFalse(self.dest.exists())

    def test_path_escape_is_rejected(self):
        self.row['path']='../outside.jpg';self.write_manifest()
        with self.assertRaises(ValueError): self.restore(self.manifest)
        self.http.assert_not_called()

    def test_pixel_mismatch_is_rejected(self):
        self.row['pixel_sha256']='0'*64;self.write_manifest();self.dest.write_bytes(self.content)
        with self.assertRaises(ValueError): self.restore(self.manifest)

    def test_wrong_source_commit_is_rejected(self):
        self.write_manifest('wrong')
        with self.assertRaises(ValueError): self.restore(self.manifest)


if __name__=='__main__': unittest.main()
