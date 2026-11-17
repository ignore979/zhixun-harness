"""OCI metadata fixtures only; no official images, industrial records or labels are fabricated."""
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from zhixun.data import ROOT


def load_helper():
    spec = importlib.util.spec_from_file_location(
        'offline_base_helper', ROOT / 'submission/fetch_offline_python_base.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class OfflineBaseTests(unittest.TestCase):
    def test_external_descriptor_cannot_choose_a_path(self):
        helper = load_helper()
        for value in ['sha256:../../escape', 'sha512:' + 'a' * 128, 'sha256:' + 'a' * 63]:
            with self.assertRaises(ValueError):
                helper.blob_path(ROOT, value)

    def test_cached_blob_is_rehashed_and_tampering_stops_verification(self):
        helper = load_helper()
        with tempfile.TemporaryDirectory(dir=ROOT / '指导/临时') as temp:
            layout = Path(temp)
            (layout / 'blobs/sha256').mkdir(parents=True)
            original = b'{"os":"linux","architecture":"amd64"}'
            digest = 'sha256:' + hashlib.sha256(original).hexdigest()
            helper.blob_path(layout, digest).write_bytes(original.replace(b'amd64', b'arm64'))
            manifest = json.dumps({'config': {'digest': digest, 'size': len(original)},
                                   'layers': []}).encode()
            manifest_digest = 'sha256:' + hashlib.sha256(manifest).hexdigest()
            helper.blob_path(layout, manifest_digest).write_bytes(manifest)
            (layout / 'index.json').write_text(json.dumps(
                {'manifests': [{'digest': manifest_digest}]}), encoding='utf-8')
            (layout / 'BASE_IMAGE_SOURCE.json').write_text(json.dumps(
                {'official_source_digest': 'unit-fixture-not-an-official-image'}), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'blob size or digest mismatch'):
                helper.verify_layout(layout)


if __name__ == '__main__':
    unittest.main()
