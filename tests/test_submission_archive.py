"""Minimal fake tar metadata are unit fixtures only, never actual submission images."""
import gzip
import importlib.util
import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path

from zhixun.data import ROOT


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'submission' / f'{name}.py')
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


class SubmissionArchiveTests(unittest.TestCase):
    def setUp(self):
        parent = ROOT / '指导/临时'
        parent.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=parent)
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.validator = module('validate_image_archive').validate_archive

    def archive(self, *, arch='amd64', entries=1, command=None):
        path = self.folder / 'unit-fixture-not-a-real-image.tar'
        manifest = [{'Config': 'config.json', 'RepoTags': ['unit-fixture:only'],
                     'Layers': ['layer.tar']}] * entries
        config = {'os': 'linux', 'architecture': arch,
                  'config': {'Entrypoint': ['python', 'unit-fixture-only.py'],
                             'Cmd': command or []}}
        with tarfile.open(path, 'w') as archive:
            for name, content in [('manifest.json', json.dumps(manifest).encode()),
                                  ('config.json', json.dumps(config).encode()),
                                  ('layer.tar', b'unit-test-layer-not-runnable')]:
                info = tarfile.TarInfo(name)
                info.size = len(content)
                archive.addfile(info, io.BytesIO(content))
        return path

    def test_format_check_does_not_claim_runtime_compatibility(self):
        result = self.validator(self.archive())
        self.assertTrue(result['archive_format_valid'])
        self.assertFalse(result['official_harness_contract_or_runtime_compatibility_verified'])

    def test_rejects_wrong_architecture_multiple_images_and_development_doctor(self):
        for kwargs in [{'arch': 'arm64'}, {'entries': 2}, {'command': ['doctor']}]:
            with self.assertRaises(ValueError):
                self.validator(self.archive(**kwargs))

    def test_rejects_source_tar_and_disguised_gzip(self):
        source = self.folder / 'source.tar'
        with tarfile.open(source, 'w'):
            pass
        with self.assertRaises(ValueError):
            self.validator(source)
        path = self.archive()
        compressed = gzip.compress(path.read_bytes())
        path.write_bytes(compressed)
        with self.assertRaises(ValueError):
            self.validator(path)

    def test_unverified_contract_cannot_create_build_context(self):
        prepare = module('prepare_image_context').prepare_context
        target = self.folder / 'must-not-exist'
        with self.assertRaisesRegex(ValueError, 'Official Harness protocol'):
            prepare(ROOT, target, ROOT / 'submission/runtime_contract.json')
        self.assertFalse(target.exists())

    def test_source_selection_excludes_data_models_logs_and_credentials(self):
        files = module('prepare_materials').source_files(ROOT)
        self.assertTrue(files)
        for p in files:
            self.assertNotIn(p.relative_to(ROOT).parts[0], {'data', 'artifacts', '指导', '.git'})
            self.assertNotIn(p.suffix, {'.npz', '.zip', '.tar', '.log'})


if __name__ == '__main__':
    unittest.main()
