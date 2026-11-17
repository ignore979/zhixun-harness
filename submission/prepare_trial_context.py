"""Explicit user-authorized trial packaging without claiming an official runtime contract."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path

from packaging.utils import canonicalize_name, parse_wheel_filename

ROOT = Path(__file__).resolve().parents[1]


def sha(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def prepare(root: Path, destination: Path, wheels: Path) -> dict:
    from sys import path

    path.insert(0, str(root / 'src'))
    from zhixun.data import verify_inputs

    for task in 'ABC':
        verify_inputs(task, root)
    wheel_files = sorted(wheels.glob('*.whl'))
    if not wheel_files:
        raise ValueError('Download Linux cp312 dependency wheels before preparing the image')
    parsed_wheels = (parse_wheel_filename(p.name) for p in wheel_files)
    available = {(canonicalize_name(name), str(version)) for name, version, _, _ in parsed_wheels}
    pinned = set()
    for lock in ['requirements.txt', 'requirements-pytrio.txt']:
        for line in (root / lock).read_text(encoding='utf-8').splitlines():
            if '==' in line and not line.startswith('#'):
                name, version = line.strip().split('==')
                pinned.add((canonicalize_name(name), version))
    if missing := pinned - available:
        raise ValueError(f'Linux dependency wheels are incomplete: {sorted(missing)}')
    target = destination.resolve()
    if target.exists() or not target.is_relative_to((root / '指导/临时').resolve()):
        raise ValueError('Use a new context directory inside 指导/临时')
    selected = [p for name in ['src', 'configs'] for p in (root / name).rglob('*')
                if p.is_file() and p.suffix in {'.py', '.json'} and '__pycache__' not in p.parts]
    selected.extend(root / name for name in [
        'run.py', 'README.md', 'NOTICE.md', 'requirements.txt', 'requirements-pytrio.txt',
        'submission/trial_entrypoint.py'])
    selected.append(root / 'data/manifest.json')
    data_names = ['train.jsonl', 'validation.jsonl', 'checkpoint.npz', 'profile.json']
    selected.extend(root / 'data' / task / name for task in 'ABC' for name in data_names)
    for item in selected:
        if item.is_symlink():
            raise ValueError('Linked files are not allowed in the trial context')
        if item.suffix != '.npz' and re.search(
            rb'\b(?:sk-[A-Za-z0-9]{24,}|swlab_[A-Za-z0-9]{20,})\b', item.read_bytes()
        ):
            raise ValueError(f'Potential credential in {item.relative_to(root)}')
    target.mkdir(parents=True)
    manifest = {}
    for item in selected:
        relative = item.relative_to(root)
        copied = target / relative
        copied.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item, copied)
        manifest[relative.as_posix()] = sha(copied)
    (target / 'wheels').mkdir()
    for wheel in wheel_files:
        if wheel.is_symlink() or 'win_' in wheel.name or 'macosx' in wheel.name:
            raise ValueError('Only ordinary Linux-compatible wheel files are allowed')
        shutil.copy2(wheel, target / 'wheels' / wheel.name)
    shutil.copy2(root / 'submission/Dockerfile.trial', target / 'Dockerfile')
    manifest['Dockerfile'] = sha(target / 'Dockerfile')
    record = {
        'format': 'zhixun.trial-image-context.v1',
        'official_runtime_protocol_verified': False,
        'user_requested_packaging_before_protocol_verification': True,
        'default_command': 'run --task all --rounds 10 --epochs 5',
        'default_llm_calls': 0,
        'data': 'Only received original A/B/C public train/validation/profile/initial checkpoint',
        'extra_replay_labels_or_trained_weights_included': False,
        'files_sha256': manifest,
        'wheel_files_sha256': {p.name: sha(p) for p in wheel_files},
        'dependency_pins_available_as_wheels': len(pinned),
        'python_base': 'Named local OCI image context; supplied by build_trial_image.py',
    }
    (target / 'TRIAL_IMAGE.json').write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return {'context': str(target), 'project_files': len(selected),
            'linux_wheels': len(wheel_files),
            'official_runtime_protocol_verified': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--wheels', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.root.resolve(), args.destination, args.wheels), indent=2))
