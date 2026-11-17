"""Prepare a minimal context only after the real official runtime has been implemented."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def prepare_context(root: Path, destination: Path, contract_path: Path) -> dict:
    contract = json.loads(contract_path.read_text(encoding='utf-8'))
    if contract.get('verification_status') != 'implemented_and_official_contract_verified':
        raise ValueError(
            'Official Harness protocol and adapter are missing; image build is blocked')
    required = ['official_reference', 'entrypoint', 'smoke_test_args', 'workspace_mounts',
                'model_environment_variables', 'artifact_format']
    if any(contract.get(k) is None for k in required):
        raise ValueError('Official runtime verification record is incomplete')
    entrypoint = contract['entrypoint']
    if (not isinstance(entrypoint, list) or not entrypoint
            or not all(isinstance(v, str) and v for v in entrypoint)
            or 'doctor' in entrypoint):
        raise ValueError('A verified actual Harness entrypoint is required')
    destination = destination.resolve()
    if not destination.is_relative_to((root / '指导/临时').resolve()) or destination.exists():
        raise ValueError('Use a new build-context directory within 指导/临时')
    destination.mkdir(parents=True)
    for folder in ['src', 'configs']:
        shutil.copytree(root / folder, destination / folder,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for name in ['run.py', 'requirements.txt', 'requirements-pytrio.txt', 'NOTICE.md']:
        shutil.copy2(root / name, destination / name)
    # No local data, labels, historical checkpoints, credentials, caches or experiment archives.
    dockerfile = '\n'.join([
        'FROM python:3.12-slim', 'WORKDIR /opt/zhixun',
        'ENV PYTHONPATH=/opt/zhixun/src PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1',
        'ENV OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1',
        'COPY requirements.txt requirements-pytrio.txt ./',
        'RUN pip install --no-cache-dir -r requirements-pytrio.txt',
        'COPY src ./src', 'COPY configs ./configs', 'COPY run.py NOTICE.md ./',
        'ENTRYPOINT ' + json.dumps(entrypoint), '',
    ])
    (destination / 'Dockerfile').write_text(dockerfile, encoding='utf-8')
    return {'context': str(destination), 'data_and_historical_weights_included': False,
            'entrypoint': entrypoint, 'smoke_test_args': contract['smoke_test_args']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--contract', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare_context(args.root, args.destination, args.contract), indent=2))
