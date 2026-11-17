"""Validate an actual uncompressed single-image docker-save archive without extracting it."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import tarfile
from pathlib import Path, PurePosixPath


def validate_archive(path: Path) -> dict:
    if not path.is_file() or path.suffix.lower() != '.tar':
        raise ValueError('Select an existing uncompressed .tar created by docker save')
    with path.open('rb') as stream:
        if stream.read(2) == b'\x1f\x8b':
            raise ValueError('Gzip-compressed images are not accepted by the platform')
    with tarfile.open(path, mode='r:') as archive:
        members = archive.getmembers()
        names = [m.name.removeprefix('./') for m in members]
        if len(names) != len(set(names)):
            raise ValueError('Duplicate archive members')
        for name, member in zip(names, members, strict=True):
            parsed = PurePosixPath(name)
            if parsed.is_absolute() or '..' in parsed.parts or '\\' in name:
                raise ValueError('Unsafe archive member path')
            if member.issym() or member.islnk():
                raise ValueError('Unexpected outer archive link')
        by_name = dict(zip(names, members, strict=True))

        def read_json(name):
            member = by_name.get(name)
            if member is None or not member.isfile() or member.size > 10 * 1024 * 1024:
                raise ValueError(f'Missing or oversized Docker metadata: {name}')
            with archive.extractfile(member) as stream:
                return json.load(stream)

        manifest = read_json('manifest.json')
        if not isinstance(manifest, list) or len(manifest) != 1:
            raise ValueError('The platform requires exactly one image, not a source-code tar')
        entry = manifest[0]
        if not isinstance(entry, dict) or not isinstance(entry.get('Config'), str):
            raise ValueError('Invalid docker-save manifest')
        config = read_json(entry['Config'])
        if config.get('os') != 'linux' or config.get('architecture') != 'amd64':
            raise ValueError('The platform requires Linux/amd64')
        runtime = config.get('config') or {}
        entrypoint, command = runtime.get('Entrypoint') or [], runtime.get('Cmd') or []
        if not all(isinstance(v, list) and all(isinstance(s, str) for s in v)
                   for v in [entrypoint, command]) or not (entrypoint or command):
            raise ValueError('Image has no usable startup command')
        if any(s == 'doctor' for s in [*entrypoint, *command]):
            raise ValueError('A doctor-only development image is not a platform Harness')
        layers = entry.get('Layers')
        if not isinstance(layers, list) or not layers:
            raise ValueError('Image layers missing')
        for layer in layers:
            member = by_name.get(layer)
            if member is None or not member.isfile() or member.size == 0:
                raise ValueError('Image layer missing or empty')
        checked_blobs = 0
        for name in [entry['Config'], *layers]:
            match = re.fullmatch(r'blobs/sha256/([0-9a-f]{64})', name)
            if match:
                with archive.extractfile(by_name[name]) as stream:
                    if hashlib.file_digest(stream, 'sha256').hexdigest() != match.group(1):
                        raise ValueError('Docker blob digest mismatch')
                checked_blobs += 1
    with path.open('rb') as stream:
        checksum = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'archive': str(path.resolve()), 'sha256': checksum, 'bytes': path.stat().st_size,
            'images': 1, 'os': 'linux', 'architecture': 'amd64', 'layers': len(layers),
            'entrypoint': entrypoint, 'command': command, 'blob_digests_checked': checked_blobs,
            'archive_format_valid': True,
            'official_harness_contract_or_runtime_compatibility_verified': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('archive', type=Path)
    args = parser.parse_args()
    print(json.dumps(validate_archive(args.archive), ensure_ascii=False, indent=2))
