"""Download a digest-pinned official Python base into a verified local OCI layout."""
from __future__ import annotations

import gzip
import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
LAYOUT = ROOT / '指导/临时/python-base-oci-20261117'
SOURCE_DIGEST = 'sha256:935c3c5b12357a7e10caa9c1e6ca5692344d97cf229832ef584be2cec9fc03df'
REGISTRY = 'https://registry-1.docker.io/v2/library/python'
OCI_MANIFEST = 'application/vnd.oci.image.manifest.v1+json'
OCI_INDEX = 'application/vnd.oci.image.index.v1+json'
ACCEPT = ', '.join([OCI_INDEX, OCI_MANIFEST,
                    'application/vnd.docker.distribution.manifest.list.v2+json',
                    'application/vnd.docker.distribution.manifest.v2+json'])


def digest_file(path: Path) -> str:
    with path.open('rb') as stream:
        return 'sha256:' + hashlib.file_digest(stream, 'sha256').hexdigest()


def blob_path(layout: Path, digest: str) -> Path:
    if not isinstance(digest, str) or not re.fullmatch(r'sha256:[0-9a-f]{64}', digest):
        raise ValueError('Unsupported or unsafe blob digest')
    return layout / 'blobs/sha256' / digest.split(':')[1]


def token() -> str:
    response = requests.get('https://auth.docker.io/token', params={
        'service': 'registry.docker.io', 'scope': 'repository:library/python:pull'},
        timeout=(15, 45))
    response.raise_for_status()
    return response.json()['token']


def get_document(digest: str) -> bytes:
    blob_path(LAYOUT, digest)
    response = requests.get(f'{REGISTRY}/manifests/{digest}',
                            headers={'Authorization': 'Bearer ' + token(), 'Accept': ACCEPT},
                            timeout=(15, 45))
    response.raise_for_status()
    content = response.content
    if 'sha256:' + hashlib.sha256(content).hexdigest() != digest:
        raise ValueError('Official manifest digest mismatch')
    return content


def fetch_blob(descriptor: dict) -> None:
    target = blob_path(LAYOUT, descriptor['digest'])
    expected = descriptor['size']
    if target.is_file() and target.stat().st_size == expected:
        if digest_file(target) == descriptor['digest']:
            print(f"Cached verified blob {descriptor['digest'][:23]}", flush=True)
            return
    partial = target.with_suffix('.partial')
    for attempt in range(1, 5):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            headers = {'Authorization': 'Bearer ' + token()}
            if offset:
                headers['Range'] = f'bytes={offset}-'
            with requests.get(f"{REGISTRY}/blobs/{descriptor['digest']}", headers=headers,
                              stream=True, timeout=(15, 45)) as response:
                response.raise_for_status()
                append = offset > 0 and response.status_code == 206
                if append and not response.headers.get('Content-Range', '').startswith(
                        f'bytes {offset}-'):
                    raise ValueError('Invalid range response')
                written = offset if append else 0
                with partial.open('ab' if append else 'wb') as stream:
                    for chunk in response.iter_content(1024 * 1024):
                        if chunk:
                            stream.write(chunk)
                            written += len(chunk)
                            if written > expected:
                                raise ValueError('Blob exceeds declared size')
                            print(f"Blob {descriptor['digest'][:23]} {written}/{expected}",
                                  flush=True)
            if partial.stat().st_size != expected:
                raise ValueError('Incomplete blob download')
            if digest_file(partial) != descriptor['digest']:
                raise ValueError('Blob digest mismatch')
            partial.replace(target)
            return
        except Exception as exc:
            # Never print bearer tokens, headers or signed redirect URLs.
            print(f"Blob {descriptor['digest'][:23]} attempt {attempt}: {type(exc).__name__}",
                  flush=True)
    raise RuntimeError(f"Blob download failed after four attempts: {descriptor['digest']}")


def verify_layout(layout: Path = LAYOUT) -> dict:
    source = json.loads((layout / 'BASE_IMAGE_SOURCE.json').read_text(encoding='utf-8'))
    descriptor = json.loads((layout / 'index.json').read_text(encoding='utf-8'))['manifests'][0]
    manifest_path = blob_path(layout, descriptor['digest'])
    if digest_file(manifest_path) != descriptor['digest']:
        raise ValueError('Local OCI manifest digest mismatch')
    manifest = json.loads(manifest_path.read_bytes())
    for item in [manifest['config'], *manifest['layers']]:
        path = blob_path(layout, item['digest'])
        if path.stat().st_size != item['size'] or digest_file(path) != item['digest']:
            raise ValueError('Local OCI blob size or digest mismatch')
    config = json.loads(blob_path(layout, manifest['config']['digest']).read_bytes())
    if config['os'] != 'linux' or config['architecture'] != 'amd64':
        raise ValueError('Python base must be Linux/amd64')
    if source['official_source_digest'] != SOURCE_DIGEST:
        raise ValueError('Unexpected source image digest')
    return {**source, 'manifest_digest': descriptor['digest'],
            'os': 'linux', 'architecture': 'amd64'}


def main() -> None:
    (LAYOUT / 'blobs/sha256').mkdir(parents=True, exist_ok=True)
    raw_index = get_document(SOURCE_DIGEST)
    original = json.loads(raw_index)
    if 'manifests' in original:
        candidates = [d for d in original['manifests'] if
                      d.get('platform', {}).get('os') == 'linux' and
                      d.get('platform', {}).get('architecture') == 'amd64']
        if len(candidates) != 1:
            raise ValueError('Expected exactly one Linux/amd64 image')
        platform_digest = candidates[0]['digest']
        raw_manifest = get_document(platform_digest)
    else:
        platform_digest, raw_manifest = SOURCE_DIGEST, raw_index
    original_manifest = json.loads(raw_manifest)
    for digest, content in [(SOURCE_DIGEST, raw_index), (platform_digest, raw_manifest)]:
        blob_path(LAYOUT, digest).write_bytes(content)
    descriptors = [original_manifest['config'], *original_manifest['layers']]
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(fetch_blob, descriptors))
    config = json.loads(blob_path(LAYOUT, original_manifest['config']['digest']).read_bytes())
    if config['os'] != 'linux' or config['architecture'] != 'amd64':
        raise ValueError('Downloaded configuration is not Linux/amd64')
    diff_ids = config['rootfs']['diff_ids']
    if len(diff_ids) != len(original_manifest['layers']):
        raise ValueError('Layer and rootfs counts differ')
    for layer, expected in zip(original_manifest['layers'], diff_ids, strict=True):
        with gzip.open(blob_path(LAYOUT, layer['digest']), 'rb') as stream:
            actual = 'sha256:' + hashlib.file_digest(stream, 'sha256').hexdigest()
        if actual != expected:
            raise ValueError('Uncompressed layer digest mismatch')
    manifest = json.loads(raw_manifest)
    manifest['mediaType'] = OCI_MANIFEST
    manifest['config']['mediaType'] = 'application/vnd.oci.image.config.v1+json'
    for layer in manifest['layers']:
        layer['mediaType'] = 'application/vnd.oci.image.layer.v1.tar+gzip'
    encoded = json.dumps(manifest, separators=(',', ':')).encode()
    digest = 'sha256:' + hashlib.sha256(encoded).hexdigest()
    blob_path(LAYOUT, digest).write_bytes(encoded)
    index = {'schemaVersion': 2, 'mediaType': OCI_INDEX, 'manifests': [{
        'mediaType': OCI_MANIFEST, 'digest': digest, 'size': len(encoded),
        'platform': {'os': 'linux', 'architecture': 'amd64'},
        'annotations': {'org.opencontainers.image.ref.name': 'python-base'}}]}
    (LAYOUT / 'index.json').write_text(json.dumps(index), encoding='utf-8')
    (LAYOUT / 'oci-layout').write_text('{"imageLayoutVersion":"1.0.0"}', encoding='utf-8')
    source = {'source': 'docker.io/library/python:3.12-slim-bookworm',
              'official_source_digest': SOURCE_DIGEST, 'source_platform_digest': platform_digest,
              'manifest_digest': digest, 'downloaded_at': datetime.now(UTC).isoformat(),
              'compressed_and_uncompressed_layer_digests_verified': True,
              'downloaded_bytes': sum(d['size'] for d in descriptors)}
    (LAYOUT / 'BASE_IMAGE_SOURCE.json').write_text(json.dumps(source, indent=2), encoding='utf-8')
    print(json.dumps({'layout': str(LAYOUT), **verify_layout()}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
