"""Build, exercise and docker-save a user-requested trial image, without uploading it."""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from fetch_offline_python_base import LAYOUT, verify_layout
from validate_image_archive import validate_archive

ROOT = Path(__file__).resolve().parents[1]
CONTEXT = ROOT / '指导/临时/trial-image-context-20261117-v2'
OUTPUT = ROOT / '指导/交付/试上传镜像-20261116'


def sha(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify_context() -> dict:
    record = json.loads((CONTEXT / 'TRIAL_IMAGE.json').read_text(encoding='utf-8'))
    for relative, expected in record['files_sha256'].items():
        if sha(CONTEXT / relative) != expected:
            raise ValueError(f'Build context changed: {relative}')
    for name, expected in record['wheel_files_sha256'].items():
        if sha(CONTEXT / 'wheels' / name) != expected:
            raise ValueError(f'Dependency wheel changed: {name}')
    expected_files = set(record['files_sha256']) | {'TRIAL_IMAGE.json'}
    expected_files.update('wheels/' + name for name in record['wheel_files_sha256'])
    actual_files = {p.relative_to(CONTEXT).as_posix() for p in CONTEXT.rglob('*') if p.is_file()}
    if actual_files != expected_files:
        raise ValueError('Unexpected or missing files in the minimal build context')
    if any(p.is_symlink() for p in CONTEXT.rglob('*')):
        raise ValueError('Links are not allowed in the minimal build context')
    if record['official_runtime_protocol_verified'] is not False:
        raise ValueError('Trial context must not claim official protocol verification')
    return record


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='backslashreplace')
    OUTPUT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime('%Y%m%dT%H%M%S%f')
    tag = f'zhixun-local-trial:{stamp.lower()}'
    log = OUTPUT / f'build-{stamp}.log'
    status_file = OUTPUT / f'build-{stamp}.json'
    archive_path = OUTPUT / f'zhixun-trial-linux-amd64-{stamp}.tar'
    temporary_archive = OUTPUT / f'zhixun-trial-linux-amd64-{stamp}.building.tar'
    state = {
        'format': 'zhixun.trial-image-build.v1', 'started_at': datetime.now(UTC).isoformat(),
        'status': 'running', 'phase': 'context_check', 'image_tag': tag,
        'build_context': str(CONTEXT), 'log': str(log), 'archive': None,
        'official_runtime_protocol_verified': False, 'formal_submission_ready': False,
        'platform_upload_or_run_started': False, 'evaluation_attempts_used': 0,
        'container_health_passed': False, 'container_training_smoke_passed': False,
    }

    def save(phase):
        state['phase'] = phase
        state['updated_at'] = datetime.now(UTC).isoformat()
        status_file.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding='utf-8')
        (OUTPUT / 'latest_build_status.json').write_text(
            json.dumps({'status_file': str(status_file), 'status': state['status'], 'phase': phase},
                       ensure_ascii=False, indent=2), encoding='utf-8')

    def docker(arguments):
        command = [shutil.which('docker') or 'docker', *arguments]
        with log.open('a', encoding='utf-8') as journal:
            journal.write('\nCOMMAND ' + json.dumps(command, ensure_ascii=False) + '\n')
            journal.flush()
            process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, encoding='utf-8',
                                       errors='replace')
            lines = []
            for line in process.stdout:
                print(line, end='', flush=True)
                journal.write(line)
                journal.flush()
                lines.append(line)
            code = process.wait()
        if code:
            raise RuntimeError(f'Docker command failed ({code}); see {log}')
        return ''.join(lines).strip()

    try:
        record = verify_context()
        base = verify_layout(LAYOUT)
        state['context_manifest_sha256'] = sha(CONTEXT / 'TRIAL_IMAGE.json')
        state['dependency_wheels'] = len(record['wheel_files_sha256'])
        state['offline_python_base'] = base
        save('docker_engine_check')
        state['server'] = json.loads(docker(['version', '--format', '{{json .Server}}']))
        save('building_linux_amd64')
        base_path = './' + LAYOUT.relative_to(ROOT).as_posix()
        base_context = f"python_base=oci-layout://{base_path}@{base['manifest_digest']}"
        docker(['buildx', 'build', '--load', '--platform', 'linux/amd64', '--provenance=false',
                '--network', 'none', '--pull=false', '--build-context', base_context,
                '--progress', 'plain', '--tag', tag, str(CONTEXT)])
        save('image_inspect')
        image = json.loads(docker(['image', 'inspect', '--format', '{{json .}}', tag]))
        if image['Os'] != 'linux' or image['Architecture'] != 'amd64':
            raise ValueError('Wrong image architecture')
        state['image_id'] = image['Id']
        save('container_health')
        health = json.loads(docker(
            ['run', '--rm', '--network', 'none', '--read-only', tag, 'health']))
        if health['status'] != 'local_inputs_verified':
            raise ValueError('Container health did not verify original inputs')
        state['container_health_passed'] = True
        state['container_health'] = health
        save('container_training_smoke')
        smoke = OUTPUT / f'smoke-{stamp}'
        smoke.mkdir()
        docker(['run', '--rm', '--network', 'none', '--mount',
                f'type=bind,source={smoke},target=/trial-output', tag,
                'run', '--task', 'all', '--rounds', '1', '--epochs', '1',
                '--output', '/trial-output'])
        reports = list(smoke.glob('trial-*/trial_report.json'))
        if len(reports) != 1:
            raise ValueError('Container did not persist its training report')
        report = json.loads(reports[0].read_text(encoding='utf-8'))
        if report['status'] != 'completed' or len(report['rounds']) != 3:
            raise ValueError('Container A/B/C training smoke did not complete')
        if {r['task'] for r in report['rounds']} != {'A', 'B', 'C'}:
            raise ValueError('Container training smoke omitted a task')
        state['container_training_smoke_passed'] = True
        state['container_training_report'] = str(reports[0])
        save('docker_save')
        docker(['image', 'save', '--platform', 'linux/amd64',
                '--output', str(temporary_archive), tag])
        save('archive_validation')
        checked = validate_archive(temporary_archive)
        temporary_archive.rename(archive_path)
        checked['archive'] = str(archive_path)
        state.update(status='completed', archive=str(archive_path), archive_validation=checked)
        save('completed')
        (OUTPUT / f'{archive_path.name}.sha256').write_text(
            checked['sha256'] + '  ' + archive_path.name + '\n', encoding='utf-8')
        print('\nIMAGE TAR READY: ' + str(archive_path))
        print('SHA256: ' + checked['sha256'])
        print('Local image checks passed; platform runtime compatibility remains unverified.')
        return 0
    except BaseException as exc:
        state.update(status='failed', error_type=type(exc).__name__, error=str(exc))
        save(state['phase'])
        print('\nBuild stopped: ' + str(exc), file=sys.stderr)
        print('No upload or evaluation was started.', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
