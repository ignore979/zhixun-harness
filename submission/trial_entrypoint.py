"""Local project image entrypoint; this is not an official platform protocol."""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT))
for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '1'


def run_trial(task: str, rounds: int, epochs: int, output: Path) -> dict:
    from zhixun.data import write_json
    from zhixun.train import RoundPlan, run_round

    if task not in ('A', 'B', 'C', 'all') or not 1 <= rounds <= 10 or not 1 <= epochs <= 50:
        raise ValueError('Use task A/B/C/all, 1..10 rounds and 1..50 epochs')
    folder = output / f'trial-{datetime.now(UTC):%Y%m%dT%H%M%S}-{uuid4().hex[:8]}'
    folder.mkdir(parents=True, exist_ok=False)
    config = json.loads((ROOT / 'configs/round1.json').read_text(encoding='utf-8'))
    result = {
        'format': 'zhixun.local-image-trial.v1',
        'evaluation_scope': 'bundled_public_training_and_validation_only',
        'official_runtime_protocol_verified': False,
        'hidden_test_evaluation_performed': False,
        'llm_calls': 0, 'rounds_per_task': rounds, 'epochs_per_round': epochs,
        'status': 'running', 'rounds': [],
    }
    report = folder / 'trial_report.json'
    write_json(report, result)
    try:
        for profile in ('ABC' if task == 'all' else task):
            parent = None
            for number in range(1, rounds + 1):
                print(json.dumps({'task': profile, 'round': number, 'phase': 'training'}),
                      flush=True)
                plan = RoundPlan(profile, epochs, config['learning_rate'], config['max_seconds'])
                outcome = run_round(plan, mode='offline', project='zhixun-image-trial',
                                    parent=parent, output=folder)
                parent = folder / outcome['experiment_id'] / 'checkpoint.npz'
                item = {'task': profile, 'round': number, 'experiment_id': outcome['experiment_id'],
                        'accepted': outcome['accepted'], 'metrics': outcome['selected_metrics'],
                        'checkpoint': str(parent), 'status': outcome['status']}
                result['rounds'].append(item)
                write_json(report, result)
                print(json.dumps(item, ensure_ascii=False), flush=True)
        result['status'] = 'completed'
    except BaseException as exc:
        result.update(status='failed', error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        write_json(report, result)
    return {'report': str(report), **result}


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] == 'health':
        from zhixun.cli import doctor

        print(json.dumps({'image_purpose': 'local_project_trial',
                          'official_runtime_protocol_verified': False, **doctor()}, indent=2))
        return 0
    if not args or args[0] == 'run':
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument('--task', choices=['A', 'B', 'C', 'all'], default='all')
        parser.add_argument('--rounds', type=int, default=10)
        parser.add_argument('--epochs', type=int, default=5)
        parser.add_argument('--output', type=Path, default=ROOT / 'artifacts')
        options = parser.parse_args(args[1:] if args else [])
        result = run_trial(options.task, options.rounds, options.epochs, options.output)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    # Existing project commands (predict/train/stream/trio/etc.) remain available.
    import run

    sys.argv = [str(ROOT / 'run.py'), *args]
    return run.main()


if __name__ == '__main__':
    raise SystemExit(main())
