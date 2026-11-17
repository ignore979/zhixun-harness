"""Recompute and export a posthoc research candidate; explicitly not a new holdout test."""
from __future__ import annotations

import json
import os
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import numpy as np  # noqa: E402

from zhixun.adaptation_study import summarize  # noqa: E402
from zhixun.data import digest, write_json  # noqa: E402
from zhixun.domain_views import BlockCoralAdapter  # noqa: E402
from zhixun.ensemble import temporal_versions  # noqa: E402
from zhixun.fair_core import load_fair_data  # noqa: E402
from zhixun.metrics import classification_metrics  # noqa: E402
from zhixun.registry import ModelRegistry  # noqa: E402
from zhixun.tracking import Recorder  # noqa: E402
from zhixun.vendor.model import Model  # noqa: E402


def main():
    study = json.loads((ROOT / 'artifacts/study-domain-20261116T084204/study.json').read_text(
        encoding='utf-8'))
    source = study['sources'][1]
    study_run = ROOT / 'artifacts' / source['experiment_id']
    studied = json.loads((study_run / 'run.json').read_text(encoding='utf-8'))
    candidate = next(r for r in studied['cases'] if r['spec']['name'] == 'block025')
    baseline = studied['baseline']
    _, base, train, batches = load_fair_data(ROOT)
    source_folder = ROOT / 'artifacts' / source['source']
    report = json.loads((source_folder / 'run.json').read_text(encoding='utf-8'))
    state = json.loads((source_folder / 'stream-state.json').read_text(encoding='utf-8'))
    registry = ModelRegistry(source_folder, state['signature'])
    events = [e for e in report['events'] if e['status'] == 'completed']
    adapter = BlockCoralAdapter.from_source(base, train.x,
                                           source_sha256=digest(ROOT / 'data/A/train.jsonl'))
    folder = ROOT / 'artifacts' / (f'run-block-recheck-{datetime.now(UTC):%Y%m%dT%H%M%S}-'
                                   f'{uuid4().hex[:8]}')
    folder.mkdir()
    registration = {'experiment_id': folder.name, 'created_at': datetime.now(UTC).isoformat(),
                    'task': 'posthoc_block_candidate_recomputation', 'source_run': source['source'],
                    'selection_phase': 'exploratory_after_view_ablation_not_registered_winner',
                    'source_case_sha256': digest(ROOT / candidate['case_directory'] /
                                                'result.json'),
                    'spec': candidate['spec'], 'target_labels_used': False,
                    'independence': 'same_public_predictions_recomputed_not_new_test',
                    'source_sha256': {p.relative_to(ROOT).as_posix(): digest(p)
                                       for p in sorted((ROOT / 'src').rglob('*.py'))}}
    write_json(folder / 'registration.json', registration)
    recorder = Recorder(folder, registration, mode='offline', project='zhixun-school',
                        workspace=None)
    checked = []
    try:
        for index, batch in enumerate(batches):
            assert events[index]['parent_version'] in temporal_versions(events, index)
            parent = Model.load(registry.checkpoint(events[index]['parent_version']))
            prediction = adapter.predict(parent, batch.data.x)
            path = folder / f'prequential-batch-{index + 2:02d}.npz'
            np.savez_compressed(path, probabilities=prediction, ids=np.array(batch.data.ids))
            metrics, details = classification_metrics(prediction, batch.data.y,
                                                       base.meta['classes'])
            expected = candidate['events'][index]['metrics']
            assert all(np.isclose(v, expected[k], rtol=1e-12, atol=1e-12)
                       for k, v in metrics.items())
            checked.append({'batch': index + 2, 'rows': len(batch.data.ids), 'metrics': metrics,
                            'details': details, 'prediction_sha256': digest(path)})
            recorder.log(index, {
                **{f'candidate/{k}': v for k, v in metrics.items()},
                **{f'reference/{k}': v for k, v in baseline['events'][index]['metrics'].items()},
                'candidate/acetaldehyde_recall': details['Acetaldehyde']['recall'],
                'reference/acetaldehyde_recall': baseline['events'][index]['details'][
                    'Acetaldehyde']['recall'], 'batch/number': index + 2})
        summaries = {'full': summarize(checked, base.meta['classes'], list(range(2, 11))),
                     'audit': summarize(checked, base.meta['classes'], [8, 9, 10]),
                     'development': summarize(checked, base.meta['classes'], list(range(2, 8)))}
        recorder.log(100, {f'summary/{scope}/{k}': v for scope, summary in summaries.items()
                           for k, v in summary.items() if isinstance(v, (int, float))})
        recorder.finish()
        bundle = ROOT / 'artifacts/model-candidate-block-coral-v1'
        bundle.mkdir(exist_ok=False)
        shutil.copy2(registry.checkpoint(events[-1]['parent_version']), bundle / 'checkpoint.npz')
        adapter.save(bundle / 'adapter.npz')
        card = {'status': 'posthoc_research_candidate_not_default', 'source_run': source['source'],
                'checkpoint_parent_version': events[-1]['parent_version'],
                'checkpoint_sha256': digest(bundle / 'checkpoint.npz'),
                'adapter_sha256': digest(bundle / 'adapter.npz'), 'spec': candidate['spec'],
                'trajectory_metrics_not_single_checkpoint_benchmark': summaries,
                'reference_trajectory_metrics': baseline['summary'],
                'independence': 'repeated_public_development_not_blind_test',
                'not_selected_by_registered_development_rule': True,
                'whole_unlabeled_batch_required': True,
                'formal_transductive_inference_approval': 'not_verified',
                'block_meaning': 'contiguous_8_feature_engineering_partition_not_official_schema',
                'default_candidate_unchanged': True, '85_percent_achieved': False}
        write_json(bundle / 'model-card.json', card)
        result = {**registration, 'status': 'completed', 'events': checked, 'summary': summaries,
                  'tracking': recorder.reference(), 'bundle': str(bundle.relative_to(ROOT)),
                  'predeclared_selection_passed': False}
        write_json(folder / 'run.json', result)
        print(json.dumps({'experiment_id': folder.name, 'bundle': result['bundle'],
                          'summary': summaries}, ensure_ascii=False))
    except BaseException as exc:
        write_json(folder / 'failure.json', {'status': 'failed', 'message': str(exc)})
        recorder.finish(failed=True)
        raise


if __name__ == '__main__':
    main()
