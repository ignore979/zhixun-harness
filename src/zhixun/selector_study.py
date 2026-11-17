"""Replay causal selectors over saved honest predictions; no new industrial training."""
from __future__ import annotations

import json
from datetime import UTC, datetime
from uuid import uuid4

import numpy as np

from zhixun.adaptation_study import summarize
from zhixun.data import ROOT, digest, write_json
from zhixun.ensemble_study import safe_candidate, select_development
from zhixun.fair_core import load_fair_data
from zhixun.metrics import classification_metrics
from zhixun.tracking import Recorder
from zhixun.view_selector import ViewSelector


def run_selector_study(*, root=ROOT, mode='offline', project='zhixun-school'):
    config = json.loads((root / 'configs/view_selector_study.json').read_text(encoding='utf-8'))
    source_path = root / config['source_domain_study']
    domain = json.loads(source_path.read_text(encoding='utf-8'))
    _, base, _, batches = load_fair_data(root)
    classes = base.meta['classes']
    study = root / 'artifacts' / f'study-selector-{datetime.now(UTC):%Y%m%dT%H%M%S}'
    study.mkdir()
    results = []
    write_json(study / 'registration.json', {'config': config,
                                            'created_at': datetime.now(UTC).isoformat()})
    for src in domain['sources']:
        source = root / 'artifacts' / src['experiment_id']
        report = json.loads((source / 'run.json').read_text(encoding='utf-8'))
        experts = report['cases']
        baseline = next(r for r in experts if r['spec']['name'] == 'coral025')
        baseline_index = experts.index(baseline)
        folder = root / 'artifacts' / (f'run-selector-{datetime.now(UTC):%Y%m%dT%H%M%S}-'
                                       f'{uuid4().hex[:8]}')
        folder.mkdir()
        registration = {'experiment_id': folder.name, 'created_at': datetime.now(UTC).isoformat(),
                        'task': 'delayed_domain_view_selector', 'config': config,
                        'expert_specs': [r['spec'] for r in experts], 'source_run': src['source'],
                        'source_experiment': source.name,
                        'source_report_sha256': digest(source / 'run.json'),
                        'source_sha256': {p.relative_to(root).as_posix(): digest(p)
                                           for p in sorted((root / 'src').rglob('*.py'))},
                        'timing_scope': 'cached_expert_outputs_requires_14_views_in_live_inference',
                        'new_training_epochs': 0, 'target_labels_in_predict': False}
        write_json(folder / 'registration.json', registration)
        recorder = Recorder(folder, registration, mode=mode, project=project, workspace=None)
        cases = []
        try:
            for case_index, policy in enumerate(config['policies']):
                case_folder = folder / f'case-{case_index:03d}'
                case_folder.mkdir()
                selector = ViewSelector(classes, baseline=baseline_index, experts=len(experts),
                                        **{k: v for k, v in policy.items() if k != 'name'})
                events = []
                for index, batch in enumerate(batches):
                    predictions = []
                    for expert in experts:
                        path = root / expert['case_directory'] / f'batch-{index + 2:02d}.npz'
                        if digest(path) != expert['events'][index]['prediction_sha256']:
                            raise ValueError('Saved honest expert predictions changed')
                        with np.load(path, allow_pickle=False) as arrays:
                            if arrays['ids'].tolist() != list(batch.data.ids):
                                raise ValueError('Expert row order differs')
                            predictions.append(arrays['probabilities'].copy())
                    prediction = selector.predict(predictions, batch=index + 2)
                    path = case_folder / f'batch-{index + 2:02d}.npz'
                    np.savez_compressed(path, probabilities=prediction,
                                        ids=np.array(batch.data.ids))
                    metrics, details = classification_metrics(prediction, batch.data.y, classes)
                    events.append({'batch': index + 2, 'rows': len(batch.data.ids),
                                   'metrics': metrics,
                                   'details': details, 'weights': selector.last_weights,
                                   'released_batches_used': [r['batch'] for r in selector.history],
                                   'prediction_sha256': digest(path)})
                    recorder.log(case_index * 100 + index, {
                        **{f'prequential/{k}': v for k, v in metrics.items()},
                        **{f'recall/{c}': details[c]['recall'] for c in classes},
                        'batch/number': index + 2, 'case/number': case_index})
                    selector.release(batch.data.y, batch=index + 2)
                summaries = {'full': summarize(events, classes, list(range(2, 11))),
                             'development': summarize(events, classes,
                                                      config['development_batches']),
                             'audit': summarize(events, classes, config['audit_batches'])}
                row = {'case': case_index, 'spec': {**policy, 'strength': 0.0},
                       'summary': summaries,
                       'events': events, 'experiment_id': folder.name,
                       'case_directory': str(case_folder.relative_to(root))}
                cases.append(row)
                write_json(case_folder / 'result.json', row)
            selected = select_development(cases, baseline)
            accepted = safe_candidate(selected['summary']['audit'], baseline['summary']['audit'])
            recorder.log(10000, {'selection/accepted_audit_gate': float(accepted)})
            for i, (chosen, control) in enumerate(zip(selected['events'], baseline['events'],
                                                     strict=True)):
                recorder.log(11000 + i, {
                    **{f'selected/{k}': v for k, v in chosen['metrics'].items()},
                    **{f'reference/{k}': v for k, v in control['metrics'].items()},
                    'selected/acetaldehyde_recall': chosen['details']['Acetaldehyde']['recall'],
                    'reference/acetaldehyde_recall': control['details']['Acetaldehyde']['recall']})
            recorder.finish()
            result = {**registration, 'status': 'completed', 'cases': cases, 'selected': selected,
                      'baseline': baseline, 'audit_accepted': accepted,
                      'tracking': recorder.reference()}
            write_json(folder / 'run.json', result)
            results.append({'source': src['source'], 'experiment_id': folder.name,
                            'selected': selected, 'baseline': baseline, 'audit_accepted': accepted})
        except BaseException as exc:
            write_json(folder / 'failure.json', {'status': 'failed', 'message': str(exc)})
            recorder.finish(failed=True)
            raise
    final = {'status': 'completed', 'config': config, 'sources': results,
             'study_directory': str(study.relative_to(root))}
    write_json(study / 'study.json', final)
    return final
