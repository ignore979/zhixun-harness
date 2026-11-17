"""Create reproducible development figures and a portable experimental model bundle."""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ["MPLCONFIGDIR"] = str(ROOT / "指导/缓存/matplotlib")
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from zhixun.data import digest, load_dataset, write_json  # noqa: E402
from zhixun.prior_correction import CoralAdapter  # noqa: E402
from zhixun.vendor.model import Model  # noqa: E402


def main():
    study = sorted((ROOT / "artifacts").glob("study-targeted-*/study.json"))[-1]
    result = json.loads(study.read_text(encoding="utf-8"))
    methods = [m["name"] for m in result["config"]["methods"]]
    labels = ['Identity', 'Past prior', 'Soft prior', 'BBSE 0.5', 'BBSE 1.0',
              'CORAL 0.25', 'CORAL 0.5', 'CORAL 1.0']
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout='constrained')
    source_labels = ['45-epoch flat trajectory', '108-epoch quality trajectory']
    for source, label in zip(result["config"]["source_runs"], source_labels, strict=True):
        rows = {r['method']: r for r in result['results'] if r['source'] == source}
        for axis, scope, metric, title in [
            (axes[0, 0], 'full', 'accuracy', 'Whole stream accuracy'),
            (axes[0, 1], 'full', 'mean_macro_f1', 'Mean batch Macro-F1'),
            (axes[1, 0], 'audit', 'accuracy', 'Batches 8-10 accuracy (public development)'),
            (axes[1, 1], 'audit', 'mean_macro_f1', 'Batches 8-10 mean Macro-F1'),
        ]:
            axis.plot(labels, [100 * rows[m]['summary'][scope][metric] for m in methods],
                      marker='o', label=label)
            axis.set_title(title)
            axis.set_ylabel('Percent')
            axis.tick_params(axis='x', rotation=35)
            axis.grid(alpha=.25)
    axes[0, 0].legend(fontsize=9)
    fig.suptitle('Prediction-only ablations: fixed weights, no current labels in prediction',
                 fontsize=14)
    figure = study.parent / 'comparison.png'
    fig.savefig(figure, dpi=160)
    plt.close(fig)

    source_name = result['config']['source_runs'][1]
    source = ROOT / 'artifacts' / source_name
    report = json.loads((source / 'run.json').read_text(encoding='utf-8'))
    registry = json.loads((source / 'stream-state.json').read_text(encoding='utf-8'))
    last = [e for e in report['events'] if e['status'] == 'completed'][-1]
    version = next(v for v in registry['versions'] if v['version'] == last['parent_version'])
    checkpoint = source / version['path']
    if digest(checkpoint) != version['sha256']:
        raise ValueError('Candidate checkpoint changed')
    bundle = ROOT / 'artifacts/model-candidate-coral-soft-v1'
    bundle.mkdir(exist_ok=False)
    shutil.copy2(checkpoint, bundle / 'checkpoint.npz')
    model = Model.load(checkpoint)
    initial = Model.load(ROOT / 'data/A/checkpoint.npz')
    if (not np.array_equal(model.mean, initial.mean)
            or not np.array_equal(model.scale, initial.scale)):
        raise ValueError('Frozen normalization changed')
    train_path = ROOT / 'data/A/train.jsonl'
    data = load_dataset(train_path, initial, 'A')
    adapter = CoralAdapter.from_source(model, data.x, source_sha256=digest(train_path))
    adapter.save(bundle / 'adapter.npz')
    selected = next(r for r in result['results'] if r['source'] == source_name
                    and r['method'] == 'coral_soft')
    identity = next(r for r in result['results'] if r['source'] == source_name
                    and r['method'] == 'identity')
    write_json(bundle / 'model-card.json', {
        'status': 'experimental_development_candidate',
        'source_trajectory': source_name, 'source_parent_version': version['version'],
        'checkpoint_sha256': digest(bundle / 'checkpoint.npz'),
        'adapter_sha256': digest(bundle / 'adapter.npz'),
        'adapter_source_train_sha256': digest(train_path), 'strength': .25,
        'current_target_labels_used': False, 'whole_current_unlabeled_batch_required': True,
        'selection_phase': 'posthoc_balanced_candidate_after_the_registered_ablation',
        'preregistered_prefix_winner': result['selections'][source_name]['method'],
        'preregistered_prefix_winner_rejected': 'audit_macro_f1_declined',
        'trajectory_summaries_not_single_checkpoint_benchmark': selected['summary'],
        'trajectory_identity_reference': identity['summary'],
        'formal_transductive_inference_approval': 'not_verified',
        'limitations': ['No independent or hidden test; public stream reused in development',
                        'Future batches remain untested',
                        'Reported trajectory scores use each batch historical parent, '
                        'not this single final checkpoint on all batches',
                        '85 percent target not achieved'],
    })
    print(json.dumps({'figure': str(figure), 'bundle': str(bundle)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
