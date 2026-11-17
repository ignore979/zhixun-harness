"""Standalone scientific comparison of registered screens and an exploratory candidate."""
from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ['MPLCONFIGDIR'] = str(ROOT / '指导/缓存/matplotlib')

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def main():
    ensemble_path = ROOT / 'artifacts/study-ensemble-20261116T083237/study.json'
    ensemble = json.loads(ensemble_path.read_text(encoding='utf-8'))
    quality = ensemble['sources'][1]
    first = json.loads((ROOT / 'artifacts' / quality['stages'][0]['experiment_id'] /
                        'run.json').read_text(encoding='utf-8'))
    rows = [r for r in first['results'] if r['spec']['strength'] == .25]
    fine = json.loads((ROOT / 'artifacts' / quality['stages'][1]['experiment_id'] /
                       'run.json').read_text(encoding='utf-8'))
    domain = json.loads((ROOT / 'artifacts/study-domain-20261116T084204/study.json').read_text(
        encoding='utf-8'))
    domain_run = json.loads((ROOT / 'artifacts' / domain['sources'][1]['experiment_id'] /
                             'run.json').read_text(encoding='utf-8'))
    baseline = domain_run['baseline']['summary']
    block = next(r for r in domain_run['cases'] if r['spec']['name'] == 'block025')['summary']
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout='constrained')
    x = np.arange(len(rows))
    axes[0, 0].bar(x - .18, [100 * r['summary']['full']['accuracy'] for r in rows],
                   width=.36, label='Whole stream')
    axes[0, 0].bar(x + .18, [100 * r['summary']['audit']['accuracy'] for r in rows],
                   width=.36, label='Batches 8-10')
    axes[0, 0].set_xticks(x, [r['spec']['name'] for r in rows], rotation=20)
    axes[0, 0].set_title('Historical-version ensembles + CORAL 0.25')
    axes[0, 0].legend(fontsize=9)
    axes[0, 0].set_ylabel('Accuracy (%)')
    xx = [100 * r['summary']['development']['accuracy'] for r in fine['results']]
    yy = [100 * r['summary']['audit']['accuracy'] for r in fine['results']]
    axes[0, 1].scatter(xx, yy, alpha=.7, label='Registered CORAL fine grid')
    chosen = fine['selected']['summary']
    axes[0, 1].scatter([100 * chosen['development']['accuracy']],
                       [100 * chosen['audit']['accuracy']], marker='x', s=120,
                       color='red', label='Chosen using batches 2-7')
    axes[0, 1].axhline(100 * baseline['audit']['accuracy'], color='gray', linestyle='--')
    axes[0, 1].set_xlabel('Batches 2-7 accuracy (%)')
    axes[0, 1].set_ylabel('Batches 8-10 accuracy (%)')
    axes[0, 1].set_title('A better prefix does not ensure a better later period')
    axes[0, 1].legend(fontsize=8)
    categories = ['Accuracy', 'Macro-F1', 'G-mean', 'Acetaldehyde recall']
    for axis, scope, title in [(axes[1, 0], 'full', 'Whole stream: exploratory block view'),
                               (axes[1, 1], 'audit', 'Batches 8-10: exploratory block view')]:
        def values(summary, selected_scope=scope):
            d = summary[selected_scope]
            return [100 * d['accuracy'], 100 * d['mean_macro_f1'],
                    100 * d['mean_g_mean'], 100 * d['pooled_per_class_recall']['Acetaldehyde']]

        xs = np.arange(4)
        axis.bar(xs - .18, values(baseline), width=.36, label='Previous CORAL 0.25')
        axis.bar(xs + .18, values(block), width=.36, label='Block CORAL 0.25 (posthoc)')
        axis.set_xticks(xs, categories, rotation=10)
        axis.set_ylabel('Percent')
        axis.set_title(title)
        axis.legend(fontsize=8)
    for axis in axes.flat:
        axis.grid(axis='y', alpha=.2)
        axis.set_axisbelow(True)
    fig.suptitle('Round 8: reused public development data; block candidate is not a locked winner',
                 fontsize=13)
    output = ensemble_path.parent / 'comparison.png'
    fig.savefig(output, dpi=160)
    plt.close(fig)
    print(output)


if __name__ == '__main__':
    main()
