"""Build a development validation panel; never a new independent final test."""
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from diagnose_leakage import fingerprint_all, nearest, to_matrix

PROJECT = Path(__file__).resolve().parents[2]


def read(name):
    with (PROJECT / 'dataset/splits' / f'{name}.csv').open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def spread(rows, limit):
    return [rows[i] for i in np.linspace(0, len(rows)-1, min(limit, len(rows)), dtype=int)] if rows else []


def main():
    train = {r['path']: r for name in ('train_temporal', 'train_mixed_v3', 'train_with_adapt') for r in read(name)}
    candidates = read('validation_temporal') + read('test_adapt_val_smoke_v1')
    fingerprints, missing = fingerprint_all(PROJECT / 'dataset', set(train) | {r['path'] for r in candidates},
                                           PROJECT / 'backend/artifacts/label_audit/development_hash_cache.json')
    if missing:
        raise ValueError('Missing images; validation was not created.')
    train_paths = list(train)
    sha = {fingerprints[p]['sha256'] for p in train_paths}
    unique = {r['path']: r for r in candidates}
    distances, _ = nearest(to_matrix([fingerprints[p] for p in unique]),
                           to_matrix([fingerprints[p] for p in train_paths]), chunk=16)
    clean = [r for r, distance in zip(unique.values(), distances)
             if r['path'] not in train and fingerprints[r['path']]['sha256'] not in sha and distance > 8]
    panel = []
    for label in ('Fire', 'No_Fire'):
        for scene in ('lake_', 'forest'):
            group = [r for r in clean if r['path'].startswith('Training/') and r['label'] == label
                     and (('lake_' in Path(r['path']).name) == (scene == 'lake_'))]
            panel.extend(spread(group, 200))
    panel.extend(r for r in clean if r['path'].startswith('Test/'))
    if set(r['label'] for r in panel) != {'Fire', 'No_Fire'}:
        raise ValueError('Panel needs both classes after leakage filtering.')
    target = PROJECT / 'dataset/splits/validation_smoke_development_v1.csv'
    with target.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['path', 'label']); w.writeheader(); w.writerows(panel)
    report = dict(counts=dict(Counter(r['label'] for r in panel)), total=len(panel),
                  candidates=len(unique), excluded_near_training=len(unique)-len(clean),
                  dhash_minimum_distance=9, train_comparison_splits=['train_temporal','train_mixed_v3','train_with_adapt'],
                  negative_label_source='Existing validation_temporal labels; not a new provenance audit',
                  confirmed_smoke_scope='Test/No_Fire frames 1592-1913',
                  training_fog_scope='Test/No_Fire frames 0-1391 stay No_Fire per user confirmation',
                  limitations=['Previously used development validation, not independent test.',
                               'No confirmed fog/cloud negatives from the Test camera outside training.',
                               'Combined AUC mixes cameras/scenes and cannot prove same-camera smoke/fog discrimination.',
                               'Consecutive frames are correlated; image count is not independent scene count.'],
                  test_final_accessed=False)
    (PROJECT / 'backend/artifacts/label_audit/development_validation_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)


if __name__ == '__main__':
    main()
