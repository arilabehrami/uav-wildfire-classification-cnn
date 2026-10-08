"""Fresh inference on the corrected development panel, with source-specific metrics."""
import csv
import json
import os
from pathlib import Path

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
import numpy as np
from analyze_scores import confusion, roc_auc, threshold_for_recall
from evaluate_on_test import build_dataset

PROJECT = Path(__file__).resolve().parents[2]


def main():
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(min(8, os.cpu_count() or 4))
    panel = PROJECT / 'dataset/splits/validation_smoke_development_v1.csv'
    with panel.open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    labels = np.array([int(r['label'] == 'Fire') for r in rows])
    out = PROJECT / 'backend/artifacts/label_audit'
    results = {}
    models = {'v3': PROJECT / 'backend/artifacts/models/fire_mobilenet_mixed_v3_finetuned.keras',
              'v5': PROJECT / 'backend/artifacts/fire_adapted_v5/model.keras'}
    for name, path in models.items():
        print(f'Loading {name}; {len(rows)} images', flush=True)
        model = tf.keras.models.load_model(path, compile=False)
        if tuple(model.input_shape[1:]) != (254,254,3) or model.output_shape[-1] != 1:
            raise ValueError('Model input/output contract mismatch')
        parts = []
        for i, batch in enumerate(build_dataset(tf, rows, PROJECT / 'dataset', 16)):
            parts.append(np.asarray(model(batch, training=False)).reshape(-1))
            if i % 10 == 0:
                print(f'{name}: {min((i+1)*16,len(rows))}/{len(rows)}', flush=True)
        scores = np.concatenate(parts)
        if len(scores) != len(rows) or not np.all(np.isfinite(scores)) or np.any((scores<0)|(scores>1)):
            raise ValueError('Invalid predictions')
        groups = {}
        for group, mask in {
            'confirmed_smoke': np.array([r['path'].startswith('Test/No_Fire/') for r in rows]),
            'test_original_fire': np.array([r['path'].startswith('Test/Fire/') for r in rows]),
            'old_domain_fire': np.array([r['path'].startswith('Training/') and r['label']=='Fire' for r in rows]),
            'old_domain_no_fire': np.array([r['path'].startswith('Training/') and r['label']=='No_Fire' for r in rows]),
        }.items():
            groups[group] = dict(count=int(mask.sum()), metrics=confusion(labels[mask],scores[mask],0.5))
        results[name] = dict(model=str(path.relative_to(PROJECT)), roc_auc=roc_auc(labels,scores),
                             at_threshold_0_5=confusion(labels,scores,0.5), per_group=groups,
                             exploratory_at_95_recall=threshold_for_recall(labels,scores,0.95))
        positive_masks = [np.array([r['path'].startswith(prefix) and r['label']=='Fire' for r in rows])
                          for prefix in ('Test/No_Fire/', 'Test/Fire/', 'Training/')]
        group_threshold = min(threshold_for_recall(labels[m],scores[m],0.95)['threshold']
                              for m in positive_masks if m.any())
        results[name]['exploratory_at_each_group_95_recall'] = {
            'overall': confusion(labels,scores,group_threshold),
            'positive_group_recalls': [confusion(labels[m],scores[m],group_threshold)['fire_recall']
                                      for m in positive_masks if m.any()],
        }
        with (out / f'{name}_corrected_development_predictions.csv').open('w',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=['path','label','fire_score']);w.writeheader()
            w.writerows(dict(**r,fire_score=float(s)) for r,s in zip(rows,scores))
        print(json.dumps(results[name],indent=2),flush=True)
        (out / 'corrected_development_results.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
        del model
        tf.keras.backend.clear_session()


if __name__ == '__main__':
    main()
