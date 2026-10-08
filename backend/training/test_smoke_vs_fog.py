"""Evaluate fresh Smoke (confirmed fire smoke) and Fog (no fire) image folders.

Run from the project root. This evaluates a fixed model/threshold; no training
or threshold search is performed. Use new recordings from the same camera.
"""
import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
from analyze_scores import confusion, roc_auc

PROJECT = Path(__file__).resolve().parents[2]
EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', type=Path, default=PROJECT / 'dataset/smoke_fog_check')
    parser.add_argument('--model', type=Path, default=PROJECT / 'backend/artifacts/models/fire_mobilenet_mixed_v3_finetuned.keras')
    parser.add_argument('--threshold', type=float, default=0.02003161795437336)
    parser.add_argument('--output', type=Path, default=PROJECT / 'backend/artifacts/smoke_fog_check')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if not 0 <= args.threshold <= 1:
        parser.error('--threshold must be in [0,1]')
    root = args.input_root.resolve()
    rows = []
    for group, label in [('Smoke','Fire'), ('Fog','No_Fire')]:
        paths = sorted(p for p in (root / group).rglob('*') if p.is_file() and p.suffix.lower() in EXTENSIONS)
        if not paths:
            parser.error(f'Vendos foto te reja ne {root / group}; kjo kategori eshte bosh.')
        rows.extend(dict(path=p.relative_to(root).as_posix(), label=label, group=group) for p in paths)
        print(f'{group}: {len(paths)} foto', flush=True)
    if args.dry_run:
        print('Vetem kontroll i listave; nuk u trajnua ose ekzekutua modeli.')
        return
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    import tensorflow as tf
    from evaluate_on_test import build_dataset
    tf.config.threading.set_intra_op_parallelism_threads(min(8, os.cpu_count() or 4))
    model = tf.keras.models.load_model(args.model, compile=False)
    if tuple(model.input_shape[1:]) != (254,254,3) or model.output_shape[-1] != 1:
        raise ValueError('Pritet RGB 254x254, [0,1], nje dalje sigmoid.')
    scores = model.predict(build_dataset(tf,rows,root,16),verbose=1).reshape(-1)
    if len(scores)!=len(rows) or not np.all(np.isfinite(scores)) or np.any((scores<0)|(scores>1)):
        raise ValueError('Rezultate te pavlefshme nga modeli.')
    labels = np.array([int(r['label']=='Fire') for r in rows])
    metrics = confusion(labels,scores,args.threshold)
    args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'predictions.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['path','group','label','fire_score','prediction','correct'])
        writer.writeheader()
        for row,score in zip(rows,scores):
            prediction='Fire' if score>=args.threshold else 'No_Fire'
            writer.writerow(dict(**row,fire_score=float(score),prediction=prediction,correct=prediction==row['label']))
    report=dict(model=str(args.model.resolve()),input_root=str(root),threshold=args.threshold,
                metrics=metrics,roc_auc=roc_auc(labels,scores),
                note='Validity depends on new images, confirmed labels and independent recordings. No automatic leakage or provenance check; correlated frames are not independent samples.')
    (args.output/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(f"\nTym zjarri i kapur: {metrics['TP']}/{metrics['TP']+metrics['FN']} ({metrics['fire_recall']:.1%})")
    print(f"Mjegull e dalluar si No_Fire: {metrics['TN']}/{metrics['TN']+metrics['FP']} ({metrics['no_fire_specificity']:.1%})")
    print(f"Alarme te rreme mbi mjegull: {metrics['FP']}")
    print(f'Rezultatet: {args.output.resolve()}',flush=True)


if __name__=='__main__':
    main()
