"""Build flame-only splits from a completed review CSV; refuse partial reviews.

Usage: python backend/training/prepare_reviewed_flame_splits.py --review FILE.csv
The review must cover the full flame_label_audit/review_manifest.csv. Uncertain
images are excluded. Original files and test_final are never edited.
"""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]


def read(path):
    with path.open(encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review',required=True,type=Path)
    args=parser.parse_args()
    audit=PROJECT/'backend/artifacts/flame_label_audit'
    manifest=read(audit/'review_manifest.csv')
    expected={(r['split'],r['path']) for r in manifest}
    reviewed={}
    for row in read(args.review):
        key=(row['split'],row['path'])
        if key not in expected or key in reviewed:
            raise SystemExit(f'Unknown or duplicate review: {key}')
        if row['reviewed_label'] not in ('Fire','No_Fire','Uncertain') or not row.get('reason','').strip():
            raise SystemExit(f'Review needs a valid decision and reason: {key}')
        reviewed[key]=row
    missing=expected-reviewed.keys()
    if missing:
        raise SystemExit(f'Audit incomplete: {len(reviewed)}/{len(expected)} reviewed; {len(missing)} remaining. No training splits written.')
    train=[];validation=[];uncertain=[]
    for r in manifest:
        decision=reviewed[(r['split'],r['path'])]
        if decision['reviewed_label']=='Uncertain':
            uncertain.append(r['path']);continue
        dest=train if r['split']=='train_with_adapt' else validation
        dest.append(dict(path=r['path'],label=decision['reviewed_label']))
    if {r['path'] for r in train}&{r['path'] for r in validation}:
        raise SystemExit('Train/validation path overlap.')
    for rows in (train,validation):
        if {r['label'] for r in rows}!={'Fire','No_Fire'}:
            raise SystemExit('Both classes are required in train and validation.')
    for name,rows in [('train_flame_reviewed_v1',train),('validation_flame_reviewed_v1',validation)]:
        with (PROJECT/'dataset/splits'/f'{name}.csv').open('w',encoding='utf-8',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=['path','label']);writer.writeheader();writer.writerows(rows)
    (audit/'split_report.json').write_text(json.dumps(dict(policy='visible flames only',train=dict(Counter(r['label'] for r in train)),validation=dict(Counter(r['label'] for r in validation)),uncertain_excluded=len(uncertain),review_source=str(args.review.resolve()),test_final_accessed=False),indent=2),encoding='utf-8')
    print('Reviewed splits created. Run leakage filtering before training; path disjointness alone is insufficient.')


if __name__=='__main__':
    main()
