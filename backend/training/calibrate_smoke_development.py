"""Select a development operating point from saved, freshly inferred scores."""
import csv
import json
from pathlib import Path
import numpy as np
from analyze_scores import confusion, threshold_for_recall

PROJECT = Path(__file__).resolve().parents[2]


def main():
    directory = PROJECT / 'backend/artifacts/label_audit'
    results = json.loads((directory / 'corrected_development_results.json').read_text())
    calibrated = {}
    for name in ('v3','v5'):
        with (directory / f'{name}_corrected_development_predictions.csv').open(newline='') as f:
            rows = list(csv.DictReader(f))
        labels=np.array([int(r['label']=='Fire') for r in rows])
        scores=np.array([float(r['fire_score']) for r in rows])
        masks={group:np.array([r['path'].startswith(prefix) and r['label']=='Fire' for r in rows])
               for group,prefix in [('confirmed_smoke','Test/No_Fire/'),('test_original_fire','Test/Fire/'),('old_domain_fire','Training/')]}
        threshold=min(threshold_for_recall(labels[m],scores[m],.95)['threshold'] for m in masks.values() if m.any())
        calibrated[name]={'threshold':threshold,'overall':confusion(labels,scores,threshold),
                          'per_positive_group':{group:confusion(labels[m],scores[m],threshold) for group,m in masks.items()}}
    selected=max(calibrated,key=lambda name:calibrated[name]['overall']['no_fire_specificity'])
    report=dict(selected=selected,rule='Maximize development No_Fire specificity while each Fire source group has empirical recall >=95%.',
                models=calibrated,model=results[selected]['model'],
                limitations='Development calibration only; negatives use the older camera. Same-camera smoke/fog performance needs new recordings.',
                weights_retrained=False,test_final_accessed=False)
    (directory/'development_calibration.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
