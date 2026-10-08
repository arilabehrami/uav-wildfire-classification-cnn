"""Compare the flame-only pilot with the existing app on the pilot development split."""
import csv
import json
import os
from pathlib import Path
import numpy as np
from analyze_scores import confusion,roc_auc
from evaluate_on_test import build_dataset

PROJECT=Path(__file__).resolve().parents[2]


def main():
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(min(8,os.cpu_count() or 4))
    with (PROJECT/'dataset/splits/validation_flame_pilot_v1.csv').open(newline='') as f:
        rows=list(csv.DictReader(f))
    labels=np.array([int(r['label']=='Fire') for r in rows])
    models={'current_v3':('models/fire_mobilenet_mixed_v3_finetuned.keras',0.02003161795437336),
            'flame_pilot_v1':('fire_flame_pilot_v1/model.keras',0.5),
            'flame_pilot_v2':('fire_flame_pilot_v2/model.keras',0.5)}
    results={}
    for name,(path,threshold) in models.items():
        model=tf.keras.models.load_model(PROJECT/'backend/artifacts'/path,compile=False)
        scores=model.predict(build_dataset(tf,rows,PROJECT/'dataset',16),verbose=0).reshape(-1)
        diagnostic=[dict(path='Test/No_Fire/resized_test_nofire_frame585.jpg',label='No_Fire')]
        diagnostic_score=float(model.predict(build_dataset(tf,diagnostic,PROJECT/'dataset',1),verbose=0)[0,0])
        results[name]=dict(roc_auc=roc_auc(labels,scores),at_operating_threshold=confusion(labels,scores,threshold),
                           at_threshold_0_5=confusion(labels,scores,0.5),
                           frame585_diagnostic=dict(fire_score=diagnostic_score,label='Fire' if diagnostic_score>=threshold else 'No_Fire',used_in_pilot_training=True))
        print(name,json.dumps(results[name]),flush=True)
        del model;tf.keras.backend.clear_session()
    report=dict(results=results,limitations='Small development validation, already used to select epochs. v3 has previous exposure to the broader recordings. Frame585 is a training diagnostic, not a held-out test.',test_final_accessed=False)
    (PROJECT/'backend/artifacts/flame_label_audit/pilot_comparison.json').write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
