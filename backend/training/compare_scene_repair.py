"""Compare frozen-backbone models on development validation and repaired scene.

Reuse features only after verifying both extractors have identical weights.
The repaired scene is a training diagnostic, never an independent test.
"""
import csv
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import tensorflow as tf
from fit_flame_gates import fit_gates, summarize
from train_adapted import group_metrics

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "backend/artifacts"
NEW = "fire_flame_scene_repair_v1"
OLD = "fire_flame_v6"
OUT = ART / NEW


def read(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main():
    tf.config.threading.set_intra_op_parallelism_threads(min(8, os.cpu_count() or 4))
    models = {name: tf.keras.models.load_model(ART / name / "model.keras", compile=False)
              for name in (OLD, NEW)}
    extractors = {name: tf.keras.Model(model.input, model.get_layer("avg_and_max").output)
                  for name, model in models.items()}
    old_weights, new_weights = [extractors[name].get_weights() for name in (OLD, NEW)]
    if len(old_weights) != len(new_weights) or any(
            not np.array_equal(a, b) for a, b in zip(old_weights, new_weights)):
        raise SystemExit("Extractors differ; cannot share features")
    del old_weights, new_weights
    heads = {name: model.get_layer("fire_head") for name, model in models.items()}
    validation = read(ROOT / "dataset/splits/validation_flame_scene_repair_v1.csv")
    for r in validation:
        r["split"] = "validation"
    scene = [dict(path=r["path"], label="Fire", split="training_diagnostic")
             for r in read(ART / "flame_label_audit/review_2026-10-20/scene_review_confirmed.csv")]
    all_scores = {}
    for group, rows in (("training_scene", scene), ("development_validation", validation)):
        parts = {name: [] for name in models}
        for i in range(0, len(rows), 32):
            batch = tf.stack([tf.image.resize(tf.cast(tf.io.decode_image(
                tf.io.read_file(str(ROOT / "dataset" / r["path"])), channels=3,
                expand_animations=False), tf.float32), (254, 254)) / 255.0
                for r in rows[i:i+32]])
            features = extractors[NEW](batch, training=False)
            for name, head in heads.items():
                parts[name].append(np.asarray(head(features, training=False)).ravel())
            if i % 320 == 0:
                print(f"{group}: {i}/{len(rows)}", flush=True)
        all_scores[group] = {name: np.concatenate(values) for name, values in parts.items()}
        with (OUT / f"comparison_{group}_scores.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["path", "label", OLD, NEW])
            w.writerows([r["path"], r["label"], *[float(all_scores[group][n][i]) for n in models]]
                        for i, r in enumerate(rows))
        print(f"{group}: done", flush=True)
    truth = np.array([r["label"] == "Fire" for r in validation], dtype=int)
    hard = np.array(["lake" not in r["path"] for r in validation])
    gates = json.loads((ART / OLD / "inference.json").read_text())["gates"]
    low, high = fit_gates(truth[hard], all_scores["development_validation"][NEW][hard])
    if not 0 <= low < high <= 1:
        raise SystemExit("Invalid fitted gates; comparison scores saved, no contract created")
    report = {"limitation": "Development validation and training diagnostics only; no independent test", "models": {}}
    for name in models:
        lo, hi = (gates["low"], gates["high"]) if name == OLD else (low, high)
        report["models"][name] = dict(
            gates=dict(low=lo, high=hi),
            validation_at_0_5=group_metrics(validation, all_scores["development_validation"][name]),
            validation_zones=summarize(truth, all_scores["development_validation"][name], lo, hi),
            training_scene_zones=summarize(np.ones(len(scene)), all_scores["training_scene"][name], lo, hi))
    contract = json.loads((ART / OLD / "inference.json").read_text())
    for key in ("measured_on_test_final", "precision_note"):
        contract.pop(key, None)
    contract["model_sha256"] = hashlib.sha256((OUT / "model.keras").read_bytes()).hexdigest()
    contract["gates"].update(low=low, high=high,
        fitted_on="validation_flame_scene_repair_v1, excluding lake images")
    contract["limitation"] = report["limitation"] + "; 201 visually reviewed flame frames repaired; other labels inherited from v1."
    (OUT / "inference.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")
    (OUT / "comparison.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
