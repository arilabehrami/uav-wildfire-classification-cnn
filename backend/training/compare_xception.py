"""Compare frozen MobileNet/Xception runs on validation, then export the winner."""
import argparse
import json
import re
import shutil
from pathlib import Path

import numpy as np

import fire_final as core


def compare(project, mobile_dir, xception_dir, output, batch_size):
    if output.exists():
        raise FileExistsError(f"Run ekziston: {output}. Perdor nje emer te ri.")
    mobile_plan = core.read_plan(project, mobile_dir)
    xception_plan = core.read_plan(project, xception_dir)
    if mobile_plan["manifest_sha256"] != xception_plan["manifest_sha256"]:
        raise ValueError("Ndarjet ose imazhet ndryshojne; krahasimi kerkon te njejtat lista te ngrira.")
    selections = [core.check_selected(directory) for directory in (mobile_dir, xception_dir)]
    started = json.loads((xception_dir / "training_started.json").read_text(encoding="utf-8"))
    if started.get("architecture") != "xception":
        raise ValueError("Run i dyte duhet te jete trajnuar me --architecture xception.")
    tf = core.tensorflow()
    rows = core.read_csv(mobile_dir / "validation.csv")
    data = core.dataset(tf, rows, project / "dataset", batch_size)
    models = [tf.keras.models.load_model(directory / "model.keras", compile=False)
              for directory in (mobile_dir, xception_dir)]
    for model in models:
        if tuple(model.input_shape[1:]) != (*core.INPUT_SIZE, 3) or model.output_shape[-1] != 1:
            raise ValueError("Kontrate modeli e papershtatshme.")
    scores = [model.predict(data, verbose=1).reshape(-1) for model in models]
    scores.append((scores[0] + scores[1]) / 2)
    names = ["mobilenet", "xception", "ensemble_50_50"]
    results = [core.metrics(rows, p) for p in scores]
    for name, result in zip(names, results):
        core.display(name, result)
    winner = min(range(3), key=lambda i: (results[i]["group_bce"], -results[i]["group_balanced_accuracy"]))
    inputs = tf.keras.Input(shape=(*core.INPUT_SIZE, 3), name="rgb_0_to_1")
    # Named wrappers avoid collisions between nested models in a saved Functional graph.
    branches = []
    for name, model in zip(names, models):
        branch_input = tf.keras.Input(shape=(*core.INPUT_SIZE, 3))
        branch = tf.keras.Model(branch_input, model(branch_input, training=False), name=f"{name}_branch")
        branches.append(branch(inputs, training=False))
    ensemble = tf.keras.Model(inputs, tf.keras.layers.Average()(branches), name="fire_ensemble")
    # Check export/serialization against separately calculated scores before publishing selection.
    output.mkdir(parents=True)
    for filename in ["plan.json", *mobile_plan["manifest_sha256"]]:
        shutil.copyfile(mobile_dir / filename, output / filename)
    ensemble.save(output / "ensemble.keras")
    restored = tf.keras.models.load_model(output / "ensemble.keras", compile=False)
    probe = next(iter(data))
    expected = (models[0](probe, training=False).numpy() + models[1](probe, training=False).numpy()) / 2
    np.testing.assert_allclose(restored(probe, training=False).numpy(), expected, rtol=1e-5, atol=1e-6)
    if winner == 2:
        source = output / "ensemble.keras"
    else:
        source = (mobile_dir, xception_dir)[winner] / "model.keras"
    for directory, selection in zip((mobile_dir, xception_dir), selections):
        if core.sha256(directory / "model.keras") != selection["model_sha256"]:
            raise ValueError("Modeli burim ndryshoi gjate krahasimit.")
    shutil.copyfile(source, output / "model.keras")
    for name, p in zip(names, scores):
        core.save_predictions(output / f"validation_{name}.csv", rows, p)
    core.save_predictions(output / "validation_predictions.csv", rows, scores[winner])
    selection = {
        "chosen_model": names[winner], "validation": results[winner],
        "comparisons": [{"model": name, "validation": result} for name, result in zip(names, results)],
        "sources": [{"run": directory.name, "model_sha256": item["model_sha256"]}
                    for directory, item in zip((mobile_dir, xception_dir), selections)],
        "ensemble_weights": [0.5, 0.5], "threshold": core.THRESHOLD,
        "selection_rule": mobile_plan["selection_rule"],
        "model_sha256": core.sha256(output / "model.keras"),
        "plan_sha256": core.sha256(output / "plan.json"), "limitation": core.LIMITATION,
    }
    core.write_json(output / "selection.json", selection)
    core.write_json(output / "inference.json", {
        "model": "model.keras", "model_sha256": selection["model_sha256"],
        "labels": core.LABELS, "threshold": core.THRESHOLD,
        "input_size": [254, 254, 3], "input_range": [0, 1], "color": "RGB",
        "resize": "tf.image.resize bilinear to 254x254; then /255 once",
        "output": "Fire score; not a calibrated confidence.",
    })
    print(f"\nZgjedhja nga validimi: {names[winner]}. Modeli: {output / 'model.keras'}")
    print(f"Tani: python backend/training/fire_final.py evaluate --run {output.name} --batch-size {batch_size}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--mobilenet-run", default="final_fire_v2")
    parser.add_argument("--xception-run", default="xception_v1")
    parser.add_argument("--run", default="ensemble_v1")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    for name in (args.mobilenet_run, args.xception_run, args.run):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
            parser.error("Emer run i pavlefshem.")
    if args.batch_size < 1:
        parser.error("batch-size duhet te jete pozitiv.")
    project = args.project_root.resolve()
    artifacts = project / "backend" / "artifacts"
    compare(project, artifacts / args.mobilenet_run, artifacts / args.xception_run,
            artifacts / args.run, args.batch_size)


if __name__ == "__main__":
    main()
