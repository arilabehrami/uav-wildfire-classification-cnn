"""Fire / No_Fire: prepare -> train -> evaluate; komandat jane ne README_FIRE_FINAL.md.

Input i modelit: RGB, float32 [0,1], (254,254,3). Sigmoid = score Fire.
Ndarjet jane temporale nga burime te njohura, jo fluturime te pavarura.
"""
import argparse
import csv
import gc
import hashlib
import json
import math
import os
import re
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import numpy as np

VERSION = "final_fire_1"
LABELS = {"No_Fire": 0, "Fire": 1}
THRESHOLD = 0.5
INPUT_SIZE = (254, 254)
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
SOURCE_SPLITS = {
    "train": ["train_mixed_v3"],
    "validation": ["validation_temporal", "validation_extra_v3"],
    "diagnostic_202": ["holdout_extra_v2"],
}
BASELINES = ["fire_mobilenet_mixed_v3_finetuned.keras", "fire_mobilenet_avgmax_v4.keras"]
LIMITATION = (
    "Temporal validation on known image sequences. No verified flight/scene IDs were supplied. "
    "The 202 extra images and the existing Test folder have been used in earlier experiments; "
    "neither is claimed to be a fresh independent final test."
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def write_csv(path, rows, fields):
    with Path(path).open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def normalized_path(value):
    path = PurePosixPath(value.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts or ":" in str(path) or not path.parts:
        raise ValueError(f"Rruge jo relative ndaj dataset: {value}")
    return path.as_posix()


def stratum(row):
    """Strata per balancim/raportim; nuk jane ID fluturimesh."""
    path = PurePosixPath(row["path"])
    parts = path.parts
    if parts[0] in {"Training", "Test"}:
        if len(parts) < 3 or parts[1] != row["label"]:
            raise ValueError(f"Etiketa nuk perputhet me dosjen: {row}")
        if parts[0] == "Test":
            return "dataset_test"
        if path.name.startswith("lake_"):
            if row["label"] != "No_Fire":
                raise ValueError(f"Etikete e papritur per lake: {row}")
            return "base_lake"
        if not path.name.startswith("resized_frame"):
            raise ValueError(f"Prefiks i panjohur ne dataset-in baze: {path}")
        return "base_forest"
    folders = {
        "Pre-Evening Fire Incident_raw_img": ("Fire", "extra_pre_evening_fire"),
        "Pre-evening Forest condition_raw_img": ("No_Fire", "extra_pre_evening_forest"),
        "Evening Forest condition_raw_img": ("No_Fire", "extra_evening_forest"),
    }
    if len(parts) >= 3 and parts[0] == "extra_raw" and parts[1] in folders:
        label, group = folders[parts[1]]
        if row["label"] != label:
            raise ValueError(f"Etikete e gabuar ne dosjen shtese: {row}")
        return group
    raise ValueError(f"Burim i panjohur: {path}; kerkohet etikete/burim i verifikuar.")


def validate_rows(rows, name):
    if not rows or {r["label"] for r in rows} != set(LABELS):
        raise ValueError(f"{name} duhet te permbaje te dyja klasat.")
    for row in rows:
        row["path"] = normalized_path(row["path"])
        row["stratum"] = stratum(row)
    paths = [r["path"].casefold() for r in rows]
    if len(paths) != len(set(paths)):
        raise ValueError(f"Rruge te perseritura ne {name}.")


def check_disjoint(splits):
    seen = {}
    for name, rows in splits.items():
        for row in rows:
            key = row["path"].casefold()
            if key in seen:
                raise ValueError(f"Foto e perbashket: {row['path']} ({seen[key]}, {name}).")
            seen[key] = name


def load_sources(project):
    directory = project / "dataset" / "splits"
    splits, hashes = {}, {}
    for role, names in SOURCE_SPLITS.items():
        splits[role] = []
        for name in names:
            path = directory / f"{name}.csv"
            hashes[path.relative_to(project).as_posix()] = sha256(path)
            splits[role].extend(dict(r, source_split=name) for r in read_csv(path))
        validate_rows(splits[role], role)
    test = []
    test_root = project / "dataset" / "Test"
    if test_root.exists():
        for label in LABELS:
            for path in sorted((test_root / label).rglob("*")):
                if path.is_file() and path.suffix.lower() in EXTENSIONS:
                    test.append({"path": path.relative_to(project / "dataset").as_posix(),
                                 "label": label, "source_split": "Test"})
        validate_rows(test, "known_test")
        splits["known_test"] = test
    check_disjoint(splits)
    return splits, hashes


def choose_training(rows):
    cells = defaultdict(list)
    for row in rows:
        cells[(row["label"], row["stratum"])].append(row)
    selected = []
    for (label, group), members in sorted(cells.items()):
        def order(row):
            match = re.search(r"(\d+)$", PurePosixPath(row["path"]).stem)
            if match is None:
                raise ValueError(f"Mungon numri i frame: {row['path']}")
            return int(match[1]), row["path"]
        members.sort(key=order)
        cap = 1000 if group == "base_lake" else 6000 if (label, group) == ("Fire", "base_forest") else len(members)
        # Perzgjedhje e shperndare ne gjithe segmentin, pa perdorur parashikime.
        indices = np.linspace(0, len(members) - 1, min(cap, len(members)), dtype=int)
        selected.extend(dict(members[i]) for i in indices)
    counts = Counter((r["label"], r["stratum"]) for r in selected)
    denominator = {label: sum(math.sqrt(n) for (lab, _), n in counts.items() if lab == label) for label in LABELS}
    for row in selected:
        n = counts[(row["label"], row["stratum"])]
        # 50% peshe per klase. Brenda klases, masa e nje strati eshte sqrt(n).
        # Kjo e zbut dominimin e grupeve te medha pa peshuar 99 foto si 6000 foto.
        row["sample_weight"] = len(selected) * 0.5 / (math.sqrt(n) * denominator[row["label"]])
    return selected


def summary(rows):
    return dict(sorted(Counter(f"{r['label']} / {r['stratum']}" for r in rows).items()))


def verify_image_files(data_root, splits, output):
    """Identike do te thote byte-identike; ky kontroll nuk provon pavaresi skenash."""
    from PIL import Image
    digests, inventory, duplicates, errors = {}, [], [], []
    total = sum(map(len, splits.values()))
    count = 0
    for role, rows in splits.items():
        for row in rows:
            path = data_root / row["path"]
            if not path.is_file():
                raise FileNotFoundError(f"Mungon imazhi: {path}")
            digest = sha256(path)
            with Image.open(path) as image:
                image.verify()
            if digest in digests:
                previous = digests[digest]
                issue = {"first": previous["path"], "second": row["path"],
                         "first_role": previous["role"], "second_role": role,
                         "conflicting_labels": previous["label"] != row["label"]}
                duplicates.append(issue)
                if previous["role"] != role or issue["conflicting_labels"]:
                    errors.append(issue)
            else:
                digests[digest] = dict(row, role=role)
            stat = path.stat()
            inventory.append({"path": row["path"], "sha256": digest,
                              "size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
            count += 1
            if count % 2000 == 0 or count == total:
                print(f"Kontroll imazhesh: {count}/{total}", flush=True)
    write_json(output / "duplicate_check.json", {
        "method": "SHA256 of file bytes; not near-duplicate or scene detection",
        "duplicates": duplicates, "blocking_errors": errors})
    if errors:
        raise ValueError("Foto identike mes ndarjeve ose me etiketa te kunderta. Shiko duplicate_check.json; trajnimi nuk u nis.")
    write_csv(output / "image_inventory.csv", inventory, ["path", "sha256", "size", "mtime_ns"])


def prepare(project, output):
    if (output / "plan.json").exists():
        read_plan(project, output)
        print(f"Pergatitja ekziston dhe listat perputhen: {output}")
        return
    output.mkdir(parents=True, exist_ok=True)
    splits, source_hashes = load_sources(project)
    # Kontrollon edhe trajnimin origjinal te baseline-ve, jo vetem nenbashkesine e re.
    verify_image_files(project / "dataset", splits, output)
    original_counts = summary(splits["train"])
    splits["train"] = choose_training(splits["train"])
    manifests = {}
    for name, rows in splits.items():
        fields = ["path", "label", "source_split", "stratum"]
        if name == "train":
            fields.append("sample_weight")
        destination = output / f"{name}.csv"
        write_csv(destination, rows, fields)
        manifests[destination.name] = sha256(destination)
        print(f"{name}: {len(rows)} foto; {summary(rows)}")
    manifests["image_inventory.csv"] = sha256(output / "image_inventory.csv")
    plan = {"version": VERSION, "created_utc": datetime.now(timezone.utc).isoformat(),
            "source_csv_sha256": source_hashes, "manifest_sha256": manifests,
            "original_train_counts": original_counts, "counts": {n: summary(r) for n, r in splits.items()},
            "threshold": THRESHOLD, "labels": LABELS, "limitation": LIMITATION,
            "selection_rule": "Minimum validation group_BCE: equal class weight, equal weight per stratum within class; threshold fixed at 0.5.",
            "sampling": "Evenly spaced frame indices; base Fire cap 6000, lake cap 1000; all remaining training rows retained.",
            "training_weights": "Each class totals 50%; within each class, stratum weight mass proportional to sqrt(selected count).",
            "image_check": "File integrity + byte SHA256. Temporal correlation and re-encoded duplicates can remain."}
    write_json(output / "plan.json", plan)
    print("Pergatitja perfundoi. Origjinalet dhe 202 fotot e kontrollit nuk u ndryshuan.")


def read_plan(project, output):
    path = output / "plan.json"
    if not path.is_file():
        raise FileNotFoundError("Se pari ekzekuto komanden prepare.")
    plan = json.loads(path.read_text(encoding="utf-8"))
    if plan["version"] != VERSION:
        raise ValueError("Version tjeter i protokollit.")
    for name, expected in plan["source_csv_sha256"].items():
        if sha256(project / name) != expected:
            raise ValueError(f"Lista origjinale ndryshoi pas pergatitjes: {name}")
    for name, expected in plan["manifest_sha256"].items():
        if sha256(output / name) != expected:
            raise ValueError(f"Lista e ngrire ndryshoi: {name}")
    for row in read_csv(output / "image_inventory.csv"):
        path = project / "dataset" / row["path"]
        stat = path.stat()
        if stat.st_size != int(row["size"]) or stat.st_mtime_ns != int(row["mtime_ns"]):
            if sha256(path) != row["sha256"]:
                raise ValueError(f"Imazhi ndryshoi pas pergatitjes: {path}")
    return plan


def tensorflow():
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(min(8, os.cpu_count() or 4))
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.keras.utils.set_random_seed(42)
    return tf


def dataset(tf, rows, data_root, batch_size, *, training=False, augment=False):
    paths = [str(data_root / row["path"]) for row in rows]
    labels = np.asarray([LABELS[r["label"]] for r in rows], dtype=np.float32)
    weights = np.asarray([float(r.get("sample_weight", 1)) for r in rows], dtype=np.float32)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels, weights))
    if training:
        ds = ds.shuffle(len(rows), seed=42, reshuffle_each_iteration=True)
    def read(path, label, weight):
        image = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
        image.set_shape([None, None, 3])
        image = tf.image.resize(tf.cast(image, tf.float32), INPUT_SIZE) / 255.0
        return image, tf.reshape(label, (1,)), weight
    ds = ds.map(read, num_parallel_calls=2, deterministic=True).batch(batch_size)
    if training or augment:
        augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal", seed=42),
            tf.keras.layers.RandomBrightness(0.08, value_range=(0, 1), seed=43),
            tf.keras.layers.RandomContrast(0.10, seed=44),
        ])
        def change(images, labels, weights):
            return tf.clip_by_value(augmentation(images, training=True), 0., 1.), labels, weights
        ds = ds.map(change, num_parallel_calls=1, deterministic=True)
    if not training:
        ds = ds.map(lambda images, labels, weights: images, num_parallel_calls=1)
    options = tf.data.Options()
    options.experimental_deterministic = True
    options.threading.private_threadpool_size = 2
    return ds.with_options(options).prefetch(1)


def build_models(tf, weights="imagenet"):
    base = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False,
                                            weights=weights, pooling="avg")
    base.trainable = False
    inputs = tf.keras.Input(shape=(*INPUT_SIZE, 3), name="rgb_0_to_1")
    x = tf.keras.layers.Resizing(224, 224, name="resize_224")(inputs)
    x = tf.keras.layers.Rescaling(2.0, offset=-1.0, name="mobilenet_scaling")(x)
    features = base(x, training=False)
    extractor = tf.keras.Model(inputs, features)
    head = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(1280,)),
        tf.keras.layers.Dropout(0.35),
        tf.keras.layers.Dense(1, activation="sigmoid", kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
    ], name="fire_head")
    model = tf.keras.Model(inputs, head(features), name="fire_mobilenet_final")
    return model, extractor, head


def enable_finetuning(tf, model):
    bases = [layer for layer in model.layers if layer.name.startswith("mobilenetv2")]
    if len(bases) != 1:
        raise ValueError("Nuk u identifikua MobileNetV2.")
    base = bases[0]
    base.trainable = True
    reached = False
    for layer in base.layers:
        if layer.name == "block_13_expand":
            reached = True
        layer.trainable = reached and not isinstance(layer, tf.keras.layers.BatchNormalization)
    if not reached:
        raise ValueError("Nuk u gjet block_13_expand.")


def metrics(rows, probabilities):
    p = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    y = np.asarray([LABELS[r["label"]] for r in rows], dtype=int)
    if not len(y) or len(p) != len(y) or not np.all(np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError("Parashikime te pavlefshme ose renditje/gjatesi e papershtatshme.")
    pred = p >= THRESHOLD
    tn = int(np.sum((y == 0) & ~pred)); fp = int(np.sum((y == 0) & pred))
    fn = int(np.sum((y == 1) & ~pred)); tp = int(np.sum((y == 1) & pred))
    ratio = lambda numerator, denominator: numerator / denominator if denominator else None
    cells = defaultdict(list)
    for i, row in enumerate(rows):
        cells[(row["label"], row["stratum"])].append(i)
    clipped = np.clip(p, 1e-7, 1 - 1e-7)
    bce = -(y * np.log(clipped) + (1 - y) * np.log1p(-clipped))
    per_cell = {}
    for (label, group), indices in sorted(cells.items()):
        per_cell[f"{label} / {group}"] = {"label": label, "total": len(indices),
            "correct": int(np.sum(pred[indices] == y[indices])),
            "recall": float(np.mean(pred[indices] == y[indices])),
            "bce": float(np.mean(bce[indices]))}
    class_bce = [float(np.mean([v["bce"] for v in per_cell.values() if v["label"] == label]))
                 for label in LABELS if np.any(y == LABELS[label])]
    class_recall = [float(np.mean([v["recall"] for v in per_cell.values() if v["label"] == label]))
                    for label in LABELS if np.any(y == LABELS[label])]
    balanced_accuracy = ((tp / (tp + fn) + tn / (tn + fp)) / 2
                         if (tp + fn) and (tn + fp) else None)
    return {"n": len(y), "threshold": THRESHOLD, "confusion_matrix": [[tn, fp], [fn, tp]],
            "confusion_order": ["No_Fire", "Fire"], "accuracy": (tn + tp) / len(y),
            "balanced_accuracy": balanced_accuracy,
            "fire_precision": ratio(tp, tp + fp), "fire_recall": ratio(tp, tp + fn),
            "no_fire_specificity": ratio(tn, tn + fp), "false_positives": fp, "false_negatives": fn,
            "false_positive_rate": ratio(fp, tn + fp), "false_negative_rate": ratio(fn, tp + fn),
            "fire_f1": ratio(2 * tp, 2 * tp + fp + fn), "group_bce": float(np.mean(class_bce)),
            "group_balanced_accuracy": float(np.mean(class_recall)), "per_stratum": per_cell,
            "zero_errors_on_this_set": fp == 0 and fn == 0}


def display(name, result):
    tn, fp = result["confusion_matrix"][0]; fn, tp = result["confusion_matrix"][1]
    print(f"\n{name}: Fire {tp}/{tp+fn}; No_Fire {tn}/{tn+fp}; alarme te rreme {fp}; zjarre te humbura {fn}")
    print(f"  accuracy={result['accuracy']:.4f}; group_BCE={result['group_bce']:.5f}")
    for name, item in result["per_stratum"].items():
        print(f"  {name}: {item['correct']}/{item['total']}")


def prediction_rows(rows, p):
    return [dict(row, fire_score=float(score), predicted_label="Fire" if score >= THRESHOLD else "No_Fire")
            for row, score in zip(rows, np.asarray(p).reshape(-1))]


def save_predictions(path, rows, p):
    write_csv(path, prediction_rows(rows, p), ["path", "label", "source_split", "stratum", "fire_score", "predicted_label"])


def validation_callback(tf, rows, validation_input, full_model, output, stage, patience, initial_best=float("inf")):
    class Validation(tf.keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.best = initial_best
            self.wait = 0
            self.records = []

        def on_epoch_end(self, epoch, logs=None):
            p = self.model.predict(validation_input, verbose=0).reshape(-1)
            result = metrics(rows, p)
            if logs is None or not np.isfinite(float(logs.get("loss", float("nan")))):
                raise FloatingPointError("Loss i pavlefshem; nuk ruhet model perfundimtar.")
            logs["val_group_bce"] = result["group_bce"]
            logs["val_group_balanced_accuracy"] = result["group_balanced_accuracy"]
            improved = result["group_bce"] < self.best - 1e-7
            record = {"stage": stage, "epoch": epoch + 1, "train_loss": float(logs["loss"]),
                      "improved": improved, "validation": result}
            self.records.append(record)
            display(f"{stage}, epoka {epoch+1}", result)
            if improved:
                self.best = result["group_bce"]
                self.wait = 0
                full_model.save(output / "candidate.keras")
                write_json(output / "best_candidate.json", record)
            else:
                self.wait += 1
            write_json(output / f"history_{stage}.json", self.records)
            if self.wait >= patience:
                print(f"Ndalje e hershme: {patience} epoka pa permiresim te validimit.")
                self.model.stop_training = True
    return Validation()


def compile_model(tf, model, learning_rate):
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate, clipnorm=1.0),
                  loss="binary_crossentropy", metrics=["accuracy"])


def train(project, output, batch_size=16, head_epochs=30, fine_epochs=4, *, imagenet_weights="imagenet"):
    plan = read_plan(project, output)
    if (output / "selection.json").exists():
        check_selected(output)
        print("Trajnimi dhe zgjedhja kane perfunduar. Vazhdo me evaluate.")
        return
    if (output / "training_started.json").exists():
        raise FileExistsError("Ka nje trajnim te nderprere ne kete run. Ruaje dosjen dhe perdor --run final_fire_v2 per nje nisje te re; mos fshi modelet e vjetra.")
    tf = tensorflow()
    train_rows = read_csv(output / "train.csv")
    val_rows = read_csv(output / "validation.csv")
    data_root = project / "dataset"
    # Ngarkimi i peshave behet para marker-it: nese shkarkimi deshton mund te riprovohet.
    model, extractor, head = build_models(tf, weights=imagenet_weights)
    write_json(output / "training_started.json", {"tf_version": tf.__version__, "numpy_version": np.__version__,
        "seed": 42, "head_max_epochs": head_epochs, "fine_max_epochs": fine_epochs,
        "batch_size": batch_size, "head_lr": 0.001, "fine_lr": 1e-5,
        "plan_sha256": sha256(output / "plan.json"), "script_sha256": sha256(Path(__file__)),
        "initialization": "ImageNet" if imagenet_weights == "imagenet" else "TEST ONLY: random weights"})
    print("\nNxjerrje vecorish ImageNet: train origjinal")
    original = extractor.predict(dataset(tf, train_rows, data_root, batch_size), verbose=1)
    print("\nNxjerrje vecorish ImageNet: nje pamje e transformuar e seciles foto")
    augmented = extractor.predict(dataset(tf, train_rows, data_root, batch_size, augment=True), verbose=1)
    print("\nNxjerrje vecorish ImageNet: validim i plote")
    validation_data = dataset(tf, val_rows, data_root, batch_size)
    val_features = extractor.predict(validation_data, verbose=1)
    features = np.concatenate([original, augmented])
    labels = np.tile(np.asarray([LABELS[r["label"]] for r in train_rows], dtype=np.float32), 2)
    weights = np.tile(np.asarray([float(r["sample_weight"]) for r in train_rows], dtype=np.float32), 2)
    compile_model(tf, head, 0.001)
    callback = validation_callback(tf, val_rows, val_features, model, output, "head", 5)
    head.fit(features, labels, sample_weight=weights, batch_size=128, epochs=head_epochs,
             shuffle=True, callbacks=[callback], verbose=2)
    del original, augmented, features, val_features, head, model, extractor, callback
    gc.collect()
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(43)
    model = tf.keras.models.load_model(output / "candidate.keras", compile=False)
    if fine_epochs:
        best = json.loads((output / "best_candidate.json").read_text(encoding="utf-8"))
        enable_finetuning(tf, model)
        compile_model(tf, model, 1e-5)
        callback = validation_callback(tf, val_rows, validation_data, model, output, "finetune", 2,
                                       initial_best=best["validation"]["group_bce"])
        print("\nFine-tuning: block_13 e tutje; BatchNormalization mbetet frozen.")
        model.fit(dataset(tf, train_rows, data_root, batch_size, training=True), epochs=fine_epochs,
                  shuffle=False, callbacks=[callback], verbose=2)
    del model
    gc.collect()
    # Vetem validimi perdoret per zgjedhje. Kontrolli 202 dhe Test lexohen nga evaluate.
    candidates = [output / "candidate.keras"]
    candidates += [project / "backend" / "artifacts" / "models" / name for name in BASELINES
                   if (project / "backend" / "artifacts" / "models" / name).is_file()]
    comparisons = []
    best_path, best_p, best_rank = None, None, None
    for path in candidates:
        tf.keras.backend.clear_session()
        candidate = tf.keras.models.load_model(path, compile=False)
        if tuple(candidate.input_shape[1:]) != (*INPUT_SIZE, 3) or candidate.output_shape[-1] != 1:
            raise ValueError(f"Model me kontrate tjeter: {path}")
        p = candidate.predict(validation_data, verbose=1).reshape(-1)
        result = metrics(val_rows, p)
        display(path.name, result)
        comparisons.append({"model": path.name, "sha256": sha256(path), "validation": result})
        rank = (result["group_bce"], -result["group_balanced_accuracy"])
        if best_rank is None or rank < best_rank:
            best_path, best_p, best_rank = path, p.copy(), rank
        del candidate
        gc.collect()
    shutil.copyfile(best_path, output / "model.keras")
    save_predictions(output / "validation_predictions.csv", val_rows, best_p)
    chosen_metrics = metrics(val_rows, best_p)
    selection = {"chosen_model": best_path.name, "chosen_new_candidate": best_path == output / "candidate.keras",
                 "comparisons": comparisons, "validation": chosen_metrics,
                 "model_sha256": sha256(output / "model.keras"), "plan_sha256": sha256(output / "plan.json"),
                 "selection_rule": plan["selection_rule"], "threshold": THRESHOLD,
                 "limitation": LIMITATION}
    write_json(output / "inference.json", {"model": "model.keras", "model_sha256": selection["model_sha256"],
        "labels": LABELS, "threshold": THRESHOLD, "input_size": [254, 254, 3], "input_range": [0, 1],
        "color": "RGB", "resize": "tf.image.resize bilinear to 254x254; then /255 once",
        "output": "Sigmoid Fire score; not a guarantee or a calibrated confidence."})
    write_json(output / "selection.json", selection)
    print(f"\nZGJEDHJA PERFUNDOI: {best_path.name}\nModeli per perdorim: {output / 'model.keras'}")
    print("Tani ekzekuto evaluate. Zgjedhja dhe pragu jane ngrire.")


def check_selected(output):
    selection = json.loads((output / "selection.json").read_text(encoding="utf-8"))
    if sha256(output / "model.keras") != selection["model_sha256"]:
        raise ValueError("Modeli ndryshoi pas zgjedhjes.")
    if sha256(output / "plan.json") != selection["plan_sha256"]:
        raise ValueError("Protokolli ndryshoi pas zgjedhjes.")
    return selection


def evaluate(project, output, batch_size=16):
    plan = read_plan(project, output)
    selection = check_selected(output)
    result_path = output / "evaluation.json"
    if result_path.exists():
        report = json.loads(result_path.read_text(encoding="utf-8"))
        if report["model_sha256"] != selection["model_sha256"]:
            raise ValueError("Raporti i perket nje modeli tjeter.")
        for name, result in report["evaluations"].items():
            display(name, result)
        print(f"Raporti i ruajtur: {result_path}")
        return
    tf = tensorflow()
    model = tf.keras.models.load_model(output / "model.keras", compile=False)
    evaluations = {}
    for name in ("diagnostic_202", "known_test"):
        csv_path = output / f"{name}.csv"
        if csv_path.name not in plan["manifest_sha256"]:
            print(f"{name}: nuk ka liste; mungon dataset/Test ne kete projekt.")
            continue
        rows = read_csv(csv_path)
        p = model.predict(dataset(tf, rows, project / "dataset", batch_size), verbose=1).reshape(-1)
        evaluations[name] = metrics(rows, p)
        save_predictions(output / f"{name}_predictions.csv", rows, p)
        display(name, evaluations[name])
    report = {"chosen_model": selection["chosen_model"], "model_sha256": selection["model_sha256"],
              "threshold": THRESHOLD, "validation": selection["validation"], "evaluations": evaluations,
              "fresh_independent_test": False, "limitation": LIMITATION}
    write_json(result_path, report)
    print(f"\nTrajnimi dhe vleresimi perfunduan. Raporti: {result_path}")
    print("Rezultatet raportohen sic jane; ky kod nuk garanton zero gabime ne pamje te reja.")


def predict(output, image):
    check_selected(output)
    tf = tensorflow()
    model = tf.keras.models.load_model(output / "model.keras", compile=False)
    raw = tf.io.decode_image(tf.io.read_file(str(image)), channels=3, expand_animations=False)
    x = tf.image.resize(tf.cast(raw, tf.float32), INPUT_SIZE) / 255.0
    score = float(model(x[None, ...], training=False).numpy().reshape(-1)[0])
    print(json.dumps({"image": str(image), "label": "Fire" if score >= THRESHOLD else "No_Fire",
                      "fire_score": score, "threshold": THRESHOLD}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "train", "evaluate", "predict"])
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--run", default="final_fire_v1")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--head-epochs", type=int, default=30)
    parser.add_argument("--fine-epochs", type=int, default=4)
    parser.add_argument("--image", type=Path)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run):
        parser.error("Emri --run lejon vetem shkronja, numra, _ dhe -.")
    if args.batch_size < 1 or args.head_epochs < 1 or args.fine_epochs < 0:
        parser.error("Parametra te pavlefshem per trajnim.")
    project = args.project_root.resolve()
    output = project / "backend" / "artifacts" / args.run
    if args.command == "prepare":
        prepare(project, output)
    elif args.command == "train":
        train(project, output, args.batch_size, args.head_epochs, args.fine_epochs)
    elif args.command == "evaluate":
        evaluate(project, output, args.batch_size)
    else:
        if not args.image or not args.image.is_file():
            parser.error("Per predict nevojitet --image me nje foto ekzistuese.")
        predict(output, args.image)


if __name__ == "__main__":
    main()
