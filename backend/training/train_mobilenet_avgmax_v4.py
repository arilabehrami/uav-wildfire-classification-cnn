"""Vendose ne backend/training/, prane dataset_loader.py dhe audit_validation_v3.py.

Nis nga v3; ngrin MobileNetV2 dhe trajnon vetem klasifikuesin avg+max.
Perzgjedhja perdor validimin me prag 0.5. Kontrolli me 202 foto behet ne fund.
"""

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from audit_validation_v3 import check_disjoint, metrics, read_rows, source_group

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"
SOURCE = ARTIFACTS / "models/fire_mobilenet_mixed_v3_finetuned.keras"
CANDIDATE = ARTIFACTS / "models/fire_mobilenet_avgmax_v4.keras"
REPORT = ARTIFACTS / "metrics/mobilenet_avgmax_v4_results.json"
HISTORY = ARTIFACTS / "metrics/mobilenet_avgmax_v4_history.csv"
VALIDATION_SCORES = ARTIFACTS / "metrics/mobilenet_avgmax_v4_validation.csv"
VALIDATION_SPLITS = ("validation_temporal", "validation_extra_v3")
EPOCHS = 20
HEAD_BATCH_SIZE = 128


def assess(rows, scores):
    scores = np.asarray(scores).reshape(-1)
    if len(rows) != len(scores) or not np.all(np.isfinite(scores)):
        raise ValueError("Numri ose vlerat e rezultateve jane te pavlefshme.")
    if np.any(scores < 0) or np.any(scores > 1):
        raise ValueError("Fire score duhet te jete ne [0,1].")
    by_split, by_source = defaultdict(list), defaultdict(list)
    for row, score in zip(rows, scores, strict=True):
        record = {"label": row["label"], "fire_score": float(score)}
        by_split[row["split"]].append(record)
        key = f"{row['split']} / {source_group(row['path'])}"
        by_source[key].append(record)
    groups = {name: metrics(records) for name, records in by_source.items()}
    fire_rates = [r["Fire"]["correct"] / r["Fire"]["total"] for r in groups.values() if r["Fire"]["total"]]
    no_fire_rates = [r["No_Fire"]["correct"] / r["No_Fire"]["total"] for r in groups.values() if r["No_Fire"]["total"]]
    if not fire_rates or not no_fire_rates:
        raise ValueError("Mungon njera klase ne validim.")
    return {
        "group_score": float((np.mean(fire_rates) + np.mean(no_fire_rates)) / 2),
        "by_split": {name: metrics(records) for name, records in by_split.items()},
        "by_source": groups,
    }


def no_regression(result, baseline):
    if result["by_source"].keys() != baseline["by_source"].keys():
        raise ValueError("Grupet e validimit nuk perputhen.")
    for name, reference in baseline["by_source"].items():
        for label in ("Fire", "No_Fire"):
            current = result["by_source"][name][label]
            if current["total"] != reference[label]["total"]:
                raise ValueError("Numrat e fotove te validimit ndryshuan.")
            if current["correct"] < reference[label]["correct"]:
                return False
    return True


def print_assessment(title, result):
    print(f"\n{title} | rezultati sipas grupeve: {result['group_score']:.6f}")
    for name, item in result["by_split"].items():
        fire, no_fire = item["Fire"], item["No_Fire"]
        print(f"  {name}: Fire {fire['correct']}/{fire['total']}; No_Fire {no_fire['correct']}/{no_fire['total']}")
    for name, item in result["by_source"].items():
        if item["No_Fire"]["total"]:
            print(f"  Alarme te rreme [{name}]: {item['No_Fire']['false_alarms']}")


def training_weights(rows):
    # I njejti peshim si ne v3, per te provuar ndryshimin e veçorive.
    counts = Counter(row["label"] for row in rows)
    weights = []
    for row in rows:
        value = len(rows) / (2 * counts[row["label"]])
        if row["label"] == "No_Fire" and not Path(row["path"]).name.startswith("lake_"):
            value *= 2
        weights.append(value)
    return np.asarray(weights, dtype=np.float32)


def build_models(tf, source):
    backbones = [layer for layer in source.layers if layer.name.startswith("mobilenetv2")]
    if len(backbones) != 1 or tuple(source.input_shape[1:]) != (254, 254, 3):
        raise ValueError("Struktura e v3 nuk perputhet me modelin e pritur.")
    base = backbones[0]
    if not isinstance(base.layers[-1], tf.keras.layers.GlobalAveragePooling2D):
        raise ValueError("Backbone-i i v3 duhet te perfundoje me average pooling.")
    spatial_output = base.layers[-2].output
    if len(spatial_output.shape) != 4:
        raise ValueError("Nuk u gjet harta hapësinore e veçorive.")

    def walk(layer):
        yield layer
        for child in getattr(layer, "layers", []):
            yield from walk(child)

    dense_layers = [layer for layer in walk(source) if isinstance(layer, tf.keras.layers.Dense)]
    if len(dense_layers) != 1 or dense_layers[0].get_config()["activation"] != "sigmoid":
        raise ValueError("Pritet nje klasifikues Dense me sigmoid ne v3.")
    kernel, bias = dense_layers[0].get_weights()
    channels = int(spatial_output.shape[-1])
    if kernel.shape != (channels, 1) or bias.shape != (1,):
        raise ValueError("Dimensionet e klasifikuesit nuk perputhen.")

    base.trainable = False
    spatial = tf.keras.Model(base.input, spatial_output, name="mobilenetv2_spatial_v4")
    spatial.trainable = False
    inputs = tf.keras.Input(shape=(254, 254, 3), name="image")
    x = inputs
    preprocessing = []
    for layer in source.layers:
        if layer is base:
            break
        if isinstance(layer, tf.keras.layers.InputLayer):
            continue
        if not isinstance(layer, (tf.keras.layers.Resizing, tf.keras.layers.Rescaling)):
            raise ValueError(f"Paraperpunim i papritur ne v3: {layer.name}")
        preprocessing.append(layer.get_config())
        x = layer.__class__.from_config(layer.get_config())(x)
    if len(preprocessing) != 2:
        raise ValueError("Priten shtresat Resizing dhe Rescaling te modelit v3.")

    maps = spatial(x, training=False)
    average = tf.keras.layers.GlobalAveragePooling2D(name="avg_features")(maps)
    maximum = tf.keras.layers.GlobalMaxPooling2D(name="max_features")(maps)
    features = tf.keras.layers.Concatenate(name="avg_and_max")([average, maximum])
    extractor = tf.keras.Model(inputs, features, name="extractor_avgmax_v4")
    extractor.trainable = False
    head = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(2 * channels,)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ], name="head_avgmax_v4")
    # Dega avg kopjon v3; dega max nis me pesha zero.
    head.layers[-1].set_weights([np.concatenate([kernel, np.zeros_like(kernel)], axis=0), bias.copy()])
    full_model = tf.keras.Model(extractor.input, head(extractor.output), name="fire_mobilenet_avgmax_v4")
    return extractor, head, full_model, channels


def main():
    import tensorflow as tf
    from dataset_loader import DATA_DIR, LABELS, BATCH_SIZE, load_image

    tf.keras.utils.set_random_seed(42)
    if LABELS != {"No_Fire": 0, "Fire": 1}:
        raise ValueError(f"Kontrollo LABELS: {LABELS}")
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)
    for path in (CANDIDATE, REPORT, HISTORY, VALIDATION_SCORES):
        if path.exists():
            raise FileExistsError(f"Rezultati ekziston: {path}")

    splits = {
        name: [dict(row, split=name) for row in read_rows(DATA_DIR, name)]
        for name in ("train_mixed_v3", *VALIDATION_SPLITS, "holdout_extra_v2")
    }
    check_disjoint(splits)
    train_rows = splits["train_mixed_v3"]
    val_rows = [row for name in VALIDATION_SPLITS for row in splits[name]]

    def dataset(rows):
        paths = [str(DATA_DIR / row["path"]) for row in rows]
        labels = np.asarray([LABELS[row["label"]] for row in rows], dtype=np.float32)
        data = tf.data.Dataset.from_tensor_slices((paths, labels))
        data = data.map(load_image, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
        return labels, data.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    source_model = tf.keras.models.load_model(SOURCE, compile=False)
    extractor, head, model, channels = build_models(tf, source_model)
    probe_rows = [row for name in VALIDATION_SPLITS for label in LABELS
                  for row in [r for r in splits[name] if r["label"] == label][:4]]
    _, probe_data = dataset(probe_rows)
    probe_images, _ = next(iter(probe_data))
    tf.debugging.assert_all_finite(probe_images, "Imazhe te pavlefshme.")
    if tuple(probe_images.shape[1:]) != (254, 254, 3):
        raise ValueError("load_image duhet te jape imazhe 254x254 RGB.")
    if float(tf.reduce_min(probe_images)) < -1e-6 or float(tf.reduce_max(probe_images)) > 1.000001:
        raise ValueError("load_image duhet te jape vlera ne [0,1].")
    np.testing.assert_allclose(
        model(probe_images, training=False).numpy(),
        source_model(probe_images, training=False).numpy(),
        rtol=1e-5, atol=1e-6,
        err_msg="Modeli i ri nuk riprodhon v3 para trajnimit.",
    )
    print("Kontrolli fillestar kaloi: avg+max riprodhon v3 para trajnimit.")

    def extract(rows, title):
        labels, data = dataset(rows)
        print(f"\nNxjerrja e veçorive: {title} ({len(rows)} foto)")
        features = np.asarray(extractor.predict(data, verbose=1), dtype=np.float32)
        if features.shape != (len(rows), 2 * channels) or not np.all(np.isfinite(features)):
            raise ValueError("Veçori te pavlefshme.")
        return labels, features

    val_labels, val_features = extract(val_rows, "validim")
    baseline_scores = head.predict(val_features, batch_size=256, verbose=0).reshape(-1)
    baseline = assess(val_rows, baseline_scores)
    print_assessment("V3 para trajnimit", baseline)
    train_labels, train_features = extract(train_rows, "trajnim")
    initial_head_weights = [weight.copy() for weight in head.get_weights()]

    class ValidationTracker(tf.keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.records = []
            self.best_any = None
            self.best_acceptable = None

        def on_epoch_end(self, epoch, logs=None):
            if logs is None or not np.isfinite(logs.get("val_loss", np.nan)):
                raise FloatingPointError("Mungon nje validation loss i fundem.")
            scores = self.model.predict(val_features, batch_size=256, verbose=0).reshape(-1)
            result = assess(val_rows, scores)
            acceptable = no_regression(result, baseline)
            improved = acceptable and result["group_score"] > baseline["group_score"] + 1e-12
            record = {
                "epoch": epoch + 1, "val_loss": float(logs["val_loss"]),
                "no_group_regression": acceptable, "accepted_improvement": improved,
                "validation": result,
            }
            self.records.append(record)
            rank = (result["group_score"], -record["val_loss"])
            snapshot = (rank, record, [weight.copy() for weight in self.model.get_weights()])
            if self.best_any is None or rank > self.best_any[0]:
                self.best_any = snapshot
            if improved and (self.best_acceptable is None or rank > self.best_acceptable[0]):
                self.best_acceptable = snapshot
            logs["val_group_score"] = result["group_score"]
            print_assessment(f"Epoka {epoch + 1}", result)
            print(f"  Pa ulje ne grupet e validimit: {acceptable}; permiresim i pranueshem: {improved}")

    for path in (CANDIDATE, REPORT, HISTORY, VALIDATION_SCORES):
        path.parent.mkdir(parents=True, exist_ok=True)
    tracker = ValidationTracker()
    head.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss="binary_crossentropy", metrics=["accuracy"])
    head.fit(
        train_features, train_labels, sample_weight=training_weights(train_rows),
        validation_data=(val_features, val_labels), batch_size=HEAD_BATCH_SIZE,
        epochs=EPOCHS, shuffle=True,
        callbacks=[
            tracker,
            tf.keras.callbacks.EarlyStopping(monitor="val_group_score", mode="max", patience=5),
            tf.keras.callbacks.CSVLogger(str(HISTORY)),
        ],
    )
    if tracker.best_any is None:
        raise RuntimeError("Trajnimi nuk perfundoi asnje epoke.")
    accepted = tracker.best_acceptable is not None
    chosen = tracker.best_acceptable if accepted else tracker.best_any
    _, best_record, weights = chosen
    head.set_weights(weights)
    head_changed = any(not np.array_equal(a, b) for a, b in zip(initial_head_weights, head.get_weights(), strict=True))
    model.save(CANDIDATE)
    reloaded = tf.keras.models.load_model(CANDIDATE, compile=False)
    np.testing.assert_allclose(model(probe_images, training=False).numpy(), reloaded(probe_images, training=False).numpy(), rtol=1e-5, atol=1e-6)

    selected_path = CANDIDATE if accepted else SOURCE
    selected_model = reloaded if accepted else source_model
    print(f"\nKandidati i trajnuar: {CANDIDATE}")
    print(f"Peshat e klasifikuesit ndryshuan: {head_changed}")
    print(f"Modeli i zgjedhur sipas validimit: {selected_path.name}")
    if not accepted:
        print("Kandidati nuk plotesoi kushtin e permiresimit pa ulje ne grupet e validimit. Mbetet v3.")

    # Vendimi eshte marre. Keto rezultate nuk perdoren per perzgjedhje.
    holdout_rows = splits["holdout_extra_v2"]
    _, holdout_data = dataset(holdout_rows)
    print(f"\nKontroll diagnostik ne {len(holdout_rows)} fotot e njohura:")
    holdout_scores = selected_model.predict(holdout_data, verbose=1).reshape(-1)
    holdout_result = assess(holdout_rows, holdout_scores)
    print_assessment("Kontrolli diagnostik", holdout_result)

    candidate_scores = head.predict(val_features, batch_size=256, verbose=0).reshape(-1)
    with VALIDATION_SCORES.open("x", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["split", "source", "path", "label", "fire_score_v3", "fire_score_candidate"])
        for row, old, new in zip(val_rows, baseline_scores, candidate_scores, strict=True):
            writer.writerow([row["split"], source_group(row["path"]), row["path"], row["label"], float(old), float(new)])
    with SOURCE.open("rb") as file:
        source_hash = hashlib.file_digest(file, "sha256").hexdigest()
    report = {
        "source": str(SOURCE), "source_sha256": source_hash,
        "trained_candidate": str(CANDIDATE), "selected_model": str(selected_path),
        "candidate_selected": accepted, "head_weights_changed": head_changed,
        "tensorflow_version": tf.__version__,
        "architecture": "Frozen v3 MobileNetV2 spatial maps -> concat(global average, global max) -> Dropout(0.3) -> Dense sigmoid. Average weights initialized from v3; max weights initialized to zero.",
        "training": {"optimizer": "Adam", "learning_rate": 1e-4, "max_epochs": EPOCHS, "head_batch_size": HEAD_BATCH_SIZE, "sample_weight": "Same as v3: inverse class frequency, doubled for non-lake No_Fire."},
        "selection_rule": "Fixed threshold 0.5. Mean Fire recall across source groups and mean No_Fire specificity across source groups contribute equally. Require a strictly higher score and no decrease in correct count for either class in any source group relative to v3. Otherwise retain v3.",
        "validation_baseline": baseline, "best_candidate_epoch": best_record,
        "history": tracker.records,
        "split_counts": {name: len(rows) for name, rows in splits.items()},
        "diagnostic_holdout": holdout_result,
        "limitation": "The 202 control images were inspected in previous experiments. They are a diagnostic comparison, not an independent final test. Validation acceptance cannot guarantee no regressions on these or future images. Global max pooling is a hypothesis for improving local-cue sensitivity, not proof of spatial localization or correct labels.",
    }
    with REPORT.open("x", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False, allow_nan=False)
    print(f"Raporti: {REPORT}")
    print(f"Rezultatet e validimit: {VALIDATION_SCORES}")


if __name__ == "__main__":
    main()
