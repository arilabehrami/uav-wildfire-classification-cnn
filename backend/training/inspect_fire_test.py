"""Kontroll i rezultateve ekzistuese, pa trajnim ose ndryshim modeli.

Vendose prane fire_final.py, te backend/training, pastaj:
    python backend/training/inspect_fire_test.py
Krijon fire_test_review.zip me rezultate, modelin dhe shembuj imazhesh.
"""
import argparse
import csv
import json
import math
import re
import shutil
import tempfile
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import fire_final as f


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def natural_key(row):
    path = Path(row["path"])
    match = re.fullmatch(r"(.*?)(\d+)", path.stem)
    return str(path.parent), match[1] if match else path.stem, int(match[2]) if match else -1, row["path"]


def spread(rows, limit):
    """Shperndarje sipas emrit/numrit; kjo nuk krijon ID videosh apo skenash."""
    rows = sorted(rows, key=natural_key)
    if not rows:
        return []
    return [rows[i] for i in np.linspace(0, len(rows)-1, min(limit, len(rows)), dtype=int)]


def choose_examples(rows, limit):
    # Gjysma nga pozicione te shperndara, gjysma nga nivele te ndryshme score-sh.
    by_score = sorted(rows, key=lambda r: (float(r["fire_score"]), r["path"]))
    if not by_score:
        return []
    limit = min(limit, len(by_score))
    candidates = [by_score[0], by_score[-1]] + spread(rows, max(0, limit//2 - 1))
    candidates += [by_score[i] for i in np.linspace(0, len(by_score)-1, min(2*limit, len(by_score)), dtype=int)]
    chosen = {}
    for row in candidates:
        chosen.setdefault(row["path"], row)
        if len(chosen) == limit:
            break
    return list(chosen.values())


def error_runs(rows):
    groups = defaultdict(list)
    for row in rows:
        path = Path(row["path"])
        match = re.fullmatch(r"(.*?)(\d+)", path.stem)
        if match:
            groups[(str(path.parent), match[1])].append(int(match[2]))
    runs = []
    for (parent, prefix), numbers in groups.items():
        numbers = sorted(set(numbers))
        first = last = numbers[0]
        for number in numbers[1:] + [None]:
            if number is not None and number == last + 1:
                last = number
                continue
            runs.append({"folder": parent, "prefix": prefix, "start": first, "end": last, "count": last-first+1})
            first = last = number
    return sorted(runs, key=lambda r: (-r["count"], r["folder"], r["start"]))[:30]


def model_config_check(model_path):
    with zipfile.ZipFile(model_path) as archive:
        config = json.loads(archive.read("config.json"))
    nodes = []
    def visit(value):
        if isinstance(value, dict):
            nodes.append(value)
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
    visit(config)
    rescale = [n["config"] for n in nodes if n.get("class_name") == "Rescaling"]
    resize = [n["config"] for n in nodes if n.get("class_name") == "Resizing"]
    sigmoid = [n["config"] for n in nodes if n.get("class_name") == "Dense" and n.get("config", {}).get("activation") == "sigmoid"]
    inputs = [n["config"].get("batch_shape", n["config"].get("batch_input_shape")) for n in nodes if n.get("class_name") == "InputLayer"]
    expected = (any(x == [None, 254, 254, 3] for x in inputs)
                and any(x.get("height") == 224 and x.get("width") == 224 for x in resize)
                and any(x.get("scale") == 2.0 and x.get("offset") == -1.0 for x in rescale)
                and len(sigmoid) == 1 and sigmoid[0].get("units") == 1)
    return config, {"matches_expected_layer_configs": expected, "input_shapes": inputs,
                    "rescaling": rescale, "resizing": resize, "sigmoid_heads": sigmoid,
                    "note": "Configuration inspection only; full graph and prediction checks require loading the model."}


def scores_check(run, role, expected, problems):
    predictions = f.read_csv(run / f"{role}_predictions.csv")
    manifest_name = "validation.csv" if role == "validation" else f"{role}.csv"
    manifest = f.read_csv(run / manifest_name)
    expected_rows = [(r["path"], r["label"]) for r in manifest]
    predicted_rows = [(r["path"], r["label"]) for r in predictions]
    if expected_rows != predicted_rows:
        problems.append(f"{role}: renditja/etiketat e CSV-se se parashikimeve nuk perputhen me listen.")
    for row in predictions:
        row["path"] = f.normalized_path(row["path"])
        score = float(row["fire_score"])
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError(f"Score i pavlefshem: {row['path']}")
        label = "Fire" if score >= f.THRESHOLD else "No_Fire"
        if row.get("predicted_label") != label:
            problems.append(f"{role}: etikete e parashikuar e papajtueshme me score per {row['path']}")
        row["fire_score"] = score
        row["predicted_label"] = label
    p = np.asarray([r["fire_score"] for r in predictions])
    result = f.metrics(predictions, p)
    if result["confusion_matrix"] != expected.get("confusion_matrix"):
        problems.append(f"{role}: confusion matrix nuk perputhet me raportin e ruajtur.")
    result["score_quantiles"] = {}
    for label in f.LABELS:
        values = np.asarray([r["fire_score"] for r in predictions if r["label"] == label])
        if len(values):
            result["score_quantiles"][label] = dict(zip(["min", "p10", "median", "p90", "max"],
                                                        map(float, np.quantile(values, [0, .1, .5, .9, 1]))))
    errors = [r for r in predictions if r["label"] != r["predicted_label"]]
    result["error_filename_runs"] = error_runs(errors)
    result["filename_runs_note"] = "Adjacent filename numbers only. Not verified video chronology or independent scenes."
    return predictions, result


def font(size):
    for path in ["C:/Windows/Fonts/arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default()


def make_sheet(name, rows, data_root, work, selected, warnings):
    columns, tile_w, tile_h = 4, 286, 320
    height = 60 + tile_h * max(1, math.ceil(len(rows) / columns))
    sheet = Image.new("RGB", (columns * tile_w, height), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((12, 12), name + f" | {len(rows)} examples", fill="black", font=font(20))
    if not rows:
        draw.text((12, 75), "No cases in this category.", fill="black", font=font(17))
    for i, row in enumerate(rows):
        relative = f.normalized_path(row["path"])
        path = data_root / relative
        with Image.open(path) as original:
            dimensions, mode = original.size, original.mode
            image = original.convert("RGB")
        archive_path = "samples/" + relative
        destination = work / archive_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        is_original = path.stat().st_size <= 256 * 1024
        if is_original:
            shutil.copyfile(path, destination)
        else:
            archive_path += ".preview.jpg"
            destination = work / archive_path
            preview = image.copy()
            preview.thumbnail((768, 768))
            preview.save(destination, quality=92)
            warnings.append(f"Preview, not original bytes: {relative}")
        selected.append({"category": name, "path": relative, "label": row["label"],
            "fire_score": row.get("fire_score", ""), "predicted_label": row.get("predicted_label", ""),
            "source_split": row.get("source_split", ""), "archive_path": archive_path,
            "original_bytes": is_original, "width": dimensions[0], "height": dimensions[1], "mode": mode})
        x, y = (i % columns) * tile_w, 60 + (i // columns) * tile_h
        # Ne flete ruhet raporti i pamjes. Modeli perdor resize 254 -> 224.
        thumbnail = image.copy()
        thumbnail.thumbnail((270, 250))
        sheet.paste(thumbnail, (x + 8, y + 4))
        draw.text((x+8, y+256), Path(relative).name[:36], fill="black", font=font(13))
        score = row.get("fire_score")
        text = f"Label: {row['label']}" + (f" | Fire={float(score):.4f}" if score is not None else " | TRAIN")
        draw.text((x+8, y+277), text, fill="black", font=font(13))
        draw.text((x+8, y+298), f"Original: {dimensions[0]}x{dimensions[1]} {mode}", fill="black", font=font(12))
    sheet.save(work / f"{name}.jpg", quality=91)


def run_review(project, run_name, destination):
    run = project / "backend" / "artifacts" / run_name
    if destination.exists():
        raise FileExistsError(f"Ekziston {destination}. Per nje kopje tjeter perdor --output fire_test_review_2.zip")
    selection = read_json(run / "selection.json")
    evaluation = read_json(run / "evaluation.json")
    inference = read_json(run / "inference.json")
    problems, warnings, roles, report = [], [], {}, {}
    model_path = run / "model.keras"
    model_hash = f.sha256(model_path)
    for label, metadata in [("selection", selection), ("evaluation", evaluation), ("inference", inference)]:
        if metadata.get("model_sha256") != model_hash:
            problems.append(f"{label}: SHA256 nuk perputhet me model.keras aktual.")
        if metadata.get("threshold") != 0.5:
            problems.append(f"{label}: pragu nuk eshte 0.5.")
    if inference.get("labels") != {"No_Fire": 0, "Fire": 1}:
        problems.append("Mapping i papritur i etiketave ne inference.json.")
    config, config_report = model_config_check(model_path)
    if not config_report["matches_expected_layer_configs"]:
        problems.append("Konfigurimi i modelit kerkon kontroll: nuk perputhet me shtresat e pritura.")
    roles["validation"], report["validation"] = scores_check(run, "validation", selection["validation"], problems)
    for role in ("diagnostic_202", "known_test"):
        roles[role], report[role] = scores_check(run, role, evaluation["evaluations"][role], problems)
    train = f.read_csv(run / "train.csv")
    for row in train:
        row["path"] = f.normalized_path(row["path"])
    test = roles["known_test"]
    val = roles["validation"]
    categories = {
        "01_train_forest_no_fire": spread([r for r in train if r["label"] == "No_Fire" and r["stratum"] == "base_forest"], 12),
        "02_train_forest_fire": spread([r for r in train if r["label"] == "Fire" and r["stratum"] == "base_forest"], 8),
        "03_test_false_alarms": choose_examples([r for r in test if r["label"] == "No_Fire" and r["predicted_label"] == "Fire"], 16),
        "04_test_correct_no_fire": choose_examples([r for r in test if r["label"] == "No_Fire" and r["predicted_label"] == "No_Fire"], 8),
        "05_test_missed_fire": choose_examples([r for r in test if r["label"] == "Fire" and r["predicted_label"] == "No_Fire"], 12),
        "06_test_correct_fire": choose_examples([r for r in test if r["label"] == "Fire" and r["predicted_label"] == "Fire"], 8),
        "07_validation_false_alarms": choose_examples([r for r in val if r["label"] == "No_Fire" and r["predicted_label"] == "Fire" and r["stratum"] == "base_forest"], 12),
        "08_validation_missed_fire": choose_examples([r for r in val if r["label"] == "Fire" and r["predicted_label"] == "No_Fire"], 12),
    }
    with tempfile.TemporaryDirectory(prefix="fire_test_review_") as temporary:
        work = Path(temporary) / "review"
        work.mkdir()
        selected = []
        for name, rows in categories.items():
            make_sheet(name, rows, project / "dataset", work, selected, warnings)
        fields = ["category", "path", "label", "fire_score", "predicted_label", "source_split",
                  "archive_path", "original_bytes", "width", "height", "mode"]
        f.write_csv(work / "selected_images.csv", selected, fields)
        files = ["selection.json", "evaluation.json", "inference.json", "plan.json", "training_started.json",
                 "best_candidate.json", "history_head.json", "history_finetune.json", "train.csv",
                 "validation_predictions.csv", "diagnostic_202_predictions.csv", "known_test_predictions.csv"]
        source_hashes = {}
        for name in files:
            if (run / name).is_file():
                shutil.copyfile(run / name, work / name)
                source_hashes[name] = f.sha256(run / name)
        shutil.copyfile(Path(f.__file__), work / "fire_final_used_for_review.py")
        f.write_json(work / "model_config.json", config)
        included_model = model_path.stat().st_size <= 18 * 1024 * 1024
        if included_model:
            shutil.copyfile(model_path, work / "model.keras")
        else:
            warnings.append("Model exceeds 18 MiB; only configuration and hash included to keep the review small.")
        f.write_json(work / "review_report.json", {
            "model_sha256": model_hash, "model_included": included_model,
            "consistency_problems": problems, "warnings": warnings,
            "model_configuration": config_report, "recomputed_metrics": report,
            "source_file_hashes": source_hashes,
            "purpose": "Diagnose pipeline consistency, labels and train/test visual differences. No retraining or threshold tuning.",
            "limits": "A small image sample cannot verify all labels or prove causality. Similar filename numbers are not verified video IDs."})
        (work / "READ_ME.txt").write_text(
            "Kontroll diagnostik: parashikimet jane nga run ekzistues.\n"
            "01/02 = shembuj trajnimi; 03-06 = raste Test; 07/08 = gabime validimi.\n"
            "Etiketa ne pamje eshte etiketa e dataset-it, jo nje etikete e re e verifikuar vizualisht.\n"
            "selected_images.csv lidh secilen foto me rrugen, score dhe kopjen ne samples/.\n"
            "original_bytes=False shenon nje preview te zvogeluar; per te tjerat ruhen byte-t origjinale.\n"
            "Asnje model, etikete, split ose prag nuk u ndryshua.\n", encoding="utf-8")
        archive_path = Path(temporary) / "fire_test_review.zip"
        with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(work.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(work).as_posix())
        if archive_path.stat().st_size > 30 * 1024 * 1024:
            raise ValueError("Paketa doli mbi 30 MiB; nevojitet nje eksport me me pak foto.")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(archive_path, destination)
    for role, result in report.items():
        f.display(role, result)
    print(f"\nProbleme konsistence: {len(problems)}")
    for problem in problems:
        print("  " + problem)
    print(f"Modeli perfshihet ne pakete: {included_model}")
    print(f"Paketa: {destination}\nMadhesia: {destination.stat().st_size / 1024**2:.1f} MiB")
    print("Bashkengjit kete ZIP. U perdoren score-t ekzistuese; nuk u nis trajnim.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--run", default="final_fire_v1")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run):
        parser.error("Emri i run-it nuk eshte i vlefshem.")
    project = args.project_root.resolve()
    output = args.output.resolve() if args.output else project / "fire_test_review.zip"
    run_review(project, args.run, output)


if __name__ == "__main__":
    main()
