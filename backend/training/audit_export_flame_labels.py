"""Prepare a review of an app export without modifying training data or models.

Only the four original images inspected visually on 2026-10-20 have confirmed
labels here. Other exported images require individual review. Export scores
and filename labels are not ground truth. Training membership is diagnostic,
not evidence of an independent evaluation.
"""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
CONFIRMED = {
    f"Test/Fire/resized_test_fire_frame{n}.jpg": "Fire"
    for n in (89, 94, 101, 102)
}


def read(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write(path, rows, fields):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("export", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    train = {r["path"]: r["label"] for r in read(
        PROJECT / "dataset/splits/train_flame_reviewed_v1.csv")}
    valid = {r["path"]: r["label"] for r in read(
        PROJECT / "dataset/splits/validation_flame_reviewed_v1.csv")}
    rows = read(args.export)
    review = []
    for row in rows:
        matches = [p for group in ("Fire", "No_Fire")
                   if (p := PROJECT / "dataset/Test" / group / row["fajlli"]).is_file()]
        path = matches[0].relative_to(PROJECT / "dataset").as_posix() if len(matches) == 1 else ""
        review.append(dict(
            filename=row["fajlli"], path=path, model=row["modeli"],
            score=row["rezultati"], zone=row["zona"],
            train_label=train.get(path, ""), validation_label=valid.get(path, ""),
            reviewed_label=CONFIRMED.get(path, ""),
            review_note="Visible flame confirmed by visual inspection of original image"
            if path in CONFIRMED else "Pending visual review; do not infer label from filename",
        ))
    args.output.mkdir(parents=True, exist_ok=True)
    write(args.output / "export_review.csv", review, list(review[0]) if review else ["filename"])
    corrections = [dict(path=p, previous_train_label=train.get(p, "excluded"),
                        reviewed_label=label,
                        action="relabel" if p in train else "review inclusion before retraining")
                   for p, label in CONFIRMED.items()]
    write(args.output / "confirmed_corrections.csv", corrections, list(corrections[0]))
    report = dict(
        export=str(args.export.resolve()), records=len(rows),
        zones=dict(Counter(r["zona"] for r in rows)),
        models=dict(Counter(r["modeli"] for r in rows)),
        exported_records_in_training=sum(r["path"] in train for r in review),
        exported_records_in_validation=sum(r["path"] in valid for r in review),
        unresolved_paths=sum(not r["path"] for r in review),
        confirmed_corrections=corrections,
        next_steps=["Review the remaining images of the affected scene individually",
                    "Check related frames for conflicting labels and split leakage",
                    "Create versioned corrected splits before training a new model",
                    "Evaluate using reviewed, separate recordings; these examples are development diagnostics"],
        training_data_modified=False, model_modified=False,
    )
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
