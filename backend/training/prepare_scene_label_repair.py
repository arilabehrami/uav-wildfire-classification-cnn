"""Versioned repair based on visual inspection of sheets 01-09, 2026-10-20.

All original Test/Fire frames 0..200 were inspected at 254x254 in contact
sheets and show visible flames. No color heuristic assigns these labels.
This is a scoped development repair, not a full dataset audit. These frames
are training diagnostics and must not be reported as independent test data.
"""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "backend/artifacts/flame_label_audit/review_2026-10-20"
SPLITS = ROOT / "dataset/splits"


def read(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write(path, rows, fields):
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main():
    train = read(SPLITS / "train_flame_reviewed_v1.csv")
    valid = read(SPLITS / "validation_flame_reviewed_v1.csv")
    old = {r["path"]: r["label"] for r in train}
    sheets = json.loads((OUT / "sheets/sheets.json").read_text())
    reviewed = ["Test/Fire/" + c["filename"] for s in sheets for c in s["cells"]]
    expected = {f"Test/Fire/resized_test_fire_frame{n}.jpg" for n in range(201)}
    if set(reviewed) != expected or len(reviewed) != 201:
        raise SystemExit("Contact sheet inventory differs from visually reviewed frames 0..200")
    features = {r["path"]: int(r["dhash"]) for r in read(
        OUT.parent / "review_v2_2026-10-13/features.csv")}
    if any(p not in features or not (ROOT / "dataset" / p).is_file() for p in reviewed):
        raise SystemExit("Missing reviewed original or similarity hash")
    changes = [dict(path=p, previous_train_label=old.get(p, "excluded"),
                    reviewed_label="Fire", sha256=hashlib.sha256(
                        (ROOT / "dataset" / p).read_bytes()).hexdigest(),
                    review_method="Visual inspection of original-resolution contact sheet cell")
               for p in reviewed]
    repaired = {**old, **{p: "Fire" for p in reviewed}}
    new_train = [dict(path=p, label=label) for p, label in repaired.items()]
    # Screen validation against corrected scene; existing v1 filtering is preserved.
    excluded, new_valid = [], []
    for row in valid:
        if row["path"] not in features:
            raise SystemExit("Missing validation hash: " + row["path"])
        distance = min((features[row["path"]] ^ features[p]).bit_count() for p in reviewed)
        if row["path"] in repaired or distance <= 8:
            excluded.append({**row, "min_scene_dhash_distance": distance})
        else:
            new_valid.append(row)
    if set(repaired) & {r["path"] for r in new_valid}:
        raise SystemExit("Train/validation overlap")
    for name, rows in (("train_flame_scene_repair_v1", new_train),
                       ("validation_flame_scene_repair_v1", new_valid)):
        write(SPLITS / f"{name}.csv", rows, ["path", "label"])
    write(OUT / "scene_review_confirmed.csv", changes, list(changes[0]))
    write(OUT / "validation_scene_exclusions.csv", excluded,
          ["path", "label", "min_scene_dhash_distance"])
    report = dict(reviewed_frames=len(changes),
                  relabelled_no_fire=sum(r["previous_train_label"] == "No_Fire" for r in changes),
                  newly_included=sum(r["previous_train_label"] == "excluded" for r in changes),
                  already_fire=sum(r["previous_train_label"] == "Fire" for r in changes),
                  train=dict(Counter(r["label"] for r in new_train)),
                  validation=dict(Counter(r["label"] for r in new_valid)),
                  validation_near_scene_excluded=len(excluded),
                  limitations=["Other labels remain unaudited; this is a scoped experimental repair",
                               "dHash screening does not establish independence of recordings",
                               "Reviewed scene is development/training data, not an independent test"],
                  original_splits_modified=False, model_modified=False)
    (OUT / "scene_repair_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
