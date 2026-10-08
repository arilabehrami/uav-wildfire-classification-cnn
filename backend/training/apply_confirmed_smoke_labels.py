"""Create a versioned validation split for the user-confirmed smoke segment.

Only Test/No_Fire frames 1592-1913 are confirmed. Training and test_final
are deliberately outside this correction. Original CSVs remain intact.
"""
import csv
import json
import re
from collections import Counter
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]


def main():
    splits = PROJECT / "dataset/splits"
    source = splits / "test_adapt_val.csv"
    target = splits / "test_adapt_val_smoke_v1.csv"
    report_dir = PROJECT / "backend/artifacts/label_audit"
    report_dir.mkdir(parents=True, exist_ok=True)
    with source.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        fields = reader.fieldnames
        rows = list(reader)
    before = dict(Counter(row["label"] for row in rows))
    changes = []
    for row in rows:
        match = re.fullmatch(r"Test/No_Fire/resized_test_nofire_frame(\d+)\.jpg", row["path"])
        if match and 1592 <= int(match.group(1)) <= 1913 and row["label"] == "No_Fire":
            changes.append(dict(path=row["path"], original_label="No_Fire", reviewed_label="Fire",
                                reason="User confirmed fire smoke in frames 1592-1913 on 2026-10-08."))
            row["label"] = "Fire"
    with target.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with (report_dir / "confirmed_smoke_changes.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["path", "original_label", "reviewed_label", "reason"])
        writer.writeheader()
        writer.writerows(changes)
    after = dict(Counter(row["label"] for row in rows))
    report = dict(source=source.name, corrected_split=target.name, before=before, after=after,
                  changed=len(changes), confirmation_scope="Test/No_Fire frames 1592-1913 only",
                  training_labels_changed=0, test_final_accessed=False,
                  limitation="Corrected validation has only Fire. ROC AUC and No_Fire specificity are undefined. Obtain independently confirmed No_Fire validation data before model selection.")
    (report_dir / "confirmed_smoke_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
