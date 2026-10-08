import json
from pathlib import Path

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"
METRICS_DIR = ARTIFACTS / "metrics"

V2_REPORT = METRICS_DIR / "mobilenet_mixed_v2_results.json"
V3_REPORT = METRICS_DIR / "mobilenet_mixed_v3_results.json"


def load_report(path):
    if not path.exists():
        raise FileNotFoundError(f"Nuk u gjet raporti: {path}")

    with path.open(
        encoding="utf-8"
    ) as file:
        return json.load(file)


def calculate_metrics(report):
    per_class = report["holdout"]["per_class"]

    no_fire_correct = per_class["No_Fire"]["correct"]
    no_fire_total = per_class["No_Fire"]["total"]

    fire_correct = per_class["Fire"]["correct"]
    fire_total = per_class["Fire"]["total"]

    false_positives = no_fire_total - no_fire_correct
    false_negatives = fire_total - fire_correct

    true_negatives = no_fire_correct
    true_positives = fire_correct

    accuracy = (
        true_positives + true_negatives
    ) / (
        fire_total + no_fire_total
    )

    precision = (
        true_positives
        / (true_positives + false_positives)
        if (true_positives + false_positives) > 0
        else 0.0
    )

    recall = (
        true_positives
        / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0
        else 0.0
    )

    specificity = (
        true_negatives
        / (true_negatives + false_positives)
        if (true_negatives + false_positives) > 0
        else 0.0
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    false_positive_rate = (
        false_positives
        / (false_positives + true_negatives)
        if (false_positives + true_negatives) > 0
        else 0.0
    )

    false_negative_rate = (
        false_negatives
        / (false_negatives + true_positives)
        if (false_negatives + true_positives) > 0
        else 0.0
    )

    return {
        "TP": true_positives,
        "TN": true_negatives,
        "FP": false_positives,
        "FN": false_negatives,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "false_positive_rate": false_positive_rate,
        "false_negative_rate": false_negative_rate,
    }


def print_model(name, metrics):
    print(f"\n================ {name} ================")

    print("\nConfusion Matrix:")
    print()
    print("                 Pred No_Fire   Pred Fire")
    print(
        f"True No_Fire       {metrics['TN']:>3}"
        f"           {metrics['FP']:>3}"
    )
    print(
        f"True Fire          {metrics['FN']:>3}"
        f"           {metrics['TP']:>3}"
    )

    print("\nMetrics:")

    print(
        f"Accuracy:            "
        f"{metrics['accuracy']:.2%}"
    )

    print(
        f"Precision (Fire):    "
        f"{metrics['precision']:.2%}"
    )

    print(
        f"Recall (Fire):       "
        f"{metrics['recall']:.2%}"
    )

    print(
        f"Specificity:         "
        f"{metrics['specificity']:.2%}"
    )

    print(
        f"F1-score:            "
        f"{metrics['f1']:.2%}"
    )

    print(
        f"False Positive Rate: "
        f"{metrics['false_positive_rate']:.2%}"
    )

    print(
        f"False Negative Rate: "
        f"{metrics['false_negative_rate']:.2%}"
    )


def main():
    v2_report = load_report(V2_REPORT)
    v3_report = load_report(V3_REPORT)

    v2 = calculate_metrics(v2_report)
    v3 = calculate_metrics(v3_report)

    print_model("V2", v2)
    print_model("V3", v3)

    print("\n================ KRAHASIMI ================")

    print(
        f"{'Metric':<24}"
        f"{'V2':>12}"
        f"{'V3':>12}"
    )

    print("-" * 48)

    comparisons = [
        ("Accuracy", "accuracy"),
        ("Precision Fire", "precision"),
        ("Recall Fire", "recall"),
        ("Specificity", "specificity"),
        ("F1-score", "f1"),
        ("False Positive Rate", "false_positive_rate"),
        ("False Negative Rate", "false_negative_rate"),
    ]

    for label, key in comparisons:
        print(
            f"{label:<24}"
            f"{v2[key]:>11.2%}"
            f"{v3[key]:>11.2%}"
        )

    print("-" * 48)

    print(
        f"{'False Positives':<24}"
        f"{v2['FP']:>12}"
        f"{v3['FP']:>12}"
    )

    print(
        f"{'False Negatives':<24}"
        f"{v2['FN']:>12}"
        f"{v3['FN']:>12}"
    )

    print("\nNdryshimi V2 -> V3:")

    print(
        f"Accuracy: "
        f"{(v3['accuracy'] - v2['accuracy']) * 100:+.2f} "
        "percentage points"
    )

    print(
        f"False Positives: "
        f"{v2['FP']} -> {v3['FP']}"
    )

    print(
        f"False Negatives: "
        f"{v2['FN']} -> {v3['FN']}"
    )


if __name__ == "__main__":
    main()