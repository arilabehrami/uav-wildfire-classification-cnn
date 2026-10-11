"""Mat sa luhatet performanca nëpër segmente kohore të të njëjtit regjistrim.

Një numër i vetëm mbi një ndarje të vetme nuk e mat modelin — mat ndarjen.
Ky skript e ndan dosjen Test në segmente të njëpasnjëshme, me hendek mes tyre,
dhe mat të njëjtin model me të njëjtat porta në secilin. Dalja është një
shpërndarje: median dhe gamë, jo një pikë.

Portat nuk ri-nxirren për segment. Ato vijnë nga inference.json dhe mbeten
fikse — pyetja është sa ndryshon sjellja e një politike të vetme sipas kushteve
të pamjes, jo sa mirë mund të përshtatet një politikë për çdo segment.

    # nga rrënja e projektit
    python backend/training/segment_stability.py
    python backend/training/segment_stability.py --segments 8 --gap 200
    python backend/training/segment_stability.py --from-csv \
        backend/artifacts/final_fire_v1/known_test_predictions.csv

Me --from-csv nuk kërkohet TensorFlow: rezultatet lexohen nga një CSV
parashikimesh i prodhuar më parë për të njëjtin model.
"""

import argparse
import csv
import json
import re
import statistics
from pathlib import Path

FRAME = re.compile(r"frame(\d+)", re.IGNORECASE)
LABELS = ("Fire", "No_Fire")


# ----------------------------------------------------------------- metrika

def roc_auc(pairs):
    """pairs: [(score, label01)]. Mesatarizon rangjet per vlera te barabarta."""
    ordered = sorted(pairs)
    pos = sum(label for _, label in ordered)
    neg = len(ordered) - pos
    if pos == 0 or neg == 0:
        return None

    rank_sum = 0.0
    index = 0
    while index < len(ordered):
        stop = index
        while stop + 1 < len(ordered) and ordered[stop + 1][0] == ordered[index][0]:
            stop += 1
        average_rank = (index + 1 + stop + 1) / 2
        for position in range(index, stop + 1):
            if ordered[position][1] == 1:
                rank_sum += average_rank
        index = stop + 1

    return (rank_sum - pos * (pos + 1) / 2) / (pos * neg)


def zone_metrics(pairs, low, high):
    alarm = alarm_true = clear = clear_fire = review = 0
    pos = sum(label for _, label in pairs)

    for score, label in pairs:
        if score >= high:
            alarm += 1
            alarm_true += label
        elif score < low:
            clear += 1
            clear_fire += label
        else:
            review += 1

    total = len(pairs)
    return {
        "alarm_share": alarm / total,
        "alarm_precision": (alarm_true / alarm) if alarm else None,
        "clear_share": clear / total,
        "missed_fire_rate": (clear_fire / pos) if pos else None,
        "review_share": review / total,
        "auto_decision_share": (alarm + clear) / total,
    }


# --------------------------------------------------------------- segmentet

def split_with_gap(frames, segments, gap):
    """Ndan nje liste te renditur ne blloqe te njepasnjeshme, duke hequr
    `gap` kuadro ne secilin kufi te brendshem (gjysma nga secila ane)."""
    size = len(frames)
    edges = [round(index * size / segments) for index in range(segments + 1)]
    half = gap // 2
    blocks = []

    for index in range(segments):
        start, stop = edges[index], edges[index + 1]
        if index > 0:
            start += half
        if index < segments - 1:
            stop -= half
        blocks.append(frames[start:stop] if start < stop else [])

    return blocks


def build_segments(rows, segments, gap):
    by_label = {}
    for label in LABELS:
        picked = [row for row in rows if row["label"] == label]
        picked.sort(key=lambda row: row["frame"])
        by_label[label] = split_with_gap(picked, segments, gap)

    built = []
    for index in range(segments):
        fire = by_label["Fire"][index]
        no_fire = by_label["No_Fire"][index]
        if not fire or not no_fire:
            continue

        pairs = [(row["score"], 1) for row in fire] + [
            (row["score"], 0) for row in no_fire
        ]
        built.append({
            "segment": index + 1,
            "n": len(pairs),
            "fire": len(fire),
            "no_fire": len(no_fire),
            "fire_frames": [fire[0]["frame"], fire[-1]["frame"]],
            "no_fire_frames": [no_fire[0]["frame"], no_fire[-1]["frame"]],
            "pairs": pairs,
        })

    return built


# ------------------------------------------------------------------ hyrja

def read_predictions(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    out = []
    for row in rows:
        match = FRAME.search(row["path"])
        if not match or row["label"] not in LABELS:
            continue
        out.append({
            "path": row["path"],
            "label": row["label"],
            "frame": int(match.group(1)),
            "score": float(row["fire_score"]),
        })
    return out


def score_with_model(project, run, manifest):
    """Skanon vete dosjen Test me modelin e inference.json (kerkon TensorFlow)."""
    import sys

    sys.path.insert(0, str(project / "backend"))
    from app.ml.classifier import FireClassifier

    classifier = FireClassifier(run=run)
    print(f"Modeli: {classifier.model_version}  (sha256 i verifikuar)")

    data_root = project / "dataset"
    rows = []
    with manifest.open(encoding="utf-8-sig", newline="") as handle:
        entries = list(csv.DictReader(handle))

    for position, entry in enumerate(entries, 1):
        relative = entry["path"].replace("\\", "/")
        match = FRAME.search(relative)
        if not match or entry["label"] not in LABELS:
            continue
        image = (data_root / relative).read_bytes()
        rows.append({
            "path": relative,
            "label": entry["label"],
            "frame": int(match.group(1)),
            "score": classifier.classify(image)["fire_score"],
        })
        if position % 500 == 0:
            print(f"  {position}/{len(entries)}")

    return rows, classifier


# ------------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--run", default="final_fire_v1")
    parser.add_argument("--segments", type=int, default=6)
    parser.add_argument("--gap", type=int, default=200)
    parser.add_argument("--from-csv", type=Path, default=None,
                        help="CSV parashikimesh i gatshem; nuk kerkon TensorFlow.")
    parser.add_argument("--manifest", type=Path, default=None,
                        help="CSV me path,label. Parazgjedhje: known_test.csv i run-it.")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    if args.segments < 2:
        raise SystemExit("Duhen te pakten 2 segmente.")

    project = args.project_root.resolve()
    run_dir = project / "backend" / "artifacts" / args.run
    contract = json.loads((run_dir / "inference.json").read_text(encoding="utf-8"))
    low = float(contract["gates"]["low"])
    high = float(contract["gates"]["high"])

    if args.from_csv:
        source = args.from_csv if args.from_csv.is_absolute() else project / args.from_csv
        rows = read_predictions(source)
        model_version = f"{args.run} (nga {source.name})"
        verified = False
    else:
        manifest = args.manifest or (run_dir / "known_test.csv")
        manifest = manifest if manifest.is_absolute() else project / manifest
        rows, classifier = score_with_model(project, args.run, manifest)
        model_version = classifier.model_version
        verified = True
        source = manifest

    if not rows:
        raise SystemExit("Asnje rresht i perdorshem.")

    print(f"\nModeli:    {model_version}")
    print(f"Burimi:    {source}")
    print(f"Portat:    low={low:.4f}  high={high:.4f}  (fikse per te gjitha segmentet)")
    print(f"Ndarja:    {args.segments} segmente, hendek {args.gap} kuadro\n")

    segments = build_segments(rows, args.segments, args.gap)
    if len(segments) < 2:
        raise SystemExit("Shume pak segmente me te dyja klasat.")

    def cell(value, width=9):
        return "—".rjust(width) if value is None else f"{value:>{width}.1%}"

    header = (f"{'Seg':>4}{'kuadro':>8}{'Fire':>7}{'AUC':>9}{'alarm':>9}"
              f"{'saktesi':>9}{'pastruar':>9}{'humbur':>9}{'auto':>9}")
    print(header)
    print("-" * len(header))

    results = []
    for block in segments:
        auc = roc_auc(block["pairs"])
        zones = zone_metrics(block["pairs"], low, high)
        record = {k: v for k, v in block.items() if k != "pairs"}
        record["roc_auc"] = auc
        record.update(zones)
        results.append(record)

        print(f"{block['segment']:>4}{block['n']:>8}{block['fire']:>7}"
              f"{auc:>9.4f}"
              + cell(zones["alarm_share"])
              + cell(zones["alarm_precision"])
              + cell(zones["clear_share"])
              + cell(zones["missed_fire_rate"])
              + cell(zones["auto_decision_share"]))

    def spread(key):
        values = [r[key] for r in results if r[key] is not None]
        if not values:
            return None
        return {
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
            "spread": max(values) - min(values),
            "n_segments": len(values),
        }

    summary = {key: spread(key) for key in
               ("roc_auc", "alarm_precision", "missed_fire_rate",
                "auto_decision_share")}

    print("\nShperndarja nepermjet segmenteve")
    print(f"{'':<22}{'median':>9}{'min':>9}{'max':>9}{'gama':>9}")
    for key, title in (("roc_auc", "ROC AUC"),
                       ("alarm_precision", "Saktesia e alarmit"),
                       ("missed_fire_rate", "Zjarre te humbur"),
                       ("auto_decision_share", "Vendime automatike")):
        stat = summary[key]
        if not stat:
            continue
        form = "{:>9.4f}" if key == "roc_auc" else "{:>9.1%}"
        print(f"{title:<22}" + "".join(
            form.format(stat[field]) for field in ("median", "min", "max", "spread")))

    report = {
        "model": model_version,
        "model_sha256": contract.get("model_sha256"),
        "sha256_verified_at_runtime": verified,
        "source": str(source),
        "gates": {"low": low, "high": high, "refit_per_segment": False},
        "segments": args.segments,
        "gap_frames": args.gap,
        "per_segment": results,
        "summary": summary,
        "limitation": (
            "Segmentet jane blloqe te njepasnjeshme kuadrosh te te njejtit "
            "regjistrim, te ndara vetem ne kohe. Segmenti k i Fire dhe segmenti "
            "k i No_Fire nuk pretendohen te njekohshem: jane bllok i k-te i "
            "secilit sekuence. Luhatja e matur ketu eshte luhatje brenda nje "
            "regjistrimi te vetem dhe nuk e mat pergjithesimin ne regjistrime "
            "te tjera, i cili mbetet i pamatur."
        ),
    }

    out = args.out or (run_dir / f"segment_stability_{args.segments}.json")
    out = out if out.is_absolute() else project / out
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nRaporti: {out}")


if __name__ == "__main__":
    main()
