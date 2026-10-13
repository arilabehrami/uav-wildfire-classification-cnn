"""Vlereson dy modelet kundrejt etiketave te kontrollit te verber njerezor.

Etiketat e flakes per trajnim dhe per test_final u nxorren nga i njejti rregull
ngjyre. Ky skript i mat modelet kundrejt nje gjykatesi te pavarur: njeriut qe
etiketoi 200 pamje te rastesishme pa pare as rregullin as modelin.

Raporton:
  1. pajtimin e rregullit me njeriun (a eshte i besueshem rregulli?)
  2. pajtimin e etiketes origjinale te dataset-it me njeriun
  3. AUC i te dy modeleve mbi etiketat njerezore, me interval besimi 95%
     per diferencen (bootstrap i ciftuar)
  4. score-in mesatar sipas grupit: flake / skene zjarri pa flake / pa zjarr

Ekzekuto nga rrenja e projektit, pasi etiketat te jene ruajtur si
<grupi>/etiketat_njerezore.csv:
    python backend/training/evaluate_human_labels.py
    python backend/training/evaluate_human_labels.py --batch human_check_2_2026-10-13

Krahasohen te gjitha modelet e listuara ne MODELS qe ekzistojne ne disk.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
AUDIT = PROJECT / "backend/artifacts/flame_label_audit"
DEFAULT_BATCH = "human_check_2026-10-13"
MODELS = {
    "tymi (final_fire_v1)": PROJECT / "backend/artifacts/final_fire_v1",
    "flaka v1 (fire_flame_v6)": PROJECT / "backend/artifacts/fire_flame_v6",
    "flaka v2 (fire_flame_v7)": PROJECT / "backend/artifacts/fire_flame_v7",
}
LABELS = ("Fire", "No_Fire", "Uncertain")
SEED = 20261013


def read_csv(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_items(check: Path, labels_path: Path):
    key = {r["id"]: r for r in read_csv(check / "key.csv")}
    human = {}
    for r in read_csv(labels_path):
        if r["id"] not in key:
            raise SystemExit(f"ID e panjohur ne {labels_path.name}: {r['id']}")
        if r["label"] not in LABELS:
            raise SystemExit(f"Etikete e pavlefshme per {r['id']}: {r['label']!r}")
        human[r["id"]] = r["label"]
    missing = sorted(set(key) - set(human))
    if missing:
        print(f"KUJDES: {len(missing)} pamje pa etikete perjashtohen ({', '.join(missing[:5])}...)")
    return [dict(key[i], human=human[i]) for i in key if i in human]


def agreement(items, field):
    """Pajtimi mes njeriut dhe nje burimi tjeter, vetem ku te dy jane te sigurt."""
    table = Counter((it["human"], it[field]) for it in items)
    sure = [it for it in items if it["human"] != "Uncertain" and it[field] in ("Fire", "No_Fire")]
    agree = sum(1 for it in sure if it["human"] == it[field])
    return {
        "tabela_njeriu_kundrejt_burimit": {f"{h} | {o}": n for (h, o), n in sorted(table.items())},
        "te_sigurta": len(sure),
        "pajtohen": agree,
        "pajtimi": agree / len(sure) if sure else None,
    }


def auc(y, s):
    """AUC si probabiliteti qe nje flake te marre score me te larte se nje jo-flake."""
    pos, neg = s[y == 1], s[y == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float((diff > 0).mean() + 0.5 * (diff == 0).mean())


def bootstrap_diff(y, old, new, rounds=2000):
    """Diferenca AUC(new) - AUC(old) mbi te njejtat pamje te rizgjedhura."""
    rng = np.random.default_rng(SEED)
    out = []
    for _ in range(rounds):
        k = rng.integers(0, len(y), len(y))
        if y[k].min() == y[k].max():
            continue
        out.append(auc(y[k], new[k]) - auc(y[k], old[k]))
    out = np.array(out)
    return {"mesatarja": float(out.mean()),
            "ci95": [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))],
            "perseritje": int(len(out))}


def at_threshold(y, s, t):
    p = s >= t
    return {"recall_flake": float(p[y == 1].mean()), "specificity": float((~p[y == 0]).mean())}


def group(it):
    # Grupet ndahen vetem sipas njeriut. Versioni i pare i perzinte pamjet
    # 'Uncertain' te B dhe C, gje qe e bente grupin B kryesisht te paqarte.
    if it["human"] == "Fire":
        return "A. flake (sipas njeriut)"
    if it["human"] == "Uncertain":
        return "D. e paqarte (sipas njeriut)"
    if it["original_label"] == "Fire":
        return "B. skene zjarri pa flake te dukshme"
    return "C. pa zjarr"


def score(run_dir: Path, paths, batch=32):
    import tensorflow as tf  # vetem ketu: pjesa tjeter e skriptit punon pa TF

    contract = json.loads((run_dir / "inference.json").read_text(encoding="utf-8"))
    model = tf.keras.models.load_model(run_dir / contract["model"], compile=False)
    h, w = model.input_shape[1:3]
    out = np.empty(len(paths), dtype=np.float64)
    for i in range(0, len(paths), batch):
        imgs = []
        for p in paths[i:i + batch]:
            raw = tf.io.read_file(str(PROJECT / "dataset" / p))
            img = tf.io.decode_image(raw, channels=3, expand_animations=False)
            imgs.append(tf.image.resize(tf.cast(img, tf.float32), (h, w)) / 255.0)
        out[i:i + len(imgs)] = model(tf.stack(imgs), training=False).numpy().ravel()
    return out, contract.get("gates")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batch", default=DEFAULT_BATCH, help="dosja e grupit nen flame_label_audit/")
    ap.add_argument("--labels", type=Path, default=None)
    args = ap.parse_args()
    check = AUDIT / args.batch
    labels = args.labels or check / "etiketat_njerezore.csv"
    if not labels.is_file():
        raise SystemExit(
            f"Nuk u gjet {labels}.\nShkarko CSV-ne nga etiketo.html dhe vendose atje "
            f"me emrin etiketat_njerezore.csv (zakonisht shkon te dosja Downloads).")

    models = {name: run for name, run in MODELS.items() if (run / "model.keras").is_file()}
    print(f"grupi: {args.batch}   modelet: {', '.join(models)}")

    items = load_items(check, labels)
    print(f"pamje te etiketuara: {len(items)}  {dict(Counter(it['human'] for it in items))}")

    rule = agreement(items, "rule_label")
    orig = agreement(items, "original_label")
    print(f"\n1. Rregulli v1 i ngjyres kundrejt njeriut: pajtim {rule['pajtimi']:.1%} "
          f"({rule['pajtohen']}/{rule['te_sigurta']}, aty ku te dy jane te sigurt)")
    print(f"2. Etiketa origjinale kundrejt njeriut: pajtim {orig['pajtimi']:.1%} "
          f"({orig['pajtohen']}/{orig['te_sigurta']})")

    sure_idx = [i for i, it in enumerate(items) if it["human"] != "Uncertain"]
    y = np.array([1 if items[i]["human"] == "Fire" else 0 for i in sure_idx])
    print(f"\n3. Modelet mbi {len(sure_idx)} pamje te sigurta ({int(y.sum())} me flake)")

    paths = [it["path"] for it in items]
    scores, results = {}, {}
    for name, run in models.items():
        s_all, gates = score(run, paths)
        scores[name] = s_all
        s = s_all[sure_idx]
        res = {"auc": auc(y, s), "ne_prag_0.5": at_threshold(y, s, 0.5)}
        if gates:
            zone = np.where(s >= gates["high"], "alarm", np.where(s < gates["low"], "clear", "review"))
            res["portat"] = {"low": gates["low"], "high": gates["high"]}
            res["me_portat"] = {
                "flake_te_humbura": int(((zone == "clear") & (y == 1)).sum()),
                "flake_ne_alarm": int(((zone == "alarm") & (y == 1)).sum()),
                "alarme_te_rreme": int(((zone == "alarm") & (y == 0)).sum()),
                "ne_shqyrtim": int((zone == "review").sum()),
            }
        groups = {}
        for g in sorted({group(it) for it in items}):
            v = np.array([s_all[i] for i, it in enumerate(items) if group(it) == g])
            groups[g] = {"n": int(len(v)), "score_mesatar": float(v.mean()),
                         "share_mbi_0.5": float((v >= 0.5).mean())}
        res["sipas_grupit"] = groups
        results[name] = res
        print(f"\n   {name}: AUC {res['auc']:.4f}   ne 0.5: recall {res['ne_prag_0.5']['recall_flake']:.3f}, "
              f"specificity {res['ne_prag_0.5']['specificity']:.3f}")
        if "me_portat" in res:
            m = res["me_portat"]
            print(f"     me portat {gates['low']}/{gates['high']}: humbur {m['flake_te_humbura']}, "
                  f"flake ne alarm {m['flake_ne_alarm']}, alarme te rreme {m['alarme_te_rreme']}, "
                  f"shqyrtim {m['ne_shqyrtim']}")
        for g, v in groups.items():
            print(f"     {g:38} n={v['n']:3}  score mes. {v['score_mesatar']:.3f}  >=0.5: {v['share_mbi_0.5']:.1%}")

    names = list(models)
    diffs = {}
    print("\n4. Diferencat e AUC (bootstrap i ciftuar, 95% CI):")
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            old, new = names[i], names[j]
            d = bootstrap_diff(y, scores[old][sure_idx], scores[new][sure_idx])
            lo, hi = d["ci95"]
            verdict = ("me i mire" if lo > 0 else "me i dobet" if hi < 0 else "pa diference te qendrueshme")
            diffs[f"{new} - {old}"] = {**d, "perfundimi": verdict}
            print(f"   {new} - {old}: {d['mesatarja']:+.4f}  [{lo:+.4f}, {hi:+.4f}]  -> {verdict}")

    report = {
        "grupi": args.batch,
        "metoda": ("Pamje te rastesishme nga test_final, te etiketuara nga njeriu pa pare emrin "
                   "e fajllit, rregullin apo modelin. Uncertain perjashtohen nga metrikat."),
        "etiketat_njerezore": dict(Counter(it["human"] for it in items)),
        "rregulli_v1_kundrejt_njeriut": rule,
        "origjinalja_kundrejt_njeriut": orig,
        "modelet": results,
        "diferencat_auc": diffs,
    }
    out = check / "rezultati_kontrollit_njerezor.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    with (check / "score_per_pamje.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "path", "original_label", "rule_label", "human", *names])
        for i, it in enumerate(items):
            w.writerow([it["id"], it["path"], it["original_label"], it["rule_label"], it["human"],
                        *[f"{scores[n][i]:.6f}" for n in names]])
    print(f"\nu shkrua: {out}")


if __name__ == "__main__":
    main()
