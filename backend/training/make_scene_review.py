"""Render numbered contact sheets for manual review; never assign labels."""
import argparse
import json
import re
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=200)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    import tensorflow as tf
    root = Path(__file__).resolve().parents[2]
    files = sorted((p for p in (root / "dataset/Test/Fire").glob("*.jpg")
                    if (m := re.search(r"frame(\d+)\.jpg$", p.name))
                    and args.start <= int(m[1]) <= args.end),
                   key=lambda p: int(re.search(r"frame(\d+)", p.name)[1]))
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = []
    for offset in range(0, len(files), 24):
        group = files[offset:offset + 24]
        cells = [tf.image.resize(tf.io.decode_jpeg(tf.io.read_file(str(p)), channels=3),
                                 (254, 254)) for p in group]
        cells += [tf.zeros((254, 254, 3))] * (24 - len(cells))
        sheet = tf.concat([tf.concat(cells[i:i+6], axis=1) for i in range(0,24,6)], axis=0)
        name = f"sheet_{offset // 24 + 1:02d}.png"
        tf.io.write_file(str(args.output / name), tf.io.encode_png(tf.cast(sheet, tf.uint8)))
        manifest.append(dict(sheet=name, cells=[dict(cell=i+1, filename=p.name)
                                               for i,p in enumerate(group)]))
    (args.output / "sheets.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
