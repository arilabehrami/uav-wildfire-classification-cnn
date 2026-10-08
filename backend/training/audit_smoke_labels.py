"""Build a visual label review without changing splits or reading test_final."""
import csv
import argparse
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
SPLITS = ("test_adapt_val", "train_with_adapt")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy', choices=['smoke', 'flame'], default='smoke')
    args = parser.parse_args()
    splits = SPLITS if args.policy == 'smoke' else ('train_with_adapt', 'validation_temporal', 'validation_extra_v3', 'test_adapt_val')
    output = PROJECT / "backend/artifacts" / ("label_audit" if args.policy == 'smoke' else 'flame_label_audit')
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for split in splits:
        with (PROJECT / "dataset/splits" / f"{split}.csv").open(
            encoding="utf-8-sig", newline=""
        ) as source:
            for row in csv.DictReader(source):
                if args.policy == 'smoke' and row["label"] != "No_Fire":
                    continue
                image = PROJECT / "dataset" / row["path"]
                if not image.is_file():
                    raise FileNotFoundError(image)
                rows.append(dict(split=split, path=row["path"], original_label=row["label"],
                                 reviewed_label="", reason="", url=image.as_uri()))
    with (output / "review_manifest.csv").open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    html = r'''<!doctype html><meta charset="utf-8"><title>Auditimi i etiketave</title>
<style>body{font:16px system-ui;margin:24px;background:#f4f5f7}header{position:sticky;top:0;background:white;padding:16px;z-index:1}button,select,input{padding:8px;margin:4px}main{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}article{background:white;padding:12px}img{width:100%;height:240px;object-fit:contain}small{overflow-wrap:anywhere}textarea{width:95%}</style>
<header><h2>Fire = flakë ose tym i zjarrit</h2><p>Shqyrto çdo foto veçmas. Re, mjegull dhe pluhur = No_Fire. Kur burimi i tymit nuk dihet, zgjidh Uncertain. Etiketat origjinale nuk ndryshohen. Kliko foton për madhësinë e plotë.</p>
<select id="split"><option>test_adapt_val</option><option>train_with_adapt</option></select><input id="search" placeholder="Filtro emrin / skenën"><button id="prev">Mbrapa</button><button id="next">Para</button><button id="export">Eksporto vendimet CSV</button><span id="status"></span><p>Vendimet ruhen në këtë shfletues; eksporto CSV para se ta mbyllësh.</p></header><main id="cards"></main>
<script>const rows=DATA;let page=0;const key='wildfire-smoke-audit-v1';let decisions={};try{decisions=JSON.parse(localStorage.getItem(key)||'{}')}catch(e){}
const split=document.querySelector('#split'),search=document.querySelector('#search'),cards=document.querySelector('#cards');
function save(){try{localStorage.setItem(key,JSON.stringify(decisions))}catch(e){document.querySelector('#status').textContent='Ruajtja lokale dështoi; eksporto CSV.'}}
function render(){const filtered=rows.filter(r=>r.split===split.value&&r.path.toLowerCase().includes(search.value.toLowerCase()));page=Math.max(0,Math.min(page,Math.ceil(filtered.length/24)-1));cards.replaceChildren();document.querySelector('#status').textContent=`Faqja ${page+1}/${Math.max(1,Math.ceil(filtered.length/24))}; ${filtered.length} foto; ${Object.keys(decisions).length} vendime`;
for(const r of filtered.slice(page*24,(page+1)*24)){const id=r.split+'|'+r.path,a=document.createElement('article'),link=document.createElement('a'),img=document.createElement('img');link.href=r.url;link.target='_blank';img.src=r.url;img.loading='lazy';link.append(img);a.append(link);const caption=document.createElement('small');caption.textContent=r.path;a.append(caption);const select=document.createElement('select');for(const v of ['', 'Fire','No_Fire','Uncertain']){const o=document.createElement('option');o.value=v;o.textContent=v||'Pa shqyrtuar';select.append(o)}select.value=decisions[id]?.reviewed_label||'';a.append(select);const reason=document.createElement('textarea');reason.placeholder='Prova / arsyeja (e detyrueshme për ndryshim)';reason.value=decisions[id]?.reason||'';a.append(reason);function update(){if(select.value)decisions[id]={reviewed_label:select.value,reason:reason.value};else delete decisions[id];save()}select.onchange=update;reason.oninput=update;cards.append(a)}}
split.onchange=search.oninput=()=>{page=0;render()};document.querySelector('#prev').onclick=()=>{page--;render()};document.querySelector('#next').onclick=()=>{page++;render()};
document.querySelector('#export').onclick=()=>{const quote=v=>'"'+String(v).replaceAll('"','""')+'"';const data=[['split','path','original_label','reviewed_label','reason'],...rows.filter(r=>decisions[r.split+'|'+r.path]).map(r=>{const d=decisions[r.split+'|'+r.path];return[r.split,r.path,r.original_label,d.reviewed_label,d.reason]})];const url=URL.createObjectURL(new Blob(['\ufeff'+data.map(r=>r.map(quote).join(',')).join('\r\n')],{type:'text/csv;charset=utf-8'}));const a=document.createElement('a');a.href=url;a.download='reviewed_smoke_labels.csv';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)};render();</script>'''
    if args.policy == 'flame':
        html = html.replace('Fire = flakë ose tym i zjarrit', 'Fire = vetëm flakë të dukshme')
        html = html.replace('Re, mjegull dhe pluhur = No_Fire. Kur burimi i tymit nuk dihet, zgjidh Uncertain.',
                            'Tym pa flakë, re, mjegull dhe pluhur = No_Fire. Shqyrto foton me madhësi të plotë; nëse flakët janë të paqarta, zgjidh Uncertain.')
        html = html.replace('wildfire-smoke-audit-v1', 'wildfire-flame-audit-v1')
        html = html.replace('reviewed_smoke_labels.csv', 'reviewed_flame_labels.csv')
    html = html.replace('<option>test_adapt_val</option><option>train_with_adapt</option>',
                        ''.join(f'<option>{s}</option>' for s in splits))
    (output / "review.html").write_text(html.replace("DATA", json.dumps(rows).replace("</", "<\\/")), encoding="utf-8")
    print(f"Review: {output / 'review.html'}; {len(rows)} images; no labels changed.")


if __name__ == "__main__":
    main()
