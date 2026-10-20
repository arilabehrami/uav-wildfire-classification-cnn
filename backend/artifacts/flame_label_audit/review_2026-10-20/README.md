# Kontrolli i skenës me flakë të vogël, 20 tetor 2026

U kontrolluan vizualisht të 201 kuadrot `Test/Fire/resized_test_fire_frame0.jpg`
deri te `frame200.jpg`, në nëntë montazhe me qeliza në rezolucionin origjinal
254 × 254. Në të gjitha shihet flakë. Lidhja mes qelizës dhe emrit ruhet në
`sheets/sheets.json`; hash-i SHA256 i çdo origjinali ruhet në
`scene_review_confirmed.csv`.

Në listën e përdorur për modelin aktual, 66 prej këtyre kuadrove janë
`No_Fire`, 75 mungojnë dhe 60 janë `Fire`. Kjo është një mospërputhje konkrete
e etiketimit që mund të ndikojë në rezultat; nuk provon se shpjegon çdo gabim.

Listat e reja janë `train_flame_scene_repair_v1.csv` dhe
`validation_flame_scene_repair_v1.csv`. Listat v1 dhe modeli në shërbim nuk
ndryshohen. Nga validimi u kontrollua largësia dHash ndaj 201 kuadrove;
asnjë kuadër validimi nuk ishte në distancë <= 8 dhe nuk u hoq asnjë.
Ky kontroll nuk provon pavarësi mes regjistrimeve.

Kontrolli paraprak i trajnimit kaloi: 26,931 foto për trajnim dhe 7,998 për
validim, pa mbivendosje rrugësh. Etiketat e tjera trashëgohen nga v1 dhe nuk
janë verifikuar individualisht në këtë ndërhyrje.

Komanda për modelin eksperimental:

```powershell
.venv\Scripts\python.exe -u backend/training/train_adapted.py --train-split train_flame_scene_repair_v1 --validation-splits validation_flame_scene_repair_v1 --name fire_flame_scene_repair_v1
```

Gjendja e ekzekutimit shihet në `training.log`; përfundimi vërtetohet nga
`backend/artifacts/fire_flame_scene_repair_v1/result.json` dhe `model.keras`.
Nisja e procesit nuk është dëshmi e përfundimit ose e përmirësimit.

Para aktivizimit kërkohet krahasim me modelin aktual, përfshirë flakën e
humbur dhe alarmet e rreme, si dhe përshtatja e pragjeve mbi validim.
Kuadrot e korrigjuara përdoren si kontroll i trajnimit, jo si test i pavarur.
451 nga 500 regjistrimet e eksportit të fundit janë në listën e trajnimit v1;
ky eksport nuk duhet raportuar si vlerësim i pavarur.
