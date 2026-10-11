# Xception dhe kombinimi me MobileNet

Ekzekuto nga dosja kryesore e projektit ne PowerShell, nje komande pas tjetres.
Kerkohet run ekzistues `final_fire_v2` i MobileNet. Perdor emra te rinj kur perserit eksperimente.

```powershell
python backend/training/fire_final.py prepare --run xception_v1
python backend/training/fire_final.py train --run xception_v1 --architecture xception --head-epochs 30 --fine-epochs 4 --batch-size 8
python backend/training/compare_xception.py --mobilenet-run final_fire_v2 --xception-run xception_v1 --run ensemble_v1 --batch-size 8
python backend/training/fire_final.py evaluate --run ensemble_v1 --batch-size 8
```

Xception perdor pesha ImageNet (shkarkohen ne perdorimin e pare), hyrje RGB [0,1]
254x254 dhe resize te brendshem 299x299 me shkallezim [-1,1]. Trajnimi perdor
te njejtin sampling, peshim dhe augmentation si MobileNet. Fine-tuning nis te
`block13_sepconv1`; BatchNormalization mbetet frozen. Ne CPU kjo mund te zgjase shume.

Krahasimi kerkon lista te ngrira dhe inventar imazhesh identik ne te dy run-et.
Krahason MobileNet, Xception dhe mesataren 50/50 ne validim. Zgjedh minimumin
e group_BCE, me group_balanced_accuracy si zgjidhje barazimi, sic protokolli ekzistues.
Nuk optimizon pesha ose prag mbi testin. Modeli fitues ruhet ne
`backend/artifacts/ensemble_v1/model.keras`, edhe kur fiton nje model individual.
Ensemble ruhet gjithashtu ne `ensemble.keras`. Krahasimi nuk ndryshon modelin e aplikacionit.

`evaluate` raporton vetem modelin e zgjedhur ne diagnostic_202 dhe known_test.
Raporti perfundimtar eshte `backend/artifacts/ensemble_v1/evaluation.json`.
Metrikat shfaqen si perqindje, perfshire Fire recall dhe No_Fire specificity.
known_test eshte perdorur me pare; per konfirmim te pavarur duhen foto te reja.
Rezultati me i larte nuk garantohet.
