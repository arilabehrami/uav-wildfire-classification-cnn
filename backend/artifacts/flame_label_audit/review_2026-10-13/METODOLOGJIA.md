# Rishikimi i etiketave sipas politikës së flakës, 2026-10-13

## Politika

Fire do të thotë **flakë e dukshme**. Tym pa flakë është No_Fire. Rastet ku
flaka nuk dallohet dot nga dheu i zbuluar janë Uncertain dhe përjashtohen.

Kjo zëvendëson politikën e mëparshme (Fire = flakë **ose** tym), nën të cilën
u trajnua `final_fire_v1`.

## Pse u desh

Modeli në prodhim i rendit kuadrot kështu, mbi `known_test_predictions.csv`:

| grupi | n | score mesatar |
| --- | --- | --- |
| flakë me tym | 500 | 0.9883 |
| tym zjarri pa flakë | 322 | 0.8026 |
| mjegull/re, pa zjarr | 1392 | 0.5751 |
| flakë pa tym | 500 | 0.2212 |

Flaka e vërtetë pa tym renditet nën mjegullën pa zjarr. Modeli ka mësuar
tymin, jo flakën, sepse nën politikën e vjetër tymi ishte sinjali që e jepte
përgjigjen e saktë.

## Si u nxorën etiketat

Rishikimi i plotë kërkon vendim për 42,698 foto. Ato janë kuadro videoje, pra
u ndoq kjo rrugë:

1. **Statistikë ngjyre për çdo foto.** Numërohen pikselat me `r >= g >= b` dhe
   `r - b > 40`, dhe veçmas ata prej tyre me `r > 150` (fusha `bright`).
2. **Kalibrim mbi raste të konfirmuara vizualisht.** Mbi nëntë foto me flakë të
   verifikuar me sy, numri i pikselave portokalli ishte 43–212; mbi pesë foto
   pa flakë ishte 0.
3. **Pragu u nxor nga negativat e njohura, jo nga pozitivat.** Mes 3,869
   kuadrove të etiketuara No_Fire në regjistrimet me borë, **asnjë** nuk ka më
   shumë se 15 piksela të ndritshëm portokalli. Prandaj `bright >= 16` → Fire.
4. **Verifikim vizual i pragut.** 16 foto me `bright >= 16` u shikuan në
   montazh: të gjitha kishin flakë të dukshme, përfshirë atë me 17 piksela.
   16 foto me `bright == 0` u shikuan: asnjëra nuk kishte flakë.
5. **Banda 1–15 u shpall Uncertain**, sepse në të njëjtin diapazon bien edhe
   kuadro të njohura No_Fire.

### Trajtime të veçanta

- **Fluturimi mbi liqen** (`lake_resized_lake_frame*`): No_Fire për tërë
  regjistrimin. Kuadrot 7908–8336 janë perëndim dielli dhe japin 300–1361
  piksela portokalli — glow horizonti, jo flakë. Mbahen si negativa të vështira:
  i mësojnë modelit se portokallia në vetvete nuk është zjarr.
- **`extra_raw/`** (987 foto): Uncertain, të përjashtuara. Janë zjarr i
  kontrolluar me dru mbi bar të thatë, me balancë të bardhë rozë — domen krejt
  tjetër nga zjarri i pyllit në borë, dhe statistika e ngjyrës ndizet nga toka.

## Rezultati

| nga → në | numri |
| --- | --- |
| Fire → Fire | 15,967 |
| Fire → No_Fire | 4,627 |
| Fire → Uncertain | 6,993 |
| No_Fire → No_Fire | 14,621 |
| No_Fire → Uncertain | 490 |

Split-et: `train_flame_reviewed_v1` 26,856 foto (10,955 Fire / 15,901 No_Fire),
`validation_flame_reviewed_v1` 7,998 foto pas filtrimit (4,985 Fire / 3,013
No_Fire).

## Filtrimi i near-duplicate

dHash 64-bit, distancë Hamming, prag 8. Asnjë duplikat i saktë (d=0); çifti më
i afërt ishte d=5. U hoqën 361 foto nga validation (4.3%). Çiftet më të afërta
ishin kuadro liqeni, ku uji i hapur duket njësoj kudo. Raporti i plotë:
`leakage_report.json`.

Split-et ishin tashmë të ndara në kohë: asnjë numër kuadri nuk mbivendoset mes
train e validation, dhe hendeqet janë 200–870 kuadro.

## Kufizimet

- Etiketat nuk janë nxjerrë nga shqyrtim foto-për-foto. Janë nxjerrë nga një
  rregull ngjyre i kalibruar dhe i verifikuar mbi mostra; pragu është zgjedhur
  që të mos ndizet mbi asnjë negativ të njohur, pra gabimi i pritshëm është në
  drejtimin e humbjes së flakëve të vogla, jo të alarmeve të rreme.
- Në 254×254, një flakë e largët nuk dallohet me siguri nga dheu i zbuluar.
  Kjo është arsyeja e bandës Uncertain prej 7,483 fotosh, dhe është kufi i
  rezolucionit, jo i metodës.
- dHash është ekran përafrues ngjashmërie, jo provë e pavarësisë së skenës.
- `test_final` nuk u lexua dhe nuk u ndryshua.
- Asnjë peshë modeli nuk u ritrajnua në këtë hap.

## Rezultati i trajnimit dhe krahasimi

`fire_flame_v6` u trajnua mbi `train_flame_reviewed_v1` (26,856 foto).
Arkitektura: MobileNetV2 ImageNet i ngrirë → `concat(GlobalAvgPool,
GlobalMaxPool)` → Dropout(0.35) → Dense(1, sigmoid). Epoka më e mirë: 21.

Mbi validation: saktësi e balancuar e grupeve 0.8596 (flakë 79.98%, jo-flakë
91.93%). Para trajnimit, koka e patrajnuar mbi veçoritë e ngrira jep 0.4976 dhe
i quan Fire 99.4% të fotove — kjo është baza zero.

### Krahasimi mbi test_final

`test_final` (3,463 foto) nuk u përdor në trajnimin e asnjërit model dhe nuk
mbivendoset me asnjë split të ri. U rietiketua me të njëjtin rregull: 560 Fire,
1,807 No_Fire, 1,096 Uncertain të përjashtuara.

| | tymi (final_fire_v1) | flaka (fire_flame_v6) |
| --- | --- | --- |
| ROC AUC | 0.6957 | 0.8553 |
| saktësi e balancuar (pragu më i mirë) | 0.6732 | 0.8038 |
| A. flakë e dukshme (560) — score mes. | 0.9051 | 0.8896 |
| A. — % >= 0.5 | 99.8% | 92.0% |
| B. tym pa flakë (520) — score mes. | 0.9177 | 0.4986 |
| B. — % >= 0.5 | 99.8% | 50.0% |
| C. pa zjarr fare (1287) — score mes. | 0.5921 | 0.2592 |
| C. — % >= 0.5 | 59.4% | 23.7% |

Gjetja qendrore: modeli i tymit jep score mesatar **më të lartë** për tymin pa
flakë (0.9177) se për flakën e vërtetë (0.9051). Pra nuk është thjesht i
pasaktë — ai po zgjidh një detyrë tjetër. Modeli i flakës e mban flakën, e ndal
tymin dhe gati e përgjysmon alarmin e rremë mbi skena pa zjarr.

### Kërcënimet ndaj validitetit të këtij krahasimi

- Etiketat e testit u nxorën nga i njëjti rregull ngjyre si ato të trajnimit.
  Numri 0.8553 është prandaj optimist: një model që do ta kishte mësuar
  rregullin fjalë për fjalë do të dilte afër 1.0. Krahasimi mbetet i drejtë
  sepse të dy modelet u matën kundrejt të njëjtave etiketa, dhe modeli i vjetër
  nuk ka asnjë lidhje me rregullin.
- `test_final` vjen nga të njëjtat regjistrime si trajnimi, me kuadro të tjera.
  Përgjithësimi në një fluturim të ri mbetet i paprovuar.
- Grupi B qëndron në 50%: modeli i ri është vërtet i pasigurt për tymin pa
  flakë. Kjo mund të jetë sjellje e arsyeshme — tymi shpesh shoqëron një flakë
  jashtë kuadrit — por nën politikën e rreptë numërohet si gabim.

## Përbërja e negativave — pse validation-i gënjen

| burimi i negativit | validation | test_final |
| --- | --- | --- |
| liqen (fluturim pa zjarr) | 2036 (67.6%) | 0 (0%) |
| pyll me mjegull/tym | 322 (10.7%) | 1287 (71.2%) |
| pyll tjetër | 655 (21.7%) | 520 (28.8%) |

Dy të tretat e negativave të validation-it janë një fluturim mbi liqen, që
dallohet trivialisht nga një pyll me borë që digjet. Negativat e test_final
janë pyll me mjegull dhe tym — detyra e vërtetë. Prandaj specificity e modelit
të flakës në pragun 0.1 është 0.90 mbi validation dhe 0.54 mbi test_final.

E njëjta gjë shpjegon hendekun e modelit të vjetër (0.95 mbi validation,
0.69 mbi test): nuk ishte leakage kuadrosh — split-et ishin të ndara në kohë —
por përbërja e negativave. Çdo metrikë e validation-it në këtë projekt para
kësaj analize është optimiste për këtë arsye.

Rregullimi: portat përshtaten vetëm mbi negativat e pyllit, liqeni raportohet
veçmas, dhe specificity raportohet gjithmonë sipas burimit të negativit.

## Hapi tjetër

Nga **rrënja e projektit**, jo nga `backend/` — skripti e pret `dataset/` nën
dosjen aktuale:

```powershell
python backend/training/train_adapted.py --dry-run
python backend/training/train_adapted.py --name fire_flame_v6
```

Pritshmëri e sinqertë: piloti i mëparshëm (40 foto trajnimi) dha model që
humbte më shumë zjarre se ai i tymit. Tani ka 26,856 foto, pra prova është e
drejtë për herë të parë. Por nëse edhe ky dështon mbi flakët e vogla, shkaku
nuk janë etiketat: koka `MobileNetV2(pooling="avg")` e mesatarizon rrjetën 7×7,
dhe një flakë që ndez një qelizë nga 49 hollohet rreth 50 herë. Në atë rast
hapi tjetër është `train_mobilenet_avgmax_v4.py` ose rezolucion më i lartë.
