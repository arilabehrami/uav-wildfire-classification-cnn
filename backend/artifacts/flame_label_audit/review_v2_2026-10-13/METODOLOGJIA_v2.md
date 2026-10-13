# Etiketat e flakës v2, 2026-10-13

Vazhdim i `review_2026-10-13/METODOLOGJIA.md`. Lexoje atë së pari.

## Pse v2

Kontrolli i verbër njerëzor (`human_check_2026-10-13`, 200 pamje, 159 të
sigurta) tregoi se rregulli v1 (`bright >= 16`) i humbi 40 nga 61 flakët që pa
njeriu. Modeli `fire_flame_v6`, i trajnuar mbi to, i kapi 0 nga 6 flakët që
rregulli i kishte quajtur No_Fire. Mbi etiketat njerëzore nuk u dallua nga
modeli i tymit (AUC 0.794 kundrejt 0.792).

## Zgjedhja e rregullit

19 rregulla kandidate u krahasuan kundrejt 159 pamjeve të sigurta të grupit të
parë: pragje mbi `bright` dhe `n`, etiketa origjinale, dhe kombinime. Më i miri
sipas saktësisë së balancuar:

    Fire  =  etiketa origjinale Fire  DHE  n >= 1

| rregulli | recall | precision | specificity | bal. acc |
| --- | --- | --- | --- | --- |
| v2: origjinale Fire DHE n >= 1 | 0.967 | 0.952 | 0.969 | 0.968 |
| etiketa origjinale | 0.984 | 0.896 | 0.929 | 0.956 |
| bright >= 1 | 0.902 | 0.902 | 0.939 | 0.920 |
| v1: bright >= 16 | 0.344 | 1.000 | 1.000 | 0.672 |

`n` = pikselat me `r >= g >= b` dhe `r - b > 40`. Interpretimi: etiketat
origjinale të FLAME janë baza e mirë; kushti `n >= 1` heq vetëm kuadrot "Fire"
pa asnjë gjurmë ngjyre flake.

**Kërcënim ndaj validitetit:** rregulli u zgjodh mbi grupin e parë, prandaj
performanca e tij atje është optimiste. Vlerësimi përfundimtar bëhet mbi grupin
e dytë (`human_check_2_2026-10-13`): 200 pamje të reja nga `test_final`, farë
20261014, pa asnjë pamje të grupit të parë.

## Rezultati

| nga → në | numri |
| --- | --- |
| Fire → Fire | 23,139 |
| Fire → No_Fire | 3,856 |
| Fire → Uncertain (extra_raw) | 592 |
| No_Fire → No_Fire | 14,716 |
| No_Fire → Uncertain (extra_raw) | 395 |

Krahasuar me v1, në trajnim hyjnë rreth 6,000 pamje më shumë, kryesisht flakët
e vogla që v1 i kishte përjashtuar si "të paqarta".

- `train_flame_reviewed_v2`: 32,946 (17,664 Fire / 15,282 No_Fire)
- `validation_flame_reviewed_v2`: 8,350 (5,394 Fire / 2,956 No_Fire), pas
  heqjes së 415 pamjeve me dHash Hamming <= 8 ndaj trajnimit

`extra_raw/` mbetet i përjashtuar si në v1. v1 nuk u ndryshua.

## Fajllat

- `features.csv` — `n`, `bright`, `dhash` për të 42,698 pamjet
  (`backend/training/flame_features.py`)
- `labels_v2.csv` — etiketa v2 me arsyen për secilën
- `split_report_v2.json`
- skripti: `backend/training/prepare_flame_splits_v2.py`

## Hapat

```powershell
python backend/training/train_adapted.py --train-split train_flame_reviewed_v2 --validation-splits validation_flame_reviewed_v2 --name fire_flame_v7
python backend/training/fit_flame_gates.py --run fire_flame_v7 --validation validation_flame_reviewed_v2 --test-labels none
python backend/training/evaluate_human_labels.py --batch human_check_2_2026-10-13
```

## Kriteri i vendimit, i fiksuar para rezultatit (2026-10-13, 21:35)

Shkruar para trajnimit të `fire_flame_v7` dhe para etiketimit të grupit të dytë,
që vlerësimi të mos përshtatet pas rezultatit:

1. Grupi `human_check_2_2026-10-13` përdoret **një herë të vetme**.
2. Metrika kryesore: AUC kundrejt etiketave njerëzore (pa "E paqartë").
   `fire_flame_v7` quhet më i mirë se `final_fire_v1` vetëm nëse intervali 95%
   i diferencës (bootstrap i çiftuar, 2000 përsëritje, farë 20261013) është
   i tëri mbi zero.
3. Metrika dytësore: flakët e njeriut që aplikacioni i pastron automatikisht
   (të humbura) me portat e secilit model.
4. Rezultati raportohet çfarëdo që të jetë. Asnjë ndryshim i mëtejshëm i
   rregullit, modelit ose portave nuk vlerësohet mbi këtë grup; një iteracion
   tjetër do të kërkonte një grup të tretë të verbër.

Modeli nuk i sheh kurrë etiketat njerëzore: grupi i parë vendosi vetëm cilin nga
19 rregullat të përdorim për etiketimin e trajnimit, dhe asnjë pamje e të dy
grupeve nuk është në trajnim (të dy vijnë nga `test_final`).

## Vendimi: v2 nuk u trajnua (2026-10-13, 21:40)

Etiketat dhe split-et v2 u përgatitën, por `fire_flame_v7` nuk u trajnua dhe
grupi `human_check_2_2026-10-13` nuk u etiketua. Kriteri i mësipërm mbetet i
paaplikuar.

Arsyeja, sipas autores: në përdorim real droni lëviz dhe mund të ulet. Një
flakë e vogël ose e largët bëhet e madhe dhe e qartë nga afër, ku modeli
`fire_flame_v6` kap rreth 95% (mbi flakët që rregulli v1 i njihte, grupi i
parë njerëzor). Zona "Për shqyrtim" mund të shërbejë si sinjal për rivëzhgim
nga lartësi më e ulët.

Kufizimet e këtij arsyetimi:
- Vlen vetëm për flakët që modeli i çon në shqyrtim ose alarm. Në grupin e
  parë, `fire_flame_v6` pastroi automatikisht 3 nga 61 flakë të njeriut.
- Rivëzhgimi nga afër është zbutje e propozuar, jo e vlerësuar në këtë punë.

Etiketat v2 dhe kriteri mbeten si dokumentim i hapit të radhës të mundshëm.
