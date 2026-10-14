# Detektim zjarri në pamje ajrore me CNN

Sistem që klasifikon një pamje ajrore (dron) në `Fire` ose `No_Fire`, me një
backend FastAPI mbi një CNN në TensorFlow dhe një ndërfaqe React.

Sistemi **detekton** zjarr në një pamje të dhënë. Nuk parashikon zjarre të
ardhshme.

## Çka do të thotë "Fire"

**Fire = flakë e dukshme.** Tymi pa flakë është `No_Fire`.

Modeli i parë (`final_fire_v1`) u trajnua nën politikën "flakë ose tym" dhe
doli se kishte mësuar tymin, jo flakën: mbi `test_final` i jepte tymit pa flakë
score mesatar më të lartë (0.918) se flakës së vërtetë (0.905), dhe flakën pa
tym e humbiste. Etiketat u rishikuan sipas politikës së flakës dhe u trajnua
`fire_flame_v6`, që tani është modeli i parazgjedhur. Metodologjia e plotë:
`backend/artifacts/flame_label_audit/review_2026-10-13/METODOLOGJIA.md`.

## Struktura

```
backend/
  app/                 sherbimi FastAPI
    config.py          konfigurimi nga ambienti
    main.py            app factory, CORS, logging, /health
    ml/
      classifier.py    ngarkimi i modelit + verifikimi i SHA256
      preprocessing.py kontrata e hyrjes (RGB 254x254, [0,1])
      zones.py         logjika e tri zonave, pa TensorFlow
      explain.py       Grad-CAM, i verifikuar kundrejt modelit ne nisje
    routes/            /api/classify
    schemas/           modelet e pergjigjes
    services/          klasifikim + shpjegim
  artifacts/<run>/     modeli, inference.json, raportet e vleresimit
  training/            skriptet e eksperimenteve (nuk nevojiten per sherbimin)
  tests/               pytest
dataset/               fotot (jashte git-it) dhe split-et (ne git)
frontend/              React + Vite
```

## Kërkesat

- Python 3.10 ose më i ri
- Node.js 20 ose më i ri
- Modeli `backend/artifacts/fire_flame_v6/model.keras` (shih më poshtë)

## Modeli

Fajllat `*.keras` janë **të përjashtuar nga git** (`.gitignore`), prandaj një
klonim i freskët nuk e ka modelin. Mbaje një kopje rezervë jashtë kompjuterit.
Dy mundësi për ta pasur:

1. **Kopjoje** `model.keras` në `backend/artifacts/fire_flame_v6/`.
2. **Ritrajnoje** nga rrënja e projektit:
   ```powershell
   python backend/training/train_adapted.py --name fire_flame_v6
   python backend/training/fit_flame_gates.py
   ```
   Kjo kërkon `dataset/` dhe `pip install -r backend/requirements-training.txt`.
   `fit_flame_gates.py` rishkruan `inference.json` me hash-in e modelit të ri;
   numrat do të ndryshojnë pak nga ata më poshtë.

`inference.json` i ekzekutimit mban SHA256-in e modelit. Nëse fajlli nuk
përputhet, API-ja **refuzon të startojë** — me qëllim: modeli që citohet në
punim është modeli që shërben. Për zhvillim: `FIRE_VERIFY_HASH=0`.

Modeli i tymit mbetet i disponueshëm për krahasim me
`FIRE_MODEL_RUN=final_fire_v1`.

## Ngritja e backend-it

Nga **rrënja e projektit**:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r backend/requirements.txt
python -m uvicorn backend.app.main:app --reload --port 8000
```

| Rruga           | Metoda | Përshkrimi                                        |
| --------------- | ------ | ------------------------------------------------- |
| `/health`       | GET    | `{"status","model_loaded"}`                       |
| `/api/classify` | POST   | `multipart/form-data` me fushën `file`            |
| `/docs`         | GET    | dokumentacioni interaktiv i OpenAPI               |

Në nisje, logu duhet të tregojë `modeli u ngarkua: run=fire_flame_v6` dhe
`Grad-CAM aktiv dhe i verifikuar`. Nëse në vend të kësaj shfaqet
`Grad-CAM i padisponueshem`, klasifikimi punon, por pa hartë; arsyeja shkruhet
në të njëjtin rresht.

### Variablat e ambientit

| Variabla              | Parazgjedhja                                   | Përshkrimi                           |
| --------------------- | ---------------------------------------------- | ------------------------------------ |
| `FIRE_MODEL_RUN`      | `fire_flame_v6`                                | dosja nën `backend/artifacts/`       |
| `FIRE_CORS_ORIGINS`   | `http://localhost:5173,http://127.0.0.1:5173`  | origjinat e lejuara, ndarë me presje |
| `FIRE_MAX_UPLOAD_MB`  | `10`                                           | kufiri i ngarkimit                   |
| `FIRE_LOG_LEVEL`      | `INFO`                                         | niveli i logimit                     |
| `FIRE_VERIFY_HASH`    | `1`                                            | `0` e çaktivizon kontrollin e SHA256 |

## Ngritja e frontend-it

```powershell
cd frontend
npm install
copy .env.example .env
npm run dev
```

Hapet në `http://localhost:5173`. Nëse backend-i ngrihet në adresë tjetër,
caktoje `VITE_API_BASE` në `.env`, dhe shtoje origjinën e re te
`FIRE_CORS_ORIGINS`.

Ndërfaqja pranon shumë foto njëherësh (deri 50 në radhë), i klasifikon në seri,
tregon zonën e vendimit dhe hartën Grad-CAM, lejon shqyrtuesin të shënojë
verdiktin për rastet e pasigurta, dhe e ruan regjistrin në `localStorage` me
eksport në CSV ose JSON.

## Testet

```powershell
cd backend
pip install -r requirements-dev.txt
python -m pytest
```

Shumica e testeve nuk kërkojnë TensorFlow: aplikacioni ndërtohet me
`create_app(..., use_lifespan=False)` dhe shërbimi zëvendësohet me një stub.
`test_preprocessing.py` dhe `test_explain.py` kërkojnë TensorFlow; testet që
ngarkojnë modelet e ruajtura anashkalohen kur fajlli `.keras` mungon.
`test_contract.py` verifikon që vlerat rezervë në `frontend/src/config/model.js`
përputhen me `inference.json` të modelit që pretendojnë.

```powershell
cd frontend
npm run lint
npm run build
```

## Vendimi me tri zona

Dalja e modelit është një numër sigmoid. Vendimi merret me dy porta të lexuara
nga `inference.json`:

| Zona          | Kushti                      | Veprimi                     |
| ------------- | --------------------------- | --------------------------- |
| E pastër      | `score < low`               | asnjë                       |
| Për shqyrtim  | `low <= score < high`       | duhet ta shohë një njeri    |
| Alarm         | `score >= high`             | njoftim i menjëhershëm      |

Për `fire_flame_v6`: `low = 0.0033`, `high = 0.9389`.

- `low` = pragu më i lartë që ende mban recall ≥ 99% për flakën.
- `high` = pragu më i ulët që jep specificity ≥ 90% mbi jo-flakën.

Të dyja u përshtatën mbi `validation_flame_reviewed_v1` **pa kuadrot e
liqenit**. Precision nuk përdoret si cak, sepse varet nga përbërja e setit.
Liqeni përjashtohet sepse dallohet trivialisht: 67.6% e negativave të
validation-it janë liqen, kurse negativat e `test_final` janë pyll me mjegull
dhe tym. Me liqenin brenda, specificity e validation-it është optimiste dhe
nuk transferohet.

Shiriti në ndërfaqe e ndan gjerësinë në tri pjesë të barabarta, një për zonë,
sepse me këto porta një shkallë lineare do ta bënte zonën e pastër 0.3% të
gjerë.

## Rezultatet dhe kufizimet

Mbi `test_final` (2,367 foto pas rietiketimit, 23.7% me flakë; asnjëri model
nuk u trajnua mbi të):

| | tymi (`final_fire_v1`) | flaka (`fire_flame_v6`) |
| --- | --- | --- |
| ROC AUC | 0.696 | 0.855 |
| flakë e dukshme, % ≥ 0.5 | 99.8% | 92.0% |
| tym pa flakë, % ≥ 0.5 | 99.8% | 50.0% |
| pa zjarr fare, % ≥ 0.5 | 59.4% | 23.7% |

Me portat e `fire_flame_v6`: humbje flakësh 2.0%, pastrim automatik 24.1%,
alarm 32.2% me saktësi 54.9%, shqyrtim njerëzor 43.7%.

- **Kontrolli i verbër njerëzor nuk e konfirmoi përmirësimin.** Mbi 200 pamje të
  etiketuara nga njeriu, AUC është 0.792 (tymi) kundrejt 0.794 (flaka), diferenca
  +0.002 me interval 95% [−0.062, +0.069]. Rregulli i ngjyrës i humbi 40 nga 61
  flakët që pa njeriu, dhe modeli i flakës i mësoi ato pika të verbra. Detajet
  te `METODOLOGJIA.md`, seksioni "Kontrolli i verbër njerëzor".
- **Etiketat u nxorën nga një rregull ngjyre** i kalibruar mbi negativat e
  njohura dhe i verifikuar vizualisht mbi montazhe, jo nga shqyrtim
  foto-për-foto. Etiketat e `test_final` u nxorën nga i njëjti rregull, prandaj
  AUC 0.855 është optimist. Krahasimi mes dy modeleve mbetet i drejtë.
- **Tymi pa flakë** ende jep alarm në 38% të rasteve (specificity 0.619).
- **Saktësia e alarmit varet nga përbërja.** 54.9% vlen për një set me 23.7%
  flakë.
- **Përgjithësimi është i paprovuar.** `test_final` vjen nga të njëjtat
  regjistrime si trajnimi, me kuadro të tjera; nevojitet një fluturim i ri.
- **Dalja nuk është e kalibruar.** Është sigmoid, jo probabilitet.
- **Grad-CAM** tregon ku u aktivizua rrjeti, **nuk** e lokalizon zjarrin.

Sistemi është prototip akademik. Një përgjigje "E pastër" nuk e përjashton
zjarrin.
