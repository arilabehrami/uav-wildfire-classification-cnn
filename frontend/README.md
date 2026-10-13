# Ndërfaqja

React 19 + Vite. Klient i `POST /api/classify` të backend-it.

```bash
npm install
cp .env.example .env     # Windows: copy .env.example .env
npm run dev              # http://localhost:5173
npm run lint
npm run build            # prodhimi ne dist/
```

`VITE_API_BASE` cakton adresën e backend-it (parazgjedhja
`http://localhost:8000`). Çdo origjinë e re duhet shtuar edhe në
`FIRE_CORS_ORIGINS` të backend-it.

## Struktura

| Fajlli                     | Roli                                                        |
| -------------------------- | ----------------------------------------------------------- |
| `App.jsx`                  | radha e pamjeve, kërkesat, gjendja, regjistri               |
| `components/ImageUpload`   | ngarkimi, pamja e fokusuar, harta Grad-CAM, radha            |
| `components/DecisionBand`  | shiriti i tri zonave (shkallë sipas zonave) dhe verdikti    |
| `components/DetectionLog`  | regjistri i ruajtur, eksporti CSV/JSON                      |
| `components/SessionStats`  | numrat përmbledhës                                          |
| `components/ModelCard`     | si lexohet rezultati                                        |
| `components/ErrorBoundary` | kufiri i gabimeve të render-it                              |
| `lib/storage.js`           | regjistri në `localStorage`                                 |
| `lib/exportLog.js`         | CSV (me BOM) dhe JSON                                       |
| `config/api.js`            | adresat e API-së                                            |
| `config/model.js`          | vlera rezervë dhe rregulli i zonave                         |

## Dy gjëra që nuk duhen ngatërruar

**Portat vijnë nga API-ja.** `config/model.js` mban vetëm vlera rezervë për
momentin para përgjigjes së parë. Në ekzekutim, shiriti përdor portat e
`gates` të përgjigjes, pra nuk tregon kurrë kufij të ndryshëm nga ata me të
cilët u marr vendimi. Testi `backend/tests/test_contract.py` dështon nëse
vlerat rezervë largohen nga `inference.json`.

**Harta dhe klasifikimi janë rezultate të ndryshme.** Grad-CAM shfaqet vetëm
për vendimet pozitive, sepse për një dalje afër zeros gradientet janë zhurmë.
Mungesa e hartës nuk thotë që nuk ka zjarr, dhe ndërfaqja nuk e thotë atë.

## Radha

Deri 50 foto njëherësh, të klasifikuara në seri. Rezultatet shkruhen sipas
id-së së elementit, prandaj ndërrimi i pamjes gjatë përpunimit nuk mund t'i
ngatërrojë. Nëse serveri nuk përgjigjet, radha ndalet dhe fotot e mbetura
mbesin në pritje.
