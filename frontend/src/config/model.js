// Vlera rezerve per UI-ne, te kopjuara nga backend/artifacts/fire_flame_v6/
// inference.json. Ne ekzekutim, portat vijne me pergjigjen e API-se dhe keto
// vlera nuk perdoren: backend-i mbetet autoritet. Ato sherbejne vetem derisa
// te kthehet pergjigjja e pare, dhe testi backend/tests/test_contract.py
// deshton nese largohen nga fajlli qe pretendojne te pasqyrojne.
export const MODEL = {
  run: "fire_flame_v6",
  file: "model.keras",
  architecture: "MobileNetV2 + concat(avg, max) + head",
  sha256: "b80b82b06a6b999bb49ddf1d037312b271430ebe5f61787665fcdd08e742f426",
  input: "RGB 254x254, [0,1]",
  sizeLabel: "9.2 MB",
  policy: "Fire = flake e dukshme. Tym pa flake eshte No_Fire.",
};

// low dhe high jane nxjerre mbi validation_flame_reviewed_v1 pa kuadrot e
// liqenit, dhe jane zbatuar pa ndryshim mbi test_final.
export const GATES = {
  low: 0.0033,
  high: 0.9389,
  fittedOn: "validation pa liqen (5962 foto)",
};

// Te matura njehere mbi test_final, i rietiketuar sipas politikes se flakes
// (2367 foto, 23.7% me flake). Saktesia e alarmit varet nga ky perqindje.
export const MEASURED = {
  testSet: "test_final (flake)",
  n: 2367,
  fire: 560,
  autoShare: 0.5627,
  clearShare: 0.2408,
  alarmShare: 0.3219,
  alarmPrecision: 0.5486,
  missedFires: 0.0196,
};

export const ZONES = {
  alarm: { id: "alarm", title: "Alarm", note: "Flakë shumë e mundshme. Njoftim i menjëhershëm." },
  review: { id: "review", title: "Për shqyrtim", note: "Sistemi nuk është i sigurt. Duhet ta shohë një njeri." },
  clear: { id: "clear", title: "E pastër", note: "Nuk u gjet flakë. Asnjë veprim." },
};

// Portat kalohen si argument: i njejti rregull si backend/app/ml/zones.py,
// me kufijte score < low -> clear dhe score >= high -> alarm.
export function zoneIdFor(score, gates = GATES) {
  if (score >= gates.high) return "alarm";
  if (score < gates.low) return "clear";
  return "review";
}

export function zoneFor(score, gates = GATES) {
  return ZONES[zoneIdFor(score, gates)];
}
