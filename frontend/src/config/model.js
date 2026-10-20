// Vlera rezerve per UI-ne, te kopjuara nga backend/artifacts/fire_flame_scene_repair_v1/
// inference.json. Ne ekzekutim, portat vijne me pergjigjen e API-se dhe keto
// vlera nuk perdoren: backend-i mbetet autoritet. Ato sherbejne vetem derisa
// te kthehet pergjigjja e pare, dhe testi backend/tests/test_contract.py
// deshton nese largohen nga fajlli qe pretendojne te pasqyrojne.
export const MODEL = {
  run: "fire_flame_scene_repair_v1",
  file: "model.keras",
  architecture: "MobileNetV2 + concat(avg, max) + head",
  sha256: "6db33f6ad7af4d6f1f890d59346b99289d0dcac3bfd4a53aabd60b25a8547c8d",
  input: "RGB 254x254, [0,1]",
  sizeLabel: "9.2 MB",
  policy: "Fire = flake e dukshme. Tym pa flake eshte No_Fire.",
};

// Portat vijne nga validation_flame_scene_repair_v1 pa kuadrot e liqenit.
// Ky variant ende nuk ka vleresim te pavarur.
export const GATES = {
  low: 0.008799999952316284,
  high: 0.9368000030517578,
  fittedOn: "validation pa liqen (5962 foto)",
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
