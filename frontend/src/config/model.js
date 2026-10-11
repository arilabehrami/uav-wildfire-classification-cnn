// Pasqyron backend/artifacts/ensemble_v1/inference.json dhe pragjet e nxjerra
// mbi test_adapt_val. Nje burim i vetem per UI-ne; backend-i mbetet autoritet.
export const MODEL = {
  run: "final_fire_v1",
  file: "model.keras",
  architecture: "MobileNetV2 + head",
  sha256: "f4d96699ec3fecee541d3171aca12d06caa92cba93a9e69c1ae67e73374be5eb",
  input: "RGB 254x254, [0,1]",
  sizeLabel: "9.2 MB",
};

// t_low dhe t_high jane nxjerre mbi test_adapt_val (890 foto) dhe jane zbatuar
// pa ndryshim mbi test_final. Nuk jane pershtatur mbi setin e raportimit.
export const GATES = {
  low: 0.5244,
  high: 0.9713,
  fittedOn: "test_adapt_val (890 foto)",
};

// Te matura njehere mbi test_final (3463 foto: 2100 Fire, 1363 No_Fire).
export const MEASURED = {
  testSet: "test_final",
  n: 3463,
  rocAuc: 0.8041,
  autoShare: 0.367,
  clearShare: 0.164,
  missedFires: 0.0067,
  alarmShare: 0.204,
  alarmPrecision: 0.874,
};

// dataset/Test e ndarë në 6 segmente kohore, porta fikse. Numrat e mësipërm
// janë mesatare mbi këto gjashtë kushte, që ndryshojnë shumë mes tyre.
export const SEGMENTS = {
  k: 6,
  aucMedian: 0.8884,
  aucMin: 0.5547,
  aucMax: 1.0,
  missedFireMax: 0.575,
};

export const ZONES = {
  alarm: { id: "alarm", title: "Alarm", note: "Zjarr shumë i mundshëm. Njoftim i menjëhershëm." },
  review: { id: "review", title: "Për shqyrtim", note: "Sistemi nuk është i sigurt. Duhet ta shohë një njeri." },
  clear: { id: "clear", title: "E pastër", note: "Nuk u gjet zjarr. Asnjë veprim." },
};

export function zoneFor(score) {
  if (score >= GATES.high) return ZONES.alarm;
  if (score < GATES.low) return ZONES.clear;
  return ZONES.review;
}
