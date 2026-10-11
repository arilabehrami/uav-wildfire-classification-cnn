// Gjeneruar nga backend/artifacts/final_fire_v1/segment_stability_6.json.
// Mos e ndrysho me dorë — ri-gjenerohet kur ndryshon raporti.

export const SEGMENT_RUN = {
  k: 6,
  gap: 200,
  gates: { low: 0.5244, high: 0.9713 },
  refitPerSegment: false,
};

export const SEGMENTS = [
  { id: 1, n: 1236, fire: 756, frames: [0, 755], auc: 0.5547, alarmShare: 0.1683, alarmPrecision: 1.0, clearShare: 0.5615, missedFires: 0.5754, autoShare: 0.7298 },
  { id: 2, n: 1036, fire: 656, frames: [956, 1611], auc: 1.0, alarmShare: 0.584, alarmPrecision: 1.0, clearShare: 0.1602, missedFires: 0.0, autoShare: 0.7442 },
  { id: 3, n: 1036, fire: 656, frames: [1812, 2467], auc: 0.8524, alarmShare: 0.5261, alarmPrecision: 0.8771, clearShare: 0.0077, missedFires: 0.0015, autoShare: 0.5338 },
  { id: 4, n: 1037, fire: 657, frames: [2668, 3324], auc: 0.9412, alarmShare: 0.3915, alarmPrecision: 0.9951, clearShare: 0.1688, missedFires: 0.0091, autoShare: 0.5603 },
  { id: 5, n: 1036, fire: 656, frames: [3525, 4180], auc: 0.5804, alarmShare: 0.0203, alarmPrecision: 0.619, clearShare: 0.0656, missedFires: 0.0183, autoShare: 0.0859 },
  { id: 6, n: 1236, fire: 756, frames: [4381, 5136], auc: 0.9244, alarmShare: 0.2379, alarmPrecision: 0.9592, clearShare: 0.2403, missedFires: 0.0, autoShare: 0.4782 },
];

export const SEGMENT_SUMMARY = {
  aucMedian: 0.8884,
  aucMin: 0.5547,
  aucMax: 1.0,
  missedMax: 0.5754,
  autoMedian: 0.547,
  alarmPrecisionMin: 0.619,
};

// Gama e AUC-së mbetet e gjerë sido që ndahet regjistrimi.
export const BY_K = [
  { k: 4,  median: 0.8305, min: 0.6526, max: 0.956,  missedMax: 0.375 },
  { k: 6,  median: 0.8884, min: 0.5547, max: 1.0,    missedMax: 0.575 },
  { k: 8,  median: 0.9353, min: 0.361,  max: 0.9998, missedMax: 0.803 },
  { k: 10, median: 0.8528, min: 0.4343, max: 1.0,    missedMax: 0.913 },
];
