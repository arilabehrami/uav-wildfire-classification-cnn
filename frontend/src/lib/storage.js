// Regjistri i klasifikimeve ruhet ne localStorage te shfletuesit, qe te mos
// humbase me rifreskimin e faqes. Ruhen vetem fushat tekstuale: thumbnail-at
// jane object URL dhe kane kuptim vetem brenda seances qe i krijoi.
//
// Çdo lexim dhe shkrim eshte ne try/catch: ne dritare private, me kuota te
// mbushura ose me site data te bllokuar, qasja hedh gabim. Ne ate rast
// aplikacioni punon njesoj, vetem historia mbetet brenda seances.

const KEY = "wildfire.detections.v1";
const LIMIT = 300;
const ZONE_IDS = new Set(["clear", "review", "alarm"]);
const VERDICTS = new Set(["fire", "no_fire"]);

function isRecord(value) {
  return (
    value !== null &&
    typeof value === "object" &&
    typeof value.id === "string" &&
    typeof value.name === "string" &&
    typeof value.at === "string" &&
    Number.isFinite(value.score) &&
    ZONE_IDS.has(value.zoneId) &&
    (value.verdict === null ||
      value.verdict === undefined ||
      VERDICTS.has(value.verdict))
  );
}

export function loadRecords() {
  try {
    const raw = localStorage.getItem(KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed.filter(isRecord).slice(0, LIMIT);
  } catch {
    return [];
  }
}

export function saveRecords(records) {
  try {
    const slim = records.slice(0, LIMIT).map((record) => ({
      id: record.id,
      name: record.name,
      at: record.at,
      score: record.score,
      zoneId: record.zoneId,
      verdict: record.verdict ?? null,
      modelVersion: record.modelVersion ?? null,
    }));
    localStorage.setItem(KEY, JSON.stringify(slim));
    return true;
  } catch {
    return false;
  }
}

export function clearRecords() {
  try {
    localStorage.removeItem(KEY);
  } catch {
    /* nuk ka ku te pastrohet; gjendja ne memorie pastrohet nga thirresi */
  }
}

export const RECORD_LIMIT = LIMIT;
