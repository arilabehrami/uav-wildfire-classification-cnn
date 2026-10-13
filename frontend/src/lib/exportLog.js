// Eksporti i regjistrit. CSV-ja nis me BOM, qe Excel-i t'i lexoje sakte
// shkronjat shqipe; JSON-i mban edhe versionin e modelit per gjurmueshmeri.

const HEAD = ["koha", "fajlli", "rezultati", "zona", "verdikti_njerezor", "modeli"];
const VERDICT_TEXT = { fire: "zjarr", no_fire: "pa zjarr" };

function cell(value) {
  const text = value === null || value === undefined ? "" : String(value);
  return /[",;\r\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
}

export function toCsv(records) {
  const rows = records.map((record) => [
    record.at,
    record.name,
    record.score.toFixed(6),
    record.zoneId,
    record.verdict ? VERDICT_TEXT[record.verdict] : "",
    record.modelVersion ?? "",
  ]);

  return (
    "﻿" +
    [HEAD, ...rows].map((row) => row.map(cell).join(",")).join("\r\n")
  );
}

export function toJson(records) {
  return JSON.stringify(
    {
      exportuar: new Date().toISOString(),
      numri: records.length,
      shenim:
        "Rezultati eshte dalje sigmoid, jo probabilitet i kalibruar. " +
        "Zona vjen nga portat e backend-it ne momentin e klasifikimit.",
      regjistrimet: records,
    },
    null,
    2,
  );
}

export function download(filename, text, mime) {
  const url = URL.createObjectURL(new Blob([text], { type: mime }));
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  // Revokimi shtyhet nje tik, qe shfletuesi te arrije te nise shkarkimin.
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export function stamp() {
  const now = new Date();
  const pad = (value) => String(value).padStart(2, "0");
  return (
    `${now.getFullYear()}${pad(now.getMonth() + 1)}${pad(now.getDate())}` +
    `-${pad(now.getHours())}${pad(now.getMinutes())}`
  );
}
