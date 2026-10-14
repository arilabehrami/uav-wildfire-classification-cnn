import { useMemo, useState } from "react";
import { ZONES } from "../config/model";
import { download, stamp, toCsv, toJson } from "../lib/exportLog";
import SessionStats from "./SessionStats";

const VERDICT_LABEL = { fire: "zjarr", no_fire: "pa zjarr" };

/* Koha ruhet si ISO, qe eksporti te jete i perdorshem per analize; ne ekran
   shfaqet vetem ora, dhe data shtohet kur regjistrimi nuk eshte i sotem. */
function when(iso) {
  const at = new Date(iso);
  if (Number.isNaN(at.getTime())) return iso;

  const time = at.toLocaleTimeString("sq", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
  const today = new Date();
  const sameDay =
    at.getFullYear() === today.getFullYear() &&
    at.getMonth() === today.getMonth() &&
    at.getDate() === today.getDate();

  return sameDay
    ? time
    : `${at.toLocaleDateString("sq", { day: "2-digit", month: "2-digit" })} ${time}`;
}

function DetectionLog({ records, items, persisting, onVerdict, onClearHistory }) {
  const [confirming, setConfirming] = useState(false);

  // Thumbnail-at jetojne vetem brenda seances qe i krijoi; per regjistrimet e
  // ruajtura me pare shfaqet pika e zones.
  const thumbs = useMemo(() => {
    const map = new Map();
    items.forEach((item) => {
      if (item.preview) map.set(item.id, item.preview);
    });
    return map;
  }, [items]);

  const open = records.filter(
    (record) => record.zoneId === "review" && !record.verdict,
  ).length;

  const save = (format) => {
    if (!records.length) return;
    const name = `detektime-${stamp()}`;
    if (format === "csv") {
      download(`${name}.csv`, toCsv(records), "text/csv;charset=utf-8");
    } else {
      download(`${name}.json`, toJson(records), "application/json");
    }
  };

  return (
    <section className="log-panel" aria-label="Regjistri i klasifikimeve">
      <div className="log-head">
        <h3>
          Regjistri
          {records.length > 0 && <span className="log-count">{records.length}</span>}
        </h3>
        {records.length > 0 && (
          <div className="log-tools">
            {confirming ? (
              <>
                <span className="log-confirm">Të fshihet gjithë regjistri?</span>
                <button
                  type="button"
                  className="link danger"
                  onClick={() => {
                    onClearHistory();
                    setConfirming(false);
                  }}
                >
                  Fshije
                </button>
                <button type="button" className="link" onClick={() => setConfirming(false)}>
                  Anulo
                </button>
              </>
            ) : (
              <>
                <button
                  type="button"
                  className="link"
                  onClick={() => save("csv")}
                  title="Shkarko regjistrin si CSV (hapet në Excel)"
                >
                  CSV
                </button>
                <button
                  type="button"
                  className="link"
                  onClick={() => save("json")}
                  title="Shkarko regjistrin si JSON"
                >
                  JSON
                </button>
                <button
                  type="button"
                  className="link"
                  onClick={() => setConfirming(true)}
                  title="Fshi gjithë regjistrin e ruajtur"
                >
                  Fshi
                </button>
              </>
            )}
          </div>
        )}
      </div>

      {records.length > 0 && <SessionStats records={records} />}

      {!persisting && records.length > 0 && (
        <p className="log-note warn">
          Shfletuesi nuk lejon ruajtjen lokale, prandaj regjistri do të humbasë
          me rifreskimin e faqes. Shkarkoje para se ta mbyllësh.
        </p>
      )}

      {records.length === 0 ? (
        <p className="log-empty">
          Ende asnjë analizë. Rezultatet e tua do të ruhen këtu.
        </p>
      ) : (
        <>
          {open > 0 && (
            <p className="log-note">
              {open === 1
                ? "1 pamje pret vendimin tënd."
                : `${open} pamje presin vendimin tënd.`}
            </p>
          )}

          <ol className="log">
            {records.map((record) => {
              const zone = ZONES[record.zoneId];
              const thumb = thumbs.get(record.id);

              return (
                <li className={`log-row z-${record.zoneId}`} key={record.id}>
                  {thumb ? (
                    <img className="log-thumb" src={thumb} alt="" />
                  ) : (
                    <span className="log-thumb blank" aria-hidden="true" />
                  )}

                  <div className="log-main">
                    <span className="log-name" title={record.name}>
                      {record.name}
                    </span>
                    <span className="log-at">
                      {when(record.at)}
                      {record.verdict && (
                        <>
                          {", sipas njeriut "}
                          <b className={record.verdict === "fire" ? "t-alarm" : "t-clear"}>
                            {VERDICT_LABEL[record.verdict]}
                          </b>
                        </>
                      )}
                    </span>
                    {record.zoneId === "review" && !record.verdict && (
                      <span className="log-verdict">
                        <button
                          type="button"
                          className="mini"
                          onClick={() => onVerdict(record.id, "fire")}
                        >
                          Zjarr
                        </button>
                        <button
                          type="button"
                          className="mini"
                          onClick={() => onVerdict(record.id, "no_fire")}
                        >
                          Pa zjarr
                        </button>
                      </span>
                    )}
                  </div>

                  <span className={`log-zone t-${record.zoneId}`}>{zone.title}</span>
                </li>
              );
            })}
          </ol>
        </>
      )}
    </section>
  );
}

export default DetectionLog;
