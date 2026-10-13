import { useMemo, useState } from "react";
import { ZONES } from "../config/model";
import { download, stamp, toCsv, toJson } from "../lib/exportLog";

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
    <section className="panel grow">
      <div className="panel-head">
        <h3>REGJISTRI{records.length > 0 ? ` · ${records.length}` : ""}</h3>
        {records.length > 0 && (
          <div className="panel-tools">
            <button type="button" className="chip" onClick={() => save("csv")}>
              CSV
            </button>
            <button type="button" className="chip" onClick={() => save("json")}>
              JSON
            </button>
            {confirming ? (
              <>
                <button
                  type="button"
                  className="chip danger"
                  onClick={() => {
                    onClearHistory();
                    setConfirming(false);
                  }}
                >
                  Konfirmo
                </button>
                <button
                  type="button"
                  className="chip"
                  onClick={() => setConfirming(false)}
                >
                  Anulo
                </button>
              </>
            ) : (
              <button
                type="button"
                className="chip"
                onClick={() => setConfirming(true)}
                title="Fshi gjithë regjistrin e ruajtur"
              >
                Fshi
              </button>
            )}
          </div>
        )}
      </div>

      {!persisting && records.length > 0 && (
        <p className="empty">
          Shfletuesi nuk lejon ruajtjen lokale, prandaj regjistri do të humbasë
          me rifreskimin e faqes. Eksportoje para se ta mbyllësh.
        </p>
      )}

      {records.length === 0 ? (
        <p className="empty">
          Asgjë ende. Çdo pamje e kontrolluar shfaqet këtu me përgjigjen e saj
          dhe ruhet edhe pas rifreskimit.
        </p>
      ) : (
        <>
          {open > 0 && (
            <p className="empty">
              {open} {open === 1 ? "rast pret" : "raste presin"} vendimin e
              shqyrtuesit.
            </p>
          )}

          <div className="log">
            {records.map((record) => {
              const zone = ZONES[record.zoneId];
              const thumb = thumbs.get(record.id);

              return (
                <div className="log-row" key={record.id}>
                  {thumb ? (
                    <img className="log-thumb" src={thumb} alt="" />
                  ) : (
                    <i className={`z-${record.zoneId}`} />
                  )}

                  <span className="log-main">
                    <span className="name" title={record.name}>
                      {record.name}
                    </span>
                    <span className="at">
                      {when(record.at)}
                      {record.verdict && (
                        <>
                          {" · njeriu: "}
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
                          className="chip tiny"
                          onClick={() => onVerdict(record.id, "fire")}
                        >
                          zjarr
                        </button>
                        <button
                          type="button"
                          className="chip tiny"
                          onClick={() => onVerdict(record.id, "no_fire")}
                        >
                          pa zjarr
                        </button>
                      </span>
                    )}
                  </span>

                  <span className={`score t-${record.zoneId}`}>{zone.title}</span>
                </div>
              );
            })}
          </div>
        </>
      )}
    </section>
  );
}

export default DetectionLog;
