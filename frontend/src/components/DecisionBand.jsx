import { GATES } from "../config/model";

const pct = (v) => `${(v * 100).toFixed(2)}%`;

/* Shiriti tregon ku ra pamja mes tri zonave. Numrat (rezultati, portat) nuk
   shfaqen: per perdoruesin mjafton zona; vlerat e sakta jane te punimi. */
function DecisionBand({ score, zone }) {
  const has = typeof score === "number" && Number.isFinite(score);
  const position = has ? Math.min(Math.max(score, 0), 1) : 0;

  return (
    <section className={has ? "band" : "band idle"} aria-live="polite">
      <div className="band-head">
        <div className="band-verdict">
          <div className="band-label">PËRGJIGJJA</div>
          <h2 className={has ? `band-call t-${zone.id}` : "band-call"}>
            {has ? zone.title : "Asnjë pamje e kontrolluar"}
          </h2>
          <p className="band-note">
            {has ? zone.note : "Ngarko një pamje ajrore dhe shtyp Klasifiko."}
          </p>
        </div>
      </div>

      <div className="scale">
        <div className="scale-track">
          <div
            className={`seg seg-clear${has && zone.id === "clear" ? " on" : ""}`}
            style={{ width: pct(GATES.low) }}
          />
          <div
            className={`seg seg-review${has && zone.id === "review" ? " on" : ""}`}
            style={{ width: pct(GATES.high - GATES.low) }}
          />
          <div
            className={`seg seg-alarm${has && zone.id === "alarm" ? " on" : ""}`}
            style={{ width: pct(1 - GATES.high) }}
          />
        </div>
        {has && <div className="needle" style={{ left: pct(position) }} />}
      </div>

      <div className="zone-labels">
        <span className="t-clear" style={{ left: 0 }}>
          E pastër
        </span>
        <span
          className="t-review center"
          style={{ left: pct((GATES.low + GATES.high) / 2) }}
        >
          Për shqyrtim
        </span>
        <span className="t-alarm" style={{ right: 0 }}>
          Alarm
        </span>
      </div>
    </section>
  );
}

export default DecisionBand;
