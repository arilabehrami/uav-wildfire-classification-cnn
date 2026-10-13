/* Shiriti tregon ku ra pamja mes tri zonave. Portat vijne si prop nga
   pergjigjja e API-se, jo nga konfigurimi: nese modeli ndryshohet me
   FIRE_MODEL_RUN, shiriti leviz bashke me te.

   Shkalla eshte SIPAS ZONAVE, jo lineare: secila zone zene nje te treten e
   shiritit, dhe brenda saj gjilpera leviz linearisht. Me portat e modelit te
   flakes (0.0033 / 0.9389) nje shkalle lineare do ta bente zonen e paster nje
   vije 0.3% te gjere. Renditja ruhet dhe gjilpera bie gjithmone ne zonen e
   sakte; distanca vizuale nuk eshte distance ne score.

   Numrat (rezultati, portat) nuk shfaqen: per perdoruesin mjafton zona,
   kurse vlerat e sakta jane te punimi dhe ne eksportin e regjistrit. */
const clamp01 = (v) => Math.min(Math.max(v, 0), 1);
const pct = (v) => `${(clamp01(v) * 100).toFixed(2)}%`;

function place(score, gates) {
  const s = clamp01(score);
  const bounds = [
    [0, gates.low],
    [gates.low, gates.high],
    [gates.high, 1],
  ];
  const i = s < gates.low ? 0 : s < gates.high ? 1 : 2;
  const [a, b] = bounds[i];
  const t = b > a ? clamp01((s - a) / (b - a)) : 0.5;
  return (i + t) / 3;
}

function Verdict({ verdict, onVerdict }) {
  return (
    <div className="verdict">
      <span className="verdict-ask">Vendimi i shqyrtuesit</span>
      <div className="verdict-buttons">
        <button
          type="button"
          className={`btn small${verdict === "fire" ? " on t-alarm" : ""}`}
          aria-pressed={verdict === "fire"}
          onClick={() => onVerdict("fire")}
        >
          Është zjarr
        </button>
        <button
          type="button"
          className={`btn small${verdict === "no_fire" ? " on t-clear" : ""}`}
          aria-pressed={verdict === "no_fire"}
          onClick={() => onVerdict("no_fire")}
        >
          Nuk është zjarr
        </button>
      </div>
      <span className="verdict-hint">
        {verdict
          ? "Shtypja e sërishme e heq vendimin."
          : "Vendimi ruhet në regjistër dhe del në eksport."}
      </span>
    </div>
  );
}

function DecisionBand({ score, zone, gates, verdict, onVerdict }) {
  const has = typeof score === "number" && Number.isFinite(score) && Boolean(zone);
  const position = has ? place(score, gates) : 0;
  const third = 1 / 3;

  return (
    <section className={has ? "band" : "band idle"} aria-live="polite">
      <div className="band-head">
        <div className="band-verdict">
          <div className="band-label">PËRGJIGJJA</div>
          <h2 className={has ? `band-call t-${zone.id}` : "band-call"}>
            {has ? zone.title : "Asnjë pamje e kontrolluar"}
          </h2>
          <p className="band-note">
            {has ? zone.note : "Ngarko pamje ajrore dhe shtyp Klasifiko."}
          </p>
        </div>
      </div>

      <div className="scale">
        <div className="scale-track">
          <div
            className={`seg seg-clear${has && zone.id === "clear" ? " on" : ""}`}
            style={{ width: pct(third) }}
          />
          <div
            className={`seg seg-review${has && zone.id === "review" ? " on" : ""}`}
            style={{ width: pct(third) }}
          />
          <div
            className={`seg seg-alarm${has && zone.id === "alarm" ? " on" : ""}`}
            style={{ width: pct(third) }}
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
          style={{ left: "50%" }}
        >
          Për shqyrtim
        </span>
        <span className="t-alarm" style={{ right: 0 }}>
          Alarm
        </span>
      </div>

      {has && zone.id === "review" && onVerdict && (
        <Verdict verdict={verdict} onVerdict={onVerdict} />
      )}
    </section>
  );
}

export default DecisionBand;
