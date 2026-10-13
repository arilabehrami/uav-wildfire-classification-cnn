/* Shiriti tregon ku ra pamja mes tri zonave. Portat vijne si prop nga
   pergjigjja e API-se, jo nga konfigurimi: nese modeli ndryshohet me
   FIRE_MODEL_RUN, shiriti leviz bashke me te.

   Shkalla eshte SIPAS ZONAVE, jo lineare: secila zone zene nje te treten e
   shiritit, dhe brenda saj gjilpera leviz linearisht. Me portat e modelit te
   flakes (0.0033 / 0.9389) nje shkalle lineare do ta bente zonen e paster nje
   vije 0.3% te gjere. Renditja ruhet dhe gjilpera bie gjithmone ne zonen e
   sakte; distanca vizuale nuk eshte distance ne score.

   Numrat (rezultati, portat) nuk shfaqen: per perdoruesin mjafton zona,
   kurse vlerat e sakta jane te punimi dhe ne eksportin e regjistrit.
   Kur pamja ende nuk ka pergjigje, paneli tregon gjendjen e saj (ne pritje,
   duke u analizuar, gabim) ne vend te nje shiriti bosh. */
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
    <div className="review-ask">
      <p className="review-q">Çka sheh ti në këtë pamje?</p>
      <div className="review-buttons">
        <button
          type="button"
          className={`btn pick${verdict === "fire" ? " on t-alarm" : ""}`}
          aria-pressed={verdict === "fire"}
          onClick={() => onVerdict("fire")}
        >
          Është zjarr
        </button>
        <button
          type="button"
          className={`btn pick${verdict === "no_fire" ? " on t-clear" : ""}`}
          aria-pressed={verdict === "no_fire"}
          onClick={() => onVerdict("no_fire")}
        >
          Nuk është zjarr
        </button>
      </div>
      <p className="review-hint">
        {verdict
          ? "U ruajt në regjistër. Shtype sërish për ta hequr."
          : "Vendimi yt ruhet në regjistër dhe del në eksport."}
      </p>
    </div>
  );
}

const IDLE = {
  none: { title: "Në pritje", note: "Zgjidh një ose më shumë pamje ajrore për të filluar." },
  pending: { title: "Gati", note: "Shtyp Klasifiko që sistemi ta analizojë këtë pamje." },
  running: { title: "Në analizë", note: "Modeli po e kontrollon pamjen për flakë të dukshme." },
  error: { title: "Pa përgjigje", note: "Kjo pamje nuk u kontrollua. Arsyeja shkruhet nën pamje." },
};

const SEGMENTS = [
  { id: "clear", label: "E pastër" },
  { id: "review", label: "Për shqyrtim" },
  { id: "alarm", label: "Alarm" },
];

function DecisionBand({ status, score, zone, gates, verdict, onVerdict }) {
  const has = typeof score === "number" && Number.isFinite(score) && Boolean(zone);
  const position = has ? place(score, gates) : 0;
  const idle = IDLE[status ?? "none"] ?? IDLE.none;

  return (
    <section
      className={has ? `decision z-${zone.id}` : `decision idle s-${status ?? "none"}`}
      aria-live="polite"
      aria-label="Përgjigjja e sistemit"
    >
      <p className="decision-label">Përgjigjja e sistemit</p>
      <h2 className={has ? `call t-${zone.id}` : "call"} key={has ? `${zone.id}-${score}` : status}>
        {has ? zone.title : idle.title}
      </h2>
      <p className="call-note">{has ? zone.note : idle.note}</p>

      <div className="scale" aria-hidden={!has}>
        <div className="scale-bars">
          {SEGMENTS.map((segment) => (
            <span
              key={segment.id}
              className={`bar bar-${segment.id}${has && zone.id === segment.id ? " on" : ""}`}
            />
          ))}
          {has && <span className="needle" style={{ left: pct(position) }} />}
        </div>
        <div className="scale-labels">
          {SEGMENTS.map((segment) => (
            <span
              key={segment.id}
              className={`t-${segment.id}${has && zone.id === segment.id ? " on" : ""}`}
            >
              {segment.label}
            </span>
          ))}
        </div>
      </div>

      {has && zone.id === "review" && onVerdict && (
        <Verdict verdict={verdict} onVerdict={onVerdict} />
      )}
    </section>
  );
}

export default DecisionBand;
