import { GATES, MEASURED } from "../config/model";

const pct = (v) => `${(v * 100).toFixed(2)}%`;

function DecisionBand({ score, zone, modelVersion }) {
  const has = typeof score === "number" && Number.isFinite(score);
  const position = has ? Math.min(Math.max(score, 0), 1) : 0;

  return (
    <section className={has ? "band" : "band idle"} aria-live="polite">
      <div className="band-head">
        <div className="band-verdict">
          <div className="band-label">Verdikti</div>
          <h2 className={has ? `band-call t-${zone.id}` : "band-call"}>
            {has ? zone.title : "Asnjë frame i klasifikuar"}
          </h2>
          <p className="band-note">
            {has
              ? zone.note
              : "Ngarko një pamje ajrore për ta vendosur mbi shkallë."}
          </p>
        </div>

        <div className="band-facts">
          <div>
            <div className="fact-k">rezultat fire</div>
            <div className="fact-v">{has ? score.toFixed(4) : "—"}</div>
          </div>
          <div>
            <div className="fact-k">portë e poshtme</div>
            <div className="fact-v">{GATES.low.toFixed(4)}</div>
          </div>
          <div>
            <div className="fact-k">portë e sipërme</div>
            <div className="fact-v">{GATES.high.toFixed(4)}</div>
          </div>
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

        <div className="gate" style={{ left: pct(GATES.low) }} />
        <div className="gate" style={{ left: pct(GATES.high) }} />
        {has && <div className="needle" style={{ left: pct(position) }} />}
      </div>

      <div className="scale-labels">
        <div className="tick-0">0.00</div>
        <div className="tick-gate" style={{ left: pct(GATES.low) }}>
          {GATES.low.toFixed(4)}
          <em>portë e poshtme</em>
        </div>
        <div className="tick-gate" style={{ left: pct(GATES.high) }}>
          {GATES.high.toFixed(4)}
          <em>portë e sipërme</em>
        </div>
        <div className="tick-1">1.00</div>
      </div>

      <div className="band-foot">
        <span>
          Portat e nxjerra mbi {GATES.fittedOn}, të zbatuara pa ndryshim mbi{" "}
          {MEASURED.testSet}.
        </span>
        {modelVersion && <span>{modelVersion}</span>}
      </div>
    </section>
  );
}

export default DecisionBand;
