import { MODEL } from "../config/model";
import { SEGMENT_SUMMARY } from "../config/segments";

/* Modeli dhe portat rrine te headeri. Ketu mbetet ajo qe nuk hyn atje:
   gjurma e fajllit dhe kufiri i besueshmerise. */
function ModelCard() {
  return (
    <section className="panel">
      <h3>Modeli</h3>
      <dl className="spec">
        <div>
          <dt>Arkitektura</dt>
          <dd>{MODEL.architecture}</dd>
        </div>
        <div>
          <dt>Hyrja</dt>
          <dd>{MODEL.input}</dd>
        </div>
        <div>
          <dt>sha256</dt>
          <dd>{MODEL.sha256.slice(0, 12)}</dd>
        </div>
      </dl>
      <p className="caveat">
        Nëpër segmente të regjistrimit AUC-ja shkon{" "}
        {SEGMENT_SUMMARY.aucMin.toFixed(3)}–{SEGMENT_SUMMARY.aucMax.toFixed(3)}.
        Ka pamje ku ky model nuk dallon fare.
      </p>
    </section>
  );
}

export default ModelCard;
