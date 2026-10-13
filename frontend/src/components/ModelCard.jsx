/* Udhezim per lexuesin: çfare do te thote secila pergjigje dhe ku eshte kufiri
   i sistemit. Detajet teknike (modeli, portat, hash-i) jane te punimi. */
function ModelCard() {
  return (
    <section className="panel">
      <h3>SI TA LEXOSH REZULTATIN</h3>
      <ul className="guide">
        <li>
          <i className="z-clear" />
          <div>
            <b className="t-clear">E pastër</b>
            <span>Nuk u gjet zjarr. Nuk duhet asnjë veprim.</span>
          </div>
        </li>
        <li>
          <i className="z-review" />
          <div>
            <b className="t-review">Për shqyrtim</b>
            <span>Sistemi nuk është i sigurt. Duhet ta shohë një njeri.</span>
          </div>
        </li>
        <li>
          <i className="z-alarm" />
          <div>
            <b className="t-alarm">Alarm</b>
            <span>Zjarr shumë i mundshëm. Njoftim i menjëhershëm.</span>
          </div>
        </li>
      </ul>
      <p className="caveat">
        Sistemi kërkon flakë të dukshme. Tymi pa flakë nuk numërohet si zjarr,
        prandaj një kuadër me tym mund të dalë "E pastër" ose "Për shqyrtim".
        Mund të mos e dallojë një flakë shumë të vogël ose të largët; një
        përgjigje "E pastër" nuk e përjashton plotësisht zjarrin.
      </p>
    </section>
  );
}

export default ModelCard;
