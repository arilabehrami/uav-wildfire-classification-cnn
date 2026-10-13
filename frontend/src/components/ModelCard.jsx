/* Udhezim per lexuesin: çfare do te thote secila pergjigje dhe ku eshte kufiri
   i sistemit. Mbahet i mbyllur qe faqja te mbetet e thjeshte, por kufizimi
   shfaqet gjithmone ne nje rresht, sepse nuk eshte detaj opsional.
   Detajet teknike (modeli, portat, hash-i) jane te punimi. */
function ModelCard() {
  return (
    <section className="guide">
      <p className="guide-limit">
        Sistemi kërkon flakë të dukshme. Një përgjigje “E pastër” nuk e
        përjashton plotësisht zjarrin.
      </p>
      <details>
        <summary>Si ta lexosh përgjigjen</summary>
        <ul className="guide-list">
          <li className="z-clear">
            <b className="t-clear">E pastër</b>
            <span>Nuk u gjet flakë. Nuk duhet asnjë veprim.</span>
          </li>
          <li className="z-review">
            <b className="t-review">Për shqyrtim</b>
            <span>Sistemi nuk është i sigurt. Duhet ta shohë një njeri.</span>
          </li>
          <li className="z-alarm">
            <b className="t-alarm">Alarm</b>
            <span>Flakë shumë e mundshme. Njoftim i menjëhershëm.</span>
          </li>
        </ul>
        <p className="guide-more">
          Tymi pa flakë nuk numërohet si zjarr, prandaj një pamje me tym mund të
          dalë “E pastër” ose “Për shqyrtim”. Një flakë shumë e vogël ose e
          largët mund të mos dallohet.
        </p>
      </details>
    </section>
  );
}

export default ModelCard;
