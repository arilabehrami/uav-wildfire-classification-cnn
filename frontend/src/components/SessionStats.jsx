// Numrat vijne nga regjistri i ruajtur, pra mbijetojne rifreskimin e faqes.
// 'PA VENDIM' eshte numri i rasteve ne zonen e shqyrtimit per te cilat nuk
// ka vendosur ende njeri; ky numer, jo totali, tregon sa pune ka mbetur.
function SessionStats({ records }) {
  const count = (id) => records.filter((record) => record.zoneId === id).length;
  const open = records.filter(
    (record) => record.zoneId === "review" && !record.verdict,
  ).length;

  return (
    <div className="stats">
      <div className="meta-item">
        <span className="meta-k">KLASIFIKUAR</span>
        <span className="meta-v">{records.length}</span>
      </div>
      <div className="meta-item">
        <span className="meta-k">ALARM</span>
        <span className="meta-v t-alarm">{count("alarm")}</span>
      </div>
      <div className="meta-item">
        <span className="meta-k">SHQYRTIM</span>
        <span className="meta-v t-review">{count("review")}</span>
      </div>
      <div className="meta-item">
        <span className="meta-k">PASTRUAR</span>
        <span className="meta-v t-clear">{count("clear")}</span>
      </div>
      <div className="meta-item">
        <span className="meta-k">PA VENDIM</span>
        <span className="meta-v">{open}</span>
      </div>
    </div>
  );
}

export default SessionStats;
