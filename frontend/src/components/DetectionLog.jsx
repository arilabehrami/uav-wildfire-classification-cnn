function DetectionLog({ entries }) {
  return (
    <section className="panel grow">
      <h3>Kjo seancë{entries.length > 0 ? ` · ${entries.length}` : ""}</h3>

      {entries.length === 0 ? (
        <p className="empty">
          Asgjë ende. Çdo kuadro i klasifikuar shfaqet këtu, me zonën dhe
          rezultatin e tij.
        </p>
      ) : (
        <div className="log">
          {entries.map((entry) => (
            <div className="log-row" key={entry.id}>
              {entry.thumb ? (
                <img className="log-thumb" src={entry.thumb} alt="" />
              ) : (
                <i className={`z-${entry.zone.id}`} />
              )}
              <span className="log-main">
                <span className="name" title={entry.name}>
                  {entry.name}
                </span>
                <span className="at">{entry.at}</span>
              </span>
              <span className={`score t-${entry.zone.id}`}>
                {entry.score.toFixed(4)}
              </span>
            </div>
          ))}
        </div>
      )}
    </section>
  );
}

export default DetectionLog;
