function SessionStats({ entries }) {
  const count = (id) => entries.filter((entry) => entry.zone.id === id).length;

  return (
    <div className="stats">
      <div className="meta-item">
        <span className="meta-k">KLASIFIKUAR</span>
        <span className="meta-v">{entries.length}</span>
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
    </div>
  );
}

export default SessionStats;
