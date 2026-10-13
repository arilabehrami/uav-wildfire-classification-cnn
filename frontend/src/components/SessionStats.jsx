// Numrat vijne nga regjistri i ruajtur, pra mbijetojne rifreskimin e faqes.
// "Pa vendim" eshte numri i rasteve ne zonen e shqyrtimit per te cilat nuk
// ka vendosur ende njeri; ky numer, jo totali, tregon sa pune ka mbetur.
function SessionStats({ records }) {
  const count = (id) => records.filter((record) => record.zoneId === id).length;
  const open = records.filter(
    (record) => record.zoneId === "review" && !record.verdict,
  ).length;

  const tallies = [
    { id: "alarm", label: "Alarm", value: count("alarm") },
    { id: "review", label: "Shqyrtim", value: count("review") },
    { id: "clear", label: "E pastër", value: count("clear") },
  ];

  return (
    <dl className="tally">
      {tallies.map((tally) => (
        <div key={tally.id} className={`tally-item z-${tally.id}`}>
          <dt>{tally.label}</dt>
          <dd>{tally.value}</dd>
        </div>
      ))}
      <div className={open > 0 ? "tally-item open has" : "tally-item open"}>
        <dt>Pa vendim</dt>
        <dd>{open}</dd>
      </div>
    </dl>
  );
}

export default SessionStats;
