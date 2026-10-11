import { useCallback, useEffect, useRef, useState } from "react";
import ImageUpload from "./components/ImageUpload";
import DecisionBand from "./components/DecisionBand";
import ModelCard from "./components/ModelCard";
import DetectionLog from "./components/DetectionLog";
import SessionStats from "./components/SessionStats";
import { CLASSIFY_URL, HEALTH_URL } from "./config/api";
import { MODEL, GATES, ZONES, zoneFor } from "./config/model";
import "./App.css";

const clock = () =>
  new Date().toLocaleTimeString("sq", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });

/* Kuadrokopter i pare nga lart: kater rotore dhe nje trup. Pika e ndezur ne
   qender eshte e vetmja ngjyre ne shenje, sepse zjarri eshte arsyeja e saj. */
function Drone() {
  return (
    <svg className="mark-glyph" viewBox="0 0 20 20" fill="none" aria-hidden="true">
      <path
        d="M6.4 6.4 L13.6 13.6 M13.6 6.4 L6.4 13.6"
        stroke="var(--text-faint)"
        strokeWidth="1.5"
        strokeLinecap="round"
      />
      <g stroke="var(--text-dim)" strokeWidth="1.3">
        <circle cx="4.7" cy="4.7" r="2.7" />
        <circle cx="15.3" cy="4.7" r="2.7" />
        <circle cx="4.7" cy="15.3" r="2.7" />
        <circle cx="15.3" cy="15.3" r="2.7" />
      </g>
      <rect x="7.7" y="7.7" width="4.6" height="4.6" rx="1.4" fill="var(--alarm)" />
    </svg>
  );
}

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);
  const [entries, setEntries] = useState([]);
  const [health, setHealth] = useState("checking");
  const previewRef = useRef(null);
  const thumbsRef = useRef([]);

  useEffect(() => {
    let alive = true;
    fetch(HEALTH_URL)
      .then((response) => alive && setHealth(response.ok ? "up" : "down"))
      .catch(() => alive && setHealth("down"));
    return () => {
      alive = false;
    };
  }, []);

  useEffect(
    () => () => {
      if (previewRef.current) URL.revokeObjectURL(previewRef.current);
      thumbsRef.current.forEach((url) => URL.revokeObjectURL(url));
      thumbsRef.current = [];
    },
    [],
  );

  const swapPreview = useCallback((next) => {
    if (previewRef.current) URL.revokeObjectURL(previewRef.current);
    previewRef.current = next ? URL.createObjectURL(next) : null;
    setPreview(previewRef.current);
  }, []);

  const handlePick = useCallback(
    (next, message) => {
      setError(message ?? null);
      if (!next) return;
      setFile(next);
      swapPreview(next);
      setResult(null);
    },
    [swapPreview],
  );

  const handleReset = useCallback(() => {
    setFile(null);
    swapPreview(null);
    setResult(null);
    setError(null);
  }, [swapPreview]);

  const handleClassify = useCallback(async () => {
    if (!file) return;
    setBusy(true);
    setError(null);

    const body = new FormData();
    body.append("file", file);

    try {
      const response = await fetch(CLASSIFY_URL, { method: "POST", body });
      const data = await response.json().catch(() => null);

      if (!response.ok) {
        throw new Error(data?.detail ?? `API ktheu ${response.status}.`);
      }

      const score = Number(data?.fire_score);
      if (!Number.isFinite(score)) {
        throw new Error("Përgjigjja nuk përmban një fire_score të vlefshëm.");
      }

      const zone = ZONES[data?.zone] ?? zoneFor(score);
      setResult({
        score,
        zone,
        modelVersion: data?.model_version ?? MODEL.run,
        explanation: data?.explanation ?? null,
      });

      const thumb = URL.createObjectURL(file);
      thumbsRef.current.push(thumb);

      setEntries((previous) => [
        { id: `${Date.now()}`, name: file.name, score, zone, at: clock(), thumb },
        ...previous,
      ]);
      setHealth("up");
    } catch (problem) {
      const offline = problem instanceof TypeError;
      setError(
        offline
          ? "API-ja nuk u arrit. Nise backend-in me uvicorn në portin 8000."
          : problem.message,
      );
      if (offline) setHealth("down");
    } finally {
      setBusy(false);
    }
  }, [file]);

  return (
    <div className="shell">
      <header className="topbar">
        <div className="mark">
          <Drone />
          <span className="mark-name">Detektim zjarri</span>
          <span className="mark-sub">pamje ajrore</span>
        </div>

        <div className="meta">
          <div className="meta-item">
            <span className="meta-k">MODELI</span>
            <span className="meta-v">{MODEL.run}</span>
          </div>
          <div className="meta-item">
            <span className="meta-k">PORTA E POSHTME</span>
            <span className="meta-v">{GATES.low.toFixed(4)}</span>
          </div>
          <div className="meta-item">
            <span className="meta-k">PORTA E SIPËRME</span>
            <span className="meta-v">{GATES.high.toFixed(4)}</span>
          </div>
        </div>

        <SessionStats entries={entries} />

        <div className="topbar-end">
          <span className={`status ${health}`}>
            <i />
            {health === "up"
              ? "API aktive"
              : health === "down"
                ? "API joaktive"
                : "duke kontrolluar"}
          </span>
        </div>
      </header>

      <div className="main">
        <main className="work">
          <ImageUpload
            file={file}
            preview={preview}
            zone={result?.zone ?? null}
            explanation={result?.explanation ?? null}
            busy={busy}
            error={error}
            onPick={handlePick}
            onClassify={handleClassify}
            onReset={handleReset}
          />
          <DecisionBand
            score={result?.score ?? null}
            zone={result?.zone ?? null}
            modelVersion={result?.modelVersion ?? null}
          />
        </main>

        <aside className="side">
          <ModelCard />
          <DetectionLog entries={entries} />
        </aside>
      </div>
    </div>
  );
}

export default App;
