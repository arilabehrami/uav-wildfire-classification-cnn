import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import ImageUpload from "./components/ImageUpload";
import DecisionBand from "./components/DecisionBand";
import ModelCard from "./components/ModelCard";
import DetectionLog from "./components/DetectionLog";
import { CLASSIFY_URL, HEALTH_URL, MAX_BYTES } from "./config/api";
import { GATES, MODEL, ZONES, zoneIdFor } from "./config/model";
import { clearRecords, loadRecords, saveRecords } from "./lib/storage";
import "./App.css";
import "./flight.css";

const TIMEOUT_MS = 30_000;
const HEALTH_EVERY_MS = 20_000;
// Radha mban referenca te fajllave plus nga nje object URL per secilin, pra
// kufizohet qellimisht: nje dosje e tere kuadrove do te mbante gjithçka ne
// memorie te shfletuesit.
const MAX_QUEUE = 50;
const MAX_MB = Math.round(MAX_BYTES / (1024 * 1024));

let counter = 0;
const nextId = () => `${Date.now().toString(36)}-${(counter += 1).toString(36)}`;

/* Portat vijne me pergjigjen e API-se. Vlerat e config-ut perdoren vetem nese
   pergjigjja nuk i permban ose i permban te paperdorshme, qe shiriti te mos
   tregoje kurre kufij te ndryshem nga ata me te cilet u mor vendimi. */
function gatesFrom(raw) {
  const low = Number(raw?.low);
  const high = Number(raw?.high);
  const usable =
    Number.isFinite(low) && Number.isFinite(high) && low >= 0 && low <= high && high <= 1;
  return usable ? { low, high } : { low: GATES.low, high: GATES.high };
}

async function classifyOne(file) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), TIMEOUT_MS);

  try {
    const body = new FormData();
    body.append("file", file);

    const response = await fetch(CLASSIFY_URL, {
      method: "POST",
      body,
      signal: controller.signal,
    });
    const data = await response.json().catch(() => null);

    if (!response.ok) {
      const detail = typeof data?.detail === "string" ? data.detail : null;
      throw new Error(detail ?? `Pamja nuk u kontrollua dot (HTTP ${response.status}).`);
    }

    const score = Number(data?.fire_score);
    if (!Number.isFinite(score)) {
      throw new Error("Përgjigjja e sistemit nuk u kuptua.");
    }

    return { ...data, fire_score: score };
  } finally {
    clearTimeout(timer);
  }
}

const isOffline = (problem) => problem instanceof TypeError;

function failureText(problem) {
  if (problem?.name === "AbortError") {
    return `Serveri nuk u përgjigj brenda ${TIMEOUT_MS / 1000} sekondash.`;
  }
  if (isOffline(problem)) {
    return "Sistemi nuk përgjigjet. Sigurohu që serveri është i ndezur.";
  }
  return problem?.message ?? "Pamja nuk u kontrollua dot.";
}

/* Kuadrokopter i pare nga lart: kater rotore dhe nje trup. Pika e ndezur ne
   qender eshte e vetmja ngjyre ne shenje, sepse zjarri eshte arsyeja e saj. */
function Drone() {
  return (
    <svg className="mark-glyph" viewBox="0 0 20 20" fill="none" aria-hidden="true">
      <path
        d="M6.4 6.4 L13.6 13.6 M13.6 6.4 L6.4 13.6"
        stroke="var(--faint)"
        strokeWidth="1.5"
        strokeLinecap="round"
      />
      <g stroke="var(--dim)" strokeWidth="1.3">
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
  /* itemsRef eshte burimi autoritar i radhes, kurse 'items' vetem pasqyra per
     render. Pa kete, dy veprime brenda te njejtit tik (shtimi i fotos dhe
     perfundimi i kerkeses) lexojne gjendje te vjetruar, dhe rezultati i njeres
     foto perfundon mbi tjetren. Rezultatet shkruhen gjithmone sipas id-se. */
  const itemsRef = useRef([]);
  const [items, setItems] = useState([]);
  const [focusId, setFocusId] = useState(null);
  const initialRecords = useMemo(() => loadRecords(), []);
  const recordsRef = useRef(initialRecords);
  const [records, setRecords] = useState(initialRecords);
  const [health, setHealth] = useState("checking");
  const [notice, setNotice] = useState(null);
  const [running, setRunning] = useState(false);
  const [persisting, setPersisting] = useState(true);
  const pumpRef = useRef(false);

  const commit = useCallback((next) => {
    itemsRef.current = next;
    setItems(next);
  }, []);

  const patch = useCallback(
    (id, changes) => {
      commit(
        itemsRef.current.map((item) =>
          item.id === id ? { ...item, ...changes } : item,
        ),
      );
    },
    [commit],
  );

  /* Regjistri ruhet ne te njejtin hap ne te cilin ndryshon, jo ne nje efekt:
     ruajtja eshte pasoje e drejtperdrejte e veprimit, nuk eshte sinkronizim
     i gjendjes me ndonje sistem te jashtem. */
  const commitRecords = useCallback((next) => {
    recordsRef.current = next;
    setRecords(next);
    setPersisting(saveRecords(next));
  }, []);

  useEffect(
    () => () => {
      itemsRef.current.forEach((item) => {
        if (item.preview) URL.revokeObjectURL(item.preview);
      });
    },
    [],
  );

  // Shendeti kontrollohet ne hapje dhe mandej periodikisht, qe nderprerja e
  // serverit te shfaqet edhe kur perdoruesi nuk klasifikon asgje.
  useEffect(() => {
    let alive = true;

    const ping = async () => {
      try {
        const response = await fetch(HEALTH_URL);
        if (!alive) return;
        const body = await response.json().catch(() => null);
        if (!response.ok) setHealth("down");
        else setHealth(body?.model_loaded === false ? "loading" : "up");
      } catch {
        if (alive) setHealth("down");
      }
    };

    ping();
    const timer = setInterval(ping, HEALTH_EVERY_MS);
    return () => {
      alive = false;
      clearInterval(timer);
    };
  }, []);

  const handlePick = useCallback(
    (files) => {
      const problems = [];
      const accepted = [];

      for (const file of files) {
        if (!file.type.startsWith("image/")) {
          problems.push(`${file.name}: nuk është imazh`);
        } else if (file.size > MAX_BYTES) {
          problems.push(`${file.name}: mbi ${MAX_MB} MB`);
        } else {
          accepted.push(file);
        }
      }

      const room = MAX_QUEUE - itemsRef.current.length;
      if (room <= 0) {
        setNotice(`Radha është plot (${MAX_QUEUE} foto). Pastro radhën për të shtuar të tjera.`);
        return;
      }

      const taken = accepted.slice(0, room);
      if (accepted.length > taken.length) {
        problems.push(`${accepted.length - taken.length} foto nuk u shtuan: radha u mbush.`);
      }

      const added = taken.map((file) => ({
        id: nextId(),
        file,
        name: file.name,
        size: file.size,
        preview: URL.createObjectURL(file),
        status: "pending",
        score: null,
        zoneId: null,
        gates: null,
        explanation: null,
        modelVersion: null,
        message: null,
      }));

      const wasEmpty = itemsRef.current.length === 0;
      commit([...itemsRef.current, ...added]);
      if (added.length && (wasEmpty || !focusId)) setFocusId(added[0].id);
      setNotice(problems.length ? problems.join(" · ") : null);
    },
    [commit, focusId],
  );

  const pump = useCallback(async () => {
    if (pumpRef.current) return;
    pumpRef.current = true;
    setRunning(true);

    try {
      for (;;) {
        const next = itemsRef.current.find((item) => item.status === "pending");
        if (!next) break;

        patch(next.id, { status: "running", message: null });
        setFocusId(next.id);

        try {
          const data = await classifyOne(next.file);
          const gates = gatesFrom(data.gates);
          const zoneId = ZONES[data.zone]
            ? data.zone
            : zoneIdFor(data.fire_score, gates);
          const modelVersion = data.model_version ?? MODEL.run;

          patch(next.id, {
            status: "done",
            score: data.fire_score,
            zoneId,
            gates,
            explanation: data.explanation ?? null,
            modelVersion,
          });

          commitRecords([
            {
              id: next.id,
              name: next.name,
              at: new Date().toISOString(),
              score: data.fire_score,
              zoneId,
              verdict: null,
              modelVersion,
            },
            ...recordsRef.current,
          ]);
          setHealth("up");
        } catch (problem) {
          patch(next.id, { status: "error", message: failureText(problem) });

          if (isOffline(problem)) {
            // Nuk ka kuptim t'i dergohen serverit te shuar edhe 49 kerkesa.
            setHealth("down");
            setNotice(
              "Klasifikimi u ndal: serveri nuk përgjigjet. Fotot e mbetura janë ende në radhë.",
            );
            break;
          }
        }
      }
    } finally {
      pumpRef.current = false;
      setRunning(false);
    }
  }, [patch, commitRecords]);

  const handleRemove = useCallback(
    (id) => {
      const target = itemsRef.current.find((item) => item.id === id);
      if (!target || target.status === "running") return;

      if (target.preview) URL.revokeObjectURL(target.preview);
      const next = itemsRef.current.filter((item) => item.id !== id);
      commit(next);
      if (focusId === id) setFocusId(next.length ? next[next.length - 1].id : null);
    },
    [commit, focusId],
  );

  const handleClearQueue = useCallback(() => {
    if (pumpRef.current) return;
    itemsRef.current.forEach((item) => {
      if (item.preview) URL.revokeObjectURL(item.preview);
    });
    commit([]);
    setFocusId(null);
    setNotice(null);
  }, [commit]);

  const handleVerdict = useCallback(
    (id, verdict) => {
      commitRecords(
        recordsRef.current.map((record) =>
          record.id === id
            ? { ...record, verdict: record.verdict === verdict ? null : verdict }
            : record,
        ),
      );
    },
    [commitRecords],
  );

  const handleClearHistory = useCallback(() => {
    clearRecords();
    recordsRef.current = [];
    setRecords([]);
    setPersisting(true);
  }, []);

  const focused = useMemo(
    () => items.find((item) => item.id === focusId) ?? null,
    [items, focusId],
  );
  const focusedRecord = useMemo(
    () =>
      focused?.status === "done"
        ? (records.find((record) => record.id === focused.id) ?? null)
        : null,
    [focused, records],
  );
  const pending = items.filter((item) => item.status === "pending").length;

  const healthText =
    health === "up"
      ? "Sistemi aktiv"
      : health === "loading"
        ? "Modeli po ngarkohet"
        : health === "down"
          ? "Serveri nuk përgjigjet"
          : "Duke u lidhur";

  return (
    <div className="shell">
      <header className="topbar">
        <div className="mark">
          <Drone />
          <div className="mark-text">
            <span className="mark-name">AERO<span className="brand-accent">FIRE</span></span>
            <span className="mark-sub">Monitorim ajror · Detektim zjarri</span>
          </div>
        </div>

        <span className={`status ${health}`} role="status">
          <i aria-hidden="true" />
          {healthText}
        </span>
      </header>

      <section className="flight-intro" aria-label="Monitorimi nga droni">
        <div className="intro-copy">
          <p className="eyebrow"><span /> PAMJE NGA LART. VENDIME MË TË QARTA.</p>
          <h1>Një kënd tjetër.<br /><span>Një vështrim më i sigurt.</span></h1>
          <p className="intro-description">Analizo pamjet nga droni, identifiko flakën e dukshme dhe shqyrto çdo sinjal në një hapësirë të vetme.</p>
          <div className="intro-tags"><span>01 / Ngarko pamjet</span><span>02 / Analizo</span><span>03 / Shqyrto</span></div>
        </div>
        <div className="drone-art" aria-hidden="true">
          <svg viewBox="0 0 460 280" fill="none">
            <g className="orbit-lines" stroke="currentColor"><ellipse cx="230" cy="148" rx="200" ry="100" /><ellipse cx="230" cy="148" rx="152" ry="76" /><ellipse cx="230" cy="148" rx="106" ry="53" /><path d="M22 148h416M230 26v244" strokeDasharray="3 8" /></g>
            <g className="aircraft" transform="translate(230 132) rotate(-18)">
              <path d="M-26-15L-94-59M26-15L94-59M-26 15L-94 59M26 15L94 59" stroke="#566a79" strokeWidth="17" strokeLinecap="round" />
              <path d="M-26-15L-94-59M26-15L94-59M-26 15L-94 59M26 15L94 59" stroke="#bac5cc" strokeWidth="8" strokeLinecap="round" />
              {[-1, 1].flatMap(x => [-1, 1].map(y => <g key={`${x}-${y}`} transform={`translate(${x * 100} ${y * 63})`}><circle r="39" fill="#15202a" stroke="#53616e" strokeWidth="2" /><circle r="31" stroke="#91a0ac" strokeOpacity=".35" strokeDasharray="4 4" /><path d="M-31-7Q-13-18 0 0Q13 18 31 7Q13-5 0 0Q-13 5-31-7" fill="#9faeb8" /><circle r="7" fill="#25323e" stroke="#c8d3da" /></g>))}
              <path d="M-29-28Q0-42 29-28L34 21Q0 42-34 21Z" fill="#c2cdd4" stroke="#e0e8ed" strokeWidth="2" />
              <path d="M-20-20Q0-28 20-20L22 15Q0 27-22 15Z" fill="#26343f" /><path d="M-8-18h16" stroke="#ff8b56" strokeWidth="3" strokeLinecap="round" /><rect x="-11" y="27" width="22" height="18" rx="5" fill="#111a23" stroke="#768897" /><circle cy="36" r="5" fill="#6bd2bc" />
            </g>
            <path d="M293 207h53l19 18h44" stroke="#ff8b56" strokeOpacity=".6" /><circle cx="293" cy="207" r="3" fill="#ff8b56" />
          </svg>
          <span className="art-label">UAV / VIZION AJROR</span>
        </div>
      </section>
      <div className="workspace-heading"><div><span className="eyebrow">HAPËSIRA E ANALIZËS</span><h2>Çdo pamje ka rëndësi.</h2></div><span className="workspace-mode"><i /> Analizë e fotografive</span></div>
      <div className="main">
        <main className="work">
          <div className="viewer-heading"><div><span className="section-index">01</span><h2>Pamjet ajrore</h2></div><span>JPEG / PNG · deri {MAX_MB} MB</span></div>
          <ImageUpload
            items={items}
            focused={focused}
            pending={pending}
            running={running}
            notice={notice}
            maxMb={MAX_MB}
            onPick={handlePick}
            onFocus={setFocusId}
            onRemove={handleRemove}
            onClassify={pump}
            onClearQueue={handleClearQueue}
          />
        </main>

        <aside className="side">
          <DecisionBand
            status={focused?.status ?? null}
            score={focused?.status === "done" ? focused.score : null}
            zone={focused?.status === "done" ? ZONES[focused.zoneId] : null}
            gates={focused?.gates ?? GATES}
            verdict={focusedRecord?.verdict ?? null}
            onVerdict={
              focusedRecord ? (value) => handleVerdict(focusedRecord.id, value) : null
            }
          />
          <DetectionLog
            records={records}
            items={items}
            persisting={persisting}
            onVerdict={handleVerdict}
            onClearHistory={handleClearHistory}
          />
          <ModelCard />
        </aside>
      </div>
      <footer className="flight-footer"><span>AEROFIRE <span>/</span> Pamje ajrore. Analizë e kujdesshme.</span><span>Prototip akademik · Vendimi përfundimtar mbetet te njeriu.</span></footer>
    </div>
  );
}

export default App;
