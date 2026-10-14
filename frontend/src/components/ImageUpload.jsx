import { useCallback, useEffect, useRef, useState } from "react";
import { ZONES } from "../config/model";

// Shenjestra e kameres: shfaqet vetem kur nuk ka pamje, qe zona bosh te
// lexohet si objektiv i dronit, jo si kuti e zakonshme ngarkimi.
function Reticle() {
  return (
    <svg className="reticle" viewBox="0 0 200 200" fill="none" aria-hidden="true">
      <circle cx="100" cy="100" r="62" stroke="var(--line)" />
      <circle cx="100" cy="100" r="30" stroke="var(--line)" strokeDasharray="2 5" />
      <path
        d="M100 18v26M100 156v26M18 100h26M156 100h26"
        stroke="var(--faint)"
        strokeLinecap="round"
      />
      <circle cx="100" cy="100" r="2.5" fill="var(--faint)" />
    </svg>
  );
}

/* Rampa e intensitetit: qelibar -> ember, me opacitet që ndjek vlerën.
   Zonat e ftohta e lënë foton krejt të paprekur. */
function ramp(v) {
  const stops = [
    [0.0, 90, 70, 30],
    [0.5, 239, 196, 90],
    [1.0, 255, 106, 43],
  ];
  let i = 1;
  while (i < stops.length - 1 && v > stops[i][0]) i += 1;
  const [a0, ar, ag, ab] = stops[i - 1];
  const [a1, br, bg, bb] = stops[i];
  const t = a1 === a0 ? 0 : (v - a0) / (a1 - a0);
  return [
    Math.round(ar + (br - ar) * t),
    Math.round(ag + (bg - ag) * t),
    Math.round(ab + (bb - ab) * t),
  ];
}

function HeatCanvas({ explanation }) {
  const canvasRef = useRef(null);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas || !explanation?.grid?.length) return;

    const { grid, rows, cols } = explanation;

    // Një piksel për qelizë, pastaj zmadhim me interpolim.
    const small = document.createElement("canvas");
    small.width = cols;
    small.height = rows;
    const sctx = small.getContext("2d");
    const data = sctx.createImageData(cols, rows);

    for (let r = 0; r < rows; r += 1) {
      for (let c = 0; c < cols; c += 1) {
        const v = Math.min(1, Math.max(0, grid[r]?.[c] ?? 0));
        const [red, green, blue] = ramp(v);
        const idx = (r * cols + c) * 4;
        data.data[idx] = red;
        data.data[idx + 1] = green;
        data.data[idx + 2] = blue;
        data.data[idx + 3] = Math.round(v * v * 210);
      }
    }
    sctx.putImageData(data, 0, 0);

    const box = canvas.getBoundingClientRect();
    if (box.width < 1 || box.height < 1) return;
    canvas.width = Math.round(box.width);
    canvas.height = Math.round(box.height);

    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(small, 0, 0, canvas.width, canvas.height);
  }, [explanation]);

  useEffect(() => {
    draw();
    const canvas = canvasRef.current;
    if (!canvas || typeof ResizeObserver === "undefined") return undefined;
    const observer = new ResizeObserver(draw);
    observer.observe(canvas);
    return () => observer.disconnect();
  }, [draw]);

  return <canvas className="shot-heat" ref={canvasRef} aria-hidden="true" />;
}

const STATUS_TEXT = {
  pending: "pa klasifikuar",
  running: "duke u analizuar",
  error: "gabim",
};

function QueueStrip({ items, focusedId, running, onFocus, onRemove }) {
  if (items.length < 2) return null;

  return (
    <div className="strip" role="list" aria-label="Radha e pamjeve">
      {items.map((item, index) => {
        const zone = item.status === "done" ? ZONES[item.zoneId] : null;
        const state = zone ? zone.title : STATUS_TEXT[item.status];
        return (
          <div
            role="listitem"
            key={item.id}
            className={`thumb${item.id === focusedId ? " on" : ""} s-${item.status}${zone ? ` z-${zone.id}` : ""}`}
          >
            <button
              type="button"
              className="thumb-open"
              onClick={() => onFocus(item.id)}
              title={item.message ?? item.name}
              aria-label={`Pamja ${index + 1}, ${item.name}: ${state}`}
              aria-current={item.id === focusedId ? "true" : undefined}
            >
              <img src={item.preview} alt="" />
              <span className="thumb-mark" aria-hidden="true" />
            </button>
            <button
              type="button"
              className="thumb-drop"
              onClick={() => onRemove(item.id)}
              disabled={running && item.status === "running"}
              aria-label={`Hiq ${item.name} nga radha`}
              title="Hiqe nga radha"
            >
              <svg viewBox="0 0 10 10" aria-hidden="true">
                <path d="M2.5 2.5l5 5M7.5 2.5l-5 5" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" />
              </svg>
            </button>
          </div>
        );
      })}
    </div>
  );
}

function ImageUpload({
  items,
  focused,
  pending,
  running,
  notice,
  maxMb,
  onPick,
  onFocus,
  onRemove,
  onClassify,
  onClearQueue,
}) {
  const [dragging, setDragging] = useState(false);
  const [showHeat, setShowHeat] = useState(true);
  const inputRef = useRef(null);

  const take = (fileList) => {
    const files = Array.from(fileList ?? []);
    if (files.length) onPick(files);
  };

  const pick = () => inputRef.current?.click();

  const done = focused?.status === "done";
  const zone = done ? ZONES[focused.zoneId] : null;
  const explanation = focused?.explanation ?? null;
  const position = focused ? items.findIndex((item) => item.id === focused.id) + 1 : 0;

  /* Grad-CAM shpjegon një vendim pozitiv. Kur dalja është afër zeros, gradientet
     janë zhurmë dhe normalizimi do të prodhonte një njollë bindëse nga asgjëja.
     Prandaj harta shfaqet vetëm kur kuadri nuk u pastrua.

     Mungesa e hartës nuk thotë asgjë për zjarrin: harta dhe klasifikimi janë
     dy rezultate të ndryshme, dhe mesazhet më poshtë nuk i ngatërrojnë. */
  const hasMap = Boolean(explanation) && explanation.peak_raw > 0;
  const positive = Boolean(zone) && zone.id !== "clear";
  const canShow = hasMap && positive;
  const showing = canShow && showHeat;

  const heatTitle = positive
    ? hasMap
      ? "Shfaq ose fshih zonën që ndikoi në vendim"
      : "Harta e aktivizimit nuk u llogarit dot për këtë pamje."
    : "Harta shfaqet vetëm për vendimet pozitive.";

  const finderClass = [
    "finder",
    focused ? "has-shot" : "is-empty",
    zone ? `z-${zone.id}` : "",
    focused?.status === "running" ? "is-running" : "",
    dragging ? "is-dragging" : "",
  ]
    .filter(Boolean)
    .join(" ");

  return (
    <section className="viewer" aria-label="Pamja nën analizë">
      <div
        className={finderClass}
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          take(e.dataTransfer.files);
        }}
      >
        <span className="corner tl" aria-hidden="true" />
        <span className="corner tr" aria-hidden="true" />
        <span className="corner bl" aria-hidden="true" />
        <span className="corner br" aria-hidden="true" />

        {focused ? (
          <>
            <div className="shot">
              <img className="shot-img" src={focused.preview} alt={focused.name} />
              {showing && <HeatCanvas explanation={explanation} />}
              {focused.status === "running" && <div className="scan" aria-hidden="true" />}
            </div>

            <div className="hud hud-top">
              <span className="hud-label">
                Pamja {position} nga {items.length}
              </span>
              {done && (
                <button
                  type="button"
                  className="toggle"
                  aria-pressed={showing}
                  disabled={!canShow}
                  onClick={() => setShowHeat((v) => !v)}
                  title={heatTitle}
                >
                  <span className="toggle-knob" aria-hidden="true" />
                  Zona e vendimit
                </button>
              )}
            </div>

            <div className="hud hud-bottom">
              <span className="hud-label hud-name" title={focused.name}>
                {focused.name}
              </span>
              {zone && (
                <span className={`hud-zone t-${zone.id}`}>
                  <i aria-hidden="true" />
                  {zone.title}
                </span>
              )}
            </div>
          </>
        ) : (
          <div className="drop">
            <Reticle />
            <h3 className="drop-title">
              {dragging ? "Lëshoji pamjet" : "Lësho këtu pamjet nga droni"}
            </h3>
            <p className="drop-hint">
              Zgjidh ose tërhiq fotografitë këtu. Deri {maxMb} MB për foto.
            </p>
            <button type="button" className="btn primary" onClick={pick}>
              Zgjidh pamjet <span aria-hidden="true">↗</span>
            </button>
          </div>
        )}

        <input
          ref={inputRef}
          type="file"
          accept="image/*"
          multiple
          hidden
          onChange={(e) => {
            take(e.target.files);
            e.target.value = "";
          }}
        />
      </div>

      {items.length > 0 && (
        <div className="deck">
          <QueueStrip
            items={items}
            focusedId={focused?.id ?? null}
            running={running}
            onFocus={onFocus}
            onRemove={onRemove}
          />

          <div className="deck-row">
            <p className="deck-count">
              <b>{items.length}</b> pamje në radhë
              <span>
                {running
                  ? `${pending} në pritje`
                  : pending > 0
                    ? `${pending} pa klasifikuar`
                    : "të gjitha u klasifikuan"}
              </span>
            </p>
            <div className="actions">
              <button type="button" className="btn ghost" onClick={pick} disabled={running}>
                Shto pamje
              </button>
              <button type="button" className="btn ghost" onClick={onClearQueue} disabled={running}>
                Pastro radhën
              </button>
              <button
                type="button"
                className={running ? "btn primary is-busy" : "btn primary"}
                onClick={onClassify}
                disabled={running || pending === 0}
              >
                {running
                  ? "Duke klasifikuar…"
                  : pending > 1
                    ? `Klasifiko ${pending} pamje`
                    : "Klasifiko"}
              </button>
            </div>
          </div>
        </div>
      )}

      <div className="notes" aria-live="polite">
        {focused?.status === "error" && <p className="note error">{focused.message}</p>}
        {notice && <p className="note warn">{notice}</p>}

        {showing && (
          <p className="note">
            Portokallia tregon pjesën e pamjes që ndikoi më shumë në vendim, jo
            vendin e saktë të zjarrit.
          </p>
        )}
        {done && !positive && (
          <p className="note">
            Harta e aktivizimit shpjegon vetëm vendimet pozitive, prandaj nuk
            shfaqet për një pamje të pastër.
          </p>
        )}
        {done && positive && !hasMap && (
          <p className="note">
            Harta e aktivizimit nuk u llogarit dot për këtë pamje. Klasifikimi
            mbetet i vlefshëm.
          </p>
        )}
      </div>
    </section>
  );
}

export default ImageUpload;
