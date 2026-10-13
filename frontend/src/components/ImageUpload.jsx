import { useCallback, useEffect, useRef, useState } from "react";
import { ZONES } from "../config/model";

// Unaza terreni: një aluzion i asaj që shikon kamera, i mbajtur në kontrast
// shumë të ulët që të mos konkurrojë me pamjen nën analizë.
function Contour() {
  const rings = [0, 1, 2, 3, 4, 5];
  return (
    <svg
      className="stage-contour"
      viewBox="0 0 600 400"
      preserveAspectRatio="xMidYMid slice"
      aria-hidden="true"
    >
      <g fill="none" stroke="var(--line)" strokeWidth="1">
        {rings.map((i) => (
          <ellipse
            key={i}
            cx="300"
            cy="200"
            rx={70 + i * 52}
            ry={44 + i * 33}
            transform={`rotate(${-12 + i * 2} 300 200)`}
          />
        ))}
      </g>
    </svg>
  );
}

/* Rampa e intensitetit: qelibar -> ember, me opacitet që ndjek vlerën.
   Zonat e ftohta e lënë foton krejt të paprekur. */
function ramp(v) {
  const stops = [
    [0.0, 90, 70, 30],
    [0.5, 228, 176, 74],
    [1.0, 255, 106, 31],
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

function QueueStrip({ items, focusedId, running, onFocus, onRemove }) {
  if (items.length < 2) return null;

  return (
    <div className="queue" role="list" aria-label="Radha e pamjeve">
      {items.map((item, index) => {
        const zone = item.status === "done" ? ZONES[item.zoneId] : null;
        return (
          <div
            role="listitem"
            key={item.id}
            className={`q-item${item.id === focusedId ? " on" : ""} s-${item.status}`}
          >
            <button
              type="button"
              className="q-open"
              onClick={() => onFocus(item.id)}
              title={item.message ?? item.name}
            >
              {item.preview ? (
                <img src={item.preview} alt="" className="q-thumb" />
              ) : (
                <span className="q-thumb" />
              )}
              <span className="q-index">{index + 1}</span>
              <span className={`q-dot${zone ? ` z-${zone.id}` : ""}`} />
            </button>
            <button
              type="button"
              className="q-drop"
              onClick={() => onRemove(item.id)}
              disabled={running && item.status === "running"}
              aria-label={`Hiq ${item.name}`}
              title="Hiqe nga radha"
            >
              ×
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

  const done = focused?.status === "done";
  const zone = done ? ZONES[focused.zoneId] : null;
  const explanation = focused?.explanation ?? null;

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

  return (
    <>
      <div
        className={dragging ? "stage dragging" : "stage"}
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
        {focused ? (
          <>
            <div className="shot">
              <img
                className="shot-img"
                src={focused.preview}
                alt={focused.name}
              />
              {showing && <HeatCanvas explanation={explanation} />}
              {focused.status === "running" && <div className="shot-busy" />}
            </div>

            {zone && (
              <div className={`frame-tag t-${zone.id}`}>
                <i className={`z-${zone.id}`} />
                {zone.title}
              </div>
            )}

            {done && (
              <div className="frame-tools">
                <button
                  type="button"
                  className="chip"
                  aria-pressed={showing}
                  disabled={!canShow}
                  onClick={() => setShowHeat((v) => !v)}
                  title={heatTitle}
                >
                  Zona e vendimit
                </button>
              </div>
            )}

            <div className="frame-caption">{focused.name}</div>
          </>
        ) : (
          <>
            <Contour />
            <div className="drop">
              <strong>Lësho këtu pamje ajrore</strong>
              <span>një ose shumë, ose zgjidhi nga kompjuteri</span>
              <button
                type="button"
                className="btn primary"
                onClick={() => inputRef.current?.click()}
              >
                Zgjidh pamjet
              </button>
              <small>JPEG ose PNG, deri në {maxMb} MB secila</small>
            </div>
          </>
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

      <QueueStrip
        items={items}
        focusedId={focused?.id ?? null}
        running={running}
        onFocus={onFocus}
        onRemove={onRemove}
      />

      {showing && (
        <p className="heat-note">
          Ngjyra portokalli tregon pjesën e pamjes që ndikoi më shumë në vendim.
          Nuk tregon vendin e saktë të zjarrit.
        </p>
      )}

      {done && !positive && (
        <p className="heat-note">
          Harta e aktivizimit shpjegon vetëm vendimet pozitive, prandaj nuk
          shfaqet për një pamje të pastër.
        </p>
      )}

      {done && positive && !hasMap && (
        <p className="heat-note">
          Harta e aktivizimit nuk u llogarit dot për këtë pamje. Klasifikimi
          mbetet i vlefshëm.
        </p>
      )}

      {focused?.status === "error" && <p className="error">{focused.message}</p>}
      {notice && <p className="notice">{notice}</p>}

      {items.length > 0 && (
        <div className="filebar">
          <span className="filename">
            {items.length} në radhë
            {pending > 0 ? ` · ${pending} pa klasifikuar` : " · të gjitha të kryera"}
          </span>
          <div className="actions">
            <button
              type="button"
              className="btn"
              onClick={() => inputRef.current?.click()}
              disabled={running}
            >
              Shto
            </button>
            <button
              type="button"
              className="btn"
              onClick={onClearQueue}
              disabled={running}
            >
              Pastro radhën
            </button>
            <button
              type="button"
              className="btn primary"
              onClick={onClassify}
              disabled={running || pending === 0}
            >
              {running
                ? "Duke klasifikuar…"
                : pending > 1
                  ? `Klasifiko ${pending}`
                  : "Klasifiko"}
            </button>
          </div>
        </div>
      )}
    </>
  );
}

export default ImageUpload;
