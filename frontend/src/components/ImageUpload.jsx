import { useCallback, useEffect, useRef, useState } from "react";
import { MAX_BYTES } from "../config/api";

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

function ImageUpload({
  file,
  preview,
  zone,
  explanation,
  busy,
  error,
  onPick,
  onClassify,
  onReset,
}) {
  const [dragging, setDragging] = useState(false);
  const [showHeat, setShowHeat] = useState(true);
  const inputRef = useRef(null);

  const take = (candidate) => {
    if (!candidate) return;
    if (!candidate.type.startsWith("image/")) {
      onPick(null, "Zgjidh një fajll imazhi (JPEG ose PNG).");
      return;
    }
    if (candidate.size > MAX_BYTES) {
      onPick(null, "Imazhi i kalon 10 MB. Zgjidh një më të vogël.");
      return;
    }
    onPick(candidate, null);
  };

  /* Grad-CAM shpjegon një vendim pozitiv. Kur dalja është afër zeros, gradientet
     janë zhurmë dhe normalizimi do të prodhonte një njollë bindëse nga asgjëja.
     Prandaj harta shfaqet vetëm kur kuadri nuk u pastrua. */
  const meaningful =
    Boolean(explanation) &&
    explanation.peak_raw > 0 &&
    Boolean(zone) &&
    zone.id !== "clear";
  const showing = meaningful && showHeat;

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
          take(e.dataTransfer.files?.[0]);
        }}
      >
        {preview ? (
          <>
            <div className="shot">
              <img
                className="shot-img"
                src={preview}
                alt={file ? file.name : "Pamja e ngarkuar"}
              />
              {showing && <HeatCanvas explanation={explanation} />}
            </div>

            {zone && (
              <div className={`frame-tag t-${zone.id}`}>
                <i className={`z-${zone.id}`} />
                {zone.title}
              </div>
            )}

            {explanation && (
              <div className="frame-tools">
                <button
                  type="button"
                  className="chip"
                  aria-pressed={showing}
                  disabled={!meaningful}
                  onClick={() => setShowHeat((v) => !v)}
                  title={
                    meaningful
                      ? "Shfaq ose fshih zonën që ndikoi në vendim"
                      : "Nuk u gjet zjarr, prandaj nuk ka zonë për të treguar."
                  }
                >
                  Zona e vendimit
                </button>
              </div>
            )}

            {file && <div className="frame-caption">{file.name}</div>}
          </>
        ) : (
          <>
            <Contour />
            <div className="drop">
              <strong>Lësho këtu një pamje ajrore</strong>
              <span>ose zgjidhe nga kompjuteri</span>
              <button
                type="button"
                className="btn primary"
                onClick={() => inputRef.current?.click()}
              >
                Zgjidh pamjen
              </button>
              <small>JPEG ose PNG, deri në 10 MB</small>
            </div>
          </>
        )}

        <input
          ref={inputRef}
          type="file"
          accept="image/*"
          hidden
          onChange={(e) => {
            take(e.target.files?.[0]);
            e.target.value = "";
          }}
        />
      </div>

      {showing && (
        <p className="heat-note">
          Ngjyra portokalli tregon pjesën e pamjes që ndikoi më shumë në vendim.
          Nuk tregon vendin e saktë të zjarrit.
        </p>
      )}

      {explanation && !meaningful && (
        <p className="heat-note">
          Nuk u gjet zjarr, prandaj nuk ka zonë për të treguar.
        </p>
      )}

      {error && <p className="error">{error}</p>}

      {file && (
        <div className="filebar">
          <span className="filename" title={file.name}>
            {file.name} · {(file.size / 1024).toFixed(0)} KB
          </span>
          <div className="actions">
            <button type="button" className="btn" onClick={onReset} disabled={busy}>
              Hiqe
            </button>
            <button
              type="button"
              className="btn primary"
              onClick={onClassify}
              disabled={busy}
            >
              {busy ? "Duke klasifikuar…" : "Klasifiko"}
            </button>
          </div>
        </div>
      )}
    </>
  );
}

export default ImageUpload;
