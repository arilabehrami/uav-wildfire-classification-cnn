import { useState } from "react";

function ImageUpload() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);

  const handleUpload = async () => {
    if (!file) return;

    const formData = new FormData();
    formData.append("file", file);

    const response = await fetch("http://localhost:8000/api/classify", {
      method: "POST",
      body: formData,
    });

    const data = await response.json();
    setResult(data);
  };

  return (
    <div>
      <h2>Wildfire Detection</h2>
      <p>Modeli aktual mund ta klasifikojë tymin si Fire. Modeli për vetëm flakë të dukshme ende nuk është aktiv.</p>

      <input
        type="file"
        accept="image/*"
        onChange={(e) => setFile(e.target.files[0])}
      />

      <button onClick={handleUpload}>
        Detect Fire
      </button>

      {result && (
        <div>
          <p>Result: {result.label}</p>
          <p>Fire score: {result.fire_score?.toFixed(4)}</p>
        </div>
      )}
    </div>
  );
}

export default ImageUpload;
