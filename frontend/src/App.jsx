import { useState, useRef, useEffect } from "react";
import axios from "axios";
import "./App.css";

const API = "http://127.0.0.1:8000";

// ── Botox ─────────────────────────────────────────────────────────────────────
const BOTOX_KEYS = new Set([
  "forehead", "frown", "crows_left", "crows_right",
  "under_left", "under_right", "nasolabial_left", "nasolabial_right",
  "lip_lines", "marionette_left", "marionette_right", "chin",
]);

const BOTOX_COLORS = {
  forehead:          "rgba(255, 200, 50,  0.35)",
  frown:             "rgba(255, 100, 100, 0.35)",
  crows_left:        "rgba(100, 200, 255, 0.35)",
  crows_right:       "rgba(100, 200, 255, 0.35)",
  under_left:        "rgba(150, 100, 255, 0.35)",
  under_right:       "rgba(150, 100, 255, 0.35)",
  nasolabial_left:   "rgba(100, 255, 150, 0.35)",
  nasolabial_right:  "rgba(100, 255, 150, 0.35)",
  lip_lines:         "rgba(255, 150, 200, 0.35)",
  marionette_left:   "rgba(255, 220, 100, 0.35)",
  marionette_right:  "rgba(255, 220, 100, 0.35)",
  chin:              "rgba(255, 180, 80,  0.35)",
};

const BOTOX_INTENSITIES = {
  forehead:          "forehead_intensity",
  frown:             "frown_intensity",
  crows_left:        "crows_intensity",
  crows_right:       "crows_intensity",
  under_left:        "under_eye_intensity",
  under_right:       "under_eye_intensity",
  nasolabial_left:   "nasolabial_fold_intensity",
  nasolabial_right:  "nasolabial_fold_intensity",
  lip_lines:         "lip_intensity",
  marionette_left:   "marionette_intensity",
  marionette_right:  "marionette_intensity",
  chin:              "chin_intensity",
};

const BOTOX_SHRINK = {
  forehead:          { x: 0.75, y: 0.70 },
  frown:             { x: 0.80, y: 0.75 },
  crows_left:        { x: 0.90, y: 0.60 },
  crows_right:       { x: 0.90, y: 0.60 },
  under_left:        { x: 0.70, y: 0.55 },
  under_right:       { x: 0.70, y: 0.55 },
  nasolabial_left:   { x: 0.42, y: 0.58 },
  nasolabial_right:  { x: 0.42, y: 0.58 },
  lip_lines:         { x: 0.88, y: 0.68 },
  marionette_left:   { x: 0.48, y: 0.60 },
  marionette_right:  { x: 0.48, y: 0.60 },
  chin:              { x: 0.58, y: 0.50 },
};

const BOTOX_LABEL_Y_FRAC = {
  forehead:          0.22,
  frown:             0.32,
  crows_left:        0.38,
  crows_right:       0.38,
  under_left:        0.46,
  under_right:       0.46,
  nasolabial_left:   0.52,
  nasolabial_right:  0.52,
  lip_lines:         0.60,
  marionette_left:   0.68,
  marionette_right:  0.68,
  chin:              0.76,
};

const BOTOX_PARAM_LABELS = {
  forehead_intensity:        "Forehead",
  frown_intensity:           "Frown Lines",
  crows_intensity:           "Crow's Feet",
  under_eye_intensity:       "Under Eye",
  nasolabial_fold_intensity: "Nasolabial Folds",
  lip_intensity:             "Lip Lines",
  marionette_intensity:      "Marionette Lines",
  chin_intensity:            "Chin",
};

// ── Filler ────────────────────────────────────────────────────────────────────
const FILLER_KEYS = new Set([
  "cheek_left", "cheek_right", "jawline_left", "jawline_right", "chin_filler", "lip_filler",
]);

const FILLER_COLORS = {
  cheek_left:    "rgba(100, 210, 255, 0.35)",
  cheek_right:   "rgba(100, 210, 255, 0.35)",
  jawline_left:  "rgba(180, 120, 255, 0.35)",
  jawline_right: "rgba(180, 120, 255, 0.35)",
  chin_filler:   "rgba(255, 180, 80,  0.35)",
  lip_filler:    "rgba(255, 120, 180, 0.35)",
};

// lip_filler intentionally absent — it has its own dedicated controls
const FILLER_INTENSITIES = {
  cheek_left:    "cheek_intensity",
  cheek_right:   "cheek_intensity",
  jawline_left:  "jawline_intensity",
  jawline_right: "jawline_intensity",
  chin_filler:   "chin_intensity",
};

const FILLER_SHRINK = {
  cheek_left:    { x: 0.72, y: 0.68 },
  cheek_right:   { x: 0.72, y: 0.68 },
  jawline_left:  { x: 0.68, y: 0.65 },
  jawline_right: { x: 0.68, y: 0.65 },
  chin_filler:   { x: 0.60, y: 0.55 },
  lip_filler:    { x: 0.80, y: 0.65 },
};

const FILLER_LABEL_Y_FRAC = {
  cheek_left:    0.48,
  cheek_right:   0.48,
  jawline_left:  0.76,
  jawline_right: 0.76,
  chin_filler:   0.86,
  lip_filler:    0.65,
};

const FILLER_PARAM_LABELS = {
  cheek_intensity:   "Cheek Filler",
  jawline_intensity: "Jawline",
  chin_intensity:    "Chin Filler",
};

const FILLER_PARAM_GROUPS = {
  cheek_intensity:   ["cheek_left", "cheek_right"],
  jawline_intensity: ["jawline_left", "jawline_right"],
  chin_intensity:    ["chin_filler"],
};

// ── Lip filler ────────────────────────────────────────────────────────────────
const TINT_COLOR_SWATCHES = {
  nude:  "rgb(175,120,90)",
  rose:  "rgb(155,60,55)",
  berry: "rgb(110,20,30)",
  red:   "rgb(160,15,15)",
  coral: "rgb(210,130,45)",
};

// ── Component ─────────────────────────────────────────────────────────────────
export default function App() {
  const [procedure, setProcedure] = useState("botox");
  const [image, setImage]         = useState(null);
  const [imageFile, setImageFile] = useState(null);
  const [regions, setRegions]     = useState(null);
  const [imageSize, setImageSize] = useState(null);
  const [selected, setSelected]   = useState({});

  const [botoxIntensities, setBotoxIntensities] = useState({
    forehead_intensity:        0.8,
    frown_intensity:           0.7,
    crows_intensity:           0.75,
    under_eye_intensity:       0.9,
    nasolabial_fold_intensity: 0.6,
    lip_intensity:             0.6,
    marionette_intensity:      0.6,
    chin_intensity:            0.6,
  });
  const [fillerIntensities, setFillerIntensities] = useState({
    cheek_intensity:   0.7,
    jawline_intensity: 0.7,
    chin_intensity:    0.7,
  });

  // Lip filler state
  const [lipStyle, setLipStyle]                     = useState("natural");
  const [lipFillerIntensity, setLipFillerIntensity] = useState(0.8);
  const [lipTint, setLipTint]                       = useState(false);
  const [lipTintColor, setLipTintColor]             = useState("rose");
  const [lipTintStrength, setLipTintStrength]       = useState(0.3);

  // Body state
  const [breastMarks, setBreastMarks]   = useState([]);   // [{x, y}] in display coords
  const [breastRadius, setBreastRadius] = useState(80);   // display pixels
  const [breastShape, setBreastShape]   = useState("round");
  const [breastSize, setBreastSize]     = useState(0.5);
  const [breastLift, setBreastLift]     = useState(0.3);

  const [result, setResult]       = useState(null);
  const [loading, setLoading]     = useState(false);
  const [detecting, setDetecting] = useState(false);
  const canvasRef = useRef(null);
  const imgRef    = useRef(null);

  // Mode-derived shortcuts
  const isFiller = procedure === "filler";
  const isBody   = procedure === "body";

  const activeKeys         = isFiller ? FILLER_KEYS        : BOTOX_KEYS;
  const currentColors      = isFiller ? FILLER_COLORS      : BOTOX_COLORS;
  const currentShrink      = isFiller ? FILLER_SHRINK      : BOTOX_SHRINK;
  const currentLabelYFrac  = isFiller ? FILLER_LABEL_Y_FRAC : BOTOX_LABEL_Y_FRAC;
  const currentParamMap    = isFiller ? FILLER_INTENSITIES : BOTOX_INTENSITIES;
  const currentParamLabels = isFiller ? FILLER_PARAM_LABELS : BOTOX_PARAM_LABELS;
  const currentIntensities    = isFiller ? fillerIntensities    : botoxIntensities;
  const setCurrentIntensities = isFiller ? setFillerIntensities : setBotoxIntensities;

  const lipFillerSelected = isFiller && !!selected["lip_filler"];

  // ── Procedure switch ──────────────────────────────────────────────────────
  const handleProcedureSwitch = async (newProc) => {
    if (newProc === procedure) return;
    setProcedure(newProc);
    setSelected({});
    setResult(null);

    if (newProc === "body") {
      setBreastMarks([]);
    } else if (procedure === "body" && imageFile && !regions) {
      // Switching FROM body (no landmarks loaded) TO a face procedure — fetch them now
      setDetecting(true);
      try {
        const form = new FormData();
        form.append("file", imageFile);
        const res = await axios.post(`${API}/landmarks`, form);
        setRegions(res.data.regions);
        setImageSize(res.data.image_size);
      } catch {
        alert("Could not detect face regions.");
      } finally {
        setDetecting(false);
      }
    }
  };

  // ── Upload ────────────────────────────────────────────────────────────────
  const handleUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    setImageFile(file);
    setResult(null);
    setSelected({});
    setRegions(null);
    setBreastMarks([]);
    setImage(URL.createObjectURL(file));

    if (isBody) return; // Body mode doesn't need landmarks

    setDetecting(true);
    try {
      const form = new FormData();
      form.append("file", file);
      const res = await axios.post(`${API}/landmarks`, form);
      setRegions(res.data.regions);
      setImageSize(res.data.image_size);
    } catch {
      alert("Could not detect face. Please try another image.");
    } finally {
      setDetecting(false);
    }
  };

  // ── Canvas draw ───────────────────────────────────────────────────────────
  useEffect(() => {
    if (!canvasRef.current || !imgRef.current) return;

    const img    = imgRef.current;
    const canvas = canvasRef.current;
    canvas.width  = img.clientWidth;
    canvas.height = img.clientHeight;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // ── Body mode: draw breast placement circles ──────────────────────────
    if (isBody) {
      breastMarks.forEach((mark, i) => {
        const label = i === 0 ? "L" : "R";

        // Dashed circle
        ctx.beginPath();
        ctx.arc(mark.x, mark.y, breastRadius, 0, Math.PI * 2);
        ctx.strokeStyle = "#c9a96e";
        ctx.lineWidth   = 2;
        ctx.setLineDash([6, 4]);
        ctx.stroke();
        ctx.setLineDash([]);

        // Centre dot
        ctx.beginPath();
        ctx.arc(mark.x, mark.y, 4, 0, Math.PI * 2);
        ctx.fillStyle = "#c9a96e";
        ctx.fill();

        // Label
        ctx.font         = "bold 14px Inter, sans-serif";
        ctx.fillStyle    = "#c9a96e";
        ctx.shadowColor  = "rgba(0,0,0,1)";
        ctx.shadowBlur   = 8;
        ctx.textAlign    = "left";
        ctx.textBaseline = "middle";
        ctx.fillText(label, mark.x + breastRadius + 8, mark.y);
        ctx.shadowBlur   = 0;
        ctx.textBaseline = "alphabetic";
      });
      return;
    }

    // ── Botox / Filler: draw landmark ellipses ────────────────────────────
    if (!regions || !imageSize) return;

    const scale = img.clientWidth / imageSize.width;

    Object.entries(regions)
      .filter(([key]) => activeKeys.has(key))
      .forEach(([key, region]) => {
        const [x1, y1, x2, y2] = region.box;
        const sx = x1 * scale, sy = y1 * scale;
        const sw = (x2 - x1) * scale, sh = (y2 - y1) * scale;
        const isSelected = !!selected[key];

        let ecx = sx + sw / 2;
        let ecy = sy + sh / 2;

        if (key === "chin_filler") ecx = ecx - 10;

        if (!isFiller) {
          if (key === "crows_left")  { ecx = sx + sw * 0.18; ecy = sy + sh * 0.42; }
          if (key === "crows_right") { ecx = sx + sw * 0.82; ecy = sy + sh * 0.42; }
          if (key === "chin")        { ecx = canvas.width / 2 + 9; ecy = sy + sh * 0.50; }
        }

        const sf = currentShrink[key] || { x: 0.70, y: 0.70 };
        const rx  = (sw / 2) * sf.x;
        const ry  = (sh / 2) * sf.y;

        if (isSelected) {
          ctx.beginPath();
          ctx.ellipse(ecx, ecy, rx, ry, 0, 0, Math.PI * 2);
          ctx.fillStyle = (currentColors[key] || "rgba(200,200,200,0.35)").replace("0.35", "0.22");
          ctx.fill();
        }

        ctx.beginPath();
        ctx.ellipse(ecx, ecy, rx, ry, 0, 0, Math.PI * 2);
        ctx.strokeStyle = isSelected ? "#c9a96e" : "rgba(255,255,255,0.55)";
        ctx.lineWidth   = isSelected ? 2.5 : 1.5;
        ctx.setLineDash(isSelected ? [] : [6, 4]);
        ctx.stroke();
        ctx.setLineDash([]);

        ctx.beginPath();
        ctx.arc(ecx, ecy, isSelected ? 4 : 2.5, 0, Math.PI * 2);
        ctx.fillStyle = isSelected ? "#c9a96e" : "rgba(255,255,255,0.7)";
        ctx.fill();

        const labelSide = region.label_side || "right";
        const fontSize  = Math.max(10, 11 * scale);
        const frac      = currentLabelYFrac[key];
        const labelY    = frac != null ? imageSize.height * frac * scale : ecy;
        const isFrown   = key === "frown";

        let edgeX, anchorX;
        if (labelSide === "left" && !isFrown) {
          edgeX   = ecx - rx;
          anchorX = Math.max(edgeX - 80 * scale, 4);
        } else {
          edgeX   = ecx + rx;
          anchorX = Math.min(edgeX + 80 * scale, canvas.width - 4);
        }

        ctx.beginPath();
        ctx.moveTo(edgeX, ecy);
        ctx.lineTo(anchorX, labelY);
        ctx.strokeStyle = isSelected ? "#c9a96e" : "rgba(255,255,255,0.45)";
        ctx.lineWidth   = 1;
        ctx.setLineDash([3, 3]);
        ctx.stroke();
        ctx.setLineDash([]);

        ctx.font         = `${isSelected ? "600 " : "400 "}${fontSize}px Inter, sans-serif`;
        ctx.fillStyle    = isSelected ? "#c9a96e" : "rgba(255,255,255,0.9)";
        ctx.shadowColor  = "rgba(0,0,0,1)";
        ctx.shadowBlur   = 10;
        ctx.textAlign    = (labelSide === "left" && !isFrown) ? "right" : "left";
        ctx.textBaseline = "middle";
        ctx.fillText(region.label, anchorX, labelY);
        ctx.shadowBlur   = 0;
        ctx.textAlign    = "left";
        ctx.textBaseline = "alphabetic";
      });
  }, [regions, selected, imageSize, procedure, breastMarks, breastRadius]);

  // ── Canvas click ──────────────────────────────────────────────────────────
  const handleCanvasClick = (e) => {
    if (isBody) {
      if (breastMarks.length >= 2) return;
      const canvas = canvasRef.current;
      if (!canvas) return;
      const rect     = canvas.getBoundingClientRect();
      const displayX = e.clientX - rect.left;
      const displayY = e.clientY - rect.top;
      setBreastMarks(prev => [...prev, { x: displayX, y: displayY }]);
      return;
    }

    if (!regions || !canvasRef.current || !imgRef.current || !imageSize) return;
    const canvas = canvasRef.current;
    const rect   = canvas.getBoundingClientRect();
    const cx     = e.clientX - rect.left;
    const cy     = e.clientY - rect.top;
    const scale  = imgRef.current.clientWidth / imageSize.width;

    let clicked = null;
    let minArea = Infinity;
    Object.entries(regions)
      .filter(([key]) => activeKeys.has(key))
      .forEach(([key, region]) => {
        const [x1, y1, x2, y2] = region.box;
        const sx = x1 * scale, sy = y1 * scale;
        const sw = (x2 - x1) * scale, sh = (y2 - y1) * scale;
        if (cx >= sx && cx <= sx + sw && cy >= sy && cy <= sy + sh) {
          const area = sw * sh;
          if (area < minArea) { minArea = area; clicked = key; }
        }
      });
    if (clicked) setSelected(prev => ({ ...prev, [clicked]: !prev[clicked] }));
  };

  // ── Simulate ──────────────────────────────────────────────────────────────
  const handleSimulate = async () => {
    if (!imageFile) return;

    // ── Body mode ──────────────────────────────────────────────────────────
    if (isBody) {
      if (breastMarks.length < 2) return;
      setLoading(true);
      setResult(null);
      try {
        const sortedMarks = [...breastMarks].sort((a, b) => a.x - b.x);
        const scaleX = imgRef.current.naturalWidth  / imgRef.current.clientWidth;
        const scaleY = imgRef.current.naturalHeight / imgRef.current.clientHeight;
        const imgW   = imageSize ? imageSize.width  : imgRef.current.naturalWidth;
        const form   = new FormData();
        form.append("file",           imageFile);
        form.append("left_x",         Math.round(sortedMarks[0].x * scaleX));
        form.append("left_y",         Math.round(sortedMarks[0].y * scaleY));
        form.append("right_x",        Math.round(sortedMarks[1].x * scaleX));
        form.append("right_y",        Math.round(sortedMarks[1].y * scaleY));
        form.append("radius",         Math.round(breastRadius * (imgW / imgRef.current.clientWidth)));
        form.append("size_intensity", breastSize);
        form.append("lift_intensity", breastLift);
        form.append("shape",          breastShape);
        const res = await axios.post(`${API}/simulate/breast`, form, { responseType: "blob" });
        setResult(URL.createObjectURL(res.data));
      } catch {
        alert("Simulation failed. Please try again.");
      } finally {
        setLoading(false);
      }
      return;
    }

    // ── Botox / Filler ─────────────────────────────────────────────────────
    if (!Object.values(selected).some(Boolean)) {
      alert("Please select at least one region to treat.");
      return;
    }
    setLoading(true);
    setResult(null);
    try {
      if (isFiller) {
        const lipsSelected   = !!selected["lip_filler"];
        const nonLipSelected = Object.entries(FILLER_PARAM_GROUPS).some(
          ([, rKeys]) => rKeys.some(r => selected[r])
        );

        let lipInputFile = imageFile;

        if (nonLipSelected) {
          const fillerForm = new FormData();
          fillerForm.append("file", imageFile);
          Object.entries(FILLER_PARAM_GROUPS).forEach(([param, rKeys]) => {
            const active = rKeys.some(r => selected[r]);
            fillerForm.append(param, active ? fillerIntensities[param] : 0);
          });

          if (!lipsSelected) {
            const res = await axios.post(`${API}/simulate/filler`, fillerForm, { responseType: "blob" });
            setResult(URL.createObjectURL(res.data));
            return;
          }

          fillerForm.append("raw", true);
          const fillerRes = await axios.post(`${API}/simulate/filler`, fillerForm, { responseType: "blob" });
          lipInputFile = new File([fillerRes.data], "filler_result.jpg", { type: "image/jpeg" });
        }

        if (lipsSelected) {
          const lipForm = new FormData();
          lipForm.append("file",          lipInputFile);
          lipForm.append("style",         lipStyle);
          lipForm.append("intensity",     lipFillerIntensity);
          lipForm.append("tint",          lipTint);
          lipForm.append("tint_color",    lipTintColor);
          lipForm.append("tint_strength", lipTintStrength);
          const lipRes = await axios.post(`${API}/simulate/advanced`, lipForm, { responseType: "blob" });
          setResult(URL.createObjectURL(lipRes.data));
        }
      } else {
        const form = new FormData();
        form.append("file", imageFile);
        Object.entries(BOTOX_INTENSITIES).forEach(([region, param]) => {
          if (selected[region]) form.append(param, botoxIntensities[param]);
        });
        const allParams = new Set(Object.values(BOTOX_INTENSITIES));
        allParams.forEach(param => { if (!form.has(param)) form.append(param, 0); });
        form.append("debug", false);
        const res = await axios.post(`${API}/simulate/botox`, form, { responseType: "blob" });
        setResult(URL.createObjectURL(res.data));
      }
    } catch {
      alert("Simulation failed. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  const selectedRegions = Object.entries(selected).filter(([, v]) => v).map(([k]) => k);
  const activeParams    = [...new Set(selectedRegions.map(r => currentParamMap[r]).filter(Boolean))];

  // Instruction hint text
  const hintText = isBody
    ? breastMarks.length === 0 ? "Click on the center of the LEFT breast"
    : breastMarks.length === 1 ? "Click on the center of the RIGHT breast"
    : "Adjust circle size, then simulate"
    : regions ? "Click on a highlighted region to select it for treatment" : "";

  // Simulate button disabled condition
  const simulateDisabled = loading || (isBody ? breastMarks.length < 2 : selectedRegions.length === 0);

  // ── Render ────────────────────────────────────────────────────────────────
  return (
    <div className="app">
      <header>
        <h1>AesthetiSim</h1>
        <p>Smart Aesthetic Treatment Simulator</p>
      </header>

      <div className="procedure-toggle">
        <button
          className={procedure === "botox" ? "active" : ""}
          onClick={() => handleProcedureSwitch("botox")}
        >
          Botox
        </button>
        <button
          className={procedure === "filler" ? "active" : ""}
          onClick={() => handleProcedureSwitch("filler")}
        >
          Filler
        </button>
        <button
          className={procedure === "body" ? "active" : ""}
          onClick={() => handleProcedureSwitch("body")}
        >
          Body
        </button>
      </div>

      <main>
        {!image ? (
          <div className="upload-zone">
            <label htmlFor="upload">
              <div className="upload-box">
                <span className="upload-icon">📷</span>
                <p>Upload patient photo</p>
                <small>Front-facing, well-lit photo recommended</small>
              </div>
            </label>
            <input id="upload" type="file" accept="image/*" onChange={handleUpload} />
          </div>
        ) : (
          <div className="workspace">
            {/* ── Image panel ─────────────────────────────────────────── */}
            <div className="image-panel">
              <div className="image-wrapper">
                <img
                  ref={imgRef}
                  src={image}
                  alt="patient"
                  onLoad={() => {
                    if (!isBody && regions) setSelected(s => ({ ...s }));
                  }}
                />
                {detecting && (
                  <div className="detecting-overlay">
                    <span>Detecting face regions...</span>
                  </div>
                )}
                {/* Canvas: always shown in body mode; shown when regions loaded in other modes */}
                {(isBody || regions) && (
                  <canvas
                    ref={canvasRef}
                    className="overlay-canvas"
                    onClick={handleCanvasClick}
                  />
                )}
              </div>
              <p className="hint">{hintText}</p>
              <button className="new-btn" onClick={() => {
                setImage(null); setImageFile(null);
                setRegions(null); setSelected({});
                setBreastMarks([]); setResult(null);
              }}>
                Upload New Photo
              </button>
            </div>

            {/* ── Controls panel ──────────────────────────────────────── */}
            <div className="controls-panel">
              <h2>Treatment Plan</h2>

              {/* ─── BODY mode ─────────────────────────────────────────── */}
              {isBody ? (
                breastMarks.length < 2 ? (
                  <p className="no-selection">
                    Mark both breast centers on the photo to access treatment controls.
                  </p>
                ) : (
                  <div className="body-controls">
                    <div className="sliders">
                      <h3>Controls</h3>

                      <div className="slider-row">
                        <label>Circle Size</label>
                        <input
                          type="range" min="30" max="250" step="5"
                          value={breastRadius}
                          onChange={e => setBreastRadius(parseInt(e.target.value))}
                        />
                        <span>{breastRadius}</span>
                      </div>

                      <div className="slider-row">
                        <label>Size</label>
                        <input
                          type="range" min="0" max="1" step="0.05"
                          value={breastSize}
                          onChange={e => setBreastSize(parseFloat(e.target.value))}
                        />
                        <span>{breastSize.toFixed(2)}</span>
                      </div>

                      <div className="slider-row">
                        <label>Lift</label>
                        <input
                          type="range" min="0" max="1" step="0.05"
                          value={breastLift}
                          onChange={e => setBreastLift(parseFloat(e.target.value))}
                        />
                        <span>{breastLift.toFixed(2)}</span>
                      </div>
                    </div>

                    <div className="control-group">
                      <span className="control-label">Shape</span>
                      <div className="style-selector">
                        {["round", "teardrop", "natural"].map(s => (
                          <button
                            key={s}
                            className={breastShape === s ? "active" : ""}
                            onClick={() => setBreastShape(s)}
                          >
                            {s[0].toUpperCase() + s.slice(1)}
                          </button>
                        ))}
                      </div>
                    </div>

                    <button
                      className="new-btn"
                      onClick={() => setBreastMarks([])}
                    >
                      Clear Marks
                    </button>
                  </div>
                )
              ) : (
                /* ─── BOTOX / FILLER mode ────────────────────────────── */
                selectedRegions.length === 0 ? (
                  <p className="no-selection">
                    No regions selected yet.<br />
                    Click on the face to select treatment areas.
                  </p>
                ) : (
                  <>
                    <div className="selected-tags">
                      {selectedRegions.map(r => (
                        <span key={r} className="tag">
                          {regions[r].label}
                          <button onClick={() => setSelected(p => ({ ...p, [r]: false }))}>×</button>
                        </span>
                      ))}
                    </div>

                    {/* Lip Filler dedicated controls */}
                    {lipFillerSelected && (
                      <div className="lip-filler-controls">
                        <h3>Lip Filler</h3>

                        <div className="control-group">
                          <span className="control-label">Style</span>
                          <div className="style-selector">
                            {["natural", "russian", "heart"].map(s => (
                              <button
                                key={s}
                                className={lipStyle === s ? "active" : ""}
                                onClick={() => setLipStyle(s)}
                              >
                                {s[0].toUpperCase() + s.slice(1)}
                              </button>
                            ))}
                          </div>
                        </div>

                        <div className="slider-row">
                          <label>Intensity</label>
                          <input
                            type="range" min="0.1" max="1.0" step="0.05"
                            value={lipFillerIntensity}
                            onChange={e => setLipFillerIntensity(parseFloat(e.target.value))}
                          />
                          <span>{lipFillerIntensity.toFixed(2)}</span>
                        </div>

                        <div className="tint-row">
                          <span className="control-label">Tint</span>
                          <button
                            className={`tint-toggle${lipTint ? " on" : ""}`}
                            onClick={() => setLipTint(t => !t)}
                          >
                            {lipTint ? "On" : "Off"}
                          </button>
                        </div>

                        {lipTint && (
                          <>
                            <div className="color-swatches">
                              {Object.entries(TINT_COLOR_SWATCHES).map(([name, rgb]) => (
                                <button
                                  key={name}
                                  className={`swatch${lipTintColor === name ? " selected" : ""}`}
                                  style={{ background: rgb }}
                                  title={name[0].toUpperCase() + name.slice(1)}
                                  onClick={() => setLipTintColor(name)}
                                />
                              ))}
                            </div>
                            <div className="slider-row">
                              <label>Strength</label>
                              <input
                                type="range" min="0.1" max="1.0" step="0.05"
                                value={lipTintStrength}
                                onChange={e => setLipTintStrength(parseFloat(e.target.value))}
                              />
                              <span>{lipTintStrength.toFixed(2)}</span>
                            </div>
                          </>
                        )}
                      </div>
                    )}

                    {/* Intensity sliders for non-lip filler / botox */}
                    {activeParams.length > 0 && (
                      <div className="sliders">
                        <h3>Intensity</h3>
                        {activeParams.map(param => (
                          <div key={param} className="slider-row">
                            <label>{currentParamLabels[param]}</label>
                            <input
                              type="range" min="0.1" max="1.0" step="0.05"
                              value={currentIntensities[param]}
                              onChange={e =>
                                setCurrentIntensities(p => ({ ...p, [param]: parseFloat(e.target.value) }))
                              }
                            />
                            <span>{currentIntensities[param].toFixed(2)}</span>
                          </div>
                        ))}
                      </div>
                    )}
                  </>
                )
              )}

              <button
                className="simulate-btn"
                onClick={handleSimulate}
                disabled={simulateDisabled}
              >
                {loading ? "Simulating..." : "Simulate Treatment"}
              </button>
              {result && (
                <a className="download-btn" href={result} download="aesthetisim_result.jpg">
                  Download Result
                </a>
              )}
            </div>
          </div>
        )}

        {result && (
          <div className="result-panel">
            <h2>Simulation Result</h2>
            <img src={result} alt="result" className="result-img" />
          </div>
        )}
      </main>
    </div>
  );
}
