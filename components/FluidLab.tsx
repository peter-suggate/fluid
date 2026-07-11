"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { FluidLabRenderer, type GPUStatus } from "@/lib/webgpu-renderer";
import { length, orbit, pan, zoom } from "@/lib/math";
import {
  BUILD_ID,
  cloneScene,
  createRunManifest,
  defaultCamera,
  defaultScene,
  parseScene,
  serializeScene,
  type CameraState,
  type MetricSample,
  type RigidBodyDescription,
  type RigidShape,
  type SceneDescription,
  type SolverMode,
  type ViewMode
} from "@/lib/model";
import { runShellValidation, type ValidationResult } from "@/lib/validation";
import { advanceRigidBodies, boundingRadius, cloneRigidBodies, createBodyDescription, initializeRigidBody, initializeRigidBodies, rigidDiagnostics, type RigidBodyState, type RigidStepDiagnostics } from "@/lib/rigid-body";

const modeNames: Record<SolverMode, string> = {
  eulerian: "Eulerian",
  particle: "Particle",
  compare: "Compare"
};

function downloadText(name: string, text: string, mime = "application/json") {
  const url = URL.createObjectURL(new Blob([text], { type: mime }));
  const link = document.createElement("a");
  link.href = url;
  link.download = name;
  link.click();
  URL.revokeObjectURL(url);
}

function formatNumber(value: number, digits = 3) {
  if (Math.abs(value) < 0.001 && value !== 0) return value.toExponential(2);
  return value.toFixed(digits);
}

interface RangeControlProps {
  label: string;
  unit: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (value: number) => void;
  displayDigits?: number;
}

function RangeControl({ label, unit, value, min, max, step, onChange, displayDigits = 3 }: RangeControlProps) {
  return (
    <label className="range-control">
      <span className="control-heading"><span>{label}</span><output>{formatNumber(value, displayDigits)} <small>{unit}</small></output></span>
      <input type="range" min={min} max={max} step={step} value={value} onChange={(event) => onChange(Number(event.target.value))} />
    </label>
  );
}

function MetricCard({ label, value, unit, tone = "neutral" }: { label: string; value: string; unit?: string; tone?: "neutral" | "good" | "warn" }) {
  return <div className={`metric-card tone-${tone}`}><span>{label}</span><strong>{value}</strong>{unit && <small>{unit}</small>}</div>;
}

function Sparkline({ samples }: { samples: MetricSample[] }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    const width = Math.max(1, Math.floor(canvas.clientWidth * ratio));
    const height = Math.max(1, Math.floor(canvas.clientHeight * ratio));
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, width, height);
    ctx.strokeStyle = "rgba(145, 242, 213, .13)";
    ctx.lineWidth = ratio;
    for (let i = 1; i < 4; i += 1) {
      const y = (height * i) / 4;
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(width, y); ctx.stroke();
    }
    if (samples.length < 2) return;
    const values = samples.map((sample) => sample.frame_ms);
    const max = Math.max(16.7, ...values);
    ctx.strokeStyle = "#83f1d1";
    ctx.lineWidth = 1.5 * ratio;
    ctx.beginPath();
    values.forEach((value, index) => {
      const x = (index / Math.max(values.length - 1, 1)) * width;
      const y = height - (value / max) * (height - 4 * ratio) - 2 * ratio;
      if (index === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();
  }, [samples]);
  return <canvas ref={canvasRef} className="sparkline" aria-label="Presentation frame time history" />;
}

function WebGPUViewport({ scene, camera, setCamera, mode, view, simulationTime, bodies, selectedBodyId, onFrame, onGPUStatus }: {
  scene: SceneDescription;
  camera: CameraState;
  setCamera: React.Dispatch<React.SetStateAction<CameraState>>;
  mode: SolverMode;
  view: ViewMode;
  simulationTime: number;
  bodies: RigidBodyState[];
  selectedBodyId?: string;
  onFrame: (frameMs: number, resolution: string) => void;
  onGPUStatus: (status: GPUStatus) => void;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rendererRef = useRef<FluidLabRenderer | null>(null);
  const stateRef = useRef({ scene, camera, mode, view, simulationTime, bodies, selectedBodyId });
  const pointerRef = useRef<{ id: number; x: number; y: number; action: "orbit" | "pan" } | null>(null);

  useEffect(() => {
    stateRef.current = { scene, camera, mode, view, simulationTime, bodies, selectedBodyId };
  }, [scene, camera, mode, view, simulationTime, bodies, selectedBodyId]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const renderer = new FluidLabRenderer(canvas, onGPUStatus);
    rendererRef.current = renderer;
    let frame = 0;
    let alive = true;
    renderer.initialize().then(() => {
      const render = () => {
        if (!alive) return;
        const state = stateRef.current;
        const frameMs = renderer.draw(state.simulationTime, state.scene, state.camera, state.mode, state.view, state.bodies, state.selectedBodyId);
        onFrame(frameMs, `${canvas.width} × ${canvas.height}`);
        frame = requestAnimationFrame(render);
      };
      render();
    }).catch((error: unknown) => onGPUStatus({ state: "unavailable", label: error instanceof Error ? error.message : "WebGPU initialization failed" }));
    return () => { alive = false; cancelAnimationFrame(frame); rendererRef.current = null; };
  }, [onFrame, onGPUStatus]);

  const pointerDown = (event: React.PointerEvent<HTMLCanvasElement>) => {
    event.currentTarget.setPointerCapture(event.pointerId);
    pointerRef.current = { id: event.pointerId, x: event.clientX, y: event.clientY, action: event.shiftKey || event.button === 1 ? "pan" : "orbit" };
  };
  const pointerMove = (event: React.PointerEvent<HTMLCanvasElement>) => {
    const active = pointerRef.current;
    if (!active || active.id !== event.pointerId) return;
    const dx = event.clientX - active.x;
    const dy = event.clientY - active.y;
    pointerRef.current = { ...active, x: event.clientX, y: event.clientY };
    setCamera((current) => active.action === "pan" ? pan(current, dx, dy) : orbit(current, dx, dy));
  };
  const pointerUp = (event: React.PointerEvent<HTMLCanvasElement>) => {
    if (pointerRef.current?.id === event.pointerId) pointerRef.current = null;
  };

  return (
    <canvas
      ref={canvasRef}
      className="gpu-canvas"
      aria-label="Interactive three-dimensional fluid laboratory viewport"
      data-testid="gpu-viewport"
      data-camera-azimuth={camera.azimuth_rad.toFixed(6)}
      data-camera-elevation={camera.elevation_rad.toFixed(6)}
      onPointerDown={pointerDown}
      onPointerMove={pointerMove}
      onPointerUp={pointerUp}
      onPointerCancel={pointerUp}
      onWheel={(event) => { event.preventDefault(); setCamera((current) => zoom(current, event.deltaY)); }}
      onContextMenu={(event) => event.preventDefault()}
    />
  );
}

function ValidationPanel({ results, onClose, gpuStatus }: { results: ValidationResult[]; onClose: () => void; gpuStatus: GPUStatus }) {
  const passed = results.filter((result) => result.passed).length;
  return (
    <section className="validation-panel" aria-label="Stage 3 validation report" data-testid="validation-panel">
      <header>
        <div><p className="eyebrow">STAGE 3 · RIGID-BODY CONTRACT</p><h2>{passed}/{results.length} in-app checks passed</h2></div>
        <button className="icon-button" onClick={onClose} aria-label="Close validation report">×</button>
      </header>
      <div className="validation-summary">
        <span className={`status-dot ${gpuStatus.state === "ready" ? "online" : "warning"}`} />
        <div><strong>GPU capability</strong><small>{gpuStatus.label}</small></div>
      </div>
      <div className="validation-list">
        {results.map((result) => (
          <article key={result.id} className={result.passed ? "pass" : "fail"}>
            <span className="result-mark">{result.passed ? "PASS" : "FAIL"}</span>
            <div><strong>{result.id} · {result.name}</strong><small>Measured {result.measured} · Acceptance {result.threshold}</small></div>
          </article>
        ))}
      </div>
      <p className="validation-note">Rigid-body equations are active and unit-tested. The water remains a presentation field: buoyancy, drag, pressure, fluid volume, divergence, and density are still unmeasured until their solver stages.</p>
    </section>
  );
}

export function FluidLab() {
  const [scene, setScene] = useState<SceneDescription>(() => cloneScene(defaultScene));
  const [mode, setMode] = useState<SolverMode>("compare");
  const [view, setView] = useState<ViewMode>("scientific");
  const [runState, setRunState] = useState<"paused" | "running">("running");
  const [simulationTime, setSimulationTime] = useState(0);
  const [bodies, setBodies] = useState<RigidBodyState[]>(() => initializeRigidBodies(defaultScene.rigidBodies));
  const [selectedBodyId, setSelectedBodyId] = useState<string | undefined>(defaultScene.rigidBodies[0]?.id);
  const [newBodyShape, setNewBodyShape] = useState<RigidShape>("sphere");
  const [rigidState, setRigidState] = useState<RigidStepDiagnostics>(() => rigidDiagnostics(initializeRigidBodies(defaultScene.rigidBodies), defaultScene.fluid.gravity_m_s2));
  const [camera, setCamera] = useState<CameraState>(defaultCamera);
  const [gpuStatus, setGPUStatus] = useState<GPUStatus>({ state: "initializing", label: "Initializing WebGPU" });
  const [frameMs, setFrameMs] = useState(0);
  const [resolution, setResolution] = useState("—");
  const [samples, setSamples] = useState<MetricSample[]>([]);
  const [validationOpen, setValidationOpen] = useState(false);
  const [notice, setNotice] = useState("Stage 3 rigid bodies · fluid coupling disabled");
  const fileRef = useRef<HTMLInputElement>(null);
  const lastClockRef = useRef<number | null>(null);
  const sampleClockRef = useRef(0);
  const accumulatorRef = useRef(0);
  const simulationTimeRef = useRef(0);
  const bodiesRef = useRef<RigidBodyState[]>(initializeRigidBodies(defaultScene.rigidBodies));
  const validationResults = useMemo(() => runShellValidation(), []);

  useEffect(() => {
    lastClockRef.current = null;
    let frame = 0;
    const tick = (now: number) => {
      if (lastClockRef.current === null) lastClockRef.current = now;
      const elapsed = Math.min((now - lastClockRef.current) / 1000, 0.05);
      lastClockRef.current = now;
      if (runState === "running") {
        accumulatorRef.current += elapsed;
        const dt = scene.numerics.fixedDt_s;
        let steps = 0;
        let diagnostics: RigidStepDiagnostics | undefined;
        while (accumulatorRef.current >= dt && steps < 64 && simulationTimeRef.current + dt <= scene.duration_s) {
          diagnostics = advanceRigidBodies(bodiesRef.current, scene, dt);
          accumulatorRef.current -= dt;
          simulationTimeRef.current += dt;
          steps += 1;
        }
        if (steps === 64 && accumulatorRef.current > dt * 64) accumulatorRef.current = dt * 64;
        if (steps > 0) {
          setBodies(cloneRigidBodies(bodiesRef.current));
          setRigidState(diagnostics ?? rigidDiagnostics(bodiesRef.current, scene.fluid.gravity_m_s2));
          setSimulationTime(simulationTimeRef.current);
        }
        if (simulationTimeRef.current + dt > scene.duration_s) setRunState("paused");
      }
      frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [runState, scene]);

  const handleFrame = useCallback((value: number, size: string) => {
    setFrameMs(value);
    setResolution(size);
    const now = performance.now();
    if (now - sampleClockRef.current > 250) {
      sampleClockRef.current = now;
      setSamples((current) => [...current.slice(-79), { t: now / 1000, frame_ms: value, volume_drift_pct: Number.NaN, constraint_error: Number.NaN, kinetic_energy_J: Number.NaN }]);
    }
  }, []);
  const handleGPUStatus = useCallback((status: GPUStatus) => setGPUStatus(status), []);

  const patchScene = <K extends keyof SceneDescription>(key: K, value: SceneDescription[K]) => setScene((current) => ({ ...current, [key]: value }));
  const patchContainer = (patch: Partial<SceneDescription["container"]>) => setScene((current) => ({ ...current, container: { ...current.container, ...patch } }));
  const patchFluid = (patch: Partial<SceneDescription["fluid"]>) => setScene((current) => ({ ...current, fluid: { ...current.fluid, ...patch } }));

  const setRuntimeBodies = (next: RigidBodyState[], gravity = scene.fluid.gravity_m_s2) => {
    bodiesRef.current = next;
    const snapshot = cloneRigidBodies(next);
    setBodies(snapshot);
    setRigidState(rigidDiagnostics(snapshot, gravity));
  };
  const resetSimulation = (source = scene) => {
    const next = initializeRigidBodies(source.rigidBodies);
    setRuntimeBodies(next, source.fluid.gravity_m_s2);
    simulationTimeRef.current = 0; accumulatorRef.current = 0;
    setSimulationTime(0); setRunState("paused"); setSamples([]);
    setSelectedBodyId(source.rigidBodies[0]?.id);
    setNotice("Rigid-body state reset to scene initial conditions");
  };
  const saveScene = () => {
    localStorage.setItem("fluid-lab.scene.v1", serializeScene(scene));
    downloadText(`${scene.sceneId}.fluid.json`, serializeScene(scene));
    setNotice("Scene saved locally and exported");
  };
  const loadLocal = () => {
    const stored = localStorage.getItem("fluid-lab.scene.v1");
    if (!stored) { fileRef.current?.click(); return; }
    try { const loaded = parseScene(stored); setScene(loaded); resetSimulation(loaded); setNotice("Loaded the last local scene"); }
    catch { setNotice("Stored scene failed validation"); }
  };
  const importScene = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    try { const loaded = parseScene(await file.text()); setScene(loaded); resetSimulation(loaded); setNotice(`Loaded ${file.name}`); }
    catch (error) { setNotice(error instanceof Error ? error.message : "Scene import failed"); }
    event.target.value = "";
  };
  const exportMetrics = () => {
    const adapter = gpuStatus.state === "ready" ? gpuStatus.adapter : gpuStatus.label;
    const payload = { manifest: createRunManifest(scene, mode, adapter), shellMetrics: { presentationFrame_ms: frameMs, canvasResolution: resolution, samples }, rigidBodyState: bodies, rigidBodyDiagnostics: rigidState, fluidMetrics: null };
    downloadText(`fluid-lab-run-${Date.now()}.json`, JSON.stringify(payload, null, 2) + "\n");
    setNotice("Run manifest and shell metrics exported");
  };
  const setPresetCamera = (preset: "front" | "side" | "top" | "reset") => {
    if (preset === "reset") setCamera(defaultCamera);
    else if (preset === "front") setCamera({ ...defaultCamera, azimuth_rad: 0, elevation_rad: 0.08 });
    else if (preset === "side") setCamera({ ...defaultCamera, azimuth_rad: Math.PI / 2, elevation_rad: 0.08 });
    else setCamera({ ...defaultCamera, azimuth_rad: 0, elevation_rad: 1.34, distance_m: 2.25 });
  };

  const selectedBody = bodies.find((body) => body.description.id === selectedBodyId);
  const updateSelectedDescription = (patch: Partial<RigidBodyDescription>) => {
    if (!selectedBodyId) return;
    setRunState("paused");
    const descriptions = scene.rigidBodies.map((body) => body.id === selectedBodyId ? { ...body, ...patch } : body);
    const updatedScene = { ...scene, rigidBodies: descriptions };
    setScene(updatedScene);
    const next = bodiesRef.current.map((body) => body.description.id === selectedBodyId ? initializeRigidBody(descriptions.find((item) => item.id === selectedBodyId)!) : body);
    setRuntimeBodies(next, updatedScene.fluid.gravity_m_s2);
    setNotice("Body parameters updated; simulation paused");
  };
  const addBody = () => {
    if (scene.rigidBodies.length >= 12) { setNotice("Renderer limit is 12 bodies in this verified increment"); return; }
    let bodyIndex = 1;
    while (scene.rigidBodies.some((body) => body.id === `body-${newBodyShape}-${bodyIndex}`)) bodyIndex += 1;
    const description = createBodyDescription(newBodyShape, bodyIndex, scene.container.height_m);
    const updated = { ...scene, rigidBodies: [...scene.rigidBodies, description] };
    setScene(updated);
    setRuntimeBodies([...bodiesRef.current, initializeRigidBody(description)], updated.fluid.gravity_m_s2);
    setSelectedBodyId(description.id); setRunState("paused");
    setNotice(`${description.name} added above the container`);
  };
  const removeSelectedBody = () => {
    if (!selectedBodyId) return;
    const descriptions = scene.rigidBodies.filter((body) => body.id !== selectedBodyId);
    const updated = { ...scene, rigidBodies: descriptions };
    setScene(updated);
    setRuntimeBodies(bodiesRef.current.filter((body) => body.description.id !== selectedBodyId), updated.fluid.gravity_m_s2);
    setSelectedBodyId(descriptions[0]?.id); setRunState("paused");
    setNotice("Selected body removed");
  };
  const dropSelectedBody = () => {
    if (!selectedBodyId) return;
    const next = bodiesRef.current.map((body) => {
      if (body.description.id !== selectedBodyId) return body;
      const dropped = initializeRigidBody({ ...body.description, position_m: { x: body.position_m.x, y: scene.container.height_m + boundingRadius(body.description) + 0.08, z: body.position_m.z }, linearVelocity_m_s: { x: 0, y: 0, z: 0 } });
      return dropped;
    });
    setRuntimeBodies(next);
    setRunState("running");
    setNotice("Body released under gravity; fluid force is disabled");
  };
  const singleRigidStep = () => {
    setRunState("paused");
    if (simulationTimeRef.current + scene.numerics.fixedDt_s > scene.duration_s) return;
    const diagnostics = advanceRigidBodies(bodiesRef.current, scene, scene.numerics.fixedDt_s);
    simulationTimeRef.current += scene.numerics.fixedDt_s;
    setSimulationTime(simulationTimeRef.current);
    setBodies(cloneRigidBodies(bodiesRef.current)); setRigidState(diagnostics);
  };

  const estimatedCells = Math.ceil(scene.container.width_m / scene.nominalResolution.length_m) * Math.ceil(scene.container.height_m / scene.nominalResolution.length_m) * Math.ceil(scene.container.depth_m / scene.nominalResolution.length_m);
  const estimatedParticles = Math.round(scene.container.width_m * scene.container.height_m * scene.container.depth_m * scene.container.fillFraction / Math.pow(scene.numerics.particleSpacing_m, 3));

  return (
    <main className="lab-shell" data-run-state={runState} data-solver-mode={mode} data-simulation-time={simulationTime.toFixed(6)} data-body-count={bodies.length}>
      <header className="topbar">
        <div className="brand"><span className="brand-mark">FL</span><div><strong>Fluid Lab</strong><small>WEBGPU CFD WORKBENCH</small></div></div>
        <nav className="solver-tabs" aria-label="Solver presentation mode">
          {(Object.keys(modeNames) as SolverMode[]).map((value) => <button key={value} className={mode === value ? "active" : ""} onClick={() => setMode(value)} aria-pressed={mode === value}>{modeNames[value]}{value === "compare" && <span className="beta">SYNC</span>}</button>)}
        </nav>
        <div className="top-actions">
          <button className="quiet-button" onClick={() => setValidationOpen(true)}><span className={`status-dot ${validationResults.every((result) => result.passed) ? "online" : "warning"}`} />Validation</button>
          <button className="quiet-button" onClick={saveScene}>Save scene</button>
          <button className="primary-button" onClick={exportMetrics}>Export run</button>
        </div>
      </header>

      <aside className="left-panel panel-scroll">
        <section className="panel-section scene-title">
          <p className="eyebrow">SCENE DESCRIPTION · SI</p>
          <input aria-label="Scene name" value={scene.sceneId} onChange={(event) => patchScene("sceneId", event.target.value)} />
          <div className="scene-meta"><span>schema {scene.schemaVersion}</span><span>seed {scene.randomSeed}</span></div>
        </section>
        <section className="panel-section">
          <div className="section-heading"><h2>Container</h2><span>rectangular glass</span></div>
          <RangeControl label="Width" unit="m" value={scene.container.width_m} min={0.4} max={2.5} step={0.05} onChange={(value) => patchContainer({ width_m: value })} displayDigits={2} />
          <RangeControl label="Height" unit="m" value={scene.container.height_m} min={0.4} max={1.8} step={0.05} onChange={(value) => patchContainer({ height_m: value })} displayDigits={2} />
          <RangeControl label="Depth" unit="m" value={scene.container.depth_m} min={0.4} max={2} step={0.05} onChange={(value) => patchContainer({ depth_m: value })} displayDigits={2} />
          <RangeControl label="Water fill" unit="%" value={scene.container.fillFraction * 100} min={5} max={90} step={1} onChange={(value) => patchContainer({ fillFraction: value / 100 })} displayDigits={0} />
          <div className="segmented compact"><button className={scene.container.top === "open" ? "active" : ""} onClick={() => patchContainer({ top: "open" })}>Open top</button><button className={scene.container.top === "closed" ? "active" : ""} onClick={() => patchContainer({ top: "closed" })}>Closed</button></div>
        </section>
        <section className="panel-section">
          <div className="section-heading"><h2>Water</h2><span>20 °C default</span></div>
          <RangeControl label="Density" unit="kg/m³" value={scene.fluid.density_kg_m3} min={700} max={1300} step={0.1} onChange={(value) => patchFluid({ density_kg_m3: value })} displayDigits={1} />
          <RangeControl label="Dynamic viscosity" unit="Pa·s" value={scene.fluid.dynamicViscosity_Pa_s} min={0} max={0.02} step={0.000001} onChange={(value) => patchFluid({ dynamicViscosity_Pa_s: value })} displayDigits={6} />
          <RangeControl label="Gravity Y" unit="m/s²" value={scene.fluid.gravity_m_s2.y} min={-20} max={0} step={0.01} onChange={(value) => patchFluid({ gravity_m_s2: { ...scene.fluid.gravity_m_s2, y: value } })} displayDigits={3} />
        </section>
        <section className="panel-section rigid-editor" data-testid="rigid-editor">
          <div className="section-heading"><h2>Rigid bodies</h2><span>{bodies.length}/12 active</span></div>
          <div className="body-list" aria-label="Rigid bodies">
            {bodies.map((body) => <button key={body.description.id} aria-pressed={selectedBodyId === body.description.id} className={selectedBodyId === body.description.id ? "active" : ""} onClick={() => setSelectedBodyId(body.description.id)}><i className={`shape-${body.description.shape}`} /><span>{body.description.name}</span><small>{body.description.shape}</small></button>)}
          </div>
          <div className="body-add-row">
            <select aria-label="New rigid body shape" value={newBodyShape} onChange={(event) => setNewBodyShape(event.target.value as RigidShape)}>
              <option value="sphere">Sphere</option><option value="box">Box</option><option value="capsule">Capsule</option><option value="cylinder">Cylinder</option>
            </select>
            <button onClick={addBody}>+ Add body</button>
          </div>
          {selectedBody && <div className="selected-editor">
            <div className="selected-heading"><div><strong>{selectedBody.description.name}</strong><small>{selectedBody.description.shape} · {selectedBody.description.shape === "sphere" ? "exact sphere narrow phase" : "bounding proxy for body contacts"}</small></div><button onClick={removeSelectedBody} aria-label="Remove selected rigid body">Remove</button></div>
            <div className="body-actions"><button className="drop-button" onClick={dropSelectedBody}>Drop selected</button><button onClick={() => { const resetBody = initializeRigidBody(selectedBody.description); setRuntimeBodies(bodiesRef.current.map((body) => body.description.id === selectedBodyId ? resetBody : body)); setRunState("paused"); }}>Reset body</button></div>
            <RangeControl label="Density" unit="kg/m³" value={selectedBody.description.density_kg_m3} min={100} max={4000} step={10} onChange={(value) => updateSelectedDescription({ density_kg_m3: value })} displayDigits={0} />
            <RangeControl label="Characteristic size" unit="m" value={selectedBody.description.dimensions_m.x} min={0.035} max={0.18} step={0.005} onChange={(value) => { const d = selectedBody.description.dimensions_m; const ratio = value / d.x; updateSelectedDescription({ dimensions_m: { x: value, y: d.y * ratio, z: d.z * ratio } }); }} displayDigits={3} />
            <RangeControl label="Position X" unit="m" value={selectedBody.position_m.x} min={-scene.container.width_m / 2} max={scene.container.width_m / 2} step={0.01} onChange={(value) => updateSelectedDescription({ position_m: { ...selectedBody.position_m, x: value } })} displayDigits={2} />
            <RangeControl label="Position Y" unit="m" value={selectedBody.position_m.y} min={0} max={scene.container.height_m + 0.8} step={0.01} onChange={(value) => updateSelectedDescription({ position_m: { ...selectedBody.position_m, y: value } })} displayDigits={2} />
            <RangeControl label="Position Z" unit="m" value={selectedBody.position_m.z} min={-scene.container.depth_m / 2} max={scene.container.depth_m / 2} step={0.01} onChange={(value) => updateSelectedDescription({ position_m: { ...selectedBody.position_m, z: value } })} displayDigits={2} />
            <RangeControl label="Restitution" unit="—" value={selectedBody.description.restitution} min={0} max={1} step={0.01} onChange={(value) => updateSelectedDescription({ restitution: value })} displayDigits={2} />
            <RangeControl label="Friction" unit="—" value={selectedBody.description.friction} min={0} max={1.2} step={0.01} onChange={(value) => updateSelectedDescription({ friction: value })} displayDigits={2} />
          </div>}
        </section>
        <section className="panel-section">
          <div className="section-heading"><h2>Discretization</h2><span>comparison mapping</span></div>
          <RangeControl label="Nominal length" unit="m" value={scene.nominalResolution.length_m} min={0.0125} max={0.08} step={0.0025} onChange={(value) => setScene((current) => ({ ...current, nominalResolution: { length_m: value } }))} displayDigits={4} />
          <div className="estimate-grid"><div><small>MAC allocation</small><strong>{estimatedCells.toLocaleString()}</strong><span>cells</span></div><div><small>SPH estimate</small><strong>{estimatedParticles.toLocaleString()}</strong><span>particles</span></div></div>
        </section>
      </aside>

      <section className="viewport-shell">
        <WebGPUViewport scene={scene} camera={camera} setCamera={setCamera} mode={mode} view={view} simulationTime={simulationTime} bodies={bodies} selectedBodyId={selectedBodyId} onFrame={handleFrame} onGPUStatus={handleGPUStatus} />
        <div className="viewport-topline">
          <div className={`gpu-badge state-${gpuStatus.state}`}><span className={`status-dot ${gpuStatus.state === "ready" ? "online" : "warning"}`} /><strong>{gpuStatus.state === "ready" ? "WEBGPU" : gpuStatus.state.toUpperCase()}</strong><span>{gpuStatus.label}</span></div>
          <div className="segmented"><button className={view === "scientific" ? "active" : ""} onClick={() => setView("scientific")}>Scientific</button><button className={view === "presentation" ? "active" : ""} onClick={() => setView("presentation")}>Presentation</button></div>
        </div>
        {mode === "compare" && <div className="compare-labels"><span><b>01</b> MAC GRID</span><span><b>02</b> DFSPH PARTICLES</span></div>}
        <div className="physics-stage-badge"><strong>STAGE 3</strong><span>Rigid-body CPU reference active</span><small>Fluid coupling disabled</small></div>
        {view === "scientific" && <>
          <div className="axis-widget"><span className="axis-y">Y</span><span className="axis-x">X</span><span className="axis-z">Z</span></div>
          <div className="probe-label probe-a"><i />P-01 · surface</div>
          <div className="probe-label probe-b"><i />P-02 · hydrostatic</div>
        </>}
        <div className="camera-toolbar" aria-label="Camera controls">
          <button onClick={() => setPresetCamera("reset")}>Reset</button><button onClick={() => setPresetCamera("front")}>Front</button><button onClick={() => setPresetCamera("side")}>Side</button><button onClick={() => setPresetCamera("top")}>Top</button>
          <span>drag orbit · ⇧ drag pan · wheel zoom</span>
        </div>
        {gpuStatus.state === "unavailable" && <div className="gpu-fallback"><strong>3D renderer unavailable</strong><p>{gpuStatus.label}</p><small>The scene editor, serialization, and CPU validation remain available.</small></div>}
      </section>

      <aside className="right-panel panel-scroll">
        <section className="panel-section diagnostics-head">
          <p className="eyebrow">LIVE DIAGNOSTICS</p>
          <div className="state-line"><span className={`status-dot ${runState === "running" ? "online pulse" : "idle"}`} /><strong>{runState === "running" ? "RIGID PHYSICS RUNNING" : "PAUSED"}</strong><span>{modeNames[mode]}</span></div>
        </section>
        <section className="metric-grid panel-section">
          <MetricCard label="Simulation time" value={simulationTime.toFixed(3)} unit="s" />
          <MetricCard label="Fixed validation dt" value={scene.numerics.fixedDt_s.toFixed(4)} unit="s" />
          <MetricCard label="Render encode" value={frameMs.toFixed(2)} unit="ms CPU" tone={frameMs < 4 ? "good" : "warn"} />
          <MetricCard label="Rigid bodies" value={String(bodies.length)} unit={`${rigidState.contactCount} contact solves`} />
        </section>
        {selectedBody && <section className="panel-section selected-diagnostics" data-testid="selected-body-diagnostics">
          <div className="section-heading"><h2>{selectedBody.description.name}</h2><span>selected state</span></div>
          <div className="body-vectors">
            <div><span>Position</span><strong>{formatNumber(selectedBody.position_m.x, 3)}, {formatNumber(selectedBody.position_m.y, 3)}, {formatNumber(selectedBody.position_m.z, 3)}</strong><small>m</small></div>
            <div><span>Linear velocity</span><strong>{formatNumber(selectedBody.linearVelocity_m_s.x, 3)}, {formatNumber(selectedBody.linearVelocity_m_s.y, 3)}, {formatNumber(selectedBody.linearVelocity_m_s.z, 3)}</strong><small>m/s</small></div>
            <div><span>Angular velocity</span><strong>{formatNumber(selectedBody.angularVelocity_rad_s.x, 2)}, {formatNumber(selectedBody.angularVelocity_rad_s.y, 2)}, {formatNumber(selectedBody.angularVelocity_rad_s.z, 2)}</strong><small>rad/s</small></div>
            <div><span>Orientation q</span><strong>{selectedBody.orientation.w.toFixed(3)}, {selectedBody.orientation.x.toFixed(3)}, {selectedBody.orientation.y.toFixed(3)}, {selectedBody.orientation.z.toFixed(3)}</strong><small>w, x, y, z</small></div>
          </div>
          <div className="force-grid">
            <div><small>Mass</small><strong>{selectedBody.mass_kg.toFixed(3)}</strong><span>kg</span></div>
            <div><small>Net force</small><strong>{length(selectedBody.netForce_N).toFixed(2)}</strong><span>N</span></div>
            <div><small>Net torque</small><strong>{length(selectedBody.netTorque_N_m).toFixed(3)}</strong><span>N·m</span></div>
            <div><small>Collision force*</small><strong>{(length(selectedBody.collisionImpulse_N_s) / scene.numerics.fixedDt_s).toFixed(1)}</strong><span>N</span></div>
          </div>
          <small className="diagnostic-footnote">* impulse divided by the current fixed step · hydrodynamic force and buoyancy: not implemented</small>
        </section>}
        <section className="panel-section chart-section">
          <div className="section-heading"><h2>Presentation frame</h2><span>CPU encode · ms</span></div>
          <Sparkline samples={samples} />
          <div className="chart-legend"><span><i className="legend-teal" />encode time</span><span>physics GPU time —</span></div>
        </section>
        <section className="panel-section">
          <div className="section-heading"><h2>Rigid system</h2><span>CPU binary64</span></div>
          <div className="invariant-list">
            <div><span>Kinetic energy</span><strong>{rigidState.kineticEnergy_J.toFixed(3)}</strong><small>J</small></div>
            <div><span>Potential energy</span><strong>{rigidState.potentialEnergy_J.toFixed(3)}</strong><small>J · zero at floor</small></div>
            <div><span>Linear momentum |P|</span><strong>{length(rigidState.linearMomentum_kg_m_s).toFixed(3)}</strong><small>kg·m/s</small></div>
            <div><span>Max pre-correction penetration</span><strong>{rigidState.maxPenetration_m.toExponential(2)}</strong><small>m · persistent penetration is zeroed</small></div>
            <div><span>NaN / infinity</span><strong>{rigidState.nanCount}</strong><small>acceptance = 0</small></div>
          </div>
        </section>
        <section className="panel-section fluid-pending">
          <div className="section-heading"><h2>Fluid invariants</h2><span>pending stages 4–7</span></div>
          <p>No buoyancy, drag, pressure projection, density constraint, or momentum exchange is claimed in this build. Bodies interact with gravity, each other, and the glass container only.</p>
        </section>
        <section className="panel-section">
          <div className="section-heading"><h2>Run identity</h2><span>reproducibility</span></div>
          <dl className="run-identity"><div><dt>Build</dt><dd>{BUILD_ID}</dd></div><div><dt>Rigid CPU</dt><dd>binary64 active</dd></div><div><dt>Render GPU</dt><dd>f32</dd></div><div><dt>Narrow phase</dt><dd>sphere exact / proxy general</dd></div><div><dt>Random seed</dt><dd>{scene.randomSeed}</dd></div></dl>
        </section>
      </aside>

      <footer className="transport-bar">
        <div className="transport-controls">
          <button className="transport-main" onClick={() => setRunState((state) => state === "running" ? "paused" : "running")} aria-label={runState === "running" ? "Pause simulation" : "Play simulation"}>{runState === "running" ? "Ⅱ" : "▶"}</button>
          <button onClick={singleRigidStep} aria-label="Single rigid-body step">STEP</button>
          <button onClick={() => resetSimulation()}>RESET</button>
        </div>
        <div className="time-readout"><span>t</span><strong>{simulationTime.toFixed(4)}</strong><small>s</small><div className="timeline"><i style={{ width: `${(simulationTime / scene.duration_s) * 100}%` }} /></div><span>{scene.duration_s.toFixed(1)} s</span></div>
        <div className="file-actions"><span className="notice">{notice}</span><button onClick={loadLocal}>Load</button><button onClick={() => fileRef.current?.click()}>Import</button><input ref={fileRef} type="file" accept="application/json,.json" onChange={importScene} hidden /></div>
      </footer>

      {validationOpen && <ValidationPanel results={validationResults} onClose={() => setValidationOpen(false)} gpuStatus={gpuStatus} />}
    </main>
  );
}
