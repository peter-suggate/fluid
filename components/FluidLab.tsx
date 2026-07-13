"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { FluidLabRenderer, type GPUStatus, type RendererFrameMetrics, type SimulationBackend } from "@/lib/webgpu-renderer";
import { add, cameraBasis, dot, length, normalize, orbit, pan, scale, sub, zoom } from "@/lib/math";
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
  type ViewMode
} from "@/lib/model";
import { runShellValidation, type ValidationResult } from "@/lib/validation";
import { advanceRigidBodies, boundingRadius, cloneRigidBodies, createBodyDescription, initializeRigidBody, initializeRigidBodies, rigidDiagnostics, type RigidBodyState, type RigidExternalLoad, type RigidStepDiagnostics } from "@/lib/rigid-body";
import { EulerianFluidSolver, type EulerianDiagnostics, type EulerianRenderState } from "@/lib/eulerian-solver";
import { applyFluidReactions, computeFluidLoads, type CouplingDiagnostics } from "@/lib/fluid-rigid-coupling";
import { consumeGPURigidLoad, mergeGPURigidLoads, type GPUEulerianInfo, type GPURigidLoad, type GPUGridMethod, type GPUQuality } from "@/lib/webgpu-eulerian";
import { createPaperScenario, paperScenarios, type PaperScenarioId } from "@/lib/paper-scenarios";

const RIGID_BODIES_ENABLED = true;

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

function formatGridLocation(location?: { x: number; y: number; z: number }) {
  return location ? `[${location.x}, ${location.y}, ${location.z}]` : "location pending";
}

function externalLoadsFromGPU(scene: SceneDescription, gpuLoads: GPURigidLoad[], dt: number) {
  const loads = new Map<string, RigidExternalLoad>();
  let displaced = 0, bodyImpulse = { x: 0, y: 0, z: 0 }, fluidReactionImpulse = { x: 0, y: 0, z: 0 };
  for (const load of gpuLoads) {
    const { impulse_N_s: stepImpulse, angularImpulse_N_m_s: stepAngularImpulse } = consumeGPURigidLoad(load, dt);
    const hydrodynamicForce = scale(stepImpulse, 1 / dt), hydrodynamicTorque = scale(stepAngularImpulse, 1 / dt);
    const buoyant = scale(scene.fluid.gravity_m_s2, -scene.fluid.density_kg_m3 * load.displacedVolume_m3), force = add(hydrodynamicForce, buoyant);
    loads.set(load.bodyId, { force_N: force, torque_N_m: hydrodynamicTorque, buoyantForce_N: buoyant, hydrodynamicForce_N: hydrodynamicForce, displacedFluidVolume_m3: load.displacedVolume_m3 });
    displaced += load.displacedVolume_m3;
    bodyImpulse = add(bodyImpulse, add(stepImpulse, scale(buoyant, dt)));
    // The VOS velocity blend applies the opposite of the sampled drag impulse
    // to the fluid.  Buoyancy is a separate pressure approximation and is not
    // present in the GPU exchange buffer, so do not fabricate its reaction in
    // the diagnostic.
    fluidReactionImpulse = add(fluidReactionImpulse, scale(stepImpulse, -1));
  }
  const diagnostics: CouplingDiagnostics = { displacedVolume_m3: displaced, bodyImpulse_N_s: bodyImpulse, fluidReactionImpulse_N_s: fluidReactionImpulse, momentumClosureError_N_s: length(add(bodyImpulse, fluidReactionImpulse)), coupledBodyCount: gpuLoads.filter((load) => load.displacedVolume_m3 > 0).length };
  return { loads, diagnostics };
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

interface PerformanceSnapshot {
  cpuSimulation_ms: number;
  cpuFrame_ms: number;
  cpuPhysicsSubmit_ms: number;
  cpuDataUpload_ms: number;
  cpuRenderEncode_ms: number;
  gpuAdvection_ms: number;
  gpuPressure_ms: number;
  gpuProjection_ms: number;
  gpuRigid_ms: number;
  gpuDiagnostics_ms: number;
  gpuOverhead_ms: number;
  gpuRender_ms: number;
}

const emptyPerformance: PerformanceSnapshot = {cpuSimulation_ms:0,cpuFrame_ms:0,cpuPhysicsSubmit_ms:0,cpuDataUpload_ms:0,cpuRenderEncode_ms:0,gpuAdvection_ms:0,gpuPressure_ms:0,gpuProjection_ms:0,gpuRigid_ms:0,gpuDiagnostics_ms:0,gpuOverhead_ms:0,gpuRender_ms:0};

function PerformanceDrawer({snapshot,history,onClose,timestampsAvailable}:{snapshot:PerformanceSnapshot;history:PerformanceSnapshot[];onClose:()=>void;timestampsAvailable:boolean}) {
  const gpuStages=[
    {key:"advection",label:"Advection + VOF",value:snapshot.gpuAdvection_ms,className:"stage-advection"},
    {key:"pressure",label:"Pressure Jacobi",value:snapshot.gpuPressure_ms,className:"stage-pressure"},
    {key:"projection",label:"Projection",value:snapshot.gpuProjection_ms,className:"stage-projection"},
    {key:"rigid",label:"Rigid coupling",value:snapshot.gpuRigid_ms,className:"stage-rigid"},
    {key:"diagnostics",label:"Reductions",value:snapshot.gpuDiagnostics_ms,className:"stage-diagnostics"},
    {key:"overhead",label:"Copies + queue gaps",value:snapshot.gpuOverhead_ms,className:"stage-overhead"},
    {key:"render",label:"Raymarch render",value:snapshot.gpuRender_ms,className:"stage-render"}
  ];
  const cpuOther=Math.max(0,snapshot.cpuFrame_ms-snapshot.cpuPhysicsSubmit_ms-snapshot.cpuDataUpload_ms-snapshot.cpuRenderEncode_ms);
  const cpuStages=[
    {label:"Rigid + CPU oracles",value:snapshot.cpuSimulation_ms},
    {label:"GPU physics encode",value:snapshot.cpuPhysicsSubmit_ms},
    {label:"Buffer uploads",value:snapshot.cpuDataUpload_ms},
    {label:"Render encode + submit",value:snapshot.cpuRenderEncode_ms},
    {label:"Frame orchestration",value:cpuOther}
  ];
  const gpuTotal=gpuStages.reduce((sum,stage)=>sum+stage.value,0),cpuTotal=snapshot.cpuSimulation_ms+snapshot.cpuFrame_ms,budget=16.67;
  const bottleneck=[...gpuStages].sort((a,b)=>b.value-a.value)[0];
  const historyValues=history.map((sample)=>({gpu:sample.gpuAdvection_ms+sample.gpuPressure_ms+sample.gpuProjection_ms+sample.gpuRigid_ms+sample.gpuDiagnostics_ms+sample.gpuOverhead_ms+sample.gpuRender_ms,cpu:sample.cpuSimulation_ms+sample.cpuFrame_ms}));
  const historyMax=Math.max(budget,...historyValues.flatMap((sample)=>[sample.gpu,sample.cpu]));
  const points=(key:"gpu"|"cpu")=>historyValues.map((sample,index)=>`${historyValues.length<2?0:index/(historyValues.length-1)*100},${48-Math.min(sample[key]/historyMax,1)*44}`).join(" ");
  const gpuTime=(value:number)=>!timestampsAvailable?"—":value>0?`${value.toFixed(3)} ms`:"< timer resolution";
  const cpuTime=(value:number)=>value>0?`${value.toFixed(3)} ms`:"< 0.1 ms";
  return <section id="performance-drawer" className="performance-drawer" aria-label="Performance profiler" data-testid="performance-drawer">
    <header className="performance-header"><div><p className="eyebrow">FRAME PROFILER · LIVE</p><h2>GPU and CPU pipeline contribution</h2></div><div className="performance-summary"><span><small>GPU work</small><strong>{gpuTotal.toFixed(2)} ms</strong></span><span><small>CPU work</small><strong>{cpuTotal.toFixed(2)} ms</strong></span><span><small>Largest GPU stage</small><strong>{bottleneck?.label ?? "—"}</strong></span><span><small>60 Hz budget</small><strong>{budget.toFixed(2)} ms</strong></span></div><button className="icon-button" onClick={onClose} aria-label="Close performance profiler">×</button></header>
    <div className="performance-body">
      <section className="performance-lane"><div className="performance-lane-heading"><strong>GPU queue</strong><span>{timestampsAvailable?"hardware timestamps":"timestamps unavailable"}</span></div><div className="performance-stack" aria-label={`GPU work ${gpuTotal.toFixed(2)} milliseconds of a 16.67 millisecond frame budget`}>{gpuStages.map((stage)=><i key={stage.key} className={stage.className} style={{width:`${Math.min(stage.value/budget*100,100)}%`}} />)}<b style={{left:`${Math.min(gpuTotal/budget*100,100)}%`}} /></div><div className="performance-rows">{gpuStages.map((stage)=><div className="performance-row" key={stage.key}><span><i className={stage.className}/>{stage.label}</span><div><i className={stage.className} style={{width:`${gpuTotal>0?stage.value/gpuTotal*100:0}%`}} /></div><strong>{gpuTime(stage.value)}</strong><small>{gpuTotal>0?(stage.value/gpuTotal*100).toFixed(1):"0.0"}%</small></div>)}</div></section>
      <section className="performance-lane"><div className="performance-lane-heading"><strong>CPU main thread</strong><span>wall-clock instrumentation</span></div><div className="performance-rows cpu-rows">{cpuStages.map((stage)=><div className="performance-row" key={stage.label}><span>{stage.label}</span><div><i style={{width:`${cpuTotal>0?stage.value/cpuTotal*100:0}%`}} /></div><strong>{cpuTime(stage.value)}</strong><small>{cpuTotal>0?(stage.value/cpuTotal*100).toFixed(1):"0.0"}%</small></div>)}</div><div className="performance-history"><div><strong>Recent frames</strong><span><i className="history-gpu"/>GPU <i className="history-cpu"/>CPU</span></div><svg viewBox="0 0 100 50" preserveAspectRatio="none" role="img" aria-label={`Recent GPU and CPU timing history; vertical scale zero to ${historyMax.toFixed(1)} milliseconds`}><line x1="0" y1={48-budget/historyMax*44} x2="100" y2={48-budget/historyMax*44}/><polyline className="history-gpu" points={points("gpu")}/><polyline className="history-cpu" points={points("cpu")}/></svg><small>0</small><small>{historyMax.toFixed(1)} ms</small></div></section>
    </div>
  </section>;
}

type BodyDragPhase = "start" | "move" | "end";

function WebGPUViewport({ scene, camera, setCamera, view, simulationTime, bodies, selectedBodyId, fluid, backend, quality, gridMethod, onFrame, onGPUStatus, onGPUInfo, onGPURigidLoads, onSelectBody, onDragBody }: {
  scene: SceneDescription;
  camera: CameraState;
  setCamera: React.Dispatch<React.SetStateAction<CameraState>>;
  view: ViewMode;
  simulationTime: number;
  bodies: RigidBodyState[];
  selectedBodyId?: string;
  fluid: EulerianRenderState;
  backend: SimulationBackend;
  quality: GPUQuality;
  gridMethod: GPUGridMethod;
  onFrame: (metrics: RendererFrameMetrics, resolution: string) => void;
  onGPUStatus: (status: GPUStatus) => void;
  onGPUInfo: (info: GPUEulerianInfo) => void;
  onGPURigidLoads: (loads: GPURigidLoad[]) => void;
  onSelectBody: (bodyId: string) => void;
  onDragBody: (bodyId: string, position: RigidBodyState["position_m"], velocity: RigidBodyState["linearVelocity_m_s"], phase: BodyDragPhase) => void;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rendererRef = useRef<FluidLabRenderer | null>(null);
  const stateRef = useRef({ scene, camera, view, simulationTime, bodies, selectedBodyId, fluid, backend, quality, gridMethod });
  const pointerRef = useRef<
    | { id: number; x: number; y: number; action: "orbit" | "pan" }
    | { id: number; action: "body"; bodyId: string; planePoint: RigidBodyState["position_m"]; planeNormal: RigidBodyState["position_m"]; grabOffset: RigidBodyState["position_m"]; lastPosition: RigidBodyState["position_m"]; lastTime: number }
    | null
  >(null);

  useEffect(() => {
    stateRef.current = { scene, camera, view, simulationTime, bodies, selectedBodyId, fluid, backend, quality, gridMethod };
  }, [scene, camera, view, simulationTime, bodies, selectedBodyId, fluid, backend, quality, gridMethod]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const renderer = new FluidLabRenderer(canvas, onGPUStatus, onGPUInfo, onGPURigidLoads);
    rendererRef.current = renderer;
    let frame = 0;
    let alive = true;
    renderer.initialize().then(() => {
      const render = () => {
        if (!alive) return;
        const state = stateRef.current;
        const metrics = renderer.draw(state.simulationTime, state.scene, state.camera, state.view, state.bodies, state.selectedBodyId, state.fluid, state.backend, state.quality, state.gridMethod);
        onFrame(metrics, renderer.presentationResolution);
        frame = requestAnimationFrame(render);
      };
      render();
    }).catch((error: unknown) => onGPUStatus({ state: "unavailable", label: error instanceof Error ? error.message : "WebGPU initialization failed" }));
    return () => { alive = false; cancelAnimationFrame(frame); rendererRef.current = null; };
  }, [onFrame, onGPUInfo, onGPURigidLoads, onGPUStatus]);

  const pointerRay = (event: React.PointerEvent<HTMLCanvasElement>) => {
    const rect = event.currentTarget.getBoundingClientRect(), basis = cameraBasis(stateRef.current.camera);
    const ndcX = ((event.clientX - rect.left) / Math.max(rect.width, 1)) * 2 - 1;
    const ndcY = 1 - ((event.clientY - rect.top) / Math.max(rect.height, 1)) * 2;
    return { origin: basis.position, direction: normalize(add(basis.forward, add(scale(basis.right, ndcX * rect.width / Math.max(rect.height, 1) * 0.72), scale(basis.up, ndcY * 0.72)))) };
  };
  const planeHit = (origin: RigidBodyState["position_m"], direction: RigidBodyState["position_m"], point: RigidBodyState["position_m"], normal: RigidBodyState["position_m"]) => {
    const denominator = dot(direction, normal); if (Math.abs(denominator) < 1e-6) return point;
    return add(origin, scale(direction, dot(sub(point, origin), normal) / denominator));
  };

  const pointerDown = (event: React.PointerEvent<HTMLCanvasElement>) => {
    event.currentTarget.setPointerCapture(event.pointerId);
    if (event.button === 0 && !event.shiftKey) {
      const ray = pointerRay(event); let nearest: { body: RigidBodyState; t: number } | undefined;
      for (const body of stateRef.current.bodies) {
        const oc = sub(ray.origin, body.position_m), radius = boundingRadius(body), b = dot(oc, ray.direction), c = dot(oc, oc) - radius * radius, discriminant = b * b - c;
        if (discriminant < 0) continue; const t = -b - Math.sqrt(discriminant);
        if (t > 0 && (!nearest || t < nearest.t)) nearest = { body, t };
      }
      if (nearest) {
        const basis = cameraBasis(stateRef.current.camera);
        const dragPoint = planeHit(ray.origin, ray.direction, nearest.body.position_m, basis.forward), grabOffset = sub(nearest.body.position_m, dragPoint);
        pointerRef.current = { id: event.pointerId, action: "body", bodyId: nearest.body.description.id, planePoint: nearest.body.position_m, planeNormal: basis.forward, grabOffset, lastPosition: nearest.body.position_m, lastTime: event.timeStamp };
        onSelectBody(nearest.body.description.id); onDragBody(nearest.body.description.id, nearest.body.position_m, { x: 0, y: 0, z: 0 }, "start"); return;
      }
    }
    pointerRef.current = { id: event.pointerId, x: event.clientX, y: event.clientY, action: event.shiftKey || event.button === 1 ? "pan" : "orbit" };
  };
  const pointerMove = (event: React.PointerEvent<HTMLCanvasElement>) => {
    const active = pointerRef.current;
    if (!active || active.id !== event.pointerId) return;
    if (active.action === "body") {
      const ray = pointerRay(event), position = add(planeHit(ray.origin, ray.direction, active.planePoint, active.planeNormal), active.grabOffset);
      const dt = Math.max((event.timeStamp - active.lastTime) / 1000, 1 / 240), rawVelocity = scale(sub(position, active.lastPosition), 1 / dt), speed = length(rawVelocity), velocity = speed > 6 ? scale(rawVelocity, 6 / speed) : rawVelocity;
      pointerRef.current = { ...active, lastPosition: position, lastTime: event.timeStamp };
      onDragBody(active.bodyId, position, velocity, "move"); return;
    }
    const dx = event.clientX - active.x;
    const dy = event.clientY - active.y;
    pointerRef.current = { ...active, x: event.clientX, y: event.clientY };
    setCamera((current) => active.action === "pan" ? pan(current, dx, dy) : orbit(current, dx, dy));
  };
  const pointerUp = (event: React.PointerEvent<HTMLCanvasElement>) => {
    const active = pointerRef.current;
    if (active?.id === event.pointerId) {
      if (active.action === "body") onDragBody(active.bodyId, active.lastPosition, { x: 0, y: 0, z: 0 }, "end");
      pointerRef.current = null;
    }
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
    <section className="validation-panel" aria-label="Numerical validation report" data-testid="validation-panel">
      <header>
        <div><p className="eyebrow">STAGES 3–4 · NUMERICAL CONTRACT</p><h2>{passed}/{results.length} in-app checks passed</h2></div>
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
      <p className="validation-note">The regression suite gates rigid bodies, the Eulerian MAC oracle, buoyancy, sinking, and conservative two-way impulse exchange. WebGPU supplies the high-resolution interactive Eulerian path.</p>
    </section>
  );
}

export function FluidLab() {
  const [scene, setScene] = useState<SceneDescription>(() => cloneScene(defaultScene));
  const [view, setView] = useState<ViewMode>("scientific");
  const [runState, setRunState] = useState<"paused" | "running">("running");
  const [simulationTime, setSimulationTime] = useState(0);
  const [bodies, setBodies] = useState<RigidBodyState[]>(() => initializeRigidBodies(defaultScene.rigidBodies));
  const [selectedBodyId, setSelectedBodyId] = useState<string | undefined>(defaultScene.rigidBodies[0]?.id);
  const [newBodyShape, setNewBodyShape] = useState<RigidShape>("sphere");
  const [rigidState, setRigidState] = useState<RigidStepDiagnostics>(() => rigidDiagnostics(initializeRigidBodies(defaultScene.rigidBodies), defaultScene.fluid.gravity_m_s2));
  const [initialFluidSolver] = useState(() => new EulerianFluidSolver(defaultScene));
  const fluidSolverRef = useRef(initialFluidSolver);
  const [fluidState, setFluidState] = useState<EulerianDiagnostics>(() => initialFluidSolver.diagnostics);
  const [fluidRenderState, setFluidRenderState] = useState<EulerianRenderState>(() => initialFluidSolver.getRenderState());
  const [backend, setBackend] = useState<SimulationBackend>("webgpu");
  const [gpuQuality, setGPUQuality] = useState<GPUQuality>("balanced");
  const [gpuGridMethod, setGPUGridMethod] = useState<GPUGridMethod>("tall-cell");
  const [gpuInfo, setGPUInfo] = useState<GPUEulerianInfo | null>(null);
  const [couplingState, setCouplingState] = useState<CouplingDiagnostics>({ displacedVolume_m3: 0, bodyImpulse_N_s: { x: 0, y: 0, z: 0 }, fluidReactionImpulse_N_s: { x: 0, y: 0, z: 0 }, momentumClosureError_N_s: 0, coupledBodyCount: 0 });
  const [camera, setCamera] = useState<CameraState>(defaultCamera);
  const [gpuStatus, setGPUStatus] = useState<GPUStatus>({ state: "initializing", label: "Initializing WebGPU" });
  const [frameMs, setFrameMs] = useState(0);
  const [resolution, setResolution] = useState("—");
  const [samples, setSamples] = useState<MetricSample[]>([]);
  const [performanceOpen, setPerformanceOpen] = useState(false);
  const [performanceSnapshot, setPerformanceSnapshot] = useState<PerformanceSnapshot>(emptyPerformance);
  const [performanceHistory, setPerformanceHistory] = useState<PerformanceSnapshot[]>([]);
  const [validationOpen, setValidationOpen] = useState(false);
  const [notice, setNotice] = useState("Dam-break initialized · Eulerian projection active");
  const fileRef = useRef<HTMLInputElement>(null);
  const lastClockRef = useRef<number | null>(null);
  const sampleClockRef = useRef(0);
  const accumulatorRef = useRef(0);
  const simulationTimeRef = useRef(0);
  const bodiesRef = useRef<RigidBodyState[]>(initializeRigidBodies(defaultScene.rigidBodies));
  const gpuRigidLoadsRef = useRef<GPURigidLoad[]>([]);
  const kinematicDragRef = useRef<{ bodyId: string; position: RigidBodyState["position_m"]; velocity: RigidBodyState["linearVelocity_m_s"] } | null>(null);
  const cpuOracleStepRef = useRef(0);
  const cpuSimulationMsRef = useRef(0);
  const gpuInfoRef = useRef<GPUEulerianInfo | null>(null);
  const performanceRef = useRef<PerformanceSnapshot>(emptyPerformance);
  const validationResults = useMemo(() => runShellValidation(), []);

  useEffect(() => {
    lastClockRef.current = null;
    let frame = 0;
    const tick = (now: number) => {
      const tickStart=performance.now();
      if (lastClockRef.current === null) lastClockRef.current = now;
      const elapsed = Math.min((now - lastClockRef.current) / 1000, 0.05);
      lastClockRef.current = now;
      if (runState === "running") {
        accumulatorRef.current += elapsed;
        const dt = scene.numerics.fixedDt_s;
        let steps = 0;
        let diagnostics: RigidStepDiagnostics | undefined;
        let fluidDiagnostics: EulerianDiagnostics | undefined;
        let latestCoupling: CouplingDiagnostics | undefined;
        while (accumulatorRef.current >= dt && steps < 2 && simulationTimeRef.current + dt <= scene.duration_s) {
          const simulationBodies = RIGID_BODIES_ENABLED ? bodiesRef.current : [];
          const drag = kinematicDragRef.current;
          if (drag) {
            const body = simulationBodies.find((candidate) => candidate.description.id === drag.bodyId);
            if (body) { body.position_m = { ...drag.position }; body.linearVelocity_m_s = { ...drag.velocity }; body.angularVelocity_rad_s = { x: 0, y: 0, z: 0 }; body.angularMomentum_kg_m2_s = { x: 0, y: 0, z: 0 }; }
          }
          let loads: ReadonlyMap<string, RigidExternalLoad>;
          if (backend === "webgpu") {
            const gpuCoupling = externalLoadsFromGPU(scene, RIGID_BODIES_ENABLED ? gpuRigidLoadsRef.current : [], dt);
            loads = gpuCoupling.loads; latestCoupling = gpuCoupling.diagnostics;
          } else {
            const couplingFluid = fluidSolverRef.current, coupling = computeFluidLoads(scene, couplingFluid, simulationBodies);
            latestCoupling = applyFluidReactions(couplingFluid, simulationBodies, coupling.loads, dt); loads = coupling.loads;
          }
          diagnostics = advanceRigidBodies(simulationBodies, scene, dt, 6, loads);
          if (drag) {
            const body = simulationBodies.find((candidate) => candidate.description.id === drag.bodyId);
            if (body) { body.position_m = { ...drag.position }; body.linearVelocity_m_s = { ...drag.velocity }; body.angularVelocity_rad_s = { x: 0, y: 0, z: 0 }; body.angularMomentum_kg_m2_s = { x: 0, y: 0, z: 0 }; }
          }
          cpuOracleStepRef.current += 1;
          const oracleStride = backend === "webgpu" ? 4 : 1;
          if (cpuOracleStepRef.current % oracleStride === 0) fluidDiagnostics = fluidSolverRef.current.step(dt * oracleStride);
          accumulatorRef.current -= dt;
          simulationTimeRef.current += dt;
          steps += 1;
        }
        if (steps === 2 && accumulatorRef.current > dt * 2) accumulatorRef.current = dt * 2;
        if (steps > 0) {
          setBodies(cloneRigidBodies(bodiesRef.current));
          setRigidState(diagnostics ?? rigidDiagnostics(bodiesRef.current, scene.fluid.gravity_m_s2));
          if (fluidDiagnostics) {
            setFluidState(fluidDiagnostics);
            if (backend === "cpu-reference") setFluidRenderState(fluidSolverRef.current.getRenderState());
          }
          if (latestCoupling) setCouplingState(latestCoupling);
          setSimulationTime(simulationTimeRef.current);
        }
        if (simulationTimeRef.current + dt > scene.duration_s) setRunState("paused");
        cpuSimulationMsRef.current=performance.now()-tickStart;
      }
      frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [backend, runState, scene]);

  const handleFrame = useCallback((metrics: RendererFrameMetrics, size: string) => {
    setFrameMs(metrics.cpuFrame_ms);
    setResolution(size);
    const now = performance.now();
    if (now - sampleClockRef.current > 250) {
      sampleClockRef.current = now;
      const gpu=gpuInfoRef.current?.gpuTimings,previous=performanceRef.current,snapshot:PerformanceSnapshot={cpuSimulation_ms:cpuSimulationMsRef.current,cpuFrame_ms:metrics.cpuFrame_ms,cpuPhysicsSubmit_ms:metrics.cpuPhysicsSubmit_ms,cpuDataUpload_ms:metrics.cpuDataUpload_ms,cpuRenderEncode_ms:metrics.cpuRenderEncode_ms,gpuAdvection_ms:gpu?.advection_ms??previous.gpuAdvection_ms,gpuPressure_ms:gpu?.pressure_ms??previous.gpuPressure_ms,gpuProjection_ms:gpu?.projection_ms??previous.gpuProjection_ms,gpuRigid_ms:gpu?.rigidCoupling_ms??previous.gpuRigid_ms,gpuDiagnostics_ms:gpu?.diagnostics_ms??previous.gpuDiagnostics_ms,gpuOverhead_ms:gpu?.overhead_ms??previous.gpuOverhead_ms,gpuRender_ms:metrics.gpuRender_ms??previous.gpuRender_ms};
      performanceRef.current=snapshot;setPerformanceSnapshot(snapshot);setPerformanceHistory((current)=>[...current.slice(-119),snapshot]);
      setSamples((current) => [...current.slice(-79), { t: now / 1000, frame_ms: metrics.cpuFrame_ms, volume_drift_pct: fluidSolverRef.current.diagnostics.markerVolumeDrift * 100, constraint_error: fluidSolverRef.current.diagnostics.divergenceAfter_s, kinetic_energy_J: fluidSolverRef.current.diagnostics.kineticEnergy_J }]);
    }
  }, []);
  const handleGPUInfo=useCallback((info:GPUEulerianInfo)=>{gpuInfoRef.current=info;setGPUInfo(info);},[]);
  const handleGPUStatus = useCallback((status: GPUStatus) => setGPUStatus(status), []);
  const handleGPURigidLoads = useCallback((loads: GPURigidLoad[]) => {
    gpuRigidLoadsRef.current = mergeGPURigidLoads(gpuRigidLoadsRef.current, loads);
  }, []);
  const handleBodyDrag = useCallback((bodyId: string, position: RigidBodyState["position_m"], velocity: RigidBodyState["linearVelocity_m_s"], phase: BodyDragPhase) => {
    if (phase === "end") kinematicDragRef.current = null;
    else kinematicDragRef.current = { bodyId, position: { ...position }, velocity: { ...velocity } };
    const body = bodiesRef.current.find((candidate) => candidate.description.id === bodyId);
    if (body) {
      body.position_m = { ...position }; body.linearVelocity_m_s = phase === "end" ? { x: 0, y: 0, z: 0 } : { ...velocity };
      body.angularVelocity_rad_s = { x: 0, y: 0, z: 0 }; body.angularMomentum_kg_m2_s = { x: 0, y: 0, z: 0 };
      setBodies(cloneRigidBodies(bodiesRef.current));
    }
    if (phase === "start") { setRunState("running"); setNotice("Kinematic drag active · GPU immersed boundary coupling"); }
    if (phase === "end") setNotice("Body released to buoyancy, drag, and collision response");
  }, []);

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
    fluidSolverRef.current = new EulerianFluidSolver(source);
    setFluidState(fluidSolverRef.current.diagnostics);
    setFluidRenderState(fluidSolverRef.current.getRenderState());
    simulationTimeRef.current = 0; accumulatorRef.current = 0;
    cpuOracleStepRef.current = 0;
    cpuSimulationMsRef.current=0;performanceRef.current=emptyPerformance;setPerformanceSnapshot(emptyPerformance);setPerformanceHistory([]);
    gpuRigidLoadsRef.current = []; kinematicDragRef.current = null;
    setSimulationTime(0); setRunState("paused"); setSamples([]);
    setSelectedBodyId(source.rigidBodies[0]?.id);
    setNotice(`${source.fluid.inflow ? "Inflow scene" : source.fluid.initialCondition === "dam-break" ? "Dam-break" : "Tank fill"} reset at t = 0`);
  };
  const loadPaperScenario = (id: PaperScenarioId) => {
    const preset = createPaperScenario(id, scene), metadata = paperScenarios.find((candidate) => candidate.id === id);
    setScene(preset); resetSimulation(preset);
    setCamera({ ...defaultCamera, distance_m: 2.45, target_m: { x: 0, y: 0.42, z: 0 } });
    setNotice(`${metadata?.name ?? "Paper scenario"} loaded · ${metadata?.paperFigure ?? "paper"} · dt ${preset.numerics.fixedDt_s.toFixed(4)} s`);
  };
  const loadDeepComparison = () => {
    const deep = cloneScene(scene);
    deep.sceneId = "deep-water-grid-comparison";
    deep.container.height_m = 20;
    deep.container.fillFraction = 0.8;
    deep.fluid.initialCondition = "tank-fill";
    // The tall-cell paper does not include a capillary-force discretization.
    // Keep the A/B preset within that shared physical scope so the grid and
    // pressure methods are the only variables in the comparison.
    deep.fluid.surfaceTension_N_m = 0;
    deep.numerics.fixedDt_s = 1 / 30;
    deep.numerics.maxDt_s = 1 / 30;
    deep.rigidBodies = [];
    setScene(deep); resetSimulation(deep);
    setNotice("Deep-water A/B loaded · 20 m tank · 80% fill · 1/30 s paper step · σ = 0");
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
    const payload = { manifest: createRunManifest(scene, adapter), backend, gpuQuality, gpuGridMethod, gpuInfo, shellMetrics: { presentationFrame_ms: frameMs, canvasResolution: resolution, samples, performance: performanceSnapshot, performanceHistory }, rigidBodyState: bodies, rigidBodyDiagnostics: rigidState, eulerianMetrics: fluidState, couplingMetrics: couplingState };
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
    setNotice("Body released with buoyancy, drag, torque, and fluid reaction enabled");
  };
  const singleRigidStep = () => {
    setRunState("paused");
    if (simulationTimeRef.current + scene.numerics.fixedDt_s > scene.duration_s) return;
    const activeFluid = fluidSolverRef.current, simulationBodies = RIGID_BODIES_ENABLED ? bodiesRef.current : [];
    const gpuCoupling = backend === "webgpu" ? externalLoadsFromGPU(scene, RIGID_BODIES_ENABLED ? gpuRigidLoadsRef.current : [], scene.numerics.fixedDt_s) : undefined;
    const coupling = gpuCoupling ? undefined : computeFluidLoads(scene, activeFluid, simulationBodies);
    const couplingDiagnostics = gpuCoupling?.diagnostics ?? applyFluidReactions(activeFluid, simulationBodies, coupling!.loads, scene.numerics.fixedDt_s);
    const diagnostics = advanceRigidBodies(simulationBodies, scene, scene.numerics.fixedDt_s, 6, gpuCoupling?.loads ?? coupling!.loads);
    const fluidDiagnostics = fluidSolverRef.current.step(scene.numerics.fixedDt_s);
    simulationTimeRef.current += scene.numerics.fixedDt_s;
    setSimulationTime(simulationTimeRef.current);
    setBodies(cloneRigidBodies(bodiesRef.current)); setRigidState(diagnostics);
    setFluidState(fluidDiagnostics); setFluidRenderState(fluidSolverRef.current.getRenderState());
    setCouplingState(couplingDiagnostics);
  };

  const estimatedCells = Math.ceil(scene.container.width_m / scene.nominalResolution.length_m) * Math.ceil(scene.container.height_m / scene.nominalResolution.length_m) * Math.ceil(scene.container.depth_m / scene.nominalResolution.length_m);
  return (
    <main className="lab-shell" data-run-state={runState} data-solver-mode="eulerian" data-simulation-time={simulationTime.toFixed(6)} data-body-count={RIGID_BODIES_ENABLED ? bodies.length : 0}>
      <header className="topbar">
        <div className="brand"><span className="brand-mark">FL</span><div><strong>Fluid Lab</strong><small>WEBGPU CFD WORKBENCH</small></div></div>
        <div className="solver-identity">{gpuGridMethod === "tall-cell" ? "Tall-cell VOF" : "Uniform VOF"}</div>
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
        <section className="panel-section" data-testid="paper-scenarios">
          <div className="section-heading"><h2>Paper scenarios</h2><span>Figures 3, 4, 6</span></div>
          <div className="body-list" aria-label="Restricted tall-cell paper scenarios">
            {paperScenarios.map((preset) => <button key={preset.id} onClick={() => loadPaperScenario(preset.id)}><span>{preset.name}</span><small>{preset.paperFigure} · {preset.description}</small></button>)}
          </div>
        </section>
        <section className="panel-section">
          <div className="section-heading"><h2>Container</h2><span>rectangular glass</span></div>
          <RangeControl label="Width" unit="m" value={scene.container.width_m} min={0.4} max={2.5} step={0.05} onChange={(value) => patchContainer({ width_m: value })} displayDigits={2} />
          <RangeControl label="Height" unit="m" value={scene.container.height_m} min={0.4} max={5} step={0.05} onChange={(value) => patchContainer({ height_m: value })} displayDigits={2} />
          <RangeControl label="Depth" unit="m" value={scene.container.depth_m} min={0.4} max={2} step={0.05} onChange={(value) => patchContainer({ depth_m: value })} displayDigits={2} />
          <RangeControl label="Water fill" unit="%" value={scene.container.fillFraction * 100} min={5} max={90} step={1} onChange={(value) => patchContainer({ fillFraction: value / 100 })} displayDigits={0} />
          <div className="segmented compact"><button className={scene.container.top === "open" ? "active" : ""} onClick={() => patchContainer({ top: "open" })}>Open top</button><button className={scene.container.top === "closed" ? "active" : ""} onClick={() => patchContainer({ top: "closed" })}>Closed</button></div>
          <div className="segmented compact" aria-label="Fluid wall condition"><button className={scene.container.fluidWallMode === "no-slip" ? "active" : ""} onClick={() => patchContainer({ fluidWallMode: "no-slip" })}>No slip</button><button className={scene.container.fluidWallMode === "free-slip" ? "active" : ""} onClick={() => patchContainer({ fluidWallMode: "free-slip" })}>Free slip</button></div>
        </section>
        <section className="panel-section">
          <div className="section-heading"><h2>Water</h2><span>20 °C default</span></div>
          <div className="segmented compact" aria-label="Fluid initial condition"><button className={scene.fluid.initialCondition === "dam-break" ? "active" : ""} onClick={() => patchFluid({ initialCondition: "dam-break" })}>Dam break</button><button className={scene.fluid.initialCondition === "tank-fill" ? "active" : ""} onClick={() => patchFluid({ initialCondition: "tank-fill" })}>Tank fill</button></div>
          {scene.fluid.inflow && <div className="estimate-grid"><div><small>Active inflow</small><strong>{Math.hypot(scene.fluid.inflow.velocity_m_s.x, scene.fluid.inflow.velocity_m_s.y, scene.fluid.inflow.velocity_m_s.z).toFixed(2)}</strong><span>m/s · r {scene.fluid.inflow.radius_m.toFixed(3)} m · {scene.fluid.inflow.start_s.toFixed(1)}–{scene.fluid.inflow.end_s.toFixed(1)} s</span></div></div>}
          <RangeControl label="Density" unit="kg/m³" value={scene.fluid.density_kg_m3} min={700} max={1300} step={0.1} onChange={(value) => patchFluid({ density_kg_m3: value })} displayDigits={1} />
          <RangeControl label="Dynamic viscosity" unit="Pa·s" value={scene.fluid.dynamicViscosity_Pa_s} min={0} max={0.02} step={0.000001} onChange={(value) => patchFluid({ dynamicViscosity_Pa_s: value })} displayDigits={6} />
          <RangeControl label="Surface tension" unit="N/m" value={scene.fluid.surfaceTension_N_m} min={0} max={0.15} step={0.001} onChange={(value) => patchFluid({ surfaceTension_N_m: value })} displayDigits={3} />
          <RangeControl label="Gravity Y" unit="m/s²" value={scene.fluid.gravity_m_s2.y} min={-20} max={0} step={0.01} onChange={(value) => patchFluid({ gravity_m_s2: { ...scene.fluid.gravity_m_s2, y: value } })} displayDigits={3} />
          <button className="drop-button" onClick={() => resetSimulation()}>Apply &amp; reset fluid</button>
        </section>
        {RIGID_BODIES_ENABLED && <section className="panel-section rigid-editor" data-testid="rigid-editor">
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
            <div className="body-actions">{selectedBody.description.motion !== "static" && <button className="drop-button" onClick={dropSelectedBody}>Drop selected</button>}<button onClick={() => { const resetBody = initializeRigidBody(selectedBody.description); setRuntimeBodies(bodiesRef.current.map((body) => body.description.id === selectedBodyId ? resetBody : body)); setRunState("paused"); }}>Reset body</button></div>
            <RangeControl label="Density" unit="kg/m³" value={selectedBody.description.density_kg_m3} min={100} max={4000} step={10} onChange={(value) => updateSelectedDescription({ density_kg_m3: value })} displayDigits={0} />
            <RangeControl label="Characteristic size" unit="m" value={selectedBody.description.dimensions_m.x} min={0.035} max={0.18} step={0.005} onChange={(value) => { const d = selectedBody.description.dimensions_m; const ratio = value / d.x; updateSelectedDescription({ dimensions_m: { x: value, y: d.y * ratio, z: d.z * ratio } }); }} displayDigits={3} />
            <RangeControl label="Position X" unit="m" value={selectedBody.position_m.x} min={-scene.container.width_m / 2} max={scene.container.width_m / 2} step={0.01} onChange={(value) => updateSelectedDescription({ position_m: { ...selectedBody.position_m, x: value } })} displayDigits={2} />
            <RangeControl label="Position Y" unit="m" value={selectedBody.position_m.y} min={0} max={scene.container.height_m + 0.8} step={0.01} onChange={(value) => updateSelectedDescription({ position_m: { ...selectedBody.position_m, y: value } })} displayDigits={2} />
            <RangeControl label="Position Z" unit="m" value={selectedBody.position_m.z} min={-scene.container.depth_m / 2} max={scene.container.depth_m / 2} step={0.01} onChange={(value) => updateSelectedDescription({ position_m: { ...selectedBody.position_m, z: value } })} displayDigits={2} />
            <RangeControl label="Restitution" unit="—" value={selectedBody.description.restitution} min={0} max={1} step={0.01} onChange={(value) => updateSelectedDescription({ restitution: value })} displayDigits={2} />
            <RangeControl label="Friction" unit="—" value={selectedBody.description.friction} min={0} max={1.2} step={0.01} onChange={(value) => updateSelectedDescription({ friction: value })} displayDigits={2} />
          </div>}
        </section>}
        <section className="panel-section">
          <div className="section-heading"><h2>Compute &amp; resolution</h2><span>active backend</span></div>
          <div className="segmented compact" aria-label="Simulation backend"><button className={backend === "webgpu" ? "active" : ""} onClick={() => { setBackend("webgpu"); setNotice("WebGPU compute selected; reset to rebuild fields"); }}>WebGPU</button><button className={backend === "cpu-reference" ? "active" : ""} onClick={() => { setBackend("cpu-reference"); setNotice("CPU binary64 reference selected"); }}>CPU reference</button></div>
          <div className="segmented compact" aria-label="GPU grid method"><button className={gpuGridMethod === "tall-cell" ? "active" : ""} onClick={() => { setGPUGridMethod("tall-cell"); resetSimulation(); setNotice("Restricted tall-cell grid selected"); }}>Tall cells</button><button className={gpuGridMethod === "uniform" ? "active" : ""} onClick={() => { setGPUGridMethod("uniform"); resetSimulation(); setNotice("Uniform cubic comparison grid selected"); }}>Uniform</button></div>
          <label className="select-control"><span>GPU quality</span><select aria-label="GPU quality" value={gpuQuality} onChange={(event) => { setGPUQuality(event.target.value as GPUQuality); setSimulationTime(0); simulationTimeRef.current = 0; setRunState("paused"); }}>{gpuGridMethod === "tall-cell" ? <><option value="balanced">Balanced · ~2.5k columns · 24 layers</option><option value="high">High · ~7k columns · 32 layers</option><option value="ultra">Ultra · ~12.5k columns · 40 layers</option></> : <><option value="balanced">Balanced · matched cubic grid</option><option value="high">High · matched cubic grid</option><option value="ultra">Ultra · matched cubic grid</option></>}</select></label>
          <button className="drop-button" onClick={loadDeepComparison}>Load deep-water A/B scene</button>
          <RangeControl label="Nominal length" unit="m" value={scene.nominalResolution.length_m} min={0.0125} max={0.08} step={0.0025} onChange={(value) => setScene((current) => ({ ...current, nominalResolution: { length_m: value } }))} displayDigits={4} />
          <RangeControl label="Pressure iterations" unit="iterations" value={scene.numerics.pressureMaxIterations} min={20} max={1000} step={20} onChange={(value) => setScene((current) => ({ ...current, numerics: { ...current.numerics, pressureMaxIterations: value } }))} displayDigits={0} />
          <div className="estimate-grid"><div><small>MAC allocation</small><strong>{estimatedCells.toLocaleString()}</strong><span>cells</span></div></div>
        </section>
      </aside>

      <section className="viewport-shell">
        <WebGPUViewport scene={scene} camera={camera} setCamera={setCamera} view={view} simulationTime={simulationTime} bodies={RIGID_BODIES_ENABLED ? bodies : []} selectedBodyId={RIGID_BODIES_ENABLED ? selectedBodyId : undefined} fluid={fluidRenderState} backend={backend} quality={gpuQuality} gridMethod={gpuGridMethod} onFrame={handleFrame} onGPUStatus={handleGPUStatus} onGPUInfo={handleGPUInfo} onGPURigidLoads={handleGPURigidLoads} onSelectBody={setSelectedBodyId} onDragBody={handleBodyDrag} />
        <div className="viewport-topline">
          <div className={`gpu-badge state-${gpuStatus.state}`}><span className={`status-dot ${gpuStatus.state === "ready" ? "online" : "warning"}`} /><strong>{gpuStatus.state === "ready" ? "WEBGPU" : gpuStatus.state.toUpperCase()}</strong><span>{gpuStatus.label}</span></div>
          <div className="segmented"><button className={view === "scientific" ? "active" : ""} onClick={() => setView("scientific")}>Scientific</button><button className={view === "presentation" ? "active" : ""} onClick={() => setView("presentation")}>Presentation</button></div>
        </div>
        <div className="physics-stage-badge"><strong>{gpuGridMethod === "tall-cell" ? "TALL CELLS" : "UNIFORM GRID"}</strong><span>{backend === "webgpu" ? `${gpuGridMethod === "tall-cell" ? "restricted tall-cell · paper core σ=0" : "uniform cubic"} · VOF · immersed bodies` : "CPU validation oracle active"}</span><small>{backend === "webgpu" ? `${gpuInfo?.cellCount.toLocaleString() ?? "…"} stored samples · f32 · ${gpuInfo?.pressureSolver ?? `${gpuInfo?.pressureIterations ?? "…"} Jacobi`}` : "MAC · binary64 · PCG"}</small></div>
        {view === "scientific" && <>
          <div className="axis-widget"><span className="axis-y">Y</span><span className="axis-x">X</span><span className="axis-z">Z</span></div>
          <div className="probe-label probe-a"><i />P-01 · surface</div>
          <div className="probe-label probe-b"><i />P-02 · hydrostatic</div>
        </>}
        <div className="camera-toolbar" aria-label="Camera controls">
          <button onClick={() => setPresetCamera("reset")}>Reset</button><button onClick={() => setPresetCamera("front")}>Front</button><button onClick={() => setPresetCamera("side")}>Side</button><button onClick={() => setPresetCamera("top")}>Top</button>
          <span>{RIGID_BODIES_ENABLED ? "drag body to move · " : ""}drag to orbit · ⇧ drag pan · wheel zoom</span>
        </div>
        {gpuStatus.state === "unavailable" && <div className="gpu-fallback"><strong>3D renderer unavailable</strong><p>{gpuStatus.label}</p><small>The scene editor, serialization, and CPU validation remain available.</small></div>}
      </section>

      <aside className="right-panel panel-scroll">
        <section className="panel-section diagnostics-head">
          <p className="eyebrow">LIVE DIAGNOSTICS</p>
          <div className="state-line"><span className={`status-dot ${runState === "running" ? "online pulse" : "idle"}`} /><strong>{runState === "running" ? "COUPLED RUNNING" : "PAUSED"}</strong><span>Eulerian</span></div>
        </section>
        <section className="metric-grid panel-section">
          <MetricCard label="Simulation time" value={simulationTime.toFixed(3)} unit="s" />
          <MetricCard label="GPU simulated time" value={gpuInfo?.simulatedTime_s !== undefined ? gpuInfo.simulatedTime_s.toFixed(3) : "—"} unit={`s · lag ${gpuInfo?.simulationLag_s?.toFixed(3) ?? "—"} s`} tone={gpuInfo?.simulationLag_s !== undefined && gpuInfo.simulationLag_s <= scene.numerics.maxDt_s ? "good" : "warn"} />
          <MetricCard label="Fixed validation dt" value={scene.numerics.fixedDt_s.toFixed(4)} unit="s" />
          <MetricCard label="Render encode" value={frameMs.toFixed(2)} unit="ms CPU" tone={frameMs < 4 ? "good" : "warn"} />
          {RIGID_BODIES_ENABLED && <MetricCard label="Rigid bodies" value={String(bodies.length)} unit={`${rigidState.contactCount} contact solves`} />}
          <MetricCard label="MAC grid" value={`${fluidRenderState.nx} × ${fluidRenderState.ny} × ${fluidRenderState.nz}`} unit={`${fluidState.pressureIterations} PCG iterations`} tone={fluidState.pressureConverged ? "good" : "warn"} />
          <MetricCard label="Dam front" value={fluidState.damFront_m.toFixed(3)} unit="m" />
          <MetricCard label={gpuInfo?.gridKind === "uniform" ? "GPU uniform grid" : "GPU tall grid"} value={gpuInfo ? `${gpuInfo.nx} × ${gpuInfo.storedNy} × ${gpuInfo.nz}` : "initializing"} unit={gpuInfo ? `${gpuInfo.ny} cubic-equivalent Y · ${(gpuInfo.compressionRatio * 100).toFixed(0)}% stored` : undefined} tone={backend === "webgpu" ? "good" : "neutral"} />
          <MetricCard label={gpuInfo?.gridKind === "uniform" ? "Uniform allocation" : "Tall-cell span"} value={gpuInfo?.gridKind === "uniform" ? gpuInfo.cellCount.toLocaleString() : gpuInfo?.maximumTallCellHeight !== undefined ? String(gpuInfo.maximumTallCellHeight) : "—"} unit={gpuInfo ? `${gpuInfo.gridKind === "uniform" ? "cells" : "cells"} · ${(gpuInfo.allocatedBytes / 1048576).toFixed(1)} MiB physics` : undefined} />
          <MetricCard label="GPU dam front" value={gpuInfo?.front_m !== undefined ? gpuInfo.front_m.toFixed(3) : "—"} unit="m · volume-fraction threshold" />
          <MetricCard label="GPU stability" value={gpuInfo?.stabilityFlags ? (gpuInfo.stabilityFlags.length === 0 ? "CLEAR" : "ALERT") : "—"} unit={gpuInfo?.stabilityFlags?.join(" · ") || "all instrumented gates clear"} tone={gpuInfo?.stabilityFlags?.length ? "warn" : gpuInfo?.stabilityFlags ? "good" : "neutral"} />
          <MetricCard label="GPU liquid max speed" value={gpuInfo?.maxSpeed_m_s !== undefined ? gpuInfo.maxSpeed_m_s.toFixed(3) : "—"} unit={`m/s at ${formatGridLocation(gpuInfo?.maxSpeedLocation)} · ${gpuInfo?.encodedSteps ?? 0} steps`} />
          <MetricCard label="GPU extrapolated-air speed" value={gpuInfo?.maxAirSpeed_m_s !== undefined ? gpuInfo.maxAirSpeed_m_s.toFixed(3) : "—"} unit={`m/s at ${formatGridLocation(gpuInfo?.maxAirSpeedLocation)}`} />
          <MetricCard label="GPU divergence pre → post" value={gpuInfo?.maxDivergenceBefore_s !== undefined && gpuInfo.maxDivergenceAfter_s !== undefined ? `${gpuInfo.maxDivergenceBefore_s.toExponential(2)} → ${gpuInfo.maxDivergenceAfter_s.toExponential(2)}` : "—"} unit={`s⁻¹ · ratio ${gpuInfo?.projectionDivergenceRatio?.toExponential(2) ?? "—"} · post ${formatGridLocation(gpuInfo?.maxDivergenceAfterLocation)}`} tone={gpuInfo?.lastDt_s && gpuInfo?.maxDivergenceAfter_s !== undefined && gpuInfo.maxDivergenceAfter_s * gpuInfo.lastDt_s > 0.5 ? "warn" : "neutral"} />
          <MetricCard label="GPU pressure residual" value={gpuInfo?.pressureRelativeResidual !== undefined ? gpuInfo.pressureRelativeResidual.toExponential(2) : "—"} unit={`relative L∞ · raw ${gpuInfo?.pressureResidual?.toExponential(2) ?? "—"} at ${formatGridLocation(gpuInfo?.maxPressureResidualLocation)}`} tone={gpuInfo?.pressureRelativeResidual !== undefined && gpuInfo.pressureRelativeResidual <= 0.1 ? "good" : "warn"} />
          <MetricCard label="GPU pressure maximum" value={gpuInfo?.maxPressure_Pa !== undefined ? gpuInfo.maxPressure_Pa.toExponential(2) : "—"} unit={`Pa at ${formatGridLocation(gpuInfo?.maxPressureLocation)}`} />
          <MetricCard label="GPU component CFL" value={gpuInfo?.maxComponentCfl !== undefined ? gpuInfo.maxComponentCfl.toFixed(3) : "—"} unit={`${gpuInfo?.highCflCellCount ?? 0} wet samples above 1`} tone={gpuInfo?.maxComponentCfl !== undefined && gpuInfo.maxComponentCfl <= 1 ? "good" : "warn"} />
          <MetricCard label="GPU NaN / infinity" value={gpuInfo?.nonFiniteCount !== undefined ? String(gpuInfo.nonFiniteCount) : "—"} unit="across pre-pressure, pressure, and projected fields" tone={gpuInfo?.nonFiniteCount === 0 ? "good" : "warn"} />
          <MetricCard label="GPU step" value={gpuInfo?.gpuStep_ms !== undefined ? gpuInfo.gpuStep_ms.toFixed(2) : "—"} unit="ms · timestamp query" tone={gpuInfo?.gpuStep_ms !== undefined && gpuInfo.gpuStep_ms < 16.7 ? "good" : "neutral"} />
          <MetricCard label="GPU queue throughput" value={gpuInfo?.gpuQueueWall_ms && gpuInfo.gpuQueueSimulation_s ? (gpuInfo.gpuQueueSimulation_s * 1000 / gpuInfo.gpuQueueWall_ms).toFixed(2) : "—"} unit={gpuInfo?.gpuQueueWall_ms ? `× realtime · ${gpuInfo.gpuQueueWall_ms.toFixed(1)} ms queue wall` : "synchronized completion"} tone={gpuInfo?.gpuQueueWall_ms && gpuInfo.gpuQueueSimulation_s && gpuInfo.gpuQueueSimulation_s * 1000 >= gpuInfo.gpuQueueWall_ms ? "good" : "neutral"} />
          <MetricCard label={scene.fluid.inflow ? "GPU net volume change" : "GPU volume drift"} value={gpuInfo?.volumeDrift !== undefined ? (gpuInfo.volumeDrift * 100).toFixed(2) : "—"} unit={scene.fluid.inflow ? "% · includes configured inflow" : "% · unmodified VOF integral"} tone={scene.fluid.inflow ? "neutral" : gpuInfo?.volumeDrift !== undefined && Math.abs(gpuInfo.volumeDrift) < 0.01 ? "good" : "warn"} />
          <MetricCard label="Volume correction" value="None" unit="physical VOF rendered directly" tone="good" />
        </section>
        {RIGID_BODIES_ENABLED && selectedBody && <section className="panel-section selected-diagnostics" data-testid="selected-body-diagnostics">
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
            <div><small>Buoyancy</small><strong>{length(selectedBody.buoyantForce_N).toFixed(2)}</strong><span>N</span></div>
            <div><small>Hydrodynamic force</small><strong>{length(selectedBody.hydrodynamicForce_N).toFixed(2)}</strong><span>N</span></div>
            <div><small>Displaced volume</small><strong>{selectedBody.displacedFluidVolume_m3.toExponential(2)}</strong><span>m³</span></div>
          </div>
          <small className="diagnostic-footnote">* collision impulse divided by the current fixed step · {backend === "webgpu" ? "GPU moving-solid penalization with conservative impulse readback" : "deterministic primitive quadrature"}</small>
        </section>}
        <section className="panel-section chart-section">
          <div className="section-heading"><h2>Presentation frame</h2><span>CPU encode · ms</span></div>
          <Sparkline samples={samples} />
          <div className="chart-legend"><span><i className="legend-teal" />encode time</span><span>physics GPU {gpuInfo?.gpuStep_ms?.toFixed(2)??"—"} ms</span></div>
        </section>
        {RIGID_BODIES_ENABLED && <section className="panel-section">
          <div className="section-heading"><h2>Rigid system</h2><span>CPU binary64</span></div>
          <div className="invariant-list">
            <div><span>Kinetic energy</span><strong>{rigidState.kineticEnergy_J.toFixed(3)}</strong><small>J</small></div>
            <div><span>Potential energy</span><strong>{rigidState.potentialEnergy_J.toFixed(3)}</strong><small>J · zero at floor</small></div>
            <div><span>Linear momentum |P|</span><strong>{length(rigidState.linearMomentum_kg_m_s).toFixed(3)}</strong><small>kg·m/s</small></div>
            <div><span>Max pre-correction penetration</span><strong>{rigidState.maxPenetration_m.toExponential(2)}</strong><small>m · persistent penetration is zeroed</small></div>
            <div><span>NaN / infinity</span><strong>{rigidState.nanCount}</strong><small>acceptance = 0</small></div>
          </div>
        </section>}
        <section className="panel-section fluid-pending">
          <div className="section-heading"><h2>Eulerian fluid</h2><span>CPU binary64 reference</span></div>
          <div className="invariant-list">
            <div><span>RMS divergence</span><strong>{fluidState.divergenceAfter_s.toExponential(2)}</strong><small>s⁻¹ · before {fluidState.divergenceBefore_s.toExponential(2)}</small></div>
            <div><span>PCG relative residual</span><strong>{fluidState.pressureRelativeResidual.toExponential(2)}</strong><small>{fluidState.pressureIterations} iterations · {fluidState.pressureConverged ? "converged" : "not converged"}</small></div>
            <div><span>{scene.fluid.inflow ? "Marker volume change" : "Marker volume drift"}</span><strong>{(fluidState.markerVolumeDrift * 100).toExponential(2)}</strong><small>{scene.fluid.inflow ? "% · includes configured inflow" : "% · marker mass exactly conserved"}</small></div>
            <div><span>Kinetic energy</span><strong>{fluidState.kineticEnergy_J.toFixed(2)}</strong><small>J</small></div>
            <div><span>Time-step bound</span><strong>{fluidState.limitingCondition}</strong><small>dt {fluidState.dt_s.toFixed(4)} s</small></div>
            <div><span>NaN / infinity</span><strong>{fluidState.nanCount}</strong><small>acceptance = 0</small></div>
          </div>
          <p>CPU oracle: staggered MAC, RK2 semi-Lagrangian advection, explicit viscosity, marker free surface, closed-wall flux enforcement, and matrix-free Jacobi-PCG projection. The selected WebGPU path uses {gpuGridMethod === "tall-cell" ? "one variable-height bottom cell per x/z column plus a moving band of cubic surface cells, bounded MacCormack velocity advection, narrow-band velocity extrapolation, and a restricted full-cycle multigrid pressure solve" : "the matched full-depth cubic comparison grid with conservative VOF transport and weighted Jacobi projection"}.</p>
        </section>
        {RIGID_BODIES_ENABLED && <section className="panel-section">
          <div className="section-heading"><h2>Fluid–rigid exchange</h2><span>two-way impulses</span></div>
          <div className="invariant-list"><div><span>Displaced volume</span><strong>{couplingState.displacedVolume_m3.toExponential(2)}</strong><small>m³</small></div><div><span>Coupled bodies</span><strong>{couplingState.coupledBodyCount}</strong><small>of {bodies.length}</small></div><div><span>Momentum closure</span><strong>{couplingState.momentumClosureError_N_s.toExponential(2)}</strong><small>N·s</small></div></div>
        </section>}
        <section className="panel-section">
          <div className="section-heading"><h2>Run identity</h2><span>reproducibility</span></div>
          <dl className="run-identity"><div><dt>Build</dt><dd>{BUILD_ID}</dd></div><div><dt>Active backend</dt><dd>{backend}</dd></div><div><dt>Eulerian GPU</dt><dd>{gpuGridMethod === "tall-cell" ? "f32 restricted tall-cell VOF" : "f32 uniform cubic VOF"}</dd></div><div><dt>Eulerian CPU</dt><dd>binary64 MAC PCG</dd></div><div><dt>Random seed</dt><dd>{scene.randomSeed}</dd></div></dl>
        </section>
      </aside>

      <footer className="transport-bar">
        <div className="transport-controls">
          <button className="transport-main" onClick={() => setRunState((state) => state === "running" ? "paused" : "running")} aria-label={runState === "running" ? "Pause simulation" : "Play simulation"}>{runState === "running" ? "Ⅱ" : "▶"}</button>
          <button onClick={singleRigidStep} aria-label="Single fluid clock step">STEP</button>
          <button onClick={() => resetSimulation()}>RESET</button>
          <button className={performanceOpen?"active":""} onClick={()=>setPerformanceOpen((open)=>!open)} aria-expanded={performanceOpen} aria-controls="performance-drawer">PERF</button>
        </div>
        <div className="time-readout"><span>t</span><strong>{simulationTime.toFixed(4)}</strong><small>s</small><div className="timeline"><i style={{ width: `${(simulationTime / scene.duration_s) * 100}%` }} /></div><span>{scene.duration_s.toFixed(1)} s</span></div>
        <div className="file-actions"><span className="notice">{notice}</span><button onClick={loadLocal}>Load</button><button onClick={() => fileRef.current?.click()}>Import</button><input ref={fileRef} type="file" accept="application/json,.json" onChange={importScene} hidden /></div>
      </footer>

      {performanceOpen&&<PerformanceDrawer snapshot={performanceSnapshot} history={performanceHistory} onClose={()=>setPerformanceOpen(false)} timestampsAvailable={gpuStatus.state==="ready"&&Boolean(gpuInfo?.gpuTimings)}/>}

      {validationOpen && <ValidationPanel results={validationResults} onClose={() => setValidationOpen(false)} gpuStatus={gpuStatus} />}
    </main>
  );
}
