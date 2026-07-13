"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { FluidLabRenderer, type GPUStatus, type RendererFrameMetrics, type SimulationBackend } from "@/lib/webgpu-renderer";
import { add, cameraBasis, dot, length, normalize, orbit, pan, scale, sub, zoom } from "@/lib/math";
import {
  BUILD_ID,
  cloneScene,
  createRunManifest,
  defaultCamera,
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
import { consumeGPURigidLoad, mergeGPURigidLoads, type GPUEulerianInfo, type GPURigidLoad, type GPUQuality } from "@/lib/webgpu-eulerian";
import { appendSimulationFrame, createSimulationRecording, simulationFrameAt, type SimulationRecording } from "@/lib/simulation-recording";
import {createScenarioScene,SCENARIOS,type ScenarioId} from "@/lib/scenarios";

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

function externalLoadsFromGPU(scene: SceneDescription, gpuLoads: GPURigidLoad[], dt: number) {
  const loads = new Map<string, RigidExternalLoad>();
  let displaced = 0, bodyImpulse = { x: 0, y: 0, z: 0 };
  for (const load of gpuLoads) {
    const { impulse_N_s: stepImpulse, angularImpulse_N_m_s: stepAngularImpulse } = consumeGPURigidLoad(load, dt);
    const hydrodynamicForce = scale(stepImpulse, 1 / dt), hydrodynamicTorque = scale(stepAngularImpulse, 1 / dt);
    const buoyant = scale(scene.fluid.gravity_m_s2, -scene.fluid.density_kg_m3 * load.displacedVolume_m3), force = add(hydrodynamicForce, buoyant);
    loads.set(load.bodyId, { force_N: force, torque_N_m: hydrodynamicTorque, buoyantForce_N: buoyant, hydrodynamicForce_N: hydrodynamicForce, displacedFluidVolume_m3: load.displacedVolume_m3 });
    displaced += load.displacedVolume_m3; bodyImpulse = add(bodyImpulse, add(stepImpulse, scale(buoyant, dt)));
  }
  const diagnostics: CouplingDiagnostics = { displacedVolume_m3: displaced, bodyImpulse_N_s: bodyImpulse, fluidReactionImpulse_N_s: scale(bodyImpulse, -1), momentumClosureError_N_s: 0, coupledBodyCount: gpuLoads.filter((load) => load.displacedVolume_m3 > 0).length };
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
      <input aria-label={label} type="range" min={min} max={max} step={step} value={value} onChange={(event) => onChange(Number(event.target.value))} />
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
  wallFrame_ms: number;
  cpuPhysicsSubmit_ms: number;
  cpuCommandEncode_ms: number;
  cpuQueueSubmit_ms: number;
  cpuDataUpload_ms: number;
  cpuRenderEncode_ms: number;
  cpuTopology_ms: number;
  gpuAdvection_ms: number;
  gpuControl_ms: number;
  gpuPressure_ms: number;
  gpuProjection_ms: number;
  gpuRigid_ms: number;
  gpuDiagnostics_ms: number;
  gpuOverhead_ms: number;
  gpuRender_ms: number;
  gpuQueueLatency_ms: number;
  simulationLag_ms: number;
  simulationThroughput_x: number;
  blockedFrames: number;
  gpuComputeSampleAvailable: boolean;
  gpuRenderSampleAvailable: boolean;
}

const emptyPerformance: PerformanceSnapshot = {cpuSimulation_ms:0,cpuFrame_ms:0,wallFrame_ms:0,cpuPhysicsSubmit_ms:0,cpuCommandEncode_ms:0,cpuQueueSubmit_ms:0,cpuDataUpload_ms:0,cpuRenderEncode_ms:0,cpuTopology_ms:0,gpuAdvection_ms:0,gpuControl_ms:0,gpuPressure_ms:0,gpuProjection_ms:0,gpuRigid_ms:0,gpuDiagnostics_ms:0,gpuOverhead_ms:0,gpuRender_ms:0,gpuQueueLatency_ms:0,simulationLag_ms:0,simulationThroughput_x:0,blockedFrames:0,gpuComputeSampleAvailable:false,gpuRenderSampleAvailable:false};

function PerformanceDrawer({snapshot,history,onClose,timestampsSupported,computeTimestampsEnabled,renderTimestampsEnabled}:{snapshot:PerformanceSnapshot;history:PerformanceSnapshot[];onClose:()=>void;timestampsSupported:boolean;computeTimestampsEnabled:boolean;renderTimestampsEnabled:boolean}) {
  const gpuStages=[
    {key:"advection",label:"Hierarchical VOF",value:snapshot.gpuAdvection_ms,className:"stage-advection",available:snapshot.gpuComputeSampleAvailable,enabled:computeTimestampsEnabled},
    {key:"control",label:"Solver control + copies",value:snapshot.gpuControl_ms,className:"stage-control",available:snapshot.gpuComputeSampleAvailable,enabled:computeTimestampsEnabled},
    {key:"pressure",label:"Pressure solve",value:snapshot.gpuPressure_ms,className:"stage-pressure",available:snapshot.gpuComputeSampleAvailable,enabled:computeTimestampsEnabled},
    {key:"projection",label:"Face projection",value:snapshot.gpuProjection_ms,className:"stage-projection",available:snapshot.gpuComputeSampleAvailable,enabled:computeTimestampsEnabled},
    {key:"rigid",label:"Rigid coupling",value:snapshot.gpuRigid_ms,className:"stage-rigid",available:snapshot.gpuComputeSampleAvailable,enabled:computeTimestampsEnabled},
    {key:"diagnostics",label:"Reductions",value:snapshot.gpuDiagnostics_ms,className:"stage-diagnostics",available:snapshot.gpuComputeSampleAvailable,enabled:computeTimestampsEnabled},
    {key:"overhead",label:"Unattributed compute",value:snapshot.gpuOverhead_ms,className:"stage-overhead",available:snapshot.gpuComputeSampleAvailable,enabled:computeTimestampsEnabled},
    {key:"render",label:"Surface extraction + refraction",value:snapshot.gpuRender_ms,className:"stage-render",available:snapshot.gpuRenderSampleAvailable,enabled:renderTimestampsEnabled}
  ];
  const cpuOther=Math.max(0,snapshot.cpuFrame_ms-snapshot.cpuPhysicsSubmit_ms-snapshot.cpuDataUpload_ms-snapshot.cpuRenderEncode_ms);
  const cpuStages=[
    {label:"Rigid + CPU oracles",value:snapshot.cpuSimulation_ms},
    {label:"GPU command encoding",value:snapshot.cpuCommandEncode_ms},
    {label:"GPU queue.submit",value:snapshot.cpuQueueSubmit_ms},
    {label:"Buffer uploads",value:snapshot.cpuDataUpload_ms},
    {label:"Render encode + submit",value:snapshot.cpuRenderEncode_ms},
    {label:"Adaptive topology rebuild",value:snapshot.cpuTopology_ms},
    {label:"Frame orchestration",value:cpuOther}
  ];
  const sampledGPUStages=gpuStages.filter((stage)=>stage.enabled&&stage.available),gpuTotal=sampledGPUStages.reduce((sum,stage)=>sum+stage.value,0),cpuTotal=cpuStages.reduce((sum,stage)=>sum+stage.value,0),budget=16.67;
  const bottleneck=sampledGPUStages.length>0?[...sampledGPUStages].sort((a,b)=>b.value-a.value)[0]:undefined;
  const historyValues=history.map((sample)=>({gpu:(sample.gpuComputeSampleAvailable?sample.gpuAdvection_ms+sample.gpuControl_ms+sample.gpuPressure_ms+sample.gpuProjection_ms+sample.gpuRigid_ms+sample.gpuDiagnostics_ms+sample.gpuOverhead_ms:0)+(sample.gpuRenderSampleAvailable?sample.gpuRender_ms:0),cpu:sample.wallFrame_ms}));
  const historyMax=Math.max(budget,...historyValues.flatMap((sample)=>[sample.gpu,sample.cpu]));
  const points=(key:"gpu"|"cpu")=>historyValues.map((sample,index)=>`${historyValues.length<2?0:index/(historyValues.length-1)*100},${48-Math.min(sample[key]/historyMax,1)*44}`).join(" ");
  const gpuTime=(value:number,available:boolean,enabled:boolean)=>!timestampsSupported||!enabled?"—":!available?"sampling…":value>0?`${value.toFixed(3)} ms`:"< timer resolution";
  const cpuTime=(value:number)=>value>0?`${value.toFixed(3)} ms`:"< 0.1 ms";
  const timingLabel=!timestampsSupported?"timestamps unavailable":!computeTimestampsEnabled&&!renderTimestampsEnabled?"timestamps disabled · queue-safe metrics active":!computeTimestampsEnabled?"compute timestamps disabled · queue-safe metrics active":snapshot.gpuComputeSampleAvailable&&snapshot.gpuRenderSampleAvailable?"hardware timestamps":sampledGPUStages.length>0?"partial hardware sample":"awaiting first sample";
  return <section id="performance-drawer" className="performance-drawer" aria-label="Performance profiler" data-testid="performance-drawer">
    <header className="performance-header"><div><p className="eyebrow">FRAME PROFILER · LIVE</p><h2>GPU queue, simulation throughput, and presentation</h2></div><div className="performance-summary"><span><small>GPU work</small><strong>{sampledGPUStages.length>0?`${gpuTotal.toFixed(2)} ms`:timestampsSupported&&(computeTimestampsEnabled||renderTimestampsEnabled)?"sampling…":"—"}</strong></span><span><small>Queue latency</small><strong>{snapshot.gpuQueueLatency_ms>0?`${snapshot.gpuQueueLatency_ms.toFixed(1)} ms`:"—"}</strong></span><span><small>Simulation rate</small><strong>{snapshot.simulationThroughput_x>0?`${snapshot.simulationThroughput_x.toFixed(2)}×`:"—"}</strong></span><span><small>GPU lag</small><strong>{snapshot.simulationLag_ms>0?`${snapshot.simulationLag_ms.toFixed(1)} ms`:"< 1 ms"}</strong></span><span><small>Presentation</small><strong>{snapshot.wallFrame_ms>0?`${snapshot.wallFrame_ms.toFixed(1)} ms`:"—"}</strong></span><span><small>Largest GPU stage</small><strong>{bottleneck?.label ?? "Unavailable"}</strong></span></div><button className="icon-button" onClick={onClose} aria-label="Close performance profiler">×</button></header>
    <div className="performance-body">
      <section className="performance-lane"><div className="performance-lane-heading"><strong>GPU queue</strong><span>{timingLabel}</span></div><div className="performance-stack" aria-label={`Sampled GPU work ${gpuTotal.toFixed(2)} milliseconds of a 16.67 millisecond frame budget`}>{gpuStages.map((stage)=><i key={stage.key} className={stage.className} style={{width:`${stage.enabled&&stage.available?Math.min(stage.value/budget*100,100):0}%`}} />)}<b style={{left:`${Math.min(gpuTotal/budget*100,100)}%`}} /></div><div className="performance-rows">{gpuStages.map((stage)=><div className="performance-row" key={stage.key}><span><i className={stage.className}/>{stage.label}</span><div><i className={stage.className} style={{width:`${stage.enabled&&stage.available&&gpuTotal>0?stage.value/gpuTotal*100:0}%`}} /></div><strong>{gpuTime(stage.value,stage.available,stage.enabled)}</strong><small>{stage.enabled&&stage.available&&gpuTotal>0?`${(stage.value/gpuTotal*100).toFixed(1)}%`:"—"}</small></div>)}</div></section>
      <section className="performance-lane"><div className="performance-lane-heading"><strong>CPU main thread</strong><span>{snapshot.blockedFrames} frames blocked by queue cap</span></div><div className="performance-rows cpu-rows">{cpuStages.map((stage)=><div className="performance-row" key={stage.label}><span>{stage.label}</span><div><i style={{width:`${cpuTotal>0?stage.value/cpuTotal*100:0}%`}} /></div><strong>{cpuTime(stage.value)}</strong><small>{cpuTotal>0?(stage.value/cpuTotal*100).toFixed(1):"0.0"}%</small></div>)}</div><div className="performance-history"><div><strong>Recent frames</strong><span><i className="history-gpu"/>GPU work <i className="history-cpu"/>presentation interval</span></div><svg viewBox="0 0 100 50" preserveAspectRatio="none" role="img" aria-label={`Recent GPU work and presentation interval; vertical scale zero to ${historyMax.toFixed(1)} milliseconds`}><line x1="0" y1={48-budget/historyMax*44} x2="100" y2={48-budget/historyMax*44}/><polyline className="history-gpu" points={points("gpu")}/><polyline className="history-cpu" points={points("cpu")}/></svg><small>0</small><small>{historyMax.toFixed(1)} ms</small></div></section>
    </div>
  </section>;
}

type BodyDragPhase = "start" | "move" | "end";

function WebGPUViewport({ scene, camera, setCamera, view, simulationTime, bodies, selectedBodyId, fluid, backend, quality, onFrame, onGPUStatus, onGPUInfo, onGPURigidLoads, onSelectBody, onDragBody }: {
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
  onFrame: (metrics: RendererFrameMetrics, resolution: string) => void;
  onGPUStatus: (status: GPUStatus) => void;
  onGPUInfo: (info: GPUEulerianInfo) => void;
  onGPURigidLoads: (loads: GPURigidLoad[]) => void;
  onSelectBody: (bodyId: string) => void;
  onDragBody: (bodyId: string, position: RigidBodyState["position_m"], velocity: RigidBodyState["linearVelocity_m_s"], phase: BodyDragPhase) => void;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rendererRef = useRef<FluidLabRenderer | null>(null);
  const stateRef = useRef({ scene, camera, view, simulationTime, bodies, selectedBodyId, fluid, backend, quality });
  const pointerRef = useRef<
    | { id: number; x: number; y: number; action: "orbit" | "pan" }
    | { id: number; action: "body"; bodyId: string; planePoint: RigidBodyState["position_m"]; planeNormal: RigidBodyState["position_m"]; grabOffset: RigidBodyState["position_m"]; lastPosition: RigidBodyState["position_m"]; lastTime: number }
    | null
  >(null);

  useEffect(() => {
    stateRef.current = { scene, camera, view, simulationTime, bodies, selectedBodyId, fluid, backend, quality };
  }, [scene, camera, view, simulationTime, bodies, selectedBodyId, fluid, backend, quality]);

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
        const metrics = renderer.draw(state.simulationTime, state.scene, state.camera, state.view, state.bodies, state.selectedBodyId, state.fluid, state.backend, state.quality);
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

const createInteractiveScenario=(id:ScenarioId)=>{const scene=createScenarioScene(id);scene.hierarchy.levels=1;return scene;};
const initialWebScene=createInteractiveScenario("dam-break");
const emptyFluidState:EulerianDiagnostics={step:0,time_s:0,dt_s:0,limitingCondition:"fixed",advectiveLimit_s:0,viscousLimit_s:0,divergenceBefore_s:0,divergenceAfter_s:0,pressureResidual:0,pressureRelativeResidual:0,pressureIterations:0,pressureConverged:false,markerVolume_m3:0,markerVolumeDrift:0,occupiedVolume_m3:0,occupiedVolumeDrift:0,maxSpeed_m_s:0,kineticEnergy_J:0,damFront_m:-initialWebScene.container.width_m/2,boundaryPenetrationCount:0,nanCount:0};
const emptyFluidRenderState:EulerianRenderState={nx:1,ny:1,nz:1,occupancy:new Uint8Array(1),revision:0};

export function FluidLab() {
  const [scene, setScene] = useState<SceneDescription>(() => cloneScene(initialWebScene));
  const [view, setView] = useState<ViewMode>("scientific");
  const [runState, setRunState] = useState<"paused" | "running">("paused");
  const [simulationTime, setSimulationTime] = useState(0);
  const [bodies, setBodies] = useState<RigidBodyState[]>(() => initializeRigidBodies(initialWebScene.rigidBodies));
  const [selectedBodyId, setSelectedBodyId] = useState<string | undefined>(initialWebScene.rigidBodies[0]?.id);
  const [newBodyShape, setNewBodyShape] = useState<RigidShape>("sphere");
  const [rigidState, setRigidState] = useState<RigidStepDiagnostics>(() => rigidDiagnostics(initializeRigidBodies(initialWebScene.rigidBodies), initialWebScene.fluid.gravity_m_s2));
  const fluidSolverRef = useRef<EulerianFluidSolver|null>(null);
  const [fluidState, setFluidState] = useState<EulerianDiagnostics>(emptyFluidState);
  const [fluidRenderState, setFluidRenderState] = useState<EulerianRenderState>(emptyFluidRenderState);
  const [backend, setBackend] = useState<SimulationBackend>("webgpu");
  const [gpuQuality, setGPUQuality] = useState<GPUQuality>("balanced");
  const [liveCPUOracle,setLiveCPUOracle]=useState(false);
  const [gpuInfo, setGPUInfo] = useState<GPUEulerianInfo | null>(null);
  const [couplingState, setCouplingState] = useState<CouplingDiagnostics>({ displacedVolume_m3: 0, bodyImpulse_N_s: { x: 0, y: 0, z: 0 }, fluidReactionImpulse_N_s: { x: 0, y: 0, z: 0 }, momentumClosureError_N_s: 0, coupledBodyCount: 0 });
  const [camera, setCamera] = useState<CameraState>(defaultCamera);
  const [gpuStatus, setGPUStatus] = useState<GPUStatus>({ state: "initializing", label: "Initializing WebGPU" });
  const gpuComputeBlocked=backend==="webgpu"&&gpuStatus.state==="ready"&&!gpuStatus.computeAvailable;
  const [frameMs, setFrameMs] = useState(0);
  const [resolution, setResolution] = useState("—");
  const [samples, setSamples] = useState<MetricSample[]>([]);
  const [performanceOpen, setPerformanceOpen] = useState(false);
  const [performanceSnapshot, setPerformanceSnapshot] = useState<PerformanceSnapshot>(emptyPerformance);
  const [performanceHistory, setPerformanceHistory] = useState<PerformanceSnapshot[]>([]);
  const [validationOpen, setValidationOpen] = useState(false);
  const [activeScenario,setActiveScenario]=useState<ScenarioId>("dam-break");
  const [playbackRecording, setPlaybackRecording] = useState<SimulationRecording | null>(null);
  const [playbackTime, setPlaybackTime] = useState(0);
  const [recordingDuration, setRecordingDuration] = useState(0);
  const [notice, setNotice] = useState("Dam-break initialized · paused · press Run or STEP");
  const fileRef = useRef<HTMLInputElement>(null);
  const lastClockRef = useRef<number | null>(null);
  const sampleClockRef = useRef(0);
  const presentationClockRef = useRef<number | null>(null);
  const accumulatorRef = useRef(0);
  const simulationTimeRef = useRef(0);
  const bodiesRef = useRef<RigidBodyState[]>(initializeRigidBodies(initialWebScene.rigidBodies));
  const gpuRigidLoadsRef = useRef<GPURigidLoad[]>([]);
  const kinematicDragRef = useRef<{ bodyId: string; position: RigidBodyState["position_m"]; velocity: RigidBodyState["linearVelocity_m_s"] } | null>(null);
  const cpuOracleStepRef = useRef(0);
  const cpuSimulationMsRef = useRef(0);
  const gpuInfoRef = useRef<GPUEulerianInfo | null>(null);
  const performanceRef = useRef<PerformanceSnapshot>(emptyPerformance);
  const playbackActiveRef = useRef(false);
  const [initialRecording] = useState(() => createSimulationRecording(initialWebScene, "webgpu", "balanced", initializeRigidBodies(initialWebScene.rigidBodies), emptyFluidRenderState));
  const recordingRef = useRef<SimulationRecording | null>(initialRecording);
  const [validationResults,setValidationResults]=useState<ValidationResult[]|null>(null);

  const captureSimulationFrame = useCallback((force = false) => {
    const recording = recordingRef.current;
    if (!recording || playbackRecording) return;
    if (appendSimulationFrame(recording, simulationTimeRef.current, bodiesRef.current, fluidSolverRef.current?.getRenderState()??emptyFluidRenderState, force)) {
      setRecordingDuration(recording.duration_s);
    }
  }, [playbackRecording]);

  useEffect(() => {
    lastClockRef.current = null;
    let frame = 0;
    const tick = (now: number) => {
      const tickStart=performance.now();
      if (lastClockRef.current === null) lastClockRef.current = now;
      const elapsed = Math.min((now - lastClockRef.current) / 1000, 0.05);
      lastClockRef.current = now;
      if (runState === "running" && !playbackActiveRef.current&&!gpuComputeBlocked) {
        accumulatorRef.current += elapsed;
        const dt = backend==="webgpu"&&gpuQuality==="balanced"?scene.numerics.maxDt_s:scene.numerics.fixedDt_s;
        let steps = 0;const realtimeCatchupSteps=Math.max(2,Math.ceil(0.05/dt)),maxSteps=backend==="webgpu"?(liveCPUOracle?Math.min(16,realtimeCatchupSteps):Math.min(64,realtimeCatchupSteps)):2;
        let diagnostics: RigidStepDiagnostics | undefined;
        let fluidDiagnostics: EulerianDiagnostics | undefined;
        let latestCoupling: CouplingDiagnostics | undefined;
        while (accumulatorRef.current >= dt && steps < maxSteps && simulationTimeRef.current + dt <= scene.duration_s) {
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
            const couplingFluid = fluidSolverRef.current;if(!couplingFluid)break;const coupling = computeFluidLoads(scene, couplingFluid, simulationBodies);
            latestCoupling = applyFluidReactions(couplingFluid, simulationBodies, coupling.loads, dt); loads = coupling.loads;
          }
          diagnostics = advanceRigidBodies(simulationBodies, scene, dt, 6, loads);
          if (drag) {
            const body = simulationBodies.find((candidate) => candidate.description.id === drag.bodyId);
            if (body) { body.position_m = { ...drag.position }; body.linearVelocity_m_s = { ...drag.velocity }; body.angularVelocity_rad_s = { x: 0, y: 0, z: 0 }; body.angularMomentum_kg_m2_s = { x: 0, y: 0, z: 0 }; }
          }
          if((backend==="cpu-reference"||liveCPUOracle)&&fluidSolverRef.current){cpuOracleStepRef.current += 1;const oracleStride=backend==="webgpu"?4:1;if(cpuOracleStepRef.current%oracleStride===0)fluidDiagnostics=fluidSolverRef.current.step(dt*oracleStride);}
          accumulatorRef.current -= dt;
          simulationTimeRef.current += dt;
          steps += 1;
        }
        if (steps === maxSteps && accumulatorRef.current > dt * maxSteps) accumulatorRef.current = dt * maxSteps;
        if (steps > 0) {
          setBodies(cloneRigidBodies(bodiesRef.current));
          setRigidState(diagnostics ?? rigidDiagnostics(bodiesRef.current, scene.fluid.gravity_m_s2));
          if (fluidDiagnostics) {
            setFluidState(fluidDiagnostics);
            if (backend === "cpu-reference"&&fluidSolverRef.current) setFluidRenderState(fluidSolverRef.current.getRenderState());
          }
          if (latestCoupling) setCouplingState(latestCoupling);
          setSimulationTime(simulationTimeRef.current);
          captureSimulationFrame();
        }
        if (simulationTimeRef.current + dt > scene.duration_s) setRunState("paused");
        cpuSimulationMsRef.current=performance.now()-tickStart;
      }
      frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [backend, captureSimulationFrame,gpuComputeBlocked,gpuQuality,liveCPUOracle, playbackRecording, runState, scene]);

  useEffect(() => {
    if (!playbackRecording) return;
    const startedAt = performance.now();
    let frame = 0;
    let finishFrame = 0;
    const tick = (now: number) => {
      const time = Math.min((now - startedAt) / 1000, playbackRecording.duration_s);
      setPlaybackTime(time);
      if (time >= playbackRecording.duration_s) {
        finishFrame = requestAnimationFrame(() => {
          playbackActiveRef.current = false;
          setPlaybackRecording(null);
          setNotice(`Playback complete · ${playbackRecording.duration_s.toFixed(2)} s at realtime speed`);
        });
        return;
      }
      frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => { cancelAnimationFrame(frame); cancelAnimationFrame(finishFrame); };
  }, [playbackRecording]);

  const handleFrame = useCallback((metrics: RendererFrameMetrics, size: string) => {
    const now = performance.now();
    const wallFrame_ms=presentationClockRef.current===null?0:now-presentationClockRef.current;presentationClockRef.current=now;setFrameMs(wallFrame_ms);setResolution(size);
    if (now - sampleClockRef.current > 250) {
      sampleClockRef.current = now;
      const gpuState=gpuInfoRef.current,gpu=gpuState?.gpuTimings,snapshot:PerformanceSnapshot={cpuSimulation_ms:cpuSimulationMsRef.current,cpuFrame_ms:metrics.cpuFrame_ms,wallFrame_ms,cpuPhysicsSubmit_ms:metrics.cpuPhysicsSubmit_ms,cpuCommandEncode_ms:gpuState?.cpuCommandEncode_ms??0,cpuQueueSubmit_ms:gpuState?.cpuQueueSubmit_ms??0,cpuDataUpload_ms:metrics.cpuDataUpload_ms,cpuRenderEncode_ms:metrics.cpuRenderEncode_ms,cpuTopology_ms:gpuState?.cpuRegrid_ms??0,gpuAdvection_ms:gpu?.advection_ms??0,gpuControl_ms:gpu?.control_ms??0,gpuPressure_ms:gpu?.pressure_ms??0,gpuProjection_ms:gpu?.projection_ms??0,gpuRigid_ms:gpu?.rigidCoupling_ms??0,gpuDiagnostics_ms:gpu?.diagnostics_ms??0,gpuOverhead_ms:gpu?.overhead_ms??0,gpuRender_ms:metrics.gpuRender_ms??0,gpuQueueLatency_ms:gpuState?.queueLatency_ms??0,simulationLag_ms:(gpuState?.simulationLag_s??0)*1000,simulationThroughput_x:gpuState?.simulationThroughput_x??0,blockedFrames:gpuState?.blockedFrames??0,gpuComputeSampleAvailable:Boolean(gpu),gpuRenderSampleAvailable:metrics.gpuRender_ms!=null};
      performanceRef.current=snapshot;setPerformanceSnapshot(snapshot);setPerformanceHistory((current)=>[...current.slice(-119),snapshot]);
      const oracle=fluidSolverRef.current?.diagnostics;setSamples((current) => [...current.slice(-79), { t: now / 1000, frame_ms: wallFrame_ms, volume_drift_pct: (oracle?.markerVolumeDrift??gpuInfoRef.current?.volumeDrift??0) * 100, constraint_error: oracle?.divergenceAfter_s??gpuInfoRef.current?.divergenceMax_s??0, kinetic_energy_J: oracle?.kineticEnergy_J??0 }]);
    }
  }, []);
  const handleGPUInfo=useCallback((info:GPUEulerianInfo)=>{gpuInfoRef.current=info;setGPUInfo(info);},[]);
  const handleGPUStatus=useCallback((status:GPUStatus)=>{setGPUStatus(status);if(status.state==="ready"&&!status.computeAvailable){simulationTimeRef.current=0;accumulatorRef.current=0;setSimulationTime(0);setRunState("paused");setNotice("WebGPU compute did not execute · simulation paused · CPU reference is opt-in");}},[]);
  const handleGPURigidLoads = useCallback((loads: GPURigidLoad[]) => {
    if (playbackActiveRef.current) return;
    gpuRigidLoadsRef.current = mergeGPURigidLoads(gpuRigidLoadsRef.current, loads);
  }, []);
  const handleBodyDrag = useCallback((bodyId: string, position: RigidBodyState["position_m"], velocity: RigidBodyState["linearVelocity_m_s"], phase: BodyDragPhase) => {
    if (playbackActiveRef.current) return;
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
  const resetSimulation = (source = scene,targetBackend:SimulationBackend=backend,withOracle=liveCPUOracle) => {
    const next = initializeRigidBodies(source.rigidBodies);
    setRuntimeBodies(next, source.fluid.gravity_m_s2);
    fluidSolverRef.current=targetBackend==="cpu-reference"||withOracle?new EulerianFluidSolver(source):null;
    setFluidState(fluidSolverRef.current?.diagnostics??emptyFluidState);
    setFluidRenderState(fluidSolverRef.current?.getRenderState()??emptyFluidRenderState);
    simulationTimeRef.current = 0; accumulatorRef.current = 0;
    cpuOracleStepRef.current = 0;
    cpuSimulationMsRef.current=0;performanceRef.current=emptyPerformance;setPerformanceSnapshot(emptyPerformance);setPerformanceHistory([]);
    gpuRigidLoadsRef.current = []; kinematicDragRef.current = null;
    recordingRef.current = createSimulationRecording(source,targetBackend,gpuQuality,next,fluidSolverRef.current?.getRenderState()??emptyFluidRenderState);
    playbackActiveRef.current = false;
    setRecordingDuration(0); setPlaybackRecording(null); setPlaybackTime(0);
    setSimulationTime(0); setRunState("paused"); setSamples([]);
    setSelectedBodyId(source.rigidBodies[0]?.id);
    setNotice(`${source.fluid.initialCondition === "dam-break" ? "Dam-break" : "Tank fill"} reset at t = 0`);
  };
  const loadScenario=(id:ScenarioId)=>{const next=createInteractiveScenario(id);setActiveScenario(id);setScene(next);setGPUInfo(null);setCamera(defaultCamera);resetSimulation(next);setNotice(`${SCENARIOS.find((preset)=>preset.id===id)?.name} ready · uniform interactive grid · press play`);};
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
    const payload = { manifest: createRunManifest(scene, adapter), backend, gpuQuality, gpuInfo, shellMetrics: { presentationFrame_ms: frameMs, canvasResolution: resolution, samples, performance: performanceSnapshot, performanceHistory }, rigidBodyState: bodies, rigidBodyDiagnostics: rigidState, eulerianMetrics: fluidState, couplingMetrics: couplingState };
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
    setPlaybackRecording(null);
    setRunState("paused");
    if(backend==="webgpu"&&gpuStatus.state==="ready"&&!gpuStatus.computeAvailable){setNotice("WebGPU compute is unavailable; no simulation step was run");return;}
    const dt=backend==="webgpu"&&gpuQuality==="balanced"?scene.numerics.maxDt_s:scene.numerics.fixedDt_s;
    if (simulationTimeRef.current + dt > scene.duration_s) return;
    const activeFluid = fluidSolverRef.current, simulationBodies = RIGID_BODIES_ENABLED ? bodiesRef.current : [];
    const gpuCoupling = backend === "webgpu" ? externalLoadsFromGPU(scene, RIGID_BODIES_ENABLED ? gpuRigidLoadsRef.current : [], dt) : undefined;
    if(!gpuCoupling&&!activeFluid){setNotice("CPU reference is off; select it explicitly before stepping");return;}
    const coupling = gpuCoupling ? undefined : computeFluidLoads(scene, activeFluid!, simulationBodies);
    const couplingDiagnostics = gpuCoupling?.diagnostics ?? applyFluidReactions(activeFluid!, simulationBodies, coupling!.loads, dt);
    const diagnostics = advanceRigidBodies(simulationBodies, scene, dt, 6, gpuCoupling?.loads ?? coupling!.loads);
    const fluidDiagnostics = backend==="cpu-reference"?activeFluid!.step(dt):undefined;
    simulationTimeRef.current += dt;
    setSimulationTime(simulationTimeRef.current);
    setBodies(cloneRigidBodies(bodiesRef.current)); setRigidState(diagnostics);
    if(fluidDiagnostics){setFluidState(fluidDiagnostics);setFluidRenderState(activeFluid!.getRenderState());}
    setCouplingState(couplingDiagnostics);
    captureSimulationFrame(true);
  };

  const startPlayback = () => {
    captureSimulationFrame(true);
    const recording = recordingRef.current;
    if (!recording || recording.duration_s <= 0) return;
    setRunState("paused");
    accumulatorRef.current = 0;
    playbackActiveRef.current = true;
    setPlaybackTime(0);
    setPlaybackRecording(recording);
    setNotice(`Playing latest capture · ${recording.duration_s.toFixed(2)} s at 1× realtime`);
  };

  const stopPlayback = () => {
    playbackActiveRef.current = false;
    setPlaybackRecording(null);
    setPlaybackTime(0);
    setNotice("Playback stopped · live simulation restored");
  };

  const playbackFrame = playbackRecording ? simulationFrameAt(playbackRecording, playbackTime) : null;
  const displayedScene = playbackRecording?.scene ?? scene;
  const displayedTime = playbackRecording ? playbackTime : simulationTime;
  const displayedBodies = playbackFrame?.bodies ?? bodies;
  const displayedFluid = playbackFrame?.fluid ?? fluidRenderState;
  const displayedBackend = playbackRecording?.backend ?? backend;
  const displayedQuality = playbackRecording?.quality ?? gpuQuality;

  const estimatedCells = Math.ceil(scene.container.width_m / scene.nominalResolution.length_m) * Math.ceil(scene.container.height_m / scene.nominalResolution.length_m) * Math.ceil(scene.container.depth_m / scene.nominalResolution.length_m);
  return (
    <main className="lab-shell" data-run-state={runState} data-playback-state={playbackRecording ? "playing" : "live"} data-solver-mode="eulerian" data-simulation-time={displayedTime.toFixed(6)} data-body-count={RIGID_BODIES_ENABLED ? displayedBodies.length : 0}>
      <header className="topbar">
        <div className="brand"><span className="brand-mark">FL</span><div><strong>Fluid Lab</strong><small>WEBGPU CFD WORKBENCH</small></div></div>
        <div className="solver-identity">Eulerian VOF</div>
        <div className="top-actions">
          <button className="quiet-button" onClick={() => {setValidationResults(runShellValidation());setValidationOpen(true);}}><span className={`status-dot ${validationResults?.every((result) => result.passed) ? "online" : "idle"}`} />Validation</button>
          <button className="quiet-button" onClick={saveScene}>Save scene</button>
          <button className="primary-button" onClick={exportMetrics}>Export run</button>
        </div>
      </header>

      <aside className="left-panel panel-scroll">
        <section className="panel-section scenario-launcher">
          <div className="section-heading"><h2>Choose a scenario</h2><span>reproducible presets</span></div>
          <div className="scenario-grid" aria-label="Simulation scenarios">{SCENARIOS.map((preset)=><button key={preset.id} className={activeScenario===preset.id?"active":""} aria-pressed={activeScenario===preset.id} onClick={()=>loadScenario(preset.id)}><strong>{preset.name}</strong><span>{preset.description}</span><small>{preset.stress}</small></button>)}</div>
          <div className="prime-controls"><button className="primary-button" disabled={gpuComputeBlocked} onClick={()=>setRunState((state)=>state==="running"?"paused":"running")}>{runState==="running"?"Pause":"Run scenario"}</button><button onClick={()=>resetSimulation()}>Reset</button></div>
        </section>
        <section className="panel-section core-compute">
          <div className="section-heading"><h2>Grid mode</h2><span>primary comparison</span></div>
          <label className="select-control"><span>Hierarchy depth</span><select aria-label="Hierarchy depth" value={scene.hierarchy.levels} onChange={(event) => { const levels=Number(event.target.value);setScene((current) => ({ ...current, hierarchy: { ...current.hierarchy, levels, minimumFluidLevel: Math.min(current.hierarchy.minimumFluidLevel, levels - 1) } }));setSimulationTime(0);simulationTimeRef.current=0;setRunState("paused"); }}><option value={1}>1 · Uniform grid</option><option value={2}>2 · One adaptive split</option><option value={3}>3 · Balanced adaptive</option><option value={4}>4 · Fine adaptive</option><option value={5}>5 · Maximum depth</option></select></label>
          <label className="select-control"><span>GPU quality</span><select aria-label="GPU quality" value={gpuQuality} onChange={(event) => { setGPUQuality(event.target.value as GPUQuality); setSimulationTime(0); simulationTimeRef.current = 0; setRunState("paused"); }}><option value="balanced">Balanced · interactive</option><option value="high">High · detailed</option><option value="ultra">Ultra · maximum</option></select></label>
          <div className="estimate-grid"><div><small>Uniform equivalent</small><strong>{(gpuInfo?.equivalentUniformCells ?? estimatedCells).toLocaleString()}</strong><span>cells</span></div>{gpuInfo?.activeBrickCount != null&&<div><small>Active hierarchy</small><strong>{gpuInfo.cellCount.toLocaleString()}</strong><span>{((gpuInfo.compressionRatio??1)*100).toFixed(1)}% · {gpuInfo.activeBrickCount.toLocaleString()} bricks</span></div>}</div>
        </section>
        <details className="control-disclosure">
          <summary><span>Scene &amp; fluid setup</span><small>container, water, files</small></summary>
          <div className="disclosure-body">
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
          <div className="segmented compact" aria-label="Fluid wall condition"><button className={scene.container.fluidWallMode === "no-slip" ? "active" : ""} onClick={() => patchContainer({ fluidWallMode: "no-slip" })}>No slip</button><button className={scene.container.fluidWallMode === "free-slip" ? "active" : ""} onClick={() => patchContainer({ fluidWallMode: "free-slip" })}>Free slip</button></div>
        </section>
        <section className="panel-section">
          <div className="section-heading"><h2>Water</h2><span>20 °C default</span></div>
          <div className="segmented compact" aria-label="Fluid initial condition"><button className={scene.fluid.initialCondition === "dam-break" ? "active" : ""} onClick={() => patchFluid({ initialCondition: "dam-break" })}>Dam break</button><button className={scene.fluid.initialCondition === "tank-fill" ? "active" : ""} onClick={() => patchFluid({ initialCondition: "tank-fill" })}>Tank fill</button></div>
          <RangeControl label="Density" unit="kg/m³" value={scene.fluid.density_kg_m3} min={700} max={1300} step={0.1} onChange={(value) => patchFluid({ density_kg_m3: value })} displayDigits={1} />
          <RangeControl label="Dynamic viscosity" unit="Pa·s" value={scene.fluid.dynamicViscosity_Pa_s} min={0} max={0.02} step={0.000001} onChange={(value) => patchFluid({ dynamicViscosity_Pa_s: value })} displayDigits={6} />
          <RangeControl label="Surface tension" unit="N/m" value={scene.fluid.surfaceTension_N_m} min={0} max={0.15} step={0.001} onChange={(value) => patchFluid({ surfaceTension_N_m: value })} displayDigits={3} />
          <RangeControl label="Gravity Y" unit="m/s²" value={scene.fluid.gravity_m_s2.y} min={-20} max={0} step={0.01} onChange={(value) => patchFluid({ gravity_m_s2: { ...scene.fluid.gravity_m_s2, y: value } })} displayDigits={3} />
          <button className="drop-button" onClick={() => resetSimulation()}>Apply &amp; reset fluid</button>
        </section>
          </div>
        </details>
        {RIGID_BODIES_ENABLED&&<details className="control-disclosure"><summary><span>Rigid-body editor</span><small>{bodies.length}/12 bodies</small></summary><div className="disclosure-body"><section className="panel-section rigid-editor" data-testid="rigid-editor">
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
        </section></div></details>}
        <details className="control-disclosure"><summary><span>Advanced solver controls</span><small>backend, tolerances, halos</small></summary><div className="disclosure-body"><section className="panel-section">
          <div className="section-heading"><h2>Solver tuning</h2><span>advanced</span></div>
          <div className="segmented compact" aria-label="Simulation backend"><button disabled={gpuStatus.state==="ready"&&!gpuStatus.computeAvailable} className={backend === "webgpu" ? "active" : ""} onClick={() => {setBackend("webgpu");resetSimulation(scene,"webgpu",liveCPUOracle);setNotice("WebGPU compute selected; fields reset");}}>WebGPU</button><button className={backend === "cpu-reference" ? "active" : ""} onClick={() => {setBackend("cpu-reference");setLiveCPUOracle(false);resetSimulation(scene,"cpu-reference",false);setNotice("CPU binary64 reference explicitly enabled · fields reset");}}>CPU reference</button></div>
          <label className="check-row"><input type="checkbox" checked={liveCPUOracle} onChange={(event)=>{const enabled=event.target.checked;setLiveCPUOracle(enabled);resetSimulation(scene,backend,enabled);setNotice(enabled?"Live CPU comparison explicitly enabled · fields reset":"Live CPU comparison disabled");}}/><span>Live CPU comparison oracle (opt in)</span></label>
          <RangeControl label="Nominal length" unit="m" value={scene.nominalResolution.length_m} min={0.0125} max={0.08} step={0.0025} onChange={(value) => setScene((current) => ({ ...current, nominalResolution: { length_m: value } }))} displayDigits={4} />
          <RangeControl label="Interface halo" unit="cells" value={scene.hierarchy.interfaceHaloCells} min={1} max={8} step={1} onChange={(value) => setScene((current) => ({ ...current, hierarchy: { ...current.hierarchy, interfaceHaloCells: value } }))} displayDigits={0} />
          <RangeControl label="Solid halo" unit="cells" value={scene.hierarchy.solidHaloCells} min={1} max={8} step={1} onChange={(value) => setScene((current) => ({ ...current, hierarchy: { ...current.hierarchy, solidHaloCells: value } }))} displayDigits={0} />
          <RangeControl label="Pressure iterations" unit="iterations" value={scene.numerics.pressureMaxIterations} min={20} max={1000} step={20} onChange={(value) => setScene((current) => ({ ...current, numerics: { ...current.numerics, pressureMaxIterations: value } }))} displayDigits={0} />
          {gpuInfo?.activeBrickCount!=null&&<div className="advanced-readout">Topology revision {gpuInfo.topologyRevision??0} · {gpuInfo.regridCount??0} refinement checks</div>}
        </section></div></details>
      </aside>

      <section className="viewport-shell">
        <WebGPUViewport scene={displayedScene} camera={camera} setCamera={setCamera} view={view} simulationTime={displayedTime} bodies={RIGID_BODIES_ENABLED ? displayedBodies : []} selectedBodyId={RIGID_BODIES_ENABLED ? selectedBodyId : undefined} fluid={displayedFluid} backend={displayedBackend} quality={displayedQuality} onFrame={handleFrame} onGPUStatus={handleGPUStatus} onGPUInfo={handleGPUInfo} onGPURigidLoads={handleGPURigidLoads} onSelectBody={setSelectedBodyId} onDragBody={handleBodyDrag} />
        <div className="viewport-topline">
          <div className={`gpu-badge state-${gpuStatus.state}`}><span className={`status-dot ${gpuStatus.state === "ready" ? "online" : "warning"}`} /><strong>{gpuStatus.state === "ready" ? "WEBGPU" : gpuStatus.state.toUpperCase()}</strong><span>{gpuStatus.label}</span></div>
          <div className="segmented"><button className={view === "scientific" ? "active" : ""} onClick={() => setView("scientific")}>Scientific</button><button className={view === "presentation" ? "active" : ""} onClick={() => setView("presentation")}>Presentation</button></div>
        </div>
          <div className="physics-stage-badge"><strong>HIERARCHICAL WEBGPU</strong><span>{backend === "webgpu" ? `VOF · composite MAC · strong rigid coupling · L${gpuInfo?.hierarchyLevels ?? scene.hierarchy.levels}` : "CPU validation oracle active"}</span><small>{backend === "webgpu" ? `${gpuInfo?.cellCount.toLocaleString() ?? "…"} active cells · f32 · ${gpuInfo?.pressureIterationsExecuted ?? "…"}/${gpuInfo?.pressureIterations ?? "…"} ${(gpuInfo?.pressureMethod??"pressure").toUpperCase()}` : "MAC · binary64 · PCG"}</small></div>
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
        {gpuStatus.state === "ready"&&!gpuStatus.computeAvailable&&backend==="webgpu"&&<div className="compute-fallback-warning"><strong>WEBGPU COMPUTE UNAVAILABLE · PAUSED</strong><span>No CPU fallback is running · CPU reference is an advanced opt-in</span></div>}
      </section>

      <aside className="right-panel panel-scroll">
        <section className="panel-section diagnostics-head">
          <p className="eyebrow">LIVE DIAGNOSTICS</p>
          <div className="state-line"><span className={`status-dot ${playbackRecording || runState === "running" ? "online pulse" : "idle"}`} /><strong>{playbackRecording ? "PLAYBACK · REALTIME" : runState === "running" ? "COUPLED RUNNING" : "PAUSED"}</strong><span>Eulerian</span></div>
        </section>
        <section className="metric-grid panel-section">
          <MetricCard label={playbackRecording ? "Playback time" : "Simulation time"} value={displayedTime.toFixed(3)} unit="s" />
          <MetricCard label={backend==="webgpu"?"GPU grid":"Fluid grid"} value={backend==="webgpu"?(gpuInfo?`${gpuInfo.nx} × ${gpuInfo.ny} × ${gpuInfo.nz}`:"initializing"):`${fluidRenderState.nx} × ${fluidRenderState.ny} × ${fluidRenderState.nz}`} unit={backend==="webgpu"&&gpuInfo?`${(gpuInfo.allocatedBytes/1048576).toFixed(1)} MiB physics`:backend==="cpu-reference"?"CPU binary64 oracle":undefined} tone="good" />
          <MetricCard label="Max speed" value={backend==="webgpu"?(gpuInfo?.maxSpeed_m_s?.toFixed(3)??"—"):fluidState.maxSpeed_m_s.toFixed(3)} unit={backend==="webgpu"?`m/s · ${gpuInfo?.encodedSteps??0} encoded steps`:"m/s · CPU oracle"} />
          <MetricCard label="Volume drift" value={backend==="webgpu"?(gpuInfo?.volumeDrift!==undefined?(gpuInfo.volumeDrift*100).toFixed(2):"—"):(fluidState.markerVolumeDrift*100).toFixed(2)} unit="% · unmodified fluid volume" tone={(backend==="webgpu"?Math.abs(gpuInfo?.volumeDrift??Infinity):Math.abs(fluidState.markerVolumeDrift))<.01?"good":"warn"} />
        </section>
        <details className="control-disclosure diagnostics-disclosure"><summary><span>Detailed diagnostics</span><small>pressure, bodies, invariants</small></summary><div className="disclosure-body">
        <section className="metric-grid panel-section">
          <MetricCard label="Fixed validation dt" value={scene.numerics.fixedDt_s.toFixed(4)} unit="s" />
          <MetricCard label="Presentation interval" value={frameMs.toFixed(2)} unit="ms wall · includes GPU backpressure" tone={frameMs < 20 ? "good" : "warn"} />
          {RIGID_BODIES_ENABLED&&<MetricCard label="Rigid bodies" value={String(bodies.length)} unit={`${rigidState.contactCount} contact solves`} />}
          {(backend==="cpu-reference"||liveCPUOracle)&&<><MetricCard label="CPU oracle grid" value={`${fluidRenderState.nx} × ${fluidRenderState.ny} × ${fluidRenderState.nz}`} unit={`${fluidState.pressureIterations} PCG iterations`} tone={fluidState.pressureConverged?"good":"warn"} /><MetricCard label="CPU / GPU front" value={`${fluidState.damFront_m.toFixed(3)} / ${gpuInfo?.front_m?.toFixed(3)??"—"}`} unit="m" /></>}
          <MetricCard label="GPU pressure / finite" value={gpuInfo?.pressureMax_Pa!==undefined?gpuInfo.pressureMax_Pa.toFixed(1):"—"} unit={`Pa max · pre-div ${gpuInfo?.divergenceBefore_s?.toExponential(1)??"—"} · ${gpuInfo?.nanCount??0} non-finite`} tone={(gpuInfo?.nanCount??0)===0?"good":"warn"} />
          <MetricCard label="GPU step" value={gpuInfo?.gpuStep_ms!==undefined?gpuInfo.gpuStep_ms.toFixed(2):"—"} unit={`ms timestamp · ${gpuInfo?.substepsLast??0} substeps · ${gpuInfo?.queuedSubmissions??0} queued`} />
          <MetricCard label="GPU delivery" value={gpuInfo?.simulationThroughput_x?`${gpuInfo.simulationThroughput_x.toFixed(2)}×`:"—"} unit={`${gpuInfo?.queueLatency_ms?.toFixed(1)??"—"} ms queue · ${((gpuInfo?.simulationLag_s??0)*1000).toFixed(1)} ms behind`} tone={(gpuInfo?.simulationThroughput_x??0)>=.9&&((gpuInfo?.simulationLag_s??0)<.05)?"good":"warn"} />
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
          <div className="section-heading"><h2>Presentation frame</h2><span>wall interval · ms</span></div>
          <Sparkline samples={samples} />
          <div className="chart-legend"><span><i className="legend-teal" />presentation interval</span><span>physics GPU {gpuInfo?.gpuStep_ms?.toFixed(2)??"—"} ms · {gpuInfo?.simulationThroughput_x?.toFixed(2)??"—"}×</span></div>
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
        {(backend==="cpu-reference"||liveCPUOracle)&&<section className="panel-section fluid-pending">
          <div className="section-heading"><h2>Eulerian fluid</h2><span>CPU binary64 reference</span></div>
          <div className="invariant-list">
            <div><span>RMS divergence</span><strong>{fluidState.divergenceAfter_s.toExponential(2)}</strong><small>s⁻¹ · before {fluidState.divergenceBefore_s.toExponential(2)}</small></div>
            <div><span>PCG relative residual</span><strong>{fluidState.pressureRelativeResidual.toExponential(2)}</strong><small>{fluidState.pressureIterations} iterations · {fluidState.pressureConverged ? "converged" : "not converged"}</small></div>
            <div><span>Marker volume drift</span><strong>{(fluidState.markerVolumeDrift * 100).toExponential(2)}</strong><small>% · marker mass exactly conserved</small></div>
            <div><span>Kinetic energy</span><strong>{fluidState.kineticEnergy_J.toFixed(2)}</strong><small>J</small></div>
            <div><span>Time-step bound</span><strong>{fluidState.limitingCondition}</strong><small>dt {fluidState.dt_s.toFixed(4)} s</small></div>
            <div><span>NaN / infinity</span><strong>{fluidState.nanCount}</strong><small>acceptance = 0</small></div>
          </div>
          <p>CPU oracle: staggered MAC, RK2 semi-Lagrangian advection, explicit viscosity, marker free surface, closed-wall flux enforcement, and matrix-free Jacobi-PCG projection. The WebGPU path uses sparse 2:1 leaf bricks, conservative bounded VOF transfer, composite coarse–fine face fluxes, matrix-free PCG projection, and pressure-level rigid-body coupling.</p>
        </section>}
        {RIGID_BODIES_ENABLED && <section className="panel-section">
          <div className="section-heading"><h2>Fluid–rigid exchange</h2><span>two-way impulses</span></div>
          <div className="invariant-list"><div><span>Displaced volume</span><strong>{couplingState.displacedVolume_m3.toExponential(2)}</strong><small>m³</small></div><div><span>Coupled bodies</span><strong>{couplingState.coupledBodyCount}</strong><small>of {bodies.length}</small></div><div><span>Momentum closure</span><strong>{couplingState.momentumClosureError_N_s.toExponential(2)}</strong><small>N·s</small></div></div>
        </section>}
        <section className="panel-section">
          <div className="section-heading"><h2>Run identity</h2><span>reproducibility</span></div>
          <dl className="run-identity"><div><dt>Build</dt><dd>{BUILD_ID}</dd></div><div><dt>Active backend</dt><dd>{backend}</dd></div><div><dt>Eulerian GPU</dt><dd>f32 hierarchical composite MAC PCG</dd></div><div><dt>Eulerian CPU</dt><dd>{backend==="cpu-reference"||liveCPUOracle?"binary64 MAC PCG oracle":"off · opt in only"}</dd></div><div><dt>Random seed</dt><dd>{scene.randomSeed}</dd></div></dl>
        </section>
        </div></details>
      </aside>

      <footer className="transport-bar">
        <div className="transport-controls">
          <button className="transport-main" disabled={Boolean(playbackRecording)||gpuComputeBlocked} onClick={() => setRunState((state) => state === "running" ? "paused" : "running")} aria-label={runState === "running" ? "Pause simulation" : "Play simulation"}>{runState === "running" ? "Ⅱ" : "▶"}</button>
          <button disabled={Boolean(playbackRecording)||gpuComputeBlocked} onClick={singleRigidStep} aria-label="Single fluid clock step">STEP</button>
          <button onClick={() => resetSimulation()}>RESET</button>
          <button className={playbackRecording ? "active" : ""} disabled={!playbackRecording && recordingDuration <= 0} onClick={playbackRecording ? stopPlayback : startPlayback} aria-label={playbackRecording ? "Stop playback" : "Play latest simulation capture from the start"}>{playbackRecording ? "STOP" : "REPLAY"}</button>
          <button className={performanceOpen?"active":""} onClick={()=>setPerformanceOpen((open)=>!open)} aria-expanded={performanceOpen} aria-controls="performance-drawer">PERF</button>
        </div>
        <div className="time-readout"><span>{playbackRecording ? "replay" : "t"}</span><strong>{displayedTime.toFixed(4)}</strong><small>s</small><div className="timeline"><i style={{ width: `${(displayedTime / (playbackRecording?.duration_s ?? scene.duration_s)) * 100}%` }} /></div><span>{(playbackRecording?.duration_s ?? scene.duration_s).toFixed(1)} s</span></div>
        <div className="file-actions"><span className="notice">{notice}</span><button onClick={loadLocal}>Load</button><button onClick={() => fileRef.current?.click()}>Import</button><input ref={fileRef} type="file" accept="application/json,.json" onChange={importScene} hidden /></div>
      </footer>

      {performanceOpen&&<PerformanceDrawer snapshot={performanceSnapshot} history={performanceHistory} onClose={()=>setPerformanceOpen(false)} timestampsSupported={gpuStatus.state==="ready"&&gpuStatus.timestampQueriesAvailable} computeTimestampsEnabled={gpuInfo?.timestampSamplingEnabled??false} renderTimestampsEnabled={gpuStatus.state==="ready"&&gpuStatus.renderTimestampSamplingEnabled}/>}

      {validationOpen&&validationResults&&<ValidationPanel results={validationResults} onClose={() => setValidationOpen(false)} gpuStatus={gpuStatus} />}
    </main>
  );
}
