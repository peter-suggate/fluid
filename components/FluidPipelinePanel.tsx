"use client";

import { useEffect, useMemo, useState, type ReactNode } from "react";
import { useDiagnosticsStore } from "@/lib/stores/diagnostics-store";
import { useMethodStore, resolvedMethodValues } from "@/lib/stores/method-store";
import { useRuntimeStore } from "@/lib/stores/runtime-store";
import { useSceneStore } from "@/lib/stores/scene-store";
import { useUIStore } from "@/lib/stores/ui-store";
import { usePerformanceInstrumentationStore } from "@/lib/stores/performance-instrumentation-store";
import { averagePerformanceTraces, type PerformanceTrace } from "@/lib/performance-trace";
import {
  fluidPipelinePhaseCosts,
  type FluidPipelineContext,
  type FluidPipelineGraph,
  type FluidStageControl,
} from "@/lib/fluid-pipeline";
import { loadFluidPipeline } from "@/lib/fluid-pipelines";
import { getMethod } from "@/lib/methods";
import { simulation } from "@/lib/simulation/controller";
import { sceneHasTerrain } from "@/lib/terrain";
import { FluidPipeline } from "./FluidPipeline";
import { PipeChoice, PipeRange, PipeToggle } from "./PipeControls";

/** Advances the live readout averages over. */
const TRACE_WINDOW = 12;

/**
 * The averaged physics lane, split by honesty.
 *
 * `stages` is only ever the hardware boundary-chain partition — an exact,
 * exclusive split of the advance whose phases the graph's seam labels name.
 * When a device cannot produce one, the solver publishes the queue-wall
 * observation instead; that is a real advance total but a single number, so it
 * feeds the header as `total` and the trunk stays honestly unmeasured.
 */
function usePhysicsTiming(): {
  readonly total?: PerformanceTrace;
  readonly stages?: PerformanceTrace;
} {
  const reports = useDiagnosticsStore((state) => state.performanceReports);
  return useMemo(() => {
    const newest = reports.findLast((report) => report.physics);
    if (!newest) return {};
    const recent = reports
      .filter((report) => report.context === newest.context)
      .slice(-TRACE_WINDOW)
      .map((report) => report.physics)
      .filter((trace): trace is PerformanceTrace => trace !== undefined);
    const latest = recent.at(-1);
    // The averager refuses mixed lanes, and a window straddling the hardware →
    // queue-wall degradation would be one: average only the newest source.
    const matching = latest
      ? recent.filter((trace) => trace.measurementSource === latest.measurementSource)
      : [];
    const mean = averagePerformanceTraces(matching);
    const hardware = mean?.measurementSource === "gpu-hardware-timestamp";
    return { total: mean, stages: hardware ? mean : undefined };
  }, [reports]);
}

export function FluidPipelinePanel() {
  const setRightPanel = useUIStore((state) => state.setRightPanel);
  const methodId = useMethodStore((state) => state.methodId);
  const quality = useMethodStore((state) => state.quality);
  const overrides = useMethodStore((state) => state.overrides);
  const info = useDiagnosticsStore((state) => state.gpuInfo);
  const bodyCount = useDiagnosticsStore((state) => state.bodies.length);
  const scene = useSceneStore((state) => state.scene);
  const running = useRuntimeStore((state) => state.runState === "running");

  const method = getMethod(methodId);
  const values = useMemo(
    () => resolvedMethodValues({ methodId, quality, overrides }),
    [methodId, quality, overrides]);

  // Graphs are declared beside their encoders, so loading one loads the solver
  // module; async keeps an unsupported method from paying for that.
  const [graph, setGraph] = useState<FluidPipelineGraph | undefined>(undefined);
  useEffect(() => {
    let cancelled = false;
    setGraph(undefined);
    void loadFluidPipeline(methodId).then((loaded) => {
      if (!cancelled) setGraph(loaded);
    });
    return () => { cancelled = true; };
  }, [methodId]);

  const [liveTiming, setLiveTiming] = useState(true);

  // Physics trace samples only reach the diagnostics store while
  // instrumentation records. The panel opts in for as long as it is open and
  // hands the setting back exactly as it found it, never overriding an
  // explicit choice made in the performance panel (the guard on `off`).
  useEffect(() => {
    if (!liveTiming) return;
    const store = usePerformanceInstrumentationStore.getState();
    if (store.mode !== "off") return;
    store.setMode("timeline");
    return () => {
      const current = usePerformanceInstrumentationStore.getState();
      if (current.mode === "timeline") current.setMode("off");
    };
  }, [liveTiming]);

  const timing = usePhysicsTiming();
  const stageTrace = liveTiming ? timing.stages : undefined;
  const totalTrace = liveTiming ? timing.total : undefined;
  const costs = useMemo(() => fluidPipelinePhaseCosts(stageTrace), [stageTrace]);
  const total_ms = stageTrace?.total_ms ?? 0;
  const measured = stageTrace !== undefined;

  const context: FluidPipelineContext = useMemo(() => ({
    values,
    info,
    sceneId: scene.sceneId,
    bodyCount,
    hasTerrain: sceneHasTerrain(scene),
    hasInflow: Boolean(scene.fluid.inflow),
    running,
  }), [values, info, scene, bodyCount, running]);

  // Declarative stage controls, materialized here so the graph module stays
  // free of React. `param-*` route through the method store exactly as the
  // method panel would, so runtime-unsafe parameters rebuild the solver.
  const controls = useMemo(() => {
    if (!graph) return {};
    const rendered: Record<string, ReactNode> = {};
    const renderControl = (control: FluidStageControl, key: number): ReactNode => {
      switch (control.kind) {
        case "param-choice":
          return <PipeChoice key={key} label={control.label}
            value={String(values[control.param] ?? "")}
            options={control.options.map((option) => ({ ...option }))}
            onChange={(value) => simulation.setMethodParam(methodId, control.param, value)} />;
        case "param-range":
          return <PipeRange key={key} label={control.label} unit={control.unit}
            value={Number(values[control.param] ?? control.min)}
            min={control.min} max={control.max} step={control.step} digits={control.digits ?? 0}
            hint={control.hint}
            disabled={control.enabled ? !control.enabled(context) : false}
            onChange={(value) => simulation.setMethodParam(methodId, control.param, value)} />;
        case "readout":
          return <label key={key} className="pipe-field" title={control.hint}>
            <span>{control.label}</span>
            <output>{control.value(context)}</output>
          </label>;
      }
    };
    for (const stage of graph.stages) {
      if (!stage.controls?.length) continue;
      rendered[stage.id] = <div className="pipe-fields">
        {stage.controls.map(renderControl)}
      </div>;
    }
    return rendered;
  }, [graph, values, context, methodId]);

  const toggleStage = (stageId: string, checked: boolean) => {
    const stage = graph?.stages.find((candidate) => candidate.id === stageId);
    if (!stage?.toggle) return;
    simulation.setMethodParam(methodId, stage.toggle.param,
      checked ? stage.toggle.on : stage.toggle.off);
  };

  const grid = info ? `${info.nx}×${info.ny}×${info.nz}` : "—";
  const advanceLabel = totalTrace
    ? `${totalTrace.total_ms.toFixed(2)} ms/advance`
    : "no trace yet";
  const sourceLabel = !totalTrace
    ? "Waiting for a sampled advance."
    : totalTrace.measurementSource === "gpu-hardware-timestamp"
      ? `Hardware timestamp boundary chain: an exact, exclusive partition of the advance, so the trunk figures sum to the total. ${TRACE_WINDOW}-sample mean.`
      : `GPU queue-wall observation: the whole advance is timed but this device could not split it per stage, so the trunk stays unmeasured. ${TRACE_WINDOW}-sample mean.`;

  return <aside id="simulation-panel" className="right-panel panel-scroll performance-panel performance-v2 visual-panel"
    aria-label="Simulation pipeline" data-testid="fluid-pipeline-panel">
    <header className="performance-panel-header render-panel-header">
      <div><span>SIMULATION OBSERVATORY</span><h2>Advance pipeline</h2></div>
      <div className="performance-panel-header-actions">
        <button className="panel-close" type="button" onClick={() => setRightPanel(null)} aria-label="Close simulation panel">×</button>
      </div>
      <div className="render-status-line">
        <span className={running && info ? "online" : ""} />
        <strong data-testid="fluid-pipeline-method">{method.badge ?? method.label}</strong>
        <code data-testid="fluid-advance-cost" title={sourceLabel}>{advanceLabel}</code>
      </div>
    </header>

    <div className="render-preset-strip render-live-strip" role="group" aria-label="Live per-stage timing">
      <PipeToggle label="LIVE ms" checked={liveTiming} onChange={setLiveTiming}
        hint={`Samples the advance on a cadence with hardware timestamp boundaries spliced into passes the step already encodes, so a traced advance submits the same command graph as an untraced one.\n\n${sourceLabel}`} />
      <output title={`The solver's lattice; every pass in this diagram dispatches over it.`}>GRID {grid}</output>
    </div>

    {graph
      ? <FluidPipeline graph={graph} context={context} costs={costs}
        total_ms={total_ms} measured={measured} controls={controls}
        onToggleStage={toggleStage} />
      : <p className="render-inline-warning">
        The {method.label} method has not declared an advance pipeline yet. Select the uniform
        reference method to see its stage graph.
      </p>}
  </aside>;
}
