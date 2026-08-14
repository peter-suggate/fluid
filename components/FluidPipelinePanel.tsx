"use client";

import { useEffect, useMemo, useState, type ReactNode } from "react";
import { useDiagnosticsStore } from "../lib/core/stores/diagnostics-store";
import { useMethodStore, resolvedMethodValues } from "../lib/core/stores/method-store";
import { useRuntimeStore } from "../lib/core/stores/runtime-store";
import { useSceneStore } from "../lib/core/stores/scene-store";
import { useUIStore } from "../lib/core/stores/ui-store";
import { usePerformanceInstrumentationStore } from "../lib/core/stores/performance-instrumentation-store";
import {
  averagePerformanceTraces,
  performanceTraceMatchesLane,
  type PerformanceTrace,
} from "../lib/core/performance-trace";
import {
  fluidPipelinePhaseCosts,
  type FluidPipelineContext,
  type FluidPipelineGraph,
  type FluidStageControl,
} from "../lib/core/fluid-pipeline";
import { getMethod } from "@/lib/core/method-registry";
import { simulation } from "../lib/core/simulation/controller";
import { sceneHasTerrain } from "../lib/core/terrain";
import { FluidPipeline } from "./FluidPipeline";
import { PipeChoice, PipeRange, PipeToggle } from "./PipeControls";

/** Advances the live readout averages over. */
const TRACE_WINDOW = 12;

/**
 * The averaged physics lane, split by honesty.
 *
 * `stages` is the best exact, exclusive partition of the authoritative
 * advance. GPU solvers prefer hardware timestamp boundaries. A host-authority
 * solver instead publishes its semantic CPU boundary chain under the same
 * capture identity as the queue-wall completion; the controller has already
 * joined those traces before they reach this report. A bare queue-wall sample
 * remains a real total but cannot put figures on individual stage nodes.
 */
function usePhysicsTiming(methodId: string): {
  readonly total?: PerformanceTrace;
  readonly stages?: PerformanceTrace;
} {
  const reports = useDiagnosticsStore((state) => state.performanceReports);
  return useMemo(() => {
    const newest = reports.findLast((report) => report.methodId === methodId && report.physics);
    if (!newest) return {};
    const recent = reports
      .filter((report) => report.methodId === methodId && report.context === newest.context)
      .slice(-TRACE_WINDOW);
    const physics = recent.map((report) => report.physics)
      .filter((trace): trace is PerformanceTrace => trace !== undefined);
    const latest = physics.at(-1);
    // The averager refuses mixed lanes, and a window straddling the hardware →
    // queue-wall degradation would be one: average only the newest source.
    const matching = latest
      ? physics.filter((trace) => trace.measurementSource === latest.measurementSource)
      : [];
    const physicsMean = averagePerformanceTraces(matching);
    if (physicsMean?.measurementSource === "gpu-hardware-timestamp") {
      return { total: physicsMean, stages: physicsMean };
    }

    // `performanceReportCPUTrace` only admits a CPU trace here when its sample,
    // context and stable capture frame all match this completed physics trace.
    // Recheck the observable half of that contract locally so a future report
    // producer cannot accidentally turn an animation/render tick into solver
    // stage timing.
    const cpu = recent.flatMap((report) => {
      const trace = report.cpu;
      const completed = report.physics;
      if (!trace || !completed
        || !performanceTraceMatchesLane(trace, "cpu", "main-thread")
        || trace.sampleId !== completed.sampleId) return [];
      const suffix = ":queue-wall-fallback";
      const physicsContext = completed.context.endsWith(suffix)
        ? completed.context.slice(0, -suffix.length)
        : completed.context;
      return trace.context === physicsContext ? [trace] : [];
    });
    const cpuMean = averagePerformanceTraces(cpu);
    return cpuMean
      ? { total: cpuMean, stages: cpuMean }
      : { total: physicsMean, stages: undefined };
  }, [reports, methodId]);
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
  const [loadedGraph, setLoadedGraph] = useState<{
    readonly methodId: string;
    readonly graph: FluidPipelineGraph;
  } | undefined>(undefined);
  useEffect(() => {
    let cancelled = false;
    void getMethod(methodId).pipelineGraph?.().then((loaded) => {
      if (!cancelled) setLoadedGraph({ methodId, graph: loaded });
    });
    return () => { cancelled = true; };
  }, [methodId]);
  const graph = loadedGraph?.methodId === methodId ? loadedGraph.graph : undefined;

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

  const timing = usePhysicsTiming(methodId);
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
  // method panel would. Runtime-safe parameters reach the attached solver on
  // the next frame; structural controls take the controller's rebuild path.
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
  const liveTuning = Boolean(method.runtimeParamKeys?.length);
  const advanceLabel = totalTrace
    ? `${totalTrace.total_ms.toFixed(2)} ms/advance`
    : "no trace yet";
  const sourceLabel = !totalTrace
    ? "Waiting for a sampled advance."
    : totalTrace.measurementSource === "gpu-hardware-timestamp"
      ? `Hardware timestamp boundary chain: an exact, exclusive partition of the advance, so the trunk figures sum to the total. ${TRACE_WINDOW}-sample mean.`
      : totalTrace.measurementSource === "cpu-active-wall"
        ? `Authoritative CPU boundary chain: an exact, exclusive partition of the host solve, so the trunk figures sum to the total. ${TRACE_WINDOW}-sample mean.`
        : `GPU queue-wall observation: the whole advance is timed but no semantic boundary partition is available, so the trunk stays unmeasured. ${TRACE_WINDOW}-sample mean.`;

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
        hint={`Samples the method-owned advance boundary chain on a cadence without changing its algorithmic schedule.\n\n${sourceLabel}`} />
      <output title={liveTuning
        ? "Stage gates and tuning controls apply to the attached solver without resetting time. Pressure schedule controls are marked separately and rebuild their precomputed dispatch plan."
        : "The solver's lattice; every pass in this diagram dispatches over it."}>
        {liveTuning ? "TUNING LIVE · " : ""}GRID {grid}
      </output>
    </div>

    {graph
      ? <FluidPipeline graph={graph} context={context} costs={costs}
        total_ms={total_ms} measured={measured} controls={controls}
        measurementDomain={stageTrace?.domain}
        onToggleStage={toggleStage} />
      : <p className="render-inline-warning">
        The {method.label} method has not declared an advance pipeline yet. Select the uniform
        reference method to see its stage graph.
      </p>}
  </aside>;
}
