"use client";

import { useEffect, useMemo, useState, type ReactNode } from "react";
import { stageLensOverlayMode, type AnyStageLens } from "../lib/core/stage-lens";
import { useDiagnosticsStore } from "../lib/core/stores/diagnostics-store";
import { useMethodStore, resolvedMethodValues } from "../lib/core/stores/method-store";
import { useRuntimeStore } from "../lib/core/stores/runtime-store";
import { useSceneStore } from "../lib/core/stores/scene-store";
import { DEFAULT_GRID_OVERLAY_AXIS, useUIStore } from "../lib/core/stores/ui-store";
import { usePerformanceInstrumentationStore } from "../lib/core/stores/performance-instrumentation-store";
import {
  averagePerformanceTraces,
  performanceTraceMatchesLane,
  type PerformanceTrace,
} from "../lib/core/performance-trace";
import {
  fluidPipelinePhaseCosts,
  fluidPipelineTipText,
  measureFluidPipelineBand,
  measureFluidPipelineStage,
  type FluidPipelineContext,
  type FluidPipelineGraph,
  type FluidPipelineMeasurement,
  type FluidStageControl,
} from "../lib/core/fluid-pipeline";
import { getMethod } from "@/lib/core/method-registry";
import { simulation } from "../lib/core/simulation/controller";
import { sceneHasTerrain } from "../lib/core/terrain";
import { NumberField } from "./controls";
import { PipelineGraph, formatPipelineDuration, type PipelineBand } from "./PipelineGraph";
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

/** Why the figure on the pipe is the kind of number it is, in solver terms. */
function costExplanation(
  cost: FluidPipelineMeasurement,
  measurementDomain: "cpu" | "gpu" | undefined,
): string {
  switch (cost.kind) {
    case "withheld":
      return "0 ms — not encoded this advance. The stage's gate is closed, so this is a measured zero rather than a missing measurement.";
    case "shared":
      return `Not a seam of its own: this work runs inside ${cost.insideStage?.replace(/-/g, " ").toUpperCase()}, and ⊂ marks the figure as that stage's.`;
    case "idle":
      return "0 ms — the advance partition arrived, and this gated stage encoded nothing in the sampled advances.";
    case "structural":
      return "A decision, not a dispatch. It spends no advance time either way.";
    case "unmeasured":
      return "No phase in the advance partition names this stage yet — either no trace has arrived, or the fallback queue-wall observation is too coarse to split the advance.";
    default:
      return `${formatPipelineDuration(cost.duration_ms ?? 0)} ${measurementDomain === "cpu" ? "CPU active time" : "GPU execution time"} between this stage's trace seams, averaged across the trace window.${cost.encodedFraction !== undefined
        ? `\n\nEncoded in ${Math.round(cost.encodedFraction * 100)}% of sampled advances; the figure is the expected cost per advance, not the per-encode mean.`
        : ""}`;
  }
}

/**
 * The timing rates and budgets that belong to no single stage.
 *
 * These are the three controls the old configuration popover's Numerics tab
 * held. They land here rather than on the tank's Configure flyout because they
 * are not properties of the water: a step size, an oracle cell and an iteration
 * budget are settings of *the solve*, and this is the surface that shows what
 * the solve costs. Changing one and watching the trunk move is the whole point
 * of them being in the same panel.
 */
function NumericsSection() {
  const scene = useSceneStore((state) => state.scene);
  const patchScene = useSceneStore((state) => state.patchScene);
  const patchNumerics = useSceneStore((state) => state.patchNumerics);
  // Folded shut. These three are set once for a run and then watched rather
  // than touched, so they are the one part of this instrument that can be a
  // drawer without making "where does this live?" a question again: the drawer
  // is named, and it is the last thing in the pane.
  return <details className="scene-instrument-section instrument-drawer" aria-label="Timing and numerics">
    <summary><span>Numerics</span><small>rates &amp; budgets</small></summary>
    <div className="field-grid">
      <NumberField label="Shared simulation step" unit="ms"
        value={Math.round(scene.numerics.fixedDt_s * 1000)} step={1} min={1} max={50}
        onChange={(value) => simulation.setStepSize(value / 1000)} />
      <NumberField label="Oracle cell" unit="m"
        value={scene.nominalResolution.length_m} step={0.0025} min={0.0125} max={0.08}
        onChange={(value) => patchScene({ nominalResolution: { length_m: value } })} />
      <NumberField label="PCG budget" unit="iterations"
        value={scene.numerics.pressureMaxIterations} step={20} min={8} max={1000}
        onChange={(value) => patchNumerics({ pressureMaxIterations: Math.round(value) })} />
    </div>
    <small className="control-hint">Rigid bodies and fluid advance on the same fixed step. Changes apply to the live simulation without resetting its clock.</small>
  </details>;
}

/**
 * The advance pipeline, as an instrument over the scene.
 *
 * Everything the retired SIM tab held except its docked shell: the method's own
 * declared stage graph, the exact boundary-chain partition read at each stage's
 * junction, the gates, and the stage-declared parameter controls. The lamps here
 * are readouts wherever the method says a stage is not optional — a solver stage
 * is not something a reader withholds — which is the one honest difference from
 * the frame graph, and it survives the merge as a field rather than as a second
 * diagram.
 */
export function SimPipelineOverlay({ lenses: override }: {
  /** Overrides the running method's own roster. For tests and previews. */
  readonly lenses?: readonly AnyStageLens[];
}) {
  const methodId = useMethodStore((state) => state.methodId);
  const quality = useMethodStore((state) => state.quality);
  const overrides = useMethodStore((state) => state.overrides);
  const info = useDiagnosticsStore((state) => state.gpuInfo);
  const bodyCount = useDiagnosticsStore((state) => state.bodies.length);
  const scene = useSceneStore((state) => state.scene);
  const running = useRuntimeStore((state) => state.runState === "running");
  const overlayMode = useUIStore((state) => state.gridOverlayMode);
  const overlayAxis = useUIStore((state) => state.gridOverlayAxis);
  const setOverlayMode = useUIStore((state) => state.setGridOverlayMode);
  const setOverlayAxis = useUIStore((state) => state.setGridOverlayAxis);

  const method = getMethod(methodId);
  // An override roster (tests and previews) is matched to rows by stage id;
  // otherwise each graph node carries its own lens.
  const lensByStage = new Map((override ?? []).map((lens) => [lens.stage, lens]));
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
  // instrumentation records. The overlay opts in for as long as it is open and
  // hands the setting back exactly as it found it — measurement is real work,
  // and leaving it running after the overlay closed would charge every later
  // advance for a readout nobody is looking at. It never overrides an explicit
  // choice made elsewhere, which is why the guard is on `off` rather than on a
  // stored previous value.
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
  const measurementDomain = stageTrace?.domain;

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

  // The timing row *is* the partition, so the lens opens from the row that
  // prices the stage rather than from a list somewhere else that would have to
  // be read against this one. A lens is a slice view like any other, so an
  // overlay that is off adopts a plane the way the field picker does.
  const openLens = (stageId: string, open: boolean) => {
    if (!open) { setOverlayAxis("off"); return; }
    setOverlayMode(stageLensOverlayMode(stageId));
    if (overlayAxis === "off") setOverlayAxis(DEFAULT_GRID_OVERLAY_AXIS);
  };

  // The method's declared graph, resolved into the shared diagram's view model.
  // Computed per render rather than memoized: every input is already a render
  // input, and a memo keyed on a closure over `toggleStage` would have to list
  // the whole set anyway.
  const bands: readonly PipelineBand[] = !graph ? []
    : graph.bands.map((band): PipelineBand => {
      const bandCost = measureFluidPipelineBand(band.id, graph, costs, total_ms);
      const priced = measured && bandCost.duration_ms !== undefined;
      return {
        id: band.id,
        label: band.label,
        cost_ms: priced ? bandCost.duration_ms : undefined,
        share: priced && total_ms > 0 ? bandCost.share : undefined,
        rows: graph.stages.filter((stage) => stage.band === band.id).map((stage) => {
          const state = stage.state(context);
          const chip = stage.chip(context);
          const cost = measured
            ? measureFluidPipelineStage(stage, graph.stages, costs, total_ms, state)
            : { kind: "unmeasured" as const, share: 0 };
          const tip = fluidPipelineTipText(stage, chip);
          const toggleable = Boolean(stage.toggle);
          const statusLabel = toggleable
            ? state === "on" ? "encoding" : state === "off" ? "configured off" : "not applicable to this scene"
            : "always encoded";
          // The node carries its own lens; the roster lookup is only for a
          // caller that overrides the method's lenses.
          const lens = override ? lensByStage.get(stage.id) : stage.lens;
          const lensOpen = Boolean(lens)
            && overlayAxis !== "off" && overlayMode === stageLensOverlayMode(stage.id);
          return {
            id: stage.id,
            label: stage.label,
            state,
            tip,
            chip,
            cost: {
              kind: cost.kind,
              duration_ms: cost.duration_ms,
              encodedFraction: cost.encodedFraction,
              explanation: `${stage.label}\n\n${costExplanation(cost, measurementDomain)}`,
            },
            lamp: {
              // A stage the advance always encodes is a status, not a control:
              // it reports, takes no click and offers no switch semantics.
              kind: toggleable ? "switch" as const : "fixed" as const,
              checked: state === "on",
              disabled: !toggleable || state === "unavailable",
              ariaLabel: `${stage.label}: ${statusLabel}`,
              title: toggleable
                ? stage.toggle?.hint ? `${stage.toggle.hint}\n\n${tip}` : tip
                : `Always encoded · fixed pipeline stage\n\n${tip}`,
              onToggle: toggleable ? () => toggleStage(stage.id, state !== "on") : undefined,
            },
            tap: lens ? {
              label: "◎",
              title: `Draw what this stage did, phase by phase, on the slice.\n\n${lens.description}`,
              active: lensOpen,
              onToggle: () => openLens(stage.id, !lensOpen),
            } : undefined,
            controls: controls[stage.id],
          };
        }),
      };
    });

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
  // What the configuration actually allocated, and what the running method says
  // about the machinery it selected. These were the method panel's two readouts;
  // they belong beside the pipeline the lattice dispatches over rather than in a
  // configuration surface that cannot show what either one costs.
  const configurationReadouts = method.configurationReadouts?.(info ?? undefined, values) ?? [];

  return <>
    {/* One band, not three. What is running, whether the figures under it are
        live, and what an advance costs are the same question asked three ways,
        and they were three full-width strips of furniture before the diagram
        started. The lattice moved down to the fact line, where the numbers that
        describe it already were. */}
    <div className="render-status-line">
      <span className={running && info ? "online" : ""} />
      <strong data-testid="fluid-pipeline-method">{method.badge ?? method.label}</strong>
      <PipeToggle label="Live" checked={liveTiming} onChange={setLiveTiming}
        hint={`Samples the method-owned advance boundary chain on a cadence without changing its algorithmic schedule.\n\n${sourceLabel}`} />
      <code data-testid="fluid-advance-cost" title={sourceLabel}>{advanceLabel}</code>
    </div>

    {/* What the configuration actually allocated: the lattice, how many samples
        that is, and what it cost to hold. */}
    {info && <div className="grid-readout" title={`The grid the selected quality and parameters actually allocated.\n\n${liveTuning
      ? "Stage gates and tuning controls apply to the attached solver without resetting time. Pressure schedule controls are marked separately and rebuild their precomputed dispatch plan."
      : "The solver's lattice; every pass in this diagram dispatches over it."}`} data-testid="grid-readout">
      <strong>{grid}</strong>
      <span>{info.cellCount.toLocaleString()} samples · {(info.allocatedBytes / 1048576).toFixed(1)} MiB{liveTuning ? " · tuning live" : ""}</span>
    </div>}
    {configurationReadouts.map((readout) => <div key={readout.id} className="grid-readout" title={readout.title}>
      <strong>{readout.value}</strong>
      <span>{readout.detail}</span>
    </div>)}

    {graph
      ? <PipelineGraph bands={bands} testId="fluid-pipeline" />
      : <p className="render-inline-warning">
        The {method.label} method has not declared an advance pipeline yet. Select the uniform
        reference method to see its stage graph.
      </p>}

    <NumericsSection />
  </>;
}
