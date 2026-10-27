"use client";

import { useEffect, useMemo, useState, type ReactNode } from "react";
import { useSession } from "../lib/core/session/session-context";
import { resolvedMethodValues } from "../lib/core/stores/method-store";
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
  measureFluidPipelineLabels,
  type FluidPipelineContext,
  type FluidPipelineGraph,
  type FluidPipelineMeasurement,
  type FluidStageControl,
} from "../lib/core/fluid-pipeline";
import { getMethod } from "@/lib/core/method-registry";
import { PHYSICS_EXECUTION_BACKEND_KEY } from "@/lib/core/physics-execution-backend";
import type { MethodParamValue, SelectParamSpec } from "@/lib/core/method-contract";
import { simulation } from "../lib/core/simulation/controller";
import { sceneHasTerrain } from "../lib/core/terrain";
import { formatPipelineDuration, type PipelineBand } from "./PipelineCost";
import { RailChoice, RailRange, RailSelect, RailToggle } from "./rail";
import { Facts } from "./ui";

/**
 * The advance pipeline's view model: the method's declared stage graph, priced
 * by the newest measured advances and carrying each stage's own controls.
 *
 * Drawing it is the caller's — the rail draws a line per band — so this owns
 * only what the figures are and where they came from.
 */

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
  readonly samples?: number;
} {
  const session = useSession();
  const reports = session.diagnostics((state) => state.performanceReports);
  const unavailable = session.diagnostics(state => state.gpuInfo?.physicsTraceUnavailable);
  return useMemo(() => {
    if (unavailable) return {};
    const newest = reports.findLast((report) => report.methodId === methodId && report.physics);
    if (!newest) return {};
    // Reports can repeat one completed GPU sample while the next is in flight.
    // Pick unique advances before limiting the window, not twelve UI reports.
    const seen = new Set<string>();
    const recent = reports
      .filter((report) => report.methodId === methodId && report.context === newest.context && report.physics)
      .filter((report) => {
        const trace = report.physics!;
        const key = JSON.stringify([trace.context, trace.sampleId, trace.capturedAt_ms, trace.measurementSource]);
        if (seen.has(key)) return false;
        seen.add(key); return true;
      }).slice(-TRACE_WINDOW);
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
      return { total: physicsMean, stages: physicsMean, samples: matching.length };
    }

    // Uniform Geometric only publishes measured GPU stages. Old reports
    // cannot reintroduce a CPU or queue-wall substitute after a scene edit.
    if (methodId === "uniform-volume" || methodId === "uniform-narrow-band-flip") return {};

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
      ? { total: cpuMean, stages: cpuMean, samples: cpu.length }
      : { total: physicsMean, stages: undefined, samples: matching.length };
  }, [reports, methodId, unavailable]);
}

/** Why the figure is the kind of number it is, in solver terms. */
function costExplanation(
  cost: FluidPipelineMeasurement,
  measurementDomain: "cpu" | "gpu" | undefined,
  paged = false,
): string {
  switch (cost.kind) {
    case "withheld":
      return "0 ms — gate closed; not encoded this advance.";
    case "shared":
      return `Runs inside ${cost.insideStage?.replace(/-/g, " ")}; the figure is that stage's.`;
    case "idle":
      return "0 ms — encoded nothing in the sampled advances.";
    case "structural":
      return "A decision, not a dispatch.";
    case "unmeasured":
      return "No measurement for this advance.";
    default:
      return `${formatPipelineDuration(cost.duration_ms ?? 0)} ${paged ? "elapsed" : measurementDomain === "cpu" ? "CPU" : "GPU"}, mean of the trace window.${paged ? " Includes page-submission gaps." : ""}${cost.encodedFraction !== undefined
        ? ` Encoded in ${Math.round(cost.encodedFraction * 100)}% of advances; per-advance expectation.`
        : ""}`;
  }
}

/** The physics backend choice and the method values everything here reads. */
export function useSolverSetup() {
  const session = useSession();
  const methodId = session.method((state) => state.methodId);
  const quality = session.method((state) => state.quality);
  const overrides = session.method((state) => state.overrides);
  const method = getMethod(methodId);
  const values = resolvedMethodValues({ methodId, quality, overrides });
  const backend = method.params.find((spec): spec is SelectParamSpec =>
    spec.key === PHYSICS_EXECUTION_BACKEND_KEY && spec.kind === "select");
  const cpuPhysics = backend !== undefined && values[backend.key] === "cpu";
  return { methodId, method, values, backend, cpuPhysics };
}

type StageReadout = Extract<FluidStageControl, { kind: "readout" }>;
type StageSetting = Exclude<FluidStageControl, StageReadout>;

export interface FluidPipelineCost {
  /** Whether the method declares a stage graph at all. */
  readonly declared: boolean;
  readonly bands: readonly PipelineBand[];
  /** The advance's measured total, or why there is none. */
  readonly advance: string;
  readonly source: string;
}

/**
 * The priced pipeline. Mounting it opts physics tracing in — measurement is
 * real work — and unmounting hands the setting back exactly as it was found,
 * so nothing is sampled while nobody is reading it. It never overrides an
 * explicit choice made elsewhere, which is why the guard is on `off`.
 */
export function useFluidPipelineCost(): FluidPipelineCost {
  const session = useSession();
  const { methodId, values } = useSolverSetup();
  const info = session.diagnostics((state) => state.gpuInfo);
  const bodyCount = session.diagnostics((state) => state.bodies.length);
  const scene = session.scene((state) => state.scene);
  const running = session.runtime((state) => state.runState === "running");

  // Graphs are declared beside their encoders, so loading one loads the solver
  // module; async keeps an unsupported method from paying for that.
  const [loaded, setLoaded] = useState<{ readonly methodId: string; readonly graph: FluidPipelineGraph }>();
  useEffect(() => {
    let cancelled = false;
    void getMethod(methodId).pipelineGraph?.().then((graph) => {
      if (!cancelled) setLoaded({ methodId, graph });
    });
    return () => { cancelled = true; };
  }, [methodId]);
  const graph = loaded?.methodId === methodId ? loaded.graph : undefined;

  useEffect(() => {
    const store = usePerformanceInstrumentationStore.getState();
    if (store.mode !== "off") return;
    store.setMode("timeline");
    return () => {
      const current = usePerformanceInstrumentationStore.getState();
      if (current.mode === "timeline") current.setMode("off");
    };
  }, []);

  const timing = usePhysicsTiming(methodId);
  const stageTrace = timing.stages;
  const costs = useMemo(() => fluidPipelinePhaseCosts(stageTrace), [stageTrace]);
  const total_ms = stageTrace?.total_ms ?? 0;
  const measured = stageTrace !== undefined;
  const domain = stageTrace?.domain;
  const paged = methodId === "uniform-volume" && values.volumeStorage !== "dense" && info?.uniformMixedGeneration === undefined;

  const context: FluidPipelineContext = {
    values,
    info,
    sceneId: scene.sceneId,
    bodyCount,
    hasTerrain: sceneHasTerrain(scene),
    hasInflow: Boolean(scene.fluid.inflow),
    running,
  };
  const setParam = (param: string, value: MethodParamValue) =>
    simulation.setMethodParam(methodId, param, value, session.id);

  // Declarative stage controls in the rail's grammar. `param-*` route through
  // the method store; runtime-safe parameters reach the attached solver on the
  // next frame, structural ones take the controller's rebuild path.
  const renderSetting = (control: StageSetting, key: number): ReactNode => {
    const disabled = control.enabled ? !control.enabled(context) : false;
    switch (control.kind) {
      case "param-choice": {
        const value = String(values[control.param] ?? "");
        const on = control.options.find((option) => option.label === "On");
        const off = control.options.find((option) => option.label === "Off");
        if (control.options.length === 2 && on && off) {
          return <RailToggle key={key} label={control.label} checked={value === on.value} disabled={disabled}
            hint={control.hint} onChange={(checked) => setParam(control.param, checked ? on.value : off.value)} />;
        }
        const options = control.options.map((option) => ({ ...option }));
        // Pills only while they fit beside the name; otherwise a list.
        return options.length <= 3 && options.reduce((sum, option) => sum + option.label.length, 0) <= 16
          ? <RailChoice key={key} label={control.label} value={value} options={options} disabled={disabled}
            hint={control.hint} onChange={(next) => setParam(control.param, next)} />
          : <RailSelect key={key} label={control.label} value={value} options={options} disabled={disabled}
            hint={control.hint} onChange={(next) => setParam(control.param, next)} />;
      }
      case "param-range":
        return <RailRange key={key} label={control.label} unit={control.unit}
          value={Number(values[control.param] ?? control.min)}
          min={control.min} max={control.max} step={control.step} digits={control.digits ?? 0}
          hint={control.hint} disabled={disabled}
          onInput={control.live ? (next) => setParam(control.param, next) : undefined}
          onChange={(next) => setParam(control.param, next)} />;
    }
  };

  const toggleStage = (stageId: string, checked: boolean) => {
    const stage = graph?.stages.find((candidate) => candidate.id === stageId);
    if (!stage?.toggle) return;
    setParam(stage.toggle.param, checked ? stage.toggle.on : stage.toggle.off);
  };

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
          const settings = stage.controls?.filter((control): control is StageSetting => control.kind !== "readout") ?? [];
          const readouts = stage.controls?.filter((control): control is StageReadout => control.kind === "readout") ?? [];
          const hasBreakdown = stage.breakdown?.some(child => child.phaseLabels.some(label => costs.has(label))) ?? false;
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
              explanation: costExplanation(cost, domain, paged),
            },
            breakdown: stage.breakdown?.map(part => {
              const measuredPart = measureFluidPipelineLabels(part.phaseLabels, costs, total_ms);
              // Zero-duration seams are omitted by the recorder. A trace with
              // the new vocabulary proves a missing sibling was skipped; an
              // older parent-only trace cannot tell us its children's costs.
              const partCost = measuredPart.kind === "unmeasured" && hasBreakdown
                ? { kind: "idle" as const, duration_ms: 0, share: 0 }
                : measuredPart;
              return { label: part.label, cost: { ...partCost, explanation: part.hint } };
            }),
            lamp: {
              kind: toggleable ? "switch" as const : "fixed" as const,
              checked: state === "on",
              disabled: !toggleable || state === "unavailable",
              ariaLabel: stage.label,
              title: stage.toggle?.hint ?? tip,
              onToggle: toggleable ? () => toggleStage(stage.id, state !== "on") : undefined,
            },
            controls: settings.length + readouts.length > 0 ? <>
              {settings.map(renderSetting)}
              {readouts.length > 0 && <Facts items={readouts.map((readout) =>
                ({ label: readout.label, value: readout.value(context), hint: readout.hint }))} />}
            </> : undefined,
          };
        }),
      };
    });

  const total = timing.total;
  return {
    declared: graph !== undefined,
    bands,
    advance: total ? formatPipelineDuration(total.total_ms) : info?.physicsTraceUnavailable ? "unavailable" : "—",
    source: !total
      ? info?.physicsTraceUnavailable ?? "Waiting for a sampled advance."
      : total.measurementSource === "gpu-hardware-timestamp"
        ? `GPU timestamps, mean of ${timing.samples ?? 0} advances. Sampling adds a little overhead.`
        : total.measurementSource === "cpu-active-wall"
          ? `CPU boundary chain, mean of ${timing.samples ?? 0} advances.`
          : "GPU queue wall: total only, no stage split.",
  };
}
