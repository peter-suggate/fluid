"use client";

import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { useDiagnosticsStore } from "../lib/core/stores/diagnostics-store";
import { useRuntimeStore } from "../lib/core/stores/runtime-store";
import { useSceneStore } from "../lib/core/stores/scene-store";
import { useUIStore } from "../lib/core/stores/ui-store";
import { usePerformanceInstrumentationStore } from "../lib/core/stores/performance-instrumentation-store";
import { averagePerformanceTraces, type PerformanceTrace } from "../lib/core/performance-trace";
import { mergeRenderFrameManifests, type RenderFrameManifest } from "../lib/core/render-frame-stages";
import {
  measureRenderPipelineBand,
  measureRenderPipelineNode,
  renderPipelineEncodings,
  renderPipelineStageDurations,
  renderPipelineTipText,
  renderPipelineUnownedPhases,
  RENDER_PIPELINE_BANDS,
  RENDER_PIPELINE_COLLAPSE_GROUPS,
  RENDER_PIPELINE_NODES,
  type RenderPipelineContext,
  type RenderPipelineMeasurement,
} from "../lib/core/render-pipeline-graph";
import { disabledRenderStagesFrom } from "../lib/core/render-stage-switches";
import {
  PipelineGraph,
  formatPipelineDuration,
  type PipelineBand,
  type PipelineRow,
} from "./PipelineGraph";
import {
  SVO_RENDER_STAGE_DEFINITIONS,
  SVO_RENDER_STAGE_MAXIMUM_LIGHT_SLOT,
  svoRenderStageUsesLightSlot,
} from "../lib/svo/svo-render-diagnostics";
import { resolveSvoPrimaryTraversal } from "../lib/svo/svo-render-options";
import {
  SVO_ENVIRONMENT_BRICK_REFINEMENT_MAXIMUM,
  SVO_ENVIRONMENT_REFINEMENT_DEPTH_MAXIMUM,
  SVO_ENVIRONMENT_REFINEMENT_DEPTH_MINIMUM,
  SVO_PRIMARY_LEAF_VISIT_HARD_LIMIT,
  SVO_RENDER_QUALITY_PRESETS,
  SVO_RENDER_TUNING_PRESETS,
  svoRenderTuningKey,
  svoSceneryRefinementDepth,
  type SvoConeRadianceReconstruction,
  type SvoRenderQualityPreset,
  type SvoRenderTuning,
} from "../lib/svo/svo-render-tuning";
import { findSceneDefinition } from "../lib/core/scenes";
import { sceneDefinitionTakesLattice } from "../lib/core/scene-definition";
import { simulation } from "../lib/core/simulation/controller";
import { PipeChoice, PipeRange, PipeToggle } from "./PipeControls";

/** Frames the live readout averages over. */
const TRACE_WINDOW = 12;

/**
 * Frame totals observed under each setting of each node, so a switch can report
 * what it was worth.
 *
 * This is the only figure that can price a node whose work is a *term* inside
 * another pass — GI composition is a branch in the deferred shader and has no
 * dispatch to time — and it is the figure the question "what do I gain by
 * turning this off" literally asks for. It is a difference of two medians of
 * averaged totals, so it also catches whatever downstream got cheaper
 * alongside, which the node's own phase sum cannot.
 */
type AblationSample = { on_ms: number[]; off_ms: number[] };

/** Settled totals retained per side. The median of these prices the setting. */
const ABLATION_SAMPLE_CAP = 5;

const median = (values: readonly number[]): number => {
  const sorted = [...values].sort((a, b) => a - b);
  const mid = sorted.length >> 1;
  return sorted.length % 2 ? sorted[mid]! : (sorted[mid - 1]! + sorted[mid]!) / 2;
};

/**
 * How long a setting must hold before its total is believed.
 *
 * A switch invalidates cached frames, clears the world-GI cache and restarts
 * the trace context, so the first traces after one are measuring the recovery
 * and not the pipeline. Two full windows of settled frames is the cheapest
 * thing that reliably excludes them.
 */
const ABLATION_SETTLE_MS = 900;

const parseNodeStates = (signature: string): [string, boolean][] => signature.split(",")
  .filter(Boolean)
  .map((entry) => {
    const [id, on] = entry.split("=");
    return [id ?? "", on === "1"] as [string, boolean];
  });

/**
 * Milliseconds each node's current setting is saving, where both settings have
 * been seen against the same rest-of-pipeline.
 *
 * Positive means the current setting is the cheaper one. The comparison is only
 * meaningful while nothing else moved, so when the signature changes every
 * node's pair is discarded except the one node that actually flipped — a delta
 * measured across two different pipelines is not that node's cost, and keeping
 * it would be worse than showing nothing.
 */
function useStageAblation(
  signature: string,
  contextKey: string,
  total_ms: number,
  measured: boolean,
): ReadonlyMap<string, number> {
  const samples = useRef(new Map<string, AblationSample>());
  const previous = useRef<{ signature: string; contextKey: string; at_ms: number } | undefined>(undefined);
  const [deltas, setDeltas] = useState<ReadonlyMap<string, number>>(() => new Map());

  useEffect(() => {
    const now_ms = performance.now();
    const last = previous.current;
    if (!last || last.signature !== signature || last.contextKey !== contextKey) {
      if (last && last.contextKey === contextKey) {
        const before = new Map(parseNodeStates(last.signature));
        const moved = parseNodeStates(signature).filter(([id, on]) => before.get(id) !== on);
        const survivor = moved.length === 1 ? moved[0]?.[0] : undefined;
        const kept = survivor ? samples.current.get(survivor) : undefined;
        samples.current = new Map(survivor && kept ? [[survivor, kept]] : []);
      } else if (last) {
        // Tuning, run state or the stage view moved: every stored total was
        // measured under a different pipeline, and a delta differenced across
        // that would be the change's cost wearing a node's name.
        samples.current = new Map();
      }
      previous.current = { signature, contextKey, at_ms: now_ms };
      setDeltas(new Map());
      return;
    }
    // Everything before the settle is measuring the recovery: a switch clears
    // cached frames and the world-GI cache, and restarts the trace context.
    if (!measured || total_ms <= 0 || now_ms - last.at_ms < ABLATION_SETTLE_MS) return;
    const next = new Map<string, number>();
    for (const [id, on] of parseNodeStates(signature)) {
      const sample = samples.current.get(id) ?? { on_ms: [], off_ms: [] };
      // Medians of a bounded window, not one total against one total: this
      // lane's run-to-run drift has reached 30%, and a single-sample pair has
      // measured nothing but the noise.
      const side = on ? sample.on_ms : sample.off_ms;
      side.push(total_ms);
      if (side.length > ABLATION_SAMPLE_CAP) side.shift();
      samples.current.set(id, sample);
      if (sample.on_ms.length === 0 || sample.off_ms.length === 0) continue;
      next.set(id, on
        ? median(sample.off_ms) - median(sample.on_ms)
        : median(sample.on_ms) - median(sample.off_ms));
    }
    setDeltas((current) => (current.size === next.size
      && [...next].every(([id, value]) => current.get(id) === value) ? current : next));
  }, [signature, contextKey, total_ms, measured]);

  return deltas;
}

const rendererFailureLabels = {
  "missing-source": "waiting for structural SVO data",
  "unsupported-terrain": "terrain source could not be represented",
  "unsupported-glass-cutout": "authored glazing needs an opaque shell cutout",
  "missing-pbr-materials": "production PBR material table is unavailable",
  "missing-lighting-publications": "production light/environment publications are unavailable",
  "pipeline-compile-failure": "SVO pipeline failed to compile",
  "pipeline-compiling": "SVO pipeline is compiling",
  "frame-rejected": "live SVO frame publication was rejected",
} as const;

/**
 * A millimetre readout that keeps the whole leaf ladder exact. The sizes are
 * binary fractions of the lattice, so a fixed number of decimals either rounds
 * the fine end into a lie or pads the coarse end with zeros.
 */
const trimmed = (value: number) => value.toFixed(5).replace(/\.?0+$/, "");

/**
 * Fence-measured band phases → the graph band whose collar carries the figure.
 * The costs read where each band connects to the pipeline, never as a separate
 * readout: the collar is the junction, so its figure sits beside the work it
 * prices. Two encode segments (extraction+caustics, interfaces+composite) both
 * belong to the OUTPUT collar and are summed there.
 */
const BAND_WALL_COLLARS: Partial<Record<string, string>> = {
  "scene-upload": "source",
  "surface-extraction": "output",
  "svo-primary": "primary",
  "svo-cone-lighting": "lighting",
  "dry-scene": "shading",
  "optical-composite": "output",
};

/**
 * What is left of a band's wall once its rows are accounted for, said plainly.
 *
 * A band wall is a queue fence difference, so it contains the band's GPU work
 * *and* whatever the instrument costs — submit turnaround, and the latency of
 * the completion callback the timestamp is taken in. When every row in the
 * band is priced or is a measured zero, the residual is not hidden work in one
 * of them; it belongs to the measurement, and the panel says so rather than
 * naming a row for it. That naming is what put 27.9 ms on a world build whose
 * stages encoded nothing.
 */
function bandResidualNote(
  wall_ms: number,
  entries: readonly { cost: RenderPipelineMeasurement }[],
  manifested: boolean,
): string {
  const compute_ms = entries.reduce((sum, entry) =>
    sum + (entry.cost.kind === "measured" ? entry.cost.duration_ms ?? 0 : 0), 0);
  const silent = entries.filter((entry) => entry.cost.kind === "withheld").length;
  const residual_ms = Math.max(0, wall_ms - compute_ms);
  if (!manifested) {
    return `${formatPipelineDuration(compute_ms)} of it is measured compute; the rest is unattributed until a stage manifest arrives.`;
  }
  return `${formatPipelineDuration(compute_ms)} of it is measured compute and ${formatPipelineDuration(residual_ms)} is unattributed`
    + (silent > 0
      ? `. ${silent} row${silent === 1 ? "" : "s"} in this band encoded no pass at all this frame, so the residual is not theirs — it is submit turnaround and fence-callback latency, which a queue-wall measurement cannot separate from the work.`
      : ", spread across this band's render passes and the instrument's own submit turnaround.");
}

/** Why the figure on the pipe is the kind of number it is, in frame terms. */
function costExplanation(cost: RenderPipelineMeasurement): string {
  const measuredSuffix = "GPU execution time, summed over the passes this node owns and averaged across the trace window.";
  const unpriced = cost.unpricedRenderPasses
    ? `\n\n${cost.unpricedRenderPasses} render pass${cost.unpricedRenderPasses === 1 ? "" : "es"} in this row cannot be priced: a Metal render pass's timestamp pair brackets a tiler window, not the pass. Only the band's fence-partitioned wall can measure them.`
    : "";
  switch (cost.kind) {
    case "withheld":
      return "0 ms — this stage encoded no pass in the recorded frame. The frame's own stage manifest says so, which is why this is a measured zero rather than a missing measurement.";
    case "shared":
      return `Not a dispatch of its own: this work runs inside ${cost.insideNode?.replace(/-/g, " ").toUpperCase()}, and ⊂ marks the figure as that pass's.`;
    case "unpriced":
      return `This stage encoded ${cost.unpricedRenderPasses} render pass${cost.unpricedRenderPasses === 1 ? "" : "es"} and no compute, and a Metal render pass's timestamp pair is a tiler window rather than a cost. The band's fence-partitioned wall is the only honest figure for it; it reads here when this is the band's only such row.`;
    case "structural":
      return "A gate, not a pass. It spends no frame time either way — its worth shows up as the row it lets the frame skip going to zero.";
    case "wall":
      return `${formatPipelineDuration(cost.duration_ms ?? 0)} wall-clock, derived: this is the band's only row whose passes no timestamp can price, so the band's fence-partitioned wall minus its measured compute lands here. Render passes included; a different basis than the compute figures around it.`;
    case "unmeasured":
      return "No measurement has arrived for this row yet: no trace, or no stage manifest for the frame it would describe.";
    default:
      return `${formatPipelineDuration(cost.duration_ms ?? 0)} ${measuredSuffix}${unpriced}`;
  }
}

/**
 * The averaged frame total and the finest honest stage partition available.
 *
 * The header is the GPU queue-wall frame total. Trunk values are kept separate:
 * they contain only trustworthy hardware pass timestamps and are never filled
 * with CPU command-encoding time or another proxy.
 */
function usePresentationTiming(): {
  readonly frame?: PerformanceTrace;
  readonly stages?: PerformanceTrace;
  readonly stageSource?: "gpu";
  /** Fence-partitioned band walls: real queue-wall cost per encode band, from 1-in-16 sampling frames. */
  readonly bands?: PerformanceTrace;
  /** Which stages the recorded frames encoded, and with how many passes of each kind. */
  readonly manifest?: RenderFrameManifest;
} {
  const reports = useDiagnosticsStore((state) => state.performanceReports);
  return useMemo(() => {
    const newest = reports.findLast((report) => report.presentation || report.presentationStages);
    if (!newest) return {};
    const recent = reports
      .filter((report) => report.context === newest.context)
      .slice(-TRACE_WINDOW);
    const gpu = recent
      .map((report) => report.presentation)
      .filter((trace): trace is PerformanceTrace => trace !== undefined);
    const hardware = recent
      .map((report) => report.presentationStages)
      .filter((trace): trace is PerformanceTrace => trace !== undefined);
    // Band samples are sparse (one frame in sixteen), so the same report can
    // repeat one for many frames; the dedup inside the averager keeps each
    // observation counted once.
    const bands = recent
      .map((report) => report.presentationBands)
      .filter((trace): trace is PerformanceTrace => trace !== undefined);
    const latestGpu = gpu.at(-1);
    const frameSamples = latestGpu
      ? gpu.filter((trace) => trace.measurementSource === latestGpu.measurementSource)
      : [];
    const hardwareMean = averagePerformanceTraces(hardware);
    return {
      frame: averagePerformanceTraces(frameSamples),
      stages: hardwareMean,
      stageSource: hardwareMean ? "gpu" : undefined,
      bands: averagePerformanceTraces(bands),
      // Merged across the window rather than taken from the newest frame: an
      // intermittent stage — caustics on mesh-move frames, world maintenance
      // on edit frames — encodes on some frames and not others, and a row that
      // ever encodes must not be treated as a silent one.
      manifest: mergeRenderFrameManifests(recent
        .map((report) => report.presentationStageManifest)
        .filter((entry): entry is RenderFrameManifest => entry !== undefined)),
    };
  }, [reports]);
}

/**
 * The frame graph, as an instrument over the scene.
 *
 * Everything the retired RENDER tab held except its docked shell and its PROBES
 * strip — the probes are ring wedges now. The PROFILE rung strip and the render
 * resolution scale came with it: they are the two questions asked before any
 * node is opened, so they stay above the graph rather than inside a node.
 */
export function RenderPipelineOverlay() {
  const effectiveRendererStatus = useDiagnosticsStore((state) => state.effectiveRendererStatus);
  const svoShadowsEnabled = useUIStore((state) => state.svoShadowsEnabled);
  const setSvoShadowsEnabled = useUIStore((state) => state.setSvoShadowsEnabled);
  const svoAmbientOcclusionEnabled = useUIStore((state) => state.svoAmbientOcclusionEnabled);
  const setSvoAmbientOcclusionEnabled = useUIStore((state) => state.setSvoAmbientOcclusionEnabled);
  const silhouetteRefinementEnabled = useUIStore((state) => state.silhouetteRefinementEnabled);
  const setSilhouetteRefinementEnabled = useUIStore((state) => state.setSilhouetteRefinementEnabled);
  const svoConeTracingMode = useUIStore((state) => state.svoConeTracingMode);
  const setSvoConeTracingMode = useUIStore((state) => state.setSvoConeTracingMode);
  const svoGlobalIlluminationEnabled = useUIStore((state) => state.svoGlobalIlluminationEnabled);
  const setSvoGlobalIlluminationEnabled = useUIStore((state) => state.setSvoGlobalIlluminationEnabled);
  const disabledRenderStages = useUIStore((state) => state.disabledRenderStages);
  const setRenderStageDisabled = useUIStore((state) => state.setRenderStageDisabled);
  const svoStageView = useUIStore((state) => state.svoStageView);
  const setSvoStageView = useUIStore((state) => state.setSvoStageView);
  const svoStageLightSlot = useUIStore((state) => state.svoStageLightSlot);
  const setSvoStageLightSlot = useUIStore((state) => state.setSvoStageLightSlot);
  const svoMaximumTraversalDepth = useUIStore((state) => state.svoMaximumTraversalDepth);
  const setSvoMaximumTraversalDepth = useUIStore((state) => state.setSvoMaximumTraversalDepth);
  const svoMaximumNodeVisits = useUIStore((state) => state.svoMaximumNodeVisits);
  const setSvoMaximumNodeVisits = useUIStore((state) => state.setSvoMaximumNodeVisits);
  const tuning = useUIStore((state) => state.svoRenderTuning);
  const setTuning = useUIStore((state) => state.setSvoRenderTuning);

  // The scene facts the build node has to state for itself: what a leaf
  // actually measures, and whether the depth is legal at all. A simulated scene
  // pins every brick's node at the solver level, so the ladder is a no-op there
  // and has to say so rather than move.
  const finestCellSize_m = useSceneStore((state) => state.scene.voxelDomain.finestCellSize_m);
  const sceneIsDry = useSceneStore((state) => state.scene.systems?.fluid === false);
  const voxelDomain = useSceneStore((state) => state.scene.voxelDomain);
  const presetId = useSceneStore((state) => state.presetId);
  const authoredRefinementDepth = svoSceneryRefinementDepth(voxelDomain, { fluid: !sceneIsDry });
  // Read off the document rather than recomputed: the ladder is signed now, and
  // a negative rung enlarges the dry lattice itself, so the leaf is a published
  // fact and not a function of the finest cell.
  const leafVoxel_mm = (voxelDomain.detailCellSize_m ?? finestCellSize_m) * 1000;
  const zeroRungCellSize_m = voxelDomain.environmentRefinementBaseCellSize_m ?? finestCellSize_m;
  // Water is the only thing that can refuse a depth outright. Whether the
  // scene's *factory* takes a lattice decides only how far the change
  // propagates: with one the document is re-authored and even the terrain pitch
  // follows; without, it is a patch.
  const latticeDefinition = findSceneDefinition(presetId);
  const reauthorsDocument = latticeDefinition !== undefined && sceneDefinitionTakesLattice(latticeDefinition);

  const [liveTiming, setLiveTiming] = useState(true);

  // Per-pass costs only reach the diagnostics store while instrumentation is
  // recording, and instrumentation is off by default because measurement is
  // work the product did not ask for. The overlay opts in for as long as it is
  // open and hands the setting back exactly as it found it — an overlay that
  // left measurement running after it closed would charge every later frame for
  // a readout nobody is looking at. It never overrides an explicit choice made
  // elsewhere, which is why the guard is on `off` rather than a stored previous
  // value.
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

  const timing = usePresentationTiming();
  const trace = timing.frame;
  const stageTrace = timing.stages;
  const durations = useMemo(
    () => renderPipelineStageDurations(liveTiming ? stageTrace : undefined), [stageTrace, liveTiming]);
  // What the frame encoded, which is a different question from what it cost and
  // the one that decides whether a silent row is a zero or an unpriced pass.
  const encodings = useMemo(
    () => renderPipelineEncodings(liveTiming ? timing.manifest : undefined), [timing.manifest, liveTiming]);
  // A trace label that names no stage of the ABI is a measurement bug — a seam
  // publishing a label the registry does not own — and it is louder as a
  // console warning than as a row that quietly reads zero.
  const unowned = useMemo(() => renderPipelineUnownedPhases(stageTrace), [stageTrace]);
  useEffect(() => {
    if (unowned.length > 0) console.warn("Render trace phases owned by no pipeline stage:", unowned);
  }, [unowned]);
  const measured = liveTiming && trace !== undefined;
  const total_ms = trace?.total_ms ?? 0;
  const stageTotal_ms = stageTrace?.total_ms ?? 0;
  // The honest denominator for a share of the frame: earliest pass begin to
  // latest pass end. The pass *sum* double-counts overlapped passes (measured
  // at up to 1.9x on this path), so it is shown as itself and divides nothing.
  const stageSpan_ms = stageTrace?.span_ms ?? 0;
  // Fence-partitioned band walls, folded onto the graph's band collars. These
  // are the only wall-clock figures for the render passes Metal timestamps
  // cannot price, and they read at the junction they measure.
  const bandWalls = useMemo(() => {
    if (!timing.bands) return undefined;
    const walls = new Map<string, number>();
    for (const phase of timing.bands.phases) {
      const collar = BAND_WALL_COLLARS[phase.id];
      if (collar !== undefined) walls.set(collar, (walls.get(collar) ?? 0) + phase.duration_ms);
    }
    return walls;
  }, [timing.bands]);

  const updateTuning = <K extends keyof SvoRenderTuning>(key: K, value: SvoRenderTuning[K]) =>
    setTuning((current) => ({ ...current, [key]: value }));
  const modified = <K extends keyof SvoRenderTuning>(key: K) => tuning[key] !== SVO_RENDER_TUNING_PRESETS.balanced[key];
  const resetTuning = <K extends keyof SvoRenderTuning>(key: K) => () =>
    updateTuning(key, SVO_RENDER_TUNING_PRESETS.balanced[key]);

  const tuningKey = svoRenderTuningKey(tuning);
  // A rung is the pair, so the match is on the pair. `quality` and `reference`
  // carry the same sliders and differ only in how visibility is answered; a
  // lookup by tuning alone would report EXACT as QUALITY.
  const activePreset = (Object.keys(SVO_RENDER_QUALITY_PRESETS) as SvoRenderQualityPreset[])
    .find((preset) => svoRenderTuningKey(SVO_RENDER_QUALITY_PRESETS[preset].tuning) === tuningKey
      && SVO_RENDER_QUALITY_PRESETS[preset].coneTracingMode === svoConeTracingMode);
  const silhouetteRefinementStatus = effectiveRendererStatus.silhouetteRefinement ?? {
    state: effectiveRendererStatus.state === "pending" ? "compiling" as const
      : effectiveRendererStatus.state === "failed" ? "failed" as const
      : silhouetteRefinementEnabled ? "enabled" as const : "disabled" as const,
    detail: effectiveRendererStatus.detail,
  };
  const lightingVisibilityStatus = effectiveRendererStatus.lightingVisibility ?? { state: svoConeTracingMode };

  // What the live readout can actually answer, and why. A per-pass split needs
  // hardware timestamp queries *and* the SVO path — the fallback partition names
  // stages by position, so a queue-walled frame has a total and nothing under
  // it. Saying which of those is missing costs one line; leaving a column of
  // dashes unexplained costs a bug report.
  const perPassSplit = liveTiming && stageTrace !== undefined;
  const hardwarePerPassSplit = perPassSplit && timing.stageSource === "gpu";
  // Three different numbers, each labelled as itself: the queue wall includes
  // any solver advance still in flight, the span is how long the GPU was busy
  // with the presentation's passes, and the compute sum is what the trunk's
  // per-node figures add up to. Showing one of them unlabelled invited reading
  // the pass sum as the frame, which it is not.
  const timingLabel = !liveTiming ? "timing off"
    : !measured ? "awaiting trace"
    : hardwarePerPassSplit && stageSpan_ms > 0
      // Non-breaking spaces inside each figure and before each separator: the
      // three of them do not fit one line over a scene, so this wraps — and a
      // figure parted from its label, or a dot orphaned onto the next line, is
      // the one way three labelled numbers become unreadable.
      ? `${total_ms.toFixed(2)} wall · ${stageSpan_ms.toFixed(2)} busy · ${stageTotal_ms.toFixed(2)} Σcompute ms`
      : `${total_ms.toFixed(2)} ms / frame · total only`;
  const timingHint = !liveTiming
    ? "Live per-pass timing is off. Turn it on to price each node from the frame's own GPU timestamp partition."
    : !measured
      ? "No presentation trace has arrived yet. The next frame is captured as soon as the current timestamp readback completes."
      : hardwarePerPassSplit
        ? `Hardware GPU timestamps, ${TRACE_WINDOW}-frame mean.\n\nWALL is submit to queue-drain and includes any solver advance still in flight. BUSY is the presentation's own GPU span — earliest pass begin to latest pass end — and is the denominator for every share. ΣCOMPUTE is the sum of trustworthy compute passes, which the trunk figures add up to; Metal render-pass tiler windows are excluded from it.`
        : "This device or scene produced only a queue-wall observation and no trustworthy stage partition yet.";

  const disabledStages = useMemo(
    () => disabledRenderStagesFrom(disabledRenderStages), [disabledRenderStages]);

  const context: RenderPipelineContext = {
    disabledStages,
    coneTracingMode: svoConeTracingMode,
    shadowsEnabled: svoShadowsEnabled,
    ambientOcclusionEnabled: svoAmbientOcclusionEnabled,
    seamClosureEnabled: silhouetteRefinementEnabled,
    globalIlluminationEnabled: svoGlobalIlluminationEnabled,
    tuning,
    sceneHasFluid: !sceneIsDry,
    refinementDepth: authoredRefinementDepth,
    leafVoxel_mm,
    rendererActive: effectiveRendererStatus.state === "active",
    stageView: svoStageView,
    // Asked of the shared rule rather than assumed: it answers `traced` for
    // every production frame, so the three raster-only tiers read as
    // unavailable — but under FLUID_SVO_PRIMARY_TRAVERSAL=raster they are live
    // passes with live switches, and the panel should say which frame it is
    // looking at rather than hard-coding one.
    rasterPrimaryActive: resolveSvoPrimaryTraversal("raster") === "raster",
  };

  // The lamp is the node's own switch, and every node has one.
  //
  // Most route through the encode-time ablation set, which withholds the pass
  // outright. Cone visibility, GI composition and seam closure are switched by
  // contracts the shaders already compile against. The graph says which is
  // which.
  const toggleNode = (id: string) => {
    const node = RENDER_PIPELINE_NODES.find((candidate) => candidate.id === id);
    if (node?.stage) {
      setRenderStageDisabled(node.stage, !disabledStages.has(node.stage));
      return;
    }
    if (id === "seam-closure") setSilhouetteRefinementEnabled(!silhouetteRefinementEnabled);
    else if (id === "cone-visibility") setSvoConeTracingMode(svoConeTracingMode === "off" ? "cones" : "off");
    else if (id === "gi-composition") setSvoGlobalIlluminationEnabled(!svoGlobalIlluminationEnabled);
  };

  // One string for the whole pipeline's on/off shape. It is both the dependency
  // the ablation memory keys on and the thing that tells it which single node
  // moved, so it has to be derived from the same `state` the diagram draws.
  const pipelineSignature = RENDER_PIPELINE_NODES
    .map((node) => `${node.id}=${node.state(context) === "off" ? "0" : "1"}`).join(",");
  // Everything that changes the frame without moving a lamp. A stored total
  // from a different tuning, run state or stage view priced a different
  // pipeline; the signature alone could not see that, so an off-sample taken
  // under one tuning was differenced against an on-sample taken under another.
  const runState = useRuntimeStore((state) => state.runState);
  const ablationContextKey = `${tuningKey}|${svoConeTracingMode}|${runState}|view=${svoStageView !== "off" ? "1" : "0"}`;
  const deltas = useStageAblation(pipelineSignature, ablationContextKey, total_ms, measured);

  // Named drawers, shut by default.
  //
  // Every control a node owns still lives on that node — that is the whole
  // point of the graph — but a node with twenty cone budgets under it made the
  // instrument four screens tall over a scene it is supposed to be read
  // against. So a cluster that is *tuning* folds behind its own name, and the
  // controls that decide what the frame IS — the visibility source, the
  // shadow/AO switches — stay on the card where they were.
  const controls: Readonly<Record<string, ReactNode>> = {
    "sparse-world-build": <details className="rp-tune"><summary>Refinement</summary>
      <div className="pipe-fields">
      <PipeRange
        label={`Environment refinement depth · ${reauthorsDocument ? "RE-AUTHORS" : "REBUILD"}`}
        unit="levels" value={authoredRefinementDepth}
        min={SVO_ENVIRONMENT_REFINEMENT_DEPTH_MINIMUM} max={SVO_ENVIRONMENT_REFINEMENT_DEPTH_MAXIMUM}
        step={1} digits={0} disabled={!sceneIsDry}
        onChange={(value) => { simulation.setEnvironmentRefinementDepth(value); }}
        hint={sceneIsDry
          ? `Leaf voxel ${trimmed(leafVoxel_mm)} mm, and the set is drawn at that size — ${authoredRefinementDepth < 0
            ? `${-authoredRefinementDepth} coarser level${authoredRefinementDepth === -1 ? "" : "s"} above`
            : `${authoredRefinementDepth} level${authoredRefinementDepth === 1 ? "" : "s"} below`} the ${trimmed(zeroRungCellSize_m * 1000)} mm zero rung.\n\n`
            + "Signed octree levels relative to the scene's authored lattice. Positive descends into finer leaves; negative enlarges the dry-scene lattice by 2× per level. It writes the document's own cells, which are the single numbers the tree and the scenery both read.\n\n"
            + (reauthorsDocument
              ? "Moving it re-authors the document through this scene's own factory, so the terrain pitch follows; that reloads the preset, and scenery edits since it was opened do not survive it."
              : "This scene's factory takes no lattice, so moving it patches the document: every generator re-resolves its legibility floors at the new leaf, but anything the factory baked keeps its authored value.")
            + "\n\nEach level also adds one to the cone hierarchy; past the runtime's twelve, derived lighting withdraws and the cone node reads EXACT FALLBACK."
          : `Zero on this scene whatever the slider says: the fluid solver claims every brick of the container and a solver brick pins its node, so the leaf stays the ${trimmed(finestCellSize_m * 1000)} mm lattice. Turn water off under the tank's Water setting to move on this environment-only ladder.`} />
      <PipeRange label="Environment brick refinement" unit="levels" value={tuning.environmentBrickRefinementLevels}
        min={0} max={SVO_ENVIRONMENT_BRICK_REFINEMENT_MAXIMUM} step={1} digits={0}
        onChange={(value) => updateTuning("environmentBrickRefinementLevels", value)}
        modified={modified("environmentBrickRefinementLevels")} onReset={resetTuning("environmentBrickRefinementLevels")}
        hint="Additional SVO subdivision for authored scenery outside the simulation lattice. Already at its ceiling by default, so the only move is down; changing it rebuilds the sparse world." />
      <div className="pipe-row" role="group" aria-label="Environment refinement exemptions">
        <PipeToggle label="Flat-node exemption · REBUILD" checked={tuning.environmentPlanarRefinementExemption}
          hint="Let the refinement rule stop at a node its surface crosses flatly instead of spending the depth above. The test is second order, so it declines depth exactly where curvature is lowest — which is also where a coarse leaf shows, because the primary shades a leaf as one of six axis-aligned voxel faces. On, the tree is smaller and builds faster, at that cost."
          onChange={(value) => updateTuning("environmentPlanarRefinementExemption", value)} />
      </div>
      </div>
    </details>,

    "primary-traversal": <details className="rp-tune"><summary>Traversal budgets</summary>
      <div className="pipe-fields">
      <PipeRange label="Maximum traversal depth" unit="levels" value={svoMaximumTraversalDepth}
        min={1} max={21} step={1} digits={0} onChange={setSvoMaximumTraversalDepth}
        hint="Hierarchy depth accepted by every camera traversal. A budget, not a detail control: exceeding it reports traversal exhaustion rather than falling back to a coarser surface." />
      <PipeRange label="Maximum node visits" unit="nodes" value={svoMaximumNodeVisits}
        min={1} max={256} step={1} digits={0} onChange={setSvoMaximumNodeVisits}
        hint="Topology nodes allowed per primary traversal call." />
      <PipeRange label="Maximum leaf visits" unit="bricks" value={tuning.primaryLeafVisits}
        min={1} max={SVO_PRIMARY_LEAF_VISIT_HARD_LIMIT} step={1} digits={0}
        onChange={(value) => updateTuning("primaryLeafVisits", value)}
        modified={modified("primaryLeafVisits")} onReset={resetTuning("primaryLeafVisits")} />
      </div>
    </details>,

    "seam-closure": silhouetteRefinementStatus.state === "failed" || silhouetteRefinementStatus.state === "compiling"
      ? <p data-testid="silhouette-refinement-status" aria-live="polite"
          className={silhouetteRefinementStatus.state === "failed" ? "render-inline-warning" : "render-inline-status"}>
          Primary seam closure: {silhouetteRefinementStatus.state.toUpperCase()}
          {silhouetteRefinementStatus.detail ? ` · ${silhouetteRefinementStatus.detail}` : ""}
        </p>
      : undefined,

    "cone-visibility": <>
      <PipeChoice label="Visibility source" value={svoConeTracingMode} onChange={setSvoConeTracingMode} options={[
        { value: "cones", label: "CONES", hint: "Cone-traced soft shadows, AO and GI, fed by the reduced-rate prepass and world-GI cache." },
        { value: "exact", label: "EXACT", hint: "No cone stage runs; shadows and AO use bounded exact SVO visibility rays. Sharp reference shadows, costlier per pixel." },
        { value: "off", label: "OFF", hint: "No visibility work at all: unshadowed direct lighting, no AO, no GI. Strictly removes work." },
      ]} />
      <div className="pipe-row" role="group" aria-label="SVO lighting effects">
        <PipeToggle label="Shadows" checked={svoShadowsEnabled} disabled={svoConeTracingMode === "off"} onChange={setSvoShadowsEnabled} />
        <PipeToggle label="AO" checked={svoAmbientOcclusionEnabled} disabled={svoConeTracingMode === "off"} onChange={setSvoAmbientOcclusionEnabled} />
      </div>
      <PipeChoice label="Cone prepass rate" value={String(tuning.coneLightingScale)} disabled={svoConeTracingMode !== "cones"}
        onChange={(value) => updateTuning("coneLightingScale", Number(value) as SvoRenderTuning["coneLightingScale"])}
        options={[{ value: "1", label: "FULL" }, { value: "0.5", label: "2×2" }, { value: "0.25", label: "4×4" }, { value: "0.125", label: "8×8" }]} />
      {/* What the frame *is* stays on the card above; what it is *calibrated
          to* folds. Twenty budgets is a session's worth of tuning, not
          something read at a glance beside the picture they change. */}
      <details className="rp-tune"><summary>Cone calibration</summary>
      <div className="pipe-fields">
        <PipeRange label="Shadow cone aperture" unit="rad" value={tuning.shadowConeAperture} min={0.01} max={0.25} step={0.005} digits={3}
          onChange={(value) => updateTuning("shadowConeAperture", value)}
          modified={modified("shadowConeAperture")} onReset={resetTuning("shadowConeAperture")}
          hint="Wider cones take larger march steps and produce softer shadows; narrower cones preserve sharper shadows but need more taps. The sharp/soft decision most scenes want is the CONES/EXACT switch above; this calibrates the soft path." />
        <PipeRange label="Shadow strength" unit="%" value={tuning.shadowStrength * 100} min={0} max={100} step={1} digits={0}
          onChange={(value) => updateTuning("shadowStrength", value / 100)} modified={modified("shadowStrength")} onReset={resetTuning("shadowStrength")} />
        <PipeRange label="Shadow origin bias" unit="cells" value={tuning.shadowBiasCells} min={0} max={0.25} step={0.005} digits={3}
          onChange={(value) => updateTuning("shadowBiasCells", value)} modified={modified("shadowBiasCells")} onReset={resetTuning("shadowBiasCells")} />
        <PipeRange label="AO cone aperture" unit="rad" value={tuning.aoConeAperture} min={0.1} max={1.4} step={0.01} digits={2}
          onChange={(value) => updateTuning("aoConeAperture", value)} modified={modified("aoConeAperture")} onReset={resetTuning("aoConeAperture")} />
        <PipeRange label="AO strength" unit="%" value={tuning.aoStrength * 100} min={0} max={100} step={1} digits={0}
          onChange={(value) => updateTuning("aoStrength", value / 100)} modified={modified("aoStrength")} onReset={resetTuning("aoStrength")} />
        <PipeRange label="AO radius" unit="×" value={tuning.aoRadiusScale} min={0.1} max={3} step={0.05} digits={2}
          onChange={(value) => updateTuning("aoRadiusScale", value)} modified={modified("aoRadiusScale")} onReset={resetTuning("aoRadiusScale")} />
        <PipeRange label="Cone step budget" unit="steps" value={tuning.coneStepBudget} min={1} max={48} step={1} digits={0}
          onChange={(value) => updateTuning("coneStepBudget", value)} modified={modified("coneStepBudget")} onReset={resetTuning("coneStepBudget")} />
        <PipeRange label="Shaded lights" unit="lights" value={tuning.maximumShadedLights} min={1} max={8} step={1} digits={0}
          onChange={(value) => updateTuning("maximumShadedLights", value)} modified={modified("maximumShadedLights")} onReset={resetTuning("maximumShadedLights")} />
        <PipeRange label="Normal escape" unit="cells" value={tuning.coneNormalEscapeCells} min={0} max={2} step={0.05} digits={2}
          onChange={(value) => updateTuning("coneNormalEscapeCells", value)} modified={modified("coneNormalEscapeCells")} onReset={resetTuning("coneNormalEscapeCells")} />
        <PipeRange label="Emitter clearance" unit="cells" value={tuning.coneEmitterClearanceCells} min={0} max={8} step={0.25} digits={2}
          onChange={(value) => updateTuning("coneEmitterClearanceCells", value)} modified={modified("coneEmitterClearanceCells")} onReset={resetTuning("coneEmitterClearanceCells")} />
      </div>
      {/* The exact-ray budgets are the whole cost of a shadow under EXACT and
          are nearly inert under CONES, where the cone march has its own step
          budget and the exact traversal is never reached. They appear with the
          arm that spends them rather than sitting in a drawer that claims to
          govern the frame everyone actually renders. */}
      {svoConeTracingMode === "exact" && <div className="pipe-fields" data-testid="exact-visibility-budgets">
        <PipeRange label="Visibility nodes" unit="nodes" value={tuning.visibilityNodeVisits} min={1} max={128} step={1} digits={0}
          onChange={(value) => updateTuning("visibilityNodeVisits", value)} modified={modified("visibilityNodeVisits")} onReset={resetTuning("visibilityNodeVisits")} />
        <PipeRange label="Visibility leaves" unit="bricks" value={tuning.visibilityLeafVisits} min={1} max={32} step={1} digits={0}
          onChange={(value) => updateTuning("visibilityLeafVisits", value)} modified={modified("visibilityLeafVisits")} onReset={resetTuning("visibilityLeafVisits")} />
        <PipeRange label="Visibility voxel work" unit="tests" value={tuning.visibilityWorkItems} min={16} max={1024} step={16} digits={0}
          onChange={(value) => updateTuning("visibilityWorkItems", value)} modified={modified("visibilityWorkItems")} onReset={resetTuning("visibilityWorkItems")} />
        <PipeRange label="Intersections" unit="hits" value={tuning.visibilityIntersections} min={1} max={4} step={1} digits={0}
          onChange={(value) => updateTuning("visibilityIntersections", value)} modified={modified("visibilityIntersections")} onReset={resetTuning("visibilityIntersections")} />
      </div>}
      <div className="pipe-fields">
        <PipeRange label="Area samples · stable" unit="rays" value={tuning.stableAreaLightSamples} min={1} max={2} step={1} digits={0}
          onChange={(value) => updateTuning("stableAreaLightSamples", value)} modified={modified("stableAreaLightSamples")} onReset={resetTuning("stableAreaLightSamples")} />
        <PipeRange label="Area samples · moving" unit="rays" value={tuning.movingAreaLightSamples} min={1} max={2} step={1} digits={0}
          onChange={(value) => updateTuning("movingAreaLightSamples", value)} modified={modified("movingAreaLightSamples")} onReset={resetTuning("movingAreaLightSamples")} />
        <PipeRange label="AO samples · stable" unit="cones" value={tuning.stableAoSamples} min={1} max={4} step={1} digits={0}
          onChange={(value) => updateTuning("stableAoSamples", value)} modified={modified("stableAoSamples")} onReset={resetTuning("stableAoSamples")} />
        <PipeRange label="AO samples · moving" unit="cones" value={tuning.movingAoSamples} min={1} max={4} step={1} digits={0}
          onChange={(value) => updateTuning("movingAoSamples", value)} modified={modified("movingAoSamples")} onReset={resetTuning("movingAoSamples")} />
      </div>
      {/* Only the per-light plane consults the slot, so it appears with that
          plane rather than sitting inert beside the others. */}
      {svoRenderStageUsesLightSlot(svoStageView) && <div className="pipe-fields">
        <PipeRange label="Cached light slot" unit="slot" value={svoStageLightSlot}
          min={0} max={SVO_RENDER_STAGE_MAXIMUM_LIGHT_SLOT} step={1} digits={0}
          onChange={setSvoStageLightSlot}
          hint="Which of the eight cached per-light visibilities the prepass plane is decoded for." />
      </div>}
      </details>
      {(lightingVisibilityStatus.fallback || lightingVisibilityStatus.detail)
        && <p data-testid="lighting-visibility-status" aria-live="polite" className="render-inline-warning">
          Lighting visibility: {lightingVisibilityStatus.state.toUpperCase()}
          {lightingVisibilityStatus.fallback ? " FALLBACK" : ""}
          {lightingVisibilityStatus.detail ? ` · ${lightingVisibilityStatus.detail}` : ""}
        </p>}
    </>,

    "reduced-shade": <PipeChoice label="Reconstruction" value={tuning.coneRadianceReconstruction}
      disabled={tuning.coneLightingScale === 1 || svoConeTracingMode !== "cones"}
      onChange={(value: SvoConeRadianceReconstruction) => updateTuning("coneRadianceReconstruction", value)}
      options={[
        { value: "full-res-relight", label: "RELIGHT", hint: "Full-rate material and BRDF over the reduced visibility cache. The production arm: it preserves material and edge detail at either reduced rate." },
        { value: "wide-relight", label: "WIDE", hint: "Relight with an unguided wide gather of the reduced cache." },
        { value: "joint-bilateral", label: "BILAT", hint: "Guided upsample of the reduced radiance, weighted by depth, normal and identity." },
        { value: "gated-linear", label: "LINEAR", hint: "Bilinear upsample, gated off wherever the guide disagrees." },
        { value: "nearest", label: "EXACT", hint: "Nearest reduced texel. The fallback every other reconstruction mode lands on when its guide fails." },
      ]} />,

    "gi-composition": <details className="rp-tune"><summary>Bounce tuning</summary>
      <div className="pipe-fields" data-testid="gi-composition-controls"
      data-withheld={svoGlobalIlluminationEnabled ? undefined : "true"}>
      <PipeRange label="GI bounce" unit="%" disabled={!svoGlobalIlluminationEnabled} value={tuning.giBounceStrength * 100} min={0} max={400} step={5} digits={0}
        onChange={(value) => updateTuning("giBounceStrength", value / 100)} modified={modified("giBounceStrength")} onReset={resetTuning("giBounceStrength")}
        hint="Exposure for gathered diffuse bounce. This does not amplify direct highlights or emissive surfaces." />
      <PipeRange label="GI occlusion" unit="%" disabled={!svoGlobalIlluminationEnabled} value={tuning.giOcclusionStrength * 100} min={0} max={100} step={1} digits={0}
        onChange={(value) => updateTuning("giOcclusionStrength", value / 100)} modified={modified("giOcclusionStrength")} onReset={resetTuning("giOcclusionStrength")}
        hint="Uses the same wide GI cones to darken diffuse environment fill in enclosed regions. The AO toggle enables this in GLOBAL mode." />
      <PipeRange label="Diffuse environment" unit="%" disabled={!svoGlobalIlluminationEnabled} value={tuning.giEnvironmentStrength * 100} min={0} max={200} step={5} digits={0}
        onChange={(value) => updateTuning("giEnvironmentStrength", value / 100)} modified={modified("giEnvironmentStrength")} onReset={resetTuning("giEnvironmentStrength")}
        hint="Analytic sky fill retained alongside GI. Lower this when bounced light should carry more of the diffuse scene." />
      <PipeRange label="GI cone aperture" unit="rad" disabled={!svoGlobalIlluminationEnabled} value={tuning.giConeAperture} min={0.4} max={1.4} step={0.01} digits={2}
        onChange={(value) => updateTuning("giConeAperture", value)} modified={modified("giConeAperture")} onReset={resetTuning("giConeAperture")}
        hint="Wide apertures survey broad scene regions and produce smoother, lower-frequency bounce and occlusion." />
      <PipeRange label="GI cones" unit="cones" disabled={!svoGlobalIlluminationEnabled} value={tuning.giConeCount} min={3} max={4} step={1} digits={0}
        onChange={(value) => updateTuning("giConeCount", value)} modified={modified("giConeCount")} onReset={resetTuning("giConeCount")}
        hint="Four cones give the best hemispherical coverage; three trades the normal cone for longer marches at the same total budget." />
      </div>
    </details>,

    present: <div className="pipe-fields">
      <PipeRange label="Render resolution" unit="%" value={tuning.resolutionScale * 100}
        min={35} max={100} step={1} digits={0}
        onChange={(value) => updateTuning("resolutionScale", value / 100)}
        modified={modified("resolutionScale")} onReset={resetTuning("resolutionScale")}
        hint="Pixel-linear: the frame costs roughly 19–22 ms per megapixel with cones and 13–15 without, so this is the one dial that moves every pass at once." />
    </div>,
  };

  const graphTotal_ms = stageSpan_ms > 0 ? stageSpan_ms : stageTotal_ms;
  const walls = liveTiming ? bandWalls : undefined;

  const bands: readonly PipelineBand[] = RENDER_PIPELINE_BANDS.map((band): PipelineBand => {
    const bandCost = measureRenderPipelineBand(band.id, durations, graphTotal_ms, encodings);
    const bandWall_ms = walls?.get(band.id);
    const nodes = RENDER_PIPELINE_NODES.filter((node) => node.band === band.id);
    const entries = nodes.map((node) => {
      const state = node.state(context);
      const cost: RenderPipelineMeasurement = perPassSplit || encodings
        ? measureRenderPipelineNode(node, durations, graphTotal_ms, state, encodings)
        : { kind: "unmeasured", share: 0 };
      return { node, state, cost };
    });
    // The wall reads at a row's junction, not on the collar, whenever it can be
    // attributed — but only a row whose passes exist and cannot be timed is a
    // candidate. A row that encoded nothing is not "unmeasured", it is zero,
    // and handing it the band's residual is how a settled world came to report
    // 27.9 ms of maintenance. When more than one row is unpriceable, or none
    // is, the split is unknowable and the figure stays on the collar where it
    // reads as the band's own.
    const unpricedRows = entries.filter((entry) =>
      entry.state !== "unavailable"
      && (entry.cost.kind === "unpriced" || (entry.cost.kind === "unmeasured" && !encodings)));
    const wallRow = bandWall_ms !== undefined && unpricedRows.length === 1 ? unpricedRows[0] : undefined;
    if (wallRow && bandWall_ms !== undefined) {
      const compute_ms = entries.reduce((sum, entry) =>
        sum + (entry.cost.kind === "measured" ? entry.cost.duration_ms ?? 0 : 0), 0);
      wallRow.cost = {
        kind: "wall",
        duration_ms: Math.max(0, bandWall_ms - compute_ms),
        share: 0,
        unpricedRenderPasses: wallRow.cost.unpricedRenderPasses,
      };
    }
    const priced = perPassSplit && bandCost.duration_ms !== undefined;
    const rows: PipelineRow[] = [];
    for (const { node, state, cost } of entries) {
      // A run of rows on an unreachable arm reads as one collapsed row:
      // repeating `unavailable` per tier is diagram space spent on a path
      // the frame cannot take. The first member renders the placeholder;
      // the rest render nothing while every member is unreachable.
      if (node.collapseGroup) {
        const members = nodes.filter((other) => other.collapseGroup === node.collapseGroup);
        if (members.every((member) => member.state(context) === "unavailable")) {
          if (members.indexOf(node) !== 0) continue;
          const group = RENDER_PIPELINE_COLLAPSE_GROUPS[node.collapseGroup];
          const tip = `${group.label} · ${group.chip}\n\n${group.summary}`;
          rows.push({
            id: node.collapseGroup,
            label: group.label,
            state: "unavailable",
            tip,
            lamp: { kind: "switch", checked: true, disabled: true, ariaLabel: group.label, title: tip },
          });
          continue;
        }
      }
      const chip = node.chip(context);
      const tip = renderPipelineTipText(node, chip);
      const unavailable = state === "unavailable";
      const activeTap = node.taps.find((view) => view === svoStageView);
      const primaryTap = node.taps[0];
      const delta_ms = perPassSplit ? deltas.get(node.id) : undefined;
      rows.push({
        id: node.id,
        label: node.label,
        state,
        tip,
        chip,
        cost: {
          kind: cost.kind,
          duration_ms: cost.duration_ms,
          explanation: `${node.label}\n\n${costExplanation(cost)}`,
        },
        lamp: {
          kind: "switch",
          checked: state !== "off",
          disabled: !node.toggleable || unavailable,
          ariaLabel: node.label,
          title: tip,
          onToggle: () => toggleNode(node.id),
        },
        delta: delta_ms === undefined ? undefined : {
          ms: delta_ms,
          // The whole point of a switch: what the frame did when this last
          // moved. Measured across the toggle rather than derived from the
          // node's own phases, so it is the only figure that can price a node
          // whose work is a term inside somebody else's pass.
          title: `Switching this ${state === "off" ? "off" : "on"} moved the frame total by ${Math.abs(delta_ms).toFixed(2)} ms.\n\n`
            + "Measured: the averaged frame total under each setting, differenced. It includes anything downstream that got cheaper or dearer with it, which is why it can differ from the row's own cost.",
        },
        tap: primaryTap && !unavailable ? {
          label: "◨",
          title: `Present a plane this pass wrote instead of the composite.\n\n${tip}`,
          active: Boolean(activeTap),
          onToggle: () => setSvoStageView(activeTap ? "off" : primaryTap),
        } : undefined,
        planes: node.taps.length > 1 ? node.taps.map((view) => {
          const definition = SVO_RENDER_STAGE_DEFINITIONS[view];
          return {
            label: definition.label,
            title: `${definition.plane}\n\n${definition.description}`,
            active: svoStageView === view,
            onToggle: () => setSvoStageView(svoStageView === view ? "off" : view),
          };
        }) : undefined,
        controls: controls[node.id],
      });
    }
    return {
      id: band.id,
      label: band.label,
      cost_ms: priced ? bandCost.duration_ms : undefined,
      share: priced && graphTotal_ms > 0 ? bandCost.share : undefined,
      wall: bandWall_ms !== undefined && !wallRow ? {
        ms: bandWall_ms,
        title: `${formatPipelineDuration(bandWall_ms)} wall-clock for this band's whole encode segment, render passes included.\n\n`
          + `Measured by splitting one frame in sixteen at band boundaries into separate submits and timing each queue fence. Sampling frames are excluded from the frame mean.\n\n`
          + bandResidualNote(bandWall_ms, entries, encodings !== undefined),
      } : undefined,
      rows,
    };
  });

  return <>
    {/* One band: what is running, whether the figures are live, and what the
        frame costs. These were a status strip and a live strip stacked over the
        profile rung, three full-width bars before the first measurement. */}
    <div className="render-status-line">
      <span className={effectiveRendererStatus.state === "active" ? "online" : ""} />
      <strong data-testid="effective-renderer-status">{effectiveRendererStatus.state === "active"
        ? lightingVisibilityStatus.state === "cones" ? "SVO GI"
          : lightingVisibilityStatus.state === "exact" ? "SVO exact" : "SVO direct"
        : effectiveRendererStatus.state === "not-required" ? "SVO not required"
        : effectiveRendererStatus.state === "pending" ? "SVO pending" : "SVO failed closed"}</strong>
      <PipeToggle label="Live" checked={liveTiming} onChange={setLiveTiming}
        hint={`Continuously records GPU pass timestamps with one readback in flight, then reads each trustworthy node cost from them. Measurement is real work, so it stops when this is off and when the overlay closes.\n\n${measured
          ? `Frame source: ${trace?.measurementSource === "gpu-hardware-timestamp" ? "hardware timestamps" : "GPU queue wall"}. Stage source: ${hardwarePerPassSplit ? "hardware timestamps" : "unavailable"}. ${TRACE_WINDOW}-frame mean.`
          : "No trace yet."}`} />
      <code data-testid="render-frame-cost" title={timingHint}>{timingLabel}</code>
      {effectiveRendererStatus.terminalCounts && <code data-testid="svo-terminal-counts"
        title="Accepted unified SVO leaf terminals. Planar terminals keep exact thin slab geometry without voxel payload traversal.">
        {effectiveRendererStatus.terminalCounts.planarBoundary} planar · {effectiveRendererStatus.terminalCounts.voxel} voxel
      </code>}
    </div>

    {/* The profile rung is the question asked before any node is opened — how
        expensive is this frame allowed to be — so it stays above the graph
        rather than inside a node that can be collapsed over it. */}
    <div className="render-preset-strip" role="group" aria-label="Render performance profile">
      <span>Profile</span>
      {(Object.keys(SVO_RENDER_QUALITY_PRESETS) as SvoRenderQualityPreset[]).map((preset) =>
        <button key={preset} type="button" className={activePreset === preset ? "active" : ""} onClick={() => {
          // One click, both halves. The visibility mode has its own buttons on
          // the cone node and stays reachable there; what this strip guarantees
          // is that a named rung is never half-applied.
          setTuning(SVO_RENDER_QUALITY_PRESETS[preset].tuning);
          setSvoConeTracingMode(SVO_RENDER_QUALITY_PRESETS[preset].coneTracingMode);
        }}>{preset}</button>)}
      {!activePreset && <output>custom</output>}
      {/* A withheld stage is easy to forget and looks like a bug from the
          viewport, so the count is always visible once there is one and the way
          back is one click from where it is stated. */}
      {disabledRenderStages.length > 0 && <button type="button" className="render-withheld-reset"
        data-testid="withheld-stage-count"
        title={`Withheld from the encode:\n${disabledRenderStages.map((stage) => `· ${stage.replace(/-/g, " ")}`).join("\n")}\n\nRestore all of them.`}
        onClick={() => { for (const stage of disabledRenderStages) setRenderStageDisabled(stage, false); }}>
        {disabledRenderStages.length} withheld ↺
      </button>}
    </div>

    {effectiveRendererStatus.failureReason && <p className="render-inline-warning">SVO unavailable: {effectiveRendererStatus.detail
      ?? rendererFailureLabels[effectiveRendererStatus.failureReason]}.</p>}

    <PipelineGraph bands={bands} testId="render-pipeline" />
  </>;
}
