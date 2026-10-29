"use client";

import { renderSparseWorldBuildControls } from "../features/construction/controls";
import { renderConeVisibilityControls } from "../features/lighting-visibility/controls";
import { renderPresentControls } from "../features/presentation/controls";
import { renderFilteredDetailControls,renderPrimaryTraversalControls,renderSeamClosureControls } from "../features/primary-visibility/controls";
import { renderGiCompositionControls,renderReducedShadeControls } from "../features/radiance/controls";

import { useEffect,useMemo } from "react";
import { Facts } from "../../../components/ui";
import { PipelineCost, formatPipelineDuration, type PipelineBand, type PipelineRow } from "../../../components/PipelineCost";
import { RailReadout, RailSelect, RailToggle } from "../../../components/rail";
import { sceneUsesFlatVoxelNormals } from "../../core/model";
import {
averagePerformanceTraces,
type PerformanceTrace,
} from "../../core/performance-trace";
import { svoRenderRefinementPermitted } from "../../core/method-registry";
import { useSession } from "../../core/session/session-context";
import { usePerformanceInstrumentationStore } from "../../core/stores/performance-instrumentation-store";
import {
SVO_RENDER_STAGE_DEFINITIONS,
svoRenderStageUsesPrimaryWorkMap,
} from "../features/diagnostics/svo-render-diagnostics";
import {
RENDER_PIPELINE_BANDS,
RENDER_PIPELINE_COLLAPSE_GROUPS,
RENDER_PIPELINE_NODES,
measureRenderPipelineBand,
measureRenderPipelineNode,
renderPipelineNodeForContext,
renderPipelineStageDurations,
renderPipelineTipText,
renderPipelineUnownedPhases,
type RenderPipelineContext,
type RenderPipelineMeasurement,
} from "./render-pipeline-graph";
import type { SvoStageControls } from "./control-context";
import { disabledRenderStagesFrom } from "./render-stage-switches";
import { resolveSvoPrimaryTraversal } from "./svo-render-options";
import {
SVO_RENDER_QUALITY_PRESETS,
svoRenderTuningKey,
type SvoRenderQualityPreset,
type SvoRenderTuning,
} from "./svo-render-tuning";
import { svoSceneTuningDefaults } from "./tuning-binding";
import { SvoFeatureSlot } from "./ui-slots";

/** Frames the live readout averages over. */
const TRACE_WINDOW = 12;

const rendererFailureLabels = {
  "missing-source": "waiting for structural SVO data",
  "unsupported-terrain": "terrain source could not be represented",
  "missing-pbr-materials": "production PBR material table is unavailable",
  "missing-lighting-publications": "production light/environment publications are unavailable",
  "pipeline-compile-failure": "SVO pipeline failed to compile",
  "pipeline-compiling": "SVO pipeline is compiling",
  "frame-rejected": "live SVO frame publication was rejected",
} as const;

/** Why the figure is the kind of number it is, in frame terms. */
function costExplanation(cost: RenderPipelineMeasurement): string {
  switch (cost.kind) {
    case "withheld":
      return "0 ms — encoded no work in the sampled frame.";
    case "idle":
      return "0 ms — encoded no GPU work in the sampled frames.";
    case "shared":
      return `Runs inside ${cost.insideNode?.replace(/-/g, " ")}; the figure is that stage's.`;
    case "structural":
      return "A gate, not a pass.";
    case "unmeasured":
      return "No stage sample yet.";
    default:
      return `${formatPipelineDuration(cost.duration_ms ?? 0)} GPU, mean of the trace window.${cost.encodedFraction !== undefined
        ? ` Encoded in ${Math.round(cost.encodedFraction * 100)}% of frames; per-frame expectation.`
        : ""}`;
  }
}

/**
 * The frame's exclusive GPU completion partition, grouped at the same
 * colocated plug-in seams that own the renderer stages.
 */
function usePresentationTiming(): {
  readonly total?: PerformanceTrace;
  readonly stages?: PerformanceTrace;
  readonly cpu_ms?: number;
  readonly queue_ms?: number;
  readonly extractionCount?: number;
  readonly extractionReason?: string;
} {
  const session = useSession();
  const reports = session.diagnostics((state) => state.performanceReports);
  return useMemo(() => {
    const newest = reports.findLast((report) => report.presentation);
    if (!newest) return {};
    const recent = reports
      .filter((report) => report.context === newest.context)
      .slice(-TRACE_WINDOW);
    const presentation = recent
      .map((report) => report.presentation)
      .filter((trace): trace is PerformanceTrace => trace !== undefined);
    const latest = presentation.at(-1);
    const matching = latest
      ? presentation.filter((trace) => trace.measurementSource === latest.measurementSource)
      : [];
    const mean = averagePerformanceTraces(matching);
    const stageSamples = recent
      .map((report) => report.presentationStages)
      .filter((trace): trace is PerformanceTrace => trace !== undefined
        && trace.measurementSource === "gpu-pass-timestamp");
    const cpuSamples = recent.flatMap(report => report.cpu ? [report.cpu.total_ms] : []);
    const queueSamples = recent.flatMap(report => report.presentationQueue ? [report.presentationQueue.total_ms] : []);
    const average = (values: number[]) => values.length ? values.reduce((a, b) => a + b, 0) / values.length : undefined;
    return { total: mean, stages: averagePerformanceTraces(stageSamples), cpu_ms: average(cpuSamples), queue_ms: average(queueSamples), extractionReason: newest.surfaceExtractionReason, extractionCount: newest.surfaceExtractionCount };
  }, [reports]);
}


/**
 * Samples the renderer for as long as it is mounted, and hands the setting
 * back exactly as it found it: measurement is work the product did not ask
 * for, so it runs only while a reading of it is open. It never overrides an
 * explicit choice made elsewhere, which is why the guard is on `off`.
 */
function useFrameSampling() {
  useEffect(() => {
    const store = usePerformanceInstrumentationStore.getState();
    if (store.mode !== "off") return;
    store.setMode("timeline");
    return () => {
      const current = usePerformanceInstrumentationStore.getState();
      if (current.mode === "timeline") current.setMode("off");
    };
  }, []);
}

/**
 * The frame graph's view model: the renderer's stages in encode order, priced
 * by the frame's exclusive GPU completion partition, each with its own switch,
 * planes and controls.
 */
function useRenderPipelineBands(): { readonly bands: readonly PipelineBand[]; readonly frame: string; readonly source: string } {
  const session = useSession();
  const effectiveRendererStatus = session.diagnostics((state) => state.effectiveRendererStatus);
  const svoShadowsEnabled = session.ui((state) => state.svoShadowsEnabled);
  const svoAmbientOcclusionEnabled = session.ui((state) => state.svoAmbientOcclusionEnabled);
  const silhouetteRefinementEnabled = session.ui((state) => state.silhouetteRefinementEnabled);
  const setSilhouetteRefinementEnabled = session.ui((state) => state.setSilhouetteRefinementEnabled);
  const svoPrimaryTraversal = session.ui((state) => state.svoPrimaryTraversal);
  const svoConeTracingMode = session.ui((state) => state.svoConeTracingMode);
  const setSvoConeTracingMode = session.ui((state) => state.setSvoConeTracingMode);
  const svoGlobalIlluminationEnabled = session.ui((state) => state.svoGlobalIlluminationEnabled);
  const setSvoGlobalIlluminationEnabled = session.ui((state) => state.setSvoGlobalIlluminationEnabled);
  const svoWorldGiCacheEnabled = session.ui((state) => state.svoWorldGiCacheEnabled);
  const svoLatticeVisibilityEnabled = session.ui((state) => state.svoLatticeVisibilityEnabled);
  const setSvoWorldGiCacheEnabled = session.ui((state) => state.setSvoWorldGiCacheEnabled);
  const disabledRenderStages = session.ui((state) => state.disabledRenderStages);
  const setRenderStageDisabled = session.ui((state) => state.setRenderStageDisabled);
  const svoStageView = session.ui((state) => state.svoStageView);
  const resolvedPrimary = svoRenderStageUsesPrimaryWorkMap(svoStageView) ? "traced" : resolveSvoPrimaryTraversal(svoPrimaryTraversal);
  const setSvoStageView = session.ui((state) => state.setSvoStageView);
  const svoStageLightSlot = session.ui((state) => state.svoStageLightSlot);
  const setSvoStageLightSlot = session.ui((state) => state.setSvoStageLightSlot);
  const svoMaximumTraversalDepth = session.ui((state) => state.svoMaximumTraversalDepth);
  const setSvoMaximumTraversalDepth = session.ui((state) => state.setSvoMaximumTraversalDepth);
  const svoMaximumNodeVisits = session.ui((state) => state.svoMaximumNodeVisits);
  const setSvoMaximumNodeVisits = session.ui((state) => state.setSvoMaximumNodeVisits);
  const tuning = session.ui((state) => state.svoRenderTuning);
  const renderProfile = session.scene((state) => state.scene.renderProfile);
  const sceneTuningDefaults = svoSceneTuningDefaults(renderProfile);
  const setTuning = session.ui((state) => state.setSvoRenderTuning);

  // The scene facts the build node has to state for itself: what a leaf
  // actually measures, and whether the depth is legal at all. A simulated scene
  // pins every brick's node at the solver level, so the ladder is a no-op there
  // and has to say so rather than move.
  const finestCellSize_m = session.scene((state) => state.scene.voxelDomain.finestCellSize_m);
  const sceneIsDry = session.scene((state) => state.scene.systems?.fluid === false);
  const surfaceStyle = session.scene((state) => state.scene.surfaceStyle);
  const smoothSurfaceEnabled = !sceneUsesFlatVoxelNormals({ surfaceStyle });
  const methodId = session.method((state) => state.methodId);
  const renderRefinementPermitted = svoRenderRefinementPermitted(
    { systems: sceneIsDry ? { fluid: false } : undefined }, methodId);
  const renderRefinementDepth = renderRefinementPermitted ? tuning.environmentRefinementDepth : 0;
  const leafVoxel_mm = finestCellSize_m * 1000 / 2 ** renderRefinementDepth;

  const timing = usePresentationTiming();
  const trace = timing.total;
  const stageTrace = timing.stages;
  const durations = useMemo(() => renderPipelineStageDurations(stageTrace), [stageTrace]);
  // A trace label that names no stage of the ABI is a measurement bug — a seam
  // publishing a label the registry does not own — and it is louder as a
  // console warning than as a row that quietly reads zero.
  const unowned = useMemo(() => renderPipelineUnownedPhases(stageTrace), [stageTrace]);
  useEffect(() => {
    if (unowned.length > 0) console.warn("Render trace phases owned by no pipeline stage:", unowned);
  }, [unowned]);
  const partitioned = stageTrace !== undefined;

  const updateTuning = <K extends keyof SvoRenderTuning>(key: K, value: SvoRenderTuning[K]) =>
    setTuning((current) => ({ ...current, [key]: value,
      ...(key === "stableAreaLightSamples" ? { movingAreaLightSamples: value as number } : {}),
      ...(key === "stableAoSamples" ? { movingAoSamples: value as number } : {}),
    }));
  const modified = <K extends keyof SvoRenderTuning>(key: K) => tuning[key] !== sceneTuningDefaults[key];
  const resetTuning = <K extends keyof SvoRenderTuning>(key: K) => () =>
    updateTuning(key, sceneTuningDefaults[key]);

  const silhouetteRefinementStatus = effectiveRendererStatus.silhouetteRefinement ?? {
    state: effectiveRendererStatus.state === "pending" ? "compiling" as const
      : effectiveRendererStatus.state === "failed" ? "failed" as const
      : silhouetteRefinementEnabled ? "enabled" as const : "disabled" as const,
    detail: effectiveRendererStatus.detail,
  };
  const lightingVisibilityStatus = effectiveRendererStatus.lightingVisibility ?? { state: svoConeTracingMode };

  const disabledStages = useMemo(
    () => disabledRenderStagesFrom(disabledRenderStages), [disabledRenderStages]);

  const context: RenderPipelineContext = {
    disabledStages,
    coneTracingMode: svoConeTracingMode,
    shadowsEnabled: svoShadowsEnabled,
    ambientOcclusionEnabled: svoAmbientOcclusionEnabled,
    seamClosureEnabled: silhouetteRefinementEnabled,
    globalIlluminationEnabled: svoGlobalIlluminationEnabled,
    worldGiCacheEnabled: svoWorldGiCacheEnabled,
    latticeVisibilityEnabled: svoLatticeVisibilityEnabled,
    tuning,
    sceneHasFluid: !sceneIsDry,
    refinementDepth: renderRefinementDepth,
    leafVoxel_mm,
    rendererActive: effectiveRendererStatus.state === "active",
    stageView: svoStageView,
    // Asked of the shared rule rather than assumed: it answers `traced` for
    // every production frame, so the three raster-only tiers read as
    // unavailable — but under FLUID_SVO_PRIMARY_TRAVERSAL=raster they are live
    // passes with live switches.
    rasterPrimaryActive: resolvedPrimary !== "traced",
    surfaceMeshSelected: resolvedPrimary === "mesh",
    smoothSurfaceEnabled,
    surfaceMeshActive: resolvedPrimary === "mesh" && effectiveRendererStatus.surfaceMesh?.state === "ready",
    surfaceMeshStatus: effectiveRendererStatus.surfaceMesh,
  };

  // Most switches route through the encode-time ablation set, which withholds
  // the pass outright. Cone visibility, GI composition, the world-space GI
  // cache and seam closure are switched by contracts the shaders already
  // compile against. The graph says which is which.
  const toggleNode = (id: string) => {
    const node = RENDER_PIPELINE_NODES.find((candidate) => candidate.id === id);
    if (node?.stage) {
      setRenderStageDisabled(node.stage, !disabledStages.has(node.stage));
      return;
    }
    if (id === "filtered-detail") updateTuning("surfaceMeshFilteringEnabled", !tuning.surfaceMeshFilteringEnabled);
    else if (id === "seam-closure") setSilhouetteRefinementEnabled(!silhouetteRefinementEnabled);
    else if (id === "cone-visibility") setSvoConeTracingMode(svoConeTracingMode === "off" ? "cones" : "off");
    else if (id === "gi-composition") setSvoGlobalIlluminationEnabled(!svoGlobalIlluminationEnabled);
    else if (id === "world-gi-cache") setSvoWorldGiCacheEnabled(!svoWorldGiCacheEnabled);
  };

  const controls: Readonly<Record<string, SvoStageControls>> = {
    "sparse-world-build": renderSparseWorldBuildControls({ renderRefinementDepth, renderRefinementPermitted, sceneIsDry, updateTuning, modified, resetTuning, leafVoxel_mm, finestCellSize_m, tuning }),
    "filtered-detail": renderFilteredDetailControls({ resolvedPrimary, smoothSurfaceEnabled, tuning, updateTuning, effectiveRendererStatus, svoStageView, setSvoStageView }),
    [resolvedPrimary === "mesh" ? "surface-mesh-update" : "primary-traversal"]: renderPrimaryTraversalControls({ resolvedPrimary, effectiveRendererStatus, smoothSurfaceEnabled, svoMaximumTraversalDepth, setSvoMaximumTraversalDepth, svoMaximumNodeVisits, setSvoMaximumNodeVisits, tuning, updateTuning, modified, resetTuning }),
    "surface-mesh-cull": { node: <Facts items={[{ label: "Drawn mesh records", value: effectiveRendererStatus.surfaceMesh?.quads?.toLocaleString() ?? "—", hint: "Records surviving current-camera culling. Updated asynchronously." }]} /> },
    "seam-closure": renderSeamClosureControls({ silhouetteRefinementStatus }),
    "cone-visibility": renderConeVisibilityControls({ svoConeTracingMode, tuning, updateTuning, modified, resetTuning, svoStageView, svoStageLightSlot, setSvoStageLightSlot, lightingVisibilityStatus }),
    "reduced-shade": renderReducedShadeControls(),
    "gi-composition": renderGiCompositionControls({ svoGlobalIlluminationEnabled, svoConeTracingMode, tuning, updateTuning, modified, resetTuning }),
    present: renderPresentControls({ tuning, updateTuning, modified, resetTuning }),
  };

  // Stage-span traces keep an additive phase sum in total_ms for trace
  // accounting, but the graph's denominator is the actual GPU frame span.
  const graphTotal_ms = stageTrace?.total_ms ?? 0;

  const bands = RENDER_PIPELINE_BANDS.map((band): PipelineBand => {
    const bandCost = measureRenderPipelineBand(band.id, durations, graphTotal_ms);
    const nodes = RENDER_PIPELINE_NODES.filter((node) => node.band === band.id
      && (context.surfaceMeshSelected || (node.id !== "surface-mesh-update" && node.id !== "surface-mesh-cull")))
      .map((node) => renderPipelineNodeForContext(node, context));
    const rows: PipelineRow[] = [];
    for (const node of nodes) {
      const state = node.state(context);
      // A run of rows on an unreachable arm reads as one row: repeating
      // `unavailable` per tier is space spent on a path the frame cannot take.
      if (node.collapseGroup) {
        const members = nodes.filter((other) => other.collapseGroup === node.collapseGroup);
        if (members.every((member) => member.state(context) === "unavailable")) {
          if (members.indexOf(node) !== 0) continue;
          const group = RENDER_PIPELINE_COLLAPSE_GROUPS[node.collapseGroup];
          const tip = `${group.label} · ${group.chip}\n\n${group.summary}`;
          rows.push({ id: node.collapseGroup, label: group.label, state: "unavailable", tip,
            lamp: { kind: "switch", checked: true, disabled: true, ariaLabel: group.label, title: tip } });
          continue;
        }
      }
      const cost: RenderPipelineMeasurement = partitioned
        ? measureRenderPipelineNode(node, durations, graphTotal_ms, state)
        : { kind: "unmeasured", share: 0 };
      const chip = node.chip(context);
      const tip = renderPipelineTipText(node, chip);
      const unavailable = state === "unavailable";
      const stage = controls[node.id];
      rows.push({
        id: node.id,
        label: node.label,
        state,
        tip,
        chip,
        cost: { kind: cost.kind, duration_ms: cost.duration_ms, encodedFraction: cost.encodedFraction, explanation: costExplanation(cost) },
        lamp: {
          kind: "switch",
          checked: node.id === "world-gi-cache" ? svoWorldGiCacheEnabled
            : node.id === "gi-composition" ? svoGlobalIlluminationEnabled : state !== "off",
          disabled: !node.toggleable || unavailable,
          ariaLabel: node.label,
          title: tip,
          onToggle: () => toggleNode(node.id),
        },
        // The planes this pass wrote, presented instead of the composite.
        planes: unavailable ? undefined : node.taps.map((view) => {
          const definition = SVO_RENDER_STAGE_DEFINITIONS[view];
          return {
            label: definition.label,
            title: `${definition.plane}\n\n${definition.description}`,
            active: svoStageView === view,
            onToggle: () => setSvoStageView(svoStageView === view ? "off" : view),
          };
        }),
        controls: stage?.node,
        notice: stage?.notice,
      });
    }
    const priced = partitioned && bandCost.duration_ms !== undefined;
    return {
      id: band.id,
      label: band.label,
      cost_ms: priced ? bandCost.duration_ms : undefined,
      share: priced && graphTotal_ms > 0 ? bandCost.share : undefined,
      rows,
    };
  });

  return {
    bands,
    frame: trace ? formatPipelineDuration(trace.total_ms) : "—",
    source: !trace ? "Waiting for a sampled frame."
      : trace.measurementSource === "gpu-queue-wall"
        ? "Queue completion wall time: includes backlog and callback delivery."
        : `GPU execution span, mean of up to ${TRACE_WINDOW} frames${partitioned ? "; one exclusive timing per stage" : ""}. Not the reciprocal of fps.`,
  };
}

/** The quality rung: how expensive the frame is allowed to be. */
function FrameProfileLine() {
  const session = useSession();
  const tuning = session.ui((state) => state.svoRenderTuning);
  const setTuning = session.ui((state) => state.setSvoRenderTuning);
  const coneTracingMode = session.ui((state) => state.svoConeTracingMode);
  const setSvoConeTracingMode = session.ui((state) => state.setSvoConeTracingMode);
  const tuningKey = svoRenderTuningKey(tuning);
  // A rung is the pair, so the match is on the pair. `quality` and `reference`
  // carry the same sliders and differ only in how visibility is answered.
  const presets = Object.keys(SVO_RENDER_QUALITY_PRESETS) as SvoRenderQualityPreset[];
  const active = presets.find((preset) => svoRenderTuningKey(SVO_RENDER_QUALITY_PRESETS[preset].tuning) === tuningKey
    && SVO_RENDER_QUALITY_PRESETS[preset].coneTracingMode === coneTracingMode);
  return <RailSelect label="Profile" value={active ?? "custom"} testId="render-profile"
    options={[...presets.map((preset) => ({ value: preset, label: preset })), ...(active ? [] : [{ value: "custom", label: "custom" }])]}
    onChange={(preset) => {
      if (preset === "custom") return;
      // One choice, both halves: a named rung is never half-applied.
      setTuning(SVO_RENDER_QUALITY_PRESETS[preset as SvoRenderQualityPreset].tuning);
      setSvoConeTracingMode(SVO_RENDER_QUALITY_PRESETS[preset as SvoRenderQualityPreset].coneTracingMode);
    }} />;
}

/** What a frame costs: the rung, the total, then a line per band. */
export function FrameCostLines() {
  useFrameSampling();
  const session = useSession();
  const status = session.diagnostics((state) => state.effectiveRendererStatus);
  const withheld = session.ui((state) => state.disabledRenderStages);
  const setRenderStageDisabled = session.ui((state) => state.setRenderStageDisabled);
  const cost = useRenderPipelineBands();
  return <>
    {status.failureReason && <small className="rail-note" data-tone="attention" title={status.detail ?? undefined}>
      SVO unavailable: {rendererFailureLabels[status.failureReason]}</small>}
    <FrameProfileLine />
    <RailReadout label="Frame" value={cost.frame} hint={cost.source} testId="render-frame-cost" />
    <PipelineCost id="render-pipeline" bands={cost.bands} />
    {/* A withheld stage looks like a bug from the viewport, so the count stays
        in view and the way back is one click from where it is stated. */}
    {withheld.length > 0 && <button type="button" className="rail-text-button" data-tone="attention" data-testid="withheld-stage-count"
      title={withheld.map((stage) => stage.replace(/-/g, " ")).join("\n")}
      onClick={() => { for (const stage of withheld) setRenderStageDisabled(stage, false); }}>
      {withheld.length} withheld · restore
    </button>}
  </>;
}

/** Whether More was left open, for the life of the page. */
let frameMoreOpen = false;

/** The frame's other facts and its surface options, folded. */
export function FrameMoreLines() {
  const session = useSession();
  const status = session.diagnostics((state) => state.effectiveRendererStatus);
  const surfaceStyle = session.scene((state) => state.scene.surfaceStyle);
  const patchScene = session.scene((state) => state.patchScene);
  const lighting = status.lightingVisibility?.state ?? session.ui.getState().svoConeTracingMode;
  const timing = usePresentationTiming();
  const smooth = !sceneUsesFlatVoxelNormals({ surfaceStyle });
  return <details className="rail-more" open={frameMoreOpen} onToggle={(event) => { frameMoreOpen = event.currentTarget.open; }}>
    <summary>More</summary>
    <RailReadout label="Renderer" testId="effective-renderer-status" hint={status.detail ?? undefined}
      value={status.state === "active" ? lighting === "cones" ? "SVO GI" : lighting === "exact" ? "SVO exact"
        : lighting === "raster-ao" ? "SVO raster AO" : "SVO direct"
        : status.state === "not-required" ? "not required" : status.state === "pending" ? "pending" : "failed"} />
    {timing.queue_ms !== undefined && <RailReadout label="Host · queue" testId="render-wall-timing"
      hint="Host frame work, then submit-to-completion wall time. Both overlap GPU execution."
      value={`${timing.cpu_ms?.toFixed(2) ?? "—"} · ${timing.queue_ms.toFixed(2)} ms`} />}
    {status.terminalCounts && <RailReadout label="Terminals" testId="svo-terminal-counts"
      value={`${status.terminalCounts.planarBoundary} planar · ${status.terminalCounts.voxel} voxel`} />}
    {timing.extractionCount !== undefined && <RailReadout label="Mesh builds" hint={timing.extractionReason}
      value={timing.extractionCount} />}
    <SvoFeatureSlot slot="frame.options" />
    <RailToggle label="Smooth surface" checked={smooth}
      onChange={(enabled) => patchScene({ surfaceStyle: enabled ? "smooth" : "voxel-flat" })}
      hint="Reconstruct a continuous surface instead of drawing axis-aligned voxel faces." />
    {status.surfaceMesh?.state === "blocked" && <small className="rail-note">{status.surfaceMesh.detail}</small>}
  </details>;
}
