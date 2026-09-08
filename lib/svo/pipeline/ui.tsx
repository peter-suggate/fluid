"use client";

import { renderSparseWorldBuildControls } from "../features/construction/controls";
import { renderConeVisibilityControls } from "../features/lighting-visibility/controls";
import { renderPresentControls } from "../features/presentation/controls";
import { renderPrimaryTraversalControls,renderSeamClosureControls } from "../features/primary-visibility/controls";
import { renderGiCompositionControls,renderReducedShadeControls } from "../features/radiance/controls";

import { useEffect,useMemo,useState,type ReactNode } from "react";
import { PipeToggle } from "../../../components/PipeControls";
import { SVO_SURFACE_MESH_LOD_PIXELS_DEFAULT } from "./svo-render-tuning";
import {
PipelineGraph,
formatPipelineDuration,
type PipelineBand,
type PipelineRow,
} from "../../../components/PipelineGraph";
import { sceneUsesFlatVoxelNormals } from "../../core/model";
import {
averagePerformanceTraces,
type PerformanceTrace,
} from "../../core/performance-trace";
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
import { disabledRenderStagesFrom } from "./render-stage-switches";
import { resolveSvoPrimaryTraversal } from "./svo-render-options";
import {
SVO_RENDER_QUALITY_PRESETS,
SVO_RENDER_TUNING_PRESETS,
svoRenderTuningKey,
type SvoRenderQualityPreset,
type SvoRenderTuning,
} from "./svo-render-tuning";
import { SvoFeatureSlot } from "./ui-slots";

/** Frames the live readout averages over. */
const TRACE_WINDOW = 12;

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

/** Why the figure on the pipe is the kind of number it is, in frame terms. */
function costExplanation(cost: RenderPipelineMeasurement): string {
  switch (cost.kind) {
    case "withheld":
      return "0 ms — this stage encoded no work in the sampled frame.";
    case "idle":
      return "0 ms — the frame partition arrived, and this stage encoded no GPU work in the sampled frames.";
    case "shared":
      return `Not a stage of its own: this work runs inside ${cost.insideNode?.replace(/-/g, " ").toUpperCase()}, and ⊂ marks the figure as that stage's.`;
    case "structural":
      return "A gate, not a pass. It spends no frame time either way — its worth shows up as the row it lets the frame skip going to zero.";
    case "unmeasured":
      return "No stage sample has arrived yet.";
    default:
      return `${formatPipelineDuration(cost.duration_ms ?? 0)} for this stage on the frame's exclusive GPU completion timeline.${cost.encodedFraction !== undefined
        ? `\n\nEncoded in ${Math.round(cost.encodedFraction * 100)}% of sampled frames; the figure is the expected cost per frame, not the per-encode mean.`
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
    return { total: mean, stages: averagePerformanceTraces(stageSamples) };
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
  const session = useSession();
  const effectiveRendererStatus = session.diagnostics((state) => state.effectiveRendererStatus);
  const svoShadowsEnabled = session.ui((state) => state.svoShadowsEnabled);
  const setSvoShadowsEnabled = session.ui((state) => state.setSvoShadowsEnabled);
  const svoAmbientOcclusionEnabled = session.ui((state) => state.svoAmbientOcclusionEnabled);
  const setSvoAmbientOcclusionEnabled = session.ui((state) => state.setSvoAmbientOcclusionEnabled);
  const silhouetteRefinementEnabled = session.ui((state) => state.silhouetteRefinementEnabled);
  const setSilhouetteRefinementEnabled = session.ui((state) => state.setSilhouetteRefinementEnabled);
  const svoPrimaryTraversal = session.ui((state) => state.svoPrimaryTraversal);
  const svoConeTracingMode = session.ui((state) => state.svoConeTracingMode);
  const setSvoConeTracingMode = session.ui((state) => state.setSvoConeTracingMode);
  const svoGlobalIlluminationEnabled = session.ui((state) => state.svoGlobalIlluminationEnabled);
  const setSvoGlobalIlluminationEnabled = session.ui((state) => state.setSvoGlobalIlluminationEnabled);
  const svoWorldGiCacheEnabled = session.ui((state) => state.svoWorldGiCacheEnabled);
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
  const setTuning = session.ui((state) => state.setSvoRenderTuning);

  // The scene facts the build node has to state for itself: what a leaf
  // actually measures, and whether the depth is legal at all. A simulated scene
  // pins every brick's node at the solver level, so the ladder is a no-op there
  // and has to say so rather than move.
  const finestCellSize_m = session.scene((state) => state.scene.voxelDomain.finestCellSize_m);
  const sceneIsDry = session.scene((state) => state.scene.systems?.fluid === false);
  const surfaceStyle = session.scene((state) => state.scene.surfaceStyle);
  const patchScene = session.scene((state) => state.patchScene);
  const smoothSurfaceEnabled = !sceneUsesFlatVoxelNormals({ surfaceStyle });
  const renderRefinementDepth = sceneIsDry ? tuning.environmentRefinementDepth : 0;
  const leafVoxel_mm = finestCellSize_m * 1000 / 2 ** renderRefinementDepth;

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
  const trace = liveTiming ? timing.total : undefined;
  const stageTrace = timing.stages;
  const durations = useMemo(
    () => renderPipelineStageDurations(liveTiming ? stageTrace : undefined), [stageTrace, liveTiming]);
  // A trace label that names no stage of the ABI is a measurement bug — a seam
  // publishing a label the registry does not own — and it is louder as a
  // console warning than as a row that quietly reads zero.
  const unowned = useMemo(() => renderPipelineUnownedPhases(stageTrace), [stageTrace]);
  useEffect(() => {
    if (unowned.length > 0) console.warn("Render trace phases owned by no pipeline stage:", unowned);
  }, [unowned]);
  const measured = liveTiming && trace !== undefined;
  const total_ms = trace?.total_ms ?? 0;

  const updateTuning = <K extends keyof SvoRenderTuning>(key: K, value: SvoRenderTuning[K]) =>
    setTuning((current) => ({ ...current, [key]: value,
      ...(key === "stableAreaLightSamples" ? { movingAreaLightSamples: value as number } : {}),
      ...(key === "stableAoSamples" ? { movingAoSamples: value as number } : {}),
    }));
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

  const partitioned = liveTiming && stageTrace !== undefined;
  const timingLabel = !liveTiming ? "timing off"
    : !measured ? "awaiting trace"
    : `${total_ms.toFixed(2)} ms/frame`;
  const timingHint = !liveTiming
    ? "Live frame timing is off. Turn it on to sample the presentation boundary chain."
    : !measured
      ? "No presentation trace has arrived yet."
      : partitioned
        ? `One exclusive timing per stage, cut at the same plug-in seam that owns it in code. Overlapped work is charged once, to the stage that advances GPU completion. ${TRACE_WINDOW}-sample mean.`
        : "The frame total is ready; this device has not supplied a detailed stage sample.";

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
    tuning,
    sceneHasFluid: !sceneIsDry,
    refinementDepth: renderRefinementDepth,
    leafVoxel_mm,
    rendererActive: effectiveRendererStatus.state === "active",
    stageView: svoStageView,
    // Asked of the shared rule rather than assumed: it answers `traced` for
    // every production frame, so the three raster-only tiers read as
    // unavailable — but under FLUID_SVO_PRIMARY_TRAVERSAL=raster they are live
    // passes with live switches, and the panel should say which frame it is
    // looking at rather than hard-coding one.
    rasterPrimaryActive: resolvedPrimary !== "traced",
    surfaceMeshSelected: resolvedPrimary === "mesh",
    surfaceMeshActive: resolvedPrimary === "mesh" && !smoothSurfaceEnabled && effectiveRendererStatus.surfaceMesh?.state === "ready",
    surfaceMeshStatus: effectiveRendererStatus.surfaceMesh,
  };

  // The lamp is the node's own switch, and every node has one.
  //
  // Most route through the encode-time ablation set, which withholds the pass
  // outright. Cone visibility, GI composition, the world-space GI cache and
  // seam closure are switched by contracts the shaders already compile
  // against. The graph says which is which.
  const toggleNode = (id: string) => {
    const node = RENDER_PIPELINE_NODES.find((candidate) => candidate.id === id);
    if (node?.stage) {
      setRenderStageDisabled(node.stage, !disabledStages.has(node.stage));
      return;
    }
    if (id === "seam-closure") setSilhouetteRefinementEnabled(!silhouetteRefinementEnabled);
    else if (id === "cone-visibility") setSvoConeTracingMode(svoConeTracingMode === "off" ? "cones" : "off");
    else if (id === "gi-composition") setSvoGlobalIlluminationEnabled(!svoGlobalIlluminationEnabled);
    else if (id === "world-gi-cache") setSvoWorldGiCacheEnabled(!svoWorldGiCacheEnabled);
  };

  // Named drawers, shut by default.
  //
  // Every control a node owns still lives on that node — that is the whole
  // point of the graph — but a node with twenty cone budgets under it made the
  // instrument four screens tall over a scene it is supposed to be read
  // against. So a cluster that is *tuning* folds behind its own name, and the
  // controls that decide what the frame IS — the visibility source, the
  // shadow/AO switches — stay on the card where they were.
  const controls: Readonly<Record<string, ReactNode>> = {
    "sparse-world-build": renderSparseWorldBuildControls({ renderRefinementDepth, sceneIsDry, updateTuning, modified, resetTuning, leafVoxel_mm, finestCellSize_m, tuning }),

    "primary-traversal": renderPrimaryTraversalControls({ resolvedPrimary, partitioned, disabledStages, durations, effectiveRendererStatus, smoothSurfaceEnabled, svoMaximumTraversalDepth, setSvoMaximumTraversalDepth, svoMaximumNodeVisits, setSvoMaximumNodeVisits, tuning, updateTuning, modified, resetTuning }),

    "seam-closure": renderSeamClosureControls({ silhouetteRefinementStatus }),

    "cone-visibility": renderConeVisibilityControls({ svoConeTracingMode, setSvoConeTracingMode, svoShadowsEnabled, setSvoShadowsEnabled, svoAmbientOcclusionEnabled, setSvoAmbientOcclusionEnabled, tuning, updateTuning, modified, resetTuning, svoStageView, svoStageLightSlot, setSvoStageLightSlot, lightingVisibilityStatus }),

    "reduced-shade": renderReducedShadeControls(),

    "gi-composition": renderGiCompositionControls({ svoGlobalIlluminationEnabled, tuning, updateTuning, modified, resetTuning }),

    present: renderPresentControls({ tuning, updateTuning, modified, resetTuning }),
  };

  // Stage-span traces keep an additive phase sum in total_ms for trace
  // accounting, but the graph's denominator is the actual GPU frame span.
  const graphTotal_ms = stageTrace?.total_ms ?? 0;

  const bands: readonly PipelineBand[] = RENDER_PIPELINE_BANDS.map((band): PipelineBand => {
    const bandCost = measureRenderPipelineBand(band.id, durations, graphTotal_ms);
    const nodes = RENDER_PIPELINE_NODES.filter((node) => node.band === band.id)
      .map((node) => renderPipelineNodeForContext(node, context));
    const entries = nodes.map((node) => {
      const state = node.state(context);
      const cost: RenderPipelineMeasurement = partitioned
        ? measureRenderPipelineNode(node, durations, graphTotal_ms, state)
        : { kind: "unmeasured", share: 0 };
      return { node, state, cost };
    });
    const priced = partitioned && bandCost.duration_ms !== undefined;
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
      rows.push({
        id: node.id,
        label: node.label,
        state,
        tip,
        chip,
        cost: {
          kind: cost.kind,
          duration_ms: cost.duration_ms,
          encodedFraction: cost.encodedFraction,
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
        hint={`Samples the renderer while this panel is open. Measurement stops when this is off or the panel closes.\n\n${timingHint}`} />
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

    <div className="render-frame-options" role="group" aria-label="Frame surface options">
      <SvoFeatureSlot slot="frame.options" />
      <span>Surface</span>
      <PipeToggle label="Smooth surface" checked={smoothSurfaceEnabled}
        onChange={(enabled) => patchScene({ surfaceStyle: enabled ? "smooth" : "voxel-flat" })}
        hint="Reconstruct a sub-voxel tangent surface from each cell's coverage and baked normal, changing both surface depth and orientation. Off draws the entered axis-aligned voxel face." />
      {resolvedPrimary === "mesh" && <PipeToggle label="Filtered detail" checked={tuning.surfaceMeshLodPixels > 0}
        onChange={(enabled) => updateTuning("surfaceMeshLodPixels", enabled ? SVO_SURFACE_MESH_LOD_PIXELS_DEFAULT : 0)}
        hint="Draw each brick from the coarsest of its cached levels whose cells still project under the threshold, and shade baked voxel normals instead of six-axis faces. Off is the exact voxel boundary. Runtime only: no rebuild of the mesh or a pipeline. The threshold sits on the Primary rasterization row." />}
    </div>

    {svoPrimaryTraversal === "mesh" && smoothSurfaceEnabled && <p className="render-inline-status">
      Rasterized visibility uses voxel faces. Turn off Smooth surface to use the mesh; geometry is withheld while Smooth surface is enabled.
    </p>}

    {svoPrimaryTraversal === "mesh" && !smoothSurfaceEnabled && effectiveRendererStatus.surfaceMesh?.state === "blocked"
      && <p className="render-inline-status">{effectiveRendererStatus.surfaceMesh.detail}</p>}

    {effectiveRendererStatus.failureReason && <p className="render-inline-warning">SVO unavailable: {effectiveRendererStatus.detail
      ?? rendererFailureLabels[effectiveRendererStatus.failureReason]}.</p>}

    <PipelineGraph bands={bands} testId="render-pipeline" />
  </>;
}
