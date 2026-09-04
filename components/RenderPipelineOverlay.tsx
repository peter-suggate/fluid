"use client";

import { useEffect, useMemo, useState, type ReactNode } from "react";
import { useSession } from "../lib/core/session/session-context";
import { usePerformanceInstrumentationStore } from "../lib/core/stores/performance-instrumentation-store";
import {
  averagePerformanceTraces,
  type PerformanceTrace,
} from "../lib/core/performance-trace";
import {
  measureRenderPipelineBand,
  measureRenderPipelineNode,
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
  SVO_PRIMARY_LEAF_VISIT_HARD_LIMIT,
  SVO_RENDER_QUALITY_PRESETS,
  SVO_RENDER_TUNING_PRESETS,
  svoRenderTuningKey,
  type SvoConeRadianceReconstruction,
  type SvoRenderQualityPreset,
  type SvoRenderTuning,
} from "../lib/svo/svo-render-tuning";
import { sceneUsesFlatVoxelNormals } from "../lib/core/model";
import { PipeChoice, PipeRange, PipeToggle } from "./PipeControls";

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

/**
 * A millimetre readout that keeps the whole leaf ladder exact. The sizes are
 * binary fractions of the lattice, so a fixed number of decimals either rounds
 * the fine end into a lie or pads the coarse end with zeros.
 */
const trimmed = (value: number) => value.toFixed(5).replace(/\.?0+$/, "");

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
  const svoConeTracingMode = session.ui((state) => state.svoConeTracingMode);
  const setSvoConeTracingMode = session.ui((state) => state.setSvoConeTracingMode);
  const svoGlobalIlluminationEnabled = session.ui((state) => state.svoGlobalIlluminationEnabled);
  const setSvoGlobalIlluminationEnabled = session.ui((state) => state.setSvoGlobalIlluminationEnabled);
  const disabledRenderStages = session.ui((state) => state.disabledRenderStages);
  const setRenderStageDisabled = session.ui((state) => state.setRenderStageDisabled);
  const svoStageView = session.ui((state) => state.svoStageView);
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
        label="Environment refinement depth · REBUILD"
        unit="levels" value={renderRefinementDepth}
        min={0} max={SVO_ENVIRONMENT_REFINEMENT_DEPTH_MAXIMUM}
        step={1} digits={0} disabled={!sceneIsDry}
        onChange={(value) => updateTuning("environmentRefinementDepth", value)}
        modified={modified("environmentRefinementDepth")} onReset={resetTuning("environmentRefinementDepth")}
        hint={sceneIsDry
          ? `Render leaf ${trimmed(leafVoxel_mm)} mm · ${renderRefinementDepth} level${renderRefinementDepth === 1 ? "" : "s"} below the ${trimmed(finestCellSize_m * 1000)} mm simulation lattice.\n\nThis rebuilds only the renderer-owned SVO derivative. The scene lattice, SolidWorld, and simulation state do not change. Scenery and terrain are sampled at the render leaf.\n\nEach level also adds one to the cone hierarchy; past the runtime's twelve, derived lighting withdraws and the cone node reads EXACT FALLBACK.`
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

  // Stage-span traces keep an additive phase sum in total_ms for trace
  // accounting, but the graph's denominator is the actual GPU frame span.
  const graphTotal_ms = stageTrace?.total_ms ?? 0;

  const bands: readonly PipelineBand[] = RENDER_PIPELINE_BANDS.map((band): PipelineBand => {
    const bandCost = measureRenderPipelineBand(band.id, durations, graphTotal_ms);
    const nodes = RENDER_PIPELINE_NODES.filter((node) => node.band === band.id);
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
      <span>Surface</span>
      <PipeToggle label="Smooth surface" checked={smoothSurfaceEnabled}
        onChange={(enabled) => patchScene({ surfaceStyle: enabled ? "smooth" : "voxel-flat" })}
        hint="Reconstruct a sub-voxel tangent surface from each cell's coverage and baked normal, changing both surface depth and orientation. Off draws the entered axis-aligned voxel face." />
    </div>

    {effectiveRendererStatus.failureReason && <p className="render-inline-warning">SVO unavailable: {effectiveRendererStatus.detail
      ?? rendererFailureLabels[effectiveRendererStatus.failureReason]}.</p>}

    <PipelineGraph bands={bands} testId="render-pipeline" />
  </>;
}
