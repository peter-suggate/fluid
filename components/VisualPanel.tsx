"use client";

import type { ReactNode } from "react";
import { useDiagnosticsStore } from "@/lib/stores/diagnostics-store";
import { useSceneStore } from "@/lib/stores/scene-store";
import { useUIStore } from "@/lib/stores/ui-store";
import {
  SVO_RENDER_STAGE_DEFINITIONS,
  SVO_RENDER_STAGE_GROUPS,
  SVO_RENDER_STAGE_LABELS,
  SVO_RENDER_STAGE_MAXIMUM_LIGHT_SLOT,
  SVO_RENDER_STAGE_VIEWS,
  svoRenderStageUsesLightSlot,
  type SvoRenderStageView,
} from "@/lib/svo-render-diagnostics";
import {
  SVO_ENVIRONMENT_BRICK_REFINEMENT_MAXIMUM,
  SVO_ENVIRONMENT_REFINEMENT_DEPTH_MAXIMUM,
  SVO_LOD_FIXED_LEVEL_MAXIMUM,
  SVO_LOD_SCREEN_SPACE_PIXELS_MAXIMUM,
  SVO_PRIMARY_LEAF_VISIT_HARD_LIMIT,
  SVO_RENDER_QUALITY_PRESETS,
  SVO_RENDER_TUNING_PRESETS,
  svoRenderTuningKey,
  svoSceneryDetailCellSize_m,
  svoSceneryRefinementDepth,
  type SvoRenderQualityPreset,
  type SvoRenderTuning,
} from "@/lib/svo-render-tuning";
import { findSceneDefinition } from "@/lib/scenes";
import { sceneDefinitionTakesLattice } from "@/lib/scene-definition";
import { simulation } from "@/lib/simulation/controller";
import { RangeControl } from "./controls";

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
 * Categorical palettes read as bands, not as a blend: a stage view names a
 * class per pixel, and a gradient between two classes would imply a value
 * between them that the plane never carries.
 */
const stageGradient = (view: SvoRenderStageView) => {
  if (view === "off") return "linear-gradient(90deg,#071411,#65b7a4,#f2bc72)";
  const { legend, palette } = SVO_RENDER_STAGE_DEFINITIONS[view];
  if (palette !== "categorical" && palette !== "identity") {
    return `linear-gradient(90deg,${legend.map((stop) => `${stop.color} ${Math.round(stop.at * 100)}%`).join(",")})`;
  }
  const band = 100 / legend.length;
  return `linear-gradient(90deg,${legend
    .map((stop, index) => `${stop.color} ${(index * band).toFixed(1)}%,${stop.color} ${((index + 1) * band).toFixed(1)}%`)
    .join(",")})`;
};

/**
 * A millimetre readout that keeps the whole leaf ladder exact.
 *
 * The sizes here are binary fractions of the lattice — 6.25, 3.125, 1.5625,
 * 0.78125 — so a fixed number of decimals either rounds the fine end into a lie
 * or pads the coarse end with zeros. Five decimals covers every rung the depth
 * ceiling allows, and the trim removes what that costs everywhere else.
 */
const trimmed = (value: number) => value.toFixed(5).replace(/\.?0+$/, "");

function Toggle({ label, checked, onChange, disabled = false, hint }: {
  label: string; checked: boolean; onChange: (checked: boolean) => void; disabled?: boolean; hint?: string;
}) {
  return <button className="render-toggle" type="button" role="switch" aria-checked={checked}
    disabled={disabled} title={hint} onClick={() => onChange(!checked)}>
    <i aria-hidden="true" /><span>{label}</span>
  </button>;
}

function ControlGroup({ title, note, children, open = false }: {
  title: string; note: string; children: ReactNode; open?: boolean;
}) {
  return <details className="render-control-group" open={open}>
    <summary><span>{title}</span><small>{note}</small><i aria-hidden="true" /></summary>
    <div className="render-control-group-body">{children}</div>
  </details>;
}

/**
 * A named block inside a group.
 *
 * The panel used to answer "where does this control live?" by adding another
 * peer disclosure, which is how nine of them accumulated. A subheading names a
 * neighbourhood without costing a section, so the two collapsed groups can hold
 * a long tail without the open ones growing.
 */
function SubHeading({ title, note }: { title: string; note: string }) {
  return <h3 className="render-subhead"><span>{title}</span><small>{note}</small></h3>;
}

export function VisualPanel() {
  const effectiveRendererStatus = useDiagnosticsStore((state) => state.effectiveRendererStatus);
  const svoShadowsEnabled = useUIStore((state) => state.svoShadowsEnabled);
  const setSvoShadowsEnabled = useUIStore((state) => state.setSvoShadowsEnabled);
  const svoAmbientOcclusionEnabled = useUIStore((state) => state.svoAmbientOcclusionEnabled);
  const setSvoAmbientOcclusionEnabled = useUIStore((state) => state.setSvoAmbientOcclusionEnabled);
  const silhouetteRefinementEnabled = useUIStore((state) => state.silhouetteRefinementEnabled);
  const setSilhouetteRefinementEnabled = useUIStore((state) => state.setSilhouetteRefinementEnabled);
  const svoConeTracingMode = useUIStore((state) => state.svoConeTracingMode);
  const setSvoConeTracingMode = useUIStore((state) => state.setSvoConeTracingMode);
  const svoPrimaryTraversal = useUIStore((state) => state.svoPrimaryTraversal);
  const setSvoPrimaryTraversal = useUIStore((state) => state.setSvoPrimaryTraversal);
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
  const setRightPanel = useUIStore((state) => state.setRightPanel);
  const pixelTraceEnabled = useUIStore((state) => state.pixelTraceEnabled);
  const setPixelTraceEnabled = useUIStore((state) => state.setPixelTraceEnabled);
  const pixelTracePinned = useUIStore((state) => state.pixelTracePinned);
  const setPixelTracePinned = useUIStore((state) => state.setPixelTracePinned);
  const requestPixelTracePin = useUIStore((state) => state.requestPixelTracePin);
  const fluidCellTraceEnabled = useUIStore((state) => state.fluidCellTraceEnabled);
  const fluidCellTracePinned = useUIStore((state) => state.fluidCellTracePinned);
  const setFluidCellTraceEnabled = useUIStore((state) => state.setFluidCellTraceEnabled);
  const setFluidCellTracePinned = useUIStore((state) => state.setFluidCellTracePinned);
  const requestFluidCellTracePin = useUIStore((state) => state.requestFluidCellTracePin);
  // The two scene facts the refinement-depth control has to state for itself:
  // what a leaf actually measures at the chosen depth, and whether the depth is
  // legal at all. A simulated scene pins every brick's node at the solver level,
  // so the same slider is a no-op there and has to say so rather than move.
  const finestCellSize_m = useSceneStore((state) => state.scene.voxelDomain.finestCellSize_m);
  const sceneIsDry = useSceneStore((state) => state.scene.systems?.fluid === false);
  // The depth, read where it lives. There is no tuning field to disagree with.
  const voxelDomain = useSceneStore((state) => state.scene.voxelDomain);
  const presetId = useSceneStore((state) => state.presetId);
  const authoredRefinementDepth = svoSceneryRefinementDepth(voxelDomain, { fluid: !sceneIsDry });
  // Water is the only thing that can refuse a depth outright — a solver brick
  // pins its node. Whether the scene's *factory* takes a lattice decides only
  // how far the change propagates: with one, the document is re-authored and
  // even the terrain pitch follows; without, it is a patch, and every generator
  // still re-resolves its floors because they read the leaf at expansion time.
  const latticeDefinition = findSceneDefinition(presetId);
  const reauthorsDocument = latticeDefinition !== undefined && sceneDefinitionTakesLattice(latticeDefinition);

  const selectedView = SVO_RENDER_STAGE_DEFINITIONS[svoStageView];
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
  const lightingVisibilityStatus = effectiveRendererStatus.lightingVisibility ?? {
    state: svoConeTracingMode,
  };
  const updateTuning = <K extends keyof SvoRenderTuning>(key: K, value: SvoRenderTuning[K]) =>
    setTuning((current) => ({ ...current, [key]: value }));
  const modified = <K extends keyof SvoRenderTuning>(key: K) => tuning[key] !== SVO_RENDER_TUNING_PRESETS.balanced[key];
  const resetTuning = <K extends keyof SvoRenderTuning>(key: K) => () => updateTuning(key, SVO_RENDER_TUNING_PRESETS.balanced[key]);
  // The same rule the tree resolves at construction, so this readout cannot
  // drift from the leaf that actually gets built.
  const leafVoxelSize_mm = svoSceneryDetailCellSize_m(finestCellSize_m, {
    environmentRefinementDepth: authoredRefinementDepth,
    fluid: !sceneIsDry,
  }) * 1000;

  const selectStageView = (view: SvoRenderStageView) => {
    const next = view === svoStageView && view !== "off" ? "off" : view;
    setSvoStageView(next);
  };

  return <aside id="render-panel" className="right-panel panel-scroll performance-panel performance-v2 visual-panel"
    aria-label="Rendering diagnostics and scene visualizations" data-testid="visual-panel">
    <header className="performance-panel-header render-panel-header">
      <div><span>RENDER OBSERVATORY</span><h2>SVO performance tuning</h2></div>
      <div className="performance-panel-header-actions">
        <button className="panel-close" type="button" onClick={() => setRightPanel(null)} aria-label="Close render panel">×</button>
      </div>
      <div className="render-status-line">
        <span className={effectiveRendererStatus.state === "active" ? "online" : ""} />
        <strong data-testid="effective-renderer-status">{effectiveRendererStatus.state === "active"
          ? lightingVisibilityStatus.state === "cones" ? "SVO GI ACTIVE"
            : lightingVisibilityStatus.state === "exact" ? "SVO EXACT ACTIVE" : "SVO DIRECT ACTIVE"
          : effectiveRendererStatus.state === "not-required" ? "SVO NOT REQUIRED"
          : effectiveRendererStatus.state === "pending" ? "SVO PENDING" : "SVO FAILED CLOSED"}</strong>
        <code>{Math.round(tuning.resolutionScale * 100)}% · cone {svoConeTracingMode !== "cones" ? svoConeTracingMode : tuning.coneLightingScale === 1 ? "full" : `${1 / tuning.coneLightingScale}×${1 / tuning.coneLightingScale}`}</code>
      </div>
    </header>

    {/* Profile and representation are the two questions asked before any slider
        — "how expensive" and "what am I looking at" — so they stay above the
        disclosures rather than inside one that can be closed over them. */}
    <div className="render-preset-strip" role="group" aria-label="Render performance profile">
      <span>PROFILE</span>
      {(Object.keys(SVO_RENDER_QUALITY_PRESETS) as SvoRenderQualityPreset[]).map((preset) =>
        <button key={preset} className={activePreset === preset ? "active" : ""} onClick={() => {
          // One click, both halves. The visibility mode has its own buttons under
          // Lighting and stays reachable there; what this strip guarantees is that
          // a named rung is never half-applied.
          setTuning(SVO_RENDER_QUALITY_PRESETS[preset].tuning);
          setSvoConeTracingMode(SVO_RENDER_QUALITY_PRESETS[preset].coneTracingMode);
        }}>{preset}</button>)}
      {!activePreset && <output>CUSTOM</output>}
    </div>

    <div className="render-groups">
      {/* Four controls carry the frame's cost: pixels, where descent stops, which
          primary path runs, and whether it runs at all. Measured on M1 Max at
          1500²: raster vs traced 29.0/49.6 ms, stationary reuse 47.6 → 20.7 ms,
          and the screen-space threshold 24.1 → 5.0 ms on the 4 Mpx far arm. Every
          other geometry knob in this panel is a cap that only shows up when it is
          too low, so it lives under Advanced. */}
      <ControlGroup title="Detail" note="leaf size · resolution · LOD · primary path" open>
        <div className="svo-control-grid">
          {/* First, because it is the only control here that changes what the
              scene *is* rather than how much of it gets drawn — everything below
              spends or saves against the leaf this sets. It was under Advanced
              while it was a render tuning; it is not one. The depth lives on the
              document, because the document is what the scenery generators are
              expanded against, and a slider that moved only the tree gave a
              finer leaf over a set still authored at the coarse one. One number,
              one setter: `simulation.setEnvironmentRefinementDepth`, which the
              Scene config popover's lattice control also goes through. */}
          <RangeControl className="control-lead"
            label={`Environment refinement depth · ${reauthorsDocument ? "RE-AUTHORS" : "REBUILD"}`}
            unit="levels" value={authoredRefinementDepth}
            min={0} max={SVO_ENVIRONMENT_REFINEMENT_DEPTH_MAXIMUM} step={1} displayDigits={0}
            disabled={!sceneIsDry}
            onChange={(value) => { simulation.setEnvironmentRefinementDepth(value); }}
            hint="Extra octree levels the authored environment may descend into, below the scene's own lattice. It writes the document's own detail cell, which is the single number the tree and the scenery both read — a tuning that moved only the tree gave a finer leaf over a set still expanded at the coarse size. Every generator re-resolves its legibility floors at the new leaf; where the scene's factory takes a lattice the whole document is re-authored, so even the terrain pitch follows. Only legal on a scene the solver does not own: a solver brick pins its node, leaving nowhere to descend." />
          {/* The three facts the slider cannot state: what a leaf now measures,
              that the cone hierarchy grows a level with it, and that the build
              may refuse the depth and step down. The derived pyramid withdraws
              above the runtime's twelve-level ceiling, which costs roughly 15x
              with no allocation failure to notice — the Lighting visibility line
              under Lighting is where that shows. */}
          <p className="control-caption" data-testid="environment-refinement-leaf-size">
            {!sceneIsDry
              ? `Zero on this scene whatever the slider says: the fluid solver claims every brick of the container and a solver brick pins its node, so the leaf stays the ${trimmed(finestCellSize_m * 1000)} mm lattice. Turn water off under Scene configuration to spend depth here.`
              : `Leaf voxel ${trimmed(leafVoxelSize_mm)} mm, and the set is drawn at that size — a ${trimmed(finestCellSize_m * 1000)} mm lattice over ${authoredRefinementDepth} extra level${authoredRefinementDepth === 1 ? "" : "s"}. `
                + (reauthorsDocument
                  ? "Moving it re-authors the document through this scene's own factory, so the terrain pitch follows too; that reloads the preset, and scenery edits since it was opened do not survive it (it goes on the undo stack for that reason). "
                  : "This scene's factory takes no lattice, so moving it patches the document instead: every generator re-resolves its legibility floors at the new leaf, but anything the factory baked — the terrain pitch — keeps its authored value. ")
                + "Each level also adds one level to the cone hierarchy; past the runtime's twelve, derived lighting withdraws and the Lighting visibility line below reads EXACT FALLBACK. A depth the device cannot allocate degrades to the finest that fits, and the renderer status line says so."}
          </p>
          <label className="render-discrete-control control-span"><span>Level of detail</span><div>
            {([
              { mode: "screen-space", label: "SCREEN", hint: "Stop descending once a node's projected footprint falls under the threshold. Distance-adaptive: far geometry costs less without anything authored per scene." },
              { mode: "fixed-level", label: "FIXED", hint: "Stop at one level for every ray regardless of projection. A debugging view: it shows exactly what a given depth can represent, which is what tells a missing-detail bug apart from a level the tree never built." },
            ] as const)
              .map(({ mode, label, hint }) => <button key={mode} className={tuning.lodMode === mode ? "active" : ""}
                title={hint} onClick={() => updateTuning("lodMode", mode)}>{label}</button>)}
          </div></label>
          {/* Only the predicate in force is shown, and it leads the panel. The
              alternative would be a slider that silently governs nothing, and
              this panel is where a wrong image gets diagnosed — a dead control
              here costs an hour. */}
          {tuning.lodMode === "screen-space"
            ? <RangeControl className="control-lead" label="LOD threshold" unit="px" value={tuning.lodScreenSpacePixels}
                min={0} max={SVO_LOD_SCREEN_SPACE_PIXELS_MAXIMUM} step={0.5} displayDigits={1}
                onChange={(value) => updateTuning("lodScreenSpacePixels", value)}
                modified={modified("lodScreenSpacePixels")} onReset={resetTuning("lodScreenSpacePixels")}
                hint="The projected diameter of a node's enclosing sphere, authored at a 460px reference height and rescaled to the live target — so the threshold is angular and raising resolution cannot silently request more detail. At the hero's 50 mm lens over 6.25 mm cells, threshold 3 is exact below 6.9 m, 2-cell aggregates to 13.9 m, and 4-cell beyond." />
            : <RangeControl className="control-lead" label="Detail level" unit="levels" value={tuning.lodFixedLevel}
                min={0} max={SVO_LOD_FIXED_LEVEL_MAXIMUM} step={1} displayDigits={0}
                onChange={(value) => updateTuning("lodFixedLevel", value)}
                modified={modified("lodFixedLevel")} onReset={resetTuning("lodFixedLevel")}
                hint="Descent stops at this level for every ray regardless of projection, and shades a proxy. Distinct from the maximum traversal depth under Advanced, which clamps and reports exhaustion instead of drawing anything." />}
          {/* Both captions state a property the slider itself cannot: the
              guarantee the threshold makes, and — for the fixed level — that its
              coarse half is a plateau. A user should not have to find a plateau
              by experiment, and a tooltip is not where they would look. */}
          <p className="control-caption">{tuning.lodMode === "screen-space"
            ? "A fidelity budget: no aggregate is ever drawn larger than this on screen, and the error it trades for only ever adds coverage. A brick that measures under the threshold is drawn as one solid voxel, so a silhouette can grow by at most this many pixels and can never open a hole through to the background. Zero is exact traversal everywhere, verified byte-exact against the reference image."
            : "0 coarsest → 21 exact voxels. An 8-cell brick adds three levels below its leaf, so one level past the scene's brick depth draws 4-cell aggregates, two draws 2-cell, and three is one cell and exact — 21 is honestly exact, just far past the point where anything more changes. Every level at or below the brick depth draws the same image, because the raster primary instances one proxy per brick; that plateau ends when coarse-node instances arrive."}</p>
          <RangeControl label="Render resolution" unit="%" value={tuning.resolutionScale * 100} min={35} max={100} step={1} displayDigits={0}
            onChange={(value) => updateTuning("resolutionScale", value / 100)} modified={modified("resolutionScale")} onReset={resetTuning("resolutionScale")}
            hint="Pixel-linear: the frame costs roughly 19–22 ms per megapixel with cones and 13–15 without, so this is the one dial that moves every pass at once." />
          <label className="render-discrete-control"><span>Primary visibility</span><div>
            {([
              { mode: "raster", label: "RASTER", hint: "Hardware-rasterize the resident bricks as depth-tested proxies. Octree leaves partition space, so the depth test is exact: the image matches TRACED pixel for pixel. 29.0 ms against 49.6 ms at 1500x1500 garden." },
              { mode: "traced", label: "TRACED", hint: "The full-screen traversal megakernel, where every pixel marches the octree for itself. The reference the raster path is measured against." },
            ] as const)
              .map(({ mode, label, hint }) => <button key={mode} className={svoPrimaryTraversal === mode ? "active" : ""}
                title={hint} onClick={() => setSvoPrimaryTraversal(mode)}>{label}</button>)}
          </div></label>
        </div>
        <div className="render-toggle-row" role="group" aria-label="SVO primary tracing optimizations">
          <Toggle label="Reuse stationary visibility" checked={tuning.stationaryPrimaryReuseEnabled}
            disabled={svoPrimaryTraversal === "raster"}
            hint="Reuse the exact primary G-buffer while its camera and geometry dependencies are unchanged. Measured 47.6 → 20.7 ms at 1500². The raster primary is cheap enough not to cache, and its impostor pass blocks the reuse anyway."
            onChange={(value) => updateTuning("stationaryPrimaryReuseEnabled", value)} />
          <Toggle label="Close primary seams" checked={silhouetteRefinementEnabled}
            hint="Close one-pixel background seams only when opposing foreground surfaces bracket the pixel. This opt-in pass runs before sky and deferred lighting."
            onChange={setSilhouetteRefinementEnabled} />
        </div>
        <p data-testid="silhouette-refinement-status" aria-live="polite"
          className={silhouetteRefinementStatus.state === "failed" ? "render-inline-warning" : "render-inline-status"}>
          Primary seam closure: {silhouetteRefinementStatus.state.toUpperCase()}
          {silhouetteRefinementStatus.detail ? ` · ${silhouetteRefinementStatus.detail}` : ""}
        </p>
        {effectiveRendererStatus.failureReason && <p className="render-inline-warning">SVO unavailable: {effectiveRendererStatus.detail
          ?? rendererFailureLabels[effectiveRendererStatus.failureReason]}.</p>}
      </ControlGroup>

      {/* Cone tracing is ~16 ms of a 48 ms garden frame, and the visibility mode
          plus the prepass rate are what spend it. The GI trio below is the frame's
          energy balance — the only lighting values a preset ever moves for looks
          rather than for cost. Apertures, biases and sample counts calibrate those
          same passes once per scene, so they sit under Advanced. */}
      <ControlGroup title="Lighting" note="visibility · cone rate · bounce" open>
        <div className="render-toggle-row" role="group" aria-label="SVO lighting effects">
          <Toggle label="Shadows" checked={svoShadowsEnabled} disabled={svoConeTracingMode === "off"} onChange={setSvoShadowsEnabled} />
          <Toggle label="AO" checked={svoAmbientOcclusionEnabled} disabled={svoConeTracingMode === "off"} onChange={setSvoAmbientOcclusionEnabled} />
        </div>
        <div className="svo-control-grid">
          <label className="render-discrete-control"><span>Lighting visibility</span><div>
            {([
              { mode: "cones", label: "CONES", hint: "Cone-traced soft shadows, AO, and GI, fed by the reduced-rate prepass and world-GI cache." },
              { mode: "exact", label: "EXACT", hint: "No cone stage runs; shadows and AO use bounded exact SVO visibility rays. Sharp reference shadows, costlier per pixel." },
              { mode: "off", label: "OFF", hint: "No visibility work at all: unshadowed direct lighting, no AO, no GI. Strictly removes work." },
            ] as const)
              .map(({ mode, label, hint }) => <button key={mode} className={svoConeTracingMode === mode ? "active" : ""}
                title={hint} onClick={() => setSvoConeTracingMode(mode)}>{label}</button>)}
          </div></label>
          <label className="render-discrete-control"><span>Cone prepass</span><div>
            {([{ scale: 1, label: "FULL" }, { scale: 0.5, label: "2×2" }, { scale: 0.25, label: "4×4" }, { scale: 0.125, label: "8×8" }] as const)
              .map(({ scale, label }) => <button key={scale} className={tuning.coneLightingScale === scale ? "active" : ""}
                disabled={svoConeTracingMode !== "cones"} onClick={() => updateTuning("coneLightingScale", scale)}>{label}</button>)}
          </div></label>
          <RangeControl label="GI bounce" unit="%" value={tuning.giBounceStrength * 100} min={0} max={400} step={5} displayDigits={0}
            onChange={(value) => updateTuning("giBounceStrength", value / 100)} modified={modified("giBounceStrength")} onReset={resetTuning("giBounceStrength")}
            hint="Exposure for gathered diffuse bounce. This does not amplify direct highlights or emissive surfaces." />
          <RangeControl label="GI occlusion" unit="%" value={tuning.giOcclusionStrength * 100} min={0} max={100} step={1} displayDigits={0}
            onChange={(value) => updateTuning("giOcclusionStrength", value / 100)} modified={modified("giOcclusionStrength")} onReset={resetTuning("giOcclusionStrength")}
            hint="Uses the same wide GI cones to darken diffuse environment fill in enclosed regions. The AO toggle enables this in GLOBAL mode." />
          <RangeControl label="Diffuse environment" unit="%" value={tuning.giEnvironmentStrength * 100} min={0} max={200} step={5} displayDigits={0}
            onChange={(value) => updateTuning("giEnvironmentStrength", value / 100)} modified={modified("giEnvironmentStrength")} onReset={resetTuning("giEnvironmentStrength")}
            hint="Analytic sky fill retained alongside GI. Lower this when bounced light should carry more of the diffuse scene." />
        </div>
        <p data-testid="lighting-visibility-status" aria-live="polite"
          className={lightingVisibilityStatus.fallback ? "render-inline-warning" : "render-inline-status"}>
          Lighting visibility: {lightingVisibilityStatus.state.toUpperCase()}
          {lightingVisibilityStatus.fallback ? " FALLBACK" : ""}
          {lightingVisibilityStatus.detail ? ` · ${lightingVisibilityStatus.detail}` : ""}
        </p>
      </ControlGroup>

      {/* One disclosure for everything that explains a frame rather than tunes
          it. The three used to be peer sections, which put a per-pixel probe at
          the same rank as render resolution. */}
      <ControlGroup title="Diagnostics" note="per-pass planes · pointer probes">
        <SubHeading title="Pointer probes" note="ONE PIXEL, ONE PRESSURE CELL" />
        <div className="render-toggle-row" role="group" aria-label="Live ray-work diagnostic">
          <Toggle label="Trace hovered pixel" checked={pixelTraceEnabled} onChange={setPixelTraceEnabled} />
          <Toggle
            label="Pin ray"
            checked={pixelTracePinned}
            disabled={!pixelTraceEnabled}
            // The viewport owns which pixel and which view a pin records, so this
            // asks for one rather than declaring one from out here.
            onChange={(on) => (on ? requestPixelTracePin() : setPixelTracePinned(false))}
          />
        </div>
        <div className="render-toggle-row" role="group" aria-label="Per-cell fluid work diagnostic">
          <Toggle label="Trace hovered cell" checked={fluidCellTraceEnabled} onChange={setFluidCellTraceEnabled} />
          <Toggle
            label="Pin cell"
            checked={fluidCellTracePinned}
            disabled={!fluidCellTraceEnabled}
            // As with the ray probe, the viewport owns which pixel a pin records.
            onChange={(on) => (on ? requestFluidCellTracePin() : setFluidCellTracePinned(false))}
          />
        </div>
        <p className="panel-note">
          The ray probe draws what the frame actually did to produce one pixel — under the raster primary a
          tournament of competing brick proxies, under the traced one the octree walk itself, plus the shadow rays and
          cones either way. The cell probe names the pressure cell behind that pixel and reads its row, stencil and
          power neighbours from what the frame published. Click the viewport to freeze a pixel and orbit its work;
          click again to follow the pointer.
        </p>
        <SubHeading title="Render pipeline stages" note="WHAT EACH PASS ACTUALLY WROTE" />
        <div className="paper-view-grid" role="group" aria-label="Render stage view">
          {SVO_RENDER_STAGE_GROUPS.flatMap((group) =>
            SVO_RENDER_STAGE_VIEWS.filter((candidate) => SVO_RENDER_STAGE_DEFINITIONS[candidate].group === group))
            .map((candidate) => {
              const definition = SVO_RENDER_STAGE_DEFINITIONS[candidate];
              const active = svoStageView === candidate;
              return <button key={candidate} type="button" className={active ? "active" : ""} aria-pressed={active} onClick={() => selectStageView(candidate)}>
                <span>{definition.group}</span><strong>{definition.label}</strong><small>{definition.description}</small>
                <i className="render-view-palette" style={{ background: stageGradient(candidate) }} aria-hidden="true" />
              </button>;
            })}
        </div>
        {svoRenderStageUsesLightSlot(svoStageView) && <div className="svo-control-grid">
          <RangeControl label="Cached light slot" unit="slot" value={svoStageLightSlot} min={0} max={SVO_RENDER_STAGE_MAXIMUM_LIGHT_SLOT} step={1} displayDigits={0}
            onChange={setSvoStageLightSlot}
            hint="Which of the eight cached per-light visibilities the prepass plane is decoded for." />
        </div>}
        <div className="paper-view-inspector" aria-live="polite">
          <header>
            <div><span>ACTIVE STAGE VIEW</span><strong>{SVO_RENDER_STAGE_LABELS[svoStageView]}</strong></div>
            <code>{svoStageView === "off" ? "composited frame" : "published plane"}</code>
          </header>
          <p>{selectedView.description}</p>
          <small>SOURCE · {svoStageView === "off" ? "OPTICAL COMPOSITE" : `${selectedView.group.toUpperCase()} · ${selectedView.plane}`}</small>
          <div className="paper-field-legend" aria-label={`${SVO_RENDER_STAGE_LABELS[svoStageView]} legend`}>
            {selectedView.legend.map((entry, index) => <span key={`${entry.label}-${index}`}><i style={{ background: entry.color }} />{entry.label}</span>)}
          </div>
          <footer>{svoStageView === "off"
            ? "Choose a view to replace the presented image with a decode of the plane that stage wrote. Selecting one switches to the finished SVO path automatically."
            : "Read-only: the overlay is drawn after the optical composite from planes the frame already published, so the frame it explains is the frame that rendered."}</footer>
        </div>
        <p className="render-visualization-status">STAGE PLANES READY</p>
      </ControlGroup>

      {/* Demoted, not deleted. Nothing here is reachable any other way — these
          fields have no environment override and no benchmark flag, and a
          renderer under active development still needs them weekly. What they
          are not is a reason to look at this panel: no preset moves the
          calibration values at all, and the budgets the presets do move
          (`performance` vs `quality`) are moved together by the PROFILE strip
          above, which is the whole point of having presets. */}
      <ControlGroup title="Advanced" note="calibrated once · moved rarely">
        <SubHeading title="Scene build" note="REBUILDS THE SPARSE WORLD" />
        <div className="svo-control-grid">
          <RangeControl label="Environment brick refinement" unit="levels" value={tuning.environmentBrickRefinementLevels}
            min={0} max={SVO_ENVIRONMENT_BRICK_REFINEMENT_MAXIMUM} step={1} displayDigits={0}
            onChange={(value) => updateTuning("environmentBrickRefinementLevels", value)}
            modified={modified("environmentBrickRefinementLevels")} onReset={resetTuning("environmentBrickRefinementLevels")}
            hint="Additional SVO subdivision for authored scenery outside the simulation lattice. Already at its ceiling by default, so the only move is down; changing it rebuilds the sparse world." />
          {/* Sits under the depth slider because it decides how much of that
              depth is actually spent. Off is the default: the levels it saves
              are the low-curvature ones, and that is where a coarse leaf is
              most visible. */}
          <div className="render-toggle-row control-span" role="group" aria-label="Environment refinement exemptions">
            <Toggle label="Flat-node exemption · REBUILD" checked={tuning.environmentPlanarRefinementExemption}
              hint="Let the refinement rule stop at a node its surface crosses flatly instead of spending the depth above. The test is second order — the residual against the node centre's tangent plane, and for the ground the spread of its column heights — so it declines depth exactly where curvature is lowest. The primary shades a leaf as one of six axis-aligned voxel faces, so an exempted node arrives as a flat facet with a straight octree boundary against the refined slope beside it, and a broad shallow mound reads as a grid of them. On, the tree is smaller and builds faster, at that cost. Changing it rebuilds the sparse world."
              onChange={(value) => updateTuning("environmentPlanarRefinementExemption", value)} />
          </div>
          <label className="render-discrete-control reconstruction-control"><span>Lighting reconstruction</span><div>
            {([{ mode: "nearest", label: "EXACT" }, { mode: "gated-linear", label: "LINEAR" }, { mode: "joint-bilateral", label: "BILAT" },
              { mode: "wide-relight", label: "WIDE" }, { mode: "full-res-relight", label: "RELIGHT" }] as const)
              .map(({ mode, label }) => <button key={mode} className={tuning.coneRadianceReconstruction === mode ? "active" : ""}
                disabled={tuning.coneLightingScale === 1 || svoConeTracingMode !== "cones"} onClick={() => updateTuning("coneRadianceReconstruction", mode)}>{label}</button>)}
          </div></label>
        </div>

        <SubHeading title="Lighting calibration" note="APERTURES · STRENGTHS · BIAS" />
        <div className="svo-control-grid">
          <RangeControl label="Shadow cone aperture" unit="rad" value={tuning.shadowConeAperture} min={0.01} max={0.25} step={0.005} displayDigits={3}
            onChange={(value) => updateTuning("shadowConeAperture", value)}
            modified={modified("shadowConeAperture")} onReset={resetTuning("shadowConeAperture")}
            hint="Wider cones take larger march steps and produce softer shadows; narrower cones preserve sharper shadows but require more taps. The sharp/soft decision most scenes want is the CONES/EXACT switch above; this calibrates the soft path." />
          <RangeControl label="Shadow strength" unit="%" value={tuning.shadowStrength * 100} min={0} max={100} step={1} displayDigits={0} onChange={(value) => updateTuning("shadowStrength", value / 100)} modified={modified("shadowStrength")} onReset={resetTuning("shadowStrength")} />
          <RangeControl label="Shadow origin bias" unit="cells" value={tuning.shadowBiasCells} min={0} max={0.25} step={0.005} displayDigits={3} onChange={(value) => updateTuning("shadowBiasCells", value)} modified={modified("shadowBiasCells")} onReset={resetTuning("shadowBiasCells")} />
          <RangeControl label="AO strength" unit="%" value={tuning.aoStrength * 100} min={0} max={100} step={1} displayDigits={0} onChange={(value) => updateTuning("aoStrength", value / 100)} modified={modified("aoStrength")} onReset={resetTuning("aoStrength")} />
          <RangeControl label="AO radius" unit="×" value={tuning.aoRadiusScale} min={0.1} max={3} step={0.05} displayDigits={2} onChange={(value) => updateTuning("aoRadiusScale", value)} modified={modified("aoRadiusScale")} onReset={resetTuning("aoRadiusScale")} />
          <RangeControl label="AO cone aperture" unit="rad" value={tuning.aoConeAperture} min={0.1} max={1.4} step={0.01} displayDigits={2} onChange={(value) => updateTuning("aoConeAperture", value)} modified={modified("aoConeAperture")} onReset={resetTuning("aoConeAperture")} />
          <RangeControl label="Direct key" unit="%" value={tuning.giDirectStrength * 100} min={0} max={200} step={5} displayDigits={0}
            onChange={(value) => updateTuning("giDirectStrength", value / 100)} modified={modified("giDirectStrength")} onReset={resetTuning("giDirectStrength")}
            hint="Exact primary light retained in GLOBAL mode. Its correct value is 100%: derived radiance carries emitted energy, not a baked replacement for the authored lights, so anything else is an exposure error rather than a quality setting." />
          <RangeControl label="GI cone aperture" unit="rad" value={tuning.giConeAperture} min={0.4} max={1.4} step={0.01} displayDigits={2}
            onChange={(value) => updateTuning("giConeAperture", value)} modified={modified("giConeAperture")} onReset={resetTuning("giConeAperture")}
            hint="Wide apertures survey broad scene regions and produce smoother, lower-frequency bounce and occlusion." />
          <RangeControl label="Normal escape" unit="cells" value={tuning.coneNormalEscapeCells} min={0} max={2} step={0.05} displayDigits={2} onChange={(value) => updateTuning("coneNormalEscapeCells", value)} modified={modified("coneNormalEscapeCells")} onReset={resetTuning("coneNormalEscapeCells")} />
          <RangeControl label="Emitter clearance" unit="cells" value={tuning.coneEmitterClearanceCells} min={0} max={8} step={0.25} displayDigits={2} onChange={(value) => updateTuning("coneEmitterClearanceCells", value)} modified={modified("coneEmitterClearanceCells")} onReset={resetTuning("coneEmitterClearanceCells")} />
        </div>

        <SubHeading title="Work budgets" note="CAPS ON AUDITED SHADER LOOPS" />
        <div className="svo-control-grid">
          <RangeControl label="Maximum traversal depth" unit="levels" value={svoMaximumTraversalDepth} min={1} max={21} step={1} displayDigits={0} onChange={setSvoMaximumTraversalDepth}
            hint="Hierarchy depth accepted by every camera traversal. A budget, not a LOD control: exceeding it reports traversal exhaustion rather than falling back to a coarser surface, which is why it lives here and the level-of-detail predicate lives under Detail." />
          <RangeControl label="Maximum node visits" unit="nodes" value={svoMaximumNodeVisits} min={1} max={256} step={1} displayDigits={0} onChange={setSvoMaximumNodeVisits}
            hint="Topology nodes allowed per primary traversal call." />
          <RangeControl label="Maximum leaf visits" unit="bricks" value={tuning.primaryLeafVisits} min={1} max={SVO_PRIMARY_LEAF_VISIT_HARD_LIMIT} step={1} displayDigits={0}
            onChange={(value) => updateTuning("primaryLeafVisits", value)} modified={modified("primaryLeafVisits")} onReset={resetTuning("primaryLeafVisits")} />
          <RangeControl label="Cone step budget" unit="steps" value={tuning.coneStepBudget} min={1} max={48} step={1} displayDigits={0} onChange={(value) => updateTuning("coneStepBudget", value)} modified={modified("coneStepBudget")} onReset={resetTuning("coneStepBudget")} />
          <RangeControl label="GI cones" unit="cones" value={tuning.giConeCount} min={3} max={4} step={1} displayDigits={0}
            onChange={(value) => updateTuning("giConeCount", value)} modified={modified("giConeCount")} onReset={resetTuning("giConeCount")}
            hint="Four cones give the best hemispherical coverage; three trades the normal cone for longer marches at the same total budget." />
          <RangeControl label="Shaded lights" unit="lights" value={tuning.maximumShadedLights} min={1} max={8} step={1} displayDigits={0} onChange={(value) => updateTuning("maximumShadedLights", value)} modified={modified("maximumShadedLights")} onReset={resetTuning("maximumShadedLights")} />
          <RangeControl label="Area samples · stable" unit="rays" value={tuning.stableAreaLightSamples} min={1} max={2} step={1} displayDigits={0} onChange={(value) => updateTuning("stableAreaLightSamples", value)} modified={modified("stableAreaLightSamples")} onReset={resetTuning("stableAreaLightSamples")} />
          <RangeControl label="Area samples · moving" unit="rays" value={tuning.movingAreaLightSamples} min={1} max={2} step={1} displayDigits={0} onChange={(value) => updateTuning("movingAreaLightSamples", value)} modified={modified("movingAreaLightSamples")} onReset={resetTuning("movingAreaLightSamples")} />
          <RangeControl label="AO samples · stable" unit="cones" value={tuning.stableAoSamples} min={1} max={4} step={1} displayDigits={0} onChange={(value) => updateTuning("stableAoSamples", value)} modified={modified("stableAoSamples")} onReset={resetTuning("stableAoSamples")} />
          <RangeControl label="AO samples · moving" unit="cones" value={tuning.movingAoSamples} min={1} max={4} step={1} displayDigits={0} onChange={(value) => updateTuning("movingAoSamples", value)} modified={modified("movingAoSamples")} onReset={resetTuning("movingAoSamples")} />
          <RangeControl label="Visibility nodes" unit="nodes" value={tuning.visibilityNodeVisits} min={1} max={128} step={1} displayDigits={0} onChange={(value) => updateTuning("visibilityNodeVisits", value)} modified={modified("visibilityNodeVisits")} onReset={resetTuning("visibilityNodeVisits")} />
          <RangeControl label="Visibility leaves" unit="bricks" value={tuning.visibilityLeafVisits} min={1} max={32} step={1} displayDigits={0} onChange={(value) => updateTuning("visibilityLeafVisits", value)} modified={modified("visibilityLeafVisits")} onReset={resetTuning("visibilityLeafVisits")} />
          <RangeControl label="Visibility voxel work" unit="tests" value={tuning.visibilityWorkItems} min={16} max={1024} step={16} displayDigits={0} onChange={(value) => updateTuning("visibilityWorkItems", value)} modified={modified("visibilityWorkItems")} onReset={resetTuning("visibilityWorkItems")} />
          <RangeControl label="Intersections" unit="hits" value={tuning.visibilityIntersections} min={1} max={4} step={1} displayDigits={0} onChange={(value) => updateTuning("visibilityIntersections", value)} modified={modified("visibilityIntersections")} onReset={resetTuning("visibilityIntersections")} />
        </div>
      </ControlGroup>
    </div>
  </aside>;
}
