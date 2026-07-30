"use client";

import type { ReactNode } from "react";
import { useDiagnosticsStore } from "@/lib/stores/diagnostics-store";
import { useUIStore } from "@/lib/stores/ui-store";
import {
  SVO_COST_OVERLAY_DEFINITIONS,
  SVO_COST_OVERLAY_LABELS,
  SVO_COST_OVERLAY_MODES,
  type SvoCostOverlayMode,
} from "@/lib/svo-render-diagnostics";
import {
  SVO_ENVIRONMENT_BRICK_REFINEMENT_MAXIMUM,
  SVO_PRIMARY_LEAF_VISIT_HARD_LIMIT,
  SVO_RENDER_TUNING_PRESETS,
  svoRenderTuningKey,
  type SvoRenderTuning,
  type SvoRenderTuningPreset,
} from "@/lib/svo-render-tuning";
import { RangeControl } from "./controls";

const rendererFallbackLabels = {
  "missing-source": "waiting for structural SVO data",
  "unsupported-terrain": "terrain source could not be represented",
  "unsupported-glass-cutout": "authored glazing needs an opaque shell cutout",
  "missing-pbr-materials": "production PBR material table is unavailable",
  "missing-lighting-publications": "production light/environment publications are unavailable",
  "pipeline-compile-failure": "SVO pipeline failed to compile",
  "inspection-mode": "a sparse inspection view is active",
} as const;

const visualizationGradient = (mode: SvoCostOverlayMode) => mode === "off"
  ? "linear-gradient(90deg,#071411,#65b7a4,#f2bc72)"
  : `linear-gradient(90deg,${SVO_COST_OVERLAY_DEFINITIONS[mode].legend
    .map((stop) => `${stop.color} ${Math.round(stop.at * 100)}%`).join(",")})`;

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

export function VisualPanel() {
  const effectiveRendererStatus = useDiagnosticsStore((state) => state.effectiveRendererStatus);
  const voxelRenderMode = useUIStore((state) => state.voxelRenderMode);
  const setVoxelRenderMode = useUIStore((state) => state.setVoxelRenderMode);
  const svoRenderMode = useUIStore((state) => state.svoRenderMode);
  const setSvoRenderMode = useUIStore((state) => state.setSvoRenderMode);
  const svoLightingMode = useUIStore((state) => state.svoLightingMode);
  const setSvoLightingMode = useUIStore((state) => state.setSvoLightingMode);
  const svoShadowsEnabled = useUIStore((state) => state.svoShadowsEnabled);
  const setSvoShadowsEnabled = useUIStore((state) => state.setSvoShadowsEnabled);
  const svoAmbientOcclusionEnabled = useUIStore((state) => state.svoAmbientOcclusionEnabled);
  const setSvoAmbientOcclusionEnabled = useUIStore((state) => state.setSvoAmbientOcclusionEnabled);
  const svoCostOverlay = useUIStore((state) => state.svoCostOverlay);
  const setSvoCostOverlay = useUIStore((state) => state.setSvoCostOverlay);
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

  const selectedView = SVO_COST_OVERLAY_DEFINITIONS[svoCostOverlay];
  const visualizationAvailable = svoRenderMode === "svo" && voxelRenderMode === "smooth";
  const tuningKey = svoRenderTuningKey(tuning);
  const activePreset = (Object.keys(SVO_RENDER_TUNING_PRESETS) as SvoRenderTuningPreset[])
    .find((preset) => svoRenderTuningKey(SVO_RENDER_TUNING_PRESETS[preset]) === tuningKey);
  const updateTuning = <K extends keyof SvoRenderTuning>(key: K, value: SvoRenderTuning[K]) =>
    setTuning((current) => ({ ...current, [key]: value }));
  const modified = <K extends keyof SvoRenderTuning>(key: K) => tuning[key] !== SVO_RENDER_TUNING_PRESETS.balanced[key];
  const resetTuning = <K extends keyof SvoRenderTuning>(key: K) => () => updateTuning(key, SVO_RENDER_TUNING_PRESETS.balanced[key]);

  const selectVisualization = (mode: SvoCostOverlayMode) => {
    const nextMode = mode === svoCostOverlay && mode !== "off" ? "off" : mode;
    if (nextMode !== "off") { setSvoRenderMode("svo"); setVoxelRenderMode("smooth"); }
    setSvoCostOverlay(nextMode);
  };
  const selectRenderer = (mode: "raster" | "svo") => {
    setSvoRenderMode(mode);
    if (mode === "raster") setSvoCostOverlay("off");
  };
  const selectRepresentation = (mode: Parameters<typeof setVoxelRenderMode>[0]) => {
    setVoxelRenderMode(mode);
    if (mode !== "smooth") setSvoCostOverlay("off");
  };
  // The trace explains the sparse path, so arming it selects that path rather
  // than silently doing nothing over the raster renderer.
  const enablePixelTrace = (enabled: boolean) => {
    if (enabled) { setSvoRenderMode("svo"); setVoxelRenderMode("smooth"); }
    setPixelTraceEnabled(enabled);
  };

  return <aside id="render-panel" className="right-panel panel-scroll performance-panel performance-v2 visual-panel"
    aria-label="Rendering diagnostics and scene visualizations" data-testid="visual-panel">
    <header className="performance-panel-header render-panel-header">
      <div><span>RENDER OBSERVATORY</span><h2>SVO performance tuning</h2></div>
      <div className="performance-panel-header-actions">
        <div className="measurement-mode" role="group" aria-label="Renderer">
          <button type="button" aria-pressed={svoRenderMode === "svo"} onClick={() => selectRenderer("svo")}>SVO</button>
          <button type="button" aria-pressed={svoRenderMode === "raster"} onClick={() => selectRenderer("raster")}>RASTER</button>
        </div>
        <button className="panel-close" type="button" onClick={() => setRightPanel(null)} aria-label="Close render panel">×</button>
      </div>
      <div className="render-status-line">
        <span className={effectiveRendererStatus.effectiveMode === "svo" ? "online" : ""} />
        <strong data-testid="effective-renderer-status">{effectiveRendererStatus.effectiveMode === "svo" ? "SVO ACTIVE" : "RASTER ACTIVE"}</strong>
        <code>{Math.round(tuning.resolutionScale * 100)}% · cone {tuning.coneLightingScale === 1 ? "full" : `${1 / tuning.coneLightingScale}×${1 / tuning.coneLightingScale}`} · {tuning.temporalEnabled ? "TAA" : "RAW"}</code>
      </div>
    </header>

    <div className="render-preset-strip" role="group" aria-label="Render performance profile">
      <span>PROFILE</span>
      {(Object.keys(SVO_RENDER_TUNING_PRESETS) as SvoRenderTuningPreset[]).map((preset) =>
        <button key={preset} className={activePreset === preset ? "active" : ""} onClick={() => setTuning(SVO_RENDER_TUNING_PRESETS[preset])}>{preset}</button>)}
      {!activePreset && <output>CUSTOM</output>}
    </div>

    <div className="render-groups">
      <ControlGroup title="Presentation" note="path · representation · rate" open>
        <div className="render-segment-row">
          <div><span>LIGHTING PATH</span><div role="group" aria-label="SVO lighting quality">
            <button className={svoLightingMode === "direct" ? "active" : ""} disabled={svoRenderMode !== "svo"} onClick={() => setSvoLightingMode("direct")}>DIRECT</button>
            <button className={svoLightingMode === "cone" ? "active" : ""} disabled={svoRenderMode !== "svo"} onClick={() => setSvoLightingMode("cone")}>BEAUTIFUL</button>
            <button className={svoLightingMode === "gi" ? "active" : ""} disabled={svoRenderMode !== "svo"} onClick={() => setSvoLightingMode("gi")}>GLOBAL</button>
          </div></div>
          <div><span>SCENE REPRESENTATION</span><div role="group" aria-label="Scene representation">
            <button className={voxelRenderMode === "smooth" ? "active" : ""} onClick={() => selectRepresentation("smooth")}>FINISHED</button>
            <button className={voxelRenderMode === "raw-voxels" ? "active" : ""} onClick={() => selectRepresentation("raw-voxels")}>RAW</button>
            <button className={voxelRenderMode === "surface-voxels" ? "active" : ""} onClick={() => selectRepresentation("surface-voxels")}>SURFACE</button>
            <button className={voxelRenderMode === "brick-grid" ? "active" : ""} onClick={() => selectRepresentation("brick-grid")}>BRICKS</button>
            <button className={voxelRenderMode === "occupied-bricks" ? "active" : ""} onClick={() => selectRepresentation("occupied-bricks")} title="Show only bricks containing material payload">CONTENT</button>
          </div></div>
        </div>
        <div className="render-toggle-row" role="group" aria-label="SVO lighting effects">
          <Toggle label="Shadows" checked={svoShadowsEnabled} disabled={svoRenderMode !== "svo"} onChange={setSvoShadowsEnabled} />
          <Toggle label="AO" checked={svoAmbientOcclusionEnabled} disabled={svoRenderMode !== "svo"} onChange={setSvoAmbientOcclusionEnabled} />
          <Toggle label="Temporal" checked={tuning.temporalEnabled} onChange={(value) => updateTuning("temporalEnabled", value)} />
          <Toggle label="Interlaced shadows" checked={tuning.checkerboardShadowsEnabled} disabled={!tuning.temporalEnabled}
            onChange={(value) => updateTuning("checkerboardShadowsEnabled", value)} />
        </div>
        <div className="svo-control-grid">
          <RangeControl label="Render resolution" unit="%" value={tuning.resolutionScale * 100} min={35} max={100} step={1} displayDigits={0}
            onChange={(value) => updateTuning("resolutionScale", value / 100)} modified={modified("resolutionScale")} onReset={resetTuning("resolutionScale")} />
          <RangeControl label="Environment brick refinement" unit="levels" value={tuning.environmentBrickRefinementLevels}
            min={0} max={SVO_ENVIRONMENT_BRICK_REFINEMENT_MAXIMUM} step={1} displayDigits={0}
            onChange={(value) => updateTuning("environmentBrickRefinementLevels", value)}
            modified={modified("environmentBrickRefinementLevels")} onReset={resetTuning("environmentBrickRefinementLevels")}
            hint="Additional SVO subdivision for authored scenery outside the simulation lattice. Changing it rebuilds the sparse world." />
          <label className="render-discrete-control"><span>Cone prepass</span><div>
            {([{ scale: 1, label: "FULL" }, { scale: 0.5, label: "2×2" }, { scale: 0.25, label: "4×4" }, { scale: 0.125, label: "8×8" }] as const)
              .map(({ scale, label }) => <button key={scale} className={tuning.coneLightingScale === scale ? "active" : ""}
                onClick={() => updateTuning("coneLightingScale", scale)}>{label}</button>)}
          </div></label>
          <label className="render-discrete-control reconstruction-control"><span>Lighting reconstruction</span><div>
            {([{ mode: "nearest", label: "EXACT" }, { mode: "gated-linear", label: "LINEAR" }, { mode: "joint-bilateral", label: "BILAT" },
              { mode: "wide-relight", label: "WIDE" }, { mode: "full-res-relight", label: "RELIGHT" }] as const)
              .map(({ mode, label }) => <button key={mode} className={tuning.coneRadianceReconstruction === mode ? "active" : ""}
                disabled={tuning.coneLightingScale === 1} onClick={() => updateTuning("coneRadianceReconstruction", mode)}>{label}</button>)}
          </div></label>
        </div>
        {effectiveRendererStatus.fallbackReason && <p className="render-inline-warning">SVO fallback: {rendererFallbackLabels[effectiveRendererStatus.fallbackReason]}.</p>}
      </ControlGroup>

      <ControlGroup title="Primary tracing" note="camera ray work caps" open>
        <div className="svo-control-grid">
          <RangeControl label="Maximum traversal depth" unit="levels" value={svoMaximumTraversalDepth} min={1} max={21} step={1} displayDigits={0} onChange={setSvoMaximumTraversalDepth}
            hint="Hierarchy depth accepted by every camera traversal." />
          <RangeControl label="Maximum node visits" unit="nodes" value={svoMaximumNodeVisits} min={1} max={256} step={1} displayDigits={0} onChange={setSvoMaximumNodeVisits}
            hint="Topology nodes allowed per primary traversal call." />
          <RangeControl label="Maximum leaf visits" unit="bricks" value={tuning.primaryLeafVisits} min={1} max={SVO_PRIMARY_LEAF_VISIT_HARD_LIMIT} step={1} displayDigits={0}
            onChange={(value) => updateTuning("primaryLeafVisits", value)} modified={modified("primaryLeafVisits")} onReset={resetTuning("primaryLeafVisits")} />
        </div>
      </ControlGroup>

      <ControlGroup title="Direct lighting" note="lights · samples · exact rays">
        <div className="svo-control-grid">
          <RangeControl label="Shaded lights" unit="lights" value={tuning.maximumShadedLights} min={1} max={8} step={1} displayDigits={0} onChange={(value) => updateTuning("maximumShadedLights", value)} modified={modified("maximumShadedLights")} onReset={resetTuning("maximumShadedLights")} />
          <RangeControl label="Area samples · stable" unit="rays" value={tuning.stableAreaLightSamples} min={1} max={2} step={1} displayDigits={0} onChange={(value) => updateTuning("stableAreaLightSamples", value)} modified={modified("stableAreaLightSamples")} onReset={resetTuning("stableAreaLightSamples")} />
          <RangeControl label="Area samples · moving" unit="rays" value={tuning.movingAreaLightSamples} min={1} max={2} step={1} displayDigits={0} onChange={(value) => updateTuning("movingAreaLightSamples", value)} modified={modified("movingAreaLightSamples")} onReset={resetTuning("movingAreaLightSamples")} />
          <RangeControl label="Shadow strength" unit="%" value={tuning.shadowStrength * 100} min={0} max={100} step={1} displayDigits={0} onChange={(value) => updateTuning("shadowStrength", value / 100)} modified={modified("shadowStrength")} onReset={resetTuning("shadowStrength")} />
          <RangeControl label="Shadow origin bias" unit="cells" value={tuning.shadowBiasCells} min={0} max={0.25} step={0.005} displayDigits={3} onChange={(value) => updateTuning("shadowBiasCells", value)} modified={modified("shadowBiasCells")} onReset={resetTuning("shadowBiasCells")} />
          <RangeControl label="Visibility nodes" unit="nodes" value={tuning.visibilityNodeVisits} min={1} max={128} step={1} displayDigits={0} onChange={(value) => updateTuning("visibilityNodeVisits", value)} modified={modified("visibilityNodeVisits")} onReset={resetTuning("visibilityNodeVisits")} />
          <RangeControl label="Visibility leaves" unit="bricks" value={tuning.visibilityLeafVisits} min={1} max={32} step={1} displayDigits={0} onChange={(value) => updateTuning("visibilityLeafVisits", value)} modified={modified("visibilityLeafVisits")} onReset={resetTuning("visibilityLeafVisits")} />
          <RangeControl label="Visibility voxel work" unit="tests" value={tuning.visibilityWorkItems} min={16} max={1024} step={16} displayDigits={0} onChange={(value) => updateTuning("visibilityWorkItems", value)} modified={modified("visibilityWorkItems")} onReset={resetTuning("visibilityWorkItems")} />
          <RangeControl label="Intersections" unit="hits" value={tuning.visibilityIntersections} min={1} max={4} step={1} displayDigits={0} onChange={(value) => updateTuning("visibilityIntersections", value)} modified={modified("visibilityIntersections")} onReset={resetTuning("visibilityIntersections")} />
        </div>
      </ControlGroup>

      <ControlGroup title="Ambient occlusion" note="contact cone quality">
        <div className="svo-control-grid">
          <RangeControl label="AO samples · stable" unit="cones" value={tuning.stableAoSamples} min={1} max={4} step={1} displayDigits={0} onChange={(value) => updateTuning("stableAoSamples", value)} modified={modified("stableAoSamples")} onReset={resetTuning("stableAoSamples")} />
          <RangeControl label="AO samples · moving" unit="cones" value={tuning.movingAoSamples} min={1} max={4} step={1} displayDigits={0} onChange={(value) => updateTuning("movingAoSamples", value)} modified={modified("movingAoSamples")} onReset={resetTuning("movingAoSamples")} />
          <RangeControl label="AO radius" unit="×" value={tuning.aoRadiusScale} min={0.1} max={3} step={0.05} displayDigits={2} onChange={(value) => updateTuning("aoRadiusScale", value)} modified={modified("aoRadiusScale")} onReset={resetTuning("aoRadiusScale")} />
          <RangeControl label="AO strength" unit="%" value={tuning.aoStrength * 100} min={0} max={100} step={1} displayDigits={0} onChange={(value) => updateTuning("aoStrength", value / 100)} modified={modified("aoStrength")} onReset={resetTuning("aoStrength")} />
          <RangeControl label="AO cone aperture" unit="rad" value={tuning.aoConeAperture} min={0.1} max={1.4} step={0.01} displayDigits={2} onChange={(value) => updateTuning("aoConeAperture", value)} modified={modified("aoConeAperture")} onReset={resetTuning("aoConeAperture")} />
        </div>
      </ControlGroup>

      <ControlGroup title="Cone tracing" note="step size · softness · clearance" open>
        <div className="svo-control-grid">
          <RangeControl label="Cone step budget" unit="steps" value={tuning.coneStepBudget} min={1} max={48} step={1} displayDigits={0} onChange={(value) => updateTuning("coneStepBudget", value)} modified={modified("coneStepBudget")} onReset={resetTuning("coneStepBudget")} />
          <RangeControl label="Shadow cone aperture" unit="rad" value={tuning.shadowConeAperture} min={0.01} max={0.25} step={0.005} displayDigits={3}
            onChange={(value) => updateTuning("shadowConeAperture", value)} modified={modified("shadowConeAperture")} onReset={resetTuning("shadowConeAperture")}
            hint="Wider cones take larger march steps and produce softer shadows; narrower cones preserve sharper shadows but require more taps." />
          <RangeControl label="Normal escape" unit="cells" value={tuning.coneNormalEscapeCells} min={0} max={2} step={0.05} displayDigits={2} onChange={(value) => updateTuning("coneNormalEscapeCells", value)} modified={modified("coneNormalEscapeCells")} onReset={resetTuning("coneNormalEscapeCells")} />
          <RangeControl label="Emitter clearance" unit="cells" value={tuning.coneEmitterClearanceCells} min={0} max={8} step={0.25} displayDigits={2} onChange={(value) => updateTuning("coneEmitterClearanceCells", value)} modified={modified("coneEmitterClearanceCells")} onReset={resetTuning("coneEmitterClearanceCells")} />
        </div>
      </ControlGroup>

      <ControlGroup title="Temporal resolve" note="history stability">
        <div className="svo-control-grid">
          <RangeControl label="Maximum history" unit="samples" value={tuning.temporalMaximumSamples} min={1} max={128} step={1} displayDigits={0} onChange={(value) => updateTuning("temporalMaximumSamples", value)} modified={modified("temporalMaximumSamples")} onReset={resetTuning("temporalMaximumSamples")} />
          <RangeControl label="Variance clamp" unit="σ" value={tuning.temporalVarianceSigma} min={0.5} max={4} step={0.1} displayDigits={1} onChange={(value) => updateTuning("temporalVarianceSigma", value)} modified={modified("temporalVarianceSigma")} onReset={resetTuning("temporalVarianceSigma")} />
          <RangeControl label="Depth tolerance" unit="×" value={tuning.temporalDepthToleranceScale} min={0.25} max={4} step={0.05} displayDigits={2} onChange={(value) => updateTuning("temporalDepthToleranceScale", value)} modified={modified("temporalDepthToleranceScale")} onReset={resetTuning("temporalDepthToleranceScale")} />
        </div>
      </ControlGroup>

      <ControlGroup title="SVO traversal cost" note="LIVE PER-PIXEL SCENE HEATMAPS">
        <div className="paper-view-grid" role="group" aria-label="SVO cost overlay">
          {SVO_COST_OVERLAY_MODES.map((mode) => {
            const view = SVO_COST_OVERLAY_DEFINITIONS[mode];
            const active = svoCostOverlay === mode;
            return <button key={mode} type="button" className={active ? "active" : ""} aria-pressed={active} onClick={() => selectVisualization(mode)}>
              <span>{view.category}</span><strong>{view.label}</strong><small>{view.description}</small>
              <i className="render-view-palette" style={{ background: visualizationGradient(mode) }} aria-hidden="true" />
            </button>;
          })}
        </div>
        <div className="paper-view-inspector" aria-live="polite">
          <header>
            <div><span>ACTIVE VISUALIZATION</span><strong>{SVO_COST_OVERLAY_LABELS[svoCostOverlay]}</strong></div>
            <code>{svoCostOverlay === "off" ? "production radiance" : "diagnostic signal only"}</code>
          </header>
          <p>{selectedView.description}</p>
          <small>SOURCE · {svoCostOverlay === "off" ? "PRODUCTION SCENE PRESENTATION" : `${selectedView.category.toUpperCase()} SHADER COUNTERS`}</small>
          <div className="paper-field-legend" aria-label={`${SVO_COST_OVERLAY_LABELS[svoCostOverlay]} legend`}>
            {selectedView.legend.map((entry) => <span key={entry.label}><i style={{ background: entry.color }} />{entry.label}</span>)}
          </div>
          <footer>{svoCostOverlay === "off"
            ? "Choose a view above to replace scene radiance with its per-pixel measurements. Selecting one switches to the finished SVO path automatically."
            : "The heatmap replaces regular rendering and is generated from the work performed for that exact pixel; no field readback is involved."}</footer>
        </div>
        <p className="render-visualization-status">{visualizationAvailable ? "GPU COUNTERS READY" : "SELECTS SVO + FINISHED"}</p>
      </ControlGroup>

      <ControlGroup title="Ray work under the pointer" note="LIVE 3D TRACE OF ONE PIXEL" open>
        <div className="render-toggle-row" role="group" aria-label="Live ray-work diagnostic">
          <Toggle label="Trace hovered pixel" checked={pixelTraceEnabled} onChange={enablePixelTrace} />
          <Toggle
            label="Pin ray"
            checked={pixelTracePinned}
            disabled={!pixelTraceEnabled}
            // The viewport owns which pixel and which view a pin records, so this
            // asks for one rather than declaring one from out here.
            onChange={(on) => (on ? requestPixelTracePin() : setPixelTracePinned(false))}
          />
        </div>
        <p className="panel-note">
          Hovering the viewport re-traces that exact pixel with an instrumented mirror of the shipping shader and draws
          its work in the scene: the octree boxes the ray opened, the children it rejected, the leaf bricks and fine
          cells it walked, the analytic surface tests it issued, and the shadow and occlusion cones it marched. The ray
          itself is an arrow chain — one arrow per brick it walked and per empty stretch it skipped — and each cone is
          stitched from its taps, coloured by the mip level that tap was allowed to read. Click the viewport to freeze
          that exact ray and orbit around its work; click again to follow the pointer.
        </p>
        <p className="render-visualization-status">{visualizationAvailable ? "PROBE READY" : "SELECTS SVO + FINISHED"}</p>
      </ControlGroup>
    </div>
  </aside>;
}
