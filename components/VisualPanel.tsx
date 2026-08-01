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
  "pipeline-compiling": "SVO pipeline is compiling",
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
  const svoShadowsEnabled = useUIStore((state) => state.svoShadowsEnabled);
  const setSvoShadowsEnabled = useUIStore((state) => state.setSvoShadowsEnabled);
  const svoAmbientOcclusionEnabled = useUIStore((state) => state.svoAmbientOcclusionEnabled);
  const setSvoAmbientOcclusionEnabled = useUIStore((state) => state.setSvoAmbientOcclusionEnabled);
  const svoConeTracingMode = useUIStore((state) => state.svoConeTracingMode);
  const setSvoConeTracingMode = useUIStore((state) => state.setSvoConeTracingMode);
  const svoPrimaryTraversal = useUIStore((state) => state.svoPrimaryTraversal);
  const setSvoPrimaryTraversal = useUIStore((state) => state.setSvoPrimaryTraversal);
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
  const fluidCellTraceEnabled = useUIStore((state) => state.fluidCellTraceEnabled);
  const fluidCellTracePinned = useUIStore((state) => state.fluidCellTracePinned);
  const setFluidCellTraceEnabled = useUIStore((state) => state.setFluidCellTraceEnabled);
  const setFluidCellTracePinned = useUIStore((state) => state.setFluidCellTracePinned);
  const requestFluidCellTracePin = useUIStore((state) => state.requestFluidCellTracePin);

  const selectedView = SVO_COST_OVERLAY_DEFINITIONS[svoCostOverlay];
  const visualizationAvailable = voxelRenderMode === "smooth";
  const tuningKey = svoRenderTuningKey(tuning);
  const activePreset = (Object.keys(SVO_RENDER_TUNING_PRESETS) as SvoRenderTuningPreset[])
    .find((preset) => svoRenderTuningKey(SVO_RENDER_TUNING_PRESETS[preset]) === tuningKey);
  const updateTuning = <K extends keyof SvoRenderTuning>(key: K, value: SvoRenderTuning[K]) =>
    setTuning((current) => ({ ...current, [key]: value }));
  const modified = <K extends keyof SvoRenderTuning>(key: K) => tuning[key] !== SVO_RENDER_TUNING_PRESETS.balanced[key];
  const resetTuning = <K extends keyof SvoRenderTuning>(key: K) => () => updateTuning(key, SVO_RENDER_TUNING_PRESETS.balanced[key]);

  const selectVisualization = (mode: SvoCostOverlayMode) => {
    const nextMode = mode === svoCostOverlay && mode !== "off" ? "off" : mode;
    if (nextMode !== "off") setVoxelRenderMode("smooth");
    setSvoCostOverlay(nextMode);
  };
  const selectRepresentation = (mode: Parameters<typeof setVoxelRenderMode>[0]) => {
    setVoxelRenderMode(mode);
    if (mode !== "smooth") setSvoCostOverlay("off");
  };
  // The trace explains the finished sparse path, so arming it leaves any
  // representation inspection view before probing.
  const enablePixelTrace = (enabled: boolean) => {
    if (enabled) setVoxelRenderMode("smooth");
    setPixelTraceEnabled(enabled);
  };

  return <aside id="render-panel" className="right-panel panel-scroll performance-panel performance-v2 visual-panel"
    aria-label="Rendering diagnostics and scene visualizations" data-testid="visual-panel">
    <header className="performance-panel-header render-panel-header">
      <div><span>RENDER OBSERVATORY</span><h2>SVO performance tuning</h2></div>
      <div className="performance-panel-header-actions">
        <button className="panel-close" type="button" onClick={() => setRightPanel(null)} aria-label="Close render panel">×</button>
      </div>
      <div className="render-status-line">
        <span className={effectiveRendererStatus.effectiveMode === "svo" ? "online" : ""} />
        <strong data-testid="effective-renderer-status">{effectiveRendererStatus.effectiveMode === "svo" ? "SVO GI ACTIVE" : "RASTER FALLBACK"}</strong>
        <code>{Math.round(tuning.resolutionScale * 100)}% · cone {svoConeTracingMode !== "cones" ? svoConeTracingMode : tuning.coneLightingScale === 1 ? "full" : `${1 / tuning.coneLightingScale}×${1 / tuning.coneLightingScale}`}</code>
      </div>
    </header>

    <div className="render-preset-strip" role="group" aria-label="Render performance profile">
      <span>PROFILE</span>
      {(Object.keys(SVO_RENDER_TUNING_PRESETS) as SvoRenderTuningPreset[]).map((preset) =>
        <button key={preset} className={activePreset === preset ? "active" : ""} onClick={() => setTuning(SVO_RENDER_TUNING_PRESETS[preset])}>{preset}</button>)}
      {!activePreset && <output>CUSTOM</output>}
    </div>

    <div className="render-groups">
      <ControlGroup title="Presentation" note="GLOBAL view · rate" open>
        <div className="render-segment-row">
          <div><span>GLOBAL SVO VIEW</span><div role="group" aria-label="Global SVO view">
            <button className={voxelRenderMode === "smooth" ? "active" : ""} onClick={() => selectRepresentation("smooth")}>SHADED</button>
            <button className={voxelRenderMode === "raw-voxels" ? "active" : ""} onClick={() => selectRepresentation("raw-voxels")}>RAW</button>
            <button className={voxelRenderMode === "surface-voxels" ? "active" : ""} onClick={() => selectRepresentation("surface-voxels")}>SURFACE</button>
            <button className={voxelRenderMode === "brick-grid" ? "active" : ""} onClick={() => selectRepresentation("brick-grid")}>BRICKS</button>
            <button className={voxelRenderMode === "occupied-bricks" ? "active" : ""} onClick={() => selectRepresentation("occupied-bricks")} title="Show only bricks containing material payload">CONTENT</button>
          </div></div>
        </div>
        <div className="render-toggle-row" role="group" aria-label="SVO lighting effects">
          <Toggle label="Shadows" checked={svoShadowsEnabled} disabled={svoConeTracingMode === "off"} onChange={setSvoShadowsEnabled} />
          <Toggle label="AO" checked={svoAmbientOcclusionEnabled} disabled={svoConeTracingMode === "off"} onChange={setSvoAmbientOcclusionEnabled} />
        </div>
        <div className="svo-control-grid">
          <RangeControl label="Render resolution" unit="%" value={tuning.resolutionScale * 100} min={35} max={100} step={1} displayDigits={0}
            onChange={(value) => updateTuning("resolutionScale", value / 100)} modified={modified("resolutionScale")} onReset={resetTuning("resolutionScale")} />
          <RangeControl label="Environment brick refinement" unit="levels" value={tuning.environmentBrickRefinementLevels}
            min={0} max={SVO_ENVIRONMENT_BRICK_REFINEMENT_MAXIMUM} step={1} displayDigits={0}
            onChange={(value) => updateTuning("environmentBrickRefinementLevels", value)}
            modified={modified("environmentBrickRefinementLevels")} onReset={resetTuning("environmentBrickRefinementLevels")}
            hint="Additional SVO subdivision for authored scenery outside the simulation lattice. Changing it rebuilds the sparse world." />
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
          <label className="render-discrete-control reconstruction-control"><span>Lighting reconstruction</span><div>
            {([{ mode: "nearest", label: "EXACT" }, { mode: "gated-linear", label: "LINEAR" }, { mode: "joint-bilateral", label: "BILAT" },
              { mode: "wide-relight", label: "WIDE" }, { mode: "full-res-relight", label: "RELIGHT" }] as const)
              .map(({ mode, label }) => <button key={mode} className={tuning.coneRadianceReconstruction === mode ? "active" : ""}
                disabled={tuning.coneLightingScale === 1 || svoConeTracingMode !== "cones"} onClick={() => updateTuning("coneRadianceReconstruction", mode)}>{label}</button>)}
          </div></label>
        </div>
        {effectiveRendererStatus.fallbackReason && <p className="render-inline-warning">SVO fallback: {rendererFallbackLabels[effectiveRendererStatus.fallbackReason]}.</p>}
      </ControlGroup>

      <ControlGroup title="Primary tracing" note="camera ray work caps" open>
        <div className="render-toggle-row" role="group" aria-label="SVO primary tracing optimizations">
          <Toggle label="Reuse stationary visibility" checked={tuning.stationaryPrimaryReuseEnabled}
            disabled={svoPrimaryTraversal === "raster"}
            hint="Reuse the exact primary G-buffer for an unchanged camera in static scenes or while simulation is paused. The raster primary is cheap enough not to cache, and its impostor pass blocks the reuse anyway."
            onChange={(value) => updateTuning("stationaryPrimaryReuseEnabled", value)} />
        </div>
        <div className="svo-control-grid">
          <label className="render-discrete-control"><span>Primary visibility</span><div>
            {([
              { mode: "raster", label: "RASTER", hint: "Hardware-rasterize the resident bricks as depth-tested proxies. Octree leaves partition space, so the depth test is exact: the image matches TRACED pixel for pixel. 29.0 ms against 49.6 ms at 1500x1500 garden." },
              { mode: "traced", label: "TRACED", hint: "The full-screen traversal megakernel, where every pixel marches the octree for itself. The reference the raster path is measured against." },
            ] as const)
              .map(({ mode, label, hint }) => <button key={mode} className={svoPrimaryTraversal === mode ? "active" : ""}
                title={hint} onClick={() => setSvoPrimaryTraversal(mode)}>{label}</button>)}
          </div></label>
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

      <ControlGroup title="Global illumination" note="bounce · occlusion · energy balance" open>
        <div className="svo-control-grid">
          <RangeControl label="GI bounce" unit="%" value={tuning.giBounceStrength * 100} min={0} max={400} step={5} displayDigits={0}
            onChange={(value) => updateTuning("giBounceStrength", value / 100)} modified={modified("giBounceStrength")} onReset={resetTuning("giBounceStrength")}
            hint="Exposure for gathered diffuse bounce. This does not amplify direct highlights or emissive surfaces." />
          <RangeControl label="GI occlusion" unit="%" value={tuning.giOcclusionStrength * 100} min={0} max={100} step={1} displayDigits={0}
            onChange={(value) => updateTuning("giOcclusionStrength", value / 100)} modified={modified("giOcclusionStrength")} onReset={resetTuning("giOcclusionStrength")}
            hint="Uses the same wide GI cones to darken diffuse environment fill in enclosed regions. The AO toggle enables this in GLOBAL mode." />
          <RangeControl label="Diffuse environment" unit="%" value={tuning.giEnvironmentStrength * 100} min={0} max={200} step={5} displayDigits={0}
            onChange={(value) => updateTuning("giEnvironmentStrength", value / 100)} modified={modified("giEnvironmentStrength")} onReset={resetTuning("giEnvironmentStrength")}
            hint="Analytic sky fill retained alongside GI. Lower this when bounced light should carry more of the diffuse scene." />
          <RangeControl label="Direct key" unit="%" value={tuning.giDirectStrength * 100} min={0} max={200} step={5} displayDigits={0}
            onChange={(value) => updateTuning("giDirectStrength", value / 100)} modified={modified("giDirectStrength")} onReset={resetTuning("giDirectStrength")}
            hint="Exact primary light retained in GLOBAL mode. It should remain the crisp key while GI fills indirect regions." />
          <RangeControl label="GI cones" unit="cones" value={tuning.giConeCount} min={3} max={4} step={1} displayDigits={0}
            onChange={(value) => updateTuning("giConeCount", value)} modified={modified("giConeCount")} onReset={resetTuning("giConeCount")}
            hint="Four cones give the best hemispherical coverage; three trades the normal cone for longer marches at the same total budget." />
          <RangeControl label="GI cone aperture" unit="rad" value={tuning.giConeAperture} min={0.4} max={1.4} step={0.01} displayDigits={2}
            onChange={(value) => updateTuning("giConeAperture", value)} modified={modified("giConeAperture")} onReset={resetTuning("giConeAperture")}
            hint="Wide apertures survey broad scene regions and produce smoother, lower-frequency bounce and occlusion." />
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

      <ControlGroup title="Cell work under the pointer" note="ONE PRESSURE CELL, GATHERED AND SCHEDULED">
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
          Hovering the fluid names the pressure cell behind that pixel and reads what the frame published about it:
          its leaf size and compact row, the assembled operator diagonal and right-hand side, the eighteen power
          neighbours it couples to and which of them sit at a different resolution. Beside that it reports what the
          encoded solve does to a row — how many level-0 sweeps re-read its stencil, and how few stages it takes before
          the cell depends on every other cell. Gathered and scheduled figures are badged separately and never added.
        </p>
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
          Hovering the viewport reports what the frame actually did to produce that one pixel, and draws it in the
          scene. Under rasterized primary visibility that is a tournament, not a walk: every brick proxy whose box
          covers the pixel is drawn as a box coloured by its place in the draw order, each one a fragment that ran a
          bounded DDA, with the beaten ones dashed where the brick held no surface and dimmed where it simply sat
          behind the winner. The winning brick keeps its full leaf box as a hairline, so the published occupancy
          tightening the draw is visible. Under traced primary visibility the same panel shows the walk instead — the
          octree boxes the ray opened, the children it rejected, the bricks and cells it stepped. Lighting is the same
          either way: shadow rays, and cones stitched from their taps and coloured by the mip level each tap was
          allowed to read. Click the viewport to freeze that exact pixel and orbit around its work; click again to
          follow the pointer.
        </p>
        <p className="render-visualization-status">{visualizationAvailable ? "PROBE READY" : "SELECTS SVO + FINISHED"}</p>
      </ControlGroup>
    </div>
  </aside>;
}
