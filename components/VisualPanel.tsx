"use client";

import { useDiagnosticsStore } from "@/lib/stores/diagnostics-store";
import { useUIStore } from "@/lib/stores/ui-store";
import {
  SVO_COST_OVERLAY_DEFINITIONS,
  SVO_COST_OVERLAY_LABELS,
  SVO_COST_OVERLAY_MODES,
  type SvoCostOverlayMode,
} from "@/lib/svo-render-diagnostics";
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
  const svoOverlayOpacity = useUIStore((state) => state.svoOverlayOpacity);
  const setSvoOverlayOpacity = useUIStore((state) => state.setSvoOverlayOpacity);
  const setRightPanel = useUIStore((state) => state.setRightPanel);

  const selectedView = SVO_COST_OVERLAY_DEFINITIONS[svoCostOverlay];
  const visualizationAvailable = svoRenderMode === "svo" && voxelRenderMode === "smooth";

  const selectVisualization = (mode: SvoCostOverlayMode) => {
    const nextMode = mode === svoCostOverlay && mode !== "off" ? "off" : mode;
    if (nextMode !== "off") {
      setSvoRenderMode("svo");
      setVoxelRenderMode("smooth");
    }
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

  return <aside
    id="render-panel"
    className="right-panel panel-scroll performance-panel performance-v2 visual-panel"
    aria-label="Rendering diagnostics and scene visualizations"
    data-testid="visual-panel"
  >
    <header className="performance-panel-header">
      <div><span>RENDER OBSERVATORY</span><h2>Scene presentation + ray diagnostics</h2></div>
      <div className="performance-panel-header-actions">
        <div className="measurement-mode" role="group" aria-label="Renderer">
          <span>PATH</span>
          <button type="button" aria-pressed={svoRenderMode === "svo"} onClick={() => selectRenderer("svo")}>SVO</button>
          <button type="button" aria-pressed={svoRenderMode === "raster"} onClick={() => selectRenderer("raster")}>RASTER</button>
        </div>
        <button className="panel-close" type="button" onClick={() => setRightPanel(null)} aria-label="Close render panel">×</button>
      </div>
    </header>

    <section className="paper-observatory" aria-labelledby="scene-presentation-heading">
      <header>
        <div><h3 id="scene-presentation-heading">Scene presentation</h3><small>PRODUCTION RENDERING · A/B CONTROLS</small></div>
        <span data-testid="effective-renderer-status">
          {effectiveRendererStatus.effectiveMode === "svo" ? "SVO ACTIVE" : "RASTER ACTIVE"}
        </span>
      </header>

      <div className="paper-view-controls render-control-row">
        <div>
          <span>LIGHTING PATH</span>
          <div role="group" aria-label="SVO lighting quality">
            <button className={svoLightingMode === "direct" ? "active" : ""} disabled={svoRenderMode !== "svo"} onClick={() => setSvoLightingMode("direct")}>DIRECT</button>
            <button className={svoLightingMode === "cone" ? "active" : ""} disabled={svoRenderMode !== "svo"} onClick={() => setSvoLightingMode("cone")}>BEAUTIFUL</button>
          </div>
        </div>
        <div>
          <span>LIGHTING EFFECTS</span>
          <div role="group" aria-label="SVO lighting effects">
            <button className={svoShadowsEnabled ? "active" : ""} disabled={svoRenderMode !== "svo"} aria-pressed={svoShadowsEnabled} onClick={() => setSvoShadowsEnabled(!svoShadowsEnabled)}>SHADOWS</button>
            <button className={svoAmbientOcclusionEnabled ? "active" : ""} disabled={svoRenderMode !== "svo"} aria-pressed={svoAmbientOcclusionEnabled} onClick={() => setSvoAmbientOcclusionEnabled(!svoAmbientOcclusionEnabled)}>AO</button>
          </div>
        </div>
      </div>

      <div className="paper-view-controls render-control-row render-representation-row">
        <div>
          <span>SCENE REPRESENTATION</span>
          <div role="group" aria-label="Scene representation">
            <button className={voxelRenderMode === "smooth" ? "active" : ""} onClick={() => selectRepresentation("smooth")}>FINISHED</button>
            <button className={voxelRenderMode === "raw-voxels" ? "active" : ""} onClick={() => selectRepresentation("raw-voxels")}>RAW</button>
            <button className={voxelRenderMode === "surface-voxels" ? "active" : ""} onClick={() => selectRepresentation("surface-voxels")}>SURFACE</button>
            <button className={voxelRenderMode === "brick-grid" ? "active" : ""} onClick={() => selectRepresentation("brick-grid")}>BRICKS</button>
          </div>
        </div>
      </div>

      <div className="paper-view-inspector" aria-live="polite">
        <header>
          <div><span>EFFECTIVE PATH</span><strong>{effectiveRendererStatus.effectiveMode === "svo" ? "Sparse voxel scene" : "Raster scene"}</strong></div>
          <code>{svoLightingMode} · {voxelRenderMode}</code>
        </header>
        <p>{effectiveRendererStatus.fallbackReason
          ? `The requested SVO renderer fell back because ${rendererFallbackLabels[effectiveRendererStatus.fallbackReason]}.`
          : svoRenderMode === "svo"
            ? "The camera ray resolves the dry scene through the sparse hierarchy; Beautiful adds wide-mip cone visibility."
            : "Raster is an explicit comparison path. Scene heatmaps require the finished SVO path."}</p>
      </div>
    </section>

    <section className="paper-observatory" data-testid="svo-cost-controls" aria-labelledby="traversal-views-heading">
      <header>
        <div><h3 id="traversal-views-heading">SVO traversal cost</h3><small>LIVE PER-PIXEL SCENE HEATMAPS</small></div>
        <span>{visualizationAvailable ? "GPU COUNTERS READY" : "SELECTS SVO + FINISHED"}</span>
      </header>

      <div className="paper-view-grid" role="group" aria-label="SVO cost overlay">
        {SVO_COST_OVERLAY_MODES.map((mode) => {
          const view = SVO_COST_OVERLAY_DEFINITIONS[mode];
          const active = svoCostOverlay === mode;
          return <button
            key={mode}
            type="button"
            className={active ? "active" : ""}
            aria-pressed={active}
            onClick={() => selectVisualization(mode)}
          >
            <span>{view.category}</span>
            <strong>{view.label}</strong>
            <small>{view.description}</small>
            <i className="render-view-palette" style={{ background: visualizationGradient(mode) }} aria-hidden="true" />
          </button>;
        })}
      </div>

      <div className="paper-view-inspector" aria-live="polite">
        <header>
          <div><span>ACTIVE VISUALIZATION</span><strong>{SVO_COST_OVERLAY_LABELS[svoCostOverlay]}</strong></div>
          <code>{svoCostOverlay === "off" ? "production radiance" : `overlay · ${Math.round(svoOverlayOpacity * 100)}%`}</code>
        </header>
        <p>{selectedView.description}</p>
        <small>SOURCE · {svoCostOverlay === "off" ? "PRODUCTION SCENE PRESENTATION" : `${selectedView.category.toUpperCase()} SHADER COUNTERS`}</small>
        <div className="paper-field-legend" aria-label={`${SVO_COST_OVERLAY_LABELS[svoCostOverlay]} legend`}>
          {selectedView.legend.map((entry) => <span key={entry.label}><i style={{ background: entry.color }} />{entry.label}</span>)}
        </div>
        <footer>{svoCostOverlay === "off"
          ? "Choose a view above to paint its measurements directly over the scene. Selecting one switches to the finished SVO path automatically."
          : "The heatmap is generated in the dry-scene shader from the work performed for that exact pixel; no field readback or simulation diagnostic is involved."}</footer>
      </div>

      {svoCostOverlay !== "off" && <div className="utility-controls" style={{ marginTop: 14 }}>
        <RangeControl
          label="Maximum traversal depth"
          unit="levels"
          value={svoMaximumTraversalDepth}
          min={1}
          max={21}
          step={1}
          displayDigits={0}
          onChange={setSvoMaximumTraversalDepth}
          hint="Diagnostic hierarchy ceiling. Rays requiring deeper nodes fail closed and appear in Budget failures."
        />
        <RangeControl
          label="Maximum node visits"
          unit="nodes"
          value={svoMaximumNodeVisits}
          min={1}
          max={256}
          step={1}
          displayDigits={0}
          onChange={setSvoMaximumNodeVisits}
          hint="Per-call topology budget. Reduce it to turn expensive rays into high-contrast failure regions."
        />
        <RangeControl
          label="Scene blend"
          unit="%"
          value={svoOverlayOpacity * 100}
          min={10}
          max={100}
          step={1}
          displayDigits={0}
          onChange={(value) => setSvoOverlayOpacity(value / 100)}
          hint="Blend between the authored radiance and the diagnostic palette."
        />
      </div>}
    </section>
  </aside>;
}
