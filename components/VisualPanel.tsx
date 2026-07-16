"use client";

import { getMethod } from "@/lib/methods";
import { useDiagnosticsStore } from "@/lib/stores/diagnostics-store";
import { useMethodStore } from "@/lib/stores/method-store";
import { useUIStore } from "@/lib/stores/ui-store";
import { environmentPresets } from "@/lib/environments";

export function VisualPanel() {
  const methodId = useMethodStore((state) => state.methodId);
  const gpuInfo = useDiagnosticsStore((state) => state.gpuInfo);
  const view = useUIStore((state) => state.view);
  const setView = useUIStore((state) => state.setView);
  const waterRenderMode = useUIStore((state) => state.waterRenderMode);
  const setWaterRenderMode = useUIStore((state) => state.setWaterRenderMode);
  const environmentId = useUIStore((state) => state.environmentId);
  const setEnvironmentId = useUIStore((state) => state.setEnvironmentId);
  const gridOverlayAxis = useUIStore((state) => state.gridOverlayAxis);
  const setGridOverlayAxis = useUIStore((state) => state.setGridOverlayAxis);
  const gridOverlaySlice = useUIStore((state) => state.gridOverlaySlice);
  const setGridOverlaySlice = useUIStore((state) => state.setGridOverlaySlice);
  const setRightPanel = useUIStore((state) => state.setRightPanel);
  const gridKind = getMethod(methodId).backend === "cpu" ? "uniform" : gpuInfo?.gridKind ?? "uniform";
  const tall = gridKind !== "uniform";

  return <aside className="right-panel panel-scroll visual-panel" data-testid="visual-panel">
    <section className="panel-section utility-panel-head">
      <div><p className="eyebrow">VIEWPORT</p><strong>Render &amp; debug</strong></div>
      <button className="panel-close" onClick={() => setRightPanel(null)} aria-label="Close render panel">×</button>
    </section>

    <section className="panel-section utility-controls">
      <div className="section-heading"><h2>View mode</h2><span>COMPOSITION</span></div>
      <div className="segmented compact">
        <button className={view === "scientific" ? "active" : ""} onClick={() => setView("scientific")}>Scientific</button>
        <button className={view === "presentation" ? "active" : ""} onClick={() => setView("presentation")}>Presentation</button>
      </div>
      <small className="control-hint">Scientific mode exposes solver instrumentation; presentation mode keeps the clean rendered scene.</small>
    </section>

    <section className="panel-section utility-controls">
      <div className="section-heading"><h2>Environment</h2><span>ART DIRECTION</span></div>
      <div className="environment-grid" role="radiogroup" aria-label="Scene environment">
        {environmentPresets.map((preset) => <button
          key={preset.id}
          type="button"
          role="radio"
          aria-checked={environmentId === preset.id}
          className={environmentId === preset.id ? "active" : ""}
          onClick={() => setEnvironmentId(preset.id)}
          title={preset.description}
        >
          <span className="environment-swatch" aria-hidden="true">
            {preset.swatch.map((color) => <i key={color} style={{ background: color }} />)}
          </span>
          <span><strong>{preset.shortName}</strong><small>{preset.description}</small></span>
        </button>)}
      </div>
      <small className="control-hint">Architecture and foreground elements live in world space so the water bends them naturally as the camera moves.</small>
    </section>

    <section className="panel-section utility-controls">
      <div className="section-heading"><h2>Water rendering</h2><span>OPTICS</span></div>
      <div className="segmented compact">
        <button className={waterRenderMode === "rasterized" ? "active" : ""} onClick={() => setWaterRenderMode("rasterized")}>Raster optics</button>
        <button className={waterRenderMode === "ray-marched" ? "active" : ""} onClick={() => setWaterRenderMode("ray-marched")}>Ray march</button>
      </div>
      <small className="control-hint">Raster optics uses the extracted liquid surface. Ray march samples the solver volume directly for comparison.</small>
    </section>

    <section className="panel-section utility-controls">
      <div className="section-heading"><h2>Solver grid</h2><span>DEBUG LAYER</span></div>
      {view === "scientific" ? <>
        <div className="segmented compact">
          <button className={gridOverlayAxis === "off" ? "active" : ""} onClick={() => setGridOverlayAxis("off")}>Off</button>
          <button className={gridOverlayAxis === "z" ? "active" : ""} onClick={() => setGridOverlayAxis("z")}>Z slice</button>
          <button className={gridOverlayAxis === "x" ? "active" : ""} onClick={() => setGridOverlayAxis("x")}>X slice</button>
        </div>
        {gridOverlayAxis !== "off" && <label className="slice-control">
          <span><span>Slice position</span><output>{Math.round(gridOverlaySlice * 100)}%</output></span>
          <input type="range" min={0} max={1} step={0.005} value={gridOverlaySlice} onChange={(event) => setGridOverlaySlice(Number(event.target.value))} aria-label="Grid slice position" />
        </label>}
        <small className="control-hint">The slice remains an independent layer, so it can be combined with either optical renderer.</small>
      </> : <p className="panel-note">Switch to Scientific view to configure debug layers.</p>}
    </section>

    {view === "scientific" && gridOverlayAxis !== "off" && <section className="panel-section grid-key" data-testid="grid-legend">
      <strong>{gridKind === "restricted-tall-cell" ? "TALL-CELL GRID" : gridKind === "quadtree-tall-cell" ? "QUADTREE TALL-CELL GRID" : "UNIFORM GRID"} · {gridOverlayAxis.toUpperCase()} SLICE</strong>
      {tall && <span><i className="sw sw-tall" />tall cell · liquid</span>}
      {tall && <span><i className="sw sw-tall-dry" />tall cell · air</span>}
      <span><i className="sw sw-solid" />rigid body · represented cell</span>
      <span><i className="sw sw-wet" />{tall ? "regular cell · liquid" : "cell · liquid"}</span>
      <span><i className="sw sw-air" />{tall ? "regular cell · air" : "cell · air"}</span>
      {gridKind === "restricted-tall-cell" && <span><i className="sw sw-outside" />above band · not stored</span>}
      {gridKind !== "quadtree-tall-cell" && <span><i className="sw sw-dot" />stored samples (zoom in)</span>}
      {gridKind === "quadtree-tall-cell" && <span>edges follow live adaptive pressure cells</span>}
      <small>Drag the bright top edge in the viewport to sweep the slice.</small>
    </section>}
  </aside>;
}
