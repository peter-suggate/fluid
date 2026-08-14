"use client";

import { getMethod } from "../lib/core/method-registry";
import { VISUALIZATION_FIELDS } from "../lib/core/visualization-catalog";
import { useMethodStore } from "../lib/core/stores/method-store";
import { useUIStore } from "../lib/core/stores/ui-store";
import type { GridOverlayMode } from "../lib/core/webgpu-renderer";

/**
 * Focused scientific visualization surface.
 *
 * This starts with one view on purpose. New catalog entries should not drift
 * into this panel merely because they exist; each addition needs an explicit
 * product decision and an explicit placement here.
 */
export function VisualsPanel() {
  const methodId = useMethodStore((state) => state.methodId);
  const method = getMethod(methodId);
  const overlayMode = useUIStore((state) => state.gridOverlayMode);
  const overlayAxis = useUIStore((state) => state.gridOverlayAxis);
  const overlaySlice = useUIStore((state) => state.gridOverlaySlice);
  const setOverlayMode = useUIStore((state) => state.setGridOverlayMode);
  const setOverlayAxis = useUIStore((state) => state.setGridOverlayAxis);
  const setOverlaySlice = useUIStore((state) => state.setGridOverlaySlice);
  const setRightPanel = useUIStore((state) => state.setRightPanel);
  const supported = new Set(method.supportedFieldModes ?? []);
  const views = VISUALIZATION_FIELDS.filter((field) =>
    !field.hidden && supported.has(field.mode));
  const activeView = overlayAxis === "off"
    ? undefined : views.find((field) => field.mode === overlayMode);

  const selectView = (mode: GridOverlayMode) => {
    if (overlayMode === mode && overlayAxis !== "off") {
      setOverlayAxis("off");
      return;
    }
    setOverlayMode(mode);
    if (overlayAxis === "off") {
      const view = views.find((candidate) => candidate.mode === mode);
      const axis = view?.axis ?? "z";
      setOverlayAxis(axis);
      if (axis === "volume") setOverlaySlice(0.42);
    }
  };

  return <aside id="visuals-panel"
    className="right-panel panel-scroll performance-panel performance-v2 visuals-panel"
    aria-label="Scientific visualizations" data-testid="visuals-panel">
    <header className="performance-panel-header">
      <div><span>VISUALS</span><h2>Fluid field visualization</h2></div>
      <div className="performance-panel-header-actions">
        <button className="panel-close" type="button" onClick={() => setRightPanel(null)}
          aria-label="Close visuals panel">×</button>
      </div>
    </header>

    <section className="paper-observatory visuals-only-observatory">
      <header>
        <div><h3>{method.label} fields</h3><small>LIVE METHOD PUBLICATION · NO READBACK</small></div>
        <span>{views.length} AVAILABLE</span>
      </header>

      <div className="paper-view-grid visuals-view-grid">
        {views.map((view) => <button key={view.id}
          className={activeView?.id === view.id ? "active" : ""}
          onClick={() => selectView(view.mode as GridOverlayMode)}
          aria-pressed={activeView?.id === view.id} type="button">
          <span>{view.figure ?? "FIELD"}</span>
          <strong>{view.label}</strong>
          <small>{view.description}</small>
        </button>)}
      </div>

      <div className="paper-view-controls">
        <div>
          <span>VIEW PLANE</span>
          <div role="group" aria-label="Fluid field view plane">
            {(["x", "y", "z"] as const).map((axis) => <button key={axis} type="button"
              className={activeView && overlayAxis === axis ? "active" : ""}
              onClick={() => { if (activeView) setOverlayMode(activeView.mode as GridOverlayMode); setOverlayAxis(axis); }}>
              {axis.toUpperCase()}
            </button>)}
            <button type="button" className={activeView && overlayAxis === "volume" ? "active" : ""}
              onClick={() => { if (activeView) setOverlayMode(activeView.mode as GridOverlayMode); setOverlayAxis("volume"); }}>
              VOLUME
            </button>
            <button type="button" className={!activeView ? "active" : ""}
              onClick={() => setOverlayAxis("off")}>HIDE</button>
          </div>
        </div>
        {activeView && <label>
          <span>{overlayAxis === "volume" ? "VOLUME OPACITY" : `${overlayAxis.toUpperCase()} SLICE`}</span>
          <input type="range" min={overlayAxis === "volume" ? 0.05 : 0} max={1}
            step={overlayAxis === "volume" ? 0.01 : 0.005} value={overlaySlice}
            onChange={(event) => setOverlaySlice(Number(event.currentTarget.value))}
            aria-label={overlayAxis === "volume"
              ? "Adaptive velocity volume opacity" : `Adaptive velocity ${overlayAxis} slice position`} />
          <output>{Math.round(overlaySlice * 100)}%</output>
        </label>}
      </div>

      {activeView?.legend && <div className="paper-field-legend"
        aria-label={`${activeView.label} legend`}>
        {activeView.legend.map((entry) => <span key={entry.label}>
          <i style={{ background: entry.swatch }} />{entry.label}
        </span>)}
      </div>}
    </section>
  </aside>;
}
