"use client";

import { VISUALIZATION_FIELDS } from "../lib/core/visualization-catalog";
import { useUIStore } from "../lib/core/stores/ui-store";
import type { GridOverlayMode } from "../lib/core/webgpu-renderer";

const ADAPTIVE_VELOCITY_MODE: GridOverlayMode = "adaptive-velocity-arrows";
const ADAPTIVE_VELOCITY_VIEW = VISUALIZATION_FIELDS.find(
  (field) => field.mode === ADAPTIVE_VELOCITY_MODE,
);

/**
 * Focused scientific visualization surface.
 *
 * This starts with one view on purpose. New catalog entries should not drift
 * into this panel merely because they exist; each addition needs an explicit
 * product decision and an explicit placement here.
 */
export function VisualsPanel() {
  const overlayMode = useUIStore((state) => state.gridOverlayMode);
  const overlayAxis = useUIStore((state) => state.gridOverlayAxis);
  const overlaySlice = useUIStore((state) => state.gridOverlaySlice);
  const setOverlayMode = useUIStore((state) => state.setGridOverlayMode);
  const setOverlayAxis = useUIStore((state) => state.setGridOverlayAxis);
  const setOverlaySlice = useUIStore((state) => state.setGridOverlaySlice);
  const setRightPanel = useUIStore((state) => state.setRightPanel);
  const active = overlayMode === ADAPTIVE_VELOCITY_MODE && overlayAxis !== "off";

  const toggleVelocity = () => {
    if (active) {
      setOverlayAxis("off");
      return;
    }
    setOverlayMode(ADAPTIVE_VELOCITY_MODE);
    if (overlayAxis === "off") {
      setOverlayAxis("volume");
      setOverlaySlice(0.42);
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
        <div><h3>Losasso adaptive velocity</h3><small>LIVE GPU PUBLICATION · NO READBACK</small></div>
        <span>ADAPTIVE LEAVES</span>
      </header>

      <div className="paper-view-grid visuals-view-grid">
        <button className={active ? "active" : ""} onClick={toggleVelocity}
          aria-pressed={active} type="button">
          <span>{ADAPTIVE_VELOCITY_VIEW?.figure ?? "§5"}</span>
          <strong>{ADAPTIVE_VELOCITY_VIEW?.label ?? "Adaptive velocity arrows"}</strong>
          <small>{ADAPTIVE_VELOCITY_VIEW?.description
            ?? "One cell-centred direction and speed arrow per adaptive leaf."}</small>
        </button>
      </div>

      <div className="paper-view-controls">
        <div>
          <span>VIEW PLANE</span>
          <div role="group" aria-label="Adaptive velocity view plane">
            {(["x", "y", "z"] as const).map((axis) => <button key={axis} type="button"
              className={active && overlayAxis === axis ? "active" : ""}
              onClick={() => { setOverlayMode(ADAPTIVE_VELOCITY_MODE); setOverlayAxis(axis); }}>
              {axis.toUpperCase()}
            </button>)}
            <button type="button" className={active && overlayAxis === "volume" ? "active" : ""}
              onClick={() => { setOverlayMode(ADAPTIVE_VELOCITY_MODE); setOverlayAxis("volume"); }}>
              VOLUME
            </button>
            <button type="button" className={!active ? "active" : ""}
              onClick={() => setOverlayAxis("off")}>HIDE</button>
          </div>
        </div>
        {active && <label>
          <span>{overlayAxis === "volume" ? "VOLUME OPACITY" : `${overlayAxis.toUpperCase()} SLICE`}</span>
          <input type="range" min={overlayAxis === "volume" ? 0.05 : 0} max={1}
            step={overlayAxis === "volume" ? 0.01 : 0.005} value={overlaySlice}
            onChange={(event) => setOverlaySlice(Number(event.currentTarget.value))}
            aria-label={overlayAxis === "volume"
              ? "Adaptive velocity volume opacity" : `Adaptive velocity ${overlayAxis} slice position`} />
          <output>{Math.round(overlaySlice * 100)}%</output>
        </label>}
      </div>

      {ADAPTIVE_VELOCITY_VIEW?.legend && <div className="paper-field-legend"
        aria-label="Adaptive velocity legend">
        {ADAPTIVE_VELOCITY_VIEW.legend.map((entry) => <span key={entry.label}>
          <i style={{ background: entry.swatch }} />{entry.label}
        </span>)}
      </div>}
    </section>
  </aside>;
}
