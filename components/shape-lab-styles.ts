/**
 * The lab's own styling, as one string.
 *
 * Kept beside the component and out of `app/globals.css` because it is the only
 * thing that uses it, and out of a CSS module because a tool that has to be
 * readable in one sitting should not be two files. It reads the app's own custom
 * properties, so the lab follows the theme switch like everything else.
 */
export const SHAPE_LAB_STYLES = `
.sl-root {
  display: grid;
  grid-template-columns: minmax(230px, 260px) minmax(0, 1fr) minmax(280px, 340px);
  gap: 18px;
  padding: 18px;
  height: 100vh;
  box-sizing: border-box;
  color: var(--ink);
  background: var(--bg);
  font: 13px/1.45 var(--font-sans, system-ui, sans-serif);
}
@media (max-width: 1100px) {
  .sl-root { grid-template-columns: 1fr; height: auto; }
}
.sl-side, .sl-params {
  display: flex;
  flex-direction: column;
  gap: 16px;
  min-height: 0;
  overflow-y: auto;
  padding-right: 6px;
}
.sl-title { margin: 0; font-size: 17px; font-weight: 600; letter-spacing: -0.01em; }
.sl-blurb { margin: 0; color: var(--muted); font-size: 12px; }
.sl-blurb code { font-size: 11px; background: var(--well); padding: 1px 4px; border-radius: 4px; }
.sl-field { display: flex; flex-direction: column; gap: 7px; }
.sl-legend {
  font-size: 10.5px; text-transform: uppercase; letter-spacing: 0.09em;
  color: var(--label); font-weight: 600;
}
.sl-note { margin: 0; font-size: 11px; color: var(--muted); }

.sl-list { display: flex; flex-direction: column; gap: 3px; }
.sl-item {
  display: flex; flex-direction: column; gap: 1px; text-align: left;
  padding: 7px 9px; border-radius: var(--radius-s, 9px);
  border: 1px solid transparent; background: var(--well); color: inherit; cursor: pointer;
}
.sl-item:hover { background: var(--hover); }
.sl-item-on { border-color: var(--accent-line); background: var(--accent-wash); }
.sl-item-name { font-weight: 600; }
.sl-item-detail { font-size: 11px; color: var(--muted); }

.sl-segments { display: flex; gap: 4px; flex-wrap: wrap; }
.sl-segment {
  flex: 1 1 auto; min-width: 34px; padding: 5px 8px; cursor: pointer;
  border-radius: var(--radius-s, 9px); border: 1px solid var(--line);
  background: var(--panel); color: inherit; font: inherit; font-size: 12px;
}
.sl-segment:hover { border-color: var(--line-strong); }
.sl-segment-on { background: var(--accent-wash); border-color: var(--accent-line); color: var(--accent-strong); font-weight: 600; }
.sl-check { display: flex; align-items: center; gap: 7px; font-size: 12px; color: var(--muted); }

.sl-metrics { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; margin: 0; }
.sl-metrics div { background: var(--well); border-radius: var(--radius-s, 9px); padding: 6px 8px; }
.sl-metrics dt { font-size: 10px; text-transform: uppercase; letter-spacing: 0.07em; color: var(--label); }
.sl-metrics dd { margin: 0; font-variant-numeric: tabular-nums; font-weight: 600; }
.sl-kinds { margin: 0; font-size: 11px; color: var(--muted); }

.sl-stage { display: flex; flex-direction: column; align-items: center; gap: 8px; min-width: 0; }
.sl-canvas {
  border-radius: var(--radius-m, 15px);
  border: 1px solid var(--line);
  background: #ced2d6;
  touch-action: none;
  cursor: grab;
  max-width: 100%;
}
.sl-canvas:active { cursor: grabbing; }
.sl-overlay {
  display: flex; gap: 12px; flex-wrap: wrap; align-items: center;
  font-size: 11.5px; color: var(--muted); font-variant-numeric: tabular-nums;
}
.sl-busy { color: var(--amber-ink); }
.sl-error { color: var(--alert-ink); }
.sl-hint { margin: 0; font-size: 11px; color: var(--dim); }

.sl-params-head { display: flex; justify-content: space-between; align-items: center; gap: 8px; }
.sl-actions { display: flex; gap: 6px; }
.sl-actions button {
  padding: 4px 9px; font-size: 11px; cursor: pointer;
  border-radius: var(--radius-s, 9px); border: 1px solid var(--line);
  background: var(--panel); color: inherit;
}
.sl-actions button:disabled { opacity: 0.45; cursor: default; }
.sl-export {
  margin: 0; padding: 7px 9px; font-size: 11px; line-height: 1.5;
  color: var(--muted); background: var(--accent-wash);
  border: 1px solid var(--accent-line); border-radius: var(--radius-s, 9px);
}
.sl-export code { font-size: 10.5px; background: var(--well); padding: 1px 3px; border-radius: 4px; }
.sl-params-body { display: flex; flex-direction: column; gap: 10px; }
.sl-node { display: flex; flex-direction: column; gap: 4px; }
.sl-node-title { margin: 0; font-size: 12px; font-weight: 600; color: var(--accent-strong); }

.sl-row {
  display: grid; grid-template-columns: minmax(90px, 1.1fr) minmax(60px, 1.3fr) 74px;
  align-items: center; gap: 7px; padding: 2px 0;
}
.sl-row-label {
  font-size: 11px; color: var(--muted); overflow: hidden;
  text-overflow: ellipsis; white-space: nowrap;
}
.sl-row-changed .sl-row-label { color: var(--accent-strong); font-weight: 600; }
.sl-slider { width: 100%; accent-color: var(--accent); }
.sl-number, .sl-text {
  width: 100%; box-sizing: border-box; padding: 3px 5px; font: inherit; font-size: 11px;
  border-radius: 6px; border: 1px solid var(--line); background: var(--panel); color: inherit;
  font-variant-numeric: tabular-nums;
}
.sl-seed {
  padding: 3px 6px; font-size: 11px; cursor: pointer; border-radius: 6px;
  border: 1px solid var(--line); background: var(--panel); color: inherit;
}
.sl-group { border-left: 1px solid var(--line); padding-left: 8px; margin: 3px 0; }
.sl-group > summary { cursor: pointer; font-size: 11px; color: var(--label); font-weight: 600; }
.sl-group-body { display: flex; flex-direction: column; padding-top: 3px; }
`;
