/** Layout only: colours, typography and controls follow the app's shared theme. */
export const SHAPE_LAB_STYLES = `
.sl-root { height: 100dvh; display: flex; flex-direction: column; background: var(--bg); color: var(--ink); font-family: var(--sans); --control-height: 28px; --control-font-size: var(--size-3); --control-inline-padding: 10px; }
.sl-root a { text-decoration: none; }
.sl-root .library-brand { color: var(--ink); }
.sl-bar { min-height: 72px; padding: 0 24px; flex-shrink: 0; }
.sl-bar-divider { height: 22px; width: 1px; background: var(--line-strong); }
.sl-bar-title { font: 600 14px var(--display); }
.sl-bar-title > span { color: var(--muted); font-weight: 400; }
.sl-back { margin-left: auto; height: 34px; padding: 0 14px; }
.sl-workspace { display: grid; grid-template-columns: 270px minmax(0, 1fr) 290px; flex: 1; min-height: 0; }
.sl-library { display: flex; flex-direction: column; min-height: 0; padding: 25px 16px 14px; border-right: 1px solid var(--line); }
.sl-library-heading, .sl-section-heading { display: flex; justify-content: space-between; align-items: center; gap: 12px; }
.sl-eyebrow { color: var(--muted); font: 600 10px var(--sans); text-transform: uppercase; letter-spacing: .12em; }
.sl-root h1, .sl-root h2, .sl-root h3 { color: var(--ink); font-family: var(--display); }
.sl-library h1 { margin: 5px 0 0; font-size: 23px; font-weight: 600; letter-spacing: -.03em; }
.sl-description { font-size: 12px; line-height: 1.6; color: var(--muted); margin: 9px 0 18px; }
.sl-search { width: 100%; height: 36px; margin: 0 0 14px; padding: 0 12px; box-sizing: border-box; }
.sl-search input { font-size: 12px; }
.sl-library .ui-fields { --ui-name: 30%; }
.sl-results { margin: 17px 4px 8px; color: var(--dim); font-size: 11px; }
.sl-asset-list { position: relative; flex: 1; min-height: 0; overflow-y: auto; overscroll-behavior: contain; padding-right: 4px; }
.sl-list-heading { display: flex; justify-content: space-between; margin: 13px 7px 7px; font: 500 11px var(--sans) !important; color: var(--muted) !important; }
.sl-list-heading span { font-variant-numeric: tabular-nums; color: var(--dim); }
.sl-asset.ui-button { width: 100%; height: auto; min-height: 58px; display: flex; justify-content: flex-start; gap: 10px; padding: 9px; margin: 3px 0; text-align: left; white-space: normal; border: 1px solid transparent; background: transparent; border-radius: var(--radius-s); box-shadow: none; }
.sl-asset.ui-button:hover { background: var(--hover); }
.sl-asset.ui-button.is-selected { border-color: var(--accent-line); background: var(--accent-wash); }
.sl-asset-icon { display: grid; place-items: center; width: 32px; height: 34px; flex-shrink: 0; color: var(--muted); background: var(--well); border-radius: 7px; }
.sl-asset.is-selected .sl-asset-icon { color: var(--accent); }
.sl-asset-text { display: flex; flex-direction: column; gap: 4px; min-width: 0; }
.sl-asset-text strong { font-size: 12px; font-weight: 500; }
.sl-asset-text small { color: var(--muted); font-size: 10px; font-weight: 400; }
.sl-selected-dot { margin-left: auto; flex-shrink: 0; width: 5px; height: 5px; border-radius: 50%; }
.is-selected .sl-selected-dot { background: var(--accent); }
.sl-library-foot { border-top: 1px solid var(--line); padding-top: 13px; margin: 12px 0 0; color: var(--dim); font-size: 10px; }
.sl-empty { display: flex; align-items: center; flex-direction: column; gap: 12px; padding: 32px 10px; text-align: center; color: var(--muted); font-size: 12px; }
.sl-empty p { margin: 0; }
.sl-viewer { min-width: 0; min-height: 0; padding: 27px 24px 16px; display: flex; flex-direction: column; overflow-y: auto; }
.sl-viewer-heading { display: flex; align-items: center; justify-content: space-between; gap: 12px; margin-bottom: 24px; }
.sl-viewer h2 { font-size: clamp(24px, 2.2vw, 34px); letter-spacing: -.035em; font-weight: 500; margin: 6px 0 0; }
.sl-view-options { display: flex; justify-content: space-between; gap: 12px; flex-wrap: wrap; margin-bottom: 14px; }
.sl-voxel-options { max-width: 300px; margin-bottom: 14px; }
.sl-preview { flex: 1 0 auto; min-height: 0; display: flex; flex-direction: column; }
.sl-preview-stage { flex: 1; min-height: 280px; position: relative; display: grid; place-items: center; overflow: hidden; background: rgb(206 210 214); border: 1px solid var(--line-strong); border-radius: var(--radius-m); }
.sl-preview-canvas { position: absolute; inset: 0; display: block; width: 100%; height: 100%; object-fit: contain; touch-action: none; cursor: grab; background: rgb(206 210 214); }
.sl-preview-canvas:active { cursor: grabbing; }
.sl-preview-canvas:focus-visible { outline: 2px solid var(--accent); outline-offset: -4px; }
.sl-preview-status { display: flex; justify-content: space-between; gap: 12px; min-height: 18px; margin: 10px 0; color: var(--muted); font-size: 11px; font-variant-numeric: tabular-nums; }
.sl-preview-error { color: var(--red); display: flex; align-items: center; flex-wrap: wrap; gap: 8px; overflow-wrap: anywhere; }
.sl-preview-toolbar { display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 10px; }
.sl-preview-hint { color: var(--dim); font-size: 10px; line-height: 1.6; margin: 12px 0 0; }
.sl-preview-caption { margin: 12px 0 0; padding-top: 12px; border-top: 1px solid var(--line); color: var(--muted); font-size: 11px; line-height: 1.6; }
.sl-unbaked-preview { flex: 1 0 auto; min-height: 300px; display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 16px; padding: 32px 24px; border: 1px solid var(--line); border-radius: var(--radius-m); background: var(--well); text-align: center; color: var(--muted); }
.sl-unbaked-preview h3 { margin: 0; font-size: 20px; font-weight: 500; }
.sl-unbaked-preview p { max-width: 360px; margin: 0; font-size: 13px; line-height: 1.7; }
.sl-inspector { border-left: 1px solid var(--line); background: var(--panel); padding: 25px 19px; overflow-y: auto; min-height: 0; }
.sl-detail-section + .sl-detail-section { border-top: 1px solid var(--line); padding-top: 19px; margin-top: 19px; }
.sl-inspector h2 { margin: 7px 0 0; font-size: 19px; font-weight: 500; }
.sl-inspector h3 { margin: 0 0 13px; font-size: 13px; font-weight: 600; }
.sl-section-heading { margin-bottom: 13px; }
.sl-section-heading h3 { margin: 0; }
.sl-inspector .ui-facts { --ui-name: 32%; }
.sl-inspector code { font-size: 10px; overflow-wrap: anywhere; }
.sl-source-files { list-style: none; display: flex; flex-direction: column; gap: 9px; padding: 0; margin: 0; }
.sl-source-files li { border: 1px solid var(--line); border-radius: var(--radius-s); padding: 9px 10px; background: var(--well); overflow-wrap: anywhere; }
.sl-inspector .ui-fields { --ui-name: 48%; }
.sl-small { margin: 10px 0; color: var(--dim); font-size: 10px; line-height: 1.65; }
.sl-materials { display: flex; flex-direction: column; gap: 9px; max-height: 172px; overflow-y: auto; }
.sl-material { display: flex; align-items: center; gap: 9px; font-size: 11px; color: var(--muted); }
.sl-material .ui-swatch { flex-shrink: 0; }
.sl-file { display: flex; align-items: center; gap: 10px; padding: 12px 10px; border: 1px solid var(--line); border-radius: var(--radius-s); color: var(--ink); margin: 8px 0; background: var(--well); }
.sl-file:hover { border-color: var(--accent-line); background: var(--accent-wash); }
.sl-file > span { display: flex; flex: 1; flex-direction: column; gap: 4px; }
.sl-file strong { font-size: 11px; font-weight: 500; }
.sl-file small { font-size: 10px; color: var(--muted); }
.sl-file > svg { color: var(--muted); flex-shrink: 0; }
@media (max-width: 1200px) {
  .sl-workspace { grid-template-columns: 230px minmax(0, 1fr) 250px; }
  .sl-viewer { padding: 24px 16px 16px; }
  .sl-viewer-heading { flex-wrap: wrap; }
  .sl-bar-title > span { display: none; }
}
@media (max-width: 980px) {
  .sl-root { overflow-y: auto; }
  .sl-workspace { grid-template-columns: 230px minmax(0, 1fr); flex: none; min-height: auto; }
  .sl-library { grid-row: span 2; max-height: calc(100dvh - 72px); position: sticky; top: 0; }
  .sl-viewer { min-height: 660px; }
  .sl-inspector { border-left: 0; border-top: 1px solid var(--line); display: grid; grid-template-columns: 1fr 1fr; gap: 24px; overflow: visible; }
  .sl-detail-section + .sl-detail-section { margin: 0; padding: 0; border: 0; }
}
@media (max-width: 640px) {
  .sl-bar { padding: 0 14px; gap: 10px; min-height: 62px; flex-wrap: wrap; }
  .sl-bar .library-wordmark, .sl-bar-divider { display: none; }
  .sl-bar-title { font-size: 12px; }
  .sl-back { padding: 0 10px; font-size: 11px; }
  .sl-workspace { display: flex; flex-direction: column; }
  .sl-library { position: static; max-height: none; padding: 20px 16px 12px; border-right: 0; border-bottom: 1px solid var(--line); }
  .sl-library-heading h1 { font-size: 22px; }
  .sl-asset-list { max-height: 210px; flex: auto; }
  .sl-library-foot { display: none; }
  .sl-viewer { min-height: 630px; }
  .sl-inspector { grid-template-columns: 1fr; }
  .sl-detail-section + .sl-detail-section { border-top: 1px solid var(--line); padding-top: 20px; }
}
`;
