# Hero-scene UI: the cut list and the contextual-radial architecture

STATUS: LANDED in full on 2026-08-17 (four slices + flyout redesign + overlay
restyle), all in the working tree of this date. Deviations from the plan below:
the overlay host class is `scene-instrument` (`scene-overlay` is the hover-reveal
scene chip — the name collision shipped once and rendered the overlays
invisible); the overlay pane is 420px with a left-edge pipe rail rather than the
560px centred trunk; probes' pin requests now carry an optional aim so a ring
wedge pins the right-clicked cell/pixel. Follow-ups landed 2026-08-18: saved-scene
rename/delete restored on the library route (hover affordances, inline rename,
two-step on-card delete confirm); the empty-space ring slimmed to 9 wedges
(instruments/probes stay on the water/tank rings by construction); the open
overlay persists as the `overlay` URL param (listed, never counted, in the
overrides chip — the old `panel` treatment).

Decided 2026-08-17. The scene is the hero; the contextual radial ring is the
route to everything. No docked chrome, no permanent tab rail, no panel that
resizes the viewport. Displays that must exist render as overlays *on* the
scene.

## The organizing principle

Right-click a thing → its ring offers everything you can do to or learn about
that thing. This already holds for editing (place/carry/water/erase/region);
it now extends to configuration, visualization, pipelines, and diagnostics.
Nothing gets a permanent button when a ring wedge can reach it.

## Permanent chrome (the complete list)

- **Transport mini-cluster**: play/pause, STEP, plus ● REC tucked beside them,
  and a small sim-time readout. Fades when the pointer is idle.
- **FPS meter**: small, same corner behaviour as today's imperative readout.
- **Hover-reveal scene chip** (top-left): invisible until the pointer nears the
  corner; then scene name → library, and the SceneOverridesChip when the URL
  carries overrides.
- Transient by nature, unchanged: GPU bring-up cards/pills, viewport failure
  alert, manual-start / unavailable fallbacks.

Everything else appears only through selection or the ring.

## Ring extensions (new wedges)

| Ring | New wedges |
|---|---|
| Tank / water | **Configure** (container + fluid flyout sections), **Fields** (opens FluidFieldFlyout's field views), **Pipeline** (advance-pipeline overlay), **Diagnostics** (metric-card overlay) |
| Fluid cell / voxel (right-click on the liquid or a solid voxel) | **Inspect cell** (FluidCellTraceHud, pinned to that cell), **Trace ray** (PixelTraceHud for that pixel — gains its first viewport entry point) |
| Rigid body | existing Carry/Drop/Edit/Delete; **Edit** flyout absorbs the BODIES sliders (density, size, restitution, friction) |
| Empty space | existing placement wedges; add **Library**, **Render pipeline** (frame-graph overlay), **Record** |

Keyboard shortcuts survive unchanged (`q/d/w/p/t/y/b/u/g`, `C`, `F`,
`0/1/2/3`, undo/redo, axis locks).

## Survives as contextual surfaces (mostly as-is)

- SelectionFlyout, TreeCanopyFlyout, StoneLookFlyout, VesselRimFlyout
- FluidFieldFlyout + PressureFilmStrip (pressure journal stays; it is already
  contextual on the tank)
- SceneScaleOverlay (on tank/fluid selection), SceneOverridesChip
- EditorModeChip **only in armed/carrying states** — the resting hint line and
  permanent UNDO/REDO buttons go (keyboard-only; ring teaches discovery)
- AxisWidget: dropped from the default view (was decorative; framing is
  keyboard-only anyway)
- RadialMenu, gizmos, hover chips, axis-lock readout, drafts/carry machinery
- RecordingPlaybackModal

## Rehosted as scene overlays (not docked panels)

- **FluidPipeline** (advance pipeline, per-stage GPU timings, stage gates,
  method param controls — the SIM tab's content) → translucent overlay over
  the scene, opened from the tank ring. Timeline instrumentation auto-enables
  while open, as the panel does today.
- **RenderPipeline** (frame graph, stage lamps/ablation, node tuning clusters,
  stage-view taps, quality PROFILE strip + resolution scale) → same treatment,
  opened from the empty-space ring.
- **Diagnostics cards** → compact overlay from the tank ring; the per-body
  block joins the body's Edit flyout.
- The two pipeline renderers are ~70% duplicate code — merge into one
  graph-overlay component while rehosting.

## Deleted outright

Components (≈3,100 lines) plus their store state, styles, and shortcuts:

- `utility-panel-tabs` rail, `RightPanelResizer`, `rightPanel`/
  `rightPanelWidth`/`diagnosticsOpen` and the `"diagnostics"` union wart
- `VisualPanel.tsx` (684) — RENDER tab shell; PROFILE + resolution scale move
  into the render-pipeline overlay; PROBES strip dies (ring replaces it)
- `VisualsPanel.tsx` (110) — fully duplicated by FluidFieldFlyout
- `FluidPipelinePanel.tsx` (243) — shell only; graph is rehosted
- `PerformancePanel.tsx` (362) + `PerformanceActivityGrid.tsx` (1021) +
  `PerformanceDials.tsx` (124) — capture modes, activity matrix, dials;
  the "Paper field observatory" duplicate dies with it
- `DiagnosticsPanel.tsx` (107) — content rehosted as overlay cards
- `RigidBodyTray.tsx` (129) — palette lives in the ring; sliders move to
  SelectionFlyout
- `SceneConfigPopover.tsx` (280) + `MethodPanel.tsx` (99) — Container/Fluid →
  tank Configure flyout; Method → FluidFieldFlyout's SOLVER row (already
  there); Numerics → pipeline-overlay params; "My scenes" → library route
- Top-right `Pick cell` button cluster (FPS meter survives alone)
- Dead code: `PipeRow` in PipeControls, de-export
  `buildPerformanceActivityView`
- Full TransportBar (144) replaced by the mini-cluster; the middle column
  (lockstep label, ACTUAL ×, GPU-lag chip, step-size slider) is cut — step
  size moves to the pipeline overlay

ui-store trim: right-panel state; the trace HUDs keep their state but their
only entries are the ring wedges and existing shortcuts.

## Explicitly kept decisions

- Probes (cell trace, pixel trace) are **kept**, contextual via the ring on a
  voxel/fluid cell — not deleted, not behind a URL flag.
- Frame pipelines are **kept** as overlays — the perf workflow (pressure film,
  per-stage timings, ablation) survives.
- Recording is **kept**, tucked into the transport cluster.
- Both HUDs currently share one absolute CSS box (`pixel-trace-hud`); fix the
  collision when they become ring-reachable.
