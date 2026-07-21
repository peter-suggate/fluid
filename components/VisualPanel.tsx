"use client";

import { getMethod } from "@/lib/methods";
import { finePublicationGateDiagnostics } from "@/lib/octree-fine-publication-diagnostics";
import { isOctreeTechniqueOverlayMode } from "@/lib/octree-technique-debug";
import { PAPER_VISUAL_PRESETS, paperPipelineStages, paperSection5SpatialFailures, paperVisualAuthority } from "@/lib/paper-pipeline-diagnostics";
import { useDiagnosticsStore } from "@/lib/stores/diagnostics-store";
import { useMethodStore } from "@/lib/stores/method-store";
import { useUIStore } from "@/lib/stores/ui-store";
import { SVO_COST_OVERLAY_LABELS, SVO_COST_OVERLAY_MODES } from "@/lib/svo-render-diagnostics";
import { RangeControl } from "./controls";

export function VisualPanel() {
  const methodId = useMethodStore((state) => state.methodId);
  const gpuInfo = useDiagnosticsStore((state) => state.gpuInfo);
  const effectiveRendererStatus = useDiagnosticsStore((state) => state.effectiveRendererStatus);
  const waterSurfacePresentation = useDiagnosticsStore((state) => state.waterSurfacePresentation);
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
  const gridOverlayAxis = useUIStore((state) => state.gridOverlayAxis);
  const setGridOverlayAxis = useUIStore((state) => state.setGridOverlayAxis);
  const gridOverlaySlice = useUIStore((state) => state.gridOverlaySlice);
  const setGridOverlaySlice = useUIStore((state) => state.setGridOverlaySlice);
  const gridOverlayMode = useUIStore((state) => state.gridOverlayMode);
  const setGridOverlayMode = useUIStore((state) => state.setGridOverlayMode);
  const setRightPanel = useUIStore((state) => state.setRightPanel);
  const gridKind = getMethod(methodId).backend === "cpu" ? "uniform" : gpuInfo?.gridKind ?? "uniform";
  const tall = gridKind === "restricted-tall-cell";
  const adaptive = gridKind === "quadtree-tall-cell" || gridKind === "octree";
  const quadtreeTall = gridKind === "quadtree-tall-cell";
  const octree = gridKind === "octree";
  const overlayActive = gridOverlayAxis !== "off";
  const volumeOverlay = gridOverlayAxis === "volume";
  const volumeOpacity = Math.max(0.05, gridOverlaySlice);
  const sliceOverlay = overlayActive && !volumeOverlay;
  const paperTechniqueMode = isOctreeTechniqueOverlayMode(gridOverlayMode);
  const paperVolumeCapable = paperTechniqueMode && gridOverlayMode !== "global-fine-phi";
  const paperStages = paperPipelineStages(gpuInfo, waterSurfacePresentation);
  const activePaperVisual = paperVisualAuthority(gridOverlayMode, gridOverlayAxis, paperStages, gpuInfo);
  const section5SpatialFailures = paperSection5SpatialFailures(gpuInfo);
  const encodedSteps = gpuInfo?.encodedSteps ?? 0;
  const t0SceneReady = gpuInfo?.initialSparseAuthorityReady === true
    && gpuInfo?.initialRasterSurfaceReady === true;
  const volumeSource = ({
    "global-fine": "global-fine",
    "adaptive-pages": "adaptive pages",
    "dense-volume": "dense volume",
    "initial-condition": "analytic t=0",
    unavailable: "unavailable",
  } as const)[gpuInfo?.volumeTelemetrySource ?? "unavailable"];
  const physicalVolumeDrift = gpuInfo?.volumeDrift;
  const representedVolumeDrift = gpuInfo?.representedVolumeDrift;
  const globalFineVolumeEstimate = gpuInfo?.volumeTelemetrySource === "global-fine";
  const representedVolumeAliasesPrimary = physicalVolumeDrift !== undefined
    && representedVolumeDrift !== undefined
    && physicalVolumeDrift === representedVolumeDrift;
  const driftLabel = (value: number | undefined) => value === undefined
    ? "—"
    : `${value >= 0 ? "+" : ""}${(value * 100).toFixed(2)}%`;
  const finePublicationGates = finePublicationGateDiagnostics({
    generation: gpuInfo?.globalFineGeneration,
    topologyFlags: gpuInfo?.globalFineTopologyFlags,
    downstreamReason: gpuInfo?.globalFineDownstreamFinalizeReason,
    published: gpuInfo?.globalFinePublished,
    rolledBack: gpuInfo?.globalFineRolledBack,
    redistanceCommitted: gpuInfo?.globalFineRedistanceCommitted,
    redistanceUnresolved: gpuInfo?.globalFineRedistanceUnresolvedCells,
    volumeFlags: gpuInfo?.globalFineVolumeFlags,
    transportCommitted: gpuInfo?.globalFineTransportCommitted,
    transportOutside: gpuInfo?.globalFineTransportDepartureOutsideBand,
    transportUnavailable: gpuInfo?.globalFineTransportVelocityUnavailable,
    transportFaceBandUnavailable: gpuInfo?.globalFineTransportFaceBandUnavailable,
    faceBandFlags: gpuInfo?.globalFineFaceBandFlags,
    faceBandTransitionFlags: gpuInfo?.globalFineFaceBandTransitionFlags,
    faceBandPowerFlags: gpuInfo?.globalFineFaceBandPowerPublicationFlags,
    faceBandTransientFlags: gpuInfo?.globalFineFaceBandTransientPowerFlags,
    faceBandPointFlags: gpuInfo?.globalFineFaceBandPointFieldFlags,
  });
  const selectOverlayMode = (mode: Parameters<typeof setGridOverlayMode>[0]) => {
    // Full-volume drawing is intentionally limited to the compact paper
    // structures. Legacy texture fields remain well-defined slice views.
    if (volumeOverlay && !isOctreeTechniqueOverlayMode(mode)) setGridOverlayAxis("z");
    setGridOverlayMode(mode);
  };
  const motionAdaptiveOptical = gpuInfo?.quadtreeOpticalLayerMode === "adaptive-motion";
  const rendererFallbackLabels = {
    "missing-source": "waiting for structural SVO data",
    "unsupported-terrain": "terrain source could not be represented",
    "unsupported-glass-cutout": "authored glazing needs an opaque shell cutout",
    "missing-pbr-materials": "production PBR material table is unavailable",
    "missing-primitive-candidates": "static primitive candidate index is unavailable",
    "missing-lighting-publications": "production light/environment publications are unavailable",
    "pipeline-compile-failure": "SVO pipeline failed to compile",
    "inspection-mode": "a sparse inspection view is active",
  } as const;
  const applyPaperPreset = (preset: typeof PAPER_VISUAL_PRESETS[number]) => {
    if (preset.mode) setGridOverlayMode(preset.mode);
    if (preset.axis === "volume") setGridOverlaySlice(Math.max(0.28, volumeOpacity));
    setGridOverlayAxis(preset.axis);
    if (preset.renderer) setSvoRenderMode(preset.renderer);
    if (preset.voxels) setVoxelRenderMode(preset.voxels);
  };
  const inspectSection5Failure = (mode: Parameters<typeof setGridOverlayMode>[0]) => {
    setGridOverlayMode(mode);
    setGridOverlayAxis(isOctreeTechniqueOverlayMode(mode) ? "volume" : "z");
  };

  return <aside className="right-panel panel-scroll visual-panel" data-testid="visual-panel">
    <section className="panel-section utility-panel-head">
      <div><p className="eyebrow">VIEWPORT</p><strong>Render &amp; debug</strong></div>
      <button className="panel-close" onClick={() => setRightPanel(null)} aria-label="Close render panel">×</button>
    </section>

    <section className="panel-section utility-controls" data-testid="svo-cost-controls">
      <div className="section-heading"><h2>SVO traversal cost</h2><span>SCENE HEATMAP</span></div>
      <div className="svo-overlay-grid" role="group" aria-label="SVO cost overlay">
        {SVO_COST_OVERLAY_MODES.map((mode) => <button
          key={mode}
          className={svoCostOverlay === mode ? "active" : ""}
          disabled={svoRenderMode !== "svo"}
          onClick={() => setSvoCostOverlay(mode)}
        >{SVO_COST_OVERLAY_LABELS[mode]}</button>)}
      </div>
      {svoCostOverlay !== "off" && <>
        <RangeControl label="Maximum traversal depth" unit="levels" value={svoMaximumTraversalDepth} min={1} max={21} step={1} displayDigits={0} onChange={setSvoMaximumTraversalDepth} hint="Diagnostic hierarchy ceiling. Rays requiring deeper nodes fail closed and appear in the budget-failure view." />
        <RangeControl label="Maximum node visits" unit="nodes" value={svoMaximumNodeVisits} min={1} max={256} step={1} displayDigits={0} onChange={setSvoMaximumNodeVisits} hint="Per-call SVO node budget. Lower values expose expensive rays by exhausting them earlier." />
        <RangeControl label="Overlay opacity" unit="%" value={svoOverlayOpacity * 100} min={10} max={100} step={1} displayDigits={0} onChange={(value) => setSvoOverlayOpacity(value / 100)} />
      </>}
      <small className="control-hint">Heatmaps use shader counters from the rendered ray: topology nodes, tested and skipped payload bricks, signed-distance samples, analytic candidates, wide-mip cone steps, and exact shadow traversal. Depth and node limits change the diagnostic traversal budget.</small>
    </section>

    <section className="panel-section utility-controls">
      <div className="section-heading"><h2>Renderer</h2><span>SCENE GEOMETRY</span></div>
      <div className="segmented compact" role="group" aria-label="Scene renderer">
        <button className={svoRenderMode === "raster" ? "active" : ""} onClick={() => setSvoRenderMode("raster")}>Raster</button>
        <button className={svoRenderMode === "svo" ? "active" : ""} onClick={() => setSvoRenderMode("svo")}>Sparse voxels</button>
      </div>
      <small className="control-hint" data-testid="effective-renderer-status">
        Active: {effectiveRendererStatus.effectiveMode === "svo" ? "Sparse voxels" : "Raster"}
        {effectiveRendererStatus.fallbackReason ? ` fallback · ${rendererFallbackLabels[effectiveRendererStatus.fallbackReason]}` : ""}
      </small>
      <small className="control-hint">Sparse voxels is the WebGPU default and consumes the octree directly. Raster remains available for explicit fallback and A/B comparisons.</small>
      <div className="section-heading"><h2>Lighting</h2><span>SVO QUALITY</span></div>
      <div className="segmented compact" role="group" aria-label="SVO lighting quality">
        <button disabled={svoRenderMode !== "svo"} className={svoLightingMode === "direct" ? "active" : ""} onClick={() => setSvoLightingMode("direct")}>Direct</button>
        <button disabled={svoRenderMode !== "svo"} className={svoLightingMode === "cone" ? "active" : ""} onClick={() => setSvoLightingMode("cone")}>Beautiful</button>
      </div>
      <div className="segmented compact" role="group" aria-label="SVO lighting effects">
        <button disabled={svoRenderMode !== "svo"} aria-pressed={svoShadowsEnabled} className={svoShadowsEnabled ? "active" : ""} onClick={() => setSvoShadowsEnabled(!svoShadowsEnabled)}>Shadows</button>
        <button disabled={svoRenderMode !== "svo"} aria-pressed={svoAmbientOcclusionEnabled} className={svoAmbientOcclusionEnabled ? "active" : ""} onClick={() => setSvoAmbientOcclusionEnabled(!svoAmbientOcclusionEnabled)}>Ambient occlusion</button>
      </div>
      <small className="control-hint">Beautiful uses the wide-fanout mip hierarchy for soft visibility. Shadows and ambient occlusion are independent; stale mip pages fall back to exact SVO visibility.</small>
    </section>

    <section className="panel-section utility-controls">
      <div className="section-heading"><h2>Scene representation</h2><span>HYBRID OCTREE</span></div>
      <div className="segmented compact">
        <button className={voxelRenderMode === "smooth" ? "active" : ""} onClick={() => setVoxelRenderMode("smooth")}>Hybrid</button>
        <button className={voxelRenderMode === "raw-voxels" ? "active" : ""} onClick={() => setVoxelRenderMode("raw-voxels")}>Raw voxels</button>
        <button className={voxelRenderMode === "surface-voxels" ? "active" : ""} onClick={() => setVoxelRenderMode("surface-voxels")}>Finest surface</button>
        <button className={voxelRenderMode === "brick-grid" ? "active" : ""} onClick={() => setVoxelRenderMode("brick-grid")}>Brick grid</button>
      </div>
      <small className="control-hint">Hybrid renders the finished scene. Raw preserves adaptive cell sizes; Finest surface subdivides their visible shells to the scene&apos;s finest cell size; Brick grid outlines every live compact leaf. Inspection stays GPU-only with indirect draws.</small>
    </section>

    {octree && <section className="panel-section utility-controls paper-pipeline-inspector" data-testid="paper-pipeline-inspector">
      <div className="section-heading"><h2>Paper pipeline inspector</h2><span>LIVE GPU AUTHORITY</span></div>
      <div className="paper-authority-summary" data-testid="paper-authority-summary">
        <span className={t0SceneReady ? "summary-current" : "summary-waiting"}>t=0 scene {t0SceneReady ? "ready" : "waiting"}</span>
        <span>{encodedSteps === 0 ? "pre-step state" : `encoded substep ${encodedSteps}`}</span>
        <span>power gen {gpuInfo?.powerDiagramGeneration ?? "—"}</span>
        <span>fine gen {gpuInfo?.globalFineGeneration ?? "—"}</span>
        <span>raster {waterSurfacePresentation?.surfaceGeometrySource ?? "pending"} · attached {waterSurfacePresentation?.globalFineAttachedGeneration ?? "—"} / mesh {waterSurfacePresentation?.meshPublicationGeneration ?? "—"}</span>
        <span className={physicalVolumeDrift === undefined ? "summary-waiting" : Math.abs(physicalVolumeDrift) < 0.01 ? "summary-current" : "summary-warning"}>{globalFineVolumeEstimate ? "pre-correction occupancy" : "volume"} {driftLabel(physicalVolumeDrift)} · {volumeSource}</span>
        {!representedVolumeAliasesPrimary && <span className={representedVolumeDrift === undefined ? "summary-waiting" : Math.abs(representedVolumeDrift) < 0.05 ? "summary-current" : "summary-warning"}>represented {driftLabel(representedVolumeDrift)}</span>}
      </div>
      {globalFineVolumeEstimate && <small className="control-hint" data-testid="global-fine-volume-semantics">Global-fine drift is a smoothed-occupancy estimate measured before the bounded φ shift. The global φ shift is an engineering supplement, not part of the paper&apos;s Section 5 algorithm.</small>}
      <div className={`paper-active-visual stage-${activePaperVisual.tone}`} data-testid="paper-active-visual">
        <span>ACTIVE VIEW</span>
        <div><strong>{activePaperVisual.label}</strong><small>{activePaperVisual.frame} · {activePaperVisual.detail}</small></div>
        <span className="paper-stage-state">{activePaperVisual.state}{activePaperVisual.generation ? <small>{activePaperVisual.generation}</small> : null}</span>
      </div>
      <div className="paper-stage-list" role="list" aria-label="Paper pipeline stage health">
        {paperStages.map((stage) => <div className={`paper-stage stage-${stage.tone}`} role="listitem" key={stage.id} data-stage={stage.id}>
          <span className="paper-stage-section">{stage.section}</span>
          <div><strong>{stage.label}</strong><small>{stage.detail}</small></div>
          <span className="paper-stage-state">{stage.state}{stage.generation ? <small>{stage.generation}</small> : null}</span>
        </div>)}
      </div>
      <div className="section-heading technique-heading"><h2>Visual presets</h2><span>ONE CLICK · GPU SOURCES</span></div>
      <div className="paper-preset-grid" role="group" aria-label="Paper pipeline visual presets">
        {PAPER_VISUAL_PRESETS.map((preset) => <button key={preset.id} className={(preset.mode ? preset.mode === gridOverlayMode && preset.axis === gridOverlayAxis : preset.axis === gridOverlayAxis && svoRenderMode === (preset.renderer ?? svoRenderMode)) ? "active" : ""} onClick={() => applyPaperPreset(preset)} title={preset.description} data-testid={`paper-preset-${preset.id}`}>
          <strong>{preset.label}</strong><small>{preset.description}</small>
        </button>)}
      </div>
      <div className="section-heading technique-heading"><h2>Section 5 spatial audit</h2><span>BOUNDED CONTROL + LIVE OVERLAYS</span></div>
      <div className="paper-spatial-audit" role="list" aria-label="Section 5 spatial failure audit">
        {section5SpatialFailures.map((item) => <div className={`paper-spatial-row spatial-${item.state.toLowerCase()}`} role="listitem" key={item.id} data-spatial-stage={item.id}>
          <div><strong>{item.label}</strong><small>{item.counts}{item.first ? ` · ${item.first}` : ""}</small></div>
          <span>{item.state}</span>
          <button onClick={() => inspectSection5Failure(item.inspectMode)} disabled={item.state === "WAITING"}>Inspect</button>
        </div>)}
      </div>
      <small className="control-hint">Green is current GPU authority. Amber is provisional. Purple STALE and red REJECTED products are never presented as current. These views reuse exact solver publications and add no dense readback.</small>
    </section>}

    <section className="panel-section utility-controls">
      <div className="section-heading"><h2>Solver grid</h2><span>DEBUG LAYER</span></div>
      <>
        <div className="segmented compact" role="group" aria-label="Diagnostic geometry">
          <button className={gridOverlayAxis === "off" ? "active" : ""} onClick={() => setGridOverlayAxis("off")}>Off</button>
          <button className={sliceOverlay ? "active" : ""} onClick={() => setGridOverlayAxis(volumeOverlay || gridOverlayAxis === "off" ? "z" : gridOverlayAxis)}>Slice</button>
          <button className={volumeOverlay ? "active" : ""} disabled={!octree || !paperVolumeCapable} onClick={() => { setGridOverlaySlice(volumeOpacity); setGridOverlayAxis("volume"); }}>Full volume</button>
        </div>
        {sliceOverlay && <div className="segmented compact" role="group" aria-label="Slice axis">
          <button className={gridOverlayAxis === "z" ? "active" : ""} onClick={() => setGridOverlayAxis("z")}>Z</button>
          <button className={gridOverlayAxis === "x" ? "active" : ""} onClick={() => setGridOverlayAxis("x")}>X</button>
          <button className={gridOverlayAxis === "y" ? "active" : ""} onClick={() => setGridOverlayAxis("y")}>Y</button>
        </div>}
        {sliceOverlay && <label className="slice-control">
          <span><span>Slice position</span><output>{Math.round(gridOverlaySlice * 100)}%</output></span>
          <input type="range" min={0} max={1} step={0.005} value={gridOverlaySlice} onChange={(event) => setGridOverlaySlice(Number(event.target.value))} aria-label="Grid slice position" />
        </label>}
        {volumeOverlay && <label className="slice-control">
          <span><span>Volume opacity</span><output>{Math.round(volumeOpacity * 100)}%</output></span>
          <input type="range" min={0.05} max={1} step={0.01} value={volumeOpacity} onChange={(event) => setGridOverlaySlice(Number(event.target.value))} aria-label="Diagnostic volume opacity" />
        </label>}
        {overlayActive && <div className="segmented compact" role="group" aria-label="Diagnostic field">
          <button className={gridOverlayMode === "structure" ? "active" : ""} onClick={() => selectOverlayMode("structure")}>Structure</button>
          {adaptive && <button className={gridOverlayMode === "resolution" ? "active" : ""} onClick={() => selectOverlayMode("resolution")}>Cell scale</button>}
          {octree && <button className={gridOverlayMode === "surface" ? "active" : ""} onClick={() => selectOverlayMode("surface")}>Surface band</button>}
          {octree && <button className={gridOverlayMode === "faces" ? "active" : ""} onClick={() => selectOverlayMode("faces")}>Velocity faces</button>}
          {quadtreeTall && <button className={gridOverlayMode === "optical" ? "active" : ""} onClick={() => selectOverlayMode("optical")}>Optical layer</button>}
          <button className={gridOverlayMode === "cfl" ? "active" : ""} onClick={() => selectOverlayMode("cfl")}>CFL load</button>
          <button className={gridOverlayMode === "speed" ? "active" : ""} onClick={() => selectOverlayMode("speed")}>Speed</button>
          <button className={gridOverlayMode === "representation" ? "active" : ""} onClick={() => selectOverlayMode("representation")}>Coverage</button>
          <button className={gridOverlayMode === "phi" ? "active" : ""} onClick={() => selectOverlayMode("phi")}>φ</button>
          <button className={gridOverlayMode === "divergence" ? "active" : ""} onClick={() => selectOverlayMode("divergence")}>Divergence</button>
          <button className={gridOverlayMode === "pressure" ? "active" : ""} onClick={() => selectOverlayMode("pressure")}>Pressure</button>
          {octree && <button className={gridOverlayMode === "projection" ? "active" : ""} onClick={() => selectOverlayMode("projection")}>Projection Δu</button>}
        </div>}
        {overlayActive && octree && <>
          <div className="section-heading technique-heading"><h2>Paper technique</h2><span>LIVE COMPACT DATA</span></div>
          <div className="segmented compact" role="group" aria-label="Unified power octree technique">
            <button className={gridOverlayMode === "power-cells" ? "active" : ""} onClick={() => selectOverlayMode("power-cells")}>Power cells</button>
            <button className={gridOverlayMode === "power-faces" ? "active" : ""} onClick={() => selectOverlayMode("power-faces")}>Power faces</button>
            <button className={gridOverlayMode === "delaunay-tetrahedra" ? "active" : ""} onClick={() => selectOverlayMode("delaunay-tetrahedra")}>Tetrahedra</button>
            <button className={gridOverlayMode === "transition-band" ? "active" : ""} onClick={() => selectOverlayMode("transition-band")}>Transitions</button>
            <button className={gridOverlayMode === "power-operator" ? "active" : ""} onClick={() => selectOverlayMode("power-operator")}>Operator</button>
          </div>
          <div className="section-heading technique-heading"><h2>Lifecycle</h2><span>PUBLICATION STATE</span></div>
          <div className="segmented compact" role="group" aria-label="Unified octree lifecycle">
            <button className={gridOverlayMode === "octree-lifecycle" ? "active" : ""} onClick={() => selectOverlayMode("octree-lifecycle")}>Octree lifecycle</button>
            <button className={gridOverlayMode === "fine-band-lifecycle" ? "active" : ""} aria-label="Inspect fine band and interface seeds" onClick={() => selectOverlayMode("fine-band-lifecycle")}>Fine band</button>
            <button className={gridOverlayMode === "global-fine-phi" ? "active" : ""} aria-label="Inspect global fine phi values and Eikonal residual" onClick={() => { setGridOverlayMode("global-fine-phi"); if (gridOverlayAxis === "volume") setGridOverlayAxis("z"); }}>Fine φ values</button>
            <button className={gridOverlayMode === "section5-face-band" ? "active" : ""} aria-label="Inspect Section 5 face march" onClick={() => selectOverlayMode("section5-face-band")}>Face march</button>
          </div>
          <small className="control-hint" data-testid="fine-publication-gates">{finePublicationGates.map((gate) =>
            `${gate.state === "ready" ? "✓" : gate.state === "failed" ? "✕" : gate.state === "not-required" ? "–" : "…"} ${gate.label}${gate.state === "failed" ? ` (${gate.detail})` : ""}`
          ).join(" → ")}</small>
          <div className="section-heading technique-heading"><h2>Validity audit</h2><span>FAILURE LOCALIZATION</span></div>
          <div className="segmented compact" role="group" aria-label="Unified power octree validity audit">
            <button className={gridOverlayMode === "operator-diagonal" ? "active" : ""} onClick={() => selectOverlayMode("operator-diagonal")}>Diagonal</button>
            <button className={gridOverlayMode === "operator-rhs" ? "active" : ""} onClick={() => selectOverlayMode("operator-rhs")}>RHS</button>
            <button className={gridOverlayMode === "operator-reciprocity" ? "active" : ""} onClick={() => selectOverlayMode("operator-reciprocity")}>Reciprocity</button>
            <button className={gridOverlayMode === "operator-open-fraction" ? "active" : ""} onClick={() => selectOverlayMode("operator-open-fraction")}>Open fraction</button>
            <button className={gridOverlayMode === "tetra-validity" ? "active" : ""} onClick={() => selectOverlayMode("tetra-validity")}>Tetra validity</button>
          </div>
          <small className="control-hint">These views read the exact GPU power catalog, generalized faces, incidence rows, and local Delaunay tetrahedra. Full volume traces each camera ray through the complete live structure with front-to-back alpha compositing; Slice isolates one exact cross-section. Neither rebuilds a dense debug mesh or steers the simulation.</small>
        </>}
        <small className="control-hint">The diagnostic remains an independent layer over the raster scene.</small>
      </>
    </section>

    {overlayActive && <section className="panel-section grid-key" data-testid="grid-legend">
      <strong>{gridKind === "restricted-tall-cell" ? "TALL-CELL GRID" : gridKind === "quadtree-tall-cell" ? "QUADTREE TALL-CELL GRID" : gridKind === "octree" ? "OCTREE GRID" : "UNIFORM GRID"} · {volumeOverlay ? "FULL VOLUME" : `${gridOverlayAxis.toUpperCase()} SLICE`}{gridOverlayMode !== "structure" ? ` · ${{ resolution: "COMPACT LEAF LEVEL", surface: "SPARSE SURFACE BAND", faces: "SPARSE VELOCITY FACES", optical: "OPTICAL LAYER", cfl: "CFL LOAD", speed: "SPEED", representation: "PRESSURE COVERAGE", phi: "LEVEL SET φ", divergence: "POST-PROJECTION DIVERGENCE", pressure: octree ? "PRESSURE POTENTIAL dt·p/ρ" : "MAPPED PRESSURE", projection: "PRESSURE UPDATE ΔU", "power-cells": "POWER SITES / CELLS", "power-faces": "POWER PRIMAL-DUAL GEOMETRY", "delaunay-tetrahedra": "LOCAL DELAUNAY TETRAHEDRA", "transition-band": "BOUNDARY / LEVEL TRANSITIONS", "power-operator": "POWER LAPLACIAN COEFFICIENTS", "octree-lifecycle": "OCTREE REBUILD LIFECYCLE", "fine-band-lifecycle": "FINE-BAND PUBLICATION LIFECYCLE", "global-fine-phi": "PAPER FINE φ / |∇φ|−1", "section5-face-band": "SECTION 5 REGULAR-FACE MARCH", "operator-diagonal": "OPERATOR DIAGONAL", "operator-rhs": "OPERATOR RIGHT-HAND SIDE", "operator-reciprocity": "FACE RECIPROCITY AUDIT", "operator-open-fraction": "FACE OPEN FRACTION", "tetra-validity": "TETRAHEDRON VALIDITY" }[gridOverlayMode]}` : ""}</strong>
      {gridOverlayMode === "structure" && <>
        {tall && <span><i className="sw sw-tall" />tall cell · liquid</span>}
        {tall && <span><i className="sw sw-tall-dry" />tall cell · air</span>}
        <span><i className="sw sw-solid" />rigid body · represented cell</span>
        <span><i className="sw sw-wet" />{tall ? "regular cell · liquid" : "cell · liquid"}</span>
        <span><i className="sw sw-air" />{octree ? "air cells · outline only" : tall ? "regular cell · air" : "cell · air"}</span>
        {gridKind === "restricted-tall-cell" && <span><i className="sw sw-outside" />above band · not stored</span>}
        {!adaptive && <span><i className="sw sw-dot" />stored samples (zoom in)</span>}
        {adaptive && <span>edges follow live adaptive pressure cells</span>}
        {octree && <span><i className="sw" style={{ background: "#ff148c" }} />pink cells · active fine detail shell at |φ| ≤ 1.5h</span>}
        {adaptive && <span>cyan-to-blue boundaries · adaptive pressure hierarchy</span>}
      </>}
      {gridOverlayMode === "resolution" && <>
        <span><i className="sw" style={{ background: "#38adbd" }} />finest compact leaf · 1³</span>
        <span><i className="sw" style={{ background: "#55a8ba" }} />intermediate dyadic cell</span>
        <span><i className="sw" style={{ background: "#152e7a" }} />coarsest compact leaf · {(gpuInfo?.quadtreeMaximumFluidScale ?? "max")}³</span>
        {octree && <span><i className="sw" style={{ background: "#ff05b8" }} />magenta · no compact owner for this represented cell</span>}
        {octree && <span><i className="sw" style={{ background: "#f7a314" }} />orange · pressure row exists but disagrees with the surface leaf generation</span>}
        {octree && <span><i className="sw" style={{ background: "#ff0303" }} />red · compact surface arena reported a global fault</span>}
      </>}
      {gridOverlayMode === "surface" && <>
        <span><i className="sw" style={{ background: "#ff087f" }} />pink · detail-selected core at |φ| ≤ 1.5h</span>
        <span><i className="sw" style={{ background: "#6b309e" }} />violet · allocated core-page support away from φ=0</span>
        <span><i className="sw" style={{ background: "#1fb8d1" }} />cyan · interpolation and transport halo</span>
        <span><i className="sw" style={{ background: "#f5d619" }} />yellow · newly activated page</span>
        <span><i className="sw" style={{ background: "#f59214" }} />orange · desired page not resident</span>
        <span><i className="sw" style={{ background: "#ff05b8" }} />magenta · no compact owner row</span>
        <span><i className="sw" style={{ background: "#f7a314" }} />orange · pressure/surface row generation mismatch</span>
        <span><i className="sw" style={{ background: "#ff0303" }} />red · arena fault or page/state mismatch</span>
        <span><i className="sw" style={{ background: "#0d1f59" }} />blue · valid compact coarse authority outside fine allocation</span>
        <span>In Full volume, page colors expose the complete lifecycle: desired → newly activated → resident core/halo → valid coarse authority or fault.</span>
        <span>{gpuInfo?.adaptiveSurfaceActivePages ?? 0}/{gpuInfo?.adaptiveSurfacePageCapacity ?? 0} resident · {gpuInfo?.adaptiveSurfaceAdapterCandidateRows ?? 0} adapter rows/{gpuInfo?.adaptiveSurfaceAdapterDispatchX ?? 0} groups · {gpuInfo?.adaptiveSurfaceCandidatePages ?? 0} arena candidates · {gpuInfo?.adaptiveSurfaceFinestResidentPages ?? 0} finest · {gpuInfo?.adaptiveSurfaceCoarseResidentPages ?? 0} coarse · max {gpuInfo?.adaptiveSurfaceMaximumResidentLeafSize ?? 0}³ · {gpuInfo?.adaptiveSurfaceDepartureFallbacks ?? 0} departures · fault {gpuInfo?.adaptiveSurfaceOverflowCode ?? 0}</span>
      </>}
      {gridOverlayMode === "faces" && <>
        <span><i className="sw" style={{ background: "#159578" }} />green · complete x/y/z compact-face neighborhood</span>
        <span><i className="sw" style={{ background: "#8c38c7" }} />violet · high coarse/fine incidence count</span>
        <span><i className="sw" style={{ background: "#ff05b8" }} />magenta · no compact owner row</span>
        <span><i className="sw" style={{ background: "#f7a314" }} />orange · missing velocity axis or pressure/surface row mismatch</span>
        <span><i className="sw" style={{ background: "#ff0303" }} />red · face overflow, invalid index, or non-finite velocity</span>
        <span>CFL and Speed also reconstruct directly from these sparse faces; faults are red in wet cells.</span>
      </>}
      {gridOverlayMode === "power-cells" && <>
        <span><i className="sw" style={{ background: "#159578" }} />green · regular Cartesian power cell</span>
        <span><i className="sw" style={{ background: "#8c38c7" }} />violet · catalog-resolved transition power cell</span>
        <span><i className="sw" style={{ background: "#f5ba1a" }} />gold point · pressure site at the octree-cell centre</span>
        <span><i className="sw" style={{ background: "#ff0303" }} />red · missing owner, descriptor, or catalog metric</span>
      </>}
      {gridOverlayMode === "power-faces" && <>
        <span><i className="sw" style={{ background: "#13cfe8" }} />cyan · power-face plane through its exact centroid</span>
        <span><i className="sw" style={{ background: "#8c38c7" }} />violet · dual link between incident pressure sites</span>
        <span><i className="sw" style={{ background: "#f5ba1a" }} />gold · stored face-normal glyph</span>
        <span><i className="sw" style={{ background: "#ff1830" }} />red · valid open/world boundary face</span>
        <span><i className="sw" style={{ background: "#ff00aa" }} />magenta · invalid or incomplete publication</span>
      </>}
      {gridOverlayMode === "delaunay-tetrahedra" && <>
        <span><i className="sw" style={{ background: "#19bfdc" }} />cyan wire · catalog local Delaunay tetrahedron</span>
        <span><i className="sw" style={{ background: "#f5ba1a" }} />gold point · anchor pressure site</span>
        <span><i className="sw" style={{ background: "#ff0303" }} />red · inverted, degenerate, non-finite, or catalog-selector mismatch</span>
        <span>Only non-uniform transition rows have tetrahedra; quiet regular regions intentionally remain clear.</span>
      </>}
      {gridOverlayMode === "transition-band" && <>
        <span><i className="sw" style={{ background: "#ef720f" }} />orange · level-transition row</span>
        <span><i className="sw" style={{ background: "#f51867" }} />pink · domain/free boundary row</span>
        <span>These are boundary and level-transition seed rows. The pressure solver dilates them into the implementation&apos;s graph-ring approximation of the paper&apos;s about-three-voxel localized second-order smoothing band.</span>
      </>}
      {gridOverlayMode === "power-operator" && <>
        <span><i className="sw" style={{ background: "linear-gradient(90deg,#15489a,#18bf8c,#ed2d12)" }} />max incident A/d coefficient · low to high</span>
        <span><i className="sw" style={{ background: "#f5ba1a" }} />gold · high diagonal or residual contribution</span>
        <span><i className="sw" style={{ background: "#ff0303" }} />red · unpublished/asymmetric face graph, non-positive diagonal, or non-finite coefficient</span>
      </>}
      {gridOverlayMode === "octree-lifecycle" && <>
        <span><i className="sw" style={{ background: "#0a193b" }} />dark navy · tile unchanged by the current rebuild</span>
        <span><i className="sw" style={{ background: "#16cbdc" }} />cyan · active tile in the rebuild worklist</span>
        <span><i className="sw" style={{ background: "#ff6812" }} />orange · retired tile in the rebuild worklist</span>
        <span><i className="sw" style={{ background: "#ff1738" }} />red · invalid membership index or lifecycle publication</span>
        <span>Full volume exposes every active and retired rebuild tile at once; Slice isolates one cross-section.</span>
      </>}
      {gridOverlayMode === "fine-band-lifecycle" && <>
        <span><i className="sw" style={{ background: "#ff168f" }} />pink · interface core</span>
        <span><i className="sw" style={{ background: "#ffe10b" }} />yellow · fast-march trial/frontier sample</span>
        <span><i className="sw" style={{ background: "#15c8db" }} />cyan · known redistanced support sample</span>
        <span><i className="sw" style={{ background: "#793ab8" }} />violet · resident sample awaiting a valid/known state</span>
        <span><i className="sw" style={{ background: "#ff7a12" }} />orange · newly activated sample or desired page not resident</span>
        <span><i className="sw" style={{ background: "#10255e" }} />dark blue · valid compact coarse authority outside fine allocation</span>
        <span><i className="sw" style={{ background: "#ff1738" }} />red · failed, stale, or inconsistent publication</span>
        <span>Colors come from the live sparse hash, page metadata, worklist, sample flags, and transaction controls.</span>
      </>}
      {gridOverlayMode === "global-fine-phi" && <>
        <span><i className="sw" style={{ background: "linear-gradient(90deg,#1973eb,#f5f5e6,#ed7829)" }} />paper fine-lattice φ/h · liquid (−), zero crossing, air (+)</span>
        <span><i className="sw" style={{ background: "#f505b8" }} />magenta · |∇φ|−1 residual reaches 0.25 or more</span>
        <span><i className="sw" style={{ background: "#ffffff" }} />white · φ=0 crossing</span>
        <span><i className="sw" style={{ background: "#ff0610" }} />red · stale/rejected redistance or missing stencil neighbor inside resident support</span>
        <span>Direct factor-m Section 5 signed-distance samples. This slice is a paper quantity; it does not show the separate engineering volume-correction estimate.</span>
      </>}
      {gridOverlayMode === "section5-face-band" && <>
        <span><i className="sw" style={{ background: "#f51680" }} />pink · interface-core owner row</span>
        <span><i className="sw" style={{ background: "#f5610c" }} />orange · first support closure</span>
        <span><i className="sw" style={{ background: "#8a34c2" }} />violet · deeper Delaunay support</span>
        <span><i className="sw" style={{ background: "#297ae0" }} />blue · terminal endpoint with committed parent-edge φ</span>
        <span><i className="sw" style={{ background: "#ffe10b" }} />yellow · trial face on the fast-march frontier</span>
        <span><i className="sw" style={{ background: "#15c8b0" }} />cyan · accepted face velocity</span>
        <span><i className="sw" style={{ background: "#e31fb8" }} />magenta · unresolved or invalid-φ face</span>
        <span><i className="sw" style={{ background: "#ff0610" }} />red · first reported row/face key or malformed publication</span>
        <span>Direct GPU view of the regular-face rows, incidence list, φ ordering and march state used before republishing to power faces. The spatial audit reports the exact unresolved split: heap-bound trial, accepted-predecessor scheduler defect, or disconnected.</span>
      </>}
      {gridOverlayMode === "operator-diagonal" && <>
        <span><i className="sw" style={{ background: "linear-gradient(90deg,#15489a,#18bf8c,#ed2d12)" }} />positive pressure diagonal · blue low, teal mid, red high</span>
        <span><i className="sw" style={{ background: "#ff00aa" }} />magenta · zero, negative, non-finite, or unpublished diagonal</span>
      </>}
      {gridOverlayMode === "operator-rhs" && <>
        <span><i className="sw" style={{ background: "linear-gradient(90deg,#1551d8,#20b676,#ef371d)" }} />negative · near zero · positive pressure RHS</span>
        <span><i className="sw" style={{ background: "#ff00aa" }} />magenta · non-finite or unpublished RHS</span>
      </>}
      {gridOverlayMode === "operator-reciprocity" && <>
        <span><i className="sw" style={{ background: "#20bd72" }} />green · paired endpoint/sign and reverse-CSR incidence agree</span>
        <span><i className="sw" style={{ background: "#ff1738" }} />red · missing reverse incidence, endpoint/sign disagreement, or malformed publication</span>
      </>}
      {gridOverlayMode === "operator-open-fraction" && <>
        <span><i className="sw" style={{ background: "linear-gradient(90deg,#ef271f,#8f497a,#18d1c0)" }} />red blocked → cyan open · area-weighted incident-face fraction</span>
        <span><i className="sw" style={{ background: "#ff00aa" }} />magenta · fraction outside [0, 1], non-finite, or unpublished</span>
      </>}
      {gridOverlayMode === "tetra-validity" && <>
        <span><i className="sw" style={{ background: "linear-gradient(90deg,#174b9b,#18bfd8)" }} />blue/teal wire · valid reconstructed handedness</span>
        <span><i className="sw" style={{ background: "#ff1738" }} />red · selector, degeneracy, or reconstructed-volume mismatch</span>
        <span><i className="sw" style={{ background: "#ff00aa" }} />magenta · malformed or unpublished tetrahedron data</span>
      </>}
      {gridOverlayMode === "optical" && <>
        <span><i className="sw" style={{ background: "#f4c33a" }} />retained cubic optical cells · liquid</span>
        <span><i className="sw" style={{ background: "#66cdda" }} />retained cubic optical cells · air</span>
        <span><i className="sw" style={{ background: "#263b58" }} />merged tall-cell interior</span>
        <span><i className="sw" style={{ background: motionAdaptiveOptical ? "#ff3ca6" : "#ededfa" }} />{motionAdaptiveOptical ? "motion-adaptive" : "fixed"} lower boundary</span>
        <span>showing the post-quadtree layer consumed by the pressure solver · {motionAdaptiveOptical ? "motion-adaptive" : "fixed quarter-depth"}</span>
        {motionAdaptiveOptical && <span>α {(gpuInfo?.quadtreeOpticalAlpha ?? 0.5).toFixed(2)} · requested depth {gpuInfo?.quadtreeOpticalMinimumCells ?? "–"}–{gpuInfo?.quadtreeOpticalMaximumCells ?? "–"} cells</span>}
      </>}
      {gridOverlayMode === "cfl" && <>
        <span><i className="sw" style={{ background: "#213a8c" }} />CFL ≈ 0 · idle</span>
        <span><i className="sw" style={{ background: "#38bf57" }} />CFL ≈ 2 · 1 substep</span>
        <span><i className="sw" style={{ background: "#fad133" }} />CFL ≈ 4 · 2 substeps</span>
        <span><i className="sw" style={{ background: "#e63826" }} />CFL ≥ 8 · 4+ substeps</span>
        <span>the conservative speed bound targets CFL ≤ 1, with a 64-substep hard ceiling and a visible projection-clamp fallback</span>
        {gpuInfo?.maxComponentCfl !== undefined && <span>current max CFL {gpuInfo.maxComponentCfl.toFixed(2)} → {gpuInfo.lastSubsteps ?? 1} substep{(gpuInfo.lastSubsteps ?? 1) !== 1 ? "s" : ""}</span>}
      </>}
      {gridOverlayMode === "speed" && <>
        <span><i className="sw" style={{ background: "linear-gradient(90deg,#213a8c,#10a0cc,#38bf57,#fad133,#e63826)" }} />0 → max liquid speed{gpuInfo?.maxSpeed_m_s !== undefined ? ` (${gpuInfo.maxSpeed_m_s.toFixed(2)} m/s)` : ""}</span>
        <span>bright air cells show the extrapolated velocity band; dim cells are air</span>
      </>}
      {gridOverlayMode === "representation" && <>
        <span><i className="sw" style={{ background: "#ff0503" }} />liquid voxel without a liquid pressure DOF</span>
        <span><i className="sw sw-tall-dry" />air tall cells are outline-only by design</span>
      </>}
      {gridOverlayMode === "phi" && <>
        <span><i className="sw" style={{ background: "linear-gradient(90deg,#1973eb,#f5f5e6,#ed7829)" }} />liquid (−) · zero contour · air (+)</span>
      </>}
      {gridOverlayMode === "divergence" && <>
        <span><i className="sw" style={{ background: "linear-gradient(90deg,#1548df,#f5f5f5,#e21a14)" }} />compression (−) · zero · expansion (+)</span>
        <span>color saturates at |∇·u| Δt = 1</span>
      </>}
      {gridOverlayMode === "pressure" && <>
        <span><i className="sw" style={{ background: "linear-gradient(90deg,#213a8c,#10a0cc,#38bf57,#fad133,#e63826)" }} />{octree ? "affine pressure potential dt·p/ρ reconstructed from leaf DOFs (m²/s)" : "latest fine MLS pressure (Pa)"}</span>
      </>}
      {gridOverlayMode === "projection" && <>
        <span><i className="sw" style={{ background: "linear-gradient(90deg,#213a8c,#10a0cc,#38bf57,#fad133,#e63826)" }} />|u after − u before| · normalized by live max speed</span>
        <span>dark coarse-leaf interiors reveal pressure modes that do not reach the dense transport field</span>
      </>}
      {gridKind === "quadtree-tall-cell" && <>
        <span>culled debris {gpuInfo?.quadtreeCulledDebrisCells ?? 0} · CFL clamps {gpuInfo?.quadtreeVelocityClampCount ?? 0} · pressure iterations {gpuInfo?.quadtreePressureIterationsUsed ?? 0}</span>
        <span>topology stale {gpuInfo?.quadtreeTopologyStaleSteps ?? 0}/{gpuInfo?.quadtreeTopologyStaleLimit ?? 0} steps · blocked frames {gpuInfo?.quadtreeRebuildBlockedFrames ?? 0}</span>
        <span>VOF recovery {gpuInfo?.quadtreeVofReconciliationActive ? "ARMED" : "idle"}</span>
      </>}
      <small>{volumeOverlay ? "Orbit the camera to inspect the complete ray-integrated structure; Volume opacity scales front-to-back compositing." : "Drag the highlighted slice edge in the viewport to sweep the plane."}{gridOverlayMode !== "structure" ? " Field modes sample live GPU publications — no readback." : ""}</small>
    </section>}
  </aside>;
}
