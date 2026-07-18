"use client";

import { getMethod } from "@/lib/methods";
import { useDiagnosticsStore } from "@/lib/stores/diagnostics-store";
import { useMethodStore } from "@/lib/stores/method-store";
import { useSceneStore } from "@/lib/stores/scene-store";
import { useUIStore } from "@/lib/stores/ui-store";
import { getEnvironmentPreset } from "@/lib/environments";
import { getScenePreset } from "@/lib/scenes";

export function VisualPanel() {
  const methodId = useMethodStore((state) => state.methodId);
  const gpuInfo = useDiagnosticsStore((state) => state.gpuInfo);
  const view = useUIStore((state) => state.view);
  const setView = useUIStore((state) => state.setView);
  const waterRenderMode = useUIStore((state) => state.waterRenderMode);
  const setWaterRenderMode = useUIStore((state) => state.setWaterRenderMode);
  const voxelRenderMode = useUIStore((state) => state.voxelRenderMode);
  const setVoxelRenderMode = useUIStore((state) => state.setVoxelRenderMode);
  const presetId = useSceneStore((state) => state.presetId);
  const targetFps = useUIStore((state) => state.targetFps);
  const setTargetFps = useUIStore((state) => state.setTargetFps);
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
  const motionAdaptiveOptical = gpuInfo?.quadtreeOpticalLayerMode === "adaptive-motion";
  const background = getEnvironmentPreset(getScenePreset(presetId).background);

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
      <div className="section-heading"><h2>Frame pacing</h2><span>REALTIME</span></div>
      <label className="select-control">
        <span>Target presentation rate</span>
        <select aria-label="Target frames per second" value={targetFps} onChange={(event) => setTargetFps(Number(event.target.value))}>
          <option value={24}>24 fps</option>
          <option value={30}>30 fps</option>
          <option value={60}>60 fps</option>
          <option value={90}>90 fps</option>
          <option value={120}>120 fps</option>
        </select>
      </label>
      <small className="control-hint">Raster surface extraction and presentation target {targetFps} Hz ({(1000 / targetFps).toFixed(2)} ms). Physics stays at real-time ×1 using the stable scene substep.</small>
    </section>

    <section className="panel-section utility-controls">
      <div className="section-heading"><h2>Background</h2><span>SCENE ART DIRECTION</span></div>
      <div className="scene-background-card" aria-label={`Scene background: ${background.name}`}>
        <span className="environment-swatch" aria-hidden="true">
          {background.swatch.map((color) => <i key={color} style={{ background: color }} />)}
        </span>
        <span><strong>{background.shortName}</strong><small>{background.description}</small></span>
      </div>
      <small className="control-hint">The selected scene sets its background. Choose another scene to change both together.</small>
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
      <div className="section-heading"><h2>Scene representation</h2><span>HYBRID OCTREE</span></div>
      <div className="segmented compact">
        <button className={voxelRenderMode === "smooth" ? "active" : ""} onClick={() => setVoxelRenderMode("smooth")}>Hybrid</button>
        <button className={voxelRenderMode === "raw-voxels" ? "active" : ""} onClick={() => setVoxelRenderMode("raw-voxels")}>Raw voxels</button>
        <button className={voxelRenderMode === "brick-grid" ? "active" : ""} onClick={() => setVoxelRenderMode("brick-grid")}>Brick grid</button>
      </div>
      <small className="control-hint">Hybrid view renders smooth water, glass, terrain, and rigid surfaces while the octree keeps their fluid/CSG proxies in GPU-resident sparse bricks. Raw and grid views inspect that same publication with indirect GPU draws and no topology readback.</small>
    </section>

    <section className="panel-section utility-controls">
      <div className="section-heading"><h2>Solver grid</h2><span>DEBUG LAYER</span></div>
      {view === "scientific" ? <>
        <div className="segmented compact">
          <button className={gridOverlayAxis === "off" ? "active" : ""} onClick={() => setGridOverlayAxis("off")}>Off</button>
          <button className={gridOverlayAxis === "z" ? "active" : ""} onClick={() => setGridOverlayAxis("z")}>Z slice</button>
          <button className={gridOverlayAxis === "x" ? "active" : ""} onClick={() => setGridOverlayAxis("x")}>X slice</button>
          <button className={gridOverlayAxis === "y" ? "active" : ""} onClick={() => setGridOverlayAxis("y")}>Y slice</button>
        </div>
        {gridOverlayAxis !== "off" && <label className="slice-control">
          <span><span>Slice position</span><output>{Math.round(gridOverlaySlice * 100)}%</output></span>
          <input type="range" min={0} max={1} step={0.005} value={gridOverlaySlice} onChange={(event) => setGridOverlaySlice(Number(event.target.value))} aria-label="Grid slice position" />
        </label>}
        {gridOverlayAxis !== "off" && <div className="segmented compact" role="group" aria-label="Slice field">
          <button className={gridOverlayMode === "structure" ? "active" : ""} onClick={() => setGridOverlayMode("structure")}>Structure</button>
          {adaptive && <button className={gridOverlayMode === "resolution" ? "active" : ""} onClick={() => setGridOverlayMode("resolution")}>Cell scale</button>}
          {quadtreeTall && <button className={gridOverlayMode === "optical" ? "active" : ""} onClick={() => setGridOverlayMode("optical")}>Optical layer</button>}
          <button className={gridOverlayMode === "cfl" ? "active" : ""} onClick={() => setGridOverlayMode("cfl")}>CFL load</button>
          <button className={gridOverlayMode === "speed" ? "active" : ""} onClick={() => setGridOverlayMode("speed")}>Speed</button>
          <button className={gridOverlayMode === "representation" ? "active" : ""} onClick={() => setGridOverlayMode("representation")}>Coverage</button>
          <button className={gridOverlayMode === "phi" ? "active" : ""} onClick={() => setGridOverlayMode("phi")}>φ</button>
          <button className={gridOverlayMode === "divergence" ? "active" : ""} onClick={() => setGridOverlayMode("divergence")}>Divergence</button>
          <button className={gridOverlayMode === "pressure" ? "active" : ""} onClick={() => setGridOverlayMode("pressure")}>Pressure</button>
          {octree && <button className={gridOverlayMode === "projection" ? "active" : ""} onClick={() => setGridOverlayMode("projection")}>Projection Δu</button>}
        </div>}
        <small className="control-hint">The slice remains an independent layer, so it can be combined with either optical renderer.</small>
      </> : <p className="panel-note">Switch to Scientific view to configure debug layers.</p>}
    </section>

    {view === "scientific" && gridOverlayAxis !== "off" && <section className="panel-section grid-key" data-testid="grid-legend">
      <strong>{gridKind === "restricted-tall-cell" ? "TALL-CELL GRID" : gridKind === "quadtree-tall-cell" ? "QUADTREE TALL-CELL GRID" : gridKind === "octree" ? "OCTREE GRID" : "UNIFORM GRID"} · {gridOverlayAxis.toUpperCase()} SLICE{gridOverlayMode !== "structure" ? ` · ${{ resolution: "PRESSURE CELL SCALE", optical: "OPTICAL LAYER", cfl: "CFL LOAD", speed: "SPEED", representation: "PRESSURE COVERAGE", phi: "LEVEL SET φ", divergence: "POST-PROJECTION DIVERGENCE", pressure: "MAPPED PRESSURE", projection: "PRESSURE UPDATE ΔU" }[gridOverlayMode]}` : ""}</strong>
      {gridOverlayMode === "structure" && <>
        {tall && <span><i className="sw sw-tall" />tall cell · liquid</span>}
        {tall && <span><i className="sw sw-tall-dry" />tall cell · air</span>}
        <span><i className="sw sw-solid" />rigid body · represented cell</span>
        <span><i className="sw sw-wet" />{tall ? "regular cell · liquid" : "cell · liquid"}</span>
        <span><i className="sw sw-air" />{octree ? "air cells · outline only" : tall ? "regular cell · air" : "cell · air"}</span>
        {gridKind === "restricted-tall-cell" && <span><i className="sw sw-outside" />above band · not stored</span>}
        {!adaptive && <span><i className="sw sw-dot" />stored samples (zoom in)</span>}
        {adaptive && <span>edges follow live adaptive pressure cells</span>}
        {adaptive && <span><i className="sw" style={{ background: "#ff148c" }} />pink cells and boundaries · finest represented pressure cell · 1³</span>}
        {adaptive && <span>all other boundaries retain the structural grid color</span>}
      </>}
      {gridOverlayMode === "resolution" && <>
        <span><i className="sw" style={{ background: "#ff2994" }} />finest represented pressure cell · 1³</span>
        <span><i className="sw" style={{ background: "#55a8ba" }} />intermediate dyadic cell</span>
        <span><i className="sw" style={{ background: "#152e7a" }} />coarsest represented pressure cell · {(gpuInfo?.quadtreeMaximumFluidScale ?? "max")}³</span>
        <span>surface-band cells appear as the finest pink region</span>
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
        <span><i className="sw" style={{ background: "linear-gradient(90deg,#213a8c,#10a0cc,#38bf57,#fad133,#e63826)" }} />{octree ? "affine pressure reconstructed from leaf DOFs" : "latest fine MLS pressure"}</span>
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
      <small>Drag the highlighted slice edge in the viewport to sweep the plane.{gridOverlayMode !== "structure" ? " Field modes sample live GPU textures — no readback." : ""}</small>
    </section>}
  </aside>;
}
