"use client";

import { SVO_MESHING_PLUGINS } from "../meshing/plugins";
import { Button, Facts } from "../../../../components/ui";
import { RailChoice, RailRange, RailReadout, RailSelect, RailToggle } from "../../../../components/rail";
import { WorkProgress } from "../../../../components/WorkProgress";
import type { SvoFeatureControlContext, SvoStageControls } from "../../pipeline/control-context";
import { SVO_PRIMARY_LEAF_VISIT_HARD_LIMIT, SVO_SURFACE_MESH_LOD_PIXELS_DEFAULT, SVO_SURFACE_MESH_LOD_PIXELS_MAXIMUM } from "../../pipeline/svo-render-tuning";
import { surfaceMeshProgress } from "./svo-surface-mesh";

export function renderPrimaryTraversalControls({ resolvedPrimary, effectiveRendererStatus, smoothSurfaceEnabled, svoMaximumTraversalDepth, setSvoMaximumTraversalDepth, svoMaximumNodeVisits, setSvoMaximumNodeVisits, tuning, updateTuning, modified, resetTuning }: Pick<SvoFeatureControlContext, "resolvedPrimary" | "effectiveRendererStatus" | "smoothSurfaceEnabled" | "svoMaximumTraversalDepth" | "setSvoMaximumTraversalDepth" | "svoMaximumNodeVisits" | "setSvoMaximumNodeVisits" | "tuning" | "updateTuning" | "modified" | "resetTuning">): SvoStageControls {
  if (resolvedPrimary !== "mesh") return { node: <>
      <RailRange label="Maximum traversal depth" unit="levels" value={svoMaximumTraversalDepth}
        min={1} max={21} step={1} digits={0} onChange={setSvoMaximumTraversalDepth}
        hint="Hierarchy depth accepted by every camera traversal. A budget, not a detail control: exceeding it reports traversal exhaustion rather than falling back to a coarser surface." />
      <RailRange label="Maximum node visits" unit="nodes" value={svoMaximumNodeVisits}
        min={1} max={256} step={1} digits={0} onChange={setSvoMaximumNodeVisits}
        hint="Topology nodes allowed per primary traversal call." />
      <RailRange label="Maximum leaf visits" unit="bricks" value={tuning.primaryLeafVisits}
        min={1} max={SVO_PRIMARY_LEAF_VISIT_HARD_LIMIT} step={1} digits={0}
        onChange={(value) => updateTuning("primaryLeafVisits", value)}
        modified={modified("primaryLeafVisits")} onReset={resetTuning("primaryLeafVisits")} />
    </> };

  // What the rasterizer is set to, then how far the mesh has got, then what it
  // measured: settings on the rail, the figures as facts under it.
  const mesh = effectiveRendererStatus.surfaceMesh;
  const facts = [
    ...(mesh?.allocatedBytes !== undefined ? [
      { label: "Required mesh records", value: `${mesh.requirementComplete === false ? "≥ " : ""}${mesh.requiredQuads?.toLocaleString() ?? "—"}` },
      { label: "Capacity", value: `${mesh.capacityQuads?.toLocaleString() ?? "—"} records` },
      { label: "Mesh memory", value: `${(mesh.allocatedBytes / (1024 * 1024)).toFixed(1)} / ${((mesh.maximumBytes ?? 0) / (1024 * 1024)).toFixed(1)} MiB` },
      { label: "Mesh builds", value: mesh.builds ?? "—" },
    ] : []),
  ];
  return { node: <>
    <>
      <RailSelect label="Meshing method" value={smoothSurfaceEnabled ? "dual-marching-cubes" : tuning.surfaceMeshing} disabled={smoothSurfaceEnabled}
        options={SVO_MESHING_PLUGINS.map((plugin) => ({ value: plugin.id, label: plugin.label }))}
        onChange={(value) => updateTuning("surfaceMeshing", value as typeof tuning.surfaceMeshing)}
        hint="Smooth surface selects dual marching cubes on a uniform grid to preserve shared edges and smooth silhouettes. Turn it off to choose another mesher. Sub-grid features may need finer resolution." />
      <RailRange label="Contour inflation" unit="cells" value={tuning.surfaceMeshContourInflation}
        min={0} max={0.5} step={0.01} digits={2}
        disabled={smoothSurfaceEnabled || !tuning.surfaceMeshContours}
        onChange={(value) => updateTuning("surfaceMeshContourInflation", value)}
        modified={modified("surfaceMeshContourInflation")} onReset={resetTuning("surfaceMeshContourInflation")}
        hint="Expand each contoured voxel on every side before slicing with its original plane. Overlapping patches can cover indents. 0.10 adds 10% of a cell on each side; raster geometry only." />
    </>
    <WorkProgress progress={surfaceMeshProgress(mesh)} />
    <Facts items={facts} />
  </> };
}

export function renderFilteredDetailControls({ resolvedPrimary, smoothSurfaceEnabled, tuning, updateTuning, effectiveRendererStatus, svoStageView, setSvoStageView }: Pick<SvoFeatureControlContext, "resolvedPrimary" | "smoothSurfaceEnabled" | "tuning" | "updateTuning" | "effectiveRendererStatus" | "svoStageView" | "setSvoStageView">): SvoStageControls {
  const disabled = resolvedPrimary !== "mesh" || smoothSurfaceEnabled;
  const normalsDisabled = disabled || !tuning.surfaceMeshNormalSmoothing;
  const lodBricks = effectiveRendererStatus.surfaceMesh?.lodBricks;
  return { node: <>
    <RailRange label="Detail threshold" unit="px" value={tuning.surfaceMeshLodPixels}
      min={0.25} max={SVO_SURFACE_MESH_LOD_PIXELS_MAXIMUM} step={0.25} digits={2} disabled={disabled}
      onChange={(value) => updateTuning("surfaceMeshLodPixels", value)}
      modified={tuning.surfaceMeshLodPixels !== SVO_SURFACE_MESH_LOD_PIXELS_DEFAULT}
      onReset={() => updateTuning("surfaceMeshLodPixels", SVO_SURFACE_MESH_LOD_PIXELS_DEFAULT)}
      hint="Pixels at a 460px viewport height. Higher values allow coarser geometry; the hysteresis band can retain a level slightly above this threshold. The readout takes a typed value." />
    <RailChoice label="Maximum coarsening" value={String(tuning.surfaceMeshMaxCoarsening)} disabled={disabled}
      options={[{ value: "0", label: "Native" }, { value: "1", label: "2×" }, { value: "2", label: "4×" }, { value: "3", label: "Full brick" }]}
      onChange={(value) => updateTuning("surfaceMeshMaxCoarsening", Number(value))} />
    <RailToggle label="Normal smoothing" checked={tuning.surfaceMeshNormalSmoothing} disabled={disabled}
      onChange={(value) => updateTuning("surfaceMeshNormalSmoothing", value)}
      hint="Blend baked normals in voxel-flat mode. Smooth surface already uses the full baked normals, independently of detail filtering." />
    <RailRange label="Smoothing strength" unit="%" value={tuning.surfaceMeshNormalStrength * 100}
      min={0} max={100} step={5} disabled={normalsDisabled}
      onChange={(value) => updateTuning("surfaceMeshNormalStrength", value / 100)}
      modified={tuning.surfaceMeshNormalStrength !== 1} onReset={() => updateTuning("surfaceMeshNormalStrength", 1)} />
    <details className="rail-more is-nested"><summary>Tolerances</summary><>
      <RailRange label="Transition stability" unit="%" value={tuning.surfaceMeshLodHysteresis * 100}
        min={0} max={30} step={1} disabled={disabled}
        onChange={(value) => updateTuning("surfaceMeshLodHysteresis", value / 100)}
        modified={tuning.surfaceMeshLodHysteresis !== 0.15} onReset={() => updateTuning("surfaceMeshLodHysteresis", 0.15)}
        hint="A dead band around LOD transitions. Higher values reduce switching during camera movement; zero selects directly by size." />
      <RailRange label="Normal agreement" value={tuning.surfaceMeshNormalAgreement} min={0} max={1} step={0.05} digits={2} disabled={normalsDisabled}
        onChange={(value) => updateTuning("surfaceMeshNormalAgreement", value)}
        modified={tuning.surfaceMeshNormalAgreement !== 0.5} onReset={() => updateTuning("surfaceMeshNormalAgreement", 0.5)}
        hint="Minimum agreement of contributing coarse-face normals. Higher values preserve more flat faces. Stored agreement has 8-bit precision." />
      <RailToggle label="Preserve close-up face normals" checked={tuning.surfaceMeshPreserveCloseNormals} disabled={normalsDisabled}
        onChange={(value) => updateTuning("surfaceMeshPreserveCloseNormals", value)}
        hint="Fade native geometry back to face normals between one and two detail thresholds on screen." />
    </></details>
    {/* What the card is set to sits above; what it is showing you, what that
        selects, and the verb that undoes it sit below the rule. */}
    <>
      <RailToggle label="LOD colours" checked={svoStageView === "mesh-lod"} disabled={disabled}
        onChange={(enabled) => setSvoStageView(enabled ? "mesh-lod" : "off")}
        hint="Inspect the detail level actually rasterized. Cyan: native; green: 2×; amber: 4×; pink: 8×." />
      {lodBricks && <RailReadout label="Selected bricks"
        hint="Resident surface bricks by selected LOD, before frustum and occlusion rejection. Counts update asynchronously."
        value={lodBricks.map((count, level) => `${level === 0 ? "Native" : `${2 ** level}×`} ${count.toLocaleString()}`).join(" · ")} />}
      <div className="rail-actions">
        <Button disabled={disabled}
          hint="Return every filtered-detail setting on this card to its balanced value."
          onClick={() => {
            updateTuning("surfaceMeshLodPixels", 1); updateTuning("surfaceMeshNormalSmoothing", true);
            updateTuning("surfaceMeshNormalStrength", 1); updateTuning("surfaceMeshMaxCoarsening", 3);
            updateTuning("surfaceMeshLodHysteresis", 0.15); updateTuning("surfaceMeshNormalAgreement", 0.5);
            updateTuning("surfaceMeshPreserveCloseNormals", true);
          }}>Reset filtering</Button>
      </div>
    </>
  </> };
}

export function renderSeamClosureControls({ silhouetteRefinementStatus }: Pick<SvoFeatureControlContext, "silhouetteRefinementStatus">): SvoStageControls {
  if (silhouetteRefinementStatus.state !== "failed" && silhouetteRefinementStatus.state !== "compiling") return {};
  return { notice: <small data-testid="silhouette-refinement-status" aria-live="polite" className="rail-note"
    data-tone={silhouetteRefinementStatus.state === "failed" ? "attention" : undefined}>
    Primary seam closure: {silhouetteRefinementStatus.state.toUpperCase()}
    {silhouetteRefinementStatus.detail ? ` · ${silhouetteRefinementStatus.detail}` : ""}
  </small> };
}
