"use client";

import { SVO_MESHING_PLUGINS } from "../meshing/plugins";
import { Button, ChoiceField, Facts, FieldActions, FieldList, RangeField, Readout, SelectField, SwitchField } from "../../../../components/ui";
import { formatPipelineDuration } from "../../../../components/PipelineGraph";
import { WorkProgress } from "../../../../components/WorkProgress";
import type { SvoFeatureControlContext, SvoStageControls } from "../../pipeline/control-context";
import { SURFACE_MESH_TIMING_STAGES } from "../../pipeline/render-pipeline-graph";
import { SVO_PRIMARY_LEAF_VISIT_HARD_LIMIT, SVO_SURFACE_MESH_LOD_PIXELS_DEFAULT, SVO_SURFACE_MESH_LOD_PIXELS_MAXIMUM } from "../../pipeline/svo-render-tuning";
import { surfaceMeshProgress } from "./svo-surface-mesh";

export function renderPrimaryTraversalControls({ resolvedPrimary, partitioned, disabledStages, durations, effectiveRendererStatus, smoothSurfaceEnabled, svoMaximumTraversalDepth, setSvoMaximumTraversalDepth, svoMaximumNodeVisits, setSvoMaximumNodeVisits, tuning, updateTuning, modified, resetTuning }: Pick<SvoFeatureControlContext, "resolvedPrimary" | "partitioned" | "disabledStages" | "durations" | "effectiveRendererStatus" | "smoothSurfaceEnabled" | "svoMaximumTraversalDepth" | "setSvoMaximumTraversalDepth" | "svoMaximumNodeVisits" | "setSvoMaximumNodeVisits" | "tuning" | "updateTuning" | "modified" | "resetTuning">): SvoStageControls {
  if (resolvedPrimary !== "mesh") return { settings: 3, node: <FieldList>
      <RangeField label="Maximum traversal depth" unit="levels" value={svoMaximumTraversalDepth}
        min={1} max={21} step={1} digits={0} onChange={setSvoMaximumTraversalDepth}
        hint="Hierarchy depth accepted by every camera traversal. A budget, not a detail control: exceeding it reports traversal exhaustion rather than falling back to a coarser surface." />
      <RangeField label="Maximum node visits" unit="nodes" value={svoMaximumNodeVisits}
        min={1} max={256} step={1} digits={0} onChange={setSvoMaximumNodeVisits}
        hint="Topology nodes allowed per primary traversal call." />
      <RangeField label="Maximum leaf visits" unit="bricks" value={tuning.primaryLeafVisits}
        min={1} max={SVO_PRIMARY_LEAF_VISIT_HARD_LIMIT} step={1} digits={0}
        onChange={(value) => updateTuning("primaryLeafVisits", value)}
        modified={modified("primaryLeafVisits")} onReset={resetTuning("primaryLeafVisits")} />
    </FieldList> };

  // What the rasterizer is set to, then how far the mesh has got, then what it
  // measured: settings on the rail, the figures as facts under it.
  const mesh = effectiveRendererStatus.surfaceMesh;
  const facts = [
    ...SURFACE_MESH_TIMING_STAGES.map(({ stage, label, detail }) => {
      const duration = !partitioned ? undefined : disabledStages.has("primary-traversal")
        ? 0 : durations.get(stage)?.expected_ms;
      return { label, hint: detail, value: duration === undefined ? "—" : formatPipelineDuration(duration) };
    }),
    ...(mesh?.allocatedBytes !== undefined ? [
      { label: "Required mesh records", value: `${mesh.requirementComplete === false ? "≥ " : ""}${mesh.requiredQuads?.toLocaleString() ?? "—"}` },
      { label: "Capacity", value: `${mesh.capacityQuads?.toLocaleString() ?? "—"} records` },
      { label: "Mesh memory", value: `${(mesh.allocatedBytes / (1024 * 1024)).toFixed(1)} / ${((mesh.maximumBytes ?? 0) / (1024 * 1024)).toFixed(1)} MiB` },
      { label: "Mesh builds", value: mesh.builds ?? "—" },
    ] : []),
    ...(mesh?.state === "ready" ? [{ label: "Drawn mesh records", value: mesh.quads?.toLocaleString() ?? "—" }] : []),
  ];
  return { settings: 2, readouts: facts.length, node: <>
    <FieldList>
      <SelectField label="Meshing method" value={tuning.surfaceMeshing}
        options={SVO_MESHING_PLUGINS.map((plugin) => ({ value: plugin.id, label: plugin.label }))}
        disabled={smoothSurfaceEnabled} onChange={(value) => updateTuning("surfaceMeshing", value as typeof tuning.surfaceMeshing)}
        hint="The dual-grid methods construct geometry on GPU using a uniform grid. Sub-grid features may need finer resolution." />
      <RangeField label="Contour inflation" unit="cells" value={tuning.surfaceMeshContourInflation}
        min={0} max={0.5} step={0.01} digits={2} editable
        disabled={smoothSurfaceEnabled || !tuning.surfaceMeshContours}
        onChange={(value) => updateTuning("surfaceMeshContourInflation", value)}
        modified={modified("surfaceMeshContourInflation")} onReset={resetTuning("surfaceMeshContourInflation")}
        hint="Expand each contoured voxel on every side before slicing with its original plane. Overlapping patches can cover indents. 0.10 adds 10% of a cell on each side; raster geometry only." />
    </FieldList>
    <WorkProgress progress={smoothSurfaceEnabled
      ? { label: "Smooth surface unavailable", state: "waiting", detail: "Geometry is withheld. Voxel mesh rendering resumes when smooth reconstruction is disabled." }
      : surfaceMeshProgress(mesh)} />
    <Facts items={facts} />
  </> };
}

export function renderFilteredDetailControls({ resolvedPrimary, smoothSurfaceEnabled, tuning, updateTuning, effectiveRendererStatus, svoStageView, setSvoStageView }: Pick<SvoFeatureControlContext, "resolvedPrimary" | "smoothSurfaceEnabled" | "tuning" | "updateTuning" | "effectiveRendererStatus" | "svoStageView" | "setSvoStageView">): SvoStageControls {
  const disabled = resolvedPrimary !== "mesh" || smoothSurfaceEnabled;
  const normalsDisabled = disabled || !tuning.surfaceMeshNormalSmoothing;
  const lodBricks = effectiveRendererStatus.surfaceMesh?.lodBricks;
  return { settings: 8, readouts: lodBricks ? 1 : 0, node: <FieldList>
    <RangeField label="Detail threshold" unit="px" value={tuning.surfaceMeshLodPixels}
      min={0.25} max={SVO_SURFACE_MESH_LOD_PIXELS_MAXIMUM} step={0.25} digits={2} disabled={disabled} editable
      onChange={(value) => updateTuning("surfaceMeshLodPixels", value)}
      modified={tuning.surfaceMeshLodPixels !== SVO_SURFACE_MESH_LOD_PIXELS_DEFAULT}
      onReset={() => updateTuning("surfaceMeshLodPixels", SVO_SURFACE_MESH_LOD_PIXELS_DEFAULT)}
      hint="Pixels at a 460px viewport height. Higher values allow coarser geometry; the hysteresis band can retain a level slightly above this threshold. The readout takes a typed value." />
    <ChoiceField label="Maximum coarsening" value={String(tuning.surfaceMeshMaxCoarsening)} disabled={disabled}
      options={[{ value: "0", label: "Native" }, { value: "1", label: "2×" }, { value: "2", label: "4×" }, { value: "3", label: "Full brick" }]}
      onChange={(value) => updateTuning("surfaceMeshMaxCoarsening", Number(value))} />
    <SwitchField label="Normal smoothing" checked={tuning.surfaceMeshNormalSmoothing} disabled={disabled}
      onChange={(value) => updateTuning("surfaceMeshNormalSmoothing", value)}
      hint="Use baked shading normals. Choose Native geometry above to smooth lighting without coarsening voxels." />
    <RangeField label="Smoothing strength" unit="%" value={tuning.surfaceMeshNormalStrength * 100}
      min={0} max={100} step={5} disabled={normalsDisabled}
      onChange={(value) => updateTuning("surfaceMeshNormalStrength", value / 100)}
      modified={tuning.surfaceMeshNormalStrength !== 1} onReset={() => updateTuning("surfaceMeshNormalStrength", 1)} />
    <details className="rp-tune"><summary>Advanced tolerances</summary><FieldList>
      <RangeField label="Transition stability" unit="%" value={tuning.surfaceMeshLodHysteresis * 100}
        min={0} max={30} step={1} disabled={disabled}
        onChange={(value) => updateTuning("surfaceMeshLodHysteresis", value / 100)}
        modified={tuning.surfaceMeshLodHysteresis !== 0.15} onReset={() => updateTuning("surfaceMeshLodHysteresis", 0.15)}
        hint="A dead band around LOD transitions. Higher values reduce switching during camera movement; zero selects directly by size." />
      <RangeField label="Normal agreement" value={tuning.surfaceMeshNormalAgreement} min={0} max={1} step={0.05} digits={2} disabled={normalsDisabled}
        onChange={(value) => updateTuning("surfaceMeshNormalAgreement", value)}
        modified={tuning.surfaceMeshNormalAgreement !== 0.5} onReset={() => updateTuning("surfaceMeshNormalAgreement", 0.5)}
        hint="Minimum agreement of contributing coarse-face normals. Higher values preserve more flat faces. Stored agreement has 8-bit precision." />
      <SwitchField label="Preserve close-up face normals" checked={tuning.surfaceMeshPreserveCloseNormals} disabled={normalsDisabled}
        onChange={(value) => updateTuning("surfaceMeshPreserveCloseNormals", value)}
        hint="Fade native geometry back to face normals between one and two detail thresholds on screen." />
    </FieldList></details>
    {/* What the card is set to sits above; what it is showing you, what that
        selects, and the verb that undoes it sit below the rule. */}
    <FieldList section>
      <SwitchField label="LOD colours" checked={svoStageView === "mesh-lod"} disabled={disabled}
        onChange={(enabled) => setSvoStageView(enabled ? "mesh-lod" : "off")}
        hint="Inspect the detail level actually rasterized. Cyan: native; green: 2×; amber: 4×; pink: 8×." />
      {lodBricks && <Readout label="Selected bricks"
        hint="Resident surface bricks by selected LOD, before frustum and occlusion rejection. Counts update asynchronously."
        value={lodBricks.map((count, level) => `${level === 0 ? "Native" : `${2 ** level}×`} ${count.toLocaleString()}`).join(" · ")} />}
      <FieldActions>
        <Button disabled={disabled}
          hint="Return every filtered-detail setting on this card to its balanced value."
          onClick={() => {
            updateTuning("surfaceMeshLodPixels", 1); updateTuning("surfaceMeshNormalSmoothing", true);
            updateTuning("surfaceMeshNormalStrength", 1); updateTuning("surfaceMeshMaxCoarsening", 3);
            updateTuning("surfaceMeshLodHysteresis", 0.15); updateTuning("surfaceMeshNormalAgreement", 0.5);
            updateTuning("surfaceMeshPreserveCloseNormals", true);
          }}>Reset filtering</Button>
      </FieldActions>
    </FieldList>
  </FieldList> };
}

export function renderSeamClosureControls({ silhouetteRefinementStatus }: Pick<SvoFeatureControlContext, "silhouetteRefinementStatus">): SvoStageControls {
  if (silhouetteRefinementStatus.state !== "failed" && silhouetteRefinementStatus.state !== "compiling") return {};
  return { notice: <p data-testid="silhouette-refinement-status" aria-live="polite"
    className={silhouetteRefinementStatus.state === "failed" ? "render-inline-warning" : "render-inline-status"}>
    Primary seam closure: {silhouetteRefinementStatus.state.toUpperCase()}
    {silhouetteRefinementStatus.detail ? ` · ${silhouetteRefinementStatus.detail}` : ""}
  </p> };
}
