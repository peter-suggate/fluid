"use client";

import { PipeButton, PipeChoice, PipeRange, PipeReadout, PipeToggle } from "../../../../components/PipeControls";
import { formatPipelineDuration } from "../../../../components/PipelineGraph";
import { WorkProgress } from "../../../../components/WorkProgress";
import type { SvoFeatureControlContext } from "../../pipeline/control-context";
import { SURFACE_MESH_TIMING_STAGES } from "../../pipeline/render-pipeline-graph";
import { SVO_PRIMARY_LEAF_VISIT_HARD_LIMIT, SVO_SURFACE_MESH_LOD_PIXELS_DEFAULT, SVO_SURFACE_MESH_LOD_PIXELS_MAXIMUM } from "../../pipeline/svo-render-tuning";
import { surfaceMeshProgress } from "./svo-surface-mesh";

export function renderPrimaryTraversalControls({ resolvedPrimary, partitioned, disabledStages, durations, effectiveRendererStatus, smoothSurfaceEnabled, svoMaximumTraversalDepth, setSvoMaximumTraversalDepth, svoMaximumNodeVisits, setSvoMaximumNodeVisits, tuning, updateTuning, modified, resetTuning }: Pick<SvoFeatureControlContext, "resolvedPrimary" | "partitioned" | "disabledStages" | "durations" | "effectiveRendererStatus" | "smoothSurfaceEnabled" | "svoMaximumTraversalDepth" | "setSvoMaximumTraversalDepth" | "svoMaximumNodeVisits" | "setSvoMaximumNodeVisits" | "tuning" | "updateTuning" | "modified" | "resetTuning">) {
  return (resolvedPrimary === "mesh" ? <div className="pipe-fields" aria-label="Rasterization timings">
      {SURFACE_MESH_TIMING_STAGES.map(({ stage, label, detail }) => {
        const duration = !partitioned ? undefined : disabledStages.has("primary-traversal")
          ? 0 : durations.get(stage)?.expected_ms;
        return <PipeReadout key={stage} label={label} hint={detail}
          value={duration === undefined ? "—" : formatPipelineDuration(duration)} />;
      })}
      {effectiveRendererStatus.surfaceMesh?.allocatedBytes !== undefined && <>
        <PipeReadout label="Required quads" value={`${effectiveRendererStatus.surfaceMesh.requirementComplete === false ? "≥ " : ""}${effectiveRendererStatus.surfaceMesh.requiredQuads?.toLocaleString() ?? "—"}`} />
        <PipeReadout label="Capacity" value={`${effectiveRendererStatus.surfaceMesh.capacityQuads?.toLocaleString() ?? "—"} quads`} />
        <PipeReadout label="Mesh memory" value={`${(effectiveRendererStatus.surfaceMesh.allocatedBytes / (1024 * 1024)).toFixed(1)} / ${((effectiveRendererStatus.surfaceMesh.maximumBytes ?? 0) / (1024 * 1024)).toFixed(1)} MiB`} />
        <PipeReadout label="Mesh builds" value={effectiveRendererStatus.surfaceMesh.builds ?? "—"} />
      </>}
      <WorkProgress progress={smoothSurfaceEnabled
        ? { label: "Smooth surface unavailable", state: "waiting", detail: "Geometry is withheld. Voxel mesh rendering resumes when smooth reconstruction is disabled." }
        : surfaceMeshProgress(effectiveRendererStatus.surfaceMesh)} />
      {effectiveRendererStatus.surfaceMesh?.state === "ready"
        && <PipeReadout label="Drawn quads" value={effectiveRendererStatus.surfaceMesh.quads?.toLocaleString() ?? "—"} />}
    </div> : <details className="rp-tune"><summary>Traversal budgets</summary>
      <div className="pipe-fields">
      <PipeRange label="Maximum traversal depth" unit="levels" value={svoMaximumTraversalDepth}
        min={1} max={21} step={1} digits={0} onChange={setSvoMaximumTraversalDepth}
        hint="Hierarchy depth accepted by every camera traversal. A budget, not a detail control: exceeding it reports traversal exhaustion rather than falling back to a coarser surface." />
      <PipeRange label="Maximum node visits" unit="nodes" value={svoMaximumNodeVisits}
        min={1} max={256} step={1} digits={0} onChange={setSvoMaximumNodeVisits}
        hint="Topology nodes allowed per primary traversal call." />
      <PipeRange label="Maximum leaf visits" unit="bricks" value={tuning.primaryLeafVisits}
        min={1} max={SVO_PRIMARY_LEAF_VISIT_HARD_LIMIT} step={1} digits={0}
        onChange={(value) => updateTuning("primaryLeafVisits", value)}
        modified={modified("primaryLeafVisits")} onReset={resetTuning("primaryLeafVisits")} />
      </div>
    </details> );
}

export function renderFilteredDetailControls({ resolvedPrimary, smoothSurfaceEnabled, tuning, updateTuning, effectiveRendererStatus, svoStageView, setSvoStageView }: Pick<SvoFeatureControlContext, "resolvedPrimary" | "smoothSurfaceEnabled" | "tuning" | "updateTuning" | "effectiveRendererStatus" | "svoStageView" | "setSvoStageView">) {
  const disabled = resolvedPrimary !== "mesh" || smoothSurfaceEnabled;
  const normalsDisabled = disabled || !tuning.surfaceMeshNormalSmoothing;
  return <div className="pipe-fields">
    <PipeRange label="Detail threshold" unit="px" value={tuning.surfaceMeshLodPixels}
      min={0.25} max={SVO_SURFACE_MESH_LOD_PIXELS_MAXIMUM} step={0.25} digits={2} disabled={disabled} editable
      onChange={(value) => updateTuning("surfaceMeshLodPixels", value)}
      modified={tuning.surfaceMeshLodPixels !== SVO_SURFACE_MESH_LOD_PIXELS_DEFAULT}
      onReset={() => updateTuning("surfaceMeshLodPixels", SVO_SURFACE_MESH_LOD_PIXELS_DEFAULT)}
      hint="Pixels at a 460px viewport height. Higher values allow coarser geometry; the hysteresis band can retain a level slightly above this threshold. The readout takes a typed value." />
    <PipeChoice label="Maximum coarsening" value={String(tuning.surfaceMeshMaxCoarsening)} disabled={disabled}
      options={[{ value: "0", label: "Native" }, { value: "1", label: "2×" }, { value: "2", label: "4×" }, { value: "3", label: "Full brick" }]}
      onChange={(value) => updateTuning("surfaceMeshMaxCoarsening", Number(value))} />
    <div className="pipe-row" role="group" aria-label="Shading normals">
      <PipeToggle label="Normal smoothing" checked={tuning.surfaceMeshNormalSmoothing} disabled={disabled}
        onChange={(value) => updateTuning("surfaceMeshNormalSmoothing", value)}
        hint="Use baked shading normals. Choose Native geometry above to smooth lighting without coarsening voxels." />
    </div>
    <PipeRange label="Smoothing strength" unit="%" value={tuning.surfaceMeshNormalStrength * 100}
      min={0} max={100} step={5} disabled={normalsDisabled}
      onChange={(value) => updateTuning("surfaceMeshNormalStrength", value / 100)}
      modified={tuning.surfaceMeshNormalStrength !== 1} onReset={() => updateTuning("surfaceMeshNormalStrength", 1)} />
    <details className="rp-tune"><summary>Advanced tolerances</summary><div className="pipe-fields">
      <PipeRange label="Transition stability" unit="%" value={tuning.surfaceMeshLodHysteresis * 100}
        min={0} max={30} step={1} disabled={disabled}
        onChange={(value) => updateTuning("surfaceMeshLodHysteresis", value / 100)}
        modified={tuning.surfaceMeshLodHysteresis !== 0.15} onReset={() => updateTuning("surfaceMeshLodHysteresis", 0.15)}
        hint="A dead band around LOD transitions. Higher values reduce switching during camera movement; zero selects directly by size." />
      <PipeRange label="Normal agreement" value={tuning.surfaceMeshNormalAgreement} min={0} max={1} step={0.05} digits={2} disabled={normalsDisabled}
        onChange={(value) => updateTuning("surfaceMeshNormalAgreement", value)}
        modified={tuning.surfaceMeshNormalAgreement !== 0.5} onReset={() => updateTuning("surfaceMeshNormalAgreement", 0.5)}
        hint="Minimum agreement of contributing coarse-face normals. Higher values preserve more flat faces. Stored agreement has 8-bit precision." />
      <div className="pipe-row" role="group" aria-label="Close-up normals">
        <PipeToggle label="Preserve close-up face normals" checked={tuning.surfaceMeshPreserveCloseNormals} disabled={normalsDisabled}
          onChange={(value) => updateTuning("surfaceMeshPreserveCloseNormals", value)}
          hint="Fade native geometry back to face normals between one and two detail thresholds on screen." />
      </div>
    </div></details>
    <div className="pipe-row" role="group" aria-label="Filtered detail inspection">
      <PipeToggle label="LOD colours" checked={svoStageView === "mesh-lod"} disabled={disabled && svoStageView !== "mesh-lod"}
        onChange={(enabled) => setSvoStageView(enabled ? "mesh-lod" : "off")}
        hint="Inspect the detail level actually rasterized. Cyan: native; green: 2×; amber: 4×; pink: 8×." />
      <PipeButton label="Reset filtering" disabled={disabled}
        hint="Return every filtered-detail setting on this card to its balanced value."
        onClick={() => {
          updateTuning("surfaceMeshLodPixels", 1); updateTuning("surfaceMeshNormalSmoothing", true);
          updateTuning("surfaceMeshNormalStrength", 1); updateTuning("surfaceMeshMaxCoarsening", 3);
          updateTuning("surfaceMeshLodHysteresis", 0.15); updateTuning("surfaceMeshNormalAgreement", 0.5);
          updateTuning("surfaceMeshPreserveCloseNormals", true);
        }} />
    </div>
    {effectiveRendererStatus.surfaceMesh?.lodBricks && <PipeReadout label="Selected bricks"
      hint="Resident surface bricks by selected LOD, before frustum and occlusion rejection. Counts update asynchronously."
      value={effectiveRendererStatus.surfaceMesh.lodBricks.map((count, level) => `${level === 0 ? "Native" : `${2 ** level}×`} ${count.toLocaleString()}`).join(" · ")} />}
  </div>;
}

export function renderSeamClosureControls({ silhouetteRefinementStatus }: Pick<SvoFeatureControlContext, "silhouetteRefinementStatus">) {
  return (silhouetteRefinementStatus.state === "failed" || silhouetteRefinementStatus.state === "compiling"
      ? <p data-testid="silhouette-refinement-status" aria-live="polite"
          className={silhouetteRefinementStatus.state === "failed" ? "render-inline-warning" : "render-inline-status"}>
          Primary seam closure: {silhouetteRefinementStatus.state.toUpperCase()}
          {silhouetteRefinementStatus.detail ? ` · ${silhouetteRefinementStatus.detail}` : ""}
        </p>
      : undefined );
}
