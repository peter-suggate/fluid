"use client";

import { PipeRange } from "../../../../components/PipeControls";
import { formatPipelineDuration } from "../../../../components/PipelineGraph";
import { WorkProgress } from "../../../../components/WorkProgress";
import type { SvoFeatureControlContext } from "../../pipeline/control-context";
import { SURFACE_MESH_TIMING_STAGES } from "../../pipeline/render-pipeline-graph";
import { SVO_PRIMARY_LEAF_VISIT_HARD_LIMIT } from "../../pipeline/svo-render-tuning";
import { surfaceMeshProgress } from "./svo-surface-mesh";

export function renderPrimaryTraversalControls({ resolvedPrimary, partitioned, disabledStages, durations, effectiveRendererStatus, smoothSurfaceEnabled, svoMaximumTraversalDepth, setSvoMaximumTraversalDepth, svoMaximumNodeVisits, setSvoMaximumNodeVisits, tuning, updateTuning, modified, resetTuning }: Pick<SvoFeatureControlContext, "resolvedPrimary" | "partitioned" | "disabledStages" | "durations" | "effectiveRendererStatus" | "smoothSurfaceEnabled" | "svoMaximumTraversalDepth" | "setSvoMaximumTraversalDepth" | "svoMaximumNodeVisits" | "setSvoMaximumNodeVisits" | "tuning" | "updateTuning" | "modified" | "resetTuning">) {
  return (resolvedPrimary === "mesh" ? <div className="pipe-fields" aria-label="Rasterization timings">
      {SURFACE_MESH_TIMING_STAGES.map(({ stage, label, detail }) => {
        const duration = !partitioned ? undefined : disabledStages.has("primary-traversal")
          ? 0 : durations.get(stage)?.expected_ms;
        return <div key={stage} title={detail} style={{ display: "flex", justifyContent: "space-between", gap: 12 }}>
          <span>{label}</span><code>{duration === undefined ? "—" : formatPipelineDuration(duration)}</code>
        </div>;
      })}
      {effectiveRendererStatus.surfaceMesh?.allocatedBytes !== undefined && <>
        <div>Required quads: {effectiveRendererStatus.surfaceMesh.requirementComplete === false ? "≥ " : ""}{effectiveRendererStatus.surfaceMesh.requiredQuads?.toLocaleString() ?? "—"}</div>
        <div>Capacity: {effectiveRendererStatus.surfaceMesh.capacityQuads?.toLocaleString() ?? "—"} quads</div>
        <div>Mesh memory: {(effectiveRendererStatus.surfaceMesh.allocatedBytes / (1024 * 1024)).toFixed(1)} MiB / {((effectiveRendererStatus.surfaceMesh.maximumBytes ?? 0) / (1024 * 1024)).toFixed(1)} MiB limit</div>
        <div>Mesh builds: {effectiveRendererStatus.surfaceMesh.builds ?? "—"}</div>
      </>}
      <WorkProgress progress={smoothSurfaceEnabled
        ? { label: "Smooth surface unavailable", state: "waiting", detail: "Geometry is withheld. Voxel mesh rendering resumes when smooth reconstruction is disabled." }
        : surfaceMeshProgress(effectiveRendererStatus.surfaceMesh)} />
      {effectiveRendererStatus.surfaceMesh?.state === "ready" && <div>{effectiveRendererStatus.surfaceMesh.quads?.toLocaleString() ?? "—"} drawn quads</div>}
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

export function renderSeamClosureControls({ silhouetteRefinementStatus }: Pick<SvoFeatureControlContext, "silhouetteRefinementStatus">) {
  return (silhouetteRefinementStatus.state === "failed" || silhouetteRefinementStatus.state === "compiling"
      ? <p data-testid="silhouette-refinement-status" aria-live="polite"
          className={silhouetteRefinementStatus.state === "failed" ? "render-inline-warning" : "render-inline-status"}>
          Primary seam closure: {silhouetteRefinementStatus.state.toUpperCase()}
          {silhouetteRefinementStatus.detail ? ` · ${silhouetteRefinementStatus.detail}` : ""}
        </p>
      : undefined );
}
