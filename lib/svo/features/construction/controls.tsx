"use client";

import { FieldList, RangeField, SwitchField } from "../../../../components/ui";
import type { SvoFeatureControlContext, SvoStageControls } from "../../pipeline/control-context";
import { SVO_ENVIRONMENT_BRICK_REFINEMENT_MAXIMUM,SVO_ENVIRONMENT_REFINEMENT_DEPTH_MAXIMUM } from "../../pipeline/svo-render-tuning";

const trimmed = (value: number) => value.toFixed(5).replace(/\.?0+$/, "");

export function renderSparseWorldBuildControls({ renderRefinementDepth, renderRefinementPermitted, sceneIsDry, updateTuning, modified, resetTuning, leafVoxel_mm, finestCellSize_m, tuning }: Pick<SvoFeatureControlContext, "renderRefinementDepth" | "renderRefinementPermitted" | "sceneIsDry" | "updateTuning" | "modified" | "resetTuning" | "leafVoxel_mm" | "finestCellSize_m" | "tuning">): SvoStageControls {
  return { settings: 3, node: <FieldList>
      <RangeField
        label="Environment refinement depth · REBUILD"
        unit="levels" value={renderRefinementDepth}
        min={0} max={SVO_ENVIRONMENT_REFINEMENT_DEPTH_MAXIMUM}
        step={1} digits={0} disabled={!renderRefinementPermitted}
        onChange={(value) => updateTuning("environmentRefinementDepth", value)}
        modified={modified("environmentRefinementDepth")} onReset={resetTuning("environmentRefinementDepth")}
        hint={renderRefinementPermitted
          ? `Render leaf ${trimmed(leafVoxel_mm)} mm · ${renderRefinementDepth} level${renderRefinementDepth === 1 ? "" : "s"} below the ${trimmed(finestCellSize_m * 1000)} mm simulation lattice.\n\nThis rebuilds only the renderer-owned SVO derivative. The scene lattice, SolidWorld, and simulation state do not change. Scenery and terrain are sampled at the render leaf.${sceneIsDry ? "" : " The water keeps running: solids meet the fluid on the simulation lattice, so a waterline may sit up to one simulation cell off the finer surface drawn here."}\n\nEach level also adds one to the cone hierarchy; past the runtime's twelve, derived lighting withdraws and the cone node reads EXACT FALLBACK.`
          : `Zero on this scene whatever the slider says: this method's solver draws the scene from its own sparse world, and a solver brick pins its node, so the leaf stays the ${trimmed(finestCellSize_m * 1000)} mm lattice. Turn water off under the tank's Water setting, or switch to Uniform Geometric, to move on this environment-only ladder.`} />
      <RangeField label="Environment brick refinement" unit="levels" value={tuning.environmentBrickRefinementLevels}
        min={0} max={SVO_ENVIRONMENT_BRICK_REFINEMENT_MAXIMUM} step={1} digits={0}
        onChange={(value) => updateTuning("environmentBrickRefinementLevels", value)}
        modified={modified("environmentBrickRefinementLevels")} onReset={resetTuning("environmentBrickRefinementLevels")}
        hint="Additional SVO subdivision for authored scenery outside the simulation lattice. Already at its ceiling by default, so the only move is down; changing it rebuilds the sparse world." />
      <SwitchField label="Flat-node exemption · REBUILD" checked={tuning.environmentPlanarRefinementExemption}
        hint="Let the refinement rule stop at a node its surface crosses flatly instead of spending the depth above. The test is second order, so it declines depth exactly where curvature is lowest — which is also where a coarse leaf shows, because the primary shades a leaf as one of six axis-aligned voxel faces. On, the tree is smaller and builds faster, at that cost."
        onChange={(value) => updateTuning("environmentPlanarRefinementExemption", value)} />
    </FieldList> };
}
