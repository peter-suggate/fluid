import assert from "node:assert/strict";
import test from "node:test";
import {
  SVO_PRIMARY_WORK_STAGE_VIEWS,
  SVO_RENDER_STAGE_DEFINITIONS,
  svoRenderStageUsesPrimaryWorkMap,
} from "../lib/svo/svo-render-diagnostics";
import {
  createSvoRenderStageOverlayWGSL,
  SVO_RENDER_STAGE_OVERLAY_CONTRACT,
} from "../lib/svo/webgpu-svo-stage-overlay";

test("primary work views share one explicit counter-plane contract", () => {
  assert.deepEqual(SVO_PRIMARY_WORK_STAGE_VIEWS, [
    "primary-work",
    "primary-voxel-cells",
    "primary-node-visits",
    "primary-leaf-visits",
    "primary-entry-tail",
  ]);
  for (const view of SVO_PRIMARY_WORK_STAGE_VIEWS) {
    assert.equal(svoRenderStageUsesPrimaryWorkMap(view), true);
    assert.equal(SVO_RENDER_STAGE_DEFINITIONS[view].group, "Primary work");
  }
  assert.equal(svoRenderStageUsesPrimaryWorkMap("primary-depth"), false);
  assert.equal(svoRenderStageUsesPrimaryWorkMap("off"), false);

  assert.equal(SVO_RENDER_STAGE_OVERLAY_CONTRACT.bindings.primaryWork, 13);
  const wgsl = createSvoRenderStageOverlayWGSL();
  assert.match(wgsl, /stagePrimaryWork:texture_2d<u32>/);
  assert.match(wgsl, /svoStagePrimaryVoxelCells/);
  assert.match(wgsl, /work>=174u/);
});
