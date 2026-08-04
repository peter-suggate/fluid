import assert from "node:assert/strict";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { spawnSync } from "node:child_process";
import test from "node:test";
import {
  OCTREE_PIPELINED_PCG_FIXED_POINT_LIMBS,
  OCTREE_PIPELINED_PCG_PARTIAL_BYTES,
  octreePipelinedMGPCGShader,
  planOctreePipelinedMGPCG,
} from "../lib/webgpu-octree-pipelined-mgpcg";
import { octreeSection43HybridPreconditionerShader } from
  "../lib/webgpu-octree-section43-preconditioner";

test("wide MGPCG reductions use exact integer limbs", () => {
  assert.equal(OCTREE_PIPELINED_PCG_FIXED_POINT_LIMBS, 36);
  assert.equal(OCTREE_PIPELINED_PCG_PARTIAL_BYTES, 36 * 4 * 4);
  assert.match(octreePipelinedMGPCGShader,
    /partials: array<atomic<i32>>/);
  assert.match(octreePipelinedMGPCGShader,
    /fn addFixedF32[\s\S]*atomicAdd\(&partials/);
  assert.match(octreePipelinedMGPCGShader,
    /fn fixedScalarValue[\s\S]*atomicLoad\(&partials[\s\S]*floorDiv256/);
  assert.doesNotMatch(octreePipelinedMGPCGShader,
    /fn reduceMergedPartials[\s\S]*partials\[workgroup\.x\] = merged\[0\]/);
  assert.match(octreeSection43HybridPreconditionerShader,
    /fn bandWorksetBase[\s\S]*accepted\[5\]/,
    "the Section 4.3 shell must select the dynamic-boundary bank");
  assert.match(octreeSection43HybridPreconditionerShader,
    /atomicStore\(&bandWorksets\[base\], accepted\[4\]\)/,
    "compact shell worksets must carry the dynamic-boundary epoch");

  const plan = planOctreePipelinedMGPCG({ rowCapacity: 65_537 });
  assert.deepEqual(plan.rowDispatch, [1_025, 1, 1]);
  assert.equal(plan.reductionPartialBytes,
    Math.ceil(65_537 / 128) * OCTREE_PIPELINED_PCG_PARTIAL_BYTES);
});

test("wide MGPCG and its Section 4.3 shell are accepted by naga", () => {
  const naga = process.env.NAGA ?? "naga";
  if (spawnSync(naga, ["--version"], { encoding: "utf8" }).error) return;
  const directory = mkdtempSync(join(tmpdir(), "fluid-wide-mgpcg-wgsl-"));
  try {
    for (const [name, authored] of [
      ["outer", octreePipelinedMGPCGShader],
      ["section43", octreeSection43HybridPreconditionerShader],
    ] as const) {
      // Naga does not implement the standardized subgroup extension yet. The
      // base shader's subgroup builtins are activity-only and unused by its
      // arithmetic, so remove only those declarations for this parser gate.
      const source = authored.replace("enable subgroups;", "").replace(
        /\n  @builtin\(subgroup_invocation_id\) subgroupLane: u32,\n  @builtin\(subgroup_size\) subgroupSize: u32,/g,
        "",
      );
      const path = join(directory, `${name}.wgsl`);
      writeFileSync(path, source);
      const result = spawnSync(naga, [path], { encoding: "utf8" });
      assert.equal(result.status, 0, `${name}:\n${result.stderr || result.stdout}`);
    }
  } finally {
    rmSync(directory, { recursive: true, force: true });
  }
});
