import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { octreeLosassoAdaptiveVelocityWGSL }
  from "../lib/webgpu-octree-losasso-adaptive-velocity.wgsl";

test("adaptive velocity seeds interface leaves once with a leaf-owned dispatch", () => {
  const source = readFileSync(new URL(
    "../lib/webgpu-octree-losasso-adaptive-velocity.ts", import.meta.url), "utf8");
  const nodeSeed = octreeLosassoAdaptiveVelocityWGSL.slice(
    octreeLosassoAdaptiveVelocityWGSL.indexOf("fn seedAdaptiveVelocityFrontier"),
    octreeLosassoAdaptiveVelocityWGSL.indexOf(
      "fn seedAdaptiveVelocityLiquidLeaves"));
  const leafSeed = octreeLosassoAdaptiveVelocityWGSL.slice(
    octreeLosassoAdaptiveVelocityWGSL.indexOf("fn seedAdaptiveVelocityLiquidLeaves"),
    octreeLosassoAdaptiveVelocityWGSL.indexOf("fn avFrontierDonor"));

  assert.doesNotMatch(nodeSeed, /avSeedLeaf|avTopoRegion\(11u\)/,
    "node seeding must not revisit every incident leaf");
  assert.match(leafSeed, /leaf>=avHeader\(4u\)/);
  assert.match(leafSeed, /velocityAux0\[node\]!=0u/,
    "leaf seeding must preserve the old active-corner admission predicate");
  assert.match(leafSeed, /if\(admitted\)\{avSeedLeaf\(leaf\);\}/);
  assert.match(source,
    /seedAdaptiveVelocityLiquidLeaves[\s\S]*dispatchWorkgroupsIndirect\(groups\.liveDispatch, groups\.leafDispatchOffsetBytes\)/,
    "the leaf-owned seed pass must use the graph's live leaf dispatch");
});
