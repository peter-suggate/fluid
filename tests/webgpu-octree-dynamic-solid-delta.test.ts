import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { octreeProjectionShader, WebGPUOctreeProjection } from "../lib/webgpu-octree";

const source = readFileSync(new URL("../lib/webgpu-octree.ts", import.meta.url), "utf8");

test("dynamic solid occupancy emits compact old/new transform bounds", () => {
  assert.match(octreeProjectionShader,
    /fn rigidBodyChanged\(body: u32[\s\S]*compaction\[base \+ word\] != currentRigidWord\(body, word\)/,
    "the retained position, dimensions, and orientation bits are the exact solid identity");
  assert.match(octreeProjectionShader,
    /if \(body < previousBodies\) \{ appendRigidBounds\(snapshotRigidBody\(body\)[\s\S]*if \(body < currentBodies\) \{ appendRigidBounds\(rigidBodies\[body\]/,
    "a changed body must dirty both its old and new compact bounds");
  assert.doesNotMatch(octreeProjectionShader,
    /dynamicSolidOccupancyChanged|for \(var flat[\s\S]*currentSolidAt/,
    "recurring solid dirtiness must not rescan cells");
});

test("the exact dirty 1-ring owns recurring solid rasterization", () => {
  const rebuild = WebGPUOctreeProjection.prototype.encodeInlineRebuild.toString().replace(/\s+/g, "");
  assert.match(rebuild,
    /mark\.setPipeline\(this\.buildDirtyTileDeltaPipeline\).*if\(this\.hasDenseSolidCells\)\{dispatch\(this\.rasterizeSolidsPipeline,this\.rasterizeSolidsDeltaPipeline\)/,
    "compact transform bounds must precede exact dirty-tile rasterization");
  assert.match(octreeProjectionShader,
    /fn appendDirtyTileRing[\s\S]*appendDirtyTile\(/,
    "solid changes use the same complete topology-tile 1-ring as phase changes");
  assert.match(octreeProjectionShader,
    /fn appendDirtyTile\([\s\S]*tileChangeFlagsBase\(\) \+ tileIndex/,
    "one generation stamp must deduplicate page, retirement, and rigid-body evidence");
  assert.match(source,
    /const dispatch = \(full: GPUComputePipeline, delta: GPUComputePipeline\)[\s\S]*if \(active\) pass\.dispatchWorkgroupsIndirect\(this\.solveDispatch, 0\);[\s\S]*else pass\.dispatchWorkgroups\(\.\.\.this\.workgroups\)/,
    "full-domain solid rasterization is cold-only, not a recurring fallback");
});

test("solid rasterization and delta comparison share one occupancy evaluator", () => {
  assert.match(octreeProjectionShader,
    /@group\(0\) @binding\(11\) var<storage, read_write> solidCells: array<u32>/);
  assert.doesNotMatch(octreeProjectionShader, /atomic(?:Load|Store)\(&solidCells/,
    "exclusive solid writes and immutable surface reads need no synchronization atomic");
  assert.match(octreeProjectionShader,
    /fn currentSolidAt\(p: vec3i\) -> SolidCell[\s\S]*bodySolidFraction\(rigidBodies\[bodyIndex\], p\)[\s\S]*return SolidCell\(fraction, owner\)/);
  assert.match(octreeProjectionShader,
    /fn rasterizeSolidsAt[\s\S]*let solid = currentSolidAt\(vec3i\(gid\)\)/);
  const raster = octreeProjectionShader.slice(
    octreeProjectionShader.indexOf("fn rasterizeSolidsAt"),
    octreeProjectionShader.indexOf("fn resetTopologyAt"),
  );
  assert.equal((raster.match(/bodySolidFraction/g) ?? []).length, 0,
    "the duplicated legacy raster evaluator must stay deleted");
});
