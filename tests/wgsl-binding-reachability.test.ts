import assert from "node:assert/strict";
import test from "node:test";
import { fineToCoarseLevelSetWGSL } from "../lib/webgpu-octree-fine-to-coarse-levelset";
import { structuredBoundaryCoefficientWGSL } from "../lib/webgpu-octree-structured-boundary";
import {
  assertWGSLActivityBindingEligibility,
  auditExplicitComputeBufferBindings,
  auditWGSLComputeBindingReachability,
} from "../lib/wgsl-binding-reachability";

test("WGSL binding audit follows transitive calls and ignores comments and fields", () => {
  const wgsl = /* wgsl */ `
struct Params { count: u32, unusedData: u32 }
@group(0) @binding(0) var<uniform> params: Params;
@binding(1) @group(0) var<storage, read> inputData: array<u32>;
@group(0) @binding(2) var<storage, read_write> outputData: array<u32>;
@group(0) @binding(3) var<storage, read> unusedData: array<u32>;
@group(1) @binding(0) var sourceTexture: texture_2d<f32>;

fn leaf(index: u32) -> u32 {
  return inputData[index] + u32(textureLoad(sourceTexture, vec2i(0), 0).x);
}

fn write(index: u32) {
  // unusedData is deliberately only mentioned in prose.
  let fieldWithSameName = params.unusedData;
  outputData[index] = leaf(index) + params.count + fieldWithSameName;
}

fn unreachable() {
  let value = unusedData[0];
}

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) id: vec3u) {
  write(id.x);
  /* unreachable(); */
}
`;

  const audit = auditWGSLComputeBindingReachability(wgsl, "main");

  assert.deepEqual(audit.reachableFunctions, ["leaf", "main", "write"]);
  assert.deepEqual(audit.storage.map(({ name }) => name), ["inputData", "outputData"]);
  assert.deepEqual(audit.uniform.map(({ name }) => name), ["params"]);
  assert.deepEqual(audit.other.map(({ name }) => name), ["sourceTexture"]);
  assert.equal(audit.storageCount, 2);
  assert.equal(audit.uniformCount, 1);
});

test("fine-to-coarse restriction has nine reachable storage bindings", () => {
  const audit = auditWGSLComputeBindingReachability(
    fineToCoarseLevelSetWGSL,
    "restrictCoarseRows",
  );

  assert.deepEqual(audit.storage.map(({ binding }) => binding), [1, 2, 3, 4, 5, 8, 9, 12, 13]);
  assert.deepEqual(audit.uniform.map(({ binding }) => binding), [0]);
  assert.equal(audit.storageCount, 9);
  assert.equal(audit.uniformCount, 1);
  assert.deepEqual(assertWGSLActivityBindingEligibility(audit), {
    entryPoint: "restrictCoarseRows",
    productionStorageCount: 9,
    activityStorageCount: 1,
    totalStorageCount: 10,
    maximumStorageCount: 10,
  });
});

test("structured boundary resolve has ten reachable storage bindings", () => {
  const audit = auditWGSLComputeBindingReachability(
    structuredBoundaryCoefficientWGSL,
    "resolveStructuredBoundarySlots",
  );

  assert.deepEqual(audit.storage.map(({ binding }) => binding), [2, 3, 4, 6, 7, 8, 9, 11, 16, 25]);
  assert.deepEqual(audit.uniform.map(({ binding }) => binding), [0, 5]);
  assert.equal(audit.storageCount, 10);
  assert.equal(audit.uniformCount, 2);
  assert.throws(() => assertWGSLActivityBindingEligibility(audit),
    /10 production storage bindings.*exceed 10/);
});

test("explicit layout audit counts only compute-visible buffer entries", () => {
  const audit = auditExplicitComputeBufferBindings([
    { binding: 3, visibility: 0x4, buffer: {} },
    { binding: 0, visibility: 0x5, buffer: { type: "storage" } },
    { binding: 1, visibility: 0x4, buffer: { type: "read-only-storage" } },
    { binding: 2, visibility: 0x2, buffer: { type: "storage" } },
    { binding: 4, visibility: 0x4 },
  ]);

  assert.deepEqual(audit.storageBindings, [0, 1]);
  assert.deepEqual(audit.uniformBindings, [3]);
  assert.equal(audit.storageCount, 2);
  assert.equal(audit.uniformCount, 1);
});

test("WGSL binding audit rejects invalid entry points and duplicate slots", () => {
  const wgsl = /* wgsl */ `
@group(0) @binding(0) var<storage, read> first: array<u32>;
@group(0) @binding(0) var<storage, read> second: array<u32>;
fn helper() {}
@compute @workgroup_size(1) fn main() { let value = first[0]; }
`;

  assert.throws(() => auditWGSLComputeBindingReachability(wgsl, "main"), /duplicated/);
  assert.throws(() => auditWGSLComputeBindingReachability("fn helper() {}", "helper"),
    /not a compute entry point/);
  assert.throws(() => auditWGSLComputeBindingReachability("", "missing"), /is missing/);
  assert.throws(() => auditWGSLComputeBindingReachability("", "not-valid"), /is invalid/);
  assert.throws(() => auditExplicitComputeBufferBindings([
    { binding: 0, visibility: 0x4, buffer: {} },
    { binding: 0, visibility: 0x4, buffer: {} },
  ]), /duplicated/);
});
