import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import {
  octreeSparseBrickDebugPublicationShader,
  planOctreeSparseBrickEncode,
} from "../lib/webgpu-octree-sparse-bricks";
import {
  createSparseVoxelInspectionPublicationController,
  type SparseVoxelRenderSource,
} from "../lib/webgpu-voxel-debug";

test("inspection publication controller defaults on and revisions only real transitions", () => {
  let encodeCount = 0;
  const controller = createSparseVoxelInspectionPublicationController(true, () => { encodeCount += 1; });
  assert.equal(controller.enabled, true, "legacy producers retain expanded publication by default");
  assert.equal(controller.revision, 1);
  assert.equal(controller.encodePending({} as GPUCommandEncoder), true);
  assert.equal(encodeCount, 1);
  assert.equal(controller.encodePending({} as GPUCommandEncoder), false);
  assert.equal(controller.setEnabled(true), false);
  assert.equal(controller.revision, 1);
  assert.equal(controller.setEnabled(false), true);
  assert.equal(controller.enabled, false);
  assert.equal(controller.revision, 2);
  assert.equal(controller.setEnabled(true), true, "inspection can be restored before the next encode, including a paused repaint");
  assert.equal(controller.enabled, true);
  assert.equal(controller.revision, 3);
  assert.equal(controller.encodePending({} as GPUCommandEncoder), true);
  assert.equal(encodeCount, 2);
  controller.markEncoded();
  assert.equal(controller.encodePending({} as GPUCommandEncoder), false);
});

test("smooth production plan schedules zero expanded debug work without gating structure", () => {
  assert.deepEqual(planOctreeSparseBrickEncode(false), {
    structuralPublication: true,
    inspectionPublication: false,
    inspectionCountCopies: 0,
    inspectionComputePasses: 0,
    inspectionDispatches: 0,
  });
  assert.deepEqual(planOctreeSparseBrickEncode(), {
    structuralPublication: true,
    inspectionPublication: true,
    inspectionCountCopies: 2,
    inspectionComputePasses: 2,
    inspectionDispatches: 2,
  });
});

test("inspection materialization follows each leaf's authoritative payload offset", () => {
  const voxelMaterializer = octreeSparseBrickDebugPublicationShader.slice(
    octreeSparseBrickDebugPublicationShader.indexOf("fn recordForVoxel"),
    octreeSparseBrickDebugPublicationShader.indexOf("@compute @workgroup_size(256)"),
  );
  const brickMaterializer = octreeSparseBrickDebugPublicationShader.slice(
    octreeSparseBrickDebugPublicationShader.indexOf("fn materializeBricks"),
  );
  assert.match(voxelMaterializer, /let payloadIndex = leaf\.topology\.y \+ localIndex;/);
  assert.match(brickMaterializer, /let payloadIndex = leaves\[leafIndex\]\.topology\.y \+ local;/);
  assert.equal((octreeSparseBrickDebugPublicationShader.match(/payloadIndex < arrayLength\(&materialOwners\)/g) ?? []).length, 2,
    "both voxel and brick publication guard leaf-relative payload lookup");
  assert.doesNotMatch(voxelMaterializer, /materialOwners\[index\]/,
    "expanded record index is not an authoritative sparse payload index");
  assert.doesNotMatch(brickMaterializer, /materialOwners\[leafIndex \* voxelsPerBrick \+ local\]/,
    "leaf order is independent from sparse payload arena placement");
});

test("inspection control is optional on legacy render sources", () => {
  const legacySource = {
    voxelRecords: {} as GPUBufferBinding,
    voxelCount: {} as GPUBufferBinding,
    brickRecords: {} as GPUBufferBinding,
    brickCount: {} as GPUBufferBinding,
    materials: {} as GPUBufferBinding,
    voxelCapacity: 1,
    brickCapacity: 1,
    materialCount: 1,
    revision: 1,
  } satisfies SparseVoxelRenderSource;
  assert.equal(legacySource.revision, 1);
  assert.equal("inspectionPublication" in legacySource, false);
});

test("producer gates only expanded records and always reaches atlas and structural finalization", () => {
  const source = readFileSync(new URL("../lib/webgpu-octree-sparse-bricks.ts", import.meta.url), "utf8");
  const encodeStart = source.indexOf("  encode(encoder:");
  const encodeEnd = source.indexOf("  private encodeInspectionPublication", encodeStart);
  const encode = source.slice(encodeStart, encodeEnd);
  const gateStart = encode.indexOf("if (encodePlan.inspectionPublication)");
  const atlasStart = encode.indexOf("this.atlas.encodeBulkRefresh");
  const finalizerStart = encode.indexOf("const finalizer = encoder.beginComputePass", gateStart);
  assert.ok(atlasStart >= 0 && gateStart > atlasStart && finalizerStart > gateStart,
    "atlas refresh, optional inspection, and structural finalization remain ordered but independently gated");
  const gatedWork = encode.slice(gateStart, finalizerStart);
  assert.match(gatedWork, /encodeInspectionPublication/);
  assert.doesNotMatch(gatedWork, /encodePublish|encodeFromDenseFields|proxyVoxelizer|atlas|structural publication/);
  assert.match(encode.slice(atlasStart, gateStart), /this\.atlas\.encodeBulkRefresh/);
  assert.match(encode.slice(finalizerStart), /finalizer\.dispatchWorkgroups\(1\)/);
  const materializerStart = source.indexOf("private encodeInspectionPublication");
  const materializerEnd = source.indexOf("readResidencyStats", materializerStart);
  const materializer = source.slice(materializerStart, materializerEnd);
  assert.equal((materializer.match(/copyBufferToBuffer/g) ?? []).length, 2);
  assert.equal((materializer.match(/dispatchWorkgroups/g) ?? []).length, 2);
  assert.match(materializer, /Publish octree raw voxel records/);
  assert.match(materializer, /Publish octree sparse brick records/);
});

test("renderer lazily attaches inspection before solver submission and services paused repaints", () => {
  const source = readFileSync(new URL("../lib/webgpu-renderer.ts", import.meta.url), "utf8");
  const modeControl = source.indexOf("this.voxelInspectionSource = requestedVoxelDebugGeneration >= 0");
  const solverSubmit = source.indexOf("this.submitPreparedGPUFluid(readyGPUFluid", modeControl);
  const encoderCreation = source.indexOf("createCommandEncoder({ label: \"Fluid Lab frame\" })", solverSubmit);
  const pausedPublication = source.indexOf("this.voxelInspectionSource?.inspectionPublication?.encodePending?.(encoder)", encoderCreation);
  assert.ok(modeControl >= 0 && solverSubmit > modeControl, "inspection must attach before solver encode");
  assert.ok(encoderCreation > solverSubmit && pausedPublication > encoderCreation, "render encode services a pending paused repaint");
  const solverAttachment = source.slice(source.indexOf("this.gpuFluidPending=create.then"), modeControl);
  assert.doesNotMatch(solverAttachment, /solver\.sparseVoxelRenderSource/, "smooth solver attachment must not activate expanded records");
  assert.match(solverAttachment, /const sparseSceneSource=solver\.sparseVoxelSceneSource/);
});

test("octree keeps the structural scene source eager and inspection records lazy", () => {
  const source = readFileSync(new URL("../lib/webgpu-octree-sparse-bricks.ts", import.meta.url), "utf8");
  const constructorStart = source.indexOf("constructor(device: GPUDevice");
  const lazyStart = source.indexOf("ensureInspectionSource()", constructorStart);
  const encodeStart = source.indexOf("encode(encoder: GPUCommandEncoder", lazyStart);
  const constructor = source.slice(constructorStart, lazyStart);
  const lazy = source.slice(lazyStart, encodeStart);
  assert.match(constructor, /this\.sceneSource = \{/);
  assert.doesNotMatch(constructor, /Sparse voxel debug records|Materialize sparse voxel records/);
  assert.match(lazy, /Sparse voxel debug records/);
  assert.match(lazy, /Materialize sparse voxel records/);
  assert.match(lazy, /allocatedBytes: buffers\.reduce/);
});
