import assert from "node:assert/strict";
import test from "node:test";
import {
  OCTREE_POWER_FACE_TRANSFER_CONTROL_BYTES,
  OCTREE_POWER_FACE_TRANSFER_DISPATCH_OFFSET_BYTES,
  octreePowerFaceTransferShader,
  planOctreePowerFaceTransfer,
  WebGPUOctreePowerFaceTransfer,
} from "../lib/webgpu-octree-power-face-transfer";
import { WebGPUOctreeProjection } from "../lib/webgpu-octree";

test("power-face transfer plans a fixed 128-bit radix generation", () => {
  const plan = planOctreePowerFaceTransfer(1_000);
  assert.equal(plan.sortCapacity, 1_024);
  assert.equal(plan.blockCount, 4);
  assert.equal(plan.previousBytes, 32_000);
  assert.equal(plan.indexBytes, 4_096);
  assert.equal(plan.histogramBytes, 256);
  assert.equal(plan.allocatedBytes, 32_000 + 2 * 4_096 + 256 + 32 * 256
    + OCTREE_POWER_FACE_TRANSFER_CONTROL_BYTES + 12);
  assert.equal(OCTREE_POWER_FACE_TRANSFER_DISPATCH_OFFSET_BYTES, 32);
  assert.throws(() => planOctreePowerFaceTransfer(0), /positive/);
});

test("power-face transfer uses ordered-site keys and deterministic radix order", () => {
  assert.match(octreePowerFaceTransferShader, /fn faceKey/);
  assert.match(octreePowerFaceTransferShader, /metadata=ae\|\(be<<6u\)/);
  assert.match(octreePowerFaceTransferShader, /fn powerRadixHistogram/);
  assert.match(octreePowerFaceTransferShader, /fn powerRadixPrefix/);
  assert.match(octreePowerFaceTransferShader, /fn powerRadixScatter/);
  assert.match(octreePowerFaceTransferShader, /atomicStore\(&control\[8\],\(count\+255u\)\/256u\)/,
    "the GPU face count must publish the live radix dispatch");
  assert.match(octreePowerFaceTransferShader,
    /let participating=id\.x<atomicLoad\(&control\[0\]\)[\s\S]*workgroupBarrier\(\)[\s\S]*if\(participating\)/,
    "partial scatter workgroups must participate in barriers but suppress padded writes");
  assert.match(octreePowerFaceTransferShader, /fn finalizePowerKeys/);
  assert.match(octreePowerFaceTransferShader, /f\.normalVelocity=orientation\(f\)\*previous\[old\]\.normalVelocity/);
  assert.doesNotMatch(octreePowerFaceTransferShader, /texture_/,
    "generalized transfer must never read a retired dense velocity texture");
});

test("pressure assembly keeps the current transported axis seed instead of stale generalized DOFs", () => {
  const assembly = (WebGPUOctreeProjection.prototype as unknown as { encodePowerAssemblyMirror?: () => void })
    .encodePowerAssemblyMirror?.toString?.() ?? WebGPUOctreeProjection.toString();
  assert.match(assembly, /powerFaceSeed/);
  assert.doesNotMatch(assembly, /powerFaceTransfer\?\.encodeApply/,
    "previous-frame projected faces must not erase current transport or body forces before pressure assembly");
  assert.equal(typeof WebGPUOctreePowerFaceTransfer.prototype.encodeApply, "function");
  assert.equal(typeof WebGPUOctreePowerFaceTransfer.prototype.encodeCapture, "function");
  assert.match(WebGPUOctreePowerFaceTransfer.prototype.encodeCapture.toString(),
    /copyBufferToBuffer\(this\.control,OCTREE_POWER_FACE_TRANSFER_DISPATCH_OFFSET_BYTES,this\.dispatch,0,12\)/,
    "the live GPU-authored dispatch must be staged before indirect use");
  assert.match(WebGPUOctreePowerFaceTransfer.prototype.encodeCapture.toString(),
    /dispatchWorkgroupsIndirect\(this\.dispatch,0\)/,
    "all radix stages must consume the staged indirect dispatch without a count readback");
  assert.doesNotMatch(WebGPUOctreePowerFaceTransfer.prototype.encodeCapture.toString(),
    /dispatchWorkgroupsIndirect\(this\.control/,
    "a writable storage buffer cannot also be indirect in the same synchronization scope");
});
