import assert from "node:assert/strict";
import test from "node:test";

import {
  advectPowerFaceFromOldVector,
  OCTREE_POWER_OLD_MESH_CAPTURE_BINDINGS,
  OCTREE_POWER_OLD_MESH_ADVECT_BINDINGS,
  OCTREE_POWER_OLD_MESH_FINALIZE_BINDINGS,
  OCTREE_POWER_OLD_MESH_ADVECTION_PARAMETER_BYTES,
  OCTREE_POWER_OLD_MESH_PREPARE_BINDINGS,
  octreePowerOldMeshAdvectionWGSL,
  octreePowerOldMeshCaptureWGSL,
  packOctreePowerOldMeshAdvectionParameters,
  planOctreePowerOldMeshAdvection,
  WebGPUOctreePowerFaceAdvection,
} from "../lib/webgpu-octree-power-face-advection";
import {
  OCTREE_POWER_FACE_DELTA_CARRIED,
  octreePowerFaceShader,
} from "../lib/webgpu-octree-power-faces";
import { octreePowerFaceSeedShader } from "../lib/webgpu-octree-power-face-seed";
import { WebGPUOctreeProjection } from "../lib/webgpu-octree";
import type { PassBroker } from "../lib/webgpu-pass-broker";

const compact = (value: unknown) => String(value).replace(/\s+/g, "");

test("old-mesh velocity plan retains headers, topology metrics, full vectors, and the sorted row directory", () => {
  const plan = planOctreePowerOldMeshAdvection(128, 512, 256);
  assert.equal(plan.headerBytes, 128 * 48);
  assert.equal(plan.metricOffsetBytes, plan.headerBytes);
  assert.equal(plan.velocityOffsetBytes, plan.headerBytes + 128 * 16);
  assert.equal(plan.rowDirectoryOffsetBytes, plan.headerBytes + 128 * 32);
  assert.equal(plan.rowDirectoryBytes, 256 * 16);
  assert.equal(plan.arenaBytes, plan.rowDirectoryOffsetBytes + plan.rowDirectoryBytes);
  assert.equal(plan.allocatedBytes, plan.arenaBytes + 64 + 80);
});

test("old-mesh parameter upload follows the WGSL vec3 uniform ABI", () => {
  assert.equal(OCTREE_POWER_OLD_MESH_ADVECTION_PARAMETER_BYTES, 80);
  const data = packOctreePowerOldMeshAdvectionParameters({
    rowCapacity: 101, faceCapacity: 202, rowDirectoryCapacity: 256,
    metricOffsetWords: 303, velocityOffsetWords: 404,
    dimensions: [16, 32, 64], maximumLeafSize: 8, generation: 7,
    physicalCellSize: 0.125, timestep: 0.025,
    rowDirectoryOffsetWords: 505,
  });
  assert.equal(data.byteLength, 80);
  const words = new Uint32Array(data), floats = new Float32Array(data);
  assert.deepEqual(Array.from(words.slice(0, 5)), [101, 202, 256, 303, 404]);
  assert.deepEqual(Array.from(words.slice(5, 8)), [0, 0, 0], "vec3 alignment padding is explicit");
  assert.deepEqual(Array.from(words.slice(8, 13)), [16, 32, 64, 8, 7]);
  assert.equal(floats[13], 0.125); assert.ok(Math.abs(floats[14] - 0.025) < 1e-7);
  assert.equal(words[15], 0, "the deleted optional-completion switch is explicit ABI padding");
  assert.deepEqual(Array.from(words.slice(16, 19)), [505, 0, 0]);
});

test("Section 5 CPU oracle backtraces a full vector and projects only at the new face", () => {
  const calls: readonly number[][] = [];
  const observed: number[][] = calls as number[][];
  const value = advectPowerFaceFromOldVector([2, 3, 4], [0, 1, 0], 0.5, (point) => {
    observed.push([...point]);
    return [1, point[0], -2];
  });
  assert.equal(calls.length, 3);
  assert.deepEqual(calls[0], [2, 3, 4]);
  assert.deepEqual(calls[1], [1.75, 2.5, 4.5]);
  assert.deepEqual(calls[2], [1.5, 2.125, 5]);
  assert.equal(value, 1.5);
});

test("old-mesh GPU authority is generation-coherent and uses only the native transition catalog", () => {
  const functions = Array.from(octreePowerOldMeshAdvectionWGSL.matchAll(/\bfn\s+(\w+)/g),
    match => match[1]);
  assert.equal(new Set(functions).size, functions.length,
    "the combined multi-entrypoint module must not redeclare a WGSL function");
  assert.match(octreePowerOldMeshCaptureWGSL,
    /velocityControl\[0\]!=VALID[\s\S]*velocityControl\[2\]!=rows[\s\S]*velocityControl\[5\]!=rows[\s\S]*velocityControl\[7\]!=generation/,
    "only a complete full-vector publication from the same old generation may be captured");
  assert.doesNotMatch(octreePowerOldMeshCaptureWGSL, /indexOldAxisFaces|axisHash|atomicCompareExchangeWeak/,
    "the old mesh must not rebuild a second Cartesian-face hash");
  assert.match(octreePowerOldMeshAdvectionWGSL,
    /oldGeneration\+1u!=p\.generation.*fail\(GENERATION,0u\)/,
    "a missing or stale old generation must fail closed");
  assert.match(octreePowerOldMeshAdvectionWGSL,
    /let row=owner\(x\);if\(row==INVALID.*\{return bad\(1\.\);\}/s,
    "a new-face miss stays pending for mandatory Section 5 regular-face completion");
  assert.match(octreePowerOldMeshAdvectionWGSL,
    /if\(\(th\.flags&1u\)!=0u\)\{return bad\(13\.\);\}/,
    "regular-region samples stay pending for mandatory Section 5 regular-face completion");
  assert.doesNotMatch(octreePowerOldMeshAdvectionWGSL,
    /AxisFace|axisPublication|axisFace|findAxis|regularAxisComponent|regularVector|cubicWeight/,
    "the old Cartesian mirror and its interpolation fallback must be absent");
  assert.match(octreePowerOldMeshAdvectionWGSL,
    /for\(var local=0u;local<th\.count;local\+=1u\).*let w=weights\(point/s,
    "transition regions use the old catalog Delaunay tetrahedra");
  assert.match(octreePowerOldMeshAdvectionWGSL,
    /let v0=sampleOldIncident\(x,n\).*let vm=sampleOldIncident\(x-\.5\*p\.dt\*v0\.xyz,n\).*let va=sampleOldIncident\(x-p\.dt\*vm\.xyz,n\).*let value=dot\(va\.xyz,n\)/s,
    "each new centroid follows a midpoint characteristic through old full vectors before normal projection");
  assert.match(octreePowerOldMeshAdvectionWGSL, /const STATUS_VALID=0x3f800000u/,
    "float-backed scratch status uses finite 1.0 rather than a lossy negative-zero authority bit");
  assert.match(octreePowerOldMeshAdvectionWGSL, /storeStatus\(i,STATUS_VALID\)/,
    "every successfully traced face publishes the stable scratch status");
  assert.match(octreePowerOldMeshAdvectionWGSL,
    /if\(\(f\.flags&DELTA_CARRIED\)!=0u\)[\s\S]*storeStatus\(i,STATUS_VALID\);return;[\s\S]*sampleOldIncident/,
    "an exact carried face preserves its old velocity and returns before characteristic tracing");
  assert.doesNotMatch(compact(WebGPUOctreePowerFaceAdvection),
    /OctreeFaceMirrorSource|previousFacePublication|oldAxisFaces/,
    "the native old power mesh constructor has no Cartesian publication dependency");
});

test("old-mesh capture copies only the device-published live row prefix", () => {
  const capture = compact(WebGPUOctreePowerFaceAdvection.prototype.encodeCapture);
  assert.match(capture, /input\.leafHeaders.*topology\.metrics.*input\.rowVelocities.*faces\.rowDirectory/s,
    "the old full-vector interpolation mesh and canonical row ownership remain unchanged");
  assert.doesNotMatch(capture,
    /copyBufferToBuffer|plan\.headerBytes|plan\.metricOffsetBytes|plan\.velocityOffsetBytes|plan\.rowDirectoryBytes/,
    "capacity-sized row snapshots must stay deleted");
  assert.match(octreePowerOldMeshCaptureWGSL,
    /@compute @workgroup_size\(256\)fn captureOldMeshAuthority[\s\S]*for\(var row=lid;row<rows;row\+=256u\)/,
    "one deterministic workgroup copies only the exact live rows");
  assert.doesNotMatch(octreePowerOldMeshCaptureWGSL, /atomic/,
    "capture publication has one deterministic owner and needs no atomics");
  assert.deepEqual(OCTREE_POWER_OLD_MESH_CAPTURE_BINDINGS, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
  const plan = planOctreePowerOldMeshAdvection(4_096, 98_304, 8_192);
  assert.equal(plan.arenaBytes, 4_096 * (48 + 16 + 16) + 8_192 * 16,
    "the old arena contains only headers, metrics, row vectors, and the existing sorted row directory");
});

test("old-mesh interpolation admits an exact liquid-air dual boundary only through its incident element", () => {
  const source = compact(octreePowerOldMeshAdvectionWGSL);
  assert.match(source, /fnsampleOldIncident\(x:vec3f,n:vec3f\)->vec4f/);
  assert.match(source, /direct\.w!=-1\./,
    "only an absent half-open owner may invoke the incident-element resolution");
  assert.match(source, /sampleOld\(x-epsilon\*n\)/,
    "the negative-row/liquid-side dual element must be selected geometrically");
  assert.doesNotMatch(source, /sampleOld\(x\+epsilon\*n\)/,
    "the air side must not become a nearest-neighbor fallback");
});

test("recurrent advection retains its attempt verdict after the following snapshot capture", () => {
  assert.match(octreePowerOldMeshAdvectionWGSL,
    /control\.p1=control\.flags;control\.p2=control\.firstError;control\.p3=control\.advected;control\.p4=control\.valid/);
  assert.doesNotMatch(octreePowerOldMeshCaptureWGSL, /snapshot\[(?:9|10|11|12)\]=/,
    "live-prefix capture must not overwrite the preceding advection verdict");
});

test("Section 5 captures only the final extrapolated vectors for the next rebuild", () => {
  const rebuild = compact((WebGPUOctreeProjection.prototype as unknown as {
    encodeNativePowerAssembly: (encoder: GPUCommandEncoder) => void;
  }).encodeNativePowerAssembly);
  assert.match(rebuild, /seed\.encode\(broker\).*advection\.encodeAdvect\(broker,/,
    "the explicit generation-one seed transaction precedes exact recurrent delta advection");
  const publication = compact((WebGPUOctreeProjection.prototype as unknown as {
    encodePowerVelocityPublication: (broker: PassBroker) => void;
  }).encodePowerVelocityPublication);
  assert.match(publication, /this\.powerVelocity\.encodeFromFaceControl\(broker,/,
    "the projected native power-face field must reconstruct the retained old cell-vector mesh");
  assert.doesNotMatch(publication, /encodePowerToAxis|faceMirror|axis/,
    "publication must not rebuild a Cartesian velocity authority");
  assert.doesNotMatch(publication, /encodeCapture/,
    "the pre-extrapolation snapshot is overwritten in the same command stream and must not be recorded");
  assert.doesNotMatch(rebuild, /powerFaceTransfer\?\.encodeApply/,
    "exact generalized-face identity transfer is not an authoritative rebuild step");
  assert.match(compact(WebGPUOctreePowerFaceAdvection.prototype.encodeAdvect),
    /preparePipeline.*advectPipeline.*finalizePipeline/,
    "the global commit gate is prepared and finalized in dependency order");
  assert.doesNotMatch(compact(WebGPUOctreePowerFaceAdvection.prototype.encodeAdvect), /beginComputePass/,
    "prepare, bulk advection, and publication must be broker-owned");
  assert.match(compact(WebGPUOctreePowerFaceAdvection.prototype.encodeAdvect),
    /encodeAdvect\(broker,input\)[\s\S]*prepare=broker\.compute[\s\S]*advect=broker\.compute[\s\S]*finalize=broker\.compute/,
    "prepare, bulk advection, and publication use the caller-owned compute pass");
  assert.doesNotMatch(compact(WebGPUOctreePowerFaceAdvection.prototype.encodeAdvect), /newPassBroker|broker\.fence/,
    "the advection helper must not retain a local-broker or routine-fence fallback");
  assert.deepEqual(OCTREE_POWER_OLD_MESH_PREPARE_BINDINGS, [0, 1, 4, 5, 6, 7, 8, 9]);
  assert.deepEqual(OCTREE_POWER_OLD_MESH_ADVECT_BINDINGS, [0, 1, 2, 5, 6, 7, 10, 11, 12]);
  assert.deepEqual(OCTREE_POWER_OLD_MESH_FINALIZE_BINDINGS, [1, 7, 8]);
  assert.match(compact(WebGPUOctreePowerFaceAdvection.prototype.encodeAdvect),
    /group\(this\.preparePipeline,OCTREE_POWER_OLD_MESH_PREPARE_BINDINGS\).*group\(this\.advectPipeline,OCTREE_POWER_OLD_MESH_ADVECT_BINDINGS\).*group\(this\.finalizePipeline,OCTREE_POWER_OLD_MESH_FINALIZE_BINDINGS\)/,
    "every auto-layout entry point receives its explicit live binding ABI, including seed binding 8");
});

test("face advection consumes exact identity carry with no capacity tail or atomic tally", () => {
  const encode = compact(WebGPUOctreePowerFaceAdvection.prototype.encodeAdvect);
  assert.match(encode, /dispatchWorkgroupsIndirect\(this\.faces\.liveFaceDispatch,0\)/,
    "the face publication's exact live dispatch must replace the face-capacity dispatch");
  assert.match(encode, /if\(input\.generation===1\)/,
    "generation-one initialization publishes directly before recurring completion exists");
  assert.doesNotMatch(encode, /deferInterpolationFailures|deferFailures/,
    "recurring regular-face completion is mandatory rather than an optional fallback switch");
  assert.doesNotMatch(encode, /Math\.ceil\(this\.plan\.faceCapacity|clearBuffer/,
    "no capacity dispatch or tail clear may survive");
  assert.match(octreePowerFaceShader,
    new RegExp(`current\\\\.normalVelocity=oldFace\\\\.normalVelocity;current\\\\.flags\\\\|=FACE_DELTA_CARRIED`),
    "the sorted identity merge must preserve the old scalar before committing the new A/B publication");
  assert.match(octreePowerFaceSeedShader,
    /let carried=\(face\.flags&DELTA_CARRIED\)!=0u;[\s\S]*select\(0\.0,face\.normalVelocity,carried\)/,
    "the native seed must preserve an exact carried face in its transaction scratch");
  assert.equal(OCTREE_POWER_FACE_DELTA_CARRIED, 1 << 28);
  assert.match(octreePowerOldMeshAdvectionWGSL,
    /if\(p\.generation==1u\)[\s\S]*faceDelta\[0\]!=0u[\s\S]*control\.mode=INITIALIZE/,
    "generation one must be an explicit all-new seed transaction");
  assert.doesNotMatch(octreePowerOldMeshAdvectionWGSL,
    /coldFallback|atomic(?:Add|Min|Or|Store|Load)|array<atomic<u32>>/,
    "legacy cold fallback and synchronization/count atomics must stay deleted");
  assert.match(octreePowerOldMeshAdvectionWGSL,
    /var<workgroup>advectionReduce[\s\S]*workgroupBarrier\(\)[\s\S]*control\.advected=result\.z/,
    "one deterministic reduction must own completion and failure publication");
});

test("Section 5 retains the extrapolated old full-vector mesh for the next topology", () => {
  const source = compact(WebGPUOctreeProjection.prototype["encodeGlobalFineFaceBandPhase"]);
  const publication = source.indexOf('phase!=="power-publication"');
  const reconstruct = source.indexOf("this.powerVelocity.encodeFromFaceControl", publication);
  const capture = source.indexOf("this.powerFaceAdvection.encodeCapture", reconstruct);
  assert.ok(publication >= 0 && reconstruct > publication && capture > reconstruct,
    "the committed closest-point-extended field must become the retained old interpolation mesh");
  assert.doesNotMatch(source, /encodePowerToAxis|faceMirror|axis/,
    "the next topology must retain only native power authority");
  assert.equal(source.match(/powerFaceAdvection\.encodeCapture/g)?.length, 1,
    "exactly one old interpolation snapshot must be retained per advance");
});
