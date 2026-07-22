import assert from "node:assert/strict";
import test from "node:test";

import {
  advectPowerFaceFromOldVector,
  interpolateRegularAxisFaceComponent,
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
import { WebGPUOctreeProjection } from "../lib/webgpu-octree";

const compact = (value: unknown) => String(value).replace(/\s+/g, "");

test("old-mesh velocity plan retains headers, topology metrics, full vectors, and site directory", () => {
  const plan = planOctreePowerOldMeshAdvection(128, 512, 256);
  assert.equal(plan.headerBytes, 128 * 48);
  assert.equal(plan.metricOffsetBytes, plan.headerBytes);
  assert.equal(plan.velocityOffsetBytes, plan.headerBytes + 128 * 16);
  assert.equal(plan.axisControlOffsetBytes, plan.headerBytes + 128 * 32);
  assert.equal(plan.axisFaceOffsetBytes, plan.axisControlOffsetBytes + 32);
  assert.equal(plan.axisFaceCapacity, 512);
  assert.equal(plan.axisHashCapacity, 1024);
  assert.equal(plan.axisHashOffsetBytes, plan.axisFaceOffsetBytes + 512 * 32);
  assert.equal(plan.arenaBytes, plan.axisHashOffsetBytes + 1024 * 4);
  assert.equal(plan.siteBytes, 256 * 16);
});

test("old-mesh parameter upload follows the WGSL vec3 uniform ABI", () => {
  assert.equal(OCTREE_POWER_OLD_MESH_ADVECTION_PARAMETER_BYTES, 80);
  const data = packOctreePowerOldMeshAdvectionParameters({
    rowCapacity: 101, faceCapacity: 202, siteHashCapacity: 256,
    metricOffsetWords: 303, velocityOffsetWords: 404,
    dimensions: [16, 32, 64], maximumLeafSize: 8, generation: 7,
    physicalCellSize: 0.125, timestep: 0.025,
    axisControlOffsetWords: 505, axisFaceOffsetWords: 606, axisHashOffsetWords: 707,
  });
  assert.equal(data.byteLength, 80);
  const words = new Uint32Array(data), floats = new Float32Array(data);
  assert.deepEqual(Array.from(words.slice(0, 5)), [101, 202, 256, 303, 404]);
  assert.deepEqual(Array.from(words.slice(5, 8)), [0, 0, 0], "vec3 alignment padding is explicit");
  assert.deepEqual(Array.from(words.slice(8, 13)), [16, 32, 64, 8, 7]);
  assert.equal(floats[13], 0.125); assert.ok(Math.abs(floats[14] - 0.025) < 1e-7);
  assert.deepEqual(Array.from(words.slice(16, 19)), [505, 606, 707]);
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

test("regular staggered interpolation is an exact zero-dt identity at a face centroid", () => {
  const target = 17.25;
  const observed: string[] = [];
  const value = interpolateRegularAxisFaceComponent([4, 5.5, 6.5], [4, 5, 6], 1, 0,
    (origin, axis, size) => {
      observed.push(`${origin.join(",")}/${axis}/${size}`);
      return origin[0] === 4 && origin[1] === 5 && origin[2] === 6 ? target : -900;
    });
  assert.equal(value, target);
  assert.deepEqual(observed, ["4,5,6/0/1"],
    "zero displacement must read the exact staggered face, not neighboring cell-centre averages");
});

test("regular staggered interpolation uses bounded higher-order support away from a face centroid", () => {
  const value = interpolateRegularAxisFaceComponent([4.25, 4.5, 4.5], [4, 4, 4], 1, 0,
    (origin) => origin[0] ** 2);
  assert.ok(value !== undefined);
  assert.ok(Math.abs(value - 4.25 ** 2) < 1e-10,
    `Catmull-Rom should reproduce this quadratic profile; observed ${value}`);
  assert.notEqual(value, 0.75 * 4 ** 2 + 0.25 * 5 ** 2,
    "the recurrent sampler must not silently fall back to the first-order low-pass value");
});

test("old-mesh GPU authority is generation-coherent and uses cube/catalog interpolation", () => {
  assert.match(octreePowerOldMeshCaptureWGSL,
    /velocityControl\[0\]!=VALID.*velocityControl\[2\]!=rows.*velocityControl\[5\]!=rows.*velocityControl\[7\]!=generation/,
    "only a complete full-vector publication from the same old generation may be captured");
  assert.match(octreePowerOldMeshCaptureWGSL,
    /indexOldAxisFaces.*axisHash\(face\).*atomicCompareExchangeWeak/s,
    "the same old generation retains a bounded spatial index of staggered regular faces");
  assert.match(octreePowerOldMeshAdvectionWGSL,
    /oldGeneration\+1u!=p\.generation.*fail\(GENERATION,0u\)/,
    "a missing or stale old generation must fail closed");
  assert.match(octreePowerOldMeshAdvectionWGSL,
    /if\(\(th\.flags&1u\)!=0u\).*regularAxisComponent\(x,h,0u\).*regularAxisComponent\(x,h,1u\).*regularAxisComponent\(x,h,2u\)/s,
    "ordinary regions use the paper's staggered per-axis interpolant");
  assert.doesNotMatch(octreePowerOldMeshAdvectionWGSL,
    /if\(\(th\.flags&1u\)!=0u\).*result\+=w\*v\.xyz/s,
    "ordinary regions must not reconstruct through the dissipative cell-centred round trip");
  assert.match(octreePowerOldMeshAdvectionWGSL,
    /cubicWeight.*stencil<64u.*clamp\(cubic,minimum,maximum\)/s,
    "regular advection uses a bounded higher-order tensor reconstruction instead of repeated trilinear diffusion");
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
  assert.doesNotMatch(octreePowerOldMeshAdvectionWGSL, /faceKey|homologous|applyExactPowerTransfer/,
    "the authoritative recurrent path has no same-face identity transfer masquerading as advection");
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
    /control\.p1=atomicLoad\(&control\.flags\);control\.p2=atomicLoad\(&control\.firstError\);control\.p3=atomicLoad\(&control\.advected\);control\.p4=atomicLoad\(&control\.valid\)/);
});

test("projection captures old vectors after projection and consumes them on the next rebuild", () => {
  const rebuild = compact((WebGPUOctreeProjection.prototype as unknown as {
    encodePowerAssemblyMirror: (encoder: GPUCommandEncoder) => void;
  }).encodePowerAssemblyMirror);
  assert.match(rebuild, /this\.powerFaceSeed\?\.encode\(encoder\).*this\.powerFaceAdvection\?\.encodeAdvect\(encoder,/,
    "generation one may use the cold seed, but every recurring face is overwritten from the old mesh");
  const publication = compact((WebGPUOctreeProjection.prototype as unknown as {
    encodePowerVelocityPublication: (encoder: GPUCommandEncoder) => void;
  }).encodePowerVelocityPublication);
  assert.match(publication,
    /this\.powerVelocity\.encodeFromFaceControl\(encoder,.*this\.powerFaceAdvection\?\.encodeCapture\(encoder,/,
    "the projected full cell vectors are captured before a subsequent topology can overwrite their mesh");
  assert.ok(publication.indexOf("this.powerFaceSeed.encodePowerToAxis")
    < publication.indexOf("this.powerFaceAdvection?.encodeCapture"),
  "the projected staggered field must publish before the old generation snapshot");
  assert.doesNotMatch(rebuild, /powerFaceTransfer\?\.encodeApply/,
    "exact generalized-face identity transfer is not an authoritative rebuild step");
  assert.match(compact(WebGPUOctreePowerFaceAdvection.prototype.encodeAdvect),
    /preparePipeline.*advectPipeline.*finalizePipeline/,
    "the global commit gate is prepared and finalized in dependency order");
  assert.equal(compact(WebGPUOctreePowerFaceAdvection.prototype.encodeAdvect)
    .match(/encoder\.beginComputePass/g)?.length, 3,
  "prepare, bulk advection, and publication use separate storage-dependency passes");
  assert.deepEqual(OCTREE_POWER_OLD_MESH_PREPARE_BINDINGS, [0, 1, 4, 5, 6, 7, 8]);
  assert.deepEqual(OCTREE_POWER_OLD_MESH_ADVECT_BINDINGS, [0, 1, 2, 3, 5, 6, 7, 8, 10, 11, 12]);
  assert.deepEqual(OCTREE_POWER_OLD_MESH_FINALIZE_BINDINGS, [1, 8]);
  assert.match(compact(WebGPUOctreePowerFaceAdvection.prototype.encodeAdvect),
    /group\(this\.preparePipeline,OCTREE_POWER_OLD_MESH_PREPARE_BINDINGS\).*group\(this\.advectPipeline,OCTREE_POWER_OLD_MESH_ADVECT_BINDINGS\).*group\(this\.finalizePipeline,OCTREE_POWER_OLD_MESH_FINALIZE_BINDINGS\)/,
    "every auto-layout entry point receives its explicit live binding ABI, including seed binding 8");
});

test("Section 5 retains the extrapolated old full-vector mesh for the next topology", () => {
  const source = compact(WebGPUOctreeProjection.prototype["encodeGlobalFineFaceBandPhase"]);
  const publication = source.indexOf('phase!=="power-publication"');
  const reverse = source.indexOf("this.powerFaceSeed.encodePowerToAxis", publication);
  const reconstruct = source.indexOf("this.powerVelocity.encodeFromFaceControl", reverse);
  const capture = source.indexOf("this.powerFaceAdvection?.encodeCapture", reconstruct);
  assert.ok(publication >= 0 && reverse > publication && reconstruct > reverse && capture > reconstruct,
    "the committed face-marched field must become the retained old interpolation mesh");
});
