import assert from "node:assert/strict";
import test from "node:test";
import {
  OCTREE_GPU_FACE_CANDIDATES_PER_ROW,
  OCTREE_GPU_FACE_INCIDENCE_PER_ROW,
  OCTREE_GPU_FACE_RECORD_BYTES,
  octreeFaceMirrorShader,
  planOctreeFaceMirror,
  WebGPUOctreeFaceMirror,
} from "../lib/webgpu-octree-face-mirror";
import { adaptiveFaceRhsIsSupported, WebGPUOctreeProjection } from "../lib/webgpu-octree";
import { WebGPUUniformEulerianSolver } from "../lib/webgpu-uniform-eulerian";

test("octree face mirror has a compact bounded ABI", () => {
  const plan = planOctreeFaceMirror(1024);
  assert.equal(OCTREE_GPU_FACE_RECORD_BYTES, 32);
  assert.equal(OCTREE_GPU_FACE_INCIDENCE_PER_ROW, 48);
  assert.equal(OCTREE_GPU_FACE_CANDIDATES_PER_ROW, 24);
  assert.equal(plan.faceCapacity, 24_576);
  assert.equal(plan.faceBytes, plan.faceCapacity * OCTREE_GPU_FACE_RECORD_BYTES);
  assert.equal(plan.incidenceBytes, 1024 * 49 * 4);
  assert.equal(plan.offsetBytes, 1025 * 4);
  assert.equal(plan.allocatedBytes, 84 + plan.faceBytes + plan.incidenceBytes + plan.offsetBytes);
  assert.throws(() => planOctreeFaceMirror(0), /positive/);
});

test("adaptive face RHS authority fails closed around unsupported boundary operators", () => {
  assert.equal(adaptiveFaceRhsIsSupported(true, false, 0, false), true);
  assert.equal(adaptiveFaceRhsIsSupported(false, false, 0, false), false);
  assert.equal(adaptiveFaceRhsIsSupported(true, true, 0, false), false);
  assert.equal(adaptiveFaceRhsIsSupported(true, true, 0, false, false, true), true,
    "terrain face authority requires its embedded-boundary publication");
  assert.equal(adaptiveFaceRhsIsSupported(true, false, 1, false), false);
  assert.equal(adaptiveFaceRhsIsSupported(true, false, 1, false, true), true);
  assert.equal(adaptiveFaceRhsIsSupported(true, false, 0, true), false);
  const construction = WebGPUOctreeProjection.toString();
  assert.match(construction, /const faceTransportEnabled\s*=\s*faceTransportRequested\s*&&\s*this\.faceRhsAuthority/);
  assert.match(construction, /options\.faceVelocityMirror\s*\|\|\s*options\.faceVelocityRhs\s*\|\|\s*faceTransportEnabled/,
    "a default transport request must not allocate the compact face store when authority is unsupported");
});

test("octree solver construction forwards adaptive face migration flags", () => {
  const construction = WebGPUUniformEulerianSolver.toString();
  assert.match(construction, /faceVelocityMirror:\s*options\.octree\.faceVelocityMirror/);
  assert.match(construction, /faceVelocityRhs:\s*options\.octree\.faceVelocityRhs/);
});

test("GPU face store deterministically publishes canonical orientation and signed incidence", () => {
  assert.match(octreeFaceMirrorShader, /neighborLiquid=frontierRow\(neighbor\)!=INVALID/,
    "frontier membership, not representation-dependent phi sign, identifies internal liquid faces");
  assert.match(octreeFaceMirrorShader, /!neighborLiquid\|\|side>0/,
    "internal faces must publish only from their spatially negative leaf");
  assert.match(octreeFaceMirrorShader, /negativeRow: u32, positiveRow: u32/);
  assert.match(octreeFaceMirrorShader, /originX: u32, originY: u32, originZ: u32/);
  assert.doesNotMatch(octreeFaceMirrorShader, /face\.packedOrigin/);
  assert.match(octreeFaceMirrorShader, /appendIncidence\(negative,faceIndex\);appendIncidence\(positive,faceIndex\)/);
  assert.match(octreeFaceMirrorShader, /fn countFaces/);
  assert.match(octreeFaceMirrorShader, /fn scanFaceCounts/);
  assert.match(octreeFaceMirrorShader, /faceIndex=faceOffsets\[row\]\+local/, "face IDs must not depend on atomic publication order");
  assert.doesNotMatch(octreeFaceMirrorShader, /faceIndex=atomicAdd\(&control\[0\]/);
  assert.match(octreeFaceMirrorShader, /sorted:array<u32,48>/, "incidence reduction must be stable despite parallel append order");
  assert.match(octreeFaceMirrorShader, /atomicStore\(&control\[1\],1u\)/, "capacity and incidence overflow must fail closed");
  assert.match(octreeFaceMirrorShader, /fn reduceRhsParity/);
  assert.match(octreeFaceMirrorShader, /fn applyFaceRhs/);
  assert.match(octreeFaceMirrorShader, /fn reduceProjectedDivergence/);
  assert.match(octreeFaceMirrorShader, /divergence=faceRhs\(row\)\/max\(volume,1e-30\)/);
  assert.match(octreeFaceMirrorShader, /atomicAddFloat\(&parity\[11\],divergence\*divergence\)/);
  assert.match(octreeFaceMirrorShader, /rhs\+=sign\*face\.area\*face\.normalVelocity/);
  assert.match(octreeFaceMirrorShader, /difference>max\(1e-5,abs\(reference\)\*1e-5\)/);
  const mirrorEncode = `${WebGPUOctreeFaceMirror.prototype.encodeTopology.toString()}${WebGPUOctreeFaceMirror.prototype.encodeRhs.toString()}`.replace(/\s+/g, "");
  assert.match(mirrorEncode, /dispatchWorkgroupsIndirect\(rowDispatch,0\)/);
  assert.match(mirrorEncode, /this\.parityPipeline/);
  assert.match(mirrorEncode, /if\(applyRhs\)/);
  assert.match(mirrorEncode, /this\.applyRhsPipeline/);
  const solve = WebGPUOctreeProjection.prototype.encode.toString().replace(/\s+/g, "");
  assert.match(solve, /this\.faceMirror\.encodeTopology\(encoder,this\.solveDispatch\)/);
  assert.match(solve, /this\.faceMirror\.encodeRhs\(encoder,this\.solveDispatch,this\.faceRhsAuthority\)/);
  assert.ok(solve.indexOf("this.faceMirror.encodeTopology") > solve.indexOf("pressure.end()"));
  assert.ok(solve.indexOf("this.faceMirror.encodeRhs") < solve.indexOf("this.iterateChebyshevPipeline"));
  assert.match(solve, /if\(this\.leafSolver==="dense"\)this\.faceMirror\?\.encode\(encoder,this\.solveDispatch,false\)/);
  assert.match(Object.getOwnPropertyDescriptor(WebGPUOctreeProjection.prototype, "adaptiveFaceMirrorSource")?.get?.toString() ?? "", /faceMirror/);
});

test("face classification switches from dense bootstrap to authoritative surface pages", () => {
  assert.match(octreeFaceMirrorShader, /fn pagedPhiAvailable/);
  assert.match(octreeFaceMirrorShader, /r==2u\|\|r==4u/);
  assert.match(octreeFaceMirrorShader, /surfaceArena\[6\]>0u/);
  assert.match(octreeFaceMirrorShader, /if\(!pagedPhiAvailable\(\)\)\{return textureLoad\(levelSetIn,p,0\)\.x;\}/);
  assert.match(octreeFaceMirrorShader, /fn airCellKey\(p: vec3u\) -> u32 \{ return index\(p\) \+ 1u; \}/,
    "face classification must query the exact full-domain linear air-alias key");
  assert.doesNotMatch(octreeFaceMirrorShader, /p\.x \| \(p\.y << 10u\)/,
    "face classification must not truncate air-side coordinates to 10:10:10");
  assert.match(octreeFaceMirrorShader, /!surfaceContains\(leaf,point\)/,
    "air-side face classification must continue the owner plane beyond its page rather than clamp liquid phi");
  assert.equal(typeof WebGPUOctreeFaceMirror.prototype.setSurfacePageSource, "function");
});

test("adaptive face projection applies the matched pressure gradient and checks dense parity", () => {
  assert.match(octreeFaceMirrorShader, /fn projectFaces/);
  assert.match(octreeFaceMirrorShader, /face\.normalVelocity-=\(facePressure\(face\.positiveRow\)-facePressure\(face\.negativeRow\)\)/);
  assert.match(octreeFaceMirrorShader, /clamp\(abs\(liquidPhi\)\/max\(abs\(liquidPhi\)\+abs\(airPhi\),1e-12\),0\.05,1\.0\)\*full/,
    "compact projection must retain the tall-cell 20x ghost-kick bound");
  assert.match(octreeFaceMirrorShader, /fn reduceProjectionParity/);
  assert.match(octreeFaceMirrorShader, /textureLoad\(projectedVelocity/);
  const solve = WebGPUOctreeProjection.prototype.encode.toString().replace(/\s+/g, "");
  assert.match(solve, /this\.faceMirror\?\.encodeProjection\(encoder,finalInA\)/);
  assert.match(solve, /this\.faceMirror\?\.encodeProjectionParity\(encoder,finalInA\)/);
  assert.ok(solve.indexOf("encodeProjection(encoder,finalInA)") < solve.indexOf("Octreefinite-volumevelocityprojection"));
  assert.ok(solve.indexOf("encodeProjectionParity(encoder,finalInA)") > solve.indexOf("project.end()"));
});

test("paged authority disables obsolete dense projection parity", () => {
  assert.match(WebGPUOctreeFaceMirror.prototype.encodeProjectionParity.toString(), /pagedSurfaceAttached/);
});
