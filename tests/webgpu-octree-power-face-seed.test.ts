import assert from "node:assert/strict";
import test from "node:test";
import {
  OCTREE_POWER_FACE_SEED_CONTROL_BYTES,
  octreePowerFaceSeedShader,
  planOctreePowerFaceSeed,
  WebGPUOctreePowerFaceSeed,
} from "../lib/webgpu-octree-power-face-seed";

test("native power-face seed allocates only its face transaction", () => {
  assert.deepEqual(planOctreePowerFaceSeed(100, 800), {
    rowCapacity: 100,
    faceCapacity: 800,
    velocityScratchBytes: 3_200,
    faceStatusBytes: 3_200,
    allocatedBytes: 3_200 + 3_200 + OCTREE_POWER_FACE_SEED_CONTROL_BYTES + 32,
  });
  assert.throws(() => planOctreePowerFaceSeed(0, 1), /positive integers/);
  assert.throws(() => planOctreePowerFaceSeed(1, 0), /positive integers/);
});

test("native power-face seed has no Cartesian bridge or reverse publication", () => {
  assert.doesNotMatch(octreePowerFaceSeedShader,
    /AxisFace|axisControl|axisFaces|axisIncidence|PowerToAxis|homologous/i);
  assert.equal("encodePowerToAxis" in WebGPUOctreePowerFaceSeed.prototype, false);
  assert.match(octreePowerFaceSeedShader,
    /let carried=\(face\.flags&DELTA_CARRIED\)!=0u/);
  assert.match(octreePowerFaceSeedShader,
    /var value=select\(0\.0,face\.normalVelocity,carried\)/);
  assert.match(octreePowerFaceSeedShader,
    /powerFaces\[index\]\.normalVelocity=velocityScratch\[index\]/);
});

test("cold seed fails closed on invalid native power publication", () => {
  assert.match(octreePowerFaceSeedShader,
    /powerControl\[3\]!=0u\|\|powerControl\[8\]!=VALID/);
  assert.match(octreePowerFaceSeedShader,
    /seed\.valid=select\(0u,VALID,flags==0u&&seeded==faceCount\(\)\)/);
  assert.match(octreePowerFaceSeedShader,
    /if\(seed\.valid==VALID&&index<faceCount\(\)/);
});

test("closed walls remain locked through seed and acceleration", () => {
  const closedWall = /face\.positiveRow==INVALID&&\(face\.flags&BOUNDARY\)!=0u[\s\S]*?\(face\.flags&OPEN_BOUNDARY\)==0u/;
  assert.match(octreePowerFaceSeedShader, closedWall);
  assert.match(octreePowerFaceSeedShader,
    /if\(closed\)\{velocityScratch\[index\]=0\.0;faceStatus\[index\]=VALID;return;\}/);
});
