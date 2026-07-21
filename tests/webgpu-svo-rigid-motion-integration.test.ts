import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { SVO_PRIMITIVE_MOTION_STRIDE_BYTES } from "../lib/svo-primitive-motion";
import {
  GPU_RIGID_BODY_CAPACITY,
  GPU_RIGID_MOTION_BYTES,
  gpuRigidBodyShader,
} from "../lib/webgpu-rigid-body";
import {
  SVO_DRY_SCENE_BINDING_CONTRACT,
  SVO_DRY_RIGID_MOTION_CAPACITY,
  SVO_DRY_RIGID_MOTION_UNIFORM_BYTES,
  svoDrySceneShader,
} from "../lib/webgpu-svo-dry-scene";
import { sparseVoxelTemporalAccumulatorShader } from "../lib/webgpu-svo-temporal-accumulator";

const rigidSource = readFileSync(new URL("../lib/webgpu-rigid-body.ts", import.meta.url), "utf8");
const rendererSource = readFileSync(new URL("../lib/webgpu-renderer.ts", import.meta.url), "utf8");
const methodsSource = readFileSync(new URL("../lib/methods/types.ts", import.meta.url), "utf8");
const restrictedSource = readFileSync(new URL("../lib/webgpu-eulerian.ts", import.meta.url), "utf8");
const uniformSource = readFileSync(new URL("../lib/webgpu-uniform-eulerian.ts", import.meta.url), "utf8");
const drySource = readFileSync(new URL("../lib/webgpu-svo-dry-scene.ts", import.meta.url), "utf8");

test("GPU-resident rigid producer publishes the exact 128-byte sidecar and conservative swept support", () => {
  assert.equal(GPU_RIGID_BODY_CAPACITY, 12);
  assert.equal(GPU_RIGID_MOTION_BYTES, GPU_RIGID_BODY_CAPACITY * SVO_PRIMITIVE_MOTION_STRIDE_BYTES);
  assert.match(rigidSource, /GPU rigid primitive motion sidecar[\s\S]*?GPUBufferUsage\.STORAGE \| GPUBufferUsage\.COPY_DST \| GPUBufferUsage\.COPY_SRC/);
  assert.match(gpuRigidBodyShader, /@binding\(7\) var<storage, read_write> rigidMotion: array<SvoPrimitiveMotionRecord, 12>/);
  assert.match(gpuRigidBodyShader, /var previousBodies:array<RigidBody,12>[\s\S]*?publishMotion\(index,previousBodies\[index\],bodies\[index\],dt\)/);
  assert.match(gpuRigidBodyShader, /svoPrimitiveMotionSweptBounds/,
    "the published previous/current centers plus rotation-independent radius are directly consumable for preactivation");
  assert.match(gpuRigidBodyShader, /maximumDisplacement=length\(deltaPosition\)\+2\.0\*radius\*sin\(\.5\*angularDisplacement\)/);
  assert.match(gpuRigidBodyShader, /motionLimit=min\(\.5,2\.0\*max\(params\.coupling\.w,1e-6\)\)/);
});

test("continuity, teleport, and roster changes fail closed while revision and generation remain exact", () => {
  assert.match(gpuRigidBodyShader, /generation=bitcast<u32>\(currentBody\.material\.y\)/,
    "the CPU command epoch is carried bit-exactly through the otherwise unused resident-state lane");
  assert.match(gpuRigidBodyShader, /generationContinuous=generation!=0u&&generation==previousGeneration/);
  assert.match(gpuRigidBodyShader, /revisionContinuous=generationContinuous&&motionTransformMatches\(old,previousBody\)/);
  assert.match(gpuRigidBodyShader, /valid=dt>1e-8&&generationContinuous&&revisionContinuous&&!teleport/);
  assert.match(gpuRigidBodyShader, /linearVelocity=select\(vec3f\(0\.0\),currentBody\.linearVelocity\.xyz,valid\)/);
  assert.match(rigidSource, /stateWords\[o\+29\]=nextMotionGenerations\[index\]/);
  assert.match(rigidSource, /encoder\.copyBufferToBuffer\(this\.motionBuffer,previous\*SVO_PRIMITIVE_MOTION_STRIDE_BYTES,this\.motionScratch/,
    "roster compaction keeps each body's motion publication beside the same stable body ID");
});

test("production dry pass uses a renderer-owned uniform mirror without adding an eleventh fragment storage binding", () => {
  assert.equal(SVO_DRY_RIGID_MOTION_CAPACITY, 12);
  assert.equal(SVO_DRY_RIGID_MOTION_UNIFORM_BYTES, 12 * 128);
  assert.equal((svoDrySceneShader.match(/var<storage,\s*read>/g) ?? []).length, 10);
  assert.match(svoDrySceneShader, /@binding\(14\) var<uniform> rigidMotion:array<SvoPrimitiveMotionRecord,12>/);
  assert.deepEqual(SVO_DRY_SCENE_BINDING_CONTRACT.find(({ binding }) => binding === 14), { binding: 14, type: "uniform" });
  assert.match(drySource, /return \{ binding, visibility: GPUShaderStage\.FRAGMENT, buffer: \{ type \} \}/);
  assert.match(drySource, /copyBufferToBuffer\(this\.rigidMotionSource, 0, this\.rigidMotionUniformBuffer, 0, SVO_DRY_RIGID_MOTION_UNIFORM_BYTES\)/);
  assert.match(rendererSource, /setRigidMotionSource\(backend === "webgpu" \? this\.gpuFluid\?\.rigidMotionBuffer : undefined\)/);
  assert.match(methodsSource, /readonly rigidMotionBuffer\?: GPUBuffer/);
  assert.match(restrictedSource, /get rigidMotionBuffer\(\)\{return this\.rigidSystem\.motionBuffer;\}/);
  assert.match(uniformSource, /get rigidMotionBuffer\(\) \{ return this\.rigidSystem\.motionBuffer; \}/);
});

test("G-buffer identity gates exact surface velocity and moving-rigid temporal reprojection", () => {
  assert.match(svoDrySceneShader, /svoPrimitiveMotionOwnerId\(record\)==hit\.ownerId/);
  assert.match(svoDrySceneShader, /svoPrimitiveMotionMaterialId\(record\)==dryResolvedMaterialId\(hit\)/);
  assert.match(svoDrySceneShader, /transformValid=distance\(record\.currentPositionDt\.xyz,bodies\[hit\.ownerId\]\.positionRadius\.xyz\)<=1e-5/);
  assert.match(svoDrySceneShader, /svoPrimitiveMotionVelocityAt\(record,worldSurfacePosition_m\)/);
  assert.match(svoDrySceneShader, /motionVelocity[\s\S]*?motionGeneration[\s\S]*?motionValid[\s\S]*?svoGBufferSurface/);
  assert.match(sparseVoxelTemporalAccumulatorShader, /temporalVelocity\(packed\.z\)/);
  assert.match(sparseVoxelTemporalAccumulatorShader, /previousWorld=world-velocity\*temporal\.control\.y/);
  assert.match(sparseVoxelTemporalAccumulatorShader, /supportedMotion=motionKind==SVO_TEMPORAL_MOTION_STATIC\|\|motionKind==SVO_TEMPORAL_MOTION_RIGID/);
});
