import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { gpuRigidBodyShader, GPU_RIGID_MOTION_BYTES, GPU_RIGID_RENDER_BYTES, GPU_RIGID_STATE_BYTES } from "../lib/webgpu-rigid-body";

const uniform = readFileSync(new URL("../lib/webgpu-uniform-eulerian.ts", import.meta.url), "utf8");
const restricted = readFileSync(new URL("../lib/webgpu-eulerian.ts", import.meta.url), "utf8");
const renderer = readFileSync(new URL("../lib/webgpu-renderer.ts", import.meta.url), "utf8");
const controller = readFileSync(new URL("../lib/simulation/controller.ts", import.meta.url), "utf8");
const viewport = readFileSync(new URL("../components/WebGPUViewport.tsx", import.meta.url), "utf8");

test("rigid state and render records have stable storage ABIs", () => {
  assert.equal(GPU_RIGID_STATE_BYTES, 12 * 32 * 4);
  assert.equal(GPU_RIGID_RENDER_BYTES, 12 * 16 * 4);
  assert.equal(GPU_RIGID_MOTION_BYTES, 12 * 128);
  assert.match(gpuRigidBodyShader, /array<RigidBody, 12>/);
  assert.match(gpuRigidBodyShader, /@compute @workgroup_size\(1\)/);
});

test("resident rigid kernel consumes fluid exchange and resolves contacts", () => {
  assert.match(gpuRigidBodyShader, /atomicLoad\(&exchange\[base\]\)/);
  assert.match(gpuRigidBodyShader, /let buoyancy=-rho\*displaced\*params\.gravity\.xyz/);
  assert.match(gpuRigidBodyShader, /for\(var iteration=0u;iteration<6u;iteration\+\+\)/);
  assert.match(gpuRigidBodyShader, /solveBodyPair\(a,b\)/);
  assert.match(gpuRigidBodyShader, /terrainPlane\(body\.positionShape\.xyz\)/);
  assert.doesNotMatch(gpuRigidBodyShader, /mapAsync|getMappedRange/);
});

test("all WebGPU solvers integrate rigid bodies without a physics readback", () => {
  const uniformAdvance = uniform.slice(uniform.indexOf("advanceTo(time_s"), uniform.indexOf("async readStats()"));
  const restrictedAdvance = restricted.slice(restricted.indexOf("advanceTo(time_s"), restricted.indexOf("async readStats()"));
  for (const source of [uniformAdvance, restrictedAdvance]) {
    assert.match(source, /rigidSystem\.encode\(/);
    assert.doesNotMatch(source, /mapAsync|getMappedRange|copyBufferToBuffer\(this\.rigidExchangeBuffer/);
  }
});

test("WebGPU rendering consumes GPU-authored body records", () => {
  assert.match(renderer, /residentRigidBuffer/);
  assert.match(renderer, /encoder\.copyBufferToBuffer\(residentRigidBuffer/);
  assert.match(renderer, /if \(residentRigidBuffer\)[\s\S]*else \{[\s\S]*queue\.writeBuffer\(this\.bodyBuffer/, "CPU body uploads remain only as the CPU-reference fallback");
  assert.match(controller,
    /if \(backend === "cpu-reference"\) \{[\s\S]*advanceRigidBodies\(this\.bodies[\s\S]*if \(backend === "cpu-reference"\) \{[\s\S]*this\.publishBodies\(diagnostics\)/,
    "the host controller must neither integrate nor publish a shadow rigid state for WebGPU");
});

test("mouse interaction picks live GPU poses with one bounded click readback", () => {
  assert.match(gpuRigidBodyShader, /fn pickRigidBody/);
  assert.match(gpuRigidBodyShader, /body\.dimensions\.w\*body\.dimensions\.w/);
  assert.match(renderer, /async pickRigidBody/);
  assert.match(viewport, /await rendererRef\.current\.pickRigidBody/);
  assert.match(viewport, /picked\.position_m,picked\.orientation/);
  assert.match(controller, /if \(orientation\) body\.orientation/);
  const residentSource = readFileSync(new URL("../lib/webgpu-rigid-body.ts", import.meta.url), "utf8");
  const pick = residentSource.slice(residentSource.indexOf("async pick("), residentSource.indexOf("encode(encoder:"));
  assert.match(pick, /GPU_RIGID_PICK_BYTES/);
  assert.match(pick, /mapAsync\(GPUMapMode\.READ\)/);
});
