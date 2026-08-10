/**
 * Minimal Dawn/webgpu-node async-pipeline reproducer extracted from the
 * abandoned compact MacCormack predictor snapshot. In the focused test
 * process on the affected Metal Dawn build, createComputePipelineAsync
 * terminates instead of returning a validation error; the same focused
 * process also crashes on known-old product shaders, while the production
 * pipeline compiler succeeds. This file is deliberately not authoritative
 * and is not part of the normal suite.
 */
import assert from "node:assert/strict";
import { pathToFileURL } from "node:url";

const modulePath = process.env.WEBGPU_NODE_MODULE;
assert.ok(modulePath, "set WEBGPU_NODE_MODULE to node_modules/webgpu/index.js");
const dawn = await import(pathToFileURL(modulePath).href) as {
  create(options: string[]): GPU;
  globals: Record<string, unknown>;
};
Object.assign(globalThis, dawn.globals);
const adapter = await dawn.create(["backend=metal"]).requestAdapter();
assert.ok(adapter);
const device = await adapter.requestDevice();
const module = device.createShaderModule({ code: /* wgsl */ `
@group(0) @binding(0) var<storage, read> source: array<vec2f>;
@group(0) @binding(1) var<storage, read_write> scratch: array<f32>;
@compute @workgroup_size(64)
fn snapshot(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x < arrayLength(&source) && gid.x < arrayLength(&scratch)) {
    scratch[gid.x] = source[gid.x].y;
  }
}
` });
await device.createComputePipelineAsync({
  layout: "auto", compute: { module, entryPoint: "snapshot" },
});
device.destroy();
