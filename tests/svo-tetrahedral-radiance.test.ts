import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";

import {
  SVO_TETRAHEDRAL_RADIANCE_DIRECTIONS,
  SVO_TETRAHEDRAL_RADIANCE_LAYOUT,
  evaluateSvoTetrahedralDiffuseIrradiance,
  evaluateSvoTetrahedralRadiance,
  evaluateSvoTetrahedralRadianceRaw,
  packSvoRadianceRgb9e5,
  packSvoTetrahedralRadianceTexel,
  reduceSvoTetrahedralRadianceChildren,
  svoTetrahedralLambertianEmission,
  svoTetrahedralRadianceAtlasBytes,
  svoTetrahedralRadianceWGSL,
  unpackSvoRadianceRgb9e5,
  unpackSvoTetrahedralRadianceTexel,
  type SvoRadianceRgb,
  type SvoTetrahedralRadiance,
} from "../lib/svo-tetrahedral-radiance";
import { SVO_NODE_MIP_LAYOUT, planSvoNodeMipPyramid } from "../lib/svo-node-mip-pyramid";

const close = (actual: number, expected: number, tolerance = 1e-9) => assert.ok(Math.abs(actual - expected) <= tolerance,
  `expected ${actual} to be within ${tolerance} of ${expected}`);
const closeRgb = (actual: SvoRadianceRgb, expected: SvoRadianceRgb, tolerance = 1e-9) => actual.forEach((value, index) => close(value, expected[index], tolerance));
const dot = (left: readonly number[], right: readonly number[]) => left[0] * right[0] + left[1] * right[1] + left[2] * right[2];

test("regular tetrahedral directions are centered and maximally separated", () => {
  const sum = [0, 0, 0];
  SVO_TETRAHEDRAL_RADIANCE_DIRECTIONS.forEach((direction, index) => {
    close(Math.hypot(...direction), 1);
    direction.forEach((value, axis) => { sum[axis] += value; });
    for (let other = 0; other < index; other += 1) close(dot(direction, SVO_TETRAHEDRAL_RADIANCE_DIRECTIONS[other]), -1 / 3);
  });
  sum.forEach((value) => close(value, 0));
});

test("four samples exactly reconstruct constant-plus-linear directional radiance", () => {
  const mean: SvoRadianceRgb = [2, 3, 4];
  const x: SvoRadianceRgb = [.2, -.1, .3], y: SvoRadianceRgb = [-.3, .4, .1], z: SvoRadianceRgb = [.1, .2, -.2];
  const sample = (direction: readonly number[]): SvoRadianceRgb => [0, 1, 2].map((channel) => mean[channel]
    + x[channel] * direction[0] + y[channel] * direction[1] + z[channel] * direction[2]) as unknown as SvoRadianceRgb;
  const tetra = SVO_TETRAHEDRAL_RADIANCE_DIRECTIONS.map(sample) as unknown as SvoTetrahedralRadiance;
  const direction = [2 / 3, -1 / 3, 2 / 3] as const;
  closeRgb(evaluateSvoTetrahedralRadianceRaw(tetra, direction), sample(direction));
  closeRgb(evaluateSvoTetrahedralRadiance(tetra, direction), sample(direction));
});

test("isotropic tetrahedral radiance has direction-independent radiance and pi irradiance", () => {
  const color: SvoRadianceRgb = [.25, 2, 8];
  const tetra = [color, color, color, color] as SvoTetrahedralRadiance;
  closeRgb(evaluateSvoTetrahedralRadiance(tetra, [1, -2, 3]), color);
  closeRgb(evaluateSvoTetrahedralDiffuseIrradiance(tetra, [-2, 1, .5]), color.map((channel) => Math.PI * channel) as unknown as SvoRadianceRgb);
});

test("Lambertian injection is coverage-premultiplied and mip reduction is an arithmetic mean", () => {
  const emission = svoTetrahedralLambertianEmission([4, 2, 1], [0, 1, 0], .5);
  assert.deepEqual(emission, [[2, 1, .5], [0, 0, 0], [2, 1, .5], [0, 0, 0]]);
  const black = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]] as SvoTetrahedralRadiance;
  const reduced = reduceSvoTetrahedralRadianceChildren([emission, black, black, black, black, black, black, black]);
  assert.deepEqual(reduced, [[.25, .125, .0625], [0, 0, 0], [.25, .125, .0625], [0, 0, 0]]);
});

test("RGB9E5 packing preserves practical HDR radiance and four-word texel order", () => {
  for (const color of [[0, 0, 0], [.125, .5, 1], [12.5, 6.25, 3.125], [1024, 512, 256]] as const) {
    const unpacked = unpackSvoRadianceRgb9e5(packSvoRadianceRgb9e5(color));
    color.forEach((expected, channel) => close(unpacked[channel], expected, Math.max(1 / 256, expected / 256)));
  }
  const tetra = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]] as SvoTetrahedralRadiance;
  const packed = packSvoTetrahedralRadianceTexel(tetra);
  assert.equal(packed.length, 4);
  const unpacked = unpackSvoTetrahedralRadianceTexel(packed);
  tetra.forEach((expected, direction) => closeRgb(unpacked[direction], expected, .04));
});

test("radiance atlas shares sparse page topology at sixteen bytes per texel", () => {
  const plan = planSvoNodeMipPyramid({ generation: 1, occupiedPages: [[0, 0, 0]], levelCount: 2, atlasPages: [2, 1, 1] });
  assert.equal(SVO_TETRAHEDRAL_RADIANCE_LAYOUT.textureFormat, "rgb9e5ufloat");
  assert.equal(SVO_TETRAHEDRAL_RADIANCE_LAYOUT.bytesPerTexel, 16);
  const physicalPageBytes = SVO_NODE_MIP_LAYOUT.physicalSize ** 3 * 16;
  assert.equal(SVO_TETRAHEDRAL_RADIANCE_LAYOUT.bytesPerPhysicalPage, physicalPageBytes);
  assert.equal(svoTetrahedralRadianceAtlasBytes(plan), 2 * physicalPageBytes, "physical atlas capacity includes its unused slot");
});

test("WGSL ABI uses four filterable lobes and analytic direction/diffuse reconstruction", () => {
  assert.match(svoTetrahedralRadianceWGSL, /fn svoTetraSample\(lobe0:texture_3d<f32>,lobe1:texture_3d<f32>,lobe2:texture_3d<f32>,lobe3:texture_3d<f32>,atlasSampler:sampler/);
  assert.equal((svoTetrahedralRadianceWGSL.match(/textureSampleLevel\(/g) ?? []).length, 4);
  assert.match(svoTetrahedralRadianceWGSL, /fn svoTetraRadianceAlong/);
  assert.match(svoTetrahedralRadianceWGSL, /fn svoTetraDiffuseIrradiance/);
});

const modulePath = process.env.WEBGPU_NODE_MODULE;
test("tetrahedral RGB9E5 textures and WGSL compile on the selected WebGPU adapter", {
  skip: !modulePath && "set WEBGPU_NODE_MODULE for GPU radiance-ABI checks",
}, async () => {
  const { create, globals } = await import(pathToFileURL(modulePath!).href) as { create(options: string[]): GPU; globals: Record<string, unknown> };
  Object.assign(globalThis, globals);
  const gpu = create(["backend=metal"]), adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
  assert.ok(adapter);
  const device = await adapter.requestDevice();
  const textures: GPUTexture[] = [];
  try {
    const code = `${svoTetrahedralRadianceWGSL}
@group(0) @binding(0) var lobe0:texture_3d<f32>;
@group(0) @binding(1) var lobe1:texture_3d<f32>;
@group(0) @binding(2) var lobe2:texture_3d<f32>;
@group(0) @binding(3) var lobe3:texture_3d<f32>;
@group(0) @binding(4) var radianceSampler:sampler;
@compute @workgroup_size(1) fn validate(){let value=svoTetraSample(lobe0,lobe1,lobe2,lobe3,radianceSampler,vec3f(.5));_=svoTetraRadianceAlong(value,vec3f(0.0,1.0,0.0));_=svoTetraDiffuseIrradiance(value,vec3f(0.0,1.0,0.0));}`;
    const shaderModule = device.createShaderModule({ label: "Tetrahedral radiance ABI validation", code });
    const info = await shaderModule.getCompilationInfo();
    assert.deepEqual(info.messages.filter(({ type }) => type === "error"), []);
    const entries: GPUBindGroupLayoutEntry[] = [0, 1, 2, 3].map((binding) => ({
      binding, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "float" as const, viewDimension: "3d" as const },
    }));
    entries.push({ binding: 4, visibility: GPUShaderStage.COMPUTE, sampler: { type: "filtering" } });
    const layout = device.createBindGroupLayout({ entries });
    const pipeline = await device.createComputePipelineAsync({
      layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
      compute: { module: shaderModule, entryPoint: "validate" },
    });
    for (let index = 0; index < 4; index += 1) textures.push(device.createTexture({
      label: `Tetrahedral RGB9E5 lobe ${index}`, size: [1, 1, 1], dimension: "3d", format: "rgb9e5ufloat",
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    }));
    const bindGroup = device.createBindGroup({ layout, entries: [
      ...textures.map((texture, binding) => ({ binding, resource: texture.createView({ dimension: "3d" as const }) })),
      { binding: 4, resource: device.createSampler({ minFilter: "linear", magFilter: "linear" }) },
    ] });
    const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
    pass.setPipeline(pipeline); pass.setBindGroup(0, bindGroup); pass.dispatchWorkgroups(1); pass.end();
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
  } finally {
    textures.forEach((texture) => texture.destroy());
    device.destroy();
  }
});
