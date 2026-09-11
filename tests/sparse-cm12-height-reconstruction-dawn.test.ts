import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { sparseCM12HeightReconstructionWGSL, SPARSE_CM12_HEIGHT_ENTRY_POINTS,
  SPARSE_CM12_HEIGHT_ITERATIONS } from "../lib/methods/adaptive-volume/sparse-cm12-height-reconstruction.wgsl";

const live = new Set<GPU>();
const dawnTest = process.env.WEBGPU_NODE_MODULE ? test : test.skip;
const size = 128, count = size * size;
type Leaf = { x: number; z: number; width: number; mean: number };

function fixture(field: (x: number, z: number) => number, maximumWidth = 8) {
  const input = new Float32Array(2 * count);
  const exact = new Float32Array(count);
  const leaves: Leaf[] = [];
  // Conforming 1:2:4:8 bands. Every 8x8 publication page contains whole cells,
  // and both directions cross coarse/fine corners as well as straight seams.
  for (let z = 0; z < size;) {
    const rowWidth = Math.min(maximumWidth, z < 16 ? 1 : z < 48 ? 2 : z < 80 ? 4 : 8);
    for (let x = 0; x < size;) {
      const width = Math.min(rowWidth, x < 16 ? 1 : x < 48 ? 2 : x < 80 ? 4 : 8);
      for (let dz = 0; dz < rowWidth; dz += width) {
        let mean = 0;
        for (let j = 0; j < 32; j++) for (let i = 0; i < 32; i++) {
          mean += field(x + width * (i + .5) / 32, z + dz + width * (j + .5) / 32) / 1024;
        }
        mean = Math.fround(mean);
        leaves.push({ x, z: z + dz, width, mean });
        for (let iz = 0; iz < width; iz++) for (let ix = 0; ix < width; ix++) {
          const at = x + ix + size * (z + dz + iz);
          input[2 * at] = mean; input[2 * at + 1] = width;
          exact[at] = field(x + ix + .5, z + dz + iz + .5);
        }
      }
      x += width;
    }
    z += rowWidth;
  }
  return { input, exact, leaves };
}

dawnTest("common height preserves adaptive cell averages and reconstructs smooth waves", async () => {
  await acquireWebGPUExclusiveLock("dawn-test", "common-height-reconstruction");
  let gpu: GPU | undefined, device: GPUDevice | undefined;
  try {
    const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href);
    Object.assign(globalThis, dawn.globals);
    gpu = dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]) as GPU;
    live.add(gpu);
    const adapter = await gpu.requestAdapter(); assert.ok(adapter);
    device = await adapter.requestDevice();
    const errors: string[] = [];
    device.addEventListener("uncapturederror", e => errors.push(e.error.message));
    const shader = `
@group(0) @binding(0) var<storage,read_write> fineSamples:array<u32>;
@group(0) @binding(1) var<storage,read> input:array<vec2f>;
fn heightDomain()->vec2u{return vec2u(${size});}
fn heightMaximum()->f32{return 96.0;}
fn heightGeneration()->u32{return 1u;}
fn heightBufferBase()->u32{return 0u;}
fn heightIsEnabled()->bool{return true;}
fn heightRawReceipt(q:vec2u)->vec2f{return input[q.x+${size}u*q.y];}
${sparseCM12HeightReconstructionWGSL(8)}`;
    const module = device.createShaderModule({ code: shader });
    const layout = device.createBindGroupLayout({ entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
    ] });
    const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [layout] });
    const pipelines = new Map<string, GPUComputePipeline>();
    for (const entryPoint of SPARSE_CM12_HEIGHT_ENTRY_POINTS) {
      pipelines.set(entryPoint, await device.createComputePipelineAsync({ layout: pipelineLayout,
        compute: { module, entryPoint } }));
    }
    const data = device.createBuffer({ size: 4 * (9 * count + 16),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    const input = device.createBuffer({ size: 8 * count, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const output = device.createBuffer({ size: data.size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const group = device.createBindGroup({ layout, entries: [
      { binding: 0, resource: { buffer: data } }, { binding: 1, resource: { buffer: input } },
    ] });
    const run = async (clear: boolean): Promise<Float32Array> => {
      const encoder = device!.createCommandEncoder(); if (clear) encoder.clearBuffer(data);
      const pass = encoder.beginComputePass(); pass.setBindGroup(0, group);
      const dispatch = (name: string, reduction = false) => {
        pass.setPipeline(pipelines.get(name)!); pass.dispatchWorkgroups(reduction ? 1 : size / 8, reduction ? 1 : size / 8);
      };
      dispatch("initializeCM12Height"); dispatch("constrainCM12Height");
      dispatch("laplacianCM12Height"); dispatch("applyCM12HeightLaplacian");
      dispatch("initializeCM12HeightResidual"); dispatch("reduceCM12HeightInitial", true);
      for (let iteration = 0; iteration < SPARSE_CM12_HEIGHT_ITERATIONS; iteration++) {
        dispatch("laplacianCM12HeightDirection"); dispatch("applyCM12HeightLaplacian");
        dispatch("projectCM12HeightOperator"); dispatch("reduceCM12HeightAlpha", true);
        dispatch("updateCM12HeightResidual"); dispatch("reduceCM12HeightBeta", true);
        dispatch("updateCM12HeightDirection");
      }
      dispatch("finishCM12Height"); pass.end();
      encoder.copyBufferToBuffer(data, 0, output, 0, data.size);
      device!.queue.submit([encoder.finish()]);
      await output.mapAsync(GPUMapMode.READ);
      const values = new Float32Array(output.getMappedRange()).slice(); output.unmap();
      return values;
    };
    for (const [name, field] of [
      ["constant", () => 32],
      ["mask", () => 32],
      ["bounded step", (x: number) => x < size / 2 ? 1 : 95],
      ["plane", (x: number, z: number) => 32 + .002 * (x - size / 2) + .003 * (z - size / 2)],
      ["ring", (x: number, z: number) => 32 + .36 * Math.exp(-(((Math.hypot(x - size / 2, z - size / 2) - 33) / 8.4) ** 2))],
    ] as const) {
      const f = fixture(field);
      if (name === "mask") {
        for (let z = 51; z < 69; z++) for (let x = 51; x < 69; x++) {
          f.input[2 * (x + size * z) + 1] = 0;
        }
      }
      device.queue.writeBuffer(input, 0, f.input);
      const values = await run(true);
      const reconstructed = values.subarray(count, 2 * count);
      let maximumMeanError = 0, rawError = 0, error = 0, maximumError = 0, samples = 0;
      for (const leaf of f.leaves) {
        let mean = 0;
        for (let z = 0; z < leaf.width; z++) for (let x = 0; x < leaf.width; x++) {
          const height = reconstructed[leaf.x + x + size * (leaf.z + z)]!;
          assert.ok(Number.isFinite(height) && height >= 0 && height <= 96);
          mean += height / leaf.width ** 2;
        }
        maximumMeanError = Math.max(maximumMeanError, Math.abs(mean - leaf.mean));
      }
      for (let z = 8; z < size - 8; z++) for (let x = 8; x < size - 8; x++) {
        const at = x + size * z;
        rawError += (f.input[2 * at]! - f.exact[at]!) ** 2;
        error += (reconstructed[at]! - f.exact[at]!) ** 2;
        maximumError = Math.max(maximumError, Math.abs(reconstructed[at]! - f.exact[at]!));
        samples++;
      }
      const rms = Math.sqrt(error / samples), rawRms = Math.sqrt(rawError / samples);
      console.log(JSON.stringify({ name, maximumMeanError, rms, rawRms, maximumError }));
      assert.ok(maximumMeanError < 1e-5, `${name}: every accepted column average must survive`);
      if (name === "constant" || name === "mask") assert.equal(maximumError, 0);
      if (name === "mask") assert.equal(values[6 * count + 60 + size * 60], 0,
        "invalid columns must retain their fallback instead of gaining a height authority");
      if (name === "plane") assert.ok(rms < 0.0005, `tilted plane: ${rms}`);
      if (name === "ring") {
        assert.ok(rms < rawRms * .2, `circular wave error: ${rms} / ${rawRms}`);
        let seamError = 0, seamRawError = 0, seamSamples = 0;
        for (const boundary of [16, 48, 80]) for (let along = 8; along < size - 8; along++) {
          for (const axis of [1, size]) {
            const at = axis === 1 ? boundary + size * along : along + size * boundary;
            const curvature = (a: Float32Array, stride = 1) =>
              a[stride * (at - axis)]! - 2 * a[stride * at]! + a[stride * (at + axis)]!;
            const truth = curvature(f.exact);
            seamError += (curvature(reconstructed) - truth) ** 2;
            seamRawError += (curvature(f.input, 2) - truth) ** 2;
            seamSamples++;
          }
        }
        assert.ok(seamError < seamRawError * .04,
          "2:1 seams must remove at least 80% of the artificial change in slope");
        const settled = (await run(false)).subarray(count, 2 * count);
        let warmShift = 0;
        for (let i = 0; i < count; i++) warmShift = Math.max(warmShift,
          Math.abs(settled[i]! - reconstructed[i]!));
        console.log(JSON.stringify({ seamRms: Math.sqrt(seamError / seamSamples),
          rawSeamRms: Math.sqrt(seamRawError / seamSamples), warmShift }));
        assert.ok(warmShift < .02,
          "an unchanged accepted field must not acquire a visible warm-start jump");
        const refined = fixture(field, 4);
        device.queue.writeBuffer(input, 0, refined.input);
        const rerung = (await run(false)).subarray(count, 2 * count);
        let rerungError = 0;
        for (let z = 8; z < size - 8; z++) for (let x = 8; x < size - 8; x++) {
          const at = x + size * z;
          rerungError += (rerung[at]! - reconstructed[at]!) ** 2;
        }
        for (const leaf of refined.leaves) {
          let mean = 0;
          for (let z = 0; z < leaf.width; z++) for (let x = 0; x < leaf.width; x++) {
            mean += rerung[leaf.x + x + size * (leaf.z + z)]! / leaf.width ** 2;
          }
          assert.ok(Math.abs(mean - leaf.mean) < 1e-5,
            "re-runging must constrain the new accepted averages");
        }
        const rerungRms = Math.sqrt(rerungError / samples);
        console.log(JSON.stringify({ rerungRms }));
        assert.ok(rerungRms < .002,
          "resolving the same circular wave at the next rung must preserve its shape");
      }
    }
    assert.deepEqual(errors, []);
    data.destroy(); input.destroy(); output.destroy();
  } finally {
    device?.destroy(); await releaseWebGPUExclusiveLock(); if (gpu) live.delete(gpu);
  }
});
