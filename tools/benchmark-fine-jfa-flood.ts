import { pathToFileURL } from "node:url";
import {
  fineLevelSetJFACPTSubgroupOracleWGSL,
  fineLevelSetJFACPTWGSL,
} from "../lib/webgpu-octree-fine-levelset-redistance";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "./webgpu-smoke-isolation";

const brickDimension = Number(process.env.FLUID_FINE_JFA_FLOOD_BRICK_DIMENSION ?? 8);
if (!Number.isSafeInteger(brickDimension) || brickDimension < 2 || brickDimension > 16) {
  throw new RangeError("brick dimension must be an integer in [2, 16]");
}
const brickDimensions = [brickDimension, brickDimension, brickDimension] as const;
const pageCount = brickDimensions[0] * brickDimensions[1] * brickDimensions[2];
const samplesPerBrick = 64;
const sampleCount = pageCount * samplesPerBrick;
const generation = 1;
const repeats = Number(process.env.FLUID_FINE_JFA_FLOOD_REPEATS ?? 8);
const rounds = Number(process.env.FLUID_FINE_JFA_FLOOD_ROUNDS ?? 1);
if (!Number.isSafeInteger(repeats) || repeats < 1 || !Number.isSafeInteger(rounds) || rounds < 1) {
  throw new RangeError("repeat and round counts must be positive integers");
}

await acquireWebGPUExclusiveLock("dawn-benchmark", "tools/benchmark-fine-jfa-flood.ts");
try {
  const modulePath = process.env.WEBGPU_NODE_MODULE ?? `${process.cwd()}/node_modules/webgpu/index.js`;
  const dawn = await import(pathToFileURL(modulePath).href) as {
    create(options: string[]): GPU; globals: Record<string, unknown>;
  };
  Object.assign(globalThis, dawn.globals);
  const gpu = dawn.create([`backend=${process.env.WEBGPU_BACKEND ?? "metal"}`]);
  const adapter = await gpu.requestAdapter();
  if (!adapter?.features.has("subgroups")) {
    throw new Error("Fine JFA flood benchmark requires subgroups");
  }
  const device = await adapter.requestDevice({ requiredFeatures: ["subgroups"],
    requiredLimits: { maxStorageBuffersPerShaderStage: adapter.limits.maxStorageBuffersPerShaderStage } });
  const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
  const make = (label: string, size: number) => device.createBuffer({ label, size, usage: storage });
  const upload = (label: string, values: Uint32Array | Float32Array) => {
    const buffer = make(label, values.byteLength);
    device.queue.writeBuffer(buffer, 0, values.buffer as ArrayBuffer, values.byteOffset, values.byteLength);
    return buffer;
  };

  const paramsWords = new Uint32Array(28); const paramsFloats = new Float32Array(paramsWords.buffer);
  paramsWords.set([...brickDimensions, 4, ...brickDimensions.map((value) => value * 4), samplesPerBrick,
    pageCount, 7, pageCount, generation, 23], 0);
  paramsFloats[13] = 0.0125; paramsFloats[14] = 0.1;
  paramsWords[15] = sampleCount; paramsWords[16] = device.limits.maxComputeWorkgroupsPerDimension;
  paramsWords[19] = 16 + 3 * pageCount;
  const params = device.createBuffer({ label: "Fine JFA flood parameters", size: paramsWords.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(params, 0, paramsWords);

  const metadataWords = new Uint32Array(pageCount * 10);
  for (let id = 0; id < pageCount; id += 1) metadataWords.set([id, id, generation], id * 10);
  const metadata = upload("Fine JFA flood metadata", metadataWords);

  const directoryBase = 7 + pageCount, haloBase = directoryBase + pageCount;
  const worklistWords = new Uint32Array(haloBase + pageCount * 27).fill(0xffff_ffff);
  worklistWords.set([generation, pageCount, pageCount, 3, pageCount - 1, 1, 1]);
  for (let id = 0; id < pageCount; id += 1) {
    worklistWords[7 + id] = id; worklistWords[directoryBase + id] = id;
    const x = id % brickDimensions[0];
    const y = Math.floor(id / brickDimensions[0]) % brickDimensions[1];
    const z = Math.floor(id / (brickDimensions[0] * brickDimensions[1]));
    for (let slot = 0; slot < 27; slot += 1) {
      const sx = slot % 3, sy = Math.floor(slot / 3) % 3, sz = Math.floor(slot / 9);
      const nx = x + sx - 1, ny = y + sy - 1, nz = z + sz - 1;
      if (nx >= 0 && nx < brickDimensions[0] && ny >= 0 && ny < brickDimensions[1]
        && nz >= 0 && nz < brickDimensions[2]) {
        worklistWords[haloBase + id * 27 + slot] = nx
          + brickDimensions[0] * (ny + brickDimensions[1] * nz);
      }
    }
  }
  const worklist = upload("Fine JFA flood worklist", worklistWords);
  const pageDeltaWords = new Uint32Array(16 + 4 * pageCount);
  pageDeltaWords[3] = pageCount;
  for (let id = 0; id < pageCount; id += 1) pageDeltaWords[16 + 3 * pageCount + id] = id;
  const pageDelta = upload("Fine JFA flood page delta", pageDeltaWords);
  const flags = upload("Fine JFA flood flags", new Uint32Array(sampleCount).fill(0xe000_0001));
  const phi = upload("Fine JFA flood phi", new Float32Array(sampleCount).fill(1));
  const seeds = Uint32Array.from({ length: sampleCount }, (_, index) => index);
  const workA = upload("Fine JFA flood work A", seeds);
  const workB = upload("Fine JFA flood work B", seeds);
  const supportMask = upload("Fine JFA flood support mask", new Uint32Array(pageCount).fill(generation));
  const frontierPages = make("Fine JFA flood unused frontier pages", 4);
  const frontierPublished = upload("Fine JFA flood disabled frontier", new Uint32Array([0]));
  const buffers = [[0, params], [1, worklist], [2, metadata], [3, pageDelta], [4, flags],
    [5, phi], [6, workA], [7, workB], [10, supportMask], [13, frontierPages],
    [14, frontierPublished]] as const;

  const variants = [
    { name: "voxel-lanes", source: fineLevelSetJFACPTWGSL },
    { name: "subgroup-per-voxel", source: fineLevelSetJFACPTSubgroupOracleWGSL },
  ];
  const measurements: Record<string, Record<string, number>> = {};
  for (const stride of [1, 8]) {
    measurements[`stride-${stride}`] = {};
    const runnable = variants.map((variant) => {
      const module = device.createShaderModule({ label: `Fine JFA ${variant.name}`, code: variant.source });
      const pipeline = device.createComputePipeline({ layout: "auto", compute: {
        module, entryPoint: "jumpFloodAToB", constants: { JFA_STRIDE: stride },
      } });
      const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: buffers.map(
        ([binding, buffer]) => ({ binding, resource: { buffer } })),
      });
      const measure = async () => {
        const encoder = device.createCommandEncoder();
        const pass = encoder.beginComputePass();
        pass.setPipeline(pipeline); pass.setBindGroup(0, group);
        for (let repeat = 0; repeat < repeats; repeat += 1) pass.dispatchWorkgroups(pageCount);
        pass.end(); const commands = encoder.finish(); const started = performance.now();
        device.queue.submit([commands]); await device.queue.onSubmittedWorkDone();
        return (performance.now() - started) / repeats;
      };
      return { ...variant, measure };
    });
    for (const variant of runnable) await variant.measure();
    const samples = Object.fromEntries(runnable.map((variant) => [variant.name, [] as number[]]));
    for (let round = 0; round < rounds; round += 1) {
      const order = round % 2 === 0 ? runnable : [...runnable].reverse();
      for (const variant of order) samples[variant.name]!.push(await variant.measure());
    }
    for (const variant of runnable) {
      const ordered = samples[variant.name]!.sort((a, b) => a - b);
      measurements[`stride-${stride}`]![variant.name] = ordered[Math.floor(ordered.length / 2)]!;
    }
  }
  console.log(JSON.stringify({ pageCount, samplesPerBrick, repeats, rounds,
    millisecondsPerFlood: measurements,
    speedup: Object.fromEntries(Object.entries(measurements).map(([stride, result]) =>
      [stride, result["subgroup-per-voxel"]! / result["voxel-lanes"]!])) }));
  for (const buffer of [params, metadata, worklist, pageDelta, flags, phi, workA, workB,
    supportMask, frontierPages, frontierPublished]) buffer.destroy();
  device.destroy();
} finally {
  await releaseWebGPUExclusiveLock();
}
