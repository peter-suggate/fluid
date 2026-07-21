#!/usr/bin/env node
/**
 * A/B GPU benchmark for the cone-traced node-mip lighting marcher
 * (`createSvoDryConeMarcherWGSL`). Compiles the baseline and optimized marcher
 * variants from the same builder in one process, proves bit-identical
 * per-ray results (transmittance bits, valid, step count) plus morton/find
 * probe parity, then times the variants with interleaved timestamp queries.
 *
 * Rerun: node --import tsx tools/benchmark-svo-cone-gpu.ts
 * Note: dawn-node occasionally segfaults mid-run on repeated dispatch/readback
 * cycles (independent of variant); rerun until the JSON report is produced.
 * Env: WEBGPU_NODE_MODULE, FLUID_SVO_CONE_WIDTH/_HEIGHT/_WARMUPS/_CYCLES/_DISPATCHES.
 */
import assert from "node:assert/strict";
import { fileURLToPath, pathToFileURL } from "node:url";

import {
  SVO_NODE_MIP_LAYOUT,
  createSvoNodeMipPageWithApron,
  planSvoNodeMipPyramid,
  type SvoNodeMipCoordinate,
  type SvoNodeMipRgba8,
  reduceSvoNodeMipChildren,
} from "../lib/svo-node-mip-pyramid";
import { svoNodeMipSamplingWGSL } from "../lib/svo-node-mip-sampling";
import { WebGpuSvoNodeMipPyramid } from "../lib/webgpu-svo-node-mip-pyramid";
import { createSvoDryConeMarcherWGSL, type SvoDryConeMarcherOptions } from "../lib/webgpu-svo-dry-scene";

const width = positiveInteger(process.env.FLUID_SVO_CONE_WIDTH ?? "256", "width");
const height = positiveInteger(process.env.FLUID_SVO_CONE_HEIGHT ?? "256", "height");
const warmups = positiveInteger(process.env.FLUID_SVO_CONE_WARMUPS ?? "4", "warmups");
const cycles = positiveInteger(process.env.FLUID_SVO_CONE_CYCLES ?? "12", "cycles");
const dispatches = positiveInteger(process.env.FLUID_SVO_CONE_DISPATCHES ?? "4", "dispatches");
const modulePath = process.env.WEBGPU_NODE_MODULE
  ?? fileURLToPath(new URL("../node_modules/webgpu/index.js", import.meta.url));

function positiveInteger(value: string, label: string): number {
  const parsed = Number(value);
  if (!Number.isSafeInteger(parsed) || parsed < 1) throw new RangeError(`${label} must be a positive integer`);
  return parsed;
}

const VARIANTS = [
  { name: "baseline", options: {} },
  { name: "morton", options: { branchlessMorton: true } },
  { name: "morton+ranged", options: { branchlessMorton: true, rangedDirectorySearch: true } },
  { name: "full", options: { branchlessMorton: true, rangedDirectorySearch: true, emptySpaceElision: true } },
] as const satisfies ReadonlyArray<{ name: string; options: SvoDryConeMarcherOptions }>;
type VariantName = (typeof VARIANTS)[number]["name"];

// ---------------------------------------------------------------------------
// Fixture: 128^3 fine voxels (16^3 level-0 pages), 5 mip levels. Floor slab,
// pillars, a sphere, explicit resident-empty pages, and non-resident air gaps.
// ---------------------------------------------------------------------------
const GENERATION = 7;
const FINE = 128;
const LEVELS = 5;
const CELL = 0.1;
const EXTENT = FINE * CELL;
const N = SVO_NODE_MIP_LAYOUT.interiorSize;

function solidAt(x: number, y: number, z: number): boolean {
  if (y < 6) return true; // floor slab
  const pillar = (x % 32 >= 14 && x % 32 <= 17 && z % 32 >= 14 && z % 32 <= 17 && y < 90);
  if (pillar) return true;
  const dx = x - 64.5, dy = y - 70.5, dz = z - 64.5;
  return dx * dx + dy * dy + dz * dz <= 20 * 20; // sphere
}

interface LevelGrid { size: number; texels: Uint8Array }
const levelGrids: LevelGrid[] = [];
{
  const base = new Uint8Array(FINE * FINE * FINE * 4);
  for (let z = 0; z < FINE; z += 1) for (let y = 0; y < FINE; y += 1) for (let x = 0; x < FINE; x += 1) {
    if (!solidAt(x, y, z)) continue;
    const offset = ((z * FINE + y) * FINE + x) * 4;
    base[offset] = 255; base[offset + 1] = 255;
  }
  levelGrids.push({ size: FINE, texels: base });
  for (let level = 1; level < LEVELS; level += 1) {
    const childGrid = levelGrids[level - 1];
    const size = childGrid.size / 2;
    const texels = new Uint8Array(size * size * size * 4);
    for (let z = 0; z < size; z += 1) for (let y = 0; y < size; y += 1) for (let x = 0; x < size; x += 1) {
      const children: SvoNodeMipRgba8[] = [];
      for (let cz = 0; cz < 2; cz += 1) for (let cy = 0; cy < 2; cy += 1) for (let cx = 0; cx < 2; cx += 1) {
        const offset = (((z * 2 + cz) * childGrid.size + y * 2 + cy) * childGrid.size + x * 2 + cx) * 4;
        children.push([childGrid.texels[offset], childGrid.texels[offset + 1], childGrid.texels[offset + 2], childGrid.texels[offset + 3]]);
      }
      texels.set(reduceSvoNodeMipChildren(children), ((z * size + y) * size + x) * 4);
    }
    levelGrids.push({ size, texels });
  }
}

const key3 = (page: readonly number[]) => page.join(",");
const occupied = new Map<string, SvoNodeMipCoordinate>();
for (let z = 0; z < FINE / N; z += 1) for (let y = 0; y < FINE / N; y += 1) for (let x = 0; x < FINE / N; x += 1) {
  let any = false;
  for (let tz = 0; tz < N && !any; tz += 1) for (let ty = 0; ty < N && !any; ty += 1) for (let tx = 0; tx < N && !any; tx += 1) {
    if (levelGrids[0].texels[(((z * N + tz) * FINE + y * N + ty) * FINE + x * N + tx) * 4 + 1] !== 0) any = true;
  }
  if (any) occupied.set(key3([x, y, z]), [x, y, z]);
}
// Resident-empty pages: an all-air band that still owns level-0 residency.
for (let x = 0; x < FINE / N; x += 1) for (let z = 0; z < FINE / N; z += 1) {
  if ((x + z) % 3 === 0) occupied.set(key3([x, 13, z]), [x, 13, z]);
}

const plan = planSvoNodeMipPyramid({ generation: GENERATION, occupiedPages: [...occupied.values()], levelCount: LEVELS });
assert.ok(plan.complete, "fixture plan must be complete");
const residentByLevel = Array.from({ length: LEVELS }, () => new Set<string>());
for (const page of plan.pages) residentByLevel[page.key.level].add(key3(page.key.coordinate));

function denseTexel(level: number, coordinate: readonly number[]): SvoNodeMipRgba8 | undefined {
  const grid = levelGrids[level];
  if (coordinate.some((component) => component < 0 || component >= grid.size)) return undefined;
  const offset = ((coordinate[2] * grid.size + coordinate[1]) * grid.size + coordinate[0]) * 4;
  return [grid.texels[offset], grid.texels[offset + 1], grid.texels[offset + 2], grid.texels[offset + 3]];
}

function pageInterior(level: number, page: SvoNodeMipCoordinate): Uint8Array {
  const interior = new Uint8Array(N * N * N * 4);
  for (let z = 0; z < N; z += 1) for (let y = 0; y < N; y += 1) for (let x = 0; x < N; x += 1) {
    const texel = denseTexel(level, [page[0] * N + x, page[1] * N + y, page[2] * N + z]);
    if (texel) interior.set(texel, ((z * N + y) * N + x) * 4);
  }
  return interior;
}

// ---------------------------------------------------------------------------
// WGSL harness around the production marcher block.
// ---------------------------------------------------------------------------
function harnessShader(options: SvoDryConeMarcherOptions): string {
  let marcher = createSvoDryConeMarcherWGSL(options);
  // Non-semantic instrumentation for the fetch/search histogram.
  const fetchAnchor = "let local=virtualVoxel-";
  const searchAnchor = "let middle=low+(high-low)/2u;";
  assert.ok(marcher.includes(fetchAnchor) && marcher.includes(searchAnchor), "instrumentation anchors missing");
  marcher = marcher.replace(fetchAnchor, "benchFetches+=1u;let local=virtualVoxel-");
  marcher = marcher.replace(searchAnchor, "benchSearchIterations+=1u;let middle=low+(high-low)/2u;");
  return /* wgsl */ `
${svoNodeMipSamplingWGSL}
struct BenchMapping{worldOrigin:vec3f,pad0:u32,cellSize:vec3f,pad1:u32}
struct DryParams{mapping:BenchMapping,nodeMip:vec4u,nodeMipLevelStart:array<vec4u,3>}
struct RayResult{transmittanceBits:u32,valid:u32,steps:u32,fetches:u32,searchIterations:u32,pad0:u32,pad1:u32,pad2:u32}
struct ProbeResult{mortonLow:u32,mortonHigh:u32,findResult:u32,pad:u32}
@group(0) @binding(0) var<storage,read_write> rayResults:array<RayResult>;
@group(0) @binding(1) var<storage,read_write> probeResults:array<ProbeResult>;
@group(0) @binding(8) var<storage,read> publicationState:array<u32>;
@group(0) @binding(9) var<uniform> dry:DryParams;
@group(0) @binding(16) var nodeMipAtlas:texture_3d<f32>;
@group(0) @binding(17) var nodeMipSampler:sampler;
@group(0) @binding(18) var nodeMipDirectory:texture_2d<u32>;
var<private> dryMipSteps:u32;
var<private> benchFetches:u32;
var<private> benchSearchIterations:u32;
${marcher}
fn benchHash(seed:u32)->u32{var value=seed;value^=value>>16u;value*=0x7feb352du;value^=value>>15u;value*=0x846ca68bu;value^=value>>16u;return value;}
@compute @workgroup_size(64)
fn benchmarkMain(@builtin(global_invocation_id) gid:vec3u){
  if(gid.x>=${width * height}u){return;}
  let px=gid.x%${width}u;let py=gid.x/${width}u;
  let extent=${EXTENT};
  var origin=vec3f((f32(px)+.5)/${width}.0*extent,0.0,(f32(py)+.5)/${height}.0*extent);
  let heightSelector=gid.x%5u;
  origin.y=select(select(select(select(.65,3.1,heightSelector==1u),6.4,heightSelector==2u),9.05,heightSelector==3u),11.3,heightSelector==4u);
  var direction=normalize(vec3f(cos(f32(gid.x)*2.39996323),.35+.6*sin(f32(gid.x)*.7302),sin(f32(gid.x)*2.39996323)));
  if(px%16u==0u){direction=vec3f(1.0,0.0,0.0);origin.z=f32(py%16u)*.8;}
  else if(py%16u==0u){direction=vec3f(0.0,1.0,0.0);origin.x=f32(px%16u)*.8;}
  else if(px%16u==7u){direction=normalize(vec3f(1.0,.0004,.0002));}
  let aperture=select(select(select(.065,.15,gid.x%4u==1u),.35,gid.x%4u==2u),.6,gid.x%4u==3u);
  let maximumDistance=select(select(3.0,8.0,gid.x%3u==1u),22.0,gid.x%3u==2u);
  dryMipSteps=0u;benchFetches=0u;benchSearchIterations=0u;
  let cone=dryConeVisibility(origin,direction,aperture,maximumDistance,vec3f(0.0),false);
  rayResults[gid.x]=RayResult(bitcast<u32>(cone.transmittance),cone.valid,dryMipSteps,benchFetches,benchSearchIterations,0u,0u,0u);
}
@compute @workgroup_size(64)
fn probeMain(@builtin(global_invocation_id) gid:vec3u){
  if(gid.x>=${width * height}u){return;}
  _=publicationState[0];
  _=svoNodeMipSamplePage(nodeMipAtlas,nodeMipSampler,vec3u(0u),vec3f(0.0));
  var coordinate:vec3u;
  if(gid.x%2u==0u){coordinate=vec3u(benchHash(gid.x)%20u,benchHash(gid.x+7u)%20u,benchHash(gid.x+13u)%20u);}
  else{coordinate=vec3u(benchHash(gid.x)&0x1fffffu,benchHash(gid.x+101u)&0x1fffffu,benchHash(gid.x+211u)&0x1fffffu);}
  if(gid.x%17u==0u){coordinate=vec3u(0x7ffu,0x800u,0x3ffu);}
  if(gid.x%17u==1u){coordinate=vec3u(0x400u,0x1fffffu,0x801u);}
  let level=gid.x%7u;
  let morton=dryNodeMipMorton(coordinate);
  probeResults[gid.x]=ProbeResult(morton.x,morton.y,dryNodeMipFind(level,coordinate),0u);
}`;
}

// ---------------------------------------------------------------------------
// GPU bring-up, fixture upload.
// ---------------------------------------------------------------------------
const { create, globals } = await import(pathToFileURL(modulePath).href) as {
  create(options: string[]): GPU;
  globals: Record<string, unknown>;
};
Object.assign(globalThis, globals);
const gpu = create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
assert.ok(adapter, "WebGPU adapter unavailable");
assert.ok(adapter.features.has("timestamp-query"), "timestamp-query is required");
const device = await adapter.requestDevice({ requiredFeatures: ["timestamp-query"] });
const validationErrors: string[] = [];
device.addEventListener("uncapturederror", (event) => validationErrors.push((event as GPUUncapturedErrorEvent).error.message));

const pyramid = new WebGpuSvoNodeMipPyramid(device);
pyramid.beginGeneration(plan);
for (const page of plan.pages) {
  const interior = pageInterior(page.key.level, page.key.coordinate);
  const physical = createSvoNodeMipPageWithApron(page.key.coordinate, interior, ({ page: neighbour, texel }) => {
    if (!residentByLevel[page.key.level].has(key3(neighbour))) return undefined;
    return denseTexel(page.key.level, [neighbour[0] * N + texel[0], neighbour[1] * N + texel[1], neighbour[2] * N + texel[2]]);
  });
  pyramid.uploadPhysicalPage(page.key, physical);
}
assert.ok(pyramid.publish().published, "fixture pyramid publication failed");
const visible = pyramid.visibleGeneration()!;

const levelStart = new Uint32Array(12);
for (const page of plan.pages) if (page.key.level < 11) levelStart[page.key.level + 1] += 1;
for (let boundary = 1; boundary < levelStart.length; boundary += 1) levelStart[boundary] += levelStart[boundary - 1];
const paramWords = new Uint32Array(24);
const paramFloats = new Float32Array(paramWords.buffer);
paramFloats.set([0, 0, 0], 0);
paramFloats.set([CELL, CELL, CELL], 4);
paramWords.set([GENERATION, plan.pages.length, LEVELS, 1], 8);
paramWords.set(levelStart, 12);
const paramsBuffer = device.createBuffer({ size: paramWords.byteLength, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
device.queue.writeBuffer(paramsBuffer, 0, paramWords);
const publicationBuffer = device.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
device.queue.writeBuffer(publicationBuffer, 0, new Uint32Array([0, 0, GENERATION, 0]));

const rayCount = width * height;
const rayResultBytes = rayCount * 32;
const probeResultBytes = rayCount * 16;
const rayResults = device.createBuffer({ size: rayResultBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
const probeResults = device.createBuffer({ size: probeResultBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
const rayReadback = device.createBuffer({ size: rayResultBytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
const probeReadback = device.createBuffer({ size: probeResultBytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });

interface Target { ray: GPUComputePipeline; probe: GPUComputePipeline; rayBind: GPUBindGroup; probeBind: GPUBindGroup }
const targets = new Map<VariantName, Target>();
for (const variant of VARIANTS) {
  const module = device.createShaderModule({ label: `cone marcher ${variant.name}`, code: harnessShader(variant.options) });
  const info = await module.getCompilationInfo();
  assert.deepEqual(info.messages.filter(({ type }) => type === "error").map(({ lineNum, linePos, message }) => ({ lineNum, linePos, message })), [], `${variant.name} failed to compile`);
  // Auto layouts keep only the bindings each entry point statically uses.
  const bindEntries = (pipeline: GPUComputePipeline, output: 0 | 1): GPUBindGroup => device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: output, resource: { buffer: output === 0 ? rayResults : probeResults } },
      { binding: 8, resource: { buffer: publicationBuffer } },
      { binding: 9, resource: { buffer: paramsBuffer } },
      { binding: 16, resource: visible.view },
      { binding: 17, resource: visible.sampler },
      { binding: 18, resource: visible.directoryView },
    ],
  });
  const ray = device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "benchmarkMain" } });
  const probe = device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "probeMain" } });
  targets.set(variant.name, { ray, probe, rayBind: bindEntries(ray, 0), probeBind: bindEntries(probe, 1) });
}

async function run(variant: VariantName, entry: "ray" | "probe"): Promise<Uint32Array> {
  const target = targets.get(variant)!;
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(entry === "ray" ? target.ray : target.probe);
  pass.setBindGroup(0, entry === "ray" ? target.rayBind : target.probeBind);
  pass.dispatchWorkgroups(Math.ceil(rayCount / 64));
  pass.end();
  const source = entry === "ray" ? rayResults : probeResults;
  const readback = entry === "ray" ? rayReadback : probeReadback;
  encoder.copyBufferToBuffer(source, 0, readback, 0, entry === "ray" ? rayResultBytes : probeResultBytes);
  device.queue.submit([encoder.finish()]);
  await readback.mapAsync(GPUMapMode.READ);
  const copied = new Uint32Array(readback.getMappedRange().slice(0));
  readback.unmap();
  return copied;
}

// ---------------------------------------------------------------------------
// Bit-exact parity across every variant.
// ---------------------------------------------------------------------------
const rayOutputs = new Map<VariantName, Uint32Array>();
const probeOutputs = new Map<VariantName, Uint32Array>();
for (const variant of VARIANTS) {
  rayOutputs.set(variant.name, await run(variant.name, "ray"));
  probeOutputs.set(variant.name, await run(variant.name, "probe"));
}
const baselineRays = rayOutputs.get("baseline")!;
const baselineProbes = probeOutputs.get("baseline")!;
let terminated = 0;
for (let ray = 0; ray < rayCount; ray += 1) {
  if (baselineRays[ray * 8 + 2] > 0 && new Float32Array(baselineRays.buffer, ray * 32, 1)[0] < 1) terminated += 1;
}
assert.ok(terminated > rayCount * 0.2, `only ${terminated}/${rayCount} rays attenuated — fixture coverage is too sparse`);
for (const variant of VARIANTS) {
  if (variant.name === "baseline") continue;
  const rays = rayOutputs.get(variant.name)!;
  let firstDivergence = -1;
  for (let ray = 0; ray < rayCount && firstDivergence < 0; ray += 1) {
    // Lanes 0..2: transmittance bits, valid, step count. Fetch/search counters may differ.
    if (rays[ray * 8] !== baselineRays[ray * 8] || rays[ray * 8 + 1] !== baselineRays[ray * 8 + 1]
      || rays[ray * 8 + 2] !== baselineRays[ray * 8 + 2]) firstDivergence = ray;
  }
  assert.equal(firstDivergence, -1,
    firstDivergence < 0 ? "" : `${variant.name} ray ${firstDivergence} diverged: `
      + `[0x${rays[firstDivergence * 8].toString(16)},${rays[firstDivergence * 8 + 1]},${rays[firstDivergence * 8 + 2]}] != `
      + `[0x${baselineRays[firstDivergence * 8].toString(16)},${baselineRays[firstDivergence * 8 + 1]},${baselineRays[firstDivergence * 8 + 2]}]`);
  const probes = probeOutputs.get(variant.name)!;
  let probeDivergence = -1;
  for (let word = 0; word < rayCount * 4 && probeDivergence < 0; word += 1) {
    if (probes[word] !== baselineProbes[word]) probeDivergence = word;
  }
  assert.equal(probeDivergence, -1, `${variant.name} morton/find probe word ${probeDivergence} diverged`);
}

function histogram(rays: Uint32Array): { steps: number; fetches: number; searchIterations: number; stepHistogram: number[] } {
  let steps = 0, fetches = 0, searchIterations = 0;
  const stepHistogram = new Array(7).fill(0);
  for (let ray = 0; ray < rayCount; ray += 1) {
    steps += rays[ray * 8 + 2]; fetches += rays[ray * 8 + 3]; searchIterations += rays[ray * 8 + 4];
    stepHistogram[Math.min(6, rays[ray * 8 + 2] >> 3)] += 1;
  }
  return { steps, fetches, searchIterations, stepHistogram };
}

// ---------------------------------------------------------------------------
// Interleaved timestamp timing.
// ---------------------------------------------------------------------------
process.stderr.write("parity: bit-exact across variants\n");
for (let warmup = 0; warmup < warmups; warmup += 1) for (const variant of VARIANTS) await run(variant.name, "ray");
const timestampCount = cycles * VARIANTS.length * 2;
const querySet = device.createQuerySet({ type: "timestamp", count: timestampCount });
const resolve = device.createBuffer({ size: timestampCount * 8, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
const queryReadback = device.createBuffer({ size: timestampCount * 8, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
const labels: VariantName[] = [];
let query = 0;
// One submit per cycle: a single command buffer holding every timed pass can
// exceed the Metal command-buffer watchdog and take the device down.
for (let cycle = 0; cycle < cycles; cycle += 1) {
  const cycleEncoder = device.createCommandEncoder();
  const order = cycle % 2 === 0 ? [...VARIANTS] : [...VARIANTS].reverse();
  for (const variant of order) {
    const target = targets.get(variant.name)!;
    const pass = cycleEncoder.beginComputePass({ timestampWrites: { querySet, beginningOfPassWriteIndex: query, endOfPassWriteIndex: query + 1 } });
    pass.setPipeline(target.ray);
    pass.setBindGroup(0, target.rayBind);
    for (let dispatch = 0; dispatch < dispatches; dispatch += 1) pass.dispatchWorkgroups(Math.ceil(rayCount / 64));
    pass.end();
    labels.push(variant.name);
    query += 2;
  }
  device.queue.submit([cycleEncoder.finish()]);
  await device.queue.onSubmittedWorkDone();
}
const encoder = device.createCommandEncoder();
encoder.resolveQuerySet(querySet, 0, timestampCount, resolve, 0);
encoder.copyBufferToBuffer(resolve, 0, queryReadback, 0, timestampCount * 8);
device.queue.submit([encoder.finish()]);
await queryReadback.mapAsync(GPUMapMode.READ);
const timestamps = new BigUint64Array(queryReadback.getMappedRange().slice(0));
queryReadback.unmap();
const samples = new Map<VariantName, number[]>(VARIANTS.map(({ name }) => [name, []]));
for (let index = 0; index < labels.length; index += 1) {
  samples.get(labels[index])!.push(Number(timestamps[index * 2 + 1] - timestamps[index * 2]) / 1e6 / dispatches);
}
await device.queue.onSubmittedWorkDone();
assert.deepEqual(validationErrors, []);

function median(values: readonly number[]): number {
  const ordered = [...values].sort((a, b) => a - b);
  const middle = Math.floor(ordered.length / 2);
  return ordered.length % 2 === 0 ? (ordered[middle - 1] + ordered[middle]) * 0.5 : ordered[middle];
}

const baselineMedian = median(samples.get("baseline")!);
console.log(JSON.stringify({
  phase: "svo-cone-marcher-gpu-benchmark",
  backend: process.env.FLUID_WEBGPU_BACKEND ?? "metal",
  rays: rayCount,
  fixture: {
    fineVoxels: FINE, levels: LEVELS, cellSize_m: CELL,
    residentPages: plan.pages.length,
    pagesPerLevel: residentByLevel.map((set) => set.size),
  },
  parity: {
    bitExactAcrossVariants: true,
    attenuatedRays: terminated,
    probeWordsCompared: rayCount * 4,
  },
  work: Object.fromEntries(VARIANTS.map(({ name }) => [name, histogram(rayOutputs.get(name)!)])),
  gpuMilliseconds: Object.fromEntries(VARIANTS.map(({ name }) => {
    const variantSamples = samples.get(name)!;
    const variantMedian = median(variantSamples);
    return [name, {
      median: variantMedian,
      improvementPercent: (1 - variantMedian / baselineMedian) * 100,
      samples: variantSamples.map((value) => Number(value.toFixed(4))),
    }];
  })),
}, null, 2));

querySet.destroy(); resolve.destroy(); queryReadback.destroy();
rayResults.destroy(); probeResults.destroy(); rayReadback.destroy(); probeReadback.destroy();
paramsBuffer.destroy(); publicationBuffer.destroy(); pyramid.destroy(); device.destroy();
