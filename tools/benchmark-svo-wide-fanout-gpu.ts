#!/usr/bin/env node
import assert from "node:assert/strict";
import { fileURLToPath, pathToFileURL } from "node:url";

import { planSvoWideFanout } from "../lib/svo-wide-fanout";
import { packSvoWideFanout, createWebgpuSvoWideFanoutTraversalWGSL, validateSvoWidePackedPlan } from "../lib/webgpu-svo-wide-fanout";
import { webgpuSvoTraversalWGSL } from "../lib/webgpu-svo-traversal";

type Variant = "canonical" | "wide";
const INVALID = 0xffff_ffff;
const depth = 4, brickSize = 4;
const invocationCount = Number(process.env.FLUID_SVO_WIDE_INVOCATIONS ?? 16_384);
const cycles = Number(process.env.FLUID_SVO_WIDE_CYCLES ?? 20);
const warmups = Number(process.env.FLUID_SVO_WIDE_WARMUPS ?? 4);
const dispatchesPerSample = Number(process.env.FLUID_SVO_WIDE_DISPATCHES ?? 2);
const originX = Number(process.env.FLUID_SVO_WIDE_ORIGIN_X ?? -4);
const productionFallback = process.env.FLUID_SVO_WIDE_PRODUCTION_FALLBACK !== "0";
const rayProfile = process.env.FLUID_SVO_WIDE_RAY_PROFILE ?? "camera";
const malformedWidePage = process.env.FLUID_SVO_WIDE_MALFORMED_PAGE === "1";
const modulePath = process.env.WEBGPU_NODE_MODULE
  ?? fileURLToPath(new URL("../node_modules/webgpu/index.js", import.meta.url));
assert.ok(Number.isSafeInteger(invocationCount) && invocationCount > 0);
assert.ok(Number.isSafeInteger(cycles) && cycles > 0 && Number.isSafeInteger(warmups) && warmups > 0);
assert.ok(Number.isSafeInteger(dispatchesPerSample) && dispatchesPerSample > 0);
assert.ok(Number.isFinite(originX) && originX < 0);
assert.ok(["camera", "parallel", "diagonal"].includes(rayProfile));

function denseFixture() {
  const levelCounts = Array.from({ length: depth + 1 }, (_, level) => 8 ** level);
  const offsets: number[] = []; let nodeCount = 0;
  for (const count of levelCounts) { offsets.push(nodeCount); nodeCount += count; }
  const leafCount = levelCounts[depth];
  const nodes = new Uint32Array(nodeCount * 8), leaves = new Uint32Array(leafCount * 4);
  const terminals: Array<{ sourceNodeIndex: number; sourceLeafIndex: number; level: number; coordinate: readonly [number, number, number] }> = [];
  for (let level = 0; level <= depth; level += 1) {
    for (let local = 0; local < levelCounts[level]; local += 1) {
      const node = offsets[level] + local, base = node * 8;
      nodes[base] = local; nodes[base + 2] = level;
      if (level < depth) {
        nodes[base + 3] = 0xff; nodes[base + 4] = offsets[level + 1] + local * 8;
        nodes[base + 5] = 8; nodes[base + 6] = INVALID;
      } else {
        let x = 0, y = 0, z = 0;
        for (let bit = 0; bit < level; bit += 1) {
          x |= ((local >>> (3 * bit)) & 1) << bit;
          y |= ((local >>> (3 * bit + 1)) & 1) << bit;
          z |= ((local >>> (3 * bit + 2)) & 1) << bit;
        }
        nodes[base + 4] = INVALID; nodes[base + 6] = local;
        leaves.set([node, local * brickSize ** 3, local, 0], local * 4);
        terminals.push({ sourceNodeIndex: node, sourceLeafIndex: local, level, coordinate: [x, y, z] });
      }
    }
  }
  const widePlan = planSvoWideFanout({ sourceGeneration: 1, generation: 1, maximumDepth: depth, terminals });
  const wide = packSvoWideFanout(widePlan);
  return { nodes, leaves, widePlan, wide, nodeCount, leafCount };
}

function shader(variant: Variant, nodeCount: number, leafCount: number, publication: { generation: number; pageCount: number; descriptorCount: number }): string {
  const { generation, pageCount, descriptorCount } = publication;
  const slopeScale = Math.min(1, 4 / Math.abs(originX));
  const rayExpression = rayProfile === "parallel"
    ? `SvoRay(vec3f(${originX.toFixed(1)},4.0,2.0),0.0,vec3f(1.0,0.0,0.0),${(Math.abs(originX) + 256).toFixed(1)})`
    : rayProfile === "diagonal"
      ? `SvoRay(vec3f(-1.0,3.0,2.25),0.0,normalize(vec3f(1.0,1.0,.001)),256.0)`
      : `SvoRay(vec3f(${originX.toFixed(1)},fy*63.0+.25,fz*63.0+.25),0.0,normalize(vec3f(1.0,(${(.005 * slopeScale).toExponential(9)}+fy*${(.01 * slopeScale).toExponential(9)}),(${(.003 * slopeScale).toExponential(9)}+fz*${(.008 * slopeScale).toExponential(9)}))),${(Math.abs(originX) + 256).toFixed(1)})`;
  const wide = variant === "wide" ? createWebgpuSvoWideFanoutTraversalWGSL({ pages: 3, descriptors: 4 }) : "";
  const bindings = variant === "wide"
    ? "@group(0) @binding(5) var<storage,read_write> output:array<vec4u>;"
    : "@group(0) @binding(3) var<storage,read_write> output:array<vec4u>;";
  const wideBody = productionFallback ? `
  var wideCursor:SvoWideTraversalCursor;var canonicalCursor:SvoTraversalContinuation;let publication=SvoWidePublication(${generation}u,${generation}u,${pageCount}u,${descriptorCount}u);
  var useWide=svoWideCursorInitialize(&wideCursor,ray,mapping,publication,1u);
  for(var i=0u;i<32u;i+=1u){var hit:SvoTraversalHit;if(useWide){hit=svoWideCursorNext(&wideCursor,SvoRay(ray.origin,minimum,ray.direction,ray.tMax),mapping,${depth}u,publication,1u);if(hit.status==SVO_STATUS_INVALID_TOPOLOGY){useWide=false;svoTraversalContinuationBegin(SvoRay(ray.origin,minimum,ray.direction,ray.tMax),mapping,&canonicalCursor);hit=svoTraversalContinuationNext(SvoRay(ray.origin,minimum,ray.direction,ray.tMax),mapping,${depth}u,&canonicalCursor);}}else{if(i==0u){svoTraversalContinuationBegin(ray,mapping,&canonicalCursor);}hit=svoTraversalContinuationNext(SvoRay(ray.origin,minimum,ray.direction,ray.tMax),mapping,${depth}u,&canonicalCursor);}visits+=hit.visits;status=hit.status;if(hit.status!=SVO_STATUS_HIT){break;}hits+=1u;hash=((hash*16777619u)^(hit.nodeIndex*31u+hit.leafIndex))^(hit.voxelOffset+hit.level*65537u)^bitcast<u32>(hit.tEnter)^bitcast<u32>(hit.tExit);minimum=hit.tExit+1e-5;}` : `
  var wideCursor:SvoWideTraversalCursor;let publication=SvoWidePublication(${generation}u,${generation}u,${pageCount}u,${descriptorCount}u);let ready=svoWideCursorInitialize(&wideCursor,ray,mapping,publication,1u);
  for(var i=0u;i<32u&&ready;i+=1u){let hit=svoWideCursorNext(&wideCursor,SvoRay(ray.origin,minimum,ray.direction,ray.tMax),mapping,${depth}u,publication,1u);visits+=hit.visits;status=hit.status;if(hit.status!=SVO_STATUS_HIT){break;}hits+=1u;hash=((hash*16777619u)^(hit.nodeIndex*31u+hit.leafIndex))^(hit.voxelOffset+hit.level*65537u)^bitcast<u32>(hit.tEnter)^bitcast<u32>(hit.tExit);minimum=hit.tExit+1e-5;}`;
  const body = variant === "wide" ? wideBody : `
  var cursor:SvoTraversalContinuation;svoTraversalContinuationBegin(ray,mapping,&cursor);
  for(var i=0u;i<32u;i+=1u){let hit=svoTraversalContinuationNext(SvoRay(ray.origin,minimum,ray.direction,ray.tMax),mapping,${depth}u,&cursor);visits+=hit.visits;status=hit.status;if(hit.status!=SVO_STATUS_HIT){break;}hits+=1u;hash=((hash*16777619u)^(hit.nodeIndex*31u+hit.leafIndex))^(hit.voxelOffset+hit.level*65537u)^bitcast<u32>(hit.tEnter)^bitcast<u32>(hit.tExit);minimum=hit.tExit+1e-5;}`;
  return `${webgpuSvoTraversalWGSL}\n${wide}\n${bindings}
@compute @workgroup_size(128) fn benchmarkMain(@builtin(global_invocation_id) id:vec3u){
  let index=id.x;if(index>=${invocationCount}u){return;}let fy=f32((index*1664525u+1013904223u)&65535u)/65536.0;let fz=f32((index*22695477u+1u)&65535u)/65536.0;
  let ray=${rayExpression};
  let mapping=SvoMapping(vec3f(0.0),${brickSize}u,vec3f(1.0),${depth}u,${nodeCount}u,${leafCount}u,256u,0u);
  var hits=0u;var visits=0u;var hash=2166136261u;var status=SVO_STATUS_CONTINUE;var minimum=0.0;${body}
  output[index]=vec4u(hits,status,hash,visits);
}`;
}

function buffer(device: GPUDevice, label: string, data: Uint32Array<ArrayBuffer>, usage = GPUBufferUsage.STORAGE): GPUBuffer {
  const result = device.createBuffer({ label, size: Math.max(4, data.byteLength), usage: usage | GPUBufferUsage.COPY_DST });
  if (data.byteLength) device.queue.writeBuffer(result, 0, data); return result;
}
function median(values: number[]): number { const sorted = [...values].sort((a, b) => a - b); return sorted[Math.floor(sorted.length / 2)]; }

const { create, globals } = await import(pathToFileURL(modulePath).href) as { create(options: string[]): GPU; globals: Record<string, unknown> };
Object.assign(globalThis, globals); const gpu = create(["backend=metal"]);
const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" }); assert.ok(adapter);
assert.ok(adapter.features.has("timestamp-query"));
const device = await adapter.requestDevice({ requiredFeatures: ["timestamp-query"] });
const validationErrors: string[] = [];
device.addEventListener("uncapturederror", (event) => validationErrors.push((event as GPUUncapturedErrorEvent).error.message));
const fixture = denseFixture(), control = buffer(device, "control", new Uint32Array(32));
if (malformedWidePage) fixture.wide.pages[6] += 1;
// The publication fence: full semantic validation (including canonical node and
// leaf cross-checks) runs once here. A rejected publication is never sampled;
// the wide shader sees no capability and the production path stays canonical.
const publishValidation = validateSvoWidePackedPlan(fixture.wide, fixture.widePlan,
  { nodes: fixture.nodes, leaves: fixture.leaves, nodeCount: fixture.nodeCount, leafCount: fixture.leafCount });
assert.equal(publishValidation.status === "ready", !malformedWidePage,
  `publish-time validation must ${malformedWidePage ? "reject a malformed" : "accept a well-formed"} publication`);
const widePublication = publishValidation.status === "ready"
  ? { generation: 1, pageCount: fixture.wide.control[0], descriptorCount: fixture.wide.control[1] }
  : { generation: 0, pageCount: 0, descriptorCount: 0 };
const nodes = buffer(device, "nodes", fixture.nodes), leaves = buffer(device, "leaves", fixture.leaves);
const pages = buffer(device, "wide pages", fixture.wide.pages), descriptors = buffer(device, "wide descriptors", fixture.wide.descriptors);
const outputBytes = invocationCount * 16;
const outputs = new Map<Variant, GPUBuffer>(), readbacks = new Map<Variant, GPUBuffer>();
const targets = new Map<Variant, { pipeline: GPUComputePipeline; bindGroup: GPUBindGroup }>();
for (const variant of ["canonical", "wide"] as const) {
  const output = device.createBuffer({ size: outputBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC }); outputs.set(variant, output);
  readbacks.set(variant, device.createBuffer({ size: outputBytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ }));
  const shaderModule = device.createShaderModule({ code: shader(variant, fixture.nodeCount, fixture.leafCount, widePublication) });
  const info = await shaderModule.getCompilationInfo(); assert.deepEqual(info.messages.filter(x => x.type === "error").map(x => x.message), []);
  const pipeline = device.createComputePipeline({ layout: "auto", compute: { module: shaderModule, entryPoint: "benchmarkMain" } });
  const entries: GPUBindGroupEntry[] = [{ binding: 0, resource: { buffer: control } }, { binding: 2, resource: { buffer: leaves } }];
  // The slimmed wide traversal no longer touches svoNodes; without the
  // canonical fallback composed in, the auto layout drops that binding.
  if (variant === "canonical" || productionFallback) entries.push({ binding: 1, resource: { buffer: nodes } });
  if (variant === "wide") entries.push({ binding: 3, resource: { buffer: pages } }, { binding: 4, resource: { buffer: descriptors } }, { binding: 5, resource: { buffer: output } });
  else entries.push({ binding: 3, resource: { buffer: output } });
  targets.set(variant, { pipeline, bindGroup: device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries }) });
}
for (const variant of ["canonical", "wide"] as const) {
  const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass(), target = targets.get(variant)!;
  pass.setPipeline(target.pipeline); pass.setBindGroup(0, target.bindGroup); pass.dispatchWorkgroups(Math.ceil(invocationCount / 128)); pass.end();
  encoder.copyBufferToBuffer(outputs.get(variant)!, 0, readbacks.get(variant)!, 0, outputBytes); device.queue.submit([encoder.finish()]);
  await readbacks.get(variant)!.mapAsync(GPUMapMode.READ);
}
const canonical = new Uint32Array(readbacks.get("canonical")!.getMappedRange().slice(0));
const wideResult = new Uint32Array(readbacks.get("wide")!.getMappedRange().slice(0));
let canonicalVisits = 0, wideVisits = 0;
let terminalHits = 0;
for (let i = 0; i < canonical.length; i += 4) {
  assert.deepEqual(Array.from(wideResult.slice(i, i + 3)), Array.from(canonical.slice(i, i + 3)), `output mismatch at ray ${i / 4}`);
  canonicalVisits += canonical[i + 3]; wideVisits += wideResult[i + 3];
  terminalHits += wideResult[i];
}
readbacks.get("canonical")!.unmap(); readbacks.get("wide")!.unmap();
for (let i = 0; i < warmups; i += 1) for (const variant of ["canonical", "wide"] as const) {
  const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass(), target = targets.get(variant)!;
  pass.setPipeline(target.pipeline); pass.setBindGroup(0, target.bindGroup); for (let dispatch = 0; dispatch < dispatchesPerSample; dispatch += 1) pass.dispatchWorkgroups(Math.ceil(invocationCount / 128)); pass.end(); device.queue.submit([encoder.finish()]);
}
await device.queue.onSubmittedWorkDone();
const querySet = device.createQuerySet({ type: "timestamp", count: cycles * 4 });
const resolve = device.createBuffer({ size: cycles * 32, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
const timing = device.createBuffer({ size: cycles * 32, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
let query = 0; const encoder = device.createCommandEncoder();
for (let cycle = 0; cycle < cycles; cycle += 1) for (const variant of (cycle % 2 ? ["wide", "canonical"] : ["canonical", "wide"]) as Variant[]) {
  const target = targets.get(variant)!; const pass = encoder.beginComputePass({ timestampWrites: { querySet, beginningOfPassWriteIndex: query, endOfPassWriteIndex: query + 1 } });
  pass.setPipeline(target.pipeline); pass.setBindGroup(0, target.bindGroup); for (let dispatch = 0; dispatch < dispatchesPerSample; dispatch += 1) pass.dispatchWorkgroups(Math.ceil(invocationCount / 128)); pass.end(); query += 2;
}
encoder.resolveQuerySet(querySet, 0, cycles * 4, resolve, 0); encoder.copyBufferToBuffer(resolve, 0, timing, 0, cycles * 32); device.queue.submit([encoder.finish()]);
await timing.mapAsync(GPUMapMode.READ); const stamps = new BigUint64Array(timing.getMappedRange()); query = 0;
const samples: Record<Variant, number[]> = { canonical: [], wide: [] };
for (let cycle = 0; cycle < cycles; cycle += 1) for (const variant of (cycle % 2 ? ["wide", "canonical"] : ["canonical", "wide"]) as Variant[]) {
  samples[variant].push(Number(stamps[query + 1] - stamps[query]) / 1e6 / dispatchesPerSample); query += 2;
}
const canonicalMs = median(samples.canonical), wideMs = median(samples.wide);
console.log(JSON.stringify({ phase: "svo-wide-fanout-gpu-benchmark", adapter: adapter.info, invocationCount, cycles, dispatchesPerSample, originX, productionFallback, rayProfile, malformedWidePage,
  publishValidation: publishValidation.status,
  outputParity: { exact: invocationCount, mismatches: 0 },
  // Per-terminal canonical node cross-validation now happens once at publish,
  // so terminals cost zero structural node records per ray. Leaf records are
  // payload loads (voxel offset) that both variants perform per hit.
  lookups: { canonicalNodeRecords: canonicalVisits, widePageRecords: wideVisits, wideTerminalNodeRecords: 0,
    terminalLeafPayloadRecords: terminalHits,
    wideEffectiveStructuralRecords: wideVisits,
    pageOnlyReductionPercent: (1 - wideVisits / canonicalVisits) * 100,
    effectiveStructuralReductionPercent: (1 - wideVisits / canonicalVisits) * 100 },
  timing: { canonicalMedian_ms: canonicalMs, wideMedian_ms: wideMs, speedup: canonicalMs / wideMs, improvementPercent: (1 - wideMs / canonicalMs) * 100 } }, null, 2));
assert.deepEqual(validationErrors, []);
timing.unmap(); for (const resource of [control, nodes, leaves, pages, descriptors, ...outputs.values(), ...readbacks.values(), resolve, timing]) resource.destroy(); querySet.destroy(); device.destroy();
