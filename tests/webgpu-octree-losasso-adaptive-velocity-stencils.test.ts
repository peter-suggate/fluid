import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { pathToFileURL } from "node:url";
import {
  OCTREE_LOSASSO_ADAPTIVE_VELOCITY_RECEIPT,
  OCTREE_LOSASSO_ADAPTIVE_VELOCITY_STENCIL_CONTROL_WORDS,
  OCTREE_LOSASSO_ADAPTIVE_VELOCITY_STENCIL_WORDS,
  OCTREE_LOSASSO_ADAPTIVE_VELOCITY_STENCILS_PER_NODE,
  planOctreeLosassoAdaptiveVelocity,
} from "../lib/webgpu-octree-losasso-adaptive-velocity";
import {
  octreeLosassoAdaptiveVelocitySamplerWGSL,
  octreeLosassoAdaptiveVelocityWGSL,
}
  from "../lib/webgpu-octree-losasso-adaptive-velocity.wgsl";
import {
  LOSASSO_SURFACE_GRAPH_LEAF_LOCATOR_HEADER_WORDS,
  planLosassoSurfaceGraphLeafLocatorBytes,
} from "../lib/webgpu-octree-losasso-surface-graph";
import { octreeLosassoSurfaceGraphWGSL }
  from "../lib/webgpu-octree-losasso-surface-graph.wgsl";

const functionBody = (name: string, next: string): string => {
  const start = octreeLosassoAdaptiveVelocityWGSL.indexOf(`fn ${name}`);
  const end = octreeLosassoAdaptiveVelocityWGSL.indexOf(next, start);
  assert.ok(start >= 0 && end > start, `missing WGSL section ${name}`);
  return octreeLosassoAdaptiveVelocityWGSL.slice(start, end);
};

test("adaptive velocity prices two fixed compiled face-stencil banks", () => {
  assert.equal(OCTREE_LOSASSO_ADAPTIVE_VELOCITY_STENCIL_WORDS, 36);
  assert.equal(OCTREE_LOSASSO_ADAPTIVE_VELOCITY_STENCILS_PER_NODE, 3);
  assert.equal(OCTREE_LOSASSO_ADAPTIVE_VELOCITY_STENCIL_CONTROL_WORDS, 8);
  const plan = planOctreeLosassoAdaptiveVelocity({
    nodeCapacity: 10, extensionReach: 7, minimumCellWidth: 1,
  });
  assert.equal(plan.accurateExtensionWaves, 6,
    "the accurate causal solve is bounded to the two-cell interface shell");
  assert.equal(plan.extensionWaves, 14,
    "the sparse hierarchy still covers the complete seven-cell transport reach");
  assert.equal(plan.stencilBytes, 10 * 3 * 36 * 4);
  assert.ok(plan.allocatedBytes >= 2 * plan.stencilBytes);
});

test("candidate restriction uses known fine faces and exposes renormalization", () => {
  assert.equal(OCTREE_LOSASSO_ADAPTIVE_VELOCITY_RECEIPT.renormalizedComponents, 9);
  const accumulate = functionBody("avAccumulateStencilFace", "fn avStencilComponent");
  const component = functionBody("avStencilComponent", "@compute @workgroup_size(64)fn reconstructAdaptiveVelocity");
  assert.match(accumulate,
    /reconstructFaceStatus\[slot\]==0u\)\)\{return;\}/,
    "an unknown candidate face must be omitted rather than read as zero");
  assert.doesNotMatch(accumulate,
    /reconstructFaceStatus\[slot\]==0u\)\)\{\(\*valid\)=false/,
    "one unknown fine face must not discard the other known restriction samples");
  assert.match(component, /let complete=covered==s\.header\.y.*!complete.*receiptBase\+9u/s,
    "partial known coverage must publish an explicit renormalization receipt");
  assert.match(component, /vec2f\(value,select\(2\.0,1\.0,complete\)\)/,
    "partial coverage may carry a provisional value but must not publish a valid seed");
  assert.match(component, /avExactValue\(&sum\)\/f32\(covered\)/,
    "known integer area weights must be renormalized by their actual coverage");
  assert.match(component, /covered>s\.header\.y.*AV_ERROR_FACE/s,
    "impossible over-coverage remains fail-closed");
});

test("candidate restriction renormalizes partial fine coverage on GPU", {
  skip: !process.env.WEBGPU_NODE_MODULE && "set WEBGPU_NODE_MODULE for WGSL validation",
}, async () => {
  const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href) as {
    create(options: string[]): GPU; globals: Record<string, unknown>;
  };
  Object.assign(globalThis, dawn.globals);
  const adapter = await dawn.create(["backend=metal"]).requestAdapter(); assert.ok(adapter);
  const device = await adapter.requestDevice({ requiredLimits: {
    maxStorageBuffersPerShaderStage: Math.min(10,
      adapter.limits.maxStorageBuffersPerShaderStage),
  } });
  const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
  const make = (size: number, usage = storage) => device.createBuffer({ size, usage });
  const params = make(64, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
  const graph = make(32), nodes = make(16), stencils = make(3 * 36 * 4);
  const faces = make(8), velocity = make(32), status = make(4), receipt = make(52 * 4);
  const support = make(4), faceStatus = make(8), readback = make(32 + 52 * 4,
    GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
  const paramWords = new Uint32Array(16);
  paramWords.set([1, 0, 0, 6], 0); paramWords[14] = 1;
  device.queue.writeBuffer(params, 0, paramWords);
  device.queue.writeBuffer(graph, 0, new Uint32Array([1, 0, 1, 1, 0, 1, 0, 0]));
  device.queue.writeBuffer(nodes, 0, new Uint32Array([0, 0, 0, 0xffff_ffff]));
  const stencilWords = new Uint32Array(3 * 36);
  stencilWords.set([2, 4, 1, 1, 0, 1, 0xffff_ffff, 0xffff_ffff], 0);
  stencilWords.set([1, 3, 0, 0], 20);
  stencilWords.set([0, 0, 1, 1], 36);
  stencilWords.set([0, 0, 1, 1], 72);
  device.queue.writeBuffer(stencils, 0, stencilWords);
  device.queue.writeBuffer(faces, 0, new Float32Array([2, 10]));
  const module = device.createShaderModule({ code: octreeLosassoAdaptiveVelocityWGSL });
  const pipeline = await device.createComputePipelineAsync({ layout: "auto",
    compute: { module, entryPoint: "reconstructAdaptiveVelocity" } });
  const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [
    { binding: 0, resource: { buffer: params } },
    { binding: 11, resource: { buffer: graph } },
    { binding: 12, resource: { buffer: nodes } },
    { binding: 13, resource: { buffer: stencils } },
    { binding: 14, resource: { buffer: faces } },
    { binding: 15, resource: { buffer: velocity } },
    { binding: 16, resource: { buffer: status } },
    { binding: 17, resource: { buffer: receipt } },
    { binding: 18, resource: { buffer: support } },
    { binding: 19, resource: { buffer: faceStatus } },
  ] });
  const run = async (known: readonly [number, number]) => {
    device.queue.writeBuffer(velocity, 0, new Uint32Array(8));
    device.queue.writeBuffer(status, 0, new Uint32Array(1));
    device.queue.writeBuffer(receipt, 0, new Uint32Array(52));
    device.queue.writeBuffer(faceStatus, 0, new Uint32Array(known));
    const encoder = device.createCommandEncoder(); const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline); pass.setBindGroup(0, group); pass.dispatchWorkgroups(1); pass.end();
    encoder.copyBufferToBuffer(velocity, 0, readback, 0, 32);
    encoder.copyBufferToBuffer(receipt, 0, readback, 32, 52 * 4);
    device.pushErrorScope("validation"); device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone(); assert.equal((await device.popErrorScope())?.message,
      undefined);
    await readback.mapAsync(GPUMapMode.READ); const bytes = readback.getMappedRange().slice(0);
    readback.unmap(); return bytes;
  };
  const partial = await run([1, 0]);
  assert.equal(new Float32Array(partial)[0], 2,
    "the only known face must retain its physical value after renormalization");
  assert.equal(new Uint32Array(partial)[3], 0,
    "partial coverage must remain invalid so interface extension repairs it");
  assert.equal(new Uint32Array(partial)[8 + 9], 1,
    "the partial restriction must publish one renormalized component");
  const complete = await run([1, 1]);
  assert.equal(new Float32Array(complete)[0], 8,
    "complete coverage must retain the compiled 1:3 area weighting");
  assert.equal(new Uint32Array(complete)[3], 1,
    "complete coverage is an authoritative extension seed");
  assert.equal(new Uint32Array(complete)[8 + 9], 0);
  for (const buffer of [params, graph, nodes, stencils, faces, velocity, status, receipt,
    support, faceStatus, readback]) buffer.destroy();
  device.destroy();
});

test("adaptive extrapolation keeps an accurate two-cell shell and sparse outer closure", () => {
  const extend = functionBody("extendAdaptiveVelocity", "fn avCanonicalSum");
  assert.match(extend, /let accurateReach=min\(avp\.reach,bitcast<f32>\(avp\.pad1\.y\)\)/);
  assert.match(extend, /abs\(nodePhi\)>accurateReach/);
  const source = readFileSync(new URL("../lib/webgpu-octree-losasso-adaptive-velocity.ts",
    import.meta.url), "utf8");
  assert.match(source,
    /wave < this\.plan\.accurateExtensionWaves[\s\S]*extendAdaptiveVelocity[\s\S]*wave < this\.plan\.extensionWaves[\s\S]*closeAdaptiveVelocity/,
    "the causal shell must use its short schedule while sparse closure retains full reach");
});

test("recurring nodal reconstruction consumes only fixed compiled stencil records", () => {
  const component = functionBody("avStencilComponent", "@compute @workgroup_size(64)fn reconstructAdaptiveVelocity");
  const reconstruct = functionBody("reconstructAdaptiveVelocity", "@group(0)@binding(111)");
  const accesses = component.match(/avAccumulateStencilFace\(/g) ?? [];
  assert.equal(accesses.length, 16, "each component must use sixteen explicitly unrolled slots");
  assert.doesNotMatch(component, /\b(?:for|while)\s*\(|\bloop\s*\{/);
  assert.doesNotMatch(reconstruct,
    /avStencilContainingFace|avStencilExactFace|FaceDirectory|\b(?:for|while)\s*\(|\bloop\s*\{/);
  assert.equal((reconstruct.match(/avStencilComponent\(node,/g) ?? []).length, 3);
});

test("candidate-only velocity handoff preserves exact accepted nodes", () => {
  const handoff = functionBody("handoffAdaptiveVelocity", "@group(0)@binding(61)");
  assert.match(handoff, /handoffAcceptedDirectory/);
  assert.match(handoff, /while\(low<high\)/);
  assert.match(handoff, /handoffAcceptedGraph\[6u\]!=handoffAcceptedGraph\[5u\]/);
  assert.match(handoff, /handoffStatus\[node\]=next/);
  assert.doesNotMatch(handoff, /handoffStatus\[node\]==AV_VALID_COMPONENTS/,
    "fully reconstructed candidate nodes must still receive exact coincident values");
  assert.match(handoff, /if\(\(sourceMask&bit\)!=0u\)/,
    "accepted components overwrite candidate face reconstruction at coincident nodes");
  assert.doesNotMatch(handoff, /\(old&bit\)==0u&&\(sourceMask&bit\)!=0u/);
});

test("face directory searches are confined to the fail-closed topology compiler", () => {
  const builder = functionBody("avStencilExactFace", "@group(0)@binding(101)");
  assert.match(builder, /stencilFaceDirectory/);
  assert.match(builder, /avStencilContainingFace/);
  assert.match(builder, /if\(\(\*count\)>=16u\)\{return false;\}/);
  assert.match(builder, /atomicOr\(&stencilControl\[4u\],AV_ERROR_CAPACITY\)/);
});

test("adaptive velocity demand follows the compiled transport closure, not clamped phi", () => {
  const mark = functionBody("markAdaptiveVelocitySupport", "@group(0)@binding(21)");
  const finalize = functionBody("finalizeAdaptiveVelocity", "@group(0)@binding(51)");
  assert.match(mark, /supportBandMask\[node\]==0u/);
  assert.doesNotMatch(mark, /abs\(supportPhi/);
  assert.match(finalize,
    /let demanded=node<arrayLength\(&finalSupport\)&&finalSupport\[node\]!=0u/);
  assert.match(finalize, /if\(demanded&&mask!=AV_VALID_COMPONENTS\)/);
  assert.doesNotMatch(finalize, /magnitude<=avp\.reach&&mask!=AV_VALID_COMPONENTS/);
});

test("adaptive point sampling uses a fixed sparse owner-page locator", () => {
  assert.equal(LOSASSO_SURFACE_GRAPH_LEAF_LOCATOR_HEADER_WORDS, 16);
  assert.equal(planLosassoSurfaceGraphLeafLocatorBytes(7), (16 + 7 * 512) * 4,
    "locator allocation must scale with resident page capacity, not domain volume");
  const sampler = octreeLosassoAdaptiveVelocitySamplerWGSL();
  const lookupStart = sampler.indexOf("fn adaptiveLocatedLeaf");
  const lookupEnd = sampler.indexOf("fn adaptiveSampleProduct3", lookupStart);
  const sampleStart = sampler.indexOf("fn sampleAdaptiveVelocityGrid");
  const sampleEnd = sampler.indexOf("fn sampleAdaptiveVelocity(", sampleStart);
  assert.ok(lookupStart >= 0 && lookupEnd > lookupStart && sampleEnd > sampleStart);
  const recurring = sampler.slice(lookupStart, lookupEnd)
    + sampler.slice(sampleStart, sampleEnd);
  assert.doesNotMatch(recurring,
    /adaptiveExactLeaf|adaptiveContainingLeaf|LeafDirectory|hash|probe|\b(?:for|while)\s*\(|\bloop\s*\{/);
  assert.match(recurring, /adaptiveVelocityOwnerArena\[directory\]/);
  assert.match(recurring, /adaptiveVelocityLeafLocator\[at\]/);
  assert.match(recurring, /adaptiveVelocityOwnerArena\[7u\]!=epoch/,
    "owner and graph locator epochs must agree before any sample is accepted");
});

test("surface graph compiles and transactionally commits only owner support cells", () => {
  const compilerStart = octreeLosassoSurfaceGraphWGSL.indexOf(
    "fn compileSurfaceGraphLeafLocator");
  const compilerEnd = octreeLosassoSurfaceGraphWGSL.indexOf(
    "fn isBoundaryCoordinate", compilerStart);
  assert.ok(compilerStart >= 0 && compilerEnd > compilerStart);
  const compiler = octreeLosassoSurfaceGraphWGSL.slice(compilerStart, compilerEnd);
  assert.match(compiler, /item>=support/);
  assert.match(compiler, /item\/512u/);
  assert.match(compiler, /atomicAdd\(&candidateLeafLocator\[12\],1u\)/);
  assert.match(octreeLosassoSurfaceGraphWGSL,
    /fn commitSurfaceGraphLeafLocator[\s\S]*candidateLeafLocator\[at\]/);
  assert.match(octreeLosassoSurfaceGraphWGSL,
    /atomicStore\(&acceptedLeafLocator\[6\],epoch\);atomicStore\(&acceptedControl\[3\],epoch\)/,
    "locator publication must precede the accepted graph publication in one transaction");
  assert.doesNotMatch(compiler, /dimensionsMaximum\.x\s*\*|dims\(\)\.x\s*\*/,
    "the compiler must dispatch resident support cells rather than a full domain lattice");
});
