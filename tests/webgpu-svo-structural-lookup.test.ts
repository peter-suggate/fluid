import assert from "node:assert/strict";
import { pathToFileURL } from "node:url";
import test from "node:test";

import {
  SPARSE_BRICK_GPU_LAYOUT,
  packMaterialOwner,
  packSparseBrickPlan,
  planSparseBrickOctree,
} from "../lib/sparse-brick-octree";
import {
  lookupSvoStructuralCoarseFluidCell,
  svoStructuralCoarseFluidSamplingWGSL,
  type SvoStructuralFluidPackedFixture,
} from "../lib/svo-fluid-structural-sampling";
import {
  SVO_PAYLOAD_DDA_STATUS,
  svoStructuralPayloadDdaWGSL,
  traverseSvoLeafPayload,
} from "../lib/svo-structural-payload-traversal";
import { FLUID_BRICK_RESIDENT } from "../lib/webgpu-fluid-brick-residency";
import {
  SVO_WGSL_STATUS,
  createWebgpuSvoTraversalWGSL,
  traversePackedSvo,
  type SvoRay,
  type SvoWorldMapping,
} from "../lib/webgpu-svo-traversal";
import { SPARSE_VOXEL_PUBLICATION_STATE, SPARSE_VOXEL_VALID_FIELDS } from "../lib/webgpu-voxel-debug";

const modulePath = process.env.WEBGPU_NODE_MODULE;
const mapping: SvoWorldMapping = { origin: [10, 20, 30], cellSize: [0.5, 1, 2], brickSize: 4, maximumDepth: 1 };
const plan = planSparseBrickOctree([{ x: 0, y: 0, z: 0 }, { x: 1, y: 0, z: 0 }], { brickSize: 4, maximumDepth: 1 });
const packed = packSparseBrickPlan(plan, 7);

function payloadIndex(leafIndex: number, x: number, y: number, z: number): number {
  return plan.leaves[leafIndex].voxelOffset + x + y * 4 + z * 16;
}

function makeFixture(): SvoStructuralFluidPackedFixture & { materialOwners: Uint32Array } {
  const control = new Uint32Array(32);
  control[SPARSE_BRICK_GPU_LAYOUT.controlWords.publishedNodes] = plan.nodes.length;
  control[SPARSE_BRICK_GPU_LAYOUT.controlWords.publishedLeaves] = plan.leaves.length;
  control[SPARSE_BRICK_GPU_LAYOUT.controlWords.publishedVoxels] = plan.voxelCount;
  control[SPARSE_BRICK_GPU_LAYOUT.controlWords.generation] = 7;
  control[SPARSE_BRICK_GPU_LAYOUT.controlWords.brickSize] = 4;
  const publicationState = new Uint32Array(SPARSE_VOXEL_PUBLICATION_STATE.strideBytes / 4);
  publicationState[SPARSE_VOXEL_PUBLICATION_STATE.completeGeneration] = 7;
  publicationState[SPARSE_VOXEL_PUBLICATION_STATE.validFields] = SPARSE_VOXEL_VALID_FIELDS.topology | SPARSE_VOXEL_VALID_FIELDS.coarseFluid;
  const geometry = new Float32Array(plan.voxelCount * 4);
  for (let index = 0; index < plan.voxelCount; index += 1) geometry[index * 4] = index * 0.125 - 3;
  const materialOwners = new Uint32Array(plan.voxelCount);
  materialOwners[payloadIndex(0, 1, 1, 1)] = packMaterialOwner(41, 7);
  materialOwners[payloadIndex(0, 2, 1, 1)] = packMaterialOwner(42, 8);
  materialOwners[payloadIndex(0, 3, 2, 2)] = packMaterialOwner(44, 11);
  materialOwners[payloadIndex(1, 0, 1, 1)] = packMaterialOwner(53, 10);
  materialOwners[payloadIndex(1, 2, 1, 1)] = packMaterialOwner(55, 9);
  return {
    control,
    nodes: new Uint32Array(packed.nodes),
    leaves: new Uint32Array(packed.leaves),
    geometry,
    materialOwners,
    fluidLeafStates: new Uint32Array(plan.leaves.length).fill(FLUID_BRICK_RESIDENT),
    publicationState,
    domain: { worldOrigin_m: mapping.origin, cellSize_m: mapping.cellSize, dimensionsCells: [8, 4, 4], brickSize: 4, maximumDepth: 1 },
    expectedCompleteGeneration: 7,
  };
}

interface RayCase { name: string; ray: SvoRay; ddaBudget: number; publishedVoxels: number; traversalBudget: number }
const rays: RayCase[] = [
  { name: "axis forward", ray: { origin: [9, 21.5, 33], direction: [1, 0, 0], tMax: 20 }, ddaBudget: 32, publishedVoxels: plan.voxelCount, traversalBudget: 256 },
  { name: "axis backward nearest", ray: { origin: [15, 21.5, 33], direction: [-1, 0, 0], tMax: 20 }, ddaBudget: 32, publishedVoxels: plan.voxelCount, traversalBudget: 256 },
  { name: "shared face", ray: { origin: [9, 21, 33], direction: [1, 0, 0], tMax: 20 }, ddaBudget: 32, publishedVoxels: plan.voxelCount, traversalBudget: 256 },
  { name: "corner tie", ray: { origin: [9.5, 19, 28], direction: [0.5, 1, 2], tMax: 20 }, ddaBudget: 32, publishedVoxels: plan.voxelCount, traversalBudget: 256 },
  { name: "inside occupied voxel", ray: { origin: [10.75, 21.5, 33], direction: [1, 0, 0], tMax: 20 }, ddaBudget: 32, publishedVoxels: plan.voxelCount, traversalBudget: 256 },
  { name: "payload miss", ray: { origin: [9, 23.5, 37], direction: [1, 0, 0], tMax: 20 }, ddaBudget: 32, publishedVoxels: plan.voxelCount, traversalBudget: 256 },
  { name: "payload exhaustion", ray: { origin: [9, 22.5, 35], direction: [1, 0, 0], tMax: 20 }, ddaBudget: 1, publishedVoxels: plan.voxelCount, traversalBudget: 256 },
  { name: "traversal exhaustion", ray: { origin: [9, 21.5, 33], direction: [1, 0, 0], tMax: 20 }, ddaBudget: 32, publishedVoxels: plan.voxelCount, traversalBudget: 1 },
  { name: "invalid payload publication", ray: { origin: [9, 21.5, 33], direction: [1, 0, 0], tMax: 20 }, ddaBudget: 32, publishedVoxels: 4, traversalBudget: 256 },
];

test("CPU structural leaf DDA returns payload distance, voxel, material, owner, and explicit exhaustion", () => {
  const fixture = makeFixture();
  for (const entry of rays) {
    const traversal = traversePackedSvo(entry.ray, {
      nodes: fixture.nodes, leaves: fixture.leaves, publishedNodeCount: plan.nodes.length, publishedLeafCount: plan.leaves.length,
    }, mapping, { maxNodeVisits: entry.traversalBudget });
    if (traversal.status !== "hit") {
      assert.equal(entry.name, "traversal exhaustion");
      assert.equal(traversal.status, "work-exhausted");
      continue;
    }
    const result = traverseSvoLeafPayload(entry.ray, traversal.hit, fixture.materialOwners, mapping, {
      maxVoxelVisits: entry.ddaBudget, publishedVoxelCount: entry.publishedVoxels,
    });
    if (entry.name === "payload exhaustion") assert.equal(result.status, "work-exhausted");
    else if (entry.name === "invalid payload publication") assert.equal(result.status, "invalid");
    else if (entry.name === "payload miss") assert.equal(result.status, "miss");
    else assert.equal(result.status, "hit", entry.name);
  }
});

test("structural GPU oracle composes production topology and binding-free payload/phi helpers", () => {
  const source = createWebgpuSvoTraversalWGSL();
  assert.match(source, /fn svoTraverse\(/);
  assert.match(svoStructuralPayloadDdaWGSL, /fn svoTraverseLeafPayload\(/);
  assert.match(svoStructuralPayloadDdaWGSL, /packed&0xffffu/);
  assert.doesNotMatch(source + svoStructuralPayloadDdaWGSL + svoStructuralCoarseFluidSamplingWGSL, /DebugRecord|voxelRecords|brickRecords/);
});

const rayShader = `${createWebgpuSvoTraversalWGSL()}
@group(0) @binding(3) var<storage,read> svoMaterialOwners:array<u32>;
${svoStructuralPayloadDdaWGSL}
struct OracleRayInput { ray:SvoRay, limits:vec4u }
struct OracleRayOutput { traversal:SvoTraversalHit, payload:SvoPayloadDdaHit, boundsMinimum:vec4f, boundsMaximum:vec4f }
@group(0) @binding(4) var<storage,read> oracleRays:array<OracleRayInput>;
@group(0) @binding(5) var<storage,read_write> oracleResults:array<OracleRayOutput>;
@group(0) @binding(6) var<uniform> oracleMapping:SvoMapping;
@compute @workgroup_size(1)
fn structuralRayOracle(@builtin(global_invocation_id) id:vec3u){
  if(id.x>=arrayLength(&oracleResults)||id.x>=arrayLength(&oracleRays)){return;}
  let input=oracleRays[id.x];var rayMapping=oracleMapping;rayMapping.maxVisits=input.limits.z;
  let traversal=svoTraverse(input.ray,rayMapping);var payload=svoPayloadDdaMiss(SVO_PAYLOAD_STATUS_INVALID,0u,traversal.leafIndex);
  var minimum=vec3f(0.0);var maximum=vec3f(0.0);
  if(traversal.status==SVO_STATUS_HIT){let bounds=svoNodeBounds(svoNodes[traversal.nodeIndex],rayMapping);minimum=bounds[0];maximum=bounds[1];payload=svoTraverseLeafPayload(input.ray,rayMapping,traversal,input.limits.y,input.limits.x);}
  oracleResults[id.x]=OracleRayOutput(traversal,payload,vec4f(minimum,0.0),vec4f(maximum,0.0));
}`;

const pointShader = `
@group(0) @binding(0) var<storage,read> svoStructuralControl:array<u32>;
@group(0) @binding(1) var<storage,read> svoStructuralNodes:array<u32>;
@group(0) @binding(2) var<storage,read> svoStructuralLeaves:array<u32>;
@group(0) @binding(3) var<storage,read> svoStructuralGeometry:array<vec4f>;
@group(0) @binding(4) var<storage,read> svoStructuralLeafStates:array<u32>;
@group(0) @binding(5) var<storage,read> svoStructuralPublication:array<u32>;
${svoStructuralCoarseFluidSamplingWGSL}
@group(0) @binding(6) var<uniform> oracleDomain:SvoStructuralSamplingDomain;
@group(0) @binding(7) var<uniform> oracleCell:vec4i;
@group(0) @binding(8) var<storage,read_write> oraclePointResult:array<SvoStructuralCellSample>;
@compute @workgroup_size(1)
fn structuralPointOracle(){oraclePointResult[0]=svoStructuralCoarseFluidCell(oracleDomain,oracleCell.xyz);}`;

function gpuBytes(data: ArrayBufferView): Uint8Array<ArrayBuffer> {
  const copy = new Uint8Array(data.byteLength);
  copy.set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
  return copy;
}

function bufferWith(device: GPUDevice, data: ArrayBufferView, usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST): GPUBuffer {
  const buffer = device.createBuffer({ size: Math.max(16, Math.ceil(data.byteLength / 4) * 4), usage });
  device.queue.writeBuffer(buffer, 0, gpuBytes(data));
  return buffer;
}

test("Metal structural lookup and DDA agree with CPU across boundaries and publication failures", {
  skip: !modulePath && "set WEBGPU_NODE_MODULE for GPU structural lookup checks",
}, async () => {
  const { create, globals } = await import(pathToFileURL(modulePath!).href) as { create(options: string[]): GPU; globals: Record<string, unknown> };
  Object.assign(globalThis, globals);
  const gpu = create(["backend=metal"]); const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" }); assert.ok(adapter);
  const device = await adapter.requestDevice(); const validationErrors: string[] = [];
  device.addEventListener("uncapturederror", (event: unknown) => validationErrors.push((event as { error: { message: string } }).error.message));
  const fixture = makeFixture();
  const buffers: GPUBuffer[] = [];
  const own = (buffer: GPUBuffer) => { buffers.push(buffer); return buffer; };
  try {
    const control = own(bufferWith(device, fixture.control)); const nodes = own(bufferWith(device, fixture.nodes));
    const leaves = own(bufferWith(device, fixture.leaves)); const owners = own(bufferWith(device, fixture.materialOwners));
    const rayWords = new ArrayBuffer(rays.length * 48); const rayFloats = new Float32Array(rayWords); const rayU32 = new Uint32Array(rayWords);
    rays.forEach((entry, index) => {
      const base = index * 12; rayFloats.set([...entry.ray.origin, entry.ray.tMin ?? 0, ...entry.ray.direction, entry.ray.tMax ?? 1e6], base);
      rayU32.set([entry.ddaBudget, entry.publishedVoxels, entry.traversalBudget, 0], base + 8);
    });
    const rayInputs = own(bufferWith(device, new Uint8Array(rayWords)));
    const mappingWords = new ArrayBuffer(48); const mappingFloats = new Float32Array(mappingWords); const mappingU32 = new Uint32Array(mappingWords);
    mappingFloats.set(mapping.origin, 0); mappingU32[3] = 4; mappingFloats.set(mapping.cellSize, 4); mappingU32[7] = 1;
    mappingU32.set([plan.nodes.length, plan.leaves.length, 256, 0], 8);
    const mappingBuffer = own(bufferWith(device, new Uint8Array(mappingWords), GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST));
    const rayResultBytes = rays.length * 112;
    const rayResults = own(device.createBuffer({ size: rayResultBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC }));
    const rayReadback = own(device.createBuffer({ size: rayResultBytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ }));
    const rayModule = device.createShaderModule({ code: rayShader });
    assert.deepEqual((await rayModule.getCompilationInfo()).messages.filter((message) => message.type === "error"), []);
    const rayPipeline = await device.createComputePipelineAsync({ layout: "auto", compute: { module: rayModule, entryPoint: "structuralRayOracle" } });
    const rayBindGroup = device.createBindGroup({ layout: rayPipeline.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: control } }, { binding: 1, resource: { buffer: nodes } }, { binding: 2, resource: { buffer: leaves } },
      { binding: 3, resource: { buffer: owners } }, { binding: 4, resource: { buffer: rayInputs } }, { binding: 5, resource: { buffer: rayResults } },
      { binding: 6, resource: { buffer: mappingBuffer } },
    ] });
    const encoder = device.createCommandEncoder(); const pass = encoder.beginComputePass(); pass.setPipeline(rayPipeline); pass.setBindGroup(0, rayBindGroup); pass.dispatchWorkgroups(rays.length); pass.end();
    encoder.copyBufferToBuffer(rayResults, 0, rayReadback, 0, rayResultBytes); device.queue.submit([encoder.finish()]);
    await rayReadback.mapAsync(GPUMapMode.READ); const rayCopy = rayReadback.getMappedRange().slice(0); rayReadback.unmap();
    const gpuWords = new Uint32Array(rayCopy); const gpuFloats = new Float32Array(rayCopy);
    rays.forEach((entry, index) => {
      const base = index * 28;
      const cpuTraversal = traversePackedSvo(entry.ray, { nodes: fixture.nodes, leaves: fixture.leaves, publishedNodeCount: plan.nodes.length, publishedLeafCount: plan.leaves.length }, mapping, { maxNodeVisits: entry.traversalBudget });
      const traversalStatus = cpuTraversal.status === "hit" ? SVO_WGSL_STATUS.hit : cpuTraversal.status === "work-exhausted" ? SVO_WGSL_STATUS.workExhausted : SVO_WGSL_STATUS.miss;
      assert.equal(gpuWords[base], traversalStatus, `${entry.name}: traversal status`);
      if (cpuTraversal.status !== "hit") return;
      assert.deepEqual(Array.from(gpuWords.slice(base + 2, base + 6)), [cpuTraversal.hit.nodeIndex, cpuTraversal.hit.leafIndex, cpuTraversal.hit.voxelOffset, cpuTraversal.hit.level], `${entry.name}: leaf identity`);
      for (let axis = 0; axis < 3; axis += 1) {
        assert.ok(Math.abs(gpuFloats[base + 20 + axis] - cpuTraversal.hit.bounds.minimum[axis]) < 1e-5, `${entry.name}: minimum ${axis}`);
        assert.ok(Math.abs(gpuFloats[base + 24 + axis] - cpuTraversal.hit.bounds.maximum[axis]) < 1e-5, `${entry.name}: maximum ${axis}`);
      }
      const cpuPayload = traverseSvoLeafPayload(entry.ray, cpuTraversal.hit, fixture.materialOwners, mapping, { maxVoxelVisits: entry.ddaBudget, publishedVoxelCount: entry.publishedVoxels });
      const expectedStatus = cpuPayload.status === "hit" ? SVO_PAYLOAD_DDA_STATUS.hit : cpuPayload.status === "miss" ? SVO_PAYLOAD_DDA_STATUS.miss : cpuPayload.status === "invalid" ? SVO_PAYLOAD_DDA_STATUS.invalid : SVO_PAYLOAD_DDA_STATUS.workExhausted;
      assert.equal(gpuWords[base + 8], expectedStatus, `${entry.name}: payload status`);
      assert.equal(gpuWords[base + 9], cpuPayload.visits, `${entry.name}: DDA visits`);
      if (cpuPayload.status === "hit") {
        assert.equal(gpuWords[base + 10], cpuPayload.payloadIndex, `${entry.name}: payload offset`);
        assert.deepEqual(Array.from(gpuWords.slice(base + 12, base + 16)), [cpuPayload.materialId, cpuPayload.ownerId, cpuPayload.local[0], cpuPayload.local[1]], `${entry.name}: material/owner/local xy`);
        assert.equal(gpuFloats[base + 16], cpuPayload.local[2], `${entry.name}: local z`);
        assert.ok(Math.abs(gpuFloats[base + 17] - cpuPayload.tEnter) < 2e-5, `${entry.name}: entry distance`);
      }
    });

    const geometry = own(bufferWith(device, fixture.geometry)); const states = own(bufferWith(device, fixture.fluidLeafStates));
    const publication = own(bufferWith(device, fixture.publicationState));
    const domainWords = new ArrayBuffer(64); const domainFloats = new Float32Array(domainWords); const domainU32 = new Uint32Array(domainWords);
    domainFloats.set([...mapping.origin, 0], 0); domainFloats.set([...mapping.cellSize, 0], 4); domainU32.set([8, 4, 4, 4], 8); domainU32.set([1, 7, 0, 0], 12);
    const domain = own(bufferWith(device, new Uint8Array(domainWords), GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST));
    const cell = own(bufferWith(device, new Int32Array([4, 1, 1, 0]), GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST));
    const pointResult = own(device.createBuffer({ size: 32, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC }));
    const pointModule = device.createShaderModule({ code: pointShader });
    assert.deepEqual((await pointModule.getCompilationInfo()).messages.filter((message) => message.type === "error"), []);
    const pointPipeline = await device.createComputePipelineAsync({ layout: "auto", compute: { module: pointModule, entryPoint: "structuralPointOracle" } });
    const pointBindGroup = device.createBindGroup({ layout: pointPipeline.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: control } }, { binding: 1, resource: { buffer: nodes } }, { binding: 2, resource: { buffer: leaves } },
      { binding: 3, resource: { buffer: geometry } }, { binding: 4, resource: { buffer: states } }, { binding: 5, resource: { buffer: publication } },
      { binding: 6, resource: { buffer: domain } }, { binding: 7, resource: { buffer: cell } }, { binding: 8, resource: { buffer: pointResult } },
    ] });
    const runPoint = async () => {
      const readback = own(device.createBuffer({ size: 32, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ }));
      const command = device.createCommandEncoder(); const compute = command.beginComputePass(); compute.setPipeline(pointPipeline); compute.setBindGroup(0, pointBindGroup); compute.dispatchWorkgroups(1); compute.end();
      command.copyBufferToBuffer(pointResult, 0, readback, 0, 32); device.queue.submit([command.finish()]); await readback.mapAsync(GPUMapMode.READ);
      const copy = readback.getMappedRange().slice(0); readback.unmap(); return { words: new Uint32Array(copy), floats: new Float32Array(copy) };
    };
    let point = await runPoint(); const cpuPoint = lookupSvoStructuralCoarseFluidCell(fixture, [4, 1, 1]); assert.equal(cpuPoint.status, "valid");
    assert.equal(point.words[0], 1); if (cpuPoint.status === "valid") { assert.deepEqual(Array.from(point.words.slice(1, 4)), [cpuPoint.nodeIndex, cpuPoint.leafIndex, cpuPoint.voxelIndex]); assert.ok(Math.abs(point.floats[4] - cpuPoint.phi_m) < 1e-6); }
    domainU32[13] = 8; device.queue.writeBuffer(domain, 0, new Uint8Array(domainWords)); point = await runPoint(); assert.equal(point.words[0], 2, "stale generation is invalid");
    domainU32[13] = 7; device.queue.writeBuffer(domain, 0, new Uint8Array(domainWords)); const rightLeaf = plan.leaves.find((leaf) => leaf.coordinate.x === 1)!.index;
    fixture.fluidLeafStates[rightLeaf] = 0; device.queue.writeBuffer(states, 0, gpuBytes(fixture.fluidLeafStates)); point = await runPoint(); assert.equal(point.words[0], 2, "retired leaf is invalid");
    fixture.fluidLeafStates[rightLeaf] = FLUID_BRICK_RESIDENT; device.queue.writeBuffer(states, 0, gpuBytes(fixture.fluidLeafStates));
    fixture.publicationState[SPARSE_VOXEL_PUBLICATION_STATE.completeGeneration] = 0; device.queue.writeBuffer(publication, 0, gpuBytes(fixture.publicationState)); point = await runPoint(); assert.equal(point.words[0], 2, "unpublished generation is invalid");
    fixture.publicationState[SPARSE_VOXEL_PUBLICATION_STATE.completeGeneration] = 7; device.queue.writeBuffer(publication, 0, gpuBytes(fixture.publicationState));
    const savedBacklink = fixture.leaves[rightLeaf * 4]; fixture.leaves[rightLeaf * 4] = 0; device.queue.writeBuffer(leaves, 0, gpuBytes(fixture.leaves)); point = await runPoint(); assert.equal(point.words[0], 2, "invalid topology is invalid");
    fixture.leaves[rightLeaf * 4] = savedBacklink; device.queue.writeBuffer(leaves, 0, gpuBytes(fixture.leaves));
    await device.queue.onSubmittedWorkDone(); assert.deepEqual(validationErrors, []);
  } finally {
    for (const buffer of buffers) buffer.destroy(); device.destroy();
  }
});
