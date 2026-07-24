import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { decodeGeneratedOctreePowerCatalog } from "../lib/generated/octree-power-catalog";
import { OCTREE_POWER_CATALOG_FACE_FLOATS } from "../lib/octree-power-catalog";
import { OCTREE_POWER_FACE_RECORD_BYTES, OCTREE_POWER_INVALID_ROW } from "../lib/octree-power-operator";
import { PassBroker } from "../lib/webgpu-pass-broker";
import {
  OCTREE_POWER_FACE_BOUNDARY,
  OCTREE_POWER_FACE_EMPTY_FINE_BINDINGS_BYTES,
  OCTREE_POWER_CLOSED_BOUNDARY_MASK_WITH_CLOSED_TOP,
  OCTREE_POWER_CLOSED_BOUNDARY_MASK_WITH_OPEN_TOP,
  OCTREE_POWER_FACE_ERROR,
  OCTREE_POWER_FACE_OPEN_BOUNDARY,
  OCTREE_POWER_FACE_QUADRATURE_BYTES,
  OCTREE_POWER_FACE_VALID,
  OCTREE_POWER_FACE_WORLD_BOUNDARY_SHIFT,
  WebGPUOctreePowerFaces,
  octreePowerClosedBoundaryMask,
  octreePowerBoundaryCoarsePhiShader,
  octreePowerBoundaryPhiShader,
  octreePowerFaceShader,
  mergeOctreePowerFaceIdentities,
  planOctreePowerFaces,
  unpackOctreePowerFaceControl,
} from "../lib/webgpu-octree-power-faces";
import { OCTREE_POWER_TOPOLOGY_VALID, type OctreePowerTopologySource } from "../lib/webgpu-octree-power-topology";
import {
  createColdPowerRowPublication,
  createIdentityPowerRowPublication,
} from "./webgpu-octree-power-row-delta-fixture";

// The native Metal binding can release a shared Dawn instance while a later
// test in this file is still creating pipelines. Retain wrappers/devices until
// process exit; submitted work and mapped readbacks remain the correctness gate.
const retainedNativeGPUs: GPU[] = [];
const retainedDevices: GPUDevice[] = [];
const octreeProjectionSource = readFileSync(new URL("../lib/webgpu-octree.ts", import.meta.url), "utf8");
const powerFaceSource = readFileSync(new URL("../lib/webgpu-octree-power-faces.ts", import.meta.url), "utf8");

function catalogViews() {
  const bytes = readFileSync(new URL("../lib/generated/octree-power-catalog.bin", import.meta.url));
  return decodeGeneratedOctreePowerCatalog(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
}

function morton3(cell: number, dimensions: readonly [number, number, number]): number {
  const part10 = (coordinate: number) => {
    let value = coordinate & 1023;
    value = (value | (value << 16)) & 0x030000ff;
    value = (value | (value << 8)) & 0x0300f00f;
    value = (value | (value << 4)) & 0x030c30c3;
    value = (value | (value << 2)) & 0x09249249;
    return value >>> 0;
  };
  const x = cell % dimensions[0];
  const y = Math.floor(cell / dimensions[0]) % dimensions[1];
  const z = Math.floor(cell / (dimensions[0] * dimensions[1]));
  return (part10(x) | (part10(y) << 1) | (part10(z) << 2)) >>> 0;
}

function reachableBindings(shader: string, entryPoint: string): number[] {
  const globals = new Map<string, number>();
  for (const match of shader.matchAll(
    /@group\(0\)\s*@binding\((\d+)\)\s*var(?:<[^>]+>)?\s+([A-Za-z_]\w*)/g,
  )) globals.set(match[2], Number(match[1]));
  const bodies = new Map<string, string>();
  for (const match of shader.matchAll(/\bfn\s+([A-Za-z_]\w*)\s*\(/g)) {
    const open = shader.indexOf("{", match.index); let depth = 0; let close = -1;
    for (let at = open; at < shader.length; at += 1) {
      if (shader[at] === "{") depth += 1;
      if (shader[at] === "}" && --depth === 0) { close = at; break; }
    }
    assert.ok(open >= 0 && close > open, `WGSL function ${match[1]} must be complete`);
    bodies.set(match[1], shader.slice(open + 1, close));
  }
  const pending = [entryPoint]; const reached = new Set<string>(); const bindings = new Set<number>();
  while (pending.length > 0) {
    const name = pending.pop()!;
    if (reached.has(name)) continue;
    reached.add(name);
    const body = bodies.get(name);
    assert.notEqual(body, undefined, `reachable WGSL function ${name} must exist`);
    for (const [global, binding] of globals) {
      if (new RegExp(`\\b${global}\\b`).test(body!)) bindings.add(binding);
    }
    for (const callee of bodies.keys()) {
      if (!reached.has(callee) && new RegExp(`\\b${callee}\\s*\\(`).test(body!)) pending.push(callee);
    }
  }
  return [...bindings].sort((left, right) => left - right);
}

function reachableStorageBindings(shader: string, entryPoint: string): number[] {
  const storageBindings = new Set([...shader.matchAll(
    /@group\(0\)\s*@binding\((\d+)\)\s*var<storage(?:,[^>]+)?>/g,
  )].map((match) => Number(match[1])));
  return reachableBindings(shader, entryPoint).filter((binding) => storageBindings.has(binding));
}

test("every power-face pipeline binds exactly its transitively reachable WGSL resources", () => {
  const observed = new Map<string, number[]>();
  const buffer = { destroy() {}, size: 4_096 } as unknown as GPUBuffer;
  const device = {
    queue: { writeBuffer() {} },
    createBuffer: () => buffer,
    createShaderModule: ({ code }: GPUShaderModuleDescriptor) => ({ code }),
    createComputePipeline: ({ compute }: GPUComputePipelineDescriptor) => ({
      entryPoint: compute.entryPoint,
      shader: (compute.module as unknown as { code: string }).code,
      getBindGroupLayout() {
        return { entryPoint: this.entryPoint, shader: this.shader };
      },
    }),
    createBindGroup: ({ layout, entries }: {
      layout: { entryPoint: string; shader: string };
      entries: Iterable<GPUBindGroupEntry>;
    }) => {
      const actual = Array.from(entries, ({ binding }) => binding).sort((left, right) => left - right);
      const key = `${layout.entryPoint}\n${layout.shader}`;
      const prior = observed.get(key);
      if (prior) assert.deepEqual(actual, prior, `${layout.entryPoint} bind contract changed between calls`);
      observed.set(key, actual);
      return {};
    },
  } as unknown as GPUDevice;
  const topology: OctreePowerTopologySource = {
    plan: { rowCapacity: 2, entryCount: 1, lookupCount: 1, metricBytes: 32,
      catalogBytes: 4, allocatedBytes: 0 },
    metrics: buffer, control: buffer, catalogEntryHeaders: buffer, catalogVolumes: buffer,
    catalogFaces: buffer, catalogCoefficients: buffer, catalogLookup: buffer,
    sameOrFinerDirect: buffer, sameOrCoarserDirect: buffer,
  };
  const pass = { setPipeline() {}, setBindGroup() {}, dispatchWorkgroups() {},
    dispatchWorkgroupsIndirect() {}, end() {} };
  const encoderObject = { beginComputePass: () => pass, clearBuffer() {}, copyBufferToBuffer() {} };
  const encoder = encoderObject as unknown as GPUCommandEncoder;
  const previousUsage = Object.getOwnPropertyDescriptor(globalThis, "GPUBufferUsage");
  Object.defineProperty(globalThis, "GPUBufferUsage", { configurable: true,
    value: { STORAGE: 1, COPY_SRC: 2, COPY_DST: 4, UNIFORM: 8, INDIRECT: 16 } });
  try {
    const builder = new WebGPUOctreePowerFaces(device, 2, 16, topology, 32);
    const broker = new PassBroker(encoder);
    builder.encode(broker, buffer, {
      dimensions: [2, 1, 1], rowCount: buffer,
      rowDelta: { rows: buffer, rowCapacity: 2, controlOffsetWords: 0,
        newToOldOffsetWords: 16, oldToNewOffsetWords: 18, dirtyRowsOffsetWords: 20,
        affectedRowsOffsetWords: 22 },
      boundaryPhi: {
        mode: "fine",
        fine: { params: buffer, metadata: buffer, worklist: buffer, flags: buffer, phi: buffer },
        coarse: { directory: buffer },
        container: [2, 1, 1], fillFraction: 0.5, initialCondition: "dam-break",
      },
    });
  } finally {
    if (previousUsage) Object.defineProperty(globalThis, "GPUBufferUsage", previousUsage);
    else Reflect.deleteProperty(globalThis, "GPUBufferUsage");
  }
  assert.ok(observed.size >= 12, "the audit must exercise the complete cut-over face publication graph");
  for (const [key, actual] of observed) {
    const split = key.indexOf("\n");
    const entryPoint = key.slice(0, split);
    const shader = key.slice(split + 1);
    assert.deepEqual(actual, reachableBindings(shader, entryPoint),
      `${entryPoint} host bindings must equal transitive WGSL reachability`);
  }
});

test("power-face planner allocates compact live face and incidence stores", () => {
  const plan = planOctreePowerFaces(100, 640, 1_100);
  assert.equal(plan.faceBytes, 640 * OCTREE_POWER_FACE_RECORD_BYTES);
  assert.equal(plan.normalBytes, 640 * 16);
  assert.equal(plan.centroidBytes, 640 * 16);
  assert.equal(plan.quadratureBytes, 640 * OCTREE_POWER_FACE_QUADRATURE_BYTES);
  assert.equal(plan.incidenceBytes, 1_100 * 8);
  assert.equal(plan.workspaceBytes, 101 * 16);
  assert.equal(plan.rowDirectoryCapacity, 100);
  assert.equal(plan.rowDirectoryBytes, 100 * 16);
  assert.equal(plan.rowDiagnosticBytes, 100 * 16);
  assert.equal(plan.deltaFaceBytes, plan.faceBytes);
  assert.equal(plan.boundaryQueryBytes, 640 * 32);
  assert.equal(plan.liveFaceDispatchBytes, 64,
    "four authored dispatch records and one exact CSR block record share one portable storage arena");
  assert.equal(plan.workDispatchBytes, 48,
    "the copied indirect-only arena contains exactly four dispatch records");
  assert.equal(plan.allocatedBytes,
    2 * (plan.faceBytes + plan.normalBytes + plan.centroidBytes + plan.quadratureBytes + plan.incidenceBytes
      + plan.workspaceBytes + plan.boundaryQueryBytes + 64)
    + 2 * plan.rowDirectoryBytes + plan.rowDiagnosticBytes + 2 * plan.deltaFaceBytes + plan.faceCapacity * 8
    + plan.liveFaceDispatchBytes + plan.workDispatchBytes + 112 + 16 + 12
    + OCTREE_POWER_FACE_EMPTY_FINE_BINDINGS_BYTES);
  assert.throws(() => planOctreePowerFaces(1, 2, 5), /two incidences/);
});

test("power-face diagnostics reset inside prepare without a recurring clear command", () => {
  const compactShader = octreePowerFaceShader.replace(/\s+/g, "");
  assert.match(compactShader,
    /@compute@workgroup_size\(256\)fnpreparePowerFaces\(@builtin\(local_invocation_index\)lane:u32\)/);
  assert.match(compactShader,
    /for\(varrow=lane;row<arrayLength\(&rowDiagnostics\);row\+=256u\)\{rowDiagnostics\[row\]=RowDiagnostic\(0u,INVALID,INVALID,0u\)/);
  assert.doesNotMatch(WebGPUOctreePowerFaces.prototype.encode.toString(),
    /clearBuffer\(this\.rowDiagnostics\)/);
});

test("power-face boundary policy preserves geometric world identity and scene-open ceiling authority", () => {
  assert.equal(OCTREE_POWER_CLOSED_BOUNDARY_MASK_WITH_OPEN_TOP, 47);
  assert.equal(OCTREE_POWER_CLOSED_BOUNDARY_MASK_WITH_CLOSED_TOP, 63);
  assert.equal(octreePowerClosedBoundaryMask(false), 47, "the dam ceiling (+y) is open");
  assert.equal(octreePowerClosedBoundaryMask(true), 63, "a closed container closes all six world faces");
  assert.match(octreeProjectionSource,
    /closedBoundaryMask:\s*octreePowerClosedBoundaryMask\(this\.scene\.container\.top === "closed"\)/,
    "power-face construction must receive the authored container policy used by the Section 5 face band");

  const compactShader = octreePowerFaceShader.replace(/\s+/g, "");
  assert.match(compactShader, /structParams\{dimensionsRowCount:vec4u,capacitiesGeneration:vec4u,physical:vec4f,boundaryPolicy:vec4u,containerFill:vec4f,phiPolicy:vec4u,delta:vec4u\}/);
  assert.match(compactShader,
    /letgeometricWorld=world&declared;if\(geometricWorld!=0u\)\{letopen=select\(OPEN_BOUNDARY,0u,\(params\.boundaryPolicy\.x&geometricWorld\)!=0u\);returnBOUNDARY\|open\|\(geometricWorld<<WORLD_BOUNDARY_SHIFT\)/,
    "catalog boundary bits identify world geometry while the scene mask alone selects open versus closed");
  assert.match(compactShader, /returnBOUNDARY\|OPEN_BOUNDARY;/,
    "a missing non-world phase neighbor remains an internal free surface");
  assert.match(compactShader,
    /letworld=\(face\.flags>>WORLD_BOUNDARY_SHIFT\)&63u;if\(world!=0u\).*letboundaryDistance=dot\(geometry\.centroid-rowCenter\(face\.negativeRow\),geometry\.normal\).*face\.inverseDistance=1\.0\/boundaryDistance/s,
    "an open world plane uses exact row-centre-to-boundary distance instead of free-surface theta");
});

test("power free-surface coefficients use strict current signed cell-centre crossings", () => {
  const geometry = octreePowerFaceShader.replace(/\s+/g, "");
  const sampler = octreePowerBoundaryPhiShader.replace(/\s+/g, "");
  assert.match(geometry,
    /BoundaryPhiQuery\(vec4f\(rowCenter\(face\.negativeRow\),geometry\.inverseDistance\),vec4f\(geometry\.neighborCenter,1\.0\)\)/,
    "geometry must publish the actual liquid and absent-air power-cell centres plus the immutable geometric coefficient");
  assert.match(sampler, /letliquid=sampleAuthority\(query\.liquidCenter\.xyz\);letair=sampleAuthority\(query\.airCenter\.xyz\)/,
    "both pressure samples must come from the selected signed-distance authority");
  assert.match(sampler, /theta=\(-liquid\.x\)\/\(air\.x-liquid\.x\)/,
    "the Ghost Fluid fraction must use the two signed cell-centre values");
  assert.doesNotMatch(sampler, /abs\(liquid\.x\)|abs\(air\.x\)|clamp\([^)]*theta|0\.0[15]/,
    "authoritative power pressure must not repair signs or floor theta");
  assert.match(sampler, /!\(liquid\.x<0\.0\)\|\|air\.x<0\.0/,
    "boundary sampling must use the exact phi < 0 pressure-membership predicate, including a theta=1 zero-air endpoint");
  assert.match(sampler,
    /if\(faceParams\.phiPolicy\.z==1u\)\{powerFaces\[faceIndex\]\.flags=face\.flags\|COARSE_PENDING;return;\}\s*failBoundary/,
    "missing narrow-band support defers to the published coarse octree authority (paper Section 5) and rejects only when that authority is unbound");
  const coarseResolve = octreePowerBoundaryCoarsePhiShader.replace(/\s+/g, "");
  assert.match(coarseResolve,
    /letliquid=sampleCoarseOctreePhi\(query\.liquidCenter\.xyz\);\s*letair=sampleCoarseOctreePhi\(query\.airCenter\.xyz\)/,
    "the coarse resolve pass samples both dual-edge centres from one authority so a face never mixes fine and coarse phi");
  assert.match(coarseResolve,
    /if\(!\(liquid<0\.0\)\|\|air<0\.0\)\{failBoundary[\s\S]*lettheta=\(-liquid\)\/\(air-liquid\)/,
    "the coarse authority must match phi < 0 pressure membership and retain the exact current crossing");
  assert.doesNotMatch(coarseResolve, /clamp\([^)]*theta|0\.0[15]/,
    "the coarse authority must not hide a generation mismatch behind a theta floor");
  assert.match(coarseResolve, /if\(!coarseAvailable\(liquid\)\|\|!coarseAvailable\(air\)\)\{failBoundary/,
    "an unpublished coarse directory still rejects the generation instead of estimating phi");
  assert.match(sampler, /worklist\[1\]!=fineParams\.generation\|\|worklist\[3\]!=1u\|\|worklist\[4\]!=1u/,
    "fine pressure sampling must require the current all-or-nothing GPU publication header");
  assert.match(sampler, /letdirectoryBase=5u\+fineParams\.worklistCapacity[\s\S]*worklist\[directoryBase\+key\]/,
    "fine pressure sampling must use the immutable direct brick-to-page directory");
  assert.doesNotMatch(sampler, /letmiddle=low\+\(high-low\)\/2u/,
    "fine pressure sampling must not binary-search the published page list");
  assert.doesNotMatch(sampler, /pageHash|hashCapacity|maximumHashProbes|hashKey/,
    "the removed open-hash page table must not survive in the pressure-boundary sampler");
  assert.match(octreeProjectionSource,
    /const useCurrentFineBoundary = this\.globalFineBootstrapped && this\.powerAdvancingPressureSteps > 0;[\s\S]*const useCurrentCoarseBoundary = !useCurrentFineBoundary[\s\S]*this\.powerCoarseLevelSetBootstrapped/,
    "advancing pressure assembly must choose the current fine band or the independently evolved coarse field");
  assert.match(octreeProjectionSource,
    /mode: useCurrentFineBoundary \? "fine" as const[\s\S]*useCurrentCoarseBoundary \? "coarse" as const : "analytic" as const/,
    "only the authored t=0 operator may use analytic phi; coarse-only recurrence must bind coarse mode");
  assert.match(octreeProjectionSource,
    /this\.powerCoarseLevelSet = new WebGPUOctreeCoarseLevelSet[\s\S]*if \(this\.globalFineSourceA && this\.globalFineSourceB\)/,
    "coarse phi must be allocated before the mandatory fine-band correction resources");
  assert.doesNotMatch(octreeProjectionSource, /Coarse-only paper mode:|coarseOnlyBroker/,
    "the retired fine-off recurrence and its backing broker must stay deleted");
  assert.match(octreeProjectionSource,
    /this\.fineToPowerCoarseLevelSet\.encode\(restrictionBroker, correctedFine,[\s\S]*this\.powerCoarseLevelSetSchedule\.encode\(restrictionBroker,/,
    "coarse octree phi must evolve in the same mandatory fine-band correction transaction");
  const bindings = [...octreePowerBoundaryPhiShader.matchAll(/@binding\((\d+)\)/g)].map((match) => Number(match[1]));
  assert.equal(new Set(bindings).size, 9, "deleting the hash table also deletes its legacy binding");
});

test("Dawn sorted row directory and block scan scale across blocks with world-boundary identity", {
  skip: !process.env.WEBGPU_NODE_MODULE && "set WEBGPU_NODE_MODULE for GPU power-face scale checks",
}, async () => {
  const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href) as { create(options: string[]): GPU; globals: Record<string, unknown> };
  Object.assign(globalThis, dawn.globals);
  const gpu = dawn.create(["backend=metal"]);
  retainedNativeGPUs.push(gpu);
  const adapter = await gpu.requestAdapter(); assert.ok(adapter);
  const device = await adapter.requestDevice();
  retainedDevices.push(device);
  const validationErrors: string[] = [];
  device.addEventListener("uncapturederror", (event: unknown) => {
    validationErrors.push((event as { error: { message: string } }).error.message);
  });
  const catalog = catalogViews();
  const packedUniform = catalog.sameOrFinerDirect[0x3_ffff];
  assert.notEqual(packedUniform, 0xffff_ffff);
  const entry = packedUniform & 0xffff;
  const transform = packedUniform >>> 16;
  const header = catalog.entryHeaders.slice(entry * 2, entry * 2 + 2);
  assert.equal(header[1], 6);
  const faceData = catalog.faceData.slice(header[0] * OCTREE_POWER_CATALOG_FACE_FLOATS, (header[0] + header[1]) * OCTREE_POWER_CATALOG_FACE_FLOATS);
  const upload = (data: ArrayBufferView) => {
    const buffer = device.createBuffer({ size: Math.max(4, data.byteLength), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, mappedAtCreation: true });
    new Uint8Array(buffer.getMappedRange(), 0, data.byteLength).set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
    buffer.unmap(); return buffer;
  };
  const dimensions: [number, number, number] = [10, 10, 3];
  const rowCount = dimensions[0] * dimensions[1] * dimensions[2];
  const headerWords = new Uint32Array(rowCount * 12);
  const metricWords = new Uint32Array(rowCount * 4);
  const metricFloats = new Float32Array(metricWords.buffer);
  const sortedCells = Array.from({ length: rowCount }, (_, cell) => cell)
    .sort((left, right) => morton3(left, dimensions) - morton3(right, dimensions));
  sortedCells.forEach((cell, row) => {
    const x = cell % dimensions[0];
    const y = Math.floor(cell / dimensions[0]) % dimensions[1];
    const z = Math.floor(cell / (dimensions[0] * dimensions[1]));
    headerWords[row * 12] = cell; headerWords[row * 12 + 3] = 1;
    let boundary = 0;
    if (x === 0) boundary |= 1; if (y === 0) boundary |= 2; if (z === 0) boundary |= 4;
    if (z === dimensions[2] - 1) boundary |= 8; if (y === dimensions[1] - 1) boundary |= 16;
    if (x === dimensions[0] - 1) boundary |= 32;
    metricWords[row * 4] = 0;
    metricWords[row * 4 + 1] = (OCTREE_POWER_TOPOLOGY_VALID | transform | (boundary << 8)) >>> 0;
    metricFloats[row * 4 + 2] = catalog.entryVolumes[entry];
  });
  const headers = upload(headerWords);
  const metrics = upload(metricWords);
  const entryHeaders = upload(new Uint32Array([0, 6]));
  const facesCatalog = upload(faceData);
  const placeholder = upload(new Uint32Array([0]));
  const topology: OctreePowerTopologySource = {
    plan: { rowCapacity: rowCount, entryCount: 1, lookupCount: 1, metricBytes: metricWords.byteLength,
      catalogBytes: faceData.byteLength + 8, allocatedBytes: 0 },
    metrics, control: placeholder, catalogEntryHeaders: entryHeaders, catalogVolumes: placeholder,
    catalogFaces: facesCatalog, catalogCoefficients: placeholder,
    catalogLookup: placeholder, sameOrFinerDirect: placeholder, sameOrCoarserDirect: placeholder,
  };
  const expectedInterior = 9 * 10 * 3 + 10 * 9 * 3 + 10 * 10 * 2;
  const expectedBoundary = 2 * (10 * 3 + 10 * 3 + 10 * 10);
  const expectedFaces = expectedInterior + expectedBoundary;
  const builder = new WebGPUOctreePowerFaces(device, rowCount, expectedFaces, topology, rowCount * 6);
  assert.equal(builder.plan.deltaFaceBytes, builder.plan.faceBytes);
  const cold = createColdPowerRowPublication(device, rowCount, rowCount, 19);

  const run = async () => {
    const outputBytes = builder.plan.workspaceBytes + builder.plan.faceBytes + builder.plan.incidenceBytes
      + builder.plan.rowDirectoryBytes;
    const readback = device.createBuffer({ size: 64 + outputBytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const encoder = device.createCommandEncoder();
    const broker = new PassBroker(encoder);
    builder.encode(broker, headers, {
      dimensions, rowCount: cold.rowCount, generation: 19, rowDelta: cold.rowDelta,
    });
    broker.fence();
    let offset = 0;
    encoder.copyBufferToBuffer(builder.control, 0, readback, offset, 64); offset += 64;
    encoder.copyBufferToBuffer(builder.source.incidenceRows, 0, readback, offset, builder.plan.workspaceBytes); offset += builder.plan.workspaceBytes;
    encoder.copyBufferToBuffer(builder.faces, 0, readback, offset, builder.plan.faceBytes); offset += builder.plan.faceBytes;
    encoder.copyBufferToBuffer(builder.incidence, 0, readback, offset, builder.plan.incidenceBytes); offset += builder.plan.incidenceBytes;
    encoder.copyBufferToBuffer(builder.rowDirectory, 0, readback, offset, builder.plan.rowDirectoryBytes);
    device.queue.submit([encoder.finish()]); await device.queue.onSubmittedWorkDone();
    await readback.mapAsync(GPUMapMode.READ);
    const result = readback.getMappedRange().slice(0); readback.unmap(); readback.destroy(); return result;
  };
  const first = await run();
  const control = unpackOctreePowerFaceControl(new Uint32Array(first, 0, 16));
  assert.equal(control.flags, 0); assert.equal(control.valid, OCTREE_POWER_FACE_VALID);
  assert.equal(control.rowCount, rowCount); assert.equal(control.faceCount, expectedFaces);
  assert.equal(control.incidenceCount, rowCount * 6); assert.equal(control.boundaryCount, expectedBoundary);
  assert.equal(control.lookupMissCount, expectedBoundary); assert.equal(control.worldBoundaryCount, expectedBoundary);
  assert.equal(control.maximumLookupSteps, 9,
    "the 300-row canonical directory publishes its exact nine-step binary-search bound");
  const workspace = new Uint32Array(first, 64, builder.plan.workspaceBytes / 4);
  assert.ok(workspace[255 * 4 + 2] < workspace[256 * 4 + 2]);
  assert.equal(workspace[rowCount * 4 + 2], expectedFaces);
  assert.equal(workspace[rowCount * 4 + 3], rowCount * 6);
  device.queue.writeBuffer(cold.rowCount, 0, new Uint32Array([0]));
  const empty = unpackOctreePowerFaceControl(new Uint32Array(await run(), 0, 16));
  assert.equal(empty.valid, 0); assert.equal(empty.faceCount, 0); assert.equal(empty.incidenceCount, 0);
  assert.equal(empty.flags & OCTREE_POWER_FACE_ERROR.invalidHeader, OCTREE_POWER_FACE_ERROR.invalidHeader,
    "an empty generation must fail closed instead of publishing valid=true");
  device.queue.writeBuffer(cold.rowCount, 0, new Uint32Array([rowCount]));
  const faceWordOffset = 64 + builder.plan.workspaceBytes;
  const faceWords = new Uint32Array(first, faceWordOffset, builder.plan.faceBytes / 4);
  let observedBoundary = 0;
  for (let face = 0; face < expectedFaces; face += 1) {
    const word = face * 8;
    if (faceWords[word + 1] === 0xffff_ffff) {
      observedBoundary += 1;
      assert.equal(faceWords[word + 3] & OCTREE_POWER_FACE_BOUNDARY, OCTREE_POWER_FACE_BOUNDARY);
      assert.equal(faceWords[word + 3] & OCTREE_POWER_FACE_OPEN_BOUNDARY, 0);
      assert.notEqual((faceWords[word + 3] >>> OCTREE_POWER_FACE_WORLD_BOUNDARY_SHIFT) & 63, 0);
    }
  }
  assert.equal(observedBoundary, expectedBoundary);
  const directoryOffset = 64 + builder.plan.workspaceBytes + builder.plan.faceBytes + builder.plan.incidenceBytes;
  const directory = new Uint32Array(first, directoryOffset, builder.plan.rowDirectoryBytes / 4);
  sortedCells.forEach((cell, row) => {
    assert.deepEqual(Array.from(directory.subarray(row * 4, row * 4 + 4)),
      [cell + 1, 1, row, morton3(cell, dimensions)],
      `row-directory record ${row} must be the exact deterministic (cell,size,row,Morton) publication`);
  });
  const second = await run();
  assert.deepEqual(new Uint8Array(second, 64), new Uint8Array(first, 64),
    "multi-block public face/CSR/directory output must be stable");

  const duplicateHeaders = new Uint32Array(33 * 12);
  for (let row = 0; row < 33; row += 1) {
    duplicateHeaders[row * 12] = Math.max(0, row - 1);
    duplicateHeaders[row * 12 + 3] = 1;
  }
  const duplicateHeaderBuffer = upload(duplicateHeaders);
  const duplicateBuilder = new WebGPUOctreePowerFaces(device, 33, 198, topology, 198);
  assert.equal(duplicateBuilder.plan.rowDirectoryCapacity, 33);
  const duplicateReadback = device.createBuffer({ size: 64, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const duplicateEncoder = device.createCommandEncoder();
  const duplicateBroker = new PassBroker(duplicateEncoder);
  const duplicateCold = createColdPowerRowPublication(device, 33, 33, 0);
  duplicateBuilder.encode(duplicateBroker, duplicateHeaderBuffer, {
    dimensions: [8_192, 1, 1], rowCount: duplicateCold.rowCount, rowDelta: duplicateCold.rowDelta,
  });
  duplicateBroker.fence();
  duplicateEncoder.copyBufferToBuffer(duplicateBuilder.control, 0, duplicateReadback, 0, 64);
  device.queue.submit([duplicateEncoder.finish()]); await device.queue.onSubmittedWorkDone();
  await duplicateReadback.mapAsync(GPUMapMode.READ);
  const duplicateControl = unpackOctreePowerFaceControl(new Uint32Array(duplicateReadback.getMappedRange().slice(0)));
  duplicateReadback.unmap();
  assert.equal(duplicateControl.faceCount, 0); assert.equal(duplicateControl.incidenceCount, 0); assert.equal(duplicateControl.valid, 0);
  assert.notEqual(duplicateControl.flags & OCTREE_POWER_FACE_ERROR.rowDirectory, 0,
    "duplicate directory keys must reject the generation rather than select an arbitrary owner");
  assert.deepEqual(validationErrors, []);
  duplicateCold.destroy(); duplicateBuilder.destroy(); duplicateHeaderBuffer.destroy(); duplicateReadback.destroy();
  cold.destroy(); builder.destroy(); headers.destroy(); metrics.destroy(); entryHeaders.destroy();
  facesCatalog.destroy(); placeholder.destroy();
});

test("power-face WGSL builds, merges, and publishes CSR in parallel without atomic public append", () => {
  const configuredEntryPoints = [...powerFaceSource.matchAll(/\bpipeline\([^,]+,\s*"([^"]+)"\)/g)]
    .map((match) => match[1]);
  for (const entryPoint of configuredEntryPoints) {
    assert.match(octreePowerFaceShader, new RegExp(`\\bfn\\s+${entryPoint}\\b`),
      `configured power-face pipeline entry point ${entryPoint} must exist`);
    assert.ok(reachableStorageBindings(octreePowerFaceShader, entryPoint).length <= 10,
      `${entryPoint} must fit the portable WebGPU limit of ten storage buffers`);
  }
  assert.match(octreePowerFaceShader, /@compute @workgroup_size\(256\) fn buildAffectedPowerFaceRows/);
  assert.match(octreePowerFaceShader, /@compute @workgroup_size\(256\) fn compactPowerFaceIdentityDelta/);
  assert.match(octreePowerFaceShader, /@compute @workgroup_size\(256\) fn mergePowerFaceIdentityDelta/);
  assert.match(octreePowerFaceShader, /@compute @workgroup_size\(64\) fn preparePowerFaceCandidateCSR/);
  assert.match(octreePowerFaceShader, /@compute @workgroup_size\(256\) fn scanPowerFaceCandidateCSR/);
  assert.match(octreePowerFaceShader, /@compute @workgroup_size\(256\) fn prefixPowerFaceCandidateCSR/);
  assert.match(octreePowerFaceShader, /@compute @workgroup_size\(256\) fn offsetPowerFaceCandidateCSR/);
  assert.match(octreePowerFaceShader, /@compute @workgroup_size\(64\) fn publishAffectedPowerFaceCandidateCSR/);
  assert.match(octreePowerFaceShader, /@compute @workgroup_size\(64\) fn publishCarriedPowerFaceCandidateCSR/);
  assert.match(octreePowerFaceShader,
    /let mapped=mapOldFace\(oldIncidence\.face\)[\s\S]*powerFaceLowerBound\(mapped\.face,control\.faceCount\)/,
    "clean rows carry incidence through exact old/new owning-pair identity");
  assert.match(octreePowerFaceShader,
    /fn uniformParallelEnable[\s\S]*workgroupUniformLoad\(&parallelEnabled\)/,
    "storage-backed transaction state must be broadcast uniformly before any barrier-containing stage");
  for (const entryPoint of [
    "buildAffectedPowerFaceRows", "compactPowerFaceIdentityDelta", "mergePowerFaceIdentityDelta",
  ]) {
    assert.match(octreePowerFaceShader,
      new RegExp(`fn ${entryPoint}\\([\\s\\S]*?\\{\\s*if\\(uniformParallelEnable\\(lane,`),
      `${entryPoint} must make its enable decision workgroup-uniform`);
  }
  for (const entryPoint of ["scanPowerFaceCandidateCSR", "prefixPowerFaceCandidateCSR"]) {
    assert.match(octreePowerFaceShader,
      new RegExp(`fn ${entryPoint}\\([\\s\\S]*?\\{\\s*let enabled=uniformParallelEnable\\(lane,control\\.flags==0u\\);[\\s\\S]*?parallelInclusiveScanPair`),
      `${entryPoint} must mask storage-backed failure state without returning before its parallel scan barriers`);
  }
  assert.doesNotMatch(octreePowerFaceShader,
    /fn (?:buildAffectedPowerFaces|mergePowerFaceDelta|buildPowerFaceCandidateCSR)\b/,
    "the three deleted singleton transactions must not retain shader backing code");
  assert.match(powerFaceSource.replace(/\s+/g, ""),
    /run\("Buildaffectedpower-facerowsinparallel",this\.buildAffectedRowsPipeline,1,\[params,headers,metrics,entries,catalog,rows,control,rowDelta,diagnostics,deltaFaceScratch\]\)/,
    "one cooperative workgroup must consume the exact affected-row delta directly");
  assert.doesNotMatch(powerFaceSource,
    /countAffectedPipeline|scanAffected|addAffected|emitAffected|boundaryQueryScan|boundaryQueryCompact/,
    "the deleted row-scan and cut-face-compaction schedules must not retain host backing code");
  assert.doesNotMatch(octreePowerFaceShader,
    /fn (?:countPowerFaces|scanPowerFaceRows|scanPowerFaceBlocks|emitPowerFaces|carryPowerFaces|rebuildPowerIncidence|sortPowerIncidenceRows|validatePowerFaceDelta|rowDirectoryCapacity|scanBlockCount)/,
    "the legacy full-capacity count/scan/emit/carry/rebuild/sort/validation graph must stay deleted");
  assert.match(octreePowerFaceShader, /while\(insertion>0u&&local\[insertion-1u\]\.face>faceIndex\)/,
    "row-parallel CSR construction must sort each bounded catalog incidence by exact public face ID");
  assert.match(octreePowerFaceShader, /deltaFaces\[output\]=result\.face/);
  assert.match(octreePowerFaceShader, /parallelInclusiveScan\(lane,emitted\)/);
  assert.match(octreePowerFaceShader, /oldFaceLowerBound\(face,survivorCount\)/);
  assert.doesNotMatch(octreePowerFaceShader, /atomicAdd\(&control\.faceCount/);
  assert.match(octreePowerFaceShader,
    /PowerFaceRecord\(row,neighbor,\s*\(slot&0xffffu\)\|\(\(metric\.transformAndFlags&63u\)<<16u\),\s*flags,0\.0,geometry\.area,geometry\.inverseDistance,1\.0\)/);
  assert.match(octreePowerFaceShader, /fn publishAddedPowerFaceGeometry/);
  assert.match(octreePowerFaceShader, /fn carryPowerFacePayload/);
  assert.doesNotMatch(octreePowerFaceShader, /scanPowerBoundaryPhiQueryFlags|compactPowerBoundaryPhiQueries/);
  assert.doesNotMatch(octreePowerFaceShader, /\batomic(?:Add|And|CompareExchangeWeak|Exchange|Load|Max|Min|Or|Store|Sub|Xor)\b|atomic<u32>/,
    "fixed row/face records and deterministic reductions must eliminate every power-face synchronization atomic");
  assert.match(octreePowerFaceShader.replace(/\s+/g, ""),
    /fninvalidatePowerFace\(faceIndex:u32,flag:u32,detail:u32\)[\s\S]*face\.openFraction=bitcast<f32>\(flag\)[\s\S]*fnfinalizeAndPublishPowerFaces\(\)[\s\S]*letfailureFlag=bitcast<u32>\(face\.openFraction\);control\.flags\|=failureFlag/,
    "face-parallel failures must publish one deterministic error flag/detail record for finalization");
  assert.match(octreePowerFaceShader, /let faceGroups=\(faceTotal\+63u\)\/64u;liveFaceDispatch\[0\]=faceGroups;liveFaceDispatch\[1\]=1u/,
    "compact face consumers must launch only the published live prefix");
  assert.match(octreePowerFaceShader,
    /let\s+reconstructionFailure=reconstructionError\(geometry\);\s*if\(reconstructionFailure!=0u\)\{failGeometryTopology\(row,slot,4096u\|reconstructionFailure,metric\.topologyCode,metric\.transformAndFlags\);return AffectedFaceResult/,
    "an invalid catalog reconstruction must retain its row, slot, topology code, and transform for the viewport failure marker");
  assert.match(octreePowerFaceShader, /fn reconstructionError\(face:ReconstructedPowerFace\)->u32\{/,
    "catalog validation must expose a scalar reason code instead of relying on a compound backend predicate");
  assert.doesNotMatch(octreePowerFaceShader, /airPhi=abs|liquidPhi\+dot|0\.05/,
    "face geometry must not synthesize or floor a free-surface coefficient");
});

test("dependent live-face kernels share broker passes only across valid synchronization regions", () => {
  assert.match(powerFaceSource,
    /const runIndirectFaces = \(label: string,[\s\S]*?broker\.compute\(\{ label \}\)[\s\S]*?pass\.dispatchWorkgroupsIndirect\(this\.workDispatch, offset\);/,
    "live-prefix kernels must acquire their shared compute pass from the broker");
  const liveArenaAllocation = powerFaceSource.match(
    /this\.liveFaceDispatch = device\.createBuffer\(\{([\s\S]*?)\}\);/,
  )?.[1] ?? "";
  const workDispatchAllocation = powerFaceSource.match(
    /this\.workDispatch = device\.createBuffer\(\{([\s\S]*?)\}\);/,
  )?.[1] ?? "";
  assert.match(liveArenaAllocation,
    /label: "Octree power live-face dispatch arena"[\s\S]*?usage: GPUBufferUsage\.STORAGE \| GPUBufferUsage\.COPY_SRC/,
    "GPU-authored work records must live in a storage-only publication arena");
  assert.match(workDispatchAllocation,
    /label: "Octree power indirect-only work dispatch"[\s\S]*?usage: GPUBufferUsage\.INDIRECT \| GPUBufferUsage\.COPY_DST/,
    "all indirect launches must consume the dedicated copied dispatch arena");
  assert.doesNotMatch(liveArenaAllocation, /GPUBufferUsage\.INDIRECT/,
    "the storage-authored dispatch arena must never be used indirectly");
  assert.doesNotMatch(workDispatchAllocation, /GPUBufferUsage\.STORAGE/,
    "the indirect dispatch arena must never be storage-bound");
  for (const [sourceOffset, destinationOffset, size] of [
    ["OCTREE_POWER_FACE_LIVE_ROW_DISPATCH_OFFSET_BYTES", "OCTREE_POWER_FACE_LIVE_ROW_DISPATCH_OFFSET_BYTES", 24],
    ["0", "0", 12],
    ["12", "12", 12],
  ] as const) {
    assert.match(powerFaceSource,
      new RegExp(`broker\\.updateIndirectBuffer\\(\\s*this\\.liveFaceDispatch,\\s*${sourceOffset},\\s*this\\.workDispatch,\\s*${destinationOffset},\\s*${size},?\\s*\\)`),
      `dispatch record ${destinationOffset} must cross its semantic publication boundary by an exact copy`);
  }
  assert.doesNotMatch(powerFaceSource, /beginComputePass/,
    "the face builder must not retain a raw compute-pass escape hatch");
  assert.doesNotMatch(powerFaceSource, /power face affected compact publication/);
  assert.match(powerFaceSource, /broker\.fence\("power face exact identity publication"\)/);
  assert.match(powerFaceSource, /broker\.fence\("power face geometry publication"\)/);
  assert.doesNotMatch(powerFaceSource, /power boundary query publication/);
  assert.match(powerFaceSource, /broker\.fence\("power face candidate publication"\)/);
});

test("face publication uses exact row-delta carry and deletes legacy whole-topology reuse", () => {
  const shader = octreePowerFaceShader.replace(/\s+/g, "");
  const sampler = octreePowerBoundaryPhiShader.replace(/\s+/g, "");
  assert.doesNotMatch(shader, /reuseFaceTopology|rowTopologyReuse|if\(reuse\)/,
    "legacy whole-topology reuse and its switch must be deleted");
  assert.match(shader, /fnmapOldFace\(source:u32\)->AffectedFaceResult\{[\s\S]*oldNegative=oldToNew\(face\.negativeRow\)[\s\S]*fnremappedOldFace[\s\S]*rowAffected\(face\.negativeRow\)/,
    "unchanged faces must be carried only through exact endpoint remapping and affected-row authority");
  assert.match(shader,
    /fnrowAffectedFlag\(row:u32\)->bool\{[\s\S]*ROW_DELTA_AFFECTED[\s\S]*fnrowAffected\(row:u32\)->bool\{[\s\S]*returnrowAffectedFlag\(row\)/,
    "the exact new-to-old row publication must provide O(1) affected membership to every one-ring consumer");
  assert.doesNotMatch(shader, /fnrowAffected\(row:u32\)->bool\{[\s\S]{0,200}while\(low<high\)/,
    "affected-row membership must not re-search the immutable compact list for every face and incidence");
  assert.match(shader, /fnrowIncidenceIdentity[\s\S]*RowIncidenceResult\([\s\S]*,1,1u\)[\s\S]*,-1,1u\)/,
    "candidate incidence must be reconstructed from reciprocal catalog identity and sorted by public face ID");
  assert.match(shader, /fnmergePowerFaceIdentityDelta[\s\S]*outputCount!=carried\+added\|\|outputCount\+retired!=previousCount\+added/,
    "candidate publication must satisfy both exact face-delta count identities");
  assert.doesNotMatch(shader, /fn(?:carryPowerFaces|rebuildPowerIncidence|validatePowerFaceDelta)/,
    "the old duplicate carry, incidence rebuild, and validation backing code must stay deleted");
  assert.match(shader,
    /boundaryPhiQueries\[faceIndex\]=BoundaryPhiQuery\(vec4f\(rowCenter\(face\.negativeRow\),geometry\.inverseDistance\),vec4f\(geometry\.neighborCenter,1\.0\)\)/,
    "direct cut-face queries must retain the immutable geometric coefficient at the exact public face ID");
  assert.match(sampler, /face\.inverseDistance=query\.liquidCenter\.w\/theta/,
    "dynamic generations must recompute from the immutable base instead of compounding the previous theta");
  assert.match(powerFaceSource, /Sample deterministic power boundary phi[\s\S]*\]\);/,
    "dynamic phi must launch against the exact live-face prefix and skip non-query records in-kernel");
  assert.match(powerFaceSource, /Commit power face candidate geometry[\s\S]*\], 12\)[\s\S]*Commit power face candidate incidence[\s\S]*\], 12\)/,
    "candidate publication must commit only after exact validation");
});

test("power-face identity oracle retires endpoint changes and carries remapped identities exactly", () => {
  const boundary = OCTREE_POWER_INVALID_ROW;
  const previous = [
    { negativeRow: 0, positiveRow: 1, geometryCode: 0 },
    { negativeRow: 0, positiveRow: boundary, geometryCode: 1 },
    { negativeRow: 1, positiveRow: boundary, geometryCode: 0 },
  ];
  const current = [
    { negativeRow: 0, positiveRow: boundary, geometryCode: 1 },
    { negativeRow: 1, positiveRow: boundary, geometryCode: 0 },
    { negativeRow: 1, positiveRow: boundary, geometryCode: 2 },
  ];
  assert.deepEqual(mergeOctreePowerFaceIdentities(previous, current, [-1, 1]), {
    carried: 1, added: 2, retired: 2,
  });
  assert.throws(() => mergeOctreePowerFaceIdentities(previous, [...current].reverse(), [-1, 1]), /strictly sorted/);
});

test("fine power-boundary phi rejects a one-generation phase crossing", () => {
  assert.match(octreePowerBoundaryPhiShader,
    /!\(liquid\.x<0\.0\)\|\|air\.x<0\.0[\s\S]*failBoundary/,
    "a row crossed by the current interface must fail the generation instead of solving a stale system");
  assert.match(octreePowerBoundaryPhiShader,
    /let theta=\(-liquid\.x\)\/\(air\.x-liquid\.x\);[\s\S]*!\(theta>0\.0\)\|\|theta>1\.0/,
    "an absent neighbor exactly on phi=0 must remain the valid theta=1 endpoint");
  assert.match(octreePowerBoundaryCoarsePhiShader,
    /if\(!\(liquid<0\.0\)\|\|air<0\.0\)\{failBoundary[\s\S]*let theta=\(-liquid\)\/\(air-liquid\);/,
    "fine and coarse cell-centre authorities must use the same complementary phase predicate");
  assert.doesNotMatch(octreePowerBoundaryPhiShader, /abs\(liquid\.x\)|abs\(air\.x\)|clamp\([^)]*theta/,
    "phase validation must neither repair signs nor floor the interface fraction");
});

test("Dawn emits stable unique power faces and reciprocal compact incidence", {
  skip: !process.env.WEBGPU_NODE_MODULE && "set WEBGPU_NODE_MODULE for GPU power-face checks",
}, async () => {
  const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href) as { create(options: string[]): GPU; globals: Record<string, unknown> };
  Object.assign(globalThis, dawn.globals);
  const gpu = dawn.create(["backend=metal"]);
  retainedNativeGPUs.push(gpu);
  const adapter = await gpu.requestAdapter(); assert.ok(adapter);
  const device = await adapter.requestDevice();
  retainedDevices.push(device);
  const validationErrors: string[] = [];
  device.addEventListener("uncapturederror", (event: unknown) => {
    validationErrors.push((event as { error: { message: string } }).error.message);
  });
  const compilation = await device.createShaderModule({ code: octreePowerFaceShader }).getCompilationInfo();
  assert.deepEqual(compilation.messages.filter((message) => message.type === "error"), []);

  const catalog = catalogViews();
  // 18 same-size face/edge neighbors is the uniform catalog descriptor.
  const lookupFor = (descriptor: number) => {
    if (descriptor < catalog.sameOrFinerDirect.length) {
      const packed = catalog.sameOrFinerDirect[descriptor];
      if (packed !== 0xffff_ffff) return { entry: packed & 0xffff, transform: packed >>> 16 };
    }
    for (let offset = 0; offset < catalog.lookup.length; offset += 3) {
      if (catalog.lookup[offset] === descriptor) return { entry: catalog.lookup[offset + 1], transform: catalog.lookup[offset + 2] };
    }
    throw new Error(`Missing catalog descriptor ${descriptor}`);
  };
  const uniform = lookupFor(0x3_ffff);
  const uniformHeader = catalog.entryHeaders.slice(uniform.entry * 2, uniform.entry * 2 + 2);
  const uniformFaces = catalog.faceData.slice(uniformHeader[0] * OCTREE_POWER_CATALOG_FACE_FLOATS, (uniformHeader[0] + uniformHeader[1]) * OCTREE_POWER_CATALOG_FACE_FLOATS);
  const compactFaces = new Float32Array(uniformFaces);
  const compactHeaders = new Uint32Array([0, uniformHeader[1]]);
  const metricWords = new Uint32Array(8);
  const metricFloats = new Float32Array(metricWords.buffer);
  [31, 62].forEach((boundary, row) => {
    metricWords[row * 4] = 0;
    metricWords[row * 4 + 1] = (OCTREE_POWER_TOPOLOGY_VALID | uniform.transform | (boundary << 8)) >>> 0;
    metricFloats[row * 4 + 2] = catalog.entryVolumes[uniform.entry];
  });
  const upload = (data: ArrayBufferView) => {
    const buffer = device.createBuffer({ size: Math.max(4, data.byteLength), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, mappedAtCreation: true });
    new Uint8Array(buffer.getMappedRange(), 0, data.byteLength).set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
    buffer.unmap(); return buffer;
  };
  const metrics = upload(metricWords);
  const entryHeaders = upload(compactHeaders);
  const catalogFaces = upload(compactFaces);
  const placeholder = upload(new Uint32Array([0]));
  const topology: OctreePowerTopologySource = {
    plan: { rowCapacity: 2, entryCount: 1, lookupCount: 1, metricBytes: 32, catalogBytes: compactHeaders.byteLength + compactFaces.byteLength, allocatedBytes: 0 },
    metrics, control: placeholder, catalogEntryHeaders: entryHeaders, catalogVolumes: placeholder,
    catalogFaces, catalogCoefficients: placeholder,
    catalogLookup: placeholder, sameOrFinerDirect: placeholder, sameOrCoarserDirect: placeholder,
  };
  const builder = new WebGPUOctreePowerFaces(device, 2, 11, topology, 12);
  const dimensions: [number, number, number] = [2, 1, 1];
  const headerWords = new Uint32Array(2 * 12);
  [0, 1].forEach((cell, row) => { headerWords[row * 12] = cell; headerWords[row * 12 + 3] = 1; });
  const headers = upload(headerWords);
  const cold = createColdPowerRowPublication(device, 2, 2, 7);

  const readRun = async () => {
    const byteCount = 64 + builder.plan.workspaceBytes + builder.plan.faceBytes
      + builder.plan.incidenceBytes + builder.plan.normalBytes + builder.plan.centroidBytes + builder.plan.quadratureBytes;
    const readback = device.createBuffer({ size: byteCount, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const encoder = device.createCommandEncoder();
    const broker = new PassBroker(encoder);
    builder.encode(broker, headers, {
      dimensions, rowCount: cold.rowCount, rowDelta: cold.rowDelta,
      physicalCellSize: 0.25, generation: 7,
    });
    broker.fence();
    let offset = 0;
    encoder.copyBufferToBuffer(builder.control, 0, readback, offset, 64); offset += 64;
    encoder.copyBufferToBuffer(builder.source.incidenceRows, 0, readback, offset, builder.plan.workspaceBytes); offset += builder.plan.workspaceBytes;
    encoder.copyBufferToBuffer(builder.faces, 0, readback, offset, builder.plan.faceBytes); offset += builder.plan.faceBytes;
    encoder.copyBufferToBuffer(builder.incidence, 0, readback, offset, builder.plan.incidenceBytes); offset += builder.plan.incidenceBytes;
    encoder.copyBufferToBuffer(builder.faceNormals, 0, readback, offset, builder.plan.normalBytes); offset += builder.plan.normalBytes;
    encoder.copyBufferToBuffer(builder.faceCentroids, 0, readback, offset, builder.plan.centroidBytes); offset += builder.plan.centroidBytes;
    encoder.copyBufferToBuffer(builder.faceQuadrature, 0, readback, offset, builder.plan.quadratureBytes);
    device.queue.submit([encoder.finish()]); await device.queue.onSubmittedWorkDone();
    await readback.mapAsync(GPUMapMode.READ);
    const bytes = readback.getMappedRange().slice(0); readback.unmap(); readback.destroy();
    return bytes;
  };

  const first = await readRun();
  const controlWords = new Uint32Array(first, 0, 16);
  const control = unpackOctreePowerFaceControl(controlWords);
  assert.deepEqual(control, {
    rowCount: 2, faceCount: 11, incidenceCount: 12, flags: 0,
    firstInvalid: 0xffff_ffff, invalidCount: 0, boundaryCount: 10, generation: 7, valid: OCTREE_POWER_FACE_VALID,
    lookupMissCount: 10, maximumLookupSteps: 2, worldBoundaryCount: 10,
    firstInvalidSlot: 0xffff_ffff, firstInvalidNeighbor: 0xffff_ffff, firstInvalidDetail: 0,
    firstInvalidRow: 0xffff_ffff,
  });
  assert.equal(control.maximumLookupSteps, 2,
    "the two-row canonical directory publishes its exact two-step binary-search bound");
  assert.equal(controlWords[8], OCTREE_POWER_FACE_VALID);
  const workspaceOffset = 64;
  const workspace = new Uint32Array(first, workspaceOffset, builder.plan.workspaceBytes / 4);
  assert.deepEqual([workspace[3], workspace[7], workspace[11]], [0, 6, 12]);
  assert.deepEqual([workspace[0], workspace[4]], [6, 5]);
  assert.deepEqual([workspace[1], workspace[5]], [6, 6]);

  const faceOffset = workspaceOffset + builder.plan.workspaceBytes;
  const faceWords = new Uint32Array(first, faceOffset, builder.plan.faceBytes / 4);
  const faceFloats = new Float32Array(first, faceOffset, builder.plan.faceBytes / 4);
  const keys = new Set<string>();
  let interiorFace = -1;
  for (let face = 0; face < control.faceCount; face += 1) {
    const word = face * 8;
    const negative = faceWords[word];
    const positive = faceWords[word + 1];
    const geometryCode = faceWords[word + 2];
    const flags = faceWords[word + 3];
    assert.ok(negative < 2);
    assert.equal((flags & OCTREE_POWER_FACE_VALID) >>> 0, OCTREE_POWER_FACE_VALID);
    assert.ok(Number.isFinite(faceFloats[word + 5]) && faceFloats[word + 5] > 0);
    assert.ok(Number.isFinite(faceFloats[word + 6]) && faceFloats[word + 6] > 0);
    assert.equal(faceFloats[word + 7], 1);
    const key = `${negative}:${positive}:${geometryCode}`;
    assert.equal(keys.has(key), false); keys.add(key);
    if (positive !== 0xffff_ffff) {
      assert.equal(interiorFace, -1);
      assert.deepEqual([negative, positive], [0, 1]);
      assert.equal(flags & OCTREE_POWER_FACE_BOUNDARY, 0);
      interiorFace = face;
    } else {
      assert.equal(flags & OCTREE_POWER_FACE_BOUNDARY, OCTREE_POWER_FACE_BOUNDARY);
    }
  }
  assert.notEqual(interiorFace, -1);

  const incidenceOffset = faceOffset + builder.plan.faceBytes;
  const incidenceWords = new Uint32Array(first, incidenceOffset, builder.plan.incidenceBytes / 4);
  const normalOffset = incidenceOffset + builder.plan.incidenceBytes;
  const normalFloats = new Float32Array(first, normalOffset, builder.plan.normalBytes / 4);
  const centroidOffset = normalOffset + builder.plan.normalBytes;
  const centroidFloats = new Float32Array(first, centroidOffset, builder.plan.centroidBytes / 4);
  const quadratureOffset = centroidOffset + builder.plan.centroidBytes;
  const quadratureWords = new Uint32Array(first, quadratureOffset, builder.plan.quadratureBytes / 4);
  const quadratureFloats = new Float32Array(first, quadratureOffset, builder.plan.quadratureBytes / 4);
  for (let face = 0; face < control.faceCount; face += 1) {
    const at = face * 4; const length = Math.hypot(normalFloats[at], normalFloats[at + 1], normalFloats[at + 2]);
    assert.ok(Math.abs(length - 1) < 2e-4); assert.equal(normalFloats[at + 3], 0);
    assert.ok(Number.isFinite(centroidFloats[at]) && Number.isFinite(centroidFloats[at + 1])
      && Number.isFinite(centroidFloats[at + 2]));
    assert.equal(centroidFloats[at + 3], 0);
    const quadratureAt = face * (OCTREE_POWER_FACE_QUADRATURE_BYTES / 4);
    assert.deepEqual(Array.from(quadratureFloats.slice(quadratureAt, quadratureAt + 3)),
      Array.from(centroidFloats.slice(at, at + 3)), "polygon quadrature must retain the exact public face centroid");
    assert.ok(Math.abs(quadratureFloats[quadratureAt + 3] - faceFloats[face * 8 + 5])
      <= Math.max(2e-5, faceFloats[face * 8 + 5] * 5e-4));
    assert.ok(new Set(Array.from(quadratureWords.slice(quadratureAt + 4, quadratureAt + 20))).size > 1,
      "actual power-polygon strata must not collapse to the face centroid");
  }
  const incident = Array.from({ length: 2 }, () => [] as { face: number; sign: number }[]);
  for (let row = 0; row < 2; row += 1) {
    const begin = workspace[row * 4 + 3];
    const end = workspace[(row + 1) * 4 + 3];
    for (let index = begin; index < end; index += 1) {
      incident[row].push({ face: incidenceWords[index * 2], sign: new Int32Array(first, incidenceOffset + index * 8 + 4, 1)[0] });
    }
  }
  assert.deepEqual(incident[0].filter((item) => item.face === interiorFace), [{ face: interiorFace, sign: 1 }]);
  assert.deepEqual(incident[1].filter((item) => item.face === interiorFace), [{ face: interiorFace, sign: -1 }]);
  for (const row of incident) for (const item of row) assert.ok(item.face < control.faceCount);

  const second = await readRun();
  assert.deepEqual(new Uint8Array(second), new Uint8Array(first), "repeated GPU rebuild must be byte deterministic");

  // Exact row-identity reuse keeps every topology/geometry byte immutable,
  // while the current signed-distance authority refreshes only the compact
  // internal cut-face coefficient. The candidate face buffer must also keep
  // the prior committed publication intact when that refresh fails.
  const reuseMetricWords = new Uint32Array(4);
  const reuseMetricFloats = new Float32Array(reuseMetricWords.buffer);
  reuseMetricWords[0] = 0;
  reuseMetricWords[1] = (OCTREE_POWER_TOPOLOGY_VALID | uniform.transform | (31 << 8)) >>> 0;
  reuseMetricFloats[2] = catalog.entryVolumes[uniform.entry];
  const reuseMetrics = upload(reuseMetricWords);
  const reuseTopology: OctreePowerTopologySource = {
    plan: { rowCapacity: 1, entryCount: 1, lookupCount: 1, metricBytes: 16,
      catalogBytes: compactHeaders.byteLength + compactFaces.byteLength, allocatedBytes: 0 },
    metrics: reuseMetrics, control: placeholder, catalogEntryHeaders: entryHeaders, catalogVolumes: placeholder,
    catalogFaces, catalogCoefficients: placeholder,
    catalogLookup: placeholder, sameOrFinerDirect: placeholder, sameOrCoarserDirect: placeholder,
  };
  const reuseBuilder = new WebGPUOctreePowerFaces(device, 1, 6, reuseTopology, 6);
  const reuseHeaderWords = new Uint32Array(12); reuseHeaderWords[3] = 1;
  const reuseHeaders = upload(reuseHeaderWords);
  const reuseByteCount = 64 + reuseBuilder.plan.faceBytes + reuseBuilder.plan.incidenceBytes
    + reuseBuilder.plan.normalBytes + reuseBuilder.plan.centroidBytes + reuseBuilder.plan.quadratureBytes
    + reuseBuilder.plan.boundaryQueryBytes;
  const runReuse = async (generation: number, fillFraction: number, reuse: boolean) => {
    const readback = device.createBuffer({ size: reuseByteCount, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const encoder = device.createCommandEncoder();
    const broker = new PassBroker(encoder);
    const publication = reuse
      ? createIdentityPowerRowPublication(device, 1, 1, generation)
      : createColdPowerRowPublication(device, 1, 1, generation);
    reuseBuilder.encode(broker, reuseHeaders, {
      dimensions: [2, 1, 1], rowCount: publication.rowCount, rowDelta: publication.rowDelta,
      physicalCellSize: 0.25, generation,
      boundaryPhi: { mode: "analytic", container: [0.5, 0.25, 0.25], fillFraction, initialCondition: "dam-break" },
    });
    broker.fence();
    let offset = 0;
    encoder.copyBufferToBuffer(reuseBuilder.control, 0, readback, offset, 64); offset += 64;
    encoder.copyBufferToBuffer(reuseBuilder.faces, 0, readback, offset, reuseBuilder.plan.faceBytes); offset += reuseBuilder.plan.faceBytes;
    encoder.copyBufferToBuffer(reuseBuilder.incidence, 0, readback, offset, reuseBuilder.plan.incidenceBytes); offset += reuseBuilder.plan.incidenceBytes;
    encoder.copyBufferToBuffer(reuseBuilder.faceNormals, 0, readback, offset, reuseBuilder.plan.normalBytes); offset += reuseBuilder.plan.normalBytes;
    encoder.copyBufferToBuffer(reuseBuilder.faceCentroids, 0, readback, offset, reuseBuilder.plan.centroidBytes); offset += reuseBuilder.plan.centroidBytes;
    encoder.copyBufferToBuffer(reuseBuilder.faceQuadrature, 0, readback, offset, reuseBuilder.plan.quadratureBytes); offset += reuseBuilder.plan.quadratureBytes;
    encoder.copyBufferToBuffer(reuseBuilder.source.boundaryPhiQueries, 0, readback, offset, reuseBuilder.plan.boundaryQueryBytes);
    device.queue.submit([encoder.finish()]); await device.queue.onSubmittedWorkDone();
    await readback.mapAsync(GPUMapMode.READ); const result = readback.getMappedRange().slice(0); readback.unmap(); readback.destroy();
    publication.destroy();
    return result;
  };
  const initialReuse = await runReuse(1, 0.5, false);
  const initialReuseControl = unpackOctreePowerFaceControl(new Uint32Array(initialReuse, 0, 16));
  assert.equal(initialReuseControl.valid, OCTREE_POWER_FACE_VALID); assert.equal(initialReuseControl.faceCount, 6);
  assert.equal(initialReuseControl.worldBoundaryCount, 5); assert.equal(initialReuseControl.boundaryCount, 6);
  const reuseFaceOffset = 64;
  const initialReuseWords = new Uint32Array(initialReuse, reuseFaceOffset, reuseBuilder.plan.faceBytes / 4);
  const initialReuseFloats = new Float32Array(initialReuse, reuseFaceOffset, reuseBuilder.plan.faceBytes / 4);
  const cutFace = Array.from({ length: 6 }, (_, face) => face)
    .find((face) => (initialReuseWords[face * 8 + 3] & OCTREE_POWER_FACE_OPEN_BOUNDARY) !== 0);
  assert.notEqual(cutFace, undefined, "the one absent +x neighbor must publish one compact internal cut face");

  // These would fail every geometry kernel if reuse launched any static work.
  device.queue.writeBuffer(reuseMetrics, 0, new Uint32Array(4));
  device.queue.writeBuffer(reuseHeaders, 0, new Uint32Array(12));
  const refreshedReuse = await runReuse(2, 0.45, true);
  const refreshedControl = unpackOctreePowerFaceControl(new Uint32Array(refreshedReuse, 0, 16));
  assert.equal(refreshedControl.valid, OCTREE_POWER_FACE_VALID); assert.equal(refreshedControl.generation, 2);
  const refreshedWords = new Uint32Array(refreshedReuse, reuseFaceOffset, reuseBuilder.plan.faceBytes / 4);
  const refreshedFloats = new Float32Array(refreshedReuse, reuseFaceOffset, reuseBuilder.plan.faceBytes / 4);
  for (let face = 0; face < 6; face += 1) {
    for (const word of [0, 1, 2, 3, 4, 5, 7]) {
      assert.equal(refreshedWords[face * 8 + word], initialReuseWords[face * 8 + word],
        `immutable face word ${word} changed for face ${face}`);
    }
  }
  assert.notEqual(refreshedFloats[cutFace! * 8 + 6], initialReuseFloats[cutFace! * 8 + 6],
    "the current analytic phi publication must refresh the cut-face inverse distance");
  const staticTailOffset = 64 + reuseBuilder.plan.faceBytes;
  assert.deepEqual(new Uint8Array(refreshedReuse, staticTailOffset), new Uint8Array(initialReuse, staticTailOffset),
    "incidence, normals, centroids, quadrature, and query/worklist bytes must be retained exactly");

  const repeatedReuse = await runReuse(3, 0.45, true);
  assert.deepEqual(new Uint8Array(repeatedReuse, reuseFaceOffset, reuseBuilder.plan.faceBytes),
    new Uint8Array(refreshedReuse, reuseFaceOffset, reuseBuilder.plan.faceBytes),
    "reusing the same phi generation must not compound inverseDistance/theta");
  const failedReuse = await runReuse(4, 0.8, true);
  const failedControl = unpackOctreePowerFaceControl(new Uint32Array(failedReuse, 0, 16));
  assert.equal(failedControl.valid, 0); assert.equal(failedControl.faceCount, 0);
  assert.notEqual(failedControl.flags & OCTREE_POWER_FACE_ERROR.invalidGeometry, 0);
  assert.deepEqual(new Uint8Array(failedReuse, reuseFaceOffset, reuseBuilder.plan.faceBytes),
    new Uint8Array(repeatedReuse, reuseFaceOffset, reuseBuilder.plan.faceBytes),
    "a failed candidate phi refresh must not modify the committed face publication");
  reuseBuilder.destroy(); reuseMetrics.destroy(); reuseHeaders.destroy();

  const overflow = new WebGPUOctreePowerFaces(device, 2, 1, topology, 2);
  const overflowReadback = device.createBuffer({ size: 64, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const overflowEncoder = device.createCommandEncoder();
  const overflowBroker = new PassBroker(overflowEncoder);
  overflow.encode(overflowBroker, headers, {
    dimensions, rowCount: cold.rowCount, rowDelta: cold.rowDelta, generation: 7,
  });
  overflowBroker.fence();
  overflowEncoder.copyBufferToBuffer(overflow.control, 0, overflowReadback, 0, 64);
  device.queue.submit([overflowEncoder.finish()]); await device.queue.onSubmittedWorkDone();
  await overflowReadback.mapAsync(GPUMapMode.READ);
  const overflowWords = new Uint32Array(overflowReadback.getMappedRange().slice(0)); overflowReadback.unmap();
  assert.equal(overflowWords[1], 0); assert.equal(overflowWords[2], 0);
  assert.equal(overflowWords[3] & OCTREE_POWER_FACE_ERROR.capacity, OCTREE_POWER_FACE_ERROR.capacity);
  assert.equal(overflowWords[8], 0);

  device.queue.writeBuffer(metrics, 5 * 4, new Uint32Array([0]));
  const invalidReadback = device.createBuffer({ size: 64, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const invalidEncoder = device.createCommandEncoder();
  const invalidBroker = new PassBroker(invalidEncoder);
  builder.encode(invalidBroker, headers, {
    dimensions, rowCount: cold.rowCount, rowDelta: cold.rowDelta, generation: 7,
  });
  invalidBroker.fence();
  invalidEncoder.copyBufferToBuffer(builder.control, 0, invalidReadback, 0, 64);
  device.queue.submit([invalidEncoder.finish()]); await device.queue.onSubmittedWorkDone();
  await invalidReadback.mapAsync(GPUMapMode.READ);
  const invalidWords = new Uint32Array(invalidReadback.getMappedRange().slice(0)); invalidReadback.unmap();
  assert.equal(invalidWords[1], 0); assert.equal(invalidWords[2], 0);
  assert.equal(invalidWords[3] & OCTREE_POWER_FACE_ERROR.invalidMetric, OCTREE_POWER_FACE_ERROR.invalidMetric);
  assert.ok(invalidWords[5] >= 1); assert.equal(invalidWords[8], 0);
  assert.deepEqual(validationErrors, []);

  invalidReadback.destroy();
  overflow.destroy(); overflowReadback.destroy(); cold.destroy(); builder.destroy();
  metrics.destroy(); entryHeaders.destroy(); catalogFaces.destroy(); placeholder.destroy(); headers.destroy();
  for (const retained of retainedDevices) await retained.queue.onSubmittedWorkDone();
  for (const retained of [...retainedDevices].reverse()) retained.destroy();
  retainedDevices.length = 0;
});
