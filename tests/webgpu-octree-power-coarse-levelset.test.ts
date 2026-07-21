import assert from "node:assert/strict";
import test from "node:test";
import { readFileSync } from "node:fs";
import { pathToFileURL } from "node:url";
import { OCTREE_COARSE_PHI_FLAG } from "../lib/octree-coarse-levelset";
import { decodeGeneratedOctreePowerCatalog } from "../lib/generated/octree-power-catalog";
import { WebGPUOctreeCoarseLevelSet } from "../lib/webgpu-octree-coarse-levelset";
import {
  OCTREE_POWER_COARSE_LEVELSET_CONTROL_BYTES,
  OCTREE_POWER_COARSE_LEVELSET_ERROR,
  OCTREE_POWER_COARSE_LEVELSET_VALID,
  WebGPUOctreePowerCoarseLevelSet,
  octreePowerCoarseLevelSetShader,
  planOctreePowerCoarseLevelSet,
  unpackOctreePowerCoarseLevelSetControl,
} from "../lib/webgpu-octree-power-coarse-levelset";
import { OCTREE_POWER_TOPOLOGY_VALID, WebGPUOctreePowerTopology } from "../lib/webgpu-octree-power-topology";

test("WP8 planner is compact-row bounded and exposes independent coarse/fine diagnostics", () => {
  const plan = planOctreePowerCoarseLevelSet(32, 6);
  assert.equal(plan.scratchBytes, 32 * 16 * 2);
  assert.ok(plan.allocatedBytes < 48_000);
  assert.match(octreePowerCoarseLevelSetShader, /redistancePowerCoarsePhi/);
  assert.match(octreePowerCoarseLevelSetShader, /validatePowerCoarseFineCorrection/);
  assert.match(octreePowerCoarseLevelSetShader, /acuteAnchorSolidAngle|fn acute/);
  assert.match(octreePowerCoarseLevelSetShader,
    /source\.flags&\(PHI_VALID\|PHI_FINITE\)\)\!=\(PHI_VALID\|PHI_FINITE\)[\s\S]*fail\(row,INVALID_SOURCE\)/);
  assert.match(octreePowerCoarseLevelSetShader,
    /if\(params\.physical\.y>0\.0&&\(!finite\(velocity\.x\)[\s\S]*velocity\.w<=0\.0\)\)\{fail\(row,INVALID_VELOCITY\)/,
    "the dt=0 bootstrap must not depend on a Stage-A velocity fit");
  assert.match(octreePowerCoarseLevelSetShader,
    /var value=source\.phi;if\(params\.physical\.y>0\.0&&gradient\.w>0\.0\)\{value-=/,
    "dt=0 copies the seeded coarse phi exactly before redistance");
  assert.doesNotMatch(octreePowerCoarseLevelSetShader, /texture|readback/i);
});

test("rejected fine correction preserves every byte of the prior coarse authority", () => {
  const source = readFileSync(new URL("../lib/webgpu-octree-power-coarse-levelset.ts", import.meta.url), "utf8");
  assert.doesNotMatch(source, /guardRejectedFineCoarseEntryPoints|coarseFineTransactionEntryPoints|\.replace(?:All)?\(/,
    "transaction guards and scratch addressing must be authored directly in WGSL");
  const entryPoints = ["preparePowerCoarsePhi", "clearPowerCoarsePhiSamples", "advectPowerCoarsePhi",
    "redistancePowerCoarsePhi", "validatePowerCoarseFineCorrection", "publishPowerCoarsePhi",
    "finalizePowerCoarsePhi"];
  assert.match(octreePowerCoarseLevelSetShader,
    /fn rejectedFine\(\)->bool\{return params\.hasFine!=0u&&\(arrayLength\(&fineControl\)<6u\|\|fineControl\[0\]==INVALID\|\|fineControl\[5\]!=VALID\);\}/);
  for (const entryPoint of entryPoints) {
    assert.match(octreePowerCoarseLevelSetShader,
      new RegExp(`fn ${entryPoint}\\([^)]*(?:\\)[^{]*)?\\)\\{if\\(rejectedFine\\(\\)\\)\\{return;\\}`),
      `${entryPoint} must exit before any control, record, scratch, or directory write`);
  }
  const encode = WebGPUOctreePowerCoarseLevelSet.prototype.encode.toString().replace(/\s+/g, "");
  for (const bindings of ["[0,13,14,15,16]", "[0,15,16]", "[0,1,2,5,6,7,8,9,13,16]",
    "[0,1,2,3,4,5,6,9,13,16]", "[0,11,12,13,16]", "[0,1,2,9,11,12,8,13,15,16]",
    "[0,13,15,16]"]) assert.ok(encode.includes(bindings.replace(/\s+/g, "")));
});

test("coarse publication snapshots physical power-cell volume with phi authority", () => {
  assert.match(octreePowerCoarseLevelSetShader,
    /struct SampleEntry \{[^}]*row:u32, physicalVolume:f32 \}/);
  assert.match(octreePowerCoarseLevelSetShader,
    /let metric=metrics\[row\][\s\S]*physicalVolume=metric\.volume\*extent\*extent\*extent/,
    "the directory must remain a complete volume snapshot after pressure rows rebuild");
  assert.match(octreePowerCoarseLevelSetShader,
    /sampleDirectory\.entries\[slot\]\.physicalVolume=physicalVolume/);
});

test("every coarse schedule bind group equals transitive WGSL reachability", () => {
  const bindings = new Map<string, number>();
  for (const match of octreePowerCoarseLevelSetShader.matchAll(
    /@group\(0\)\s*@binding\((\d+)\)\s*var(?:<[^>]+>)?\s+([A-Za-z_]\w*)/g,
  )) bindings.set(match[2], Number(match[1]));
  const bodies = new Map<string, string>();
  for (const match of octreePowerCoarseLevelSetShader.matchAll(/\bfn\s+([A-Za-z_]\w*)\s*\(/g)) {
    const open = octreePowerCoarseLevelSetShader.indexOf("{", match.index); let depth = 0, close = -1;
    for (let at = open; at < octreePowerCoarseLevelSetShader.length; at += 1) {
      if (octreePowerCoarseLevelSetShader[at] === "{") depth += 1;
      if (octreePowerCoarseLevelSetShader[at] === "}" && --depth === 0) { close = at; break; }
    }
    assert.ok(open >= 0 && close > open); bodies.set(match[1], octreePowerCoarseLevelSetShader.slice(open + 1, close));
  }
  const reachable = (entryPoint: string) => {
    const pending = [entryPoint], reached = new Set<string>(), result = new Set<number>();
    while (pending.length > 0) {
      const name = pending.pop()!; if (reached.has(name)) continue; reached.add(name);
      const body = bodies.get(name); assert.notEqual(body, undefined);
      for (const [global, binding] of bindings) if (new RegExp(`\\b${global}\\b`).test(body!)) result.add(binding);
      for (const callee of bodies.keys()) if (!reached.has(callee)
        && new RegExp(`\\b${callee}\\s*\\(`).test(body!)) pending.push(callee);
    }
    return [...result].sort((a, b) => a - b);
  };

  const observed = new Map<string, number[]>(); let id = 0;
  const device = {
    queue: { writeBuffer() {} },
    createBuffer: () => ({ id: id += 1 }), createShaderModule: () => ({}),
    createComputePipeline: ({ compute }: GPUComputePipelineDescriptor) => ({ entryPoint: compute.entryPoint,
      getBindGroupLayout: () => ({ entryPoint: compute.entryPoint }) }),
    createBindGroup: ({ layout, entries }: { layout: { entryPoint: string }; entries: { binding: number }[] }) => {
      observed.set(layout.entryPoint, entries.map(({ binding }) => binding).sort((a, b) => a - b)); return {};
    },
  } as unknown as GPUDevice;
  const buffer = {} as GPUBuffer;
  const coarse = { plan: { rowCapacity: 1 }, records: buffer } as unknown as WebGPUOctreeCoarseLevelSet;
  const topology = { metrics: buffer, catalogTetrahedronHeaders: buffer, catalogTetrahedra: buffer,
    catalogTetrahedronVertices: buffer };
  const pass = { setPipeline() {}, setBindGroup() {}, dispatchWorkgroups() {}, end() {} };
  const encoder = { beginComputePass: () => pass, copyBufferToBuffer() {} } as unknown as GPUCommandEncoder;
  const previousUsage = Object.getOwnPropertyDescriptor(globalThis, "GPUBufferUsage");
  Object.defineProperty(globalThis, "GPUBufferUsage", { configurable: true,
    value: { STORAGE: 1, COPY_SRC: 2, COPY_DST: 4, UNIFORM: 8 } });
  try {
    const schedule = new WebGPUOctreePowerCoarseLevelSet(device, coarse,
      topology as unknown as ConstructorParameters<typeof WebGPUOctreePowerCoarseLevelSet>[2], 1);
    schedule.encode(encoder, { headers: buffer, cellVelocities: buffer, siteIndex: buffer, rowCount: 1 },
      { dimensions: [1, 1, 1], physicalCellSize: 1, dt: 0, hashCapacity: 1, generation: 1 });
  } finally {
    if (previousUsage) Object.defineProperty(globalThis, "GPUBufferUsage", previousUsage);
    else Reflect.deleteProperty(globalThis, "GPUBufferUsage");
  }
  const entryPoints = [...octreePowerCoarseLevelSetShader.matchAll(
    /@compute\s+@workgroup_size\([^)]*\)\s*fn\s+([A-Za-z_]\w*)/g,
  )].map((match) => match[1]);
  assert.deepEqual([...observed.keys()].sort(), [...entryPoints].sort());
  for (const entryPoint of entryPoints) assert.deepEqual(observed.get(entryPoint), reachable(entryPoint), entryPoint);
});

test("Dawn runs live-row advection, redistance, and O(rows) fine aggregate correction without readback dependencies", {
  skip: !process.env.WEBGPU_NODE_MODULE && "set WEBGPU_NODE_MODULE for WP8 GPU checks",
}, async () => {
  const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href) as { create(options: string[]): GPU; globals: Record<string, unknown> };
  Object.assign(globalThis, dawn.globals); const nativeGpu = dawn.create([`backend=${process.env.WEBGPU_BACKEND ?? "metal"}`]);
  const adapter = await nativeGpu.requestAdapter(); assert.ok(adapter); const device = await adapter.requestDevice();
  const compilation = await device.createShaderModule({ code: octreePowerCoarseLevelSetShader }).getCompilationInfo();
  assert.deepEqual(compilation.messages.filter((message) => message.type === "error"), []);
  const bytes = readFileSync(new URL("../lib/generated/octree-power-catalog.bin", import.meta.url));
  const catalog = decodeGeneratedOctreePowerCatalog(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
  const rowCount = 8, topology = new WebGPUOctreePowerTopology(device, rowCount, catalog);
  const coarse = new WebGPUOctreeCoarseLevelSet(device, rowCount);
  const uniformEntry = catalog.sameOrFinerDirect[0x3ffff] & 0xffff;
  const metricData = new ArrayBuffer(rowCount * 16), metricWords = new Uint32Array(metricData), metricFloats = new Float32Array(metricData);
  for (let row = 0; row < rowCount; row += 1) { metricWords[row * 4] = uniformEntry;
    metricWords[row * 4 + 1] = OCTREE_POWER_TOPOLOGY_VALID; metricFloats[row * 4 + 2] = 1; }
  device.queue.writeBuffer(topology.metrics, 0, metricData);
  const records = new Map(Array.from({ length: rowCount }, (_, row) => { const x = row & 1, phi = x - 0.5;
    return [row, { phi, minimumPhi: phi, maximumPhi: phi,
      flags: OCTREE_COARSE_PHI_FLAG.valid | OCTREE_COARSE_PHI_FLAG.finite, generation: 0, fineSampleCount: 0 }] as const; }));
  coarse.upload(records);
  const headerData = new ArrayBuffer(rowCount * 48), headerWords = new Uint32Array(headerData);
  for (let row = 0; row < rowCount; row += 1) { headerWords[row * 12] = row; headerWords[row * 12 + 3] = 1; }
  const velocities = new Float32Array(rowCount * 4); for (let row = 0; row < rowCount; row += 1) velocities[row * 4 + 3] = 1;
  const hashCapacity = 16, siteWords = new Uint32Array(hashCapacity * 4);
  const hash = (cell: number, size: number) => { let value = (cell ^ Math.imul(size, 0x9e3779b9)) >>> 0;
    value = Math.imul((value ^ (value >>> 16)) >>> 0, 0x7feb352d) >>> 0;
    value = Math.imul((value ^ (value >>> 15)) >>> 0, 0x846ca68b) >>> 0; return (value ^ (value >>> 16)) >>> 0; };
  for (let row = 0; row < rowCount; row += 1) { let slot = hash(row, 1) & (hashCapacity - 1);
    while (siteWords[slot * 4] !== 0) slot = (slot + 1) & (hashCapacity - 1);
    siteWords.set([row + 1, 1, row, 0], slot * 4); }
  const upload = (data: ArrayBufferView) => { const buffer = device.createBuffer({ size: Math.max(4, data.byteLength),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, mappedAtCreation: true });
    new Uint8Array(buffer.getMappedRange()).set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength)); buffer.unmap(); return buffer; };
  const headers = upload(new Uint8Array(headerData)), velocityBuffer = upload(velocities), siteIndex = upload(siteWords);
  const offsets = upload(new Uint32Array([0, 1, 2, 3, 4, 5, 6, 7, 8]));
  const aggregateData = new ArrayBuffer(rowCount * 16), aggregateFloats = new Float32Array(aggregateData);
  const aggregateWords = new Uint32Array(aggregateData);
  aggregateFloats.set([-0.05, -0.05, 0.05]); aggregateWords[3] = 1;
  const contributions = upload(new Uint8Array(aggregateData));
  const schedule = new WebGPUOctreePowerCoarseLevelSet(device, coarse, topology.source, 2);
  const directoryOffset = OCTREE_POWER_COARSE_LEVELSET_CONTROL_BYTES + coarse.plan.recordBytes;
  const readback = device.createBuffer({ size: directoryOffset + schedule.plan.sampleDirectoryBytes,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const encoder = device.createCommandEncoder(); schedule.encode(encoder, { headers, cellVelocities: velocityBuffer, siteIndex,
    rowCount, fineCorrection: { rowOffsets: offsets, contributions, contributionCount: rowCount,
      maximumContributionsPerRow: 1, aggregated: true } },
  { dimensions: [2, 2, 2], physicalCellSize: 1, dt: 0, hashCapacity, generation: 41 });
  encoder.copyBufferToBuffer(schedule.control, 0, readback, 0, OCTREE_POWER_COARSE_LEVELSET_CONTROL_BYTES);
  encoder.copyBufferToBuffer(coarse.records, 0, readback, OCTREE_POWER_COARSE_LEVELSET_CONTROL_BYTES, coarse.plan.recordBytes);
  encoder.copyBufferToBuffer(schedule.sampleDirectory, 0, readback, directoryOffset, schedule.plan.sampleDirectoryBytes);
  device.queue.submit([encoder.finish()]); schedule.retireSubmittedEncoder(encoder);
  await readback.mapAsync(GPUMapMode.READ); const result = readback.getMappedRange().slice(0); readback.unmap();
  assert.deepEqual(unpackOctreePowerCoarseLevelSetControl(new Uint32Array(result, 0, 16)), {
    flags: 0, firstErrorRow: 0xffff_ffff, rowCount, advectedRows: rowCount, uniformUpdates: rowCount * 2,
    transitionUpdates: 0, nearestFallbacks: 0, redistancePasses: 2, correctedRows: 1, interfaceRows: 1,
    contributionCount: rowCount, generation: 41, valid: OCTREE_POWER_COARSE_LEVELSET_VALID,
  });
  const output = new Float32Array(result, OCTREE_POWER_COARSE_LEVELSET_CONTROL_BYTES, rowCount * 4);
  assert.ok(Math.abs(output[0] + 0.05) < 1e-6); assert.ok(Math.abs(output[1] + 0.05) < 1e-6);
  assert.ok(Math.abs(output[2] - 0.05) < 1e-6);
  const directory = new Uint32Array(result, directoryOffset, schedule.plan.sampleDirectoryBytes / 4);
  assert.deepEqual(Array.from(directory.subarray(0, 7)),
    [OCTREE_POWER_COARSE_LEVELSET_VALID, 41, schedule.plan.sampleHashCapacity, 2, 2, 2, 2]);
  assert.equal(new Float32Array(result, directoryOffset + 28, 1)[0], 1);
  let indexedRows = 0;
  for (let slot = 0; slot < schedule.plan.sampleHashCapacity; slot += 1) {
    const base = 8 + slot * 8;
    if (directory[base] === 0) continue;
    indexedRows += 1;
    assert.equal(directory[base + 5] & (OCTREE_COARSE_PHI_FLAG.valid | OCTREE_COARSE_PHI_FLAG.finite),
      OCTREE_COARSE_PHI_FLAG.valid | OCTREE_COARSE_PHI_FLAG.finite);
  }
  assert.equal(indexedRows, rowCount, "PUBLISHED implies every requested compact row has a directory key");

  // Bootstrap is a geometry/sign publication. A rank-deficient Stage-A row
  // has w=0, but at dt=0 its velocity is mathematically unused and must not
  // block the first coarse directory. Once time advances, the same input is
  // strictly invalid and publication remains fail-closed.
  device.queue.writeBuffer(velocityBuffer, 0, new Float32Array(rowCount * 4));
  const runVelocityGuard = async (dt: number, generation: number) => {
    const guardReadback = device.createBuffer({ size: OCTREE_POWER_COARSE_LEVELSET_CONTROL_BYTES + 4,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const guardEncoder = device.createCommandEncoder();
    schedule.encode(guardEncoder, { headers, cellVelocities: velocityBuffer, siteIndex, rowCount },
      { dimensions: [2, 2, 2], physicalCellSize: 1, dt, hashCapacity, generation });
    guardEncoder.copyBufferToBuffer(schedule.control, 0, guardReadback, 0,
      OCTREE_POWER_COARSE_LEVELSET_CONTROL_BYTES);
    guardEncoder.copyBufferToBuffer(schedule.sampleDirectory, 0, guardReadback,
      OCTREE_POWER_COARSE_LEVELSET_CONTROL_BYTES, 4);
    device.queue.submit([guardEncoder.finish()]); schedule.retireSubmittedEncoder(guardEncoder);
    await guardReadback.mapAsync(GPUMapMode.READ);
    const bytes = guardReadback.getMappedRange().slice(0); guardReadback.unmap(); guardReadback.destroy();
    return { control: unpackOctreePowerCoarseLevelSetControl(new Uint32Array(bytes, 0, 16)),
      directoryState: new Uint32Array(bytes)[16] };
  };
  const bootstrapWithoutVelocity = await runVelocityGuard(0, 42);
  assert.equal(bootstrapWithoutVelocity.control.flags, 0);
  assert.equal(bootstrapWithoutVelocity.control.valid, OCTREE_POWER_COARSE_LEVELSET_VALID);
  assert.equal(bootstrapWithoutVelocity.directoryState, OCTREE_POWER_COARSE_LEVELSET_VALID);
  const advancedWithoutVelocity = await runVelocityGuard(0.01, 43);
  assert.equal(advancedWithoutVelocity.control.flags, OCTREE_POWER_COARSE_LEVELSET_ERROR.invalidVelocity);
  assert.equal(advancedWithoutVelocity.control.valid, 0);
  assert.equal(advancedWithoutVelocity.directoryState, 0);

  coarse.upload(new Map(Array.from({ length: rowCount }, (_, row) => [row, {
    phi: 0, minimumPhi: 0, maximumPhi: 0, flags: 0, generation: 0, fineSampleCount: 0,
  }] as const)));
  const invalidReadback = device.createBuffer({ size: OCTREE_POWER_COARSE_LEVELSET_CONTROL_BYTES + 4,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const invalidEncoder = device.createCommandEncoder();
  schedule.encode(invalidEncoder, { headers, cellVelocities: velocityBuffer, siteIndex, rowCount },
    { dimensions: [2, 2, 2], physicalCellSize: 1, dt: 0, hashCapacity, generation: 44 });
  invalidEncoder.copyBufferToBuffer(schedule.control, 0, invalidReadback, 0,
    OCTREE_POWER_COARSE_LEVELSET_CONTROL_BYTES);
  invalidEncoder.copyBufferToBuffer(schedule.sampleDirectory, 0, invalidReadback,
    OCTREE_POWER_COARSE_LEVELSET_CONTROL_BYTES, 4);
  device.queue.submit([invalidEncoder.finish()]); schedule.retireSubmittedEncoder(invalidEncoder);
  await invalidReadback.mapAsync(GPUMapMode.READ);
  const invalidResult = invalidReadback.getMappedRange().slice(0); invalidReadback.unmap();
  const invalidControl = unpackOctreePowerCoarseLevelSetControl(new Uint32Array(invalidResult, 0, 16));
  assert.equal(invalidControl.flags, OCTREE_POWER_COARSE_LEVELSET_ERROR.invalidSource,
    "invalid source records fail closed instead of becoming valid zero phi");
  assert.ok(invalidControl.firstErrorRow < rowCount);
  assert.equal(invalidControl.valid, 0);
  assert.equal(new Uint32Array(invalidResult)[16], 0, "invalid source cannot publish an authoritative directory");
  schedule.destroy(); headers.destroy(); velocityBuffer.destroy(); siteIndex.destroy(); offsets.destroy(); contributions.destroy();
  readback.destroy(); invalidReadback.destroy(); coarse.destroy(); topology.destroy(); device.destroy();
});
