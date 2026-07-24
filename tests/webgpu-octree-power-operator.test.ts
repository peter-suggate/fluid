import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { decodeGeneratedOctreePowerCatalog } from "../lib/generated/octree-power-catalog";
import { createOctreePowerSite, powerBoxBoundary } from "../lib/octree-power-geometry";
import {
  applyOctreePowerMatrix,
  buildOctreePowerOperator,
  octreePowerBoundaryDistance,
  octreePowerDivergence,
  projectOctreePowerFaceVelocities,
  type OctreePowerFaceRecord,
} from "../lib/octree-power-operator";
import {
  OCTREE_POWER_GPU_ASSEMBLED,
  OCTREE_POWER_GPU_CONTROL_BYTES,
  OCTREE_POWER_GPU_ERROR,
  OCTREE_POWER_GPU_PROJECTED,
  WebGPUOctreePowerOperator,
  octreePowerOperatorShader,
  planOctreePowerGPUOperator,
  type OctreePowerGPUOperatorSource,
} from "../lib/webgpu-octree-power-operator";
import {
  OCTREE_POWER_REGULAR_DESCRIPTOR,
  OCTREE_POWER_TOPOLOGY_VALID,
  WebGPUOctreePowerTopology,
} from "../lib/webgpu-octree-power-topology";
import { PassBroker } from "../lib/webgpu-pass-broker";

const close = (actual: number, expected: number, tolerance = 2e-5) => assert.ok(Math.abs(actual - expected) <= tolerance,
  `expected ${actual} to be within ${tolerance} of ${expected}`);
const encodeWithBroker = (encoder: GPUCommandEncoder, encode: (broker: PassBroker) => void) => {
  const broker = new PassBroker(encoder);
  encode(broker);
  broker.fence("power operator test publication");
};

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

test("every power-operator pipeline binds its exact transitive WGSL resources", () => {
  const observed = new Map<string, number[]>();
  const buffer = { destroy() {}, size: 1 << 20 } as unknown as GPUBuffer;
  const device = {
    queue: { writeBuffer() {} },
    createBuffer: () => buffer,
    createShaderModule: ({ code }: GPUShaderModuleDescriptor) => ({ code }),
    createComputePipeline: ({ compute }: GPUComputePipelineDescriptor) => ({
      entryPoint: compute.entryPoint,
      shader: (compute.module as unknown as { code: string }).code,
      getBindGroupLayout() { return { entryPoint: this.entryPoint, shader: this.shader }; },
    }),
    createBindGroup: ({ layout, entries }: {
      layout: { entryPoint: string; shader: string };
      entries: Iterable<GPUBindGroupEntry>;
    }) => {
      const actual = Array.from(entries, ({ binding }) => binding).sort((a, b) => a - b);
      observed.set(`${layout.entryPoint}\n${layout.shader}`, actual);
      return {};
    },
  } as unknown as GPUDevice;
  const topology = {
    plan: { rowCapacity: 2, entryCount: 1 },
    metrics: buffer, catalogCoefficients: buffer,
  } as unknown as ConstructorParameters<typeof WebGPUOctreePowerOperator>[5]["topology"];
  const pass = { setPipeline() {}, setBindGroup() {}, dispatchWorkgroups() {}, end() {} };
  const encoderObject = {
    beginComputePass: () => pass,
    clearBuffer: () => undefined,
    copyBufferToBuffer: () => undefined,
  };
  const encoder = encoderObject as unknown as GPUCommandEncoder;
  const previousUsage = Object.getOwnPropertyDescriptor(globalThis, "GPUBufferUsage");
  Object.defineProperty(globalThis, "GPUBufferUsage", { configurable: true,
    value: { STORAGE: 1, COPY_SRC: 2, COPY_DST: 4, UNIFORM: 8 } });
  try {
    const operator = new WebGPUOctreePowerOperator(device, 2, 4, 8, 4, {
      topology, leafHeaders: buffer, physicalCellSize: 1, physicalCellVolume: 1,
    });
    const broker = new PassBroker(encoder);
    const csr = { incidenceRows: buffer, incidence: buffer };
    operator.encodeAssemblyFromControl(broker, buffer, csr, buffer, buffer, buffer);
    operator.encodeProjectionFromControl(broker, buffer, csr, buffer, buffer);
    operator.encodeProjectionFromControl(broker, buffer, csr, buffer, buffer, 1, buffer);
    operator.encodeLeafRowPublication(broker, buffer, buffer);
  } finally {
    if (previousUsage) Object.defineProperty(globalThis, "GPUBufferUsage", previousUsage);
    else Reflect.deleteProperty(globalThis, "GPUBufferUsage");
  }
  assert.equal(observed.size, 13);
  for (const [key, actual] of observed) {
    const split = key.indexOf("\n");
    assert.deepEqual(actual, reachableBindings(key.slice(split + 1), key.slice(0, split)),
      `${key.slice(0, split)} host bindings must equal transitive WGSL reachability`);
  }
});

function transitionPatch() {
  const sites = [createOctreePowerSite("coarse", [0, 0, 0], 2)];
  for (let z = 0; z < 3; z += 1) for (let y = 0; y < 3; y += 1) for (let x = 0; x < 3; x += 1) {
    if (x < 2 && y < 2 && z < 2) continue;
    sites.push(createOctreePowerSite(`${x}${y}${z}`, [x, y, z], 1));
  }
  const pressureForCenter = (center: readonly number[]) => 0.5 * center[0] - center[1] + 0.25 * center[2];
  return {
    sites,
    boundaries: powerBoxBoundary([0, 0, 0], [3, 3, 3]),
    pressureForCenter,
  };
}

function packFaces(faces: readonly OctreePowerFaceRecord[]): Uint32Array {
  const data = new ArrayBuffer(faces.length * 32);
  const words = new Uint32Array(data); const floats = new Float32Array(data);
  faces.forEach((face, index) => {
    const offset = index * 8;
    words[offset] = face.negativeRow; words[offset + 1] = face.positiveRow;
    words[offset + 2] = 0; words[offset + 3] = face.positiveRow === 0xffff_ffff ? 1 : 0;
    floats[offset + 4] = face.normalVelocity; floats[offset + 5] = face.area;
    floats[offset + 6] = face.inverseDistance; floats[offset + 7] = face.openFraction;
  });
  return words;
}

function packRawFaces(records: readonly (readonly [number, number, number, number, number?])[]): Uint32Array {
  const data = new ArrayBuffer(records.length * 32);
  const words = new Uint32Array(data); const floats = new Float32Array(data);
  records.forEach(([negative, positive, velocity, coefficientData, flags = 0], index) => {
    const offset = index * 8; words[offset] = negative; words[offset + 1] = positive;
    words[offset + 3] = flags;
    floats[offset + 4] = velocity; floats[offset + 5] = 1; floats[offset + 6] = coefficientData; floats[offset + 7] = 1;
  });
  return words;
}

function packCSR(rows: readonly (readonly { face: number; sign: number }[])[]) {
  const rowWords = new Uint32Array((rows.length + 1) * 4);
  const incidenceWords = new Uint32Array(rows.reduce((sum, row) => sum + row.length, 0) * 2);
  const incidenceSigns = new Int32Array(incidenceWords.buffer); let cursor = 0;
  rows.forEach((row, rowIndex) => {
    rowWords[rowIndex * 4 + 3] = cursor;
    row.forEach((item) => { incidenceWords[cursor * 2] = item.face; incidenceSigns[cursor * 2 + 1] = item.sign; cursor += 1; });
  });
  rowWords[rows.length * 4 + 3] = cursor;
  return { rowWords, incidenceWords, incidenceCount: cursor };
}

test("GPU power-operator planner is proportional to explicit compact capacities", () => {
  const plan = planOctreePowerGPUOperator(100, 400, 640, 30);
  assert.equal(plan.rowBytes, 1_600);
  assert.equal(plan.faceBytes, 12_800);
  assert.equal(plan.entryOffsetBytes, 404);
  assert.equal(plan.entryBytes, 5_120);
  assert.equal(plan.scalarBytes, 2_000);
  assert.equal(plan.faceDiagnosticOffset, 9_124);
  assert.equal(plan.rowDiagnosticOffset, 15_524);
  assert.equal(plan.arenaBytes, 17_136);
  assert.equal(plan.allocatedBytes, 17_328);
  assert.throws(() => planOctreePowerGPUOperator(1, 1, 0, 1), /positive integer/);
});

test("GPU power shader codifies one shared coefficient and fail-closed publication", () => {
  assert.match(octreePowerOperatorShader, /fn catalogCoefficient\(face:PowerFaceRecord\)->f32/);
  assert.match(octreePowerOperatorShader, /emitBulkPowerRows/);
  assert.match(octreePowerOperatorShader, /emitCutPowerRows/);
  assert.match(octreePowerOperatorShader, /bitcast<u32>\(coefficient\)==bitcast<u32>\(dynamicCoefficient\)/);
  assert.match(octreePowerOperatorShader, /merged\.openFraction\*merged\.area\*merged\.inverseDistance/);
  assert.match(octreePowerOperatorShader, /\(positive-negative\)\*face\.inverseDistance/);
  assert.match(octreePowerOperatorShader, /let value=integrated\/volume/);
  assert.match(octreePowerOperatorShader, /item\.face<=previousFace/);
  assert.match(octreePowerOperatorShader, /control\.flags=ASSEMBLED/);
  assert.match(octreePowerOperatorShader, /control\.flags!=ASSEMBLED/);
  assert.doesNotMatch(octreePowerOperatorShader,
    /\batomic(?:Add|And|CompareExchangeWeak|Exchange|Load|Max|Min|Or|Store|Sub|Xor)\b|atomic<[ui]32>/,
    "the recurring operator must reduce fixed diagnostic records without synchronization atomics");
  assert.match(octreePowerOperatorShader, /header\.entryStart=start;header\.entryCount=finish-start/);
});

test("Dawn assembles symmetric compact rows and projects the same generalized faces", {
  skip: !process.env.WEBGPU_NODE_MODULE && "set WEBGPU_NODE_MODULE for GPU power-operator checks",
}, async () => {
  const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href) as { create(options: string[]): GPU; globals: Record<string, unknown> };
  Object.assign(globalThis, dawn.globals);
  const nativeGpu = dawn.create(["backend=metal"]);
  const adapter = await nativeGpu.requestAdapter(); assert.ok(adapter);
  const device = await adapter.requestDevice({ requiredLimits: {
    maxStorageBuffersPerShaderStage: Math.min(10, adapter.limits.maxStorageBuffersPerShaderStage),
  } });
  const validationErrors: string[] = [];
  device.addEventListener("uncapturederror", (event: unknown) => validationErrors.push((event as { error: { message: string } }).error.message));
  const compilation = await device.createShaderModule({ code: octreePowerOperatorShader }).getCompilationInfo();
  assert.deepEqual(compilation.messages.filter((message) => message.type === "error"), []);

  const patch = transitionPatch();
  const sortedSites = [...patch.sites].sort((a, b) => a.key.localeCompare(b.key));
  const pressure = sortedSites.map((site) => patch.pressureForCenter(site.center));
  const pressureByKey = new Map(sortedSites.map((site, row) => [site.key, pressure[row]]));
  const aperture = (negativeKey: string, positiveKey: string) => 0.25 + 0.125 * ((negativeKey.length + positiveKey.length) % 5);
  const cpu = buildOctreePowerOperator(patch.sites, patch.boundaries, {
    openFraction: (negative, positive) => positive ? aperture(negative.key, positive.key) : 1,
    normalVelocity: (_centroid, _normal, negative, positive) => positive
      ? aperture(negative.key, positive.key) * (pressureByKey.get(positive.key)! - pressureByKey.get(negative.key)!)
        / Math.hypot(...positive.center.map((value, axis) => value - negative.center[axis]))
      : 0,
  });
  const entryCount = cpu.rows.reduce((sum, row) => sum + row.entries.length, 0);
  const upload = (data: ArrayBufferView, usage = GPUBufferUsage.STORAGE) => {
    const buffer = device.createBuffer({ size: Math.max(4, data.byteLength), usage: usage | GPUBufferUsage.COPY_DST, mappedAtCreation: true });
    new Uint8Array(buffer.getMappedRange(), 0, data.byteLength).set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
    buffer.unmap(); return buffer;
  };
  const catalogBytes = readFileSync(new URL("../lib/generated/octree-power-catalog.bin", import.meta.url));
  const catalog = decodeGeneratedOctreePowerCatalog(
    catalogBytes.buffer.slice(catalogBytes.byteOffset, catalogBytes.byteOffset + catalogBytes.byteLength));
  const coefficientTopology = new WebGPUOctreePowerTopology(device, cpu.rows.length, catalog);
  const coefficientHeaders = device.createBuffer({ size: cpu.rows.length * 48,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const coefficientMetricWords = new Uint32Array(cpu.rows.length * 4);
  const coefficientMetricFloats = new Float32Array(coefficientMetricWords.buffer);
  const coefficientHeaderWords = new Uint32Array(cpu.rows.length * 12);
  cpu.rows.forEach((row, index) => {
    coefficientMetricFloats[index * 4 + 2] = row.volume;
    coefficientHeaderWords[index * 12 + 3] = 1;
  });
  device.queue.writeBuffer(coefficientTopology.metrics, 0, coefficientMetricWords);
  device.queue.writeBuffer(coefficientHeaders, 0, coefficientHeaderWords);
  const coefficientSource = { topology: coefficientTopology.source, leafHeaders: coefficientHeaders,
    physicalCellSize: 1, physicalCellVolume: 1 };
  const gpu = new WebGPUOctreePowerOperator(device, cpu.rows.length, cpu.faces.length, entryCount,
    cpu.maximumIncidence, coefficientSource);
  const uploadCSR = (rows: readonly (readonly { face: number; sign: number }[])[]) => {
    const packed = packCSR(rows);
    return { source: { incidenceRows: upload(packed.rowWords), incidence: upload(packed.incidenceWords) }, count: packed.incidenceCount };
  };
  const faceData = packFaces(cpu.faces);
  const faces = upload(faceData, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
  const pressureBuffer = upload(new Float32Array(pressure));
  const cpuCSRData = packCSR(cpu.incidence);
  const cpuCSR = { incidenceRows: upload(cpuCSRData.rowWords), incidence: upload(cpuCSRData.incidenceWords) };
  const gpuCounts = upload(new Uint32Array([cpu.rows.length, cpu.faces.length, cpuCSRData.incidenceCount]),
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);

  const readSource = async (source: OctreePowerGPUOperatorSource) => {
    const segments = [OCTREE_POWER_GPU_CONTROL_BYTES, source.plan.rowBytes, source.plan.entryOffsetBytes, source.plan.entryBytes,
      source.plan.faceCapacity * 4, source.plan.rowCapacity * 4];
    const total = segments.reduce((sum, size) => sum + size, 0);
    const readback = device.createBuffer({ size: total, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const encoder = device.createCommandEncoder(); let offset = 0;
    const sourceOffsets = [0, source.plan.rowOffset, source.plan.entryOffsetOffset, source.plan.entryOffset,
      source.plan.projectedOffset, source.plan.divergenceOffset];
    [source.control, source.arena, source.arena, source.arena, source.arena, source.arena].forEach((buffer, index) => {
      encoder.copyBufferToBuffer(buffer, sourceOffsets[index], readback, offset, segments[index]); offset += segments[index];
    });
    device.queue.submit([encoder.finish()]); await device.queue.onSubmittedWorkDone(); await readback.mapAsync(GPUMapMode.READ);
    const bytes = readback.getMappedRange().slice(0); readback.unmap(); readback.destroy(); return { bytes, segments };
  };
  const slice = (bytes: ArrayBuffer, segments: number[], index: number) => {
    const offset = segments.slice(0, index).reduce((sum, size) => sum + size, 0);
    return bytes.slice(offset, offset + segments[index]);
  };

  let encoder = device.createCommandEncoder();
  encodeWithBroker(encoder, (broker) => gpu.encodeAssemblyFromControl(broker, faces, cpuCSR, gpuCounts));
  device.queue.submit([encoder.finish()]); await device.queue.onSubmittedWorkDone();
  let result = await readSource(gpu.source);
  let control = new Uint32Array(slice(result.bytes, result.segments, 0));
  assert.deepEqual([...control.slice(0, 7)], [OCTREE_POWER_GPU_ASSEMBLED, 0xffff_ffff, cpu.rows.length, cpu.faces.length,
    cpu.incidence.reduce((sum, row) => sum + row.length, 0), entryCount, 0]);
  const rowFloats = new Float32Array(slice(result.bytes, result.segments, 1));
  const entryOffsets = new Uint32Array(slice(result.bytes, result.segments, 2));
  const entryWords = new Uint32Array(slice(result.bytes, result.segments, 3));
  const entryFloats = new Float32Array(entryWords.buffer);
  cpu.rows.forEach((row, rowIndex) => {
    close(rowFloats[rowIndex * 4], row.diagonal); close(rowFloats[rowIndex * 4 + 1], row.rhs);
    close(rowFloats[rowIndex * 4 + 2], row.volume);
    assert.equal(entryOffsets[rowIndex + 1] - entryOffsets[rowIndex], row.entries.length);
    row.entries.forEach((entry, local) => {
      const index = entryOffsets[rowIndex] + local;
      assert.equal(entryWords[index * 2], entry.row); close(entryFloats[index * 2 + 1], entry.coefficient);
    });
  });
  // Successful publication replaces only the Chebyshev row fields while
  // retaining topology identity and reconstructed gradients in LeafHeader.
  const leafHeaderWords = new Uint32Array(cpu.rows.length * 12);
  const leafHeaderFloats = new Float32Array(leafHeaderWords.buffer);
  cpu.rows.forEach((_row, row) => {
    leafHeaderWords[row * 12] = 1000 + row; leafHeaderWords[row * 12 + 3] = 2;
    leafHeaderWords[row * 12 + 6] = 0xdead_beef; leafHeaderWords[row * 12 + 7] = 0xcafe_babe;
    leafHeaderFloats[row * 12 + 8] = row + 0.25;
  });
  const publishedHeaders = upload(leafHeaderWords, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
  const publishedEntries = device.createBuffer({ size: entryCount * 8,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  encoder = device.createCommandEncoder();
  encodeWithBroker(encoder,
    (broker) => gpu.encodeLeafRowPublication(broker, publishedHeaders, publishedEntries));
  const publishedReadback = device.createBuffer({ size: publishedHeaders.size + publishedEntries.size,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  encoder.copyBufferToBuffer(publishedHeaders, 0, publishedReadback, 0, publishedHeaders.size);
  encoder.copyBufferToBuffer(publishedEntries, 0, publishedReadback, publishedHeaders.size, publishedEntries.size);
  device.queue.submit([encoder.finish()]); await device.queue.onSubmittedWorkDone();
  await publishedReadback.mapAsync(GPUMapMode.READ); const publication = publishedReadback.getMappedRange().slice(0);
  publishedReadback.unmap(); publishedReadback.destroy();
  const headersOut = new Uint32Array(publication, 0, publishedHeaders.size / 4);
  const headersOutFloats = new Float32Array(publication, 0, publishedHeaders.size / 4);
  const entriesOut = new Uint32Array(publication, publishedHeaders.size, publishedEntries.size / 4);
  const entriesOutFloats = new Float32Array(publication, publishedHeaders.size, publishedEntries.size / 4);
  cpu.rows.forEach((row, rowIndex) => {
    assert.equal(headersOut[rowIndex * 12], 1000 + rowIndex); assert.equal(headersOut[rowIndex * 12 + 3], 2);
    assert.equal(headersOut[rowIndex * 12 + 1], entryOffsets[rowIndex]);
    assert.equal(headersOut[rowIndex * 12 + 2], row.entries.length);
    close(headersOutFloats[rowIndex * 12 + 4], row.diagonal); close(headersOutFloats[rowIndex * 12 + 5], row.rhs);
    const expectedRowFlags = cpu.incidence[rowIndex].some(({ face }) =>
      cpu.faces[face].positiveRow === 0xffff_ffff || cpu.faces[face].openFraction < 1) ? 1 : 0;
    assert.equal(headersOut[rowIndex * 12 + 6], expectedRowFlags); assert.equal(headersOut[rowIndex * 12 + 7], 0);
    close(headersOutFloats[rowIndex * 12 + 8], rowIndex + 0.25);
  });
  for (let index = 0; index < entryCount; index += 1) {
    assert.equal(entriesOut[index * 2], entryWords[index * 2]); close(entriesOutFloats[index * 2 + 1], entryFloats[index * 2 + 1]);
  }
  // Matrix decoded from GPU rows is symmetric, has a constant nullspace, and non-negative energy.
  cpu.rows.forEach((_row, row) => {
    for (let index = entryOffsets[row]; index < entryOffsets[row + 1]; index += 1) {
      const neighbor = entryWords[index * 2]; const coefficient = entryFloats[index * 2 + 1];
      let reciprocal = -1;
      for (let candidate = entryOffsets[neighbor]; candidate < entryOffsets[neighbor + 1]; candidate += 1) {
        if (entryWords[candidate * 2] === row) reciprocal = candidate;
      }
      assert.ok(reciprocal >= 0); close(entryFloats[reciprocal * 2 + 1], coefficient);
    }
  });
  applyOctreePowerMatrix(cpu, cpu.rows.map(() => 1)).forEach((value) => close(value, 0));
  for (let trial = 0; trial < 8; trial += 1) {
    const vector = cpu.rows.map((_, row) => Math.sin(row * 1.7 + trial));
    const applied = rowFloats.filter((_value, index) => index % 4 === 0).map((diagonal, row) => {
      let value = diagonal * vector[row];
      for (let index = entryOffsets[row]; index < entryOffsets[row + 1]; index += 1) value -= entryFloats[index * 2 + 1] * vector[entryWords[index * 2]];
      return value;
    });
    assert.ok(vector.reduce((sum, value, row) => sum + value * applied[row], 0) >= -2e-5);
  }

  encoder = device.createCommandEncoder();
  encodeWithBroker(encoder,
    (broker) => gpu.encodeProjectionFromControl(broker, faces, cpuCSR, pressureBuffer, gpuCounts));
  device.queue.submit([encoder.finish()]); await device.queue.onSubmittedWorkDone(); result = await readSource(gpu.source);
  control = new Uint32Array(slice(result.bytes, result.segments, 0));
  assert.equal(control[0], (OCTREE_POWER_GPU_ASSEMBLED | OCTREE_POWER_GPU_PROJECTED) >>> 0); assert.equal(control[6], cpu.faces.length);
  const projected = [...new Float32Array(slice(result.bytes, result.segments, 4))];
  const divergence = [...new Float32Array(slice(result.bytes, result.segments, 5))];
  const expectedProjected = projectOctreePowerFaceVelocities(cpu, pressure);
  const expectedDivergence = octreePowerDivergence(cpu, expectedProjected);
  expectedProjected.forEach((value, index) => close(projected[index], value, 3e-5));
  expectedDivergence.forEach((value, index) => close(divergence[index], value, 3e-5));
  const committedReadback = device.createBuffer({ size: faceData.byteLength,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  encoder = device.createCommandEncoder(); encoder.copyBufferToBuffer(faces, 0, committedReadback, 0, faceData.byteLength);
  device.queue.submit([encoder.finish()]); await device.queue.onSubmittedWorkDone(); await committedReadback.mapAsync(GPUMapMode.READ);
  const committedFaces = new Float32Array(committedReadback.getMappedRange().slice(0)); committedReadback.unmap(); committedReadback.destroy();
  expectedProjected.forEach((value, index) => close(committedFaces[index * 8 + 4], value, 3e-5));
  assert.ok(Math.max(...octreePowerDivergence(cpu).map(Math.abs)) > 1e-3);
  assert.ok(Math.max(...divergence.slice(0, cpu.rows.length).map(Math.abs)) < 3e-5);

  // Ghost-fluid OPEN_BOUNDARY rows use the bounded dual-edge crossing in
  // both the diagonal and face projection. With p=-u*d_boundary the single
  // liquid cell projects to exactly zero integrated divergence.
  const boundaryDistance = octreePowerBoundaryDistance(-0.25, 0.75, 2);
  const boundaryInverse = 1 / boundaryDistance;
  const freeSurface = new WebGPUOctreePowerOperator(device, 1, 1, 1, 1, coefficientSource);
  const freeFace = upload(packRawFaces([[0, 0xffff_ffff, 1, boundaryInverse, 2]]));
  const freeCSR = uploadCSR([[{ face: 0, sign: 1 }]]);
  encoder = device.createCommandEncoder();
  encodeWithBroker(encoder,
    (broker) => freeSurface.encodeAssembly(broker, freeFace, freeCSR.source, 1, 1, 1));
  device.queue.submit([encoder.finish()]); await device.queue.onSubmittedWorkDone(); result = await readSource(freeSurface.source);
  const freeRows = new Float32Array(slice(result.bytes, result.segments, 1));
  close(freeRows[0], boundaryInverse); close(freeRows[1], 1);
  const freePressure = upload(new Float32Array([-boundaryDistance]));
  encoder = device.createCommandEncoder();
  encodeWithBroker(encoder,
    (broker) => freeSurface.encodeProjection(broker, freeFace, freeCSR.source, freePressure, 1, 1, 1));
  device.queue.submit([encoder.finish()]); await device.queue.onSubmittedWorkDone(); result = await readSource(freeSurface.source);
  close(new Float32Array(slice(result.bytes, result.segments, 4))[0], 0);
  close(new Float32Array(slice(result.bytes, result.segments, 5))[0], 0);

  // Every optional GPU producer is a publication prerequisite. A valid
  // velocity seed cannot mask a failed generalized-solid classification.
  const seedWords = new Uint32Array(8); seedWords[6] = OCTREE_POWER_GPU_ASSEMBLED;
  const solidWords = new Uint32Array(8);
  const seedControl = upload(seedWords, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
  const solidControl = upload(solidWords, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
  encoder = device.createCommandEncoder();
  encodeWithBroker(encoder, (broker) => gpu.encodeAssemblyFromControl(broker, faces, cpuCSR, gpuCounts,
    seedControl, solidControl));
  device.queue.submit([encoder.finish()]); await device.queue.onSubmittedWorkDone(); result = await readSource(gpu.source);
  control = new Uint32Array(slice(result.bytes, result.segments, 0));
  assert.equal(control[0], OCTREE_POWER_GPU_ERROR.invalidState);

  // Invalid pressure clears projected authority and reports the exact row.
  device.queue.writeBuffer(pressureBuffer, 0, new Float32Array([Number.NaN]));
  encoder = device.createCommandEncoder();
  encodeWithBroker(encoder, (broker) => gpu.encodeAssemblyFromControl(broker, faces, cpuCSR, gpuCounts));
  device.queue.submit([encoder.finish()]); await device.queue.onSubmittedWorkDone();
  encoder = device.createCommandEncoder();
  encodeWithBroker(encoder, (broker) => gpu.encodeProjection(broker, faces, cpuCSR, pressureBuffer,
    cpu.rows.length, cpu.faces.length, cpuCSRData.incidenceCount));
  device.queue.submit([encoder.finish()]); await device.queue.onSubmittedWorkDone(); result = await readSource(gpu.source);
  control = new Uint32Array(slice(result.bytes, result.segments, 0));
  assert.deepEqual([...control.slice(0, 2)], [OCTREE_POWER_GPU_ERROR.invalidPressure, 0]); assert.equal(control[6], 0);

  // Duplicate physical records merge into one sorted neighbor entry per row.
  const duplicate = new WebGPUOctreePowerOperator(device, 2, 2, 2, 2, coefficientSource);
  const duplicateFaces = upload(packRawFaces([[0, 1, 2, 1], [0, 1, -1, 2]]));
  const duplicateCSR = uploadCSR([[{ face: 0, sign: 1 }, { face: 1, sign: 1 }], [{ face: 0, sign: -1 }, { face: 1, sign: -1 }]]);
  encoder = device.createCommandEncoder();
  encodeWithBroker(encoder,
    (broker) => duplicate.encodeAssembly(broker, duplicateFaces, duplicateCSR.source, 2, 2, duplicateCSR.count));
  device.queue.submit([encoder.finish()]); await device.queue.onSubmittedWorkDone(); result = await readSource(duplicate.source);
  control = new Uint32Array(slice(result.bytes, result.segments, 0)); assert.equal(control[5], 2);
  const duplicateRows = new Float32Array(slice(result.bytes, result.segments, 1));
  const duplicateEntries = new Float32Array(slice(result.bytes, result.segments, 3));
  close(duplicateRows[0], 3); close(duplicateRows[4], 3); close(duplicateEntries[1], 3); close(duplicateEntries[3], 3);

  // Entry overflow and nonfinite face data never publish ASSEMBLED.
  const overflow = new WebGPUOctreePowerOperator(device, 3, 2, 1, 2, coefficientSource);
  const overflowFaces = upload(packRawFaces([[0, 1, 0, 1], [1, 2, 0, 1]]));
  const overflowCSR = uploadCSR([[{ face: 0, sign: 1 }], [{ face: 0, sign: -1 }, { face: 1, sign: 1 }], [{ face: 1, sign: -1 }]]);
  encoder = device.createCommandEncoder();
  encodeWithBroker(encoder,
    (broker) => overflow.encodeAssembly(broker, overflowFaces, overflowCSR.source, 3, 2, overflowCSR.count));
  device.queue.submit([encoder.finish()]); await device.queue.onSubmittedWorkDone(); result = await readSource(overflow.source);
  control = new Uint32Array(slice(result.bytes, result.segments, 0)); assert.equal(control[0], OCTREE_POWER_GPU_ERROR.entryOverflow);
  const nonfiniteFaces = packRawFaces([[0, 1, 0, 1]]); new Float32Array(nonfiniteFaces.buffer)[5] = Number.NaN;
  const nonfiniteBuffer = upload(nonfiniteFaces);
  const nonfiniteCSR = uploadCSR([[{ face: 0, sign: 1 }], [{ face: 0, sign: -1 }]]);
  encoder = device.createCommandEncoder();
  encodeWithBroker(encoder,
    (broker) => duplicate.encodeAssembly(broker, nonfiniteBuffer, nonfiniteCSR.source, 2, 1, nonfiniteCSR.count));
  device.queue.submit([encoder.finish()]); await device.queue.onSubmittedWorkDone(); result = await readSource(duplicate.source);
  control = new Uint32Array(slice(result.bytes, result.segments, 0)); assert.equal(control[0], OCTREE_POWER_GPU_ERROR.nonfiniteFace);

  // Canonical bulk rows are emitted from the fixed coefficient LUT without a
  // separately materialized volume input. Both endpoints publish the exact
  // same coefficient bits and the control ABI reports two bulk rows.
  const bulkTopology = new WebGPUOctreePowerTopology(device, 2, catalog);
  const regularEntry = catalog.sameOrFinerDirect[OCTREE_POWER_REGULAR_DESCRIPTOR] & 0xffff;
  const normalizedCoefficient = catalog.coefficientData[regularEntry * 19 + 1];
  const physicalCellSize = Math.fround(0.05);
  const expectedCoefficient = Math.fround(Math.fround(Math.fround(physicalCellSize * physicalCellSize)
    * normalizedCoefficient) / physicalCellSize);
  const bulkMetricWords = new Uint32Array(8); const bulkMetricFloats = new Float32Array(bulkMetricWords.buffer);
  for (let row = 0; row < 2; row += 1) {
    bulkMetricWords[row * 4] = regularEntry; bulkMetricWords[row * 4 + 1] = OCTREE_POWER_TOPOLOGY_VALID;
    bulkMetricFloats[row * 4 + 2] = catalog.entryVolumes[regularEntry];
  }
  device.queue.writeBuffer(bulkTopology.metrics, 0, bulkMetricWords);
  const bulkHeaderWords = new Uint32Array(24); bulkHeaderWords[3] = 1; bulkHeaderWords[15] = 1;
  const bulkHeaders = upload(bulkHeaderWords);
  const bulkOperator = new WebGPUOctreePowerOperator(device, 2, 1, 2, 1,
    { topology: bulkTopology.source, leafHeaders: bulkHeaders, physicalCellSize,
      physicalCellVolume: physicalCellSize ** 3 });
  const bulkFaces = upload(packRawFaces([[0, 1, 2, expectedCoefficient]]));
  const bulkCSR = uploadCSR([[{ face: 0, sign: 1 }], [{ face: 0, sign: -1 }]]);
  encoder = device.createCommandEncoder();
  encodeWithBroker(encoder,
    (broker) => bulkOperator.encodeAssembly(broker, bulkFaces, bulkCSR.source, 2, 1, bulkCSR.count));
  device.queue.submit([encoder.finish()]); await device.queue.onSubmittedWorkDone();
  const bulkResult = await readSource(bulkOperator.source);
  const bulkControl = new Uint32Array(slice(bulkResult.bytes, bulkResult.segments, 0));
  assert.equal(bulkControl[0], OCTREE_POWER_GPU_ASSEMBLED); assert.equal(bulkControl[7], 2);
  const bulkRows = new Float32Array(slice(bulkResult.bytes, bulkResult.segments, 1));
  const bulkEntryWords = new Uint32Array(slice(bulkResult.bytes, bulkResult.segments, 3));
  const bulkEntryFloats = new Float32Array(bulkEntryWords.buffer);
  assert.equal(bulkRows[0], expectedCoefficient); assert.equal(bulkRows[4], expectedCoefficient);
  assert.equal(bulkEntryWords[0], 1); assert.equal(bulkEntryWords[2], 0);
  assert.equal(bulkEntryFloats[1], expectedCoefficient); assert.equal(bulkEntryFloats[3], expectedCoefficient);
  assert.ok(Number.isFinite(bulkRows[2]) && bulkRows[2] > 0);

  assert.deepEqual(validationErrors, []);
  gpu.destroy(); duplicate.destroy(); overflow.destroy();
  faces.destroy(); pressureBuffer.destroy(); gpuCounts.destroy(); cpuCSR.incidenceRows.destroy(); cpuCSR.incidence.destroy();
  duplicateFaces.destroy(); duplicateCSR.source.incidenceRows.destroy(); duplicateCSR.source.incidence.destroy();
  overflowFaces.destroy(); overflowCSR.source.incidenceRows.destroy(); overflowCSR.source.incidence.destroy();
  nonfiniteBuffer.destroy(); nonfiniteCSR.source.incidenceRows.destroy(); nonfiniteCSR.source.incidence.destroy();
  seedControl.destroy(); solidControl.destroy();
  freeSurface.destroy(); freeFace.destroy(); freePressure.destroy();
  freeCSR.source.incidenceRows.destroy(); freeCSR.source.incidence.destroy();
  publishedHeaders.destroy(); publishedEntries.destroy();
  coefficientTopology.destroy(); coefficientHeaders.destroy();
  bulkOperator.destroy(); bulkTopology.destroy(); bulkHeaders.destroy(); bulkFaces.destroy();
  bulkCSR.source.incidenceRows.destroy(); bulkCSR.source.incidence.destroy();
  device.destroy();
});
