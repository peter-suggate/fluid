import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
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

const close = (actual: number, expected: number, tolerance = 2e-5) => assert.ok(Math.abs(actual - expected) <= tolerance,
  `expected ${actual} to be within ${tolerance} of ${expected}`);

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
    words[offset + 2] = 0; words[offset + 3] = face.positiveRow === 0xffff_ffff ? 0 : 1;
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
  assert.equal(plan.arenaBytes, 9_136);
  assert.equal(plan.allocatedBytes, 9_296);
  assert.throws(() => planOctreePowerGPUOperator(1, 1, 0, 1), /positive integer/);
});

test("GPU power shader codifies one shared coefficient and fail-closed publication", () => {
  assert.match(octreePowerOperatorShader, /merged\.openFraction\*merged\.area\*merged\.inverseDistance/);
  assert.match(octreePowerOperatorShader, /\(positive-negative\)\*face\.inverseDistance/);
  assert.match(octreePowerOperatorShader, /let value=integrated\/volume/);
  assert.match(octreePowerOperatorShader, /item\.face<=previousFace/);
  assert.match(octreePowerOperatorShader, /atomicStore\(&control\.flags,ASSEMBLED\)/);
  assert.match(octreePowerOperatorShader, /atomicLoad\(&control\.flags\)!=ASSEMBLED/);
  assert.match(octreePowerOperatorShader, /header\.entryStart=start;header\.entryCount=finish-start/);
});

test("Dawn assembles symmetric compact rows and projects the same generalized faces", {
  skip: !process.env.WEBGPU_NODE_MODULE && "set WEBGPU_NODE_MODULE for GPU power-operator checks",
}, async () => {
  const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href) as { create(options: string[]): GPU; globals: Record<string, unknown> };
  Object.assign(globalThis, dawn.globals);
  const nativeGpu = dawn.create(["backend=metal"]);
  const adapter = await nativeGpu.requestAdapter(); assert.ok(adapter);
  const device = await adapter.requestDevice();
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
  const gpu = new WebGPUOctreePowerOperator(device, cpu.rows.length, cpu.faces.length, entryCount, cpu.maximumIncidence);
  const upload = (data: ArrayBufferView, usage = GPUBufferUsage.STORAGE) => {
    const buffer = device.createBuffer({ size: Math.max(4, data.byteLength), usage: usage | GPUBufferUsage.COPY_DST, mappedAtCreation: true });
    new Uint8Array(buffer.getMappedRange(), 0, data.byteLength).set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
    buffer.unmap(); return buffer;
  };
  const uploadCSR = (rows: readonly (readonly { face: number; sign: number }[])[]) => {
    const packed = packCSR(rows);
    return { source: { incidenceRows: upload(packed.rowWords), incidence: upload(packed.incidenceWords) }, count: packed.incidenceCount };
  };
  const faceData = packFaces(cpu.faces);
  const faces = upload(faceData, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
  const volumes = upload(new Float32Array(cpu.rows.map((row) => row.volume)));
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

  let encoder = device.createCommandEncoder(); gpu.encodeAssemblyFromControl(encoder, faces, cpuCSR, volumes, gpuCounts);
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
  encoder = device.createCommandEncoder(); gpu.encodeLeafRowPublication(encoder, publishedHeaders, publishedEntries);
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
    assert.equal(headersOut[rowIndex * 12 + 6], 0); assert.equal(headersOut[rowIndex * 12 + 7], 0);
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

  encoder = device.createCommandEncoder(); gpu.encodeProjectionFromControl(encoder, faces, cpuCSR, pressureBuffer, gpuCounts);
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
  const freeSurface = new WebGPUOctreePowerOperator(device, 1, 1, 1, 1);
  const freeFace = upload(packRawFaces([[0, 0xffff_ffff, 1, boundaryInverse, 2]]));
  const freeVolume = upload(new Float32Array([1]));
  const freeCSR = uploadCSR([[{ face: 0, sign: 1 }]]);
  encoder = device.createCommandEncoder(); freeSurface.encodeAssembly(encoder, freeFace, freeCSR.source, freeVolume, 1, 1, 1);
  device.queue.submit([encoder.finish()]); await device.queue.onSubmittedWorkDone(); result = await readSource(freeSurface.source);
  const freeRows = new Float32Array(slice(result.bytes, result.segments, 1));
  close(freeRows[0], boundaryInverse); close(freeRows[1], 1);
  const freePressure = upload(new Float32Array([-boundaryDistance]));
  encoder = device.createCommandEncoder(); freeSurface.encodeProjection(encoder, freeFace, freeCSR.source, freePressure, 1, 1, 1);
  device.queue.submit([encoder.finish()]); await device.queue.onSubmittedWorkDone(); result = await readSource(freeSurface.source);
  close(new Float32Array(slice(result.bytes, result.segments, 4))[0], 0);
  close(new Float32Array(slice(result.bytes, result.segments, 5))[0], 0);

  // Every optional GPU producer is a publication prerequisite. A valid
  // velocity seed cannot mask a failed generalized-solid classification.
  const seedWords = new Uint32Array(8); seedWords[6] = OCTREE_POWER_GPU_ASSEMBLED;
  const solidWords = new Uint32Array(8);
  const seedControl = upload(seedWords, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
  const solidControl = upload(solidWords, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
  encoder = device.createCommandEncoder(); gpu.encodeAssemblyFromControl(encoder, faces, cpuCSR, volumes, gpuCounts,
    seedControl, solidControl);
  device.queue.submit([encoder.finish()]); await device.queue.onSubmittedWorkDone(); result = await readSource(gpu.source);
  control = new Uint32Array(slice(result.bytes, result.segments, 0));
  assert.equal(control[0], OCTREE_POWER_GPU_ERROR.invalidState);

  // Invalid pressure clears projected authority and reports the exact row.
  device.queue.writeBuffer(pressureBuffer, 0, new Float32Array([Number.NaN]));
  encoder = device.createCommandEncoder(); gpu.encodeAssemblyFromControl(encoder, faces, cpuCSR, volumes, gpuCounts);
  device.queue.submit([encoder.finish()]); await device.queue.onSubmittedWorkDone();
  encoder = device.createCommandEncoder(); gpu.encodeProjection(encoder, faces, cpuCSR, pressureBuffer,
    cpu.rows.length, cpu.faces.length, cpuCSRData.incidenceCount);
  device.queue.submit([encoder.finish()]); await device.queue.onSubmittedWorkDone(); result = await readSource(gpu.source);
  control = new Uint32Array(slice(result.bytes, result.segments, 0));
  assert.deepEqual([...control.slice(0, 2)], [OCTREE_POWER_GPU_ERROR.invalidPressure, 0]); assert.equal(control[6], 0);

  // Duplicate physical records merge into one sorted neighbor entry per row.
  const duplicate = new WebGPUOctreePowerOperator(device, 2, 2, 2, 2);
  const duplicateFaces = upload(packRawFaces([[0, 1, 2, 1], [0, 1, -1, 2]]));
  const duplicateVolumes = upload(new Float32Array([1, 1]));
  const duplicateCSR = uploadCSR([[{ face: 0, sign: 1 }, { face: 1, sign: 1 }], [{ face: 0, sign: -1 }, { face: 1, sign: -1 }]]);
  encoder = device.createCommandEncoder(); duplicate.encodeAssembly(encoder, duplicateFaces, duplicateCSR.source, duplicateVolumes, 2, 2, duplicateCSR.count);
  device.queue.submit([encoder.finish()]); await device.queue.onSubmittedWorkDone(); result = await readSource(duplicate.source);
  control = new Uint32Array(slice(result.bytes, result.segments, 0)); assert.equal(control[5], 2);
  const duplicateRows = new Float32Array(slice(result.bytes, result.segments, 1));
  const duplicateEntries = new Float32Array(slice(result.bytes, result.segments, 3));
  close(duplicateRows[0], 3); close(duplicateRows[4], 3); close(duplicateEntries[1], 3); close(duplicateEntries[3], 3);

  // Entry overflow and nonfinite face data never publish ASSEMBLED.
  const overflow = new WebGPUOctreePowerOperator(device, 3, 2, 1, 2);
  const overflowFaces = upload(packRawFaces([[0, 1, 0, 1], [1, 2, 0, 1]]));
  const overflowVolumes = upload(new Float32Array([1, 1, 1]));
  const overflowCSR = uploadCSR([[{ face: 0, sign: 1 }], [{ face: 0, sign: -1 }, { face: 1, sign: 1 }], [{ face: 1, sign: -1 }]]);
  encoder = device.createCommandEncoder(); overflow.encodeAssembly(encoder, overflowFaces, overflowCSR.source, overflowVolumes, 3, 2, overflowCSR.count);
  device.queue.submit([encoder.finish()]); await device.queue.onSubmittedWorkDone(); result = await readSource(overflow.source);
  control = new Uint32Array(slice(result.bytes, result.segments, 0)); assert.equal(control[0], OCTREE_POWER_GPU_ERROR.entryOverflow);
  const nonfiniteFaces = packRawFaces([[0, 1, 0, 1]]); new Float32Array(nonfiniteFaces.buffer)[5] = Number.NaN;
  const nonfiniteBuffer = upload(nonfiniteFaces);
  const nonfiniteCSR = uploadCSR([[{ face: 0, sign: 1 }], [{ face: 0, sign: -1 }]]);
  encoder = device.createCommandEncoder(); duplicate.encodeAssembly(encoder, nonfiniteBuffer, nonfiniteCSR.source, duplicateVolumes, 2, 1, nonfiniteCSR.count);
  device.queue.submit([encoder.finish()]); await device.queue.onSubmittedWorkDone(); result = await readSource(duplicate.source);
  control = new Uint32Array(slice(result.bytes, result.segments, 0)); assert.equal(control[0], OCTREE_POWER_GPU_ERROR.nonfiniteFace);

  assert.deepEqual(validationErrors, []);
  gpu.destroy(); duplicate.destroy(); overflow.destroy();
  faces.destroy(); volumes.destroy(); pressureBuffer.destroy(); gpuCounts.destroy(); cpuCSR.incidenceRows.destroy(); cpuCSR.incidence.destroy();
  duplicateFaces.destroy(); duplicateVolumes.destroy(); duplicateCSR.source.incidenceRows.destroy(); duplicateCSR.source.incidence.destroy();
  overflowFaces.destroy(); overflowVolumes.destroy(); overflowCSR.source.incidenceRows.destroy(); overflowCSR.source.incidence.destroy();
  nonfiniteBuffer.destroy(); nonfiniteCSR.source.incidenceRows.destroy(); nonfiniteCSR.source.incidence.destroy();
  seedControl.destroy(); solidControl.destroy();
  freeSurface.destroy(); freeFace.destroy(); freeVolume.destroy(); freePressure.destroy();
  freeCSR.source.incidenceRows.destroy(); freeCSR.source.incidence.destroy();
  publishedHeaders.destroy(); publishedEntries.destroy();
  device.destroy();
});
