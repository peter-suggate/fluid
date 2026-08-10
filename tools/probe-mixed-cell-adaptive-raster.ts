/**
 * Focused Dawn oracle for the factor-one renderer's adaptive nodal contract.
 *
 * The synthetic octree contains adjacent span-4, span-2 and span-1 leaves.
 * It publishes only the compact adaptive corner directory consumed by the
 * production classifier; no dense phi texture or authored fallback is bound.
 * A globally representable plane must produce the same zero set on every
 * span.  A curved SDF is constrained at T-junctions exactly as in Losasso et
 * al. and is checked for a continuous, finite, nondegenerate mesh.
 *
 * Usage:
 *   WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js \
 *     node --import tsx tools/run-webgpu-exclusive.ts \
 *     --import tsx tools/probe-mixed-cell-adaptive-raster.ts
 */
import assert from "node:assert/strict";
import { fileURLToPath, pathToFileURL } from "node:url";

import { requiredFluidDeviceLimits } from "../lib/webgpu-device-limits";
import { globalFineSurfaceClassificationShader } from
  "../lib/webgpu-water-global-fine-classify";
import { globalFineClassifiedEmitShader, globalFineClassifiedIndirectScanShader } from
  "../lib/webgpu-water-global-fine-tetra";
import { rasterMeshSymmetryMetrics } from "./raster-mesh-symmetry";

type Point = readonly [number, number, number];
interface Leaf { readonly origin: Point; readonly span: 1 | 2 | 4 }

const dimensions = [8, 8, 8] as const;
const generation = 7;
const cubeCapacity = dimensions[0] * dimensions[1] * dimensions[2];
const vertexCapacity = cubeCapacity * 36;
const INVALID = 0xffff_ffff;
const PUBLISHED = 0x8000_0000;
const ADAPTIVE_CORNERS = 0x1000_0000;

function leavesForMixedX(): Leaf[] {
  const leaves: Leaf[] = [];
  for (let x = 0; x < dimensions[0];) {
    const span = (x < 4 ? 4 : x < 6 ? 2 : 1) as 1 | 2 | 4;
    for (let z = 0; z < dimensions[2]; z += span) {
      for (let y = 0; y < dimensions[1]; y += span) leaves.push({ origin: [x, y, z], span });
    }
    x += span;
  }
  return leaves.sort((a, b) => encodeCell(a.origin) - encodeCell(b.origin) || a.span - b.span);
}

function encodeCell([x, y, z]: Point) {
  return x + dimensions[0] * (y + dimensions[1] * z);
}

function cornerPoint(leaf: Leaf, corner: number): Point {
  return [leaf.origin[0] + (corner & 1 ? leaf.span : 0),
    leaf.origin[1] + (corner & 2 ? leaf.span : 0),
    leaf.origin[2] + (corner & 4 ? leaf.span : 0)];
}

function trilinear(leaf: Leaf, point: Point, source: (point: Point) => number) {
  const fraction = point.map((value, axis) => (value - leaf.origin[axis]!) / leaf.span);
  let result = 0;
  for (let corner = 0; corner < 8; corner += 1) {
    const weight = (corner & 1 ? fraction[0]! : 1 - fraction[0]!)
      * (corner & 2 ? fraction[1]! : 1 - fraction[1]!)
      * (corner & 4 ? fraction[2]! : 1 - fraction[2]!);
    result += weight * source(cornerPoint(leaf, corner));
  }
  return result;
}

function containsClosed(leaf: Leaf, point: Point) {
  return point.every((value, axis) => value >= leaf.origin[axis]!
    && value <= leaf.origin[axis]! + leaf.span);
}

/** Constrain a T-junction to the coarsest incident leaf interpolation. */
function constrainedValue(leaves: readonly Leaf[], point: Point, analytic: (point: Point) => number) {
  let authority: Leaf | undefined;
  for (const leaf of leaves) {
    if (!containsClosed(leaf, point)) continue;
    const isCorner = point.every((value, axis) => value === leaf.origin[axis]
      || value === leaf.origin[axis]! + leaf.span);
    if (isCorner) continue;
    if (!authority || leaf.span > authority.span) authority = leaf;
  }
  return authority ? trilinear(authority, point, analytic) : analytic(point);
}

function makeDirectory(leaves: readonly Leaf[], analytic: (point: Point) => number) {
  const words = new Uint32Array(8 + 16 * leaves.length);
  const floats = new Float32Array(words.buffer);
  words.set([PUBLISHED, generation, leaves.length, 4,
    dimensions[0], dimensions[1], dimensions[2]], 0);
  floats[7] = 1;
  leaves.forEach((leaf, row) => {
    const values = Array.from({ length: 8 }, (_, corner) =>
      Math.fround(constrainedValue(leaves, cornerPoint(leaf, corner), analytic)));
    const base = 8 + 8 * row;
    words[base] = encodeCell(leaf.origin) + 1;
    words[base + 1] = leaf.span;
    floats[base + 2] = values.reduce((sum, value) => sum + value, 0) / 8;
    floats[base + 3] = Math.min(...values);
    floats[base + 4] = Math.max(...values);
    words[base + 5] = ADAPTIVE_CORNERS | 9;
    words[base + 6] = row;
    floats[base + 7] = leaf.span ** 3;
    const aux = 8 + 8 * leaves.length + 8 * row;
    for (let corner = 0; corner < 8; corner += 1) floats[aux + corner] = values[corner]!;
  });
  return words;
}

function scalarContinuityMetrics(leaves: readonly Leaf[], analytic: (point: Point) => number) {
  const samples = new Map<string, number[]>();
  const stored = (point: Point) => Math.fround(constrainedValue(leaves, point, analytic));
  let constrainedNodeCount = 0, maximumConstraintError = 0;
  const nodes = new Map<string, Point>();
  for (const leaf of leaves) for (let corner = 0; corner < 8; corner += 1) {
    const point = cornerPoint(leaf, corner); nodes.set(point.join(","), point);
  }
  for (const point of nodes.values()) {
    const hasCoarseAuthority = leaves.some(leaf => containsClosed(leaf, point)
      && !point.every((value, axis) => value === leaf.origin[axis]
        || value === leaf.origin[axis]! + leaf.span));
    if (!hasCoarseAuthority) continue;
    constrainedNodeCount += 1;
    maximumConstraintError = Math.max(maximumConstraintError,
      Math.abs(stored(point) - Math.fround(constrainedValue(leaves, point, analytic))));
  }
  for (const leaf of leaves) for (let z = 0; z <= leaf.span; z += 1) {
    for (let y = 0; y <= leaf.span; y += 1) for (let x = 0; x <= leaf.span; x += 1) {
      const point = [leaf.origin[0] + x, leaf.origin[1] + y, leaf.origin[2] + z] as const;
      const value = Math.fround(trilinear(leaf, point, stored));
      const row = samples.get(point.join(",")) ?? []; row.push(value); samples.set(point.join(","), row);
    }
  }
  let sharedSampleCount = 0, maximumSharedFaceSampleMismatch = 0;
  for (const values of samples.values()) {
    if (values.length < 2) continue;
    sharedSampleCount += 1;
    maximumSharedFaceSampleMismatch = Math.max(maximumSharedFaceSampleMismatch,
      Math.max(...values) - Math.min(...values));
  }
  return { constrainedNodeCount, maximumConstraintError,
    sharedSampleCount, maximumSharedFaceSampleMismatch };
}

function floatWord(value: number) {
  return new Uint32Array(new Float32Array([Object.is(value, -0) ? 0 : value]).buffer)[0]!;
}

function meshPositionSet(mesh: Float32Array, vertexCount: number) {
  const set = new Set<string>();
  for (let vertex = 0; vertex < vertexCount; vertex += 1) {
    set.add([0, 1, 2].map(axis => floatWord(mesh[8 * vertex + axis]!)
      .toString(16).padStart(8, "0")).join(":"));
  }
  return set;
}

function meshGeometryMetrics(mesh: Float32Array, vertexCount: number,
  analytic: (point: Point) => number) {
  const positions = Array.from({ length: vertexCount }, (_, vertex) => {
    const x = 8 * mesh[8 * vertex]! + 4;
    const y = 8 * mesh[8 * vertex + 1]!;
    const z = 8 * mesh[8 * vertex + 2]! + 4;
    return [x, y, z] as const;
  });
  const unique = new Map<string, Point>();
  positions.forEach(point => unique.set(point.map(floatWord).join(":"), point));
  const bounds = { minimum: [Infinity, Infinity, Infinity],
    maximum: [-Infinity, -Infinity, -Infinity] } as { minimum: number[]; maximum: number[] };
  const deviationBySpan: Record<string, { vertexCount: number; maximumAbsolutePhi: number }> = {};
  for (const point of unique.values()) {
    for (let axis = 0; axis < 3; axis += 1) {
      bounds.minimum[axis] = Math.min(bounds.minimum[axis]!, point[axis]!);
      bounds.maximum[axis] = Math.max(bounds.maximum[axis]!, point[axis]!);
    }
    const span = point[0] < 4 ? 4 : point[0] < 6 ? 2 : 1;
    const group = deviationBySpan[span] ?? { vertexCount: 0, maximumAbsolutePhi: 0 };
    group.vertexCount += 1;
    group.maximumAbsolutePhi = Math.max(group.maximumAbsolutePhi, Math.abs(analytic(point)));
    deviationBySpan[span] = group;
  }
  const key = (point: Point) => point.map(value => floatWord(value).toString(16)).join(":");
  const edge = (a: string, b: string) => a < b ? `${a}|${b}` : `${b}|${a}`;
  const edgeCounts = new Map<string, { count: number; a: Point; b: Point }>();
  for (let triangle = 0; triangle + 2 < vertexCount; triangle += 3) {
    const points = [positions[triangle]!, positions[triangle + 1]!, positions[triangle + 2]!] as const;
    for (const [a, b] of [[0, 1], [1, 2], [2, 0]] as const) {
      const id = edge(key(points[a]), key(points[b]));
      const old = edgeCounts.get(id);
      edgeCounts.set(id, { count: (old?.count ?? 0) + 1, a: points[a], b: points[b] });
    }
  }
  let boundaryOpenEdgeCount = 0, internalOpenEdgeCount = 0;
  for (const record of edgeCounts.values()) if (record.count === 1) {
    const onBoundary = [0, 1, 2].some(axis => {
      const boundary = dimensions[axis];
      return (Math.abs(record.a[axis]!) < 1e-5 && Math.abs(record.b[axis]!) < 1e-5)
        || (Math.abs(record.a[axis]! - boundary) < 1e-5
          && Math.abs(record.b[axis]! - boundary) < 1e-5);
    });
    if (onBoundary) boundaryOpenEdgeCount += 1; else internalOpenEdgeCount += 1;
  }
  return { uniqueVertexCount: unique.size, bounds,
    zeroSetDeviationBySpan: deviationBySpan, boundaryOpenEdgeCount, internalOpenEdgeCount };
}

function buffer(device: GPUDevice, label: string, size: number, usage: GPUBufferUsageFlags,
  contents?: ArrayBufferView) {
  const result = device.createBuffer({ label, size: Math.max(4, Math.ceil(size / 4) * 4),
    usage, mappedAtCreation: contents !== undefined });
  if (contents) {
    const bytes = new Uint8Array(contents.buffer, contents.byteOffset, contents.byteLength);
    new Uint8Array(result.getMappedRange()).set(bytes); result.unmap();
  }
  return result;
}

async function read(device: GPUDevice, source: GPUBuffer, bytes: number) {
  const target = device.createBuffer({ label: "mixed raster readback", size: bytes,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(source, 0, target, 0, bytes);
  device.queue.submit([encoder.finish()]);
  await target.mapAsync(GPUMapMode.READ);
  const copy = target.getMappedRange().slice(0); target.unmap(); target.destroy();
  return copy;
}

async function runField(device: GPUDevice, name: string, leaves: readonly Leaf[],
  analytic: (point: Point) => number) {
  const directoryWords = makeDirectory(leaves, analytic);
  const uniformWords = new Uint32Array(28);
  const uniformFloats = new Float32Array(uniformWords.buffer);
  uniformFloats.set([1, 1, 1, 1], 12); // container
  const paramsWords = new Uint32Array(28);
  const paramsFloats = new Float32Array(paramsWords.buffer);
  paramsWords.set([dimensions[0], dimensions[1], dimensions[2], 1], 0);
  paramsWords.set([1, 1, 1, 1], 4);
  paramsWords.set([cubeCapacity, 6, cubeCapacity, generation], 8);
  paramsFloats.set([0, 0, 0, 1], 12);
  paramsFloats.set([1, 0, 0, 0], 16);

  const uniforms = buffer(device, `${name} uniforms`, uniformWords.byteLength,
    GPUBufferUsage.UNIFORM, uniformWords);
  const params = buffer(device, `${name} params`, paramsWords.byteLength,
    GPUBufferUsage.UNIFORM, paramsWords);
  const argsInitial = Uint32Array.from([0, 1, 0, 0, 0, INVALID, 0, INVALID]);
  const args = buffer(device, `${name} args`, argsInitial.byteLength,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC, argsInitial);
  const cubes = buffer(device, `${name} cubes`, cubeCapacity * 8,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
  const values = buffer(device, `${name} values`, cubeCapacity * 32, GPUBufferUsage.STORAGE);
  const offsets = buffer(device, `${name} offsets`, cubeCapacity * 24, GPUBufferUsage.STORAGE);
  const vertices = buffer(device, `${name} vertices`, vertexCapacity * 32,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
  const empty = buffer(device, `${name} empty`, 256,
    GPUBufferUsage.STORAGE, new Uint32Array(64));
  const directory = buffer(device, `${name} directory`, directoryWords.byteLength,
    GPUBufferUsage.STORAGE, directoryWords);

  const classify = device.createComputePipeline({ label: `${name} production classify`,
    layout: "auto", compute: { module: device.createShaderModule({ code: globalFineSurfaceClassificationShader }),
      entryPoint: "extractGlobalCoarseMain" } });
  const scan = device.createComputePipeline({ label: `${name} production scan`, layout: "auto",
    compute: { module: device.createShaderModule({ code: globalFineClassifiedIndirectScanShader }),
      entryPoint: "scanGlobalFineTriangles" } });
  const emit = device.createComputePipeline({ label: `${name} production emit`, layout: "auto",
    compute: { module: device.createShaderModule({ code: globalFineClassifiedEmitShader }),
      entryPoint: "emitGlobalFineTetrahedra" } });
  const classifyGroup = device.createBindGroup({ layout: classify.getBindGroupLayout(0), entries: [
    { binding: 0, resource: { buffer: uniforms } }, { binding: 4, resource: { buffer: args } },
    { binding: 5, resource: { buffer: cubes } }, { binding: 6, resource: { buffer: values } },
    { binding: 8, resource: { buffer: empty } }, { binding: 9, resource: { buffer: empty } },
    { binding: 10, resource: { buffer: params } }, { binding: 12, resource: { buffer: empty } },
    { binding: 16, resource: { buffer: directory } }, { binding: 17, resource: { buffer: empty } },
  ] });
  const scanGroup = device.createBindGroup({ layout: scan.getBindGroupLayout(0), entries: [
    { binding: 3, resource: { buffer: vertices } }, { binding: 4, resource: { buffer: args } },
    { binding: 5, resource: { buffer: cubes } }, { binding: 6, resource: { buffer: values } },
    { binding: 7, resource: { buffer: offsets } }, { binding: 10, resource: { buffer: params } },
    { binding: 11, resource: { buffer: empty } },
  ] });
  const emitGroup = device.createBindGroup({ layout: emit.getBindGroupLayout(0), entries: [
    { binding: 0, resource: { buffer: uniforms } }, { binding: 3, resource: { buffer: vertices } },
    { binding: 4, resource: { buffer: args } }, { binding: 5, resource: { buffer: cubes } },
    { binding: 6, resource: { buffer: values } }, { binding: 7, resource: { buffer: offsets } },
    { binding: 8, resource: { buffer: empty } }, { binding: 9, resource: { buffer: empty } },
    { binding: 10, resource: { buffer: params } }, { binding: 12, resource: { buffer: empty } },
    { binding: 16, resource: { buffer: directory } },
  ] });
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(classify); pass.setBindGroup(0, classifyGroup);
  pass.dispatchWorkgroups(Math.ceil(cubeCapacity / 256));
  pass.setPipeline(scan); pass.setBindGroup(0, scanGroup); pass.dispatchWorkgroups(1);
  pass.setPipeline(emit); pass.setBindGroup(0, emitGroup);
  pass.dispatchWorkgroups(Math.ceil(cubeCapacity / 64), 6, 1);
  pass.end(); device.queue.submit([encoder.finish()]); await device.queue.onSubmittedWorkDone();

  const argsOut = new Uint32Array(await read(device, args, argsInitial.byteLength));
  const vertexCount = argsOut[0]!;
  assert.ok(vertexCount <= vertexCapacity, `${name}: vertex allocation overflow`);
  const mesh = new Float32Array(await read(device, vertices, Math.max(4, vertexCount * 32)));
  const metrics = rasterMeshSymmetryMetrics(mesh, vertexCount);
  const leafSpans = Object.fromEntries([1, 2, 4].map(span =>
    [span, leaves.filter(leaf => leaf.span === span).length]));
  const cubeWords = new Uint32Array(await read(device, cubes, cubeCapacity * 8));
  const cubeIdentities = new Set<string>();
  for (let cube = 0; cube < argsOut[4]!; cube += 1) {
    cubeIdentities.add(`${cubeWords[2 * cube]}:${cubeWords[2 * cube + 1]}`);
  }
  const result = { name, leafCount: leaves.length, leafSpans,
    scalar: scalarContinuityMetrics(leaves, analytic), activeCubeCount: argsOut[4],
    duplicateClassifiedCubeCount: argsOut[4]! - cubeIdentities.size,
    vertexCount, mesh: metrics, geometry: meshGeometryMetrics(mesh, vertexCount, analytic),
    positionSet: meshPositionSet(mesh, vertexCount) };
  for (const resource of [uniforms, params, args, cubes, values, offsets, vertices, empty, directory]) {
    resource.destroy();
  }
  return result;
}

const modulePath = process.env.WEBGPU_NODE_MODULE
  ?? fileURLToPath(new URL("../node_modules/webgpu/index.js", import.meta.url));
const { create, globals } = await import(pathToFileURL(modulePath).href) as {
  create(options: string[]): GPU; globals: Record<string, unknown>;
};
Object.assign(globalThis, globals);
const gpu = create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
Object.defineProperty(globalThis, "navigator", { configurable: true, value: { gpu } });
const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
assert.ok(adapter, "WebGPU did not expose an adapter");
const device = await adapter.requestDevice({ requiredLimits: requiredFluidDeviceLimits(adapter.limits) });
const validationErrors: string[] = [];
device.addEventListener("uncapturederror", event => validationErrors.push(event.error.message));

const mixedLeaves = leavesForMixedX();
const unitLeaves = Array.from({ length: dimensions[0] * dimensions[1] * dimensions[2] }, (_, cell) => ({
  origin: [cell % dimensions[0], Math.floor(cell / dimensions[0]) % dimensions[1],
    Math.floor(cell / (dimensions[0] * dimensions[1]))] as Point, span: 1 as const,
}));
const planePhi = ([x, y, z]: Point) => x + 0.375 * y - 0.25 * z - 4.125;
const spherePhi = ([x, y, z]: Point) => Math.hypot(x - 4, y - 4, z - 4) - 2.75;
const plane = await runField(device, "mixed-plane", mixedLeaves, planePhi);
const unitPlane = await runField(device, "unit-plane", unitLeaves, planePhi);
const sphere = await runField(device, "mixed-sphere", mixedLeaves, spherePhi);
let planePositionDifference = 0;
for (const key of new Set([...plane.positionSet, ...unitPlane.positionSet])) {
  planePositionDifference += Number(plane.positionSet.has(key) !== unitPlane.positionSet.has(key));
}
await device.queue.onSubmittedWorkDone(); device.destroy();
console.log(JSON.stringify({ phase: "mixed-cell-adaptive-raster", dimensions,
  validationErrors, exactPlanePositionSetDifference: planePositionDifference / 2,
  fields: [plane, unitPlane, sphere].map(({ positionSet: _positionSet, ...field }) => field) }, null, 2));
assert.deepEqual(validationErrors, []);
assert.equal(plane.mesh.nonFiniteCount, 0);
assert.equal(plane.mesh.degenerateTriangleCount, 0);
assert.equal(plane.geometry.internalOpenEdgeCount, 0);
assert.equal(planePositionDifference, 0, "representable plane must be identical across leaf spans");
assert.equal(sphere.mesh.nonFiniteCount, 0);
assert.equal(sphere.mesh.degenerateTriangleCount, 0);
assert.equal(sphere.mesh.openEdgeCount, 0, "curved mixed-cell surface must be watertight");
assert.equal(sphere.mesh.nonManifoldEdgeCount, 0, "curved mixed-cell surface must be manifold");
