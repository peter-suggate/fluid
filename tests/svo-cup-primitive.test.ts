import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";

import { cupDistance_m, cupWallThickness_m } from "../lib/core/scene-shape";
import type { Vec3 } from "../lib/core/model";
import {
  SVO_PRIMITIVE_FEATURES,
  canonicalSvoPrimitive,
  intersectSvoPrimitive,
  packSvoPrimitiveRecords,
  sampleSvoPrimitive,
  svoPrimitiveForRigidBody,
  svoPrimitiveWGSL,
  unpackSvoPrimitiveRecords,
  type SvoCupPrimitive,
} from "../lib/svo/svo-primitive-abi";
import { svoFieldProgramAbsentWGSL } from "../lib/svo/svo-primitive-abi";
import { svoProceduralNoiseWGSL } from "../lib/svo/svo-procedural-material";
import {
  sparseScenePrimitiveFeatureRadius,
  sparseScenePrimitiveForSvoDescriptor,
} from "../lib/core/webgpu-sparse-scene-proxies";

/**
 * The cup is the first shape in this repository whose interior is not solid,
 * and a hollow shape is only worth anything if every consumer agrees about the
 * hollow part. Its field is declared once, in `lib/core/scene-shape.ts`; the
 * solvers reach it through `sceneShapeWgsl` and the renderer through the SVO
 * ABI, and this file is what holds the second path to the first.
 *
 * The failure it exists to catch is specific and silent: a cup whose cavity is
 * filled draws exactly like a cup, and only stops being one when it is dipped.
 */

/** Outer radius, HALF height, wall — the render ABI's convention. */
const CUP = {
  primitiveId: 1, materialId: 20, ownerId: 3,
  kind: "cup", center_m: { x: 0, y: 0, z: 0 },
  radius_m: 0.15, halfHeight_m: 0.13, wallThickness_m: 0.02,
} as const satisfies SvoCupPrimitive;

/** The same cup in the shared table's convention, which carries the FULL height. */
const SHARED: Vec3 = { x: CUP.radius_m, y: 2 * CUP.halfHeight_m, z: CUP.wallThickness_m };

/**
 * Whether the hard-feature selection at a point is a contract at all.
 *
 * Two of a cup's three surfaces meet along its rim, along the inner floor's
 * corner, and along the medial surface inside the wall. On those curves both
 * surfaces are equidistant and which one wins is decided by the last bit of a
 * subtraction, so an f32 solve and an f64 one are entitled to disagree. That
 * jump *is* what a hard feature means; the contract is the surface either side
 * of the edge, never the edge itself.
 */
function featureIsStable(point: Vec3): boolean {
  const selected = sampleSvoPrimitive(CUP, point).featureId;
  const step = 1e-4;
  for (const delta of [
    { x: step, y: 0, z: 0 }, { x: -step, y: 0, z: 0 },
    { x: 0, y: step, z: 0 }, { x: 0, y: -step, z: 0 },
    { x: 0, y: 0, z: step }, { x: 0, y: 0, z: -step },
  ]) {
    const probe = { x: point.x + delta.x, y: point.y + delta.y, z: point.z + delta.z };
    if (sampleSvoPrimitive(CUP, probe).featureId !== selected) return false;
  }
  return true;
}

function samplePoints(): Vec3[] {
  const axis = [-0.21, -0.15, -0.13, -0.08, -0.02, 0, 0.02, 0.08, 0.13, 0.15, 0.21];
  const points: Vec3[] = [];
  for (const x of axis) for (const y of axis) for (const z of axis) points.push({ x, y, z });
  return points;
}

test("the render ABI evaluates the shared cup field, not a second copy of it", () => {
  for (const point of samplePoints()) {
    assert.equal(sampleSvoPrimitive(CUP, point).signedDistance_m, cupDistance_m(SHARED, point),
      `at ${JSON.stringify(point)}`);
  }
});

test("a cup's three authored surfaces each get their own normal", () => {
  const wall = cupWallThickness_m(SHARED);
  // Just outside the outer wall, just inside the cavity wall, just above the
  // inner floor. Each point is nearest a different one of the three surfaces.
  const outside = sampleSvoPrimitive(CUP, { x: CUP.radius_m + 0.01, y: 0, z: 0 });
  assert.ok(outside.normal && outside.normal.x > 0.99, "the outer wall faces away from the axis");
  assert.equal(outside.featureId, SVO_PRIMITIVE_FEATURES.cylinderSide);

  const cavityWall = sampleSvoPrimitive(CUP, { x: CUP.radius_m - wall - 0.005, y: 0.05, z: 0 });
  assert.ok(cavityWall.normal && cavityWall.normal.x < -0.99, "the cavity wall faces the axis");
  assert.equal(cavityWall.featureId, SVO_PRIMITIVE_FEATURES.cylinderSide);

  const floor = sampleSvoPrimitive(CUP, { x: 0, y: -CUP.halfHeight_m + wall + 0.005, z: 0 });
  assert.ok(floor.normal && floor.normal.y > 0.99, "the inner floor faces up");
  assert.equal(floor.featureId, SVO_PRIMITIVE_FEATURES.cylinderCap);
});

test("a ray down the axis lands on the inner floor, not on a filled interior", () => {
  const wall = cupWallThickness_m(SHARED);
  const hit = intersectSvoPrimitive(CUP,
    { origin_m: { x: 0, y: 1, z: 0 }, direction: { x: 0, y: -1, z: 0 }, tMin_m: 0, tMax_m: 10 });
  assert.ok(hit, "the axis ray must hit the cup");
  // A solid cylinder would answer at the rim, one wall above where the floor is.
  const floorY = -CUP.halfHeight_m + wall;
  assert.ok(Math.abs(hit.position_m.y - floorY) < 1e-4,
    `expected the floor at y=${floorY}, got y=${hit.position_m.y}`);
  assert.ok(hit.normal && hit.normal.y > 0.99, "the floor faces up");
});

test("a cup survives the record round trip and refuses a wall that would close it", () => {
  // A record holds f32s, so the round trip is exact only up to the arena's own
  // precision — the contract is that nothing else moves.
  const [recovered] = unpackSvoPrimitiveRecords(packSvoPrimitiveRecords([CUP]));
  assert.deepEqual(recovered, canonicalSvoPrimitive({
    ...CUP,
    radius_m: Math.fround(CUP.radius_m),
    halfHeight_m: Math.fround(CUP.halfHeight_m),
    wallThickness_m: Math.fround(CUP.wallThickness_m),
  }));

  // The shared field clamps a wall this thick; the ABI refuses it instead, so a
  // record can never describe a cup other than the one that gets drawn.
  assert.throws(() => canonicalSvoPrimitive({ ...CUP, wallThickness_m: CUP.radius_m }), /open cavity/);
});

test("a cup body publishes as a cup, converting the one height convention", () => {
  const published = svoPrimitiveForRigidBody(
    { shape: "cup", dimensions_m: SHARED, position_m: { x: 1, y: 2, z: 3 }, orientation: { x: 0, y: 0, z: 0, w: 1 } },
    7, 9);
  assert.equal(published.kind, "cup");
  assert.deepEqual(published, canonicalSvoPrimitive({
    ...CUP, primitiveId: 7, ownerId: 9, materialId: published.materialId,
    center_m: { x: 1, y: 2, z: 3 }, orientation: { x: 0, y: 0, z: 0, w: 1 },
  }));
});

test("the live voxelizer knows a cup's wall is its finest feature", () => {
  const live = sparseScenePrimitiveForSvoDescriptor(canonicalSvoPrimitive(CUP));
  assert.equal(live.kind, "cup");
  // Without this the centre sample of a cell straddling the wall reads the
  // cavity, and the cup voxelizes as a ring of holes.
  assert.equal(sparseScenePrimitiveFeatureRadius(live), CUP.wallThickness_m);
});

test("the WGSL cup agrees with the TypeScript cup on a device", {
  skip: !process.env.WEBGPU_NODE_MODULE && "set WEBGPU_NODE_MODULE for cup ABI parity on a device",
}, async () => {
  const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href) as {
    create(options: string[]): GPU; globals: Record<string, unknown>;
  };
  Object.assign(globalThis, dawn.globals);
  const gpu = dawn.create(["backend=metal"]);
  const adapter = await gpu.requestAdapter(); assert.ok(adapter);
  const device = await adapter.requestDevice();

  const points = samplePoints();
  const record = packSvoPrimitiveRecords([CUP]);
  const inputs: Float32Array<ArrayBuffer> = new Float32Array(points.length * 4);
  points.forEach((point, index) => inputs.set([point.x, point.y, point.z, 0], index * 4));

  const code = `${svoProceduralNoiseWGSL}
${svoFieldProgramAbsentWGSL}
${svoPrimitiveWGSL}
@group(0) @binding(0) var<storage, read> recordWords: array<vec4u, 4>;
@group(0) @binding(1) var<storage, read> points: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> answers: array<vec4f>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let index = id.x;
  if (index >= arrayLength(&points)) { return; }
  let record = SvoPrimitiveRecord(recordWords[0], recordWords[1], bitcast<vec4f>(recordWords[2]), recordWords[3]);
  let sample = svoEvaluatePrimitive(record, points[index].xyz, svoInvalidClusterPacking());
  answers[index] = vec4f(sample.signedDistance_m, sample.normal.xyz);
}`;
  device.pushErrorScope("validation");
  const module = device.createShaderModule({ code });
  const compilation = await module.getCompilationInfo();
  assert.deepEqual(compilation.messages.filter((message) => message.type === "error")
    .map((message) => `${message.lineNum}: ${message.message}`), []);

  const recordBuffer = device.createBuffer({ size: record.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(recordBuffer, 0, record);
  const pointBuffer = device.createBuffer({ size: inputs.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(pointBuffer, 0, inputs);
  const answerBytes = points.length * 4 * 4;
  const answerBuffer = device.createBuffer({ size: answerBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const readback = device.createBuffer({ size: answerBytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });

  const pipeline = device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "main" } });
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: recordBuffer } },
      { binding: 1, resource: { buffer: pointBuffer } },
      { binding: 2, resource: { buffer: answerBuffer } },
    ],
  });
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.ceil(points.length / 64));
  pass.end();
  encoder.copyBufferToBuffer(answerBuffer, 0, readback, 0, answerBytes);
  device.queue.submit([encoder.finish()]);
  const scope = await device.popErrorScope();
  assert.equal(scope, null);
  await readback.mapAsync(GPUMapMode.READ);
  const answers = new Float32Array(readback.getMappedRange().slice(0));
  readback.unmap();

  let checked = 0;
  points.forEach((point, index) => {
    const expected = sampleSvoPrimitive(CUP, point);
    assert.ok(Math.abs(answers[index * 4] - expected.signedDistance_m) < 1e-5,
      `distance at ${JSON.stringify(point)}: ${answers[index * 4]} vs ${expected.signedDistance_m}`);
    // A normal is only compared where the host publishes one; the shader's own
    // validity flag is what a renderer reads, and it is derived from the same
    // length test.
    if (expected.normal && featureIsStable(point)) {
      const gpu = { x: answers[index * 4 + 1], y: answers[index * 4 + 2], z: answers[index * 4 + 3] };
      const alignment = gpu.x * expected.normal.x + gpu.y * expected.normal.y + gpu.z * expected.normal.z;
      assert.ok(alignment > 1 - 1e-4, `normal at ${JSON.stringify(point)}: alignment ${alignment}`);
    }
    if (expected.normal && featureIsStable(point)) checked += 1;
  });
  // Most of the lattice is well away from an edge; if the stability filter ever
  // starts excluding most of it, the normals are no longer being compared.
  assert.ok(checked > 0.6 * points.length, `only ${checked} of ${points.length} normals were comparable`);
});
