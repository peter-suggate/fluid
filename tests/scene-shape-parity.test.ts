import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";

import type { Vec3 } from "../lib/core/model";
import {
  SCENE_SHAPES_BY_CODE,
  SCENE_SHAPE_TABLE,
  cupInnerRadius_m,
  cupResolutionAdvice,
  cupWallThickness_m,
  sceneShape,
  sceneShapeWgsl,
} from "../lib/core/scene-shape";

/**
 * The contract that makes one shape table safe.
 *
 * `SCENE_SHAPE_TABLE` states every shape fact twice — once in TypeScript for
 * the host and once in WGSL for the solvers — and the whole argument for
 * collapsing five hand-copied ladders into it is that the two halves cannot
 * drift. Nothing enforces that except this file: it evaluates both over the
 * same lattice and holds them to 1e-5, which is comfortably tighter than the
 * f32 the shader is computing in and looser than nothing.
 *
 * The CPU-only half runs everywhere. The parity half needs a device, so it
 * skips without `WEBGPU_NODE_MODULE` exactly as every other GPU test here does.
 */

/** A lattice of query points, deliberately including the axis and the exact surface. */
function samplePoints(): Vec3[] {
  const axis = [-0.31, -0.18, -0.05, 0, 0.05, 0.09, 0.18, 0.31];
  const points: Vec3[] = [];
  for (const x of axis) for (const y of axis) for (const z of axis) points.push({ x, y, z });
  return points;
}

const DIMENSIONS: readonly Vec3[] = [
  { x: 0.09, y: 0.14, z: 0.11 },
  { x: 0.15, y: 0.26, z: 0.05 },
  { x: 0.2, y: 0.08, z: 0.2 },
  // A wall thicker than the cup, which the clamp has to absorb identically on
  // both sides: an authored cup can be dragged into this by its own gizmo.
  { x: 0.06, y: 0.10, z: 0.4 },
];

const DIRECTIONS: readonly Vec3[] = [
  { x: 1, y: 0, z: 0 },
  { x: 0, y: 1, z: 0 },
  { x: 0, y: 0, z: 1 },
  { x: 0.5773502691896258, y: 0.5773502691896258, z: 0.5773502691896258 },
  { x: -0.8017837257372732, y: 0.2672612419124244, z: -0.5345224838248488 },
];

test("every shape declares a distance whose sign agrees with its inside test", () => {
  for (const shape of SCENE_SHAPES_BY_CODE) {
    for (const d of DIMENSIONS) {
      for (const point of samplePoints()) {
        const distance = shape.distance_m(d, point);
        // The surface itself is the one place a strict/non-strict split is
        // legitimate, so only points clearly off it are asserted.
        if (Math.abs(distance) < 1e-9) continue;
        assert.equal(shape.inside(d, point), distance < 0,
          `${shape.name} disagrees at ${JSON.stringify(point)} for ${JSON.stringify(d)}: distance ${distance}`);
      }
    }
  }
});

test("every shape's distance is 1-Lipschitz, which is what makes a sphere trace safe", () => {
  const points = samplePoints();
  for (const shape of SCENE_SHAPES_BY_CODE) {
    for (const d of DIMENSIONS) {
      for (let index = 1; index < points.length; index += 7) {
        const a = points[index - 1], b = points[index];
        const separation = Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z);
        const change = Math.abs(shape.distance_m(d, a) - shape.distance_m(d, b));
        assert.ok(change <= separation + 1e-9,
          `${shape.name} moves ${change} over ${separation}, so a march can tunnel through it`);
      }
    }
  }
});

test("every shape fits inside the bounding radius it declares", () => {
  for (const shape of SCENE_SHAPES_BY_CODE) {
    for (const d of DIMENSIONS) {
      const radius = shape.boundingRadius_m(d);
      for (const point of samplePoints()) {
        if (!shape.inside(d, point)) continue;
        assert.ok(Math.hypot(point.x, point.y, point.z) <= radius + 1e-9,
          `${shape.name} contains a point outside its own bounding radius ${radius}`);
      }
      for (const direction of DIRECTIONS) {
        assert.ok(shape.supportRadius_m(d, direction) <= radius + 1e-9,
          `${shape.name} supports past its own bounding radius along ${JSON.stringify(direction)}`);
      }
    }
  }
});

test("a cup's cavity is empty, its wall is solid, and its rim is open", () => {
  const d = { x: 0.15, y: 0.26, z: 0.05 };
  const cup = sceneShape("cup");
  const wall = cupWallThickness_m(d);
  const inner = cupInnerRadius_m(d);
  // Dead centre of the cavity: empty, and by more than a wall.
  assert.ok(!cup.inside(d, { x: 0, y: 0, z: 0 }));
  assert.ok(cup.distance_m(d, { x: 0, y: 0, z: 0 }) > 0);
  // Mid-wall, halfway up: solid.
  assert.ok(cup.inside(d, { x: inner + 0.5 * wall, y: 0, z: 0 }));
  // The base, under the cavity floor: solid, which is what holds the water in.
  assert.ok(cup.inside(d, { x: 0, y: -0.5 * d.y + 0.5 * wall, z: 0 }));
  // Above the rim on the axis: open, so water can be poured in and out.
  assert.ok(!cup.inside(d, { x: 0, y: 0.5 * d.y + 0.01, z: 0 }));
  // The shell displaces less than the cylinder around it, and more than nothing.
  const outer = Math.PI * d.x * d.x * d.y;
  assert.ok(cup.volume_m3(d) < outer && cup.volume_m3(d) > 0.1 * outer);
});

test("a cup's inertia sits between the solid cylinder and the thin shell", () => {
  const d = { x: 0.15, y: 0.26, z: 0.05 };
  const density = 600;
  const inertia = SCENE_SHAPE_TABLE.cup.inertia_kg_m2(d, density);
  assert.ok(inertia.x > 0 && inertia.y > 0 && inertia.z > 0);
  // Rotationally symmetric about its own axis, which is the one property a
  // wrong parallel-axis term breaks silently.
  assert.equal(inertia.x, inertia.z);
  // Hollowing moves mass outward, so a cup resists spinning about its axis
  // more per kilogram than the solid cylinder around it and less than a shell
  // with every gram at the rim.
  const mass = density * SCENE_SHAPE_TABLE.cup.volume_m3(d);
  assert.ok(inertia.y / mass > 0.5 * d.x ** 2, "a cup should be harder to spin than a solid cylinder of its radius");
  assert.ok(inertia.y / mass < d.x ** 2, "and easier than a shell with all of its mass at the rim");
});

test("the resolution advice names the cell size a cup needs to hold water", () => {
  const d = { x: 0.15, y: 0.26, z: 0.05 };
  const coarse = cupResolutionAdvice(d, 0.05);
  assert.equal(coarse.holdsWater, false, "a one-cell wall leaks and the advice has to say so");
  const fine = cupResolutionAdvice(d, coarse.recommendedCellSize_m);
  assert.equal(fine.holdsWater, true, "the recommendation has to be a cell size that works");
});

test("the emitted WGSL states every shape exactly once, with a neutral fallback", () => {
  const wgsl = sceneShapeWgsl();
  for (const shape of SCENE_SHAPES_BY_CODE) {
    assert.ok(wgsl.includes(`const RIGID_SHAPE_${shape.name.toUpperCase()}: i32 = ${shape.code};`),
      `${shape.name} has no emitted tag`);
  }
  // Five ladders, each ending in a neutral rather than in its last shape's arm:
  // an unknown tag must read as absent, not as whichever shape happens to be
  // written last.
  assert.equal(wgsl.split("return 1e20;").length - 1, 1);
  assert.equal(wgsl.split("return false;").length - 1, 1);
});

test("the WGSL half of every shape fact agrees with the TypeScript half", {
  skip: !process.env.WEBGPU_NODE_MODULE && "set WEBGPU_NODE_MODULE for shape-table parity on a device",
}, async () => {
  const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href) as {
    create(options: string[]): GPU; globals: Record<string, unknown>;
  };
  Object.assign(globalThis, dawn.globals);
  const gpu = dawn.create(["backend=metal"]);
  const adapter = await gpu.requestAdapter(); assert.ok(adapter);
  const device = await adapter.requestDevice();

  const points = samplePoints();
  const queries: { shape: number; d: Vec3; point: Vec3; direction: Vec3 }[] = [];
  for (const shape of SCENE_SHAPES_BY_CODE) {
    for (const d of DIMENSIONS) {
      for (let index = 0; index < points.length; index += 1) {
        queries.push({ shape: shape.code, d, point: points[index], direction: DIRECTIONS[index % DIRECTIONS.length] });
      }
    }
  }

  // One f32 lane per fact, so a mismatch names which fact drifted rather than
  // only that something did.
  const inputStride = 12, outputStride = 8;
  const input: Float32Array<ArrayBuffer> = new Float32Array(queries.length * inputStride);
  queries.forEach((query, index) => {
    const offset = index * inputStride;
    input.set([query.shape, 0, 0, 0], offset);
    input.set([query.d.x, query.d.y, query.d.z, 0], offset + 4);
    input.set([query.point.x, query.point.y, query.point.z, 0], offset + 8);
  });
  const directions: Float32Array<ArrayBuffer> = new Float32Array(queries.length * 4);
  queries.forEach((query, index) => directions.set([query.direction.x, query.direction.y, query.direction.z, 0], index * 4));

  const code = `${sceneShapeWgsl()}
struct Query { tag: vec4f, dimensions: vec4f, point: vec4f }
struct Answer { distance: f32, inside: f32, support: f32, bounding: f32, volume: f32, pad0: f32, pad1: f32, pad2: f32 }
@group(0) @binding(0) var<storage, read> queries: array<Query>;
@group(0) @binding(1) var<storage, read> directions: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> answers: array<Answer>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let index = id.x;
  if (index >= arrayLength(&queries)) { return; }
  let query = queries[index];
  let shape = i32(round(query.tag.x));
  let d = query.dimensions.xyz;
  let p = query.point.xyz;
  answers[index] = Answer(
    rigidShapeDistance(shape, d, p),
    select(0.0, 1.0, rigidShapeInside(shape, d, p)),
    rigidShapeSupportRadius(shape, d, directions[index].xyz),
    rigidShapeBoundingRadius(shape, d),
    rigidShapeVolume(shape, d),
    0.0, 0.0, 0.0);
}`;
  device.pushErrorScope("validation");
  const module = device.createShaderModule({ code });
  const compilation = await module.getCompilationInfo();
  const compileErrors = compilation.messages.filter((message) => message.type === "error");
  assert.deepEqual(compileErrors.map((message) => `${message.lineNum}: ${message.message}`), []);

  const makeBuffer = (data: Float32Array<ArrayBuffer>, usage: number) => {
    const buffer = device.createBuffer({ size: data.byteLength, usage: usage | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(buffer, 0, data);
    return buffer;
  };
  const queryBuffer = makeBuffer(input, GPUBufferUsage.STORAGE);
  const directionBuffer = makeBuffer(directions, GPUBufferUsage.STORAGE);
  const answerBytes = queries.length * outputStride * 4;
  const answerBuffer = device.createBuffer({ size: answerBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const readback = device.createBuffer({ size: answerBytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });

  const pipeline = device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "main" } });
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: queryBuffer } },
      { binding: 1, resource: { buffer: directionBuffer } },
      { binding: 2, resource: { buffer: answerBuffer } },
    ],
  });
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline); pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.ceil(queries.length / 64));
  pass.end();
  encoder.copyBufferToBuffer(answerBuffer, 0, readback, 0, answerBytes);
  device.queue.submit([encoder.finish()]);
  await readback.mapAsync(GPUMapMode.READ);
  const answers = new Float32Array(readback.getMappedRange().slice(0));
  readback.unmap();
  const validation = await device.popErrorScope();
  assert.equal(validation, null, validation?.message);

  // f32 against f64 over metre-scale geometry: 1e-5 is roughly a hundred times
  // the f32 epsilon at these magnitudes and a thousandth of the finest cell any
  // scene here uses, so it catches a wrong formula and never a rounding.
  const tolerance = 1e-5;
  let insideDisagreements = 0;
  for (let index = 0; index < queries.length; index += 1) {
    const query = queries[index];
    const shape = SCENE_SHAPES_BY_CODE.find((candidate) => candidate.code === query.shape)!;
    const offset = index * outputStride;
    const where = `${shape.name} d=${JSON.stringify(query.d)} p=${JSON.stringify(query.point)}`;
    assert.ok(Math.abs(answers[offset] - shape.distance_m(query.d, query.point)) <= tolerance,
      `distance drifted: ${where} gpu=${answers[offset]} cpu=${shape.distance_m(query.d, query.point)}`);
    assert.ok(Math.abs(answers[offset + 2] - shape.supportRadius_m(query.d, query.direction)) <= tolerance,
      `support drifted: ${where}`);
    assert.ok(Math.abs(answers[offset + 3] - shape.boundingRadius_m(query.d)) <= tolerance,
      `bounding radius drifted: ${where}`);
    assert.ok(Math.abs(answers[offset + 4] - shape.volume_m3(query.d)) <= tolerance,
      `volume drifted: ${where}`);
    // The inside test is a predicate, so it cannot be compared with a
    // tolerance: points within one f32 rounding of the surface are allowed to
    // land either way, and only a systematic disagreement is a defect.
    const inside = answers[offset + 1] > 0.5;
    if (inside !== shape.inside(query.d, query.point)
      && Math.abs(shape.distance_m(query.d, query.point)) > tolerance) insideDisagreements += 1;
  }
  assert.equal(insideDisagreements, 0, "the inside test disagrees away from the surface");
});
