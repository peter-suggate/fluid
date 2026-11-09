/**
 * The mouse grab of a cloth region. CPU: the grabbed region (connected
 * through the mesh, within the radius, smooth weights, a fold's other layer
 * left alone), the cloth probe and the implicit cloth-grab gesture. GPU (skips
 * without WEBGPU_NODE_MODULE): through the cloth system, a held region of a
 * pinned sheet follows the displacement from its live pose, the release lets
 * it fall, and stale or late presses are refused for the whole press.
 */
import assert from 'node:assert/strict';
import test from 'node:test';
import { pathToFileURL } from 'node:url';
import { createProcessRetainedDawnGPU } from '../lib/harness/node-dawn-provider';
import { requiredFluidDeviceLimits } from '../lib/core/webgpu-device-limits';
import { createEmptyScene } from '../lib/core/empty-scene';
import { validateScene, type SceneDescription } from '../lib/core/model';
import type { GPUSolverInstance } from '../lib/core/method-contract';
import type { EditorEntityContext, EditorRay } from '../lib/core/editor-entity';
import { targetAtRay } from '../lib/core/editor-probe-catalog';
import { gestureForPress } from '../lib/core/editor-gesture-catalog';
import { clothGridData } from '../lib/cloth/grid';
import { pickClothSurface } from '../lib/cloth/geometry';
import { clothGrabForHit, clothGrabRadius, clothGrabRegion, CLOTH_GRAB_CORNER_WEIGHT } from '../lib/cloth/grab';
import { CLOTH_SCENE_STEP_S, ClothSystem, withClothSystem } from '../lib/cloth/system';
import type { ClothDocument, ClothMaterial, ClothVector } from '../lib/cloth/types';

const gpu = !!process.env.WEBGPU_NODE_MODULE;
const cotton: ClothMaterial = { id: 'cotton', arealDensity_kg_m2: 0.2, stretchWarp_N_m: 400, stretchWeft_N_m: 250, shear_N_m: 40, poisson: 0.3,
  compressionRatio: 0.05, bending_N_m: 2e-5, damping_s: 0.002, thickness_m: 5e-4, strainLimit: 0.02 };
const COLUMNS = 12;
/** A horizontal 0.4 m sheet at y = 0.9, pinned at its two far corners. */
const sheetData = () => clothGridData({ columns: COLUMNS, rows: COLUMNS, width_m: 0.4, height_m: 0.4, origin_m: [-0.2, 0.9, -0.2],
  group: (i, j) => j === 0 && (i === 0 || i === COLUMNS - 1) ? 1 : 0 });
const down = (x: number, z: number) => ({ origin: [x, 2, z] as ClothVector, direction: [0, -1, 0] as ClothVector });

function sheetScene(): SceneDescription {
  const scene = createEmptyScene({ sceneId: 'cloth-grab-test' });
  const cloth: ClothDocument = { schemaVersion: '2.0.0', id: 'sheet', geometry: { kind: 'inline', data: sheetData() }, materials: [cotton],
    solver: { integrator: 'bdf2', iterations: 8 } };
  scene.cloth = [cloth];
  assert.deepEqual(validateScene(scene), []);
  return scene;
}

test('a press grabs the vertices around it that the mesh joins to the hit triangle, with weights falling to the edge', () => {
  const data = sheetData(), x = data.restPositions_m;
  const document = sheetScene().cloth![0]!;
  const hit = pickClothSurface(document, data, x, down(0.01, 0.02))!;
  assert.ok(hit);
  const R = clothGrabRadius(data), region = clothGrabRegion(data, x, hit.element, hit.point_m);
  assert.equal(region.radius_m, R);
  assert.ok(R >= 0.04 * Math.hypot(0.4, 0.4) - 1e-12, `${R}`);
  assert.equal(new Set(region.vertices).size, region.vertices.length, 'no vertex twice');
  for (let k = 0; k < 3; k++) assert.ok(region.vertices.includes(data.triangles[3 * hit.element + k]!), 'the hit triangle is held');
  assert.ok(region.vertices.length > 3 && region.vertices.length < x.length / 3 / 4, `${region.vertices.length} vertices`);
  region.vertices.forEach((v, i) => {
    const d = Math.hypot(x[3 * v]! - hit.point_m[0], x[3 * v + 1]! - hit.point_m[1], x[3 * v + 2]! - hit.point_m[2]);
    const w = region.weights[i]!;
    assert.ok(w > 0 && w <= 1);
    if (i >= 3) { assert.ok(d < R); assert.ok(Math.abs(w - (1 - (d / R) ** 2) ** 2) < 1e-12); }
    else assert.ok(w >= CLOTH_GRAB_CORNER_WEIGHT);
  });
  // Every vertex within the radius on a flat sheet is joined to the hit inside it: the region is the whole disc.
  let inside = 0;
  for (let v = 0; v < x.length / 3; v++) if (Math.hypot(x[3 * v]! - hit.point_m[0], x[3 * v + 2]! - hit.point_m[2]) < R) inside++;
  assert.equal(region.vertices.length, inside);
});

test("a fold's other layer within the radius is not grabbed unless the cloth joins it inside the region", () => {
  const data = sheetData(), folded = Float64Array.from(data.restPositions_m);
  // Fold the half z > 0.05 back over the sheet, 4 mm above it: its layer lies within reach of a press near z = 0.
  for (let v = 0; v < folded.length / 3; v++) {
    const z = folded[3 * v + 2]!;
    if (z > 0.05) { folded[3 * v + 2] = 0.1 - z; folded[3 * v + 1] = 0.9 + 0.004; }
  }
  const document = sheetScene().cloth![0]!;
  const hit = pickClothSurface(document, data, folded, { origin: [0.01, 0.5, -0.08], direction: [0, 1, 0] })!;
  assert.ok(hit && hit.point_m[1] < 0.9 + 1e-9, 'the press is on the lower layer');
  const region = clothGrabRegion(data, folded, hit.element, hit.point_m, 0.05);
  for (const v of region.vertices) assert.ok(data.restPositions_m[3 * v + 2]! <= 0.05 + 1e-9, `vertex ${v} is on the folded layer`);
});

test('the cloth probe names the sheet under the ray, and a press on it with nothing armed is a cloth grab', () => {
  const scene = sheetScene(), context: EditorEntityContext = { scene, bodies: [], pickingAvailable: true };
  const ray: EditorRay = { origin: { x: 0.01, y: 2, z: 0.02 }, direction: { x: 0, y: -1, z: 0 } };
  const target = targetAtRay(context, ray);
  assert.equal(target.kind, 'cloth');
  assert.equal(target.detail?.documentId, 'sheet');
  assert.equal(typeof target.detail?.element, 'number');
  assert.ok(Math.abs(target.point_m.y - 0.9) < 1e-9);
  const none = { shift: false, middleButton: false };
  assert.equal(gestureForPress(undefined, target, none, true), 'cloth-grab');
  // Before the scene is presented the cloth is not there to pull; shift is a pan; an armed stroke outranks the grab.
  assert.equal(gestureForPress(undefined, target, none, false), 'orbit');
  assert.equal(gestureForPress(undefined, target, { shift: true, middleButton: false }, true), 'pan');
  assert.equal(gestureForPress('fluid-paint', target, none, true), 'fluid-paint');
  // A ray that misses the sheet is not a cloth target.
  assert.notEqual(targetAtRay(context, { origin: { x: 1, y: 2, z: 1 }, direction: { x: 0, y: -1, z: 0 } }).kind, 'cloth');
});

// ---------------------------------------------------------------- GPU

async function withDevice(run: (device: GPUDevice) => Promise<void>) {
  const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href);
  Object.assign(globalThis, dawn.globals);
  const adapter = await createProcessRetainedDawnGPU(dawn, [`backend=${process.env.FLUID_WEBGPU_BACKEND ?? 'metal'}`]).requestAdapter();
  assert.ok(adapter);
  const device = await adapter.requestDevice({ requiredFeatures: adapter.features.has('subgroups') ? ['subgroups'] : [], requiredLimits: requiredFluidDeviceLimits(adapter.limits) });
  const errors: string[] = [];
  device.addEventListener('uncapturederror', (e: Event) => { e.preventDefault(); const m = (e as GPUUncapturedErrorEvent).error.message; if (!errors.length) console.error(m); errors.push(m); });
  try { await run(device); await device.queue.onSubmittedWorkDone(); assert.deepEqual(errors, []); }
  finally { device.destroy(); }
}

function dryBase(): GPUSolverInstance {
  return { info: { submittedTime_s: 0, completedTime_s: 0 } as GPUSolverInstance['info'], volumeTexture: undefined as unknown as GPUTexture,
    advanceTo: () => false, readStats: async function () { return this.info; }, destroy: () => undefined,
    stageSceneUpdate: () => undefined } as GPUSolverInstance;
}

test('a held region follows the drag from its live pose, falls when let go, and a stale or late press is refused', { skip: !gpu }, async () => withDevice(async device => {
  const scene = sheetScene(), cloth = await ClothSystem.create(device, scene, { statusInterval: 1 });
  const solver = withClothSystem(dryBase(), cloth, false);
  let tick = 0;
  const run = async (ticks: number) => {
    for (let k = 0; k < ticks; k++) assert.equal(solver.advanceTo(++tick * CLOTH_SCENE_STEP_S, []), true);
    await solver.readClothPose!();
    assert.equal(solver.info.simulationFailure, undefined, solver.info.simulationFailure?.message);
    return solver.clothSurfaces![0]!;
  };
  // Press on the sheet's middle after it has started to sag.
  const pressed = await run(6);
  const hit = pickClothSurface(scene.cloth![0]!, pressed.topology, pressed.positions_m, down(0.01, 0.02))!;
  const grab = clothGrabForHit(1, pressed, hit);
  const centroid = (x: ArrayLike<number>) => [0, 1, 2].map(c => {
    let s = 0, w = 0;
    grab.vertices.forEach((v, i) => { s += grab.weights[i]! * x[3 * v + c]!; w += grab.weights[i]!; });
    return s / w;
  });
  // A grab of another topology, or one first seen displaced (its press went to an earlier system), is refused for its press.
  assert.equal(solver.setClothGrab!({ ...grab, id: 7, topologyRevision: grab.topologyRevision + 1 }), false);
  assert.equal(solver.setClothGrab!({ ...grab, id: 7 }), false, 'the refused press stays refused');
  assert.equal(solver.setClothGrab!({ ...grab, id: 8, displacement_m: [0, 0.1, 0] }), false);
  // The real press holds; the sheet sags on while the pose read is in flight, so the anchors come from the GPU's pose.
  await run(2);
  assert.equal(solver.setClothGrab!(grab), true);
  const anchors = centroid((await run(1)).positions_m);
  // Drag the region 8 cm up and 8 cm towards the pinned edge over a third of a second, then hold. (Away from the pins
  // the sheet's 2% hard strain limit, not the grab, decides where the region can go.)
  const lift: ClothVector = [0, 0.08, -0.08];
  for (let k = 1; k <= 20; k++) {
    assert.equal(solver.setClothGrab!({ ...grab, displacement_m: lift.map(c => c * Math.min(1, k / 20)) as unknown as ClothVector }), true);
    await run(1);
  }
  const held = centroid((await run(20)).positions_m);
  const miss = Math.hypot(...[0, 1, 2].map(c => held[c]! - anchors[c]! - lift[c]!));
  // The spring holds the region against the weight of the sheet it lifts: within 3 mm of its target.
  console.log(`held region ${(1e3 * miss).toFixed(2)} mm from its target`);
  assert.ok(miss < 0.003, `held region ${miss} m from its target`);
  // Let go: the region falls back under gravity.
  assert.equal(solver.setClothGrab!(null), false);
  const fallen = centroid((await run(20)).positions_m);
  assert.ok(fallen[1]! < held[1]! - 0.05, `released region fell ${held[1]! - fallen[1]!} m`);
  solver.destroy();
}));
