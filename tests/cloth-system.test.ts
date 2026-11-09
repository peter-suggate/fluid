/**
 * Cloth documents running in scenes, and the Bridson 2002 scenes in the
 * catalog. CPU: the scene adapter (dry, valid, solids as voxel solids that
 * the shared nearest-surface query hits and a solid stroke cuts, authored
 * masses, pending solids), document validation, and the catalog cards. GPU
 * (skips without WEBGPU_NODE_MODULE): the cloth system composed onto a dry
 * scene steps one 1/60 s tick per advance and tracks the oracle, mirrors its
 * pose for picking, keeps the base instance's optional members, rebuilds a
 * cut document in place, halts explicitly at a pending solid that is not the
 * scene's moving solid, lands a sheet on the moving solid (a rigid body on a
 * pose track) and refuses any other rigid body, and captures a Bridson sheet
 * in free fall.
 */
import assert from 'node:assert/strict';
import test from 'node:test';
import { pathToFileURL } from 'node:url';
import { createProcessRetainedDawnGPU } from '../lib/harness/node-dawn-provider';
import { requiredFluidDeviceLimits } from '../lib/core/webgpu-device-limits';
import { createEmptyScene } from '../lib/core/empty-scene';
import { cloneScene, validateScene, type SceneDescription } from '../lib/core/model';
import type { GPUSolverInstance } from '../lib/core/method-contract';
import { findSceneDefinition, sceneCatalogCards } from '../lib/core/scenes';
import { sceneDocument } from '../lib/core/scene-definition';
import { sceneWithSolidStroke } from '../lib/core/solid-world';
import { visibleToolSurface } from '../lib/core/voxel-editor/surface';
import { clothGridData } from '../lib/cloth/grid';
import { validateClothDocument, prepareClothTopology, clothVertexNeighbours, colourClothVertices } from '../lib/cloth/topology';
import { buildClothSolverModel, clothOracleStep, createClothOracleState } from '../lib/cloth/oracle';
import { ClothCoarseCorrection, buildClothCoarseSpace } from '../lib/cloth/coarse';
import { sceneWithClothStroke } from '../lib/cloth/scene-edit';
import { publishSceneClothSurfaces, snapshotSceneClothPose } from '../lib/cloth/scene-geometry';
import { CLOTH_SCENE_STEP_S, ClothSystem, withClothSystem } from '../lib/cloth/system';
import type { ClothGPUSolver } from '../lib/cloth/gpu-solver';
import { CLOTH_CONTACT_FAILING, buildClothBodyField, buildClothSceneField, clothContactFieldKey, clothContactRadius, clothSceneMovingSolids } from '../lib/cloth/scene-contact';
import { proceduralSolidField } from '../lib/core/procedural-solid';
import { sampleRigidPoseTrack } from '../lib/core/prescribed-rigid-motion';
import { quaternionInverseRotate } from '../lib/core/rigid-body';
import { publishSceneRigidSurfaces, rigidBodyIsEditableSurface, sceneWithRoundedPlateCuts } from '../lib/core/rigid-surface';
import { sampleClothSolidField } from '../lib/cloth/contact/solid-field';
import { applySolidBoxCuts } from '../lib/core/solid-cuts';
import type { ClothDocument, ClothMaterial } from '../lib/cloth/types';
import { bridson2002Curtain, bridson2002RotatingSphere, bridson2002Tablecloth, BRIDSON_2002_SCENES } from '../lib/cloth/scenes/bridson-2002';
import { bridson2002SceneDescription, bridson2002SceneLayout, bridsonSolidMotionStart } from '../lib/cloth/scenes/bridson-2002/scene-description';

const gpu = !!process.env.WEBGPU_NODE_MODULE;
const BRIDSON_IDS = ['bridson-2002-curtain', 'bridson-2002-rotating-sphere', 'bridson-2002-tablecloth'];

const ray = (origin: [number, number, number], direction: [number, number, number]) => ({
  origin: { x: origin[0], y: origin[1], z: origin[2] }, direction: { x: direction[0], y: direction[1], z: direction[2] } });

/** The curtain as the catalog opens it: its ball a scene rigid body on its pose track. */
const curtain = (() => { let s: SceneDescription | undefined; return () => (s ??= bridson2002SceneDescription(bridson2002Curtain(), { movingSolids: 'rigid-body' })); })();

test('the three Bridson scenes are registered study cards that open as valid dry cloth scenes', () => {
  for (const id of BRIDSON_IDS) {
    const card = sceneCatalogCards.find(c => c.id === id);
    assert.ok(card, `${id} has a library card`);
    assert.match(card.status ?? '', /C2 M6/);
    assert.match(card.status ?? '', /diverges/);
    assert.match(card.status ?? '', /contact/);
  }
  const definition = findSceneDefinition('bridson-2002-curtain')!;
  const scene = sceneDocument(definition);
  assert.deepEqual(validateScene(scene), []);
  assert.equal(scene.systems?.fluid, false);
  assert.equal(scene.cloth?.length, 1);
});

test('held solids are voxel solids; moving solids are pending at their first motion or first reach', () => {
  assert.equal(bridsonSolidMotionStart(bridson2002Curtain().solids.find(s => s.id === 'ball')!), 2);
  assert.equal(bridsonSolidMotionStart(bridson2002RotatingSphere().solids.find(s => s.id === 'sphere')!), 0.75);
  assert.equal(bridsonSolidMotionStart(bridson2002Tablecloth().solids.find(s => s.id === 'sphere')!), 0);
  const layouts = BRIDSON_2002_SCENES.map(make => bridson2002SceneLayout(make()));
  assert.deepEqual(layouts.map(l => l.held), [['ground', 'ball'], ['ground', 'sphere', 'frustum'], ['ground', 'leg-0', 'leg-1', 'leg-2', 'leg-3']]);
  assert.deepEqual(layouts.map(l => l.pending.map(p => p.id)), [['ball'], ['sphere'], ['sphere']]);
  // The table sphere descends from 3.74 m at 0.82 m/s; it reaches the flat sheet (0.76 m) plus its radius at ~3.45 s.
  assert.ok(Math.abs(layouts[2]!.pending[0]!.start_s - 3.45) < 0.02, `${layouts[2]!.pending[0]!.start_s}`);
  for (const l of layouts) assert.ok(l.dimensions.every(d => d % 8 === 0));
});

test("the rigid-body variant makes each scene's moving solid a pose-tracked rigid sphere, not held and not pending", () => {
  for (const make of BRIDSON_2002_SCENES) {
    const source = make(), held = bridson2002SceneDescription(source, { bake: false }), scene = bridson2002SceneDescription(source, { bake: false, movingSolids: 'rigid-body' });
    assert.deepEqual(validateScene(scene), []);
    const [body, ...rest] = clothSceneMovingSolids(scene), solid = source.solids.find(s => s.id === body!.id)!, t = bridson2002SceneLayout(source).translation_m;
    assert.equal(rest.length, 0);
    assert.ok(body!.shape === 'sphere' && body!.motion === 'static' && body!.poseTrack!.keyframes.length > 1 && body!.friction === solid.friction);
    // The track is the source's, in the translated scene frame.
    assert.equal(solid.motion.kind, 'pose-track');
    const k = body!.poseTrack!.keyframes[0]!, s0 = (solid.motion as { track: { keyframes: typeof k[] } }).track.keyframes[0]!;
    assert.ok(Math.abs(k.position_m.x - s0.position_m.x - t.x) < 1e-12 && Math.abs(k.position_m.y - s0.position_m.y - t.y) < 1e-12);
    assert.deepEqual(scene.cloth![0]!.pendingSolids ?? [], (held.cloth![0]!.pendingSolids ?? []).filter(p => p.id !== body!.id));
    assert.equal(held.rigidBodies.length, 0);
  }
});

test('the cloth keeps its authored masses and lands on the translated scene frame', () => {
  const source = bridson2002Curtain(), scene = curtain(), layout = bridson2002SceneLayout(source), t = layout.translation_m;
  const data = scene.cloth![0]!.geometry.data, rest = source.cloth.geometry.data.restPositions_m;
  assert.ok(data.vertexMass_kg!.every(m => m === source.nodeMass_kg));
  assert.equal(data.restPositions_m[0], rest[0]! + t.x);
  assert.equal(data.restPositions_m[1], rest[1]! + t.y);
  // The ball is the scene's moving solid (a rigid body), so nothing is pending; held, it would stop the clock at 2 s.
  assert.equal(scene.cloth![0]!.pendingSolids, undefined);
  assert.deepEqual(bridson2002SceneDescription(source, { bake: false }).cloth![0]!.pendingSolids, [{ id: 'ball', start_s: 2 }]);
  // The container is centred on x and z, floor at y = 0.
  assert.ok(Math.abs(layout.lower_m.x + t.x + scene.container.width_m / 2) < 1e-12);
  assert.equal(layout.lower_m.y + t.y, 0);
});

test('the shared nearest-surface query hits the ground and the ball, and a solid stroke cuts the ball locally', () => {
  const scene = curtain(), t = bridson2002SceneLayout(bridson2002Curtain()).translation_m;
  // Ground plane (paper y = 0), far from the cloth. Rays sit off the lattice planes.
  const ground = visibleToolSurface(scene, ray([-2.5123 + t.x, 1, 1.4871 + t.z], [0, -1, 0]));
  assert.ok(ground && Math.abs(ground.point.y - t.y) < 0.011, `ground at ${ground?.point.y}`);
  // The ball (radius 0.25 about paper (0, 0, 0.5)), hit sideways 0.1 m above the ground, under the sheet.
  const side = ray([-1 + t.x, 0.1037 + t.y, 0.5041 + t.z], [1, 0, 0]);
  const ball = visibleToolSurface(scene, side);
  const expected = -Math.sqrt(0.25 ** 2 - 0.1037 ** 2 - 0.0041 ** 2) + t.x;
  assert.ok(ball && !ball.cloth && Math.abs(ball.point.x - expected) < 0.015, `ball at ${ball?.point.x}, expected ${expected}`);
  // Carve the ball's near cap away: the ray goes deeper, the ground elsewhere is untouched.
  const h = scene.voxelDomain.finestCellSize_m, cell = (x: number, a: 0 | 1 | 2) => Math.floor((x + [scene.container.width_m / 2, 0, scene.container.depth_m / 2][a]!) / h);
  const cut = sceneWithSolidStroke(scene, [{ operation: 'clear', affectsScenery: true,
    minimum: [cell(-0.3 + t.x, 0), cell(0.02 + t.y, 1), cell(0.35 + t.z, 2)], maximumExclusive: [cell(-0.1 + t.x, 0), cell(0.2 + t.y, 1), cell(0.65 + t.z, 2)] }]);
  const after = visibleToolSurface(cut, side);
  assert.ok(after && after.point.x > -0.1 + t.x - 0.011, `cut ball at ${after?.point.x}`);
  const groundAfter = visibleToolSurface(cut, ray([-2.5123 + t.x, 1, 1.4871 + t.z], [0, -1, 0]));
  assert.equal(groundAfter?.point.y, ground.point.y);
  // Save and load keep the same document.
  const loaded = JSON.parse(JSON.stringify(cut)) as SceneDescription;
  assert.deepEqual(validateScene(loaded), []);
  assert.equal(visibleToolSurface(loaded, side)?.point.x, after.point.x);
});

test("the moving ball is a shared editing surface at its drawn pose: hit, a local cut in its body frame that travels with it, the rest untouched, save/load and undo; a wet scene's sphere is not one", () => {
  const scene = cloneScene(curtain()), ball = scene.rigidBodies.find(b => b.id === 'ball')!, R = ball.dimensions_m.x, h = scene.voxelDomain.finestCellSize_m;
  assert.equal(rigidBodyIsEditableSurface(scene, ball), true);
  // The drawn frame at t = 3 s, as the renderer publishes it: the ball has risen and moved away from its t = 0 pose.
  const at = (t: number) => sampleRigidPoseTrack(ball.poseTrack!, t), p3 = at(3).position_m;
  assert.ok(Math.hypot(p3.x - ball.position_m.x, p3.y - ball.position_m.y) > 0.5);
  publishSceneRigidSurfaces(scene, [{ id: 'ball', ...at(3) }]);
  // Sideways rays at the posed ball: one through the near cap the cut will take, one at its far side.
  const near = ray([p3.x - 1.0037, p3.y + 0.0213, p3.z + 0.0171], [1, 0, 0]), far = ray([p3.x + 1.0041, p3.y - 0.0119, p3.z + 0.0233], [-1, 0, 0]);
  const hitNear = visibleToolSurface(scene, near)!, hitFar = visibleToolSurface(scene, far)!;
  assert.equal(hitNear.rigidBodyId, 'ball');
  assert.ok(Math.abs(Math.hypot(hitNear.point.x - p3.x, hitNear.point.y - p3.y, hitNear.point.z - p3.z) - R) < 1e-9, 'on the posed sphere');
  // The shared solid stroke: a box over the near cap, in lattice cells.
  const cell = (x: number, a: 0 | 1 | 2) => Math.floor((x + [scene.container.width_m / 2, 0, scene.container.depth_m / 2][a]!) / h);
  const stroke = [{ operation: 'clear' as const, affectsScenery: true, minimum: [cell(p3.x - 0.4, 0), cell(p3.y - 0.1, 1), cell(p3.z - 0.1, 2)] as [number, number, number],
    maximumExclusive: [cell(p3.x - 0.15, 0), cell(p3.y + 0.1, 1), cell(p3.z + 0.1, 2)] as [number, number, number] }];
  const cut = sceneWithSolidStroke(scene, stroke), body = cut.rigidBodies.find(b => b.id === 'ball')!;
  assert.equal(body.cuts?.length, 1);
  assert.deepEqual(validateScene(cut), []);
  // The cut face is the box's x = lo + 0.25-ish plane, so the near ray goes deeper; the far side and the ground keep their exact hits.
  const nearCut = visibleToolSurface(cut, near)!, farCut = visibleToolSurface(cut, far)!;
  assert.equal(nearCut.rigidBodyId, 'ball');
  assert.ok(nearCut.distance_m > hitNear.distance_m + 0.05 && Math.abs(nearCut.normal.x + 1) < 1e-9, `cut face at ${nearCut.distance_m} (was ${hitNear.distance_m})`);
  assert.equal(farCut.distance_m, hitFar.distance_m);
  const groundRay = ray([-2.5123 + p3.x, 1, 1.4871 + p3.z], [0, -1, 0]);
  assert.equal(visibleToolSurface(cut, groundRay)?.point.y, visibleToolSurface(scene, groundRay)?.point.y);
  // The cut is in the body frame: drawn at t = 3.5 s it travels with the ball.
  const p35 = at(3.5).position_m, d = { x: p35.x - p3.x, y: p35.y - p3.y, z: p35.z - p3.z }, moved = (r: ReturnType<typeof ray>) => ({ ...r, origin: { x: r.origin.x + d.x, y: r.origin.y + d.y, z: r.origin.z + d.z } });
  publishSceneRigidSurfaces(cut, [{ id: 'ball', ...at(3.5) }]);
  assert.ok(Math.hypot(d.x, d.y, d.z) > 0.2);
  assert.ok(Math.abs(at(3.5).orientation.w - at(3).orientation.w) < 1e-12, 'the ball translates on this stretch');
  assert.ok(Math.abs(visibleToolSurface(cut, moved(near))!.distance_m - nearCut.distance_m) < 1e-9);
  // Save and load keep the cut; at the same drawn pose the hits are the same.
  const loaded = JSON.parse(JSON.stringify(cut)) as SceneDescription;
  assert.deepEqual(validateScene(loaded), []);
  publishSceneRigidSurfaces(loaded, [{ id: 'ball', ...at(3) }]);
  assert.equal(visibleToolSurface(loaded, near)!.distance_m, nearCut.distance_m);
  // Undo is the previous document: the uncut ball, hit where it was.
  assert.equal(visibleToolSurface(scene, near)!.distance_m, hitNear.distance_m);
  // The cloth contact bakes the cut in the same body frame: the removed cap is outside, the far side unchanged.
  const r = clothContactRadius(cut.cloth![0]!), cutField = buildClothBodyField(cut, body, r), whole = buildClothBodyField(scene, ball, r);
  const local = (q: number[]) => quaternionInverseRotate(at(3).orientation, { x: q[0]! - p3.x, y: q[1]! - p3.y, z: q[2]! - p3.z });
  const sample = (f: typeof whole, q: number[]) => { const l = local(q); return sampleClothSolidField(f, [l.x, l.y, l.z]).phi; };
  const capPoint = [p3.x - R + 0.02, p3.y, p3.z], farPoint = [p3.x + R - 0.02, p3.y, p3.z];
  assert.ok(sample(whole, capPoint) < -0.015 && sample(cutField, capPoint) > 0, `cap ${sample(cutField, capPoint)}`);
  assert.equal(sample(cutField, farPoint), sample(whole, farPoint));
  // A wet scene's sphere keeps its object tool, and a cut of one fails validation explicitly.
  const wet = { ...cut, systems: { ...cut.systems, fluid: true } };
  assert.equal(rigidBodyIsEditableSurface(wet, ball), false);
  assert.ok(validateScene(wet).some(e => e.includes('such spheres in a dry scene')));
});

test("each Bridson bake leads solidVoxels, so the visible held solids come from the smooth procedural field", () => {
  // The curtain's only shaped solid is its ball, a rigid body in the catalog: its held world is the lattice-aligned ground alone.
  assert.equal(proceduralSolidField(curtain()), undefined);
  assert.equal(proceduralSolidField(sceneDocument(findSceneDefinition('bridson-2002-curtain')!)), undefined);
  assert.ok(proceduralSolidField(bridson2002SceneDescription(bridson2002Curtain())), 'the held ball is baked procedurally');
  for (const make of BRIDSON_2002_SCENES.filter(m => m().id !== 'bridson-2002-curtain')) {
    const scene = bridson2002SceneDescription(make(), { movingSolids: 'rigid-body' });
    const field = proceduralSolidField(scene);
    assert.ok(field, `${scene.sceneId}: the procedural boundary matches its patch prefix`);
    assert.equal(proceduralSolidField(sceneDocument(findSceneDefinition(scene.sceneId)!)) !== undefined, true, `${scene.sceneId} as the catalog opens it`);
  }
});

test('cloth contact settings: self friction is validated and carried; a solid world contact cannot see fails explicitly', () => {
  const document = curtain().cloth![0]!;
  assert.equal(document.solver?.selfFriction, 0.3);
  for (const bad of [-0.1, NaN, Infinity])
    assert.ok(validateClothDocument({ ...document, solver: { ...document.solver, selfFriction: bad } }).some(e => e.includes('self friction')));
  assert.equal(clothContactRadius(document), 1e-3);
  assert.throws(() => clothContactRadius({ ...document, materials: [{ ...document.materials[0]!, thickness_m: 0 }] }), /material thickness/);
  const empty = createEmptyScene({ sceneId: 'cloth-contact-test' }), origin = [0, 1, 0] as const;
  const field = buildClothSceneField(empty, origin, 1e-3);
  assert.ok(field.bricks.size > 0, 'the container shell is a solid');
  assert.notEqual(clothContactFieldKey(empty, origin, 1e-3), clothContactFieldKey(empty, [0, 1.5, 0], 1e-3));
  assert.throws(() => buildClothSceneField({ ...empty, terrain: { baseHeight_m: 0.1, features: [] } as unknown as SceneDescription['terrain'] }, origin, 1e-3), /terrain/);
  assert.throws(() => buildClothSceneField({ ...empty, rigidBodies: [{ id: 'b' } as SceneDescription['rigidBodies'][number]] }, origin, 1e-3), /rigid body only as a moving solid with a prescribed pose track/);
  const carved = sceneWithSolidStroke(empty, [{ operation: 'clear', affectsScenery: true, minimum: [10, 0, 10], maximumExclusive: [20, 4, 20] }]);
  assert.throws(() => buildClothSceneField(carved, origin, 1e-3), /Cloth contact cannot follow solid patch/);
});

test('pending solids are validated', () => {
  const document = curtain().cloth![0]!;
  assert.deepEqual(validateClothDocument(document), []);
  for (const bad of [[{ id: '', start_s: 1 }], [{ id: 'x', start_s: -1 }], [{ id: 'x', start_s: NaN }]])
    assert.ok(validateClothDocument({ ...document, pendingSolids: bad }).some(e => e.includes('pending solid')));
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

const cotton: ClothMaterial = { id: 'cotton', arealDensity_kg_m2: 0.2, stretchWarp_N_m: 400, stretchWeft_N_m: 250, shear_N_m: 40, poisson: 0.3,
  compressionRatio: 0.05, bending_N_m: 2e-5, damping_s: 0.002, thickness_m: 5e-4, strainLimit: 0.02 };

function sheetScene(pendingAt_s: number): SceneDescription {
  const scene = createEmptyScene({ sceneId: 'cloth-system-test' });
  const data = clothGridData({ columns: 8, rows: 8, width_m: 0.4, height_m: 0.4, origin_m: [-0.2, 0.9, -0.2],
    group: (i, j) => j === 0 && (i === 0 || i === 7) ? 1 : 0 });
  const cloth: ClothDocument = { schemaVersion: '2.0.0', id: 'sheet', geometry: { kind: 'inline', data }, materials: [cotton],
    solver: { integrator: 'bdf2', iterations: 12 }, pendingSolids: [{ id: 'probe', start_s: pendingAt_s }] };
  scene.cloth = [cloth];
  assert.deepEqual(validateScene(scene), []);
  return scene;
}

/** The dry scene's display instance as the renderer sees it: no clock of its own, no fluid members. */
function dryBase(staged: SceneDescription[]): GPUSolverInstance {
  return { info: { submittedTime_s: 0, completedTime_s: 0 } as GPUSolverInstance['info'], volumeTexture: undefined as unknown as GPUTexture,
    advanceTo: () => false, readStats: async function () { return this.info; }, destroy: () => undefined,
    stageSceneUpdate: (scene: SceneDescription) => { staged.push(scene); } } as GPUSolverInstance;
}

test('a dry scene steps its cloth one tick per advance, tracks the oracle, and halts at a pending solid', { skip: !gpu }, async () => withDevice(async device => {
  const scene = sheetScene(10 / 60), staged: SceneDescription[] = [];
  const cloth = await ClothSystem.create(device, scene, { statusInterval: 1 });
  const solver = withClothSystem(dryBase(staged), cloth, false);
  // Optional members follow the base instance.
  assert.equal('applySceneUniforms' in solver, false);
  assert.equal(solver.applySceneUniforms, undefined);
  assert.equal(typeof solver.stageSceneUpdate, 'function');
  assert.equal(solver.clothRenderSources?.length, 1);
  assert.equal(solver.clothSurfaces?.[0]?.tick, 0);
  // Half a tick owed is not a step; a whole one is.
  assert.equal(solver.advanceTo(0.4 * CLOTH_SCENE_STEP_S, []), false);
  for (let k = 1; k <= 6; k++) {
    assert.equal(solver.advanceTo(k * CLOTH_SCENE_STEP_S, []), true);
    assert.ok(Math.abs(solver.info.submittedTime_s! - k * CLOTH_SCENE_STEP_S) < 1e-12);
  }
  // Stepping never reads the pose back; a press or an edit asks for it.
  assert.equal(solver.clothSurfaces![0]!.tick, 0);
  assert.equal(typeof solver.readClothPose, 'function');
  await Promise.all([solver.readClothPose!(), solver.readClothPose!()]);
  const surface = solver.clothSurfaces![0]!;
  assert.equal(surface.tick, 6);
  // The mirrored pose carries the exact solver state (BDF2 history, AVBD duals, clock), not a restart.
  const liveSolver = () => (cloth as unknown as { entries: Map<string, { solver: ClothGPUSolver }> }).entries.get('sheet')!.solver;
  const exact = (await liveSolver().readState()).solverState;
  assert.deepEqual(surface.solverState, exact);
  assert.ok(Math.abs(exact.previousStep_s - CLOTH_SCENE_STEP_S) < 1e-9 && exact.previousVelocities_m_s.some(v => v !== 0) && exact.pinDuals.some(v => v !== 0));
  const document = scene.cloth![0]!, topology = prepareClothTopology(document.geometry.data, document.materials);
  const model = buildClothSolverModel(topology, document.materials, colourClothVertices(clothVertexNeighbours(topology)));
  const oracle = createClothOracleState(model, document.geometry.data.restPositions_m);
  // The scene's default two-level correction, from the 'previous' guess (the system's pairing).
  const coarse = new ClothCoarseCorrection(model, buildClothCoarseSpace(topology));
  for (let k = 0; k < 6; k++) clothOracleStep(model, oracle, { dt_s: CLOTH_SCENE_STEP_S, gravity_m_s2: [0, scene.fluid.gravity_m_s2.y, 0], integrator: 'bdf2', iterations: 12, chebyshevRho: 0, initialGuess: 'previous' }, false, undefined, coarse);
  let worst = 0;
  for (let i = 0; i < oracle.x.length; i++) worst = Math.max(worst, Math.abs(oracle.x[i]! - surface.positions_m[i]!));
  // The scene runs with OGC contact; with nothing in reach its truncation still bounds each
  // iteration's move, so the K = 12 iterate sits ~1.2e-4 m from the contact-free oracle's.
  assert.ok(worst < 3e-4, `scene cloth vs oracle after 6 ticks: ${worst} m`);
  assert.ok(surface.positions_m[3 * 36 + 1]! < 0.9 - 0.01, 'the free sheet fell');
  // Picking reads the mirrored pose through the shared path.
  publishSceneClothSurfaces(scene, solver.clothSurfaces!);
  const hit = visibleToolSurface(scene, ray([0, 2, 0], [0, -1, 0]));
  assert.ok(hit?.cloth && Math.abs(hit.point.y - surface.positions_m[3 * 36 + 1]!) < 0.02);

  // A cut is rebuilt in place from the edited pose; the clock waits for it.
  const edited = sceneWithClothStroke(scene, [{ kind: 'box', box: { minimum_m: [-0.06, -1, -0.06], maximum_m: [0.06, 2, 0.06] } }]);
  assert.notEqual(edited.cloth![0], scene.cloth![0]);
  const before = solver.clothRenderSources![0]!;
  solver.stageSceneUpdate!(edited);
  assert.equal(staged.at(-1), edited);
  assert.equal(solver.advanceTo(7 * CLOTH_SCENE_STEP_S, []), false, 'no step while the cut document builds');
  await cloth.settled();
  const after = solver.clothRenderSources![0]!;
  assert.ok(after.document === edited.cloth![0] && after.revision !== before.revision && after.triangleCount === edited.cloth![0]!.geometry.data.triangles.length / 3);
  assert.notEqual(after.triangleCount, before.triangleCount);
  assert.equal(solver.clothSurfaces![0]!.topologyRevision, 1);
  // The rebuilt solver continues the history: BDF2 from its first step, pin duals kept (vertex origins remap the rest);
  // its clock base is the f32 time of the cut.
  const carried = (await liveSolver().readState()).solverState;
  assert.ok(Math.abs(carried.previousStep_s - CLOTH_SCENE_STEP_S) < 1e-9 && carried.time_s === Math.fround(exact.time_s));
  assert.deepEqual(carried.pinDuals, exact.pinDuals);
  assert.ok(carried.previousVelocities_m_s.some(v => v !== 0) && carried.edgeDuals.some(v => v !== 0));
  assert.equal(solver.advanceTo(7 * CLOTH_SCENE_STEP_S, []), true);

  // The pending solid stops the clock at 10 ticks with an explicit failure.
  for (let k = 8; k <= 10; k++) assert.equal(solver.advanceTo(k * CLOTH_SCENE_STEP_S, []), true);
  assert.equal(solver.advanceTo(11 * CLOTH_SCENE_STEP_S, []), false);
  assert.equal(solver.info.simulationFailure?.code, 'cloth-pending-solid');
  assert.match(solver.info.simulationFailure!.message, /moving cloth solid 'probe' is not the scene's moving solid/);
  assert.ok(Math.abs(solver.info.submittedTime_s! - 10 * CLOTH_SCENE_STEP_S) < 1e-12);
  solver.destroy();
}));

/**
 * A sphere of radius R centred at y on a pose track (scene coordinates): it slides `dx` along x and turns `turn` rad
 * about y over `end_s`.
 */
function trackedSphere(R: number, y: number, dx: number, turn: number, end_s: number): SceneDescription['rigidBodies'][number] {
  const at = (x: number) => ({ x, y, z: 0 }), zero = { x: 0, y: 0, z: 0 }, q = (a: number) => ({ w: Math.cos(a / 2), x: 0, y: Math.sin(a / 2), z: 0 });
  return { id: 'turner', name: 'turner', shape: 'sphere', dimensions_m: { x: R, y: R, z: R }, density_kg_m3: 1000, position_m: at(0), orientation: q(0),
    linearVelocity_m_s: zero, angularVelocity_rad_s: zero, restitution: 0, friction: 0.5, motion: 'static',
    poseTrack: { keyframes: [0, 0.5, 1].map(f => ({ time_s: f * end_s, position_m: at(f * dx), orientation: q(f * turn) })) } };
}

/** The halt message (a call, so a passed `failure` check does not narrow later reads). */
const failureOf = (cloth: ClothSystem) => cloth.failure?.message;

/** A 24 x 24 sheet (1.7 cm edges) 4 cm above a sphere that slides 3 cm and turns 0.6 rad under it over 36 ticks, on a 1 cm lattice (the body field's cell). */
function movingSolidScene() {
  const R = 0.1, y = 0.76, scene = createEmptyScene({ sceneId: 'cloth-moving-solid-test', finestCellSize_m: 0.01, extents_m: { x: 1.2, y: 1.6, z: 1.2 } }), ticks = 36, end_s = ticks * CLOTH_SCENE_STEP_S;
  const data = clothGridData({ columns: 24, rows: 24, width_m: 0.4, height_m: 0.4, origin_m: [-0.2, 0.9, -0.2] });
  scene.cloth = [{ schemaVersion: '2.0.0', id: 'sheet', geometry: { kind: 'inline', data }, materials: [cotton], solver: { integrator: 'bdf2', iterations: 12 },
    pendingSolids: [{ id: 'turner', start_s: 0 }] }];
  const body = trackedSphere(R, y, 0.03, 0.6, end_s);
  scene.rigidBodies = [body];
  assert.deepEqual(validateScene(scene), []);
  return { scene, body, R, y, ticks };
}

test("the scene's moving solid (a rigid body on a pose track) is baked once and posed every frame; any other rigid body fails explicitly", { skip: !gpu }, async () => withDevice(async device => {
  const { scene, body, R, y } = movingSolidScene();
  const cloth = await ClothSystem.create(device, scene, { statusInterval: 1 }), contact = cloth.contactFor('sheet')!;
  // The pending solid at t = 0 is the scene's moving solid: the clock runs past it, with its poses uploaded every frame.
  const [frame] = await cloth.capture([8 * CLOTH_SCENE_STEP_S]);
  assert.equal(cloth.failure, undefined, cloth.failure?.message);
  assert.equal(frame![0]!.tick, 8);
  const counters = await contact.readCounters(), x = frame![0]!.positions_m, cx = 0.03 * 8 / 36;
  let nearest = Infinity;
  for (let v = 0; v < x.length / 3; v++) nearest = Math.min(nearest, Math.hypot(x[3 * v]! - cx, x[3 * v + 1]! - y, x[3 * v + 2]!));
  // The sheet falls the 4 cm onto the posed sphere in about 6 ticks and is in contact with it at tick 8, not inside it.
  assert.ok(counters.bodyContacts > 0 && nearest > R - 1e-3, `${counters.bodyContacts} body contacts, nearest vertex ${nearest - R} m off the sphere`);
  assert.equal(counters.status & CLOTH_CONTACT_FAILING, 0);
  cloth.destroy();
  // More than one moving solid, or a body the cloth cannot see as one, fails explicitly.
  scene.rigidBodies = [body, { ...body, id: 'second' }];
  await assert.rejects(ClothSystem.create(device, scene), /at most one moving solid/);
  scene.rigidBodies = [{ ...body, poseTrack: undefined, motion: 'dynamic' }];
  await assert.rejects(ClothSystem.create(device, scene), /prescribed pose track/);
  scene.rigidBodies = [{ ...body, shape: 'cup' }];
  await assert.rejects(ClothSystem.create(device, scene), /cannot bake a moving solid of shape cup/);
}));

// It halted heldConflict at tick 29 until C2's 40d73e7f.
test('a sheet lands on the moving solid and rides it: nothing passes inside, no failing status', { skip: !gpu }, async () => withDevice(async device => {
  const { scene, R, y, ticks } = movingSolidScene();
  const cloth = await ClothSystem.create(device, scene, { statusInterval: 1 }), contact = cloth.contactFor('sheet')!;
  let nearest = Infinity, bodyContacts = 0, top = -Infinity;
  for (let k = 6; k <= ticks; k += 6) {
    const [frame] = await cloth.capture([k * CLOTH_SCENE_STEP_S]);
    const x = frame![0]!.positions_m, cx = 0.03 * k / ticks;
    nearest = Infinity; top = -Infinity;
    for (let v = 0; v < x.length / 3; v++) {
      nearest = Math.min(nearest, Math.hypot(x[3 * v]! - cx, x[3 * v + 1]! - y, x[3 * v + 2]!));
      if (Math.hypot(x[3 * v]! - cx, x[3 * v + 2]!) < 0.03) top = Math.max(top, x[3 * v + 1]!);
    }
    bodyContacts = Math.max(bodyContacts, (await contact.readCounters()).bodyContacts);
    // Every 6 ticks: every vertex outside the sphere at its pose, no face inside it.
    assert.ok(nearest > R - 1e-3, `tick ${k}: a vertex ${R - nearest} m inside the moving sphere`);
    assert.deepEqual(await contact.countIntersections(), { intersections: 0, solidIntersections: 0 }, `tick ${k}`);
  }
  console.log(`moving sphere: nearest vertex ${(nearest - R).toFixed(4)} m off the sphere, cap ${(top - y - R).toFixed(4)} m above its top, peak body contacts ${bodyContacts}`);
  assert.equal(cloth.failure, undefined, cloth.failure?.message);
  // The sheet fell 4 cm onto the sphere and is held on its top: not inside, not left behind.
  assert.ok(top - y - R > 0 && top - y - R < 0.01, `cap ${top - y - R} m above the sphere`);
  cloth.destroy();
}));

test("a cut of the moving solid at edit time rebakes its body frame and rebuilds the cloth from its live pose; the sheet falls onto the cut face; undo restores the uncut body and the pose", { skip: !gpu }, async () => withDevice(async device => {
  const { scene, body, R, y, ticks } = movingSolidScene(), dt = CLOTH_SCENE_STEP_S;
  const cloth = await ClothSystem.create(device, scene, { statusInterval: 1 });
  const [rested] = await cloth.capture([12 * dt]);
  publishSceneClothSurfaces(scene, cloth.surfaces);
  const before = snapshotSceneClothPose(scene), pose = rested![0]!.positions_m.slice();
  // The editor cuts at the drawn pose: the sphere's cap above 6 cm, as a world box through the shared rigid path, stored in its body frame.
  const posed = sampleRigidPoseTrack(body.poseTrack!, 12 * dt);
  publishSceneRigidSurfaces(scene, [{ id: body.id, ...posed }]);
  const cut = sceneWithRoundedPlateCuts(scene, [{ kind: 'box', box: { minimum_m: [posed.position_m.x - 0.3, y + 0.06, -0.3], maximum_m: [posed.position_m.x + 0.3, y + 0.2, 0.3] } }]);
  const cuts = cut.rigidBodies[0]!.cuts!;
  assert.equal(cuts.length, 1);
  assert.ok(Math.abs(cuts[0]!.center_m.y - 0.13) < 1e-12 && Math.abs(cuts[0]!.halfExtents_m.y - 0.07) < 1e-12, 'the cut in the body frame');
  assert.deepEqual(validateScene(cut), []);
  assert.equal(cut.cloth, scene.cloth, 'a cut of the body alone leaves the cloth document as it is');
  cloth.adopt(cut);
  assert.equal(cloth.pending, true);
  assert.equal(cloth.stepTo(13 * dt), false, 'no step while the cut body rebuilds');
  await cloth.settled();
  await cloth.readPose();
  const resumed = cloth.surfaces[0]!;
  assert.equal(cloth.failure, undefined, failureOf(cloth));
  assert.equal(resumed.tick, 12);
  let jump = 0;
  for (let i = 0; i < pose.length; i++) jump = Math.max(jump, Math.abs(resumed.positions_m[i]! - pose[i]!));
  assert.equal(jump, 0, 'the rebuilt cloth starts from the live pose');
  // The sheet falls onto the flat cut face (body-frame y = 6 cm) and nothing passes into the cut sphere.
  const [landed] = await cloth.capture([ticks * dt]);
  assert.equal(cloth.failure, undefined, failureOf(cloth));
  const end = sampleRigidPoseTrack(body.poseTrack!, ticks * dt), x = landed![0]!.positions_m;
  let deepest = Infinity, top = -Infinity;
  for (let v = 0; v < x.length / 3; v++) {
    const local = quaternionInverseRotate(end.orientation, { x: x[3 * v]! - end.position_m.x, y: x[3 * v + 1]! - end.position_m.y, z: x[3 * v + 2]! - end.position_m.z });
    deepest = Math.min(deepest, applySolidBoxCuts(Math.hypot(local.x, local.y, local.z) - R, cuts, local));
    if (Math.hypot(local.x, local.z) < 0.03) top = Math.max(top, local.y);
  }
  console.log(`cut moving sphere: sheet centre ${(top - 0.06).toFixed(4)} m above the cut face (it rested ${(R - 0.06).toFixed(2)} m higher), deepest vertex ${deepest.toFixed(4)} m`);
  assert.ok(deepest > -1e-3, `a vertex ${-deepest} m inside the cut sphere`);
  assert.ok(top > 0.06 && top < 0.06 + 0.01, `sheet centre ${top - 0.06} m above the cut face`);
  // Undo: the history endpoint brings back the uncut body and the pose at tick 12, and steps on from there.
  cloth.adopt(before);
  await cloth.settled();
  await cloth.readPose();
  const undone = cloth.surfaces[0]!;
  assert.equal(undone.tick, 12);
  assert.ok(undone.positions_m.every((v, i) => v === pose[i]), 'the undone cloth is at the endpoint pose');
  assert.equal(cloth.failure, undefined, failureOf(cloth));
  await cloth.capture([18 * dt]);
  assert.equal(cloth.failure, undefined, failureOf(cloth));
  cloth.destroy();
}));

test('in contact, a history endpoint steps on as if never rebuilt', { skip: !gpu,
  todo: "the endpoint lacks the detector's cross-step per-vertex state (rho_v, held in the last step): 2.2 mm after 6 ticks resting on the moving sphere; C2 to expose it, C1 to carry it with the solver state" },
async () => withDevice(async device => {
  const { scene } = movingSolidScene(), dt = CLOTH_SCENE_STEP_S;
  const reference = await ClothSystem.create(device, cloneScene(scene));
  const straight = (await reference.capture([18 * dt]))[0]![0]!;
  reference.destroy();
  const cloth = await ClothSystem.create(device, scene, { statusInterval: 1 });
  await cloth.capture([12 * dt]);
  publishSceneClothSurfaces(scene, cloth.surfaces);
  cloth.adopt(snapshotSceneClothPose(scene));
  await cloth.settled();
  const [again] = await cloth.capture([18 * dt]);
  let drift = 0;
  for (let i = 0; i < straight.positions_m.length; i++) drift = Math.max(drift, Math.abs(again![0]!.positions_m[i]! - straight.positions_m[i]!));
  console.log(`endpoint rebuilt at tick 12 in contact: 6 ticks on, ${drift.toExponential(2)} m from the straight run`);
  cloth.destroy();
  assert.ok(drift < 1e-5, `drift ${drift} m`);
}));

test('undo of a cut through the system: the history endpoint rebuilds the uncut sheet with its exact state and steps on as if never cut', { skip: !gpu }, async () => withDevice(async device => {
  const scene = sheetScene(1e9), staged: SceneDescription[] = [];
  // The reference never cuts: 9 ticks straight.
  const reference = await ClothSystem.create(device, cloneScene(scene));
  const uncut = (await reference.capture([9 * CLOTH_SCENE_STEP_S]))[0]![0]!;
  reference.destroy();
  const cloth = await ClothSystem.create(device, scene, { statusInterval: 1 });
  const solver = withClothSystem(dryBase(staged), cloth, false);
  const liveState = () => (cloth as unknown as { entries: Map<string, { solver: ClothGPUSolver }> }).entries.get('sheet')!.solver.readState();
  for (let k = 1; k <= 6; k++) assert.equal(solver.advanceTo(k * CLOTH_SCENE_STEP_S, []), true);
  await solver.readClothPose!();
  publishSceneClothSurfaces(scene, solver.clothSurfaces!);
  // The editor's history endpoint before the cut: the accepted pose and the exact solver state.
  const before = snapshotSceneClothPose(scene), saved = before.cloth![0]!.geometry.data;
  assert.equal(saved.triangles, scene.cloth![0]!.geometry.data.triangles);
  assert.ok(saved.initialPositions_m && saved.solverState && Math.abs(saved.solverState.time_s - 6 * CLOTH_SCENE_STEP_S) < 1e-6);
  const edited = sceneWithClothStroke(scene, [{ kind: 'box', box: { minimum_m: [-0.06, -1, -0.06], maximum_m: [0.06, 2, 0.06] } }]);
  solver.stageSceneUpdate!(edited);
  await cloth.settled();
  for (let k = 7; k <= 9; k++) assert.equal(solver.advanceTo(k * CLOTH_SCENE_STEP_S, []), true);
  const cut = solver.clothRenderSources![0]!;
  assert.equal(cut.triangleCount, edited.cloth![0]!.geometry.data.triangles.length / 3);

  // Undo: stage the endpoint. The uncut document is rebuilt from its pose and state; the clock goes back to tick 6.
  solver.stageSceneUpdate!(before);
  assert.equal(solver.advanceTo(10 * CLOTH_SCENE_STEP_S, []), false, 'no step while the undone document builds');
  await cloth.settled();
  const undone = solver.clothRenderSources![0]!;
  assert.ok(undone.document === before.cloth![0] && undone.revision !== cut.revision && undone.triangleCount === saved.triangles.length / 3);
  assert.equal(solver.clothSurfaces![0]!.topologyRevision, 2);
  assert.equal(solver.clothSurfaces![0]!.tick, 6);
  const restored = await liveState();
  for (let i = 0; i < restored.positions_m.length; i++) assert.ok(Math.abs(restored.positions_m[i]! - saved.initialPositions_m![i]!) < 1e-6, `position ${i}`);
  const want = saved.solverState!, got = restored.solverState;
  assert.ok(Math.abs(got.previousStep_s - want.previousStep_s) < 1e-9 && got.time_s === Math.fround(want.time_s));
  for (const key of ['previousVelocities_m_s', 'previousDisplacements_m', 'pinDuals', 'edgeDuals'] as const) {
    let worst = 0;
    want[key].forEach((v, i) => { worst = Math.max(worst, Math.abs(v - got[key][i]!)); });
    assert.ok(worst <= 1e-6 * Math.max(1, ...want[key].map(Math.abs)), `${key} restored to ${worst}`);
  }
  // It steps on from tick 6 to 9 as the reference did without the cut: the host clock goes on, the scene's time follows the cloth's.
  for (let k = 10; k <= 12; k++) {
    assert.equal(solver.advanceTo(k * CLOTH_SCENE_STEP_S, []), true);
    assert.ok(Math.abs(solver.info.submittedTime_s! - (k - 3) * CLOTH_SCENE_STEP_S) < 1e-9);
  }
  await solver.readClothPose!();
  const after = solver.clothSurfaces![0]!;
  assert.equal(after.tick, 9);
  let drift = 0;
  for (let i = 0; i < uncut.positions_m.length; i++) drift = Math.max(drift, Math.abs(after.positions_m[i]! - uncut.positions_m[i]!));
  console.log(`undo: 3 ticks after the undo the sheet is ${drift.toExponential(2)} m from the never-cut run`);
  assert.ok(drift < 1e-5, `undone sheet ${drift} m from the never-cut run`);
  // Redo stages the cut document again.
  solver.stageSceneUpdate!(edited);
  await cloth.settled();
  assert.equal(solver.clothRenderSources![0]!.triangleCount, cut.triangleCount);
  assert.equal(cloth.failure, undefined, cloth.failure?.message);
  solver.destroy();
}));

test('capture runs a 150 x 150 Bridson sheet on the scene clock with contact: free fall above the floor', { skip: !gpu }, async () => withDevice(async device => {
  const scene = bridson2002SceneDescription(bridson2002Tablecloth(), { bake: false });
  const cloth = await ClothSystem.create(device, cloneScene(scene));
  const t0 = performance.now();
  const [start, later] = await cloth.capture([0, 6 * CLOTH_SCENE_STEP_S]);
  const ms = performance.now() - t0;
  const rest = scene.cloth![0]!.geometry.data.restPositions_m, x = later![0]!.positions_m;
  assert.equal(start![0]!.tick, 0);
  assert.equal(later![0]!.tick, 6);
  // An unpinned flat sheet falls undeformed: every vertex drops alike (BDF2 at 1/60 s).
  let lo = Infinity, hi = -Infinity;
  for (let i = 1; i < x.length; i += 3) { const d = rest[i]! - x[i]!; lo = Math.min(lo, d); hi = Math.max(hi, d); }
  const exact = 0.5 * 9.81 * (6 * CLOTH_SCENE_STEP_S) ** 2;
  assert.ok(Math.abs(lo - exact) < 0.25 * exact && hi - lo < 1e-3, `drop ${lo}..${hi} m, exact ${exact}`);
  const t1 = performance.now();
  await cloth.capture([18 * CLOTH_SCENE_STEP_S]);
  console.log(`tablecloth ${rest.length / 3} vertices, K=${scene.cloth![0]!.solver?.iterations}: first 6 ticks ${ms.toFixed(0)} ms, next 12 ticks ${((performance.now() - t1) / 12).toFixed(1)} ms per tick (fenced each tick)`);
  cloth.destroy();
}));

test('the rotating-sphere sheet lands on the held sphere through scene contact: no vertex inside, no failing status', { skip: !gpu }, async () => withDevice(async device => {
  const source = bridson2002RotatingSphere(), scene = bridson2002SceneDescription(source), t = bridson2002SceneLayout(source).translation_m;
  const t0 = performance.now();
  const cloth = await ClothSystem.create(device, cloneScene(scene), { statusInterval: 1 });
  const setup = performance.now() - t0, ticks = Number(process.env.CLOTH_CONTACT_TICKS ?? 20);
  const t1 = performance.now();
  const [frame] = await cloth.capture([ticks * CLOTH_SCENE_STEP_S]);
  const ms = (performance.now() - t1) / ticks;
  const counters = await cloth.contactFor(scene.cloth![0]!.id)!.readCounters();
  assert.equal(cloth.failure, undefined, cloth.failure?.message);
  const x = frame![0]!.positions_m, centre = [t.x, 0.3 + t.y, t.z], R = 0.25;
  let inside = Infinity, top = -Infinity;
  for (let i = 0; i < x.length; i += 3) {
    const d = Math.hypot(x[i]! - centre[0]!, x[i + 1]! - centre[1]!, x[i + 2]! - centre[2]!);
    inside = Math.min(inside, d);
    if (Math.hypot(x[i]! - centre[0]!, x[i + 2]! - centre[2]!) < 0.02) top = Math.max(top, x[i + 1]!);
  }
  const N = x.length / 3;
  console.log(`rotating sphere ${N} vertices with contact: setup ${setup.toFixed(0)} ms, ${ms.toFixed(1)} ms per tick over ${ticks} ticks (fenced each tick); `
    + `closest vertex ${(inside - R).toFixed(4)} m off the sphere, cap top ${(top - centre[1]! - R).toFixed(4)} m; status ${counters.status}, `
    + `solid contacts ${counters.solidContacts}, peak pair slots ${(counters.peakPairSlots / N).toFixed(0)}N, peak grid ${counters.peakGrid}, peak contacts ${counters.peakContacts}`);
  // Free fall from 1 cm above the top reaches the sphere in ~3 ticks: the cap rests on it, nothing passes inside.
  assert.ok(inside > R - 2e-3, `a vertex is ${R - inside} m inside the sphere`);
  assert.ok(top > centre[1]! + R - 2e-3 && top < centre[1]! + R + 0.012, `cap at ${top - centre[1]! - R} m above the sphere top`);
  cloth.destroy();
}));
