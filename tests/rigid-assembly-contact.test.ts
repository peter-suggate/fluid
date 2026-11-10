import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import asset from '../lib/core/imported-assets/paddleboats.generated.json';
import { createPaddleboatsScene } from '../lib/core/paddleboats-scene';
import { convexHullVertices, hullSupportDeficit, pointSupport, sphereDirections } from '../lib/core/convex-hull';
import { massProperties } from '../lib/core/rigid-body';
import {
  RIGID_HULL_ENVELOPE, RIGID_HULL_ENVELOPE_CLOUD, RIGID_SUPPORT_TIE_M, encodeMirrorBodies, mirrorShapeSupportRadius, qInverseRotate, qRotate,
  rigidAssemblyBuild, rigidAssemblyContactWGSL, rigidAssemblySolidWGSL, rigidAxleEnvelope, rigidPieceContact, rigidPieceExists, rigidPieceReach,
  rigidContactTableHeader, rigidContactTableLayout, rigidContactTableThreads,
  rigidPieceSupport, stepRigidMirror, type MirrorBody, type MirrorWorld, type V3, type V4,
} from '../lib/core/rigid-assembly-contact';
import { GPU_RIGID_BODY_CAPACITY } from '../lib/core/rigid-capacity';
import { rigidMotorRecords } from '../lib/core/webgpu-rigid-body';
import { rigidMirrorWorld } from './helpers/rigid-mirror-world';
import { Box3DMotorContacts } from './helpers/box3d-motor-oracle';

const DT = 1 / 60;
const ZERO_G = { x: 0, y: 0, z: 0 };
const wheelCloud: V3[] = asset.wheel.nodes.filter(n => n.part >= 0).flatMap(n => Array.from({ length: 8 }, (_, i) =>
  [(i & 1 ? n.hi : n.lo)[0]!, (i & 2 ? n.hi : n.lo)[1]!, (i & 4 ? n.hi : n.lo)[2]!] as V3));
const hullCloud = RIGID_HULL_ENVELOPE_CLOUD.map(p => [...p] as V3);
const dot = (a: V3, b: V3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const add = (a: V3, b: V3): V3 => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const xyz = (v: V4): V3 => [v[0], v[1], v[2]];
const yawQ = (yaw: number): V4 => [Math.cos(yaw / 2), 0, Math.sin(yaw / 2), 0];

/** One boat (hull + its eight blades), or both, placed and launched. */
function boats(count: 1 | 2, place: (hull: ReturnType<typeof createPaddleboatsScene>['rigidBodies'][number], index: number) => void, gravity = ZERO_G) {
  const scene = createPaddleboatsScene();
  scene.fluid.gravity_m_s2 = gravity;
  scene.rigidBodies = scene.rigidBodies.slice(0, 9 * count);
  scene.rigidBodies.filter(b => !b.motorAttachment).forEach(place);
  return { scene, world: rigidMirrorWorld(scene) };
}

/** Every rendered point of an assembly in world space: the hull's BVH leaf corners and each blade's at its current pose. */
function assemblyPoints(w: MirrorWorld, root: number): V3[] {
  const hull = w.bodies[root]!;
  const points = hullCloud.map(p => add(xyz(hull.positionShape), qRotate(hull.orientation, p)));
  w.motors.forEach((m, index) => {
    if (Math.round(m.pivotParent[3]) !== root + 1) return;
    const blade = w.bodies[index]!;
    for (const p of wheelCloud) points.push(add(xyz(blade.positionShape), qRotate(blade.orientation, p)));
  });
  return points;
}

function kineticEnergy(w: MirrorWorld, index: number, mass: number, inertia: V3): number {
  const b = w.bodies[index]!, local = qInverseRotate(b.orientation, xyz(b.angularVelocity));
  return .5 * mass * dot(xyz(b.linearVelocity), xyz(b.linearVelocity))
    + .5 * (inertia[0] * local[0] ** 2 + inertia[1] * local[1] ** 2 + inertia[2] * local[2] ** 2);
}

const hullMass = () => {
  const p = massProperties(createPaddleboatsScene().rigidBodies[0]!);
  return { mass: p.mass_kg, inertia: [p.inertiaBody_kg_m2.x, p.inertiaBody_kg_m2.y, p.inertiaBody_kg_m2.z] as V3 };
};

test('the hull envelope is the exact hull of its BVH leaf corners', () => {
  const directions = sphereDirections(4000);
  const exact = convexHullVertices(RIGID_HULL_ENVELOPE_CLOUD);
  assert.equal(hullSupportDeficit(RIGID_HULL_ENVELOPE_CLOUD, exact, directions), 0, 'the hull vertices carry the cloud support exactly');
  // The kernel's vertices are f32-rounded: within an f32 ulp of the cloud.
  assert.ok(hullSupportDeficit(RIGID_HULL_ENVELOPE_CLOUD, RIGID_HULL_ENVELOPE, directions) < 1e-6);
  assert.ok(RIGID_HULL_ENVELOPE.length < 100, `the single-thread kernel scans ${RIGID_HULL_ENVELOPE.length} vertices, not ${RIGID_HULL_ENVELOPE_CLOUD.length}`);
  assert.throws(() => convexHullVertices([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]]), /coplanar/, 'a flat cloud is not an envelope');
});

test('compound support equals the brute-force max over the assembly vertices', () => {
  const { world: w } = boats(1, hull => { hull.orientation = { w: .9, x: .2, y: .3, z: .1 }; });
  // Renormalize the arbitrary orientation the way the GPU record holds it.
  const q = w.bodies[0]!.orientation, l = Math.hypot(...q); w.bodies[0]!.orientation = q.map(v => v / l) as V4;
  const body = w.bodies[0]!;
  const d = xyz(body.dimensions), corners: V3[] = Array.from({ length: 8 }, (_, i) =>
    [(i & 1 ? .5 : -.5) * d[0], (i & 2 ? .5 : -.5) * d[1], (i & 4 ? .5 : -.5) * d[2]]);
  const pieces = Array.from({ length: 2 + w.bodies.length }, (_, p) => p).filter(p => rigidPieceExists(w, 0, p));
  assert.deepEqual(pieces, [0, 1, 3, 7], 'core, hull envelope, and one axle per wheel (blades share it)');
  // Each wheel sweeps its blades through every phase: sample the sweep.
  const swept: V3[] = [];
  for (let phase = 0; phase < 720; phase++) {
    const half = .5 * phase * 2 * Math.PI / 720, rot: V4 = [Math.cos(half), 0, 0, Math.sin(half)];
    for (let index = 1; index < 9; index++) {
      const m = w.motors[index]!;
      for (const p of wheelCloud) swept.push(add(xyz(m.pivotParent), qRotate(rot, add(xyz(m.offsetPhase), qRotate(m.orientation, p)))));
    }
  }
  let axleSlack = 0;
  for (const u of sphereDirections(600) as V3[]) {
    const local = qInverseRotate(body.orientation, u);
    const reach = (p: number) => rigidPieceReach(w, body, p, u);
    assert.ok(Math.abs(reach(0) - pointSupport(corners, local)) < 1e-6, 'piece 0 is the core box');
    assert.ok(Math.abs(reach(1) - pointSupport(hullCloud, local)) < 1e-6, 'piece 1 is the hull envelope, exactly');
    const sweptMax = pointSupport(swept, local), axleMax = Math.max(reach(3), reach(7));
    assert.ok(axleMax >= sweptMax - 1e-6, 'each axle cylinder encloses every phase of its wheel');
    axleSlack = Math.max(axleSlack, axleMax - sweptMax);
    // Support points lie on the support plane: they are where the contact acts.
    for (const p of pieces) {
      const s = rigidPieceSupport(w, body, p, u);
      assert.ok(Math.abs(dot(s.point, u) - s.reach) <= RIGID_SUPPORT_TIE_M + 1e-6, `piece ${p}'s contact point is within the tie band of its support plane`);
    }
    const compound = Math.max(...pieces.map(reach));
    const brute = Math.max(pointSupport(corners, local), pointSupport(hullCloud, local), sweptMax);
    assert.ok(compound >= brute - 1e-6 && compound - brute <= axleSlack + 1e-9);
  }
  // The cylinder is the sweep's bound up to its faceting in phase (720 samples) and its blade ends.
  assert.ok(axleSlack < .01, `axle envelope slack ${axleSlack} m`);
});

test('an ordinary box comes to rest on the floor without jitter or energy gain', () => {
  const scene = createPaddleboatsScene();
  scene.rigidBodies = [{ ...scene.rigidBodies[0]!, appearance: undefined, id: 'crate', position_m: { x: 0, y: .3, z: 0 } }];
  const w = rigidMirrorWorld(scene);
  for (let i = 0; i < 180; i++) stepRigidMirror(w, DT);
  const b = w.bodies[0]!;
  assert.ok(Math.abs(b.positionShape[1] - .14) < 1e-3, `rests on its half height: ${b.positionShape[1]}`);
  assert.ok(Math.hypot(...xyz(b.linearVelocity)) < 1e-6, `rests: ${Math.hypot(...xyz(b.linearVelocity))} m/s after the contact sweeps`);
  assert.ok(Math.hypot(...xyz(b.angularVelocity)) < 1e-9, 'a flat landing does not rotate it');
});

test('a paddleboat lands on the floor on its lowest pieces and settles', () => {
  const { world: w } = boats(1, hull => { hull.position_m = { x: 0, y: .5, z: 0 }; }, { x: 0, y: -9.80665, z: 0 });
  let deepest = 0;
  for (let i = 0; i < 240; i++) {
    stepRigidMirror(w, DT);
    deepest = Math.min(deepest, ...assemblyPoints(w, 0).map(p => p[1]));
  }
  assert.ok(deepest > -2e-3, `no rendered point sinks into the floor: ${deepest}`);
  assert.ok(Math.hypot(...xyz(w.bodies[0]!.linearVelocity)) < 1e-3 && Math.hypot(...xyz(w.bodies[0]!.angularVelocity)) < 1e-3,
    `it comes to rest: ${Math.hypot(...xyz(w.bodies[0]!.linearVelocity))} m/s, ${Math.hypot(...xyz(w.bodies[0]!.angularVelocity))} rad/s`);
  assert.ok(Math.min(...assemblyPoints(w, 0).map(p => p[1])) < .01, 'and rests on the floor');
});

for (const [yaw, speed] of [[0, 4], [Math.PI / 6, 4], [Math.PI / 4, 4], [Math.PI / 3, 4], [Math.PI / 2, 4], [0, 40], [Math.PI / 4, 40]] as const) {
  test(`a hull driven into the side wall at yaw ${yaw.toFixed(2)}, ${speed} m/s: no part crosses it, energy does not grow`, () => {
    const { world: w } = boats(1, hull => {
      hull.position_m = { x: 2, y: .6, z: 0 }; const q = yawQ(yaw);
      hull.orientation = { w: q[0], x: q[1], y: q[2], z: q[3] }; hull.linearVelocity_m_s = { x: 0, y: 0, z: speed };
    });
    const { mass, inertia } = hullMass(), e0 = kineticEnergy(w, 0, mass, inertia);
    let worst = -Infinity;
    for (let frame = 0; frame < 90; frame++) {
      stepRigidMirror(w, DT);
      for (const p of assemblyPoints(w, 0)) worst = Math.max(worst, Math.abs(p[2]) - 2.4, Math.abs(p[0]) - 6.4, -p[1]);
      assert.ok(kineticEnergy(w, 0, mass, inertia) <= e0 * (1 + 1e-9), 'contact does not create energy');
      assert.ok(encodeMirrorBodies(w.bodies).every(Number.isFinite));
    }
    assert.ok(worst <= 1e-3, `a rendered point crossed a wall by ${worst} m`);
    assert.ok(w.bodies[0]!.linearVelocity[2] < speed * .8, 'the wall changes the hull velocity');
  });
}

test('a 40 m/s stern-first impact into the end wall does not tunnel', () => {
  const { world: w } = boats(1, hull => { hull.position_m = { x: 0, y: .6, z: 0 }; hull.linearVelocity_m_s = { x: -40, y: 0, z: 0 }; });
  let worst = -Infinity;
  for (let frame = 0; frame < 30; frame++) {
    stepRigidMirror(w, DT);
    for (const p of assemblyPoints(w, 0)) worst = Math.max(worst, Math.abs(p[0]) - 6.4);
  }
  assert.ok(worst <= 1e-3, `the wheels crossed the end wall by ${worst} m`);
  assert.ok(w.bodies[0]!.positionShape[0] > -6.4 && w.bodies[0]!.linearVelocity[0] > -32);
});

/** Deepest SAT overlap between any two pieces of the two hulls (positive: interpenetrating). */
function pairDepth(w: MirrorWorld, a: number, b: number): number {
  let depth = -Infinity;
  for (let ap = 0; ap < 2 + w.bodies.length; ap++) for (let bp = 0; bp < 2 + w.bodies.length; bp++) {
    if (rigidPieceExists(w, a, ap) && rigidPieceExists(w, b, bp)) depth = Math.max(depth, rigidPieceContact(w, w.bodies[a]!, ap, w.bodies[b]!, bp).depth);
  }
  return depth;
}
const momentum = (w: MirrorWorld, index: number, mass: number): V3 => xyz(w.bodies[index]!.linearVelocity).map(v => v * mass) as V3;

// Clear of the floor (zero gravity): a bow-on collision pitches the hulls, and
// their stern wheels would otherwise reach the floor inside the window.
for (const [label, relative, offset] of [['head-on', 4, 0], ['head-on', 40, 0], ['glancing', 4, .15], ['glancing', 40, .15]] as const) {
  test(`two hulls ${label} at ${relative} m/s closing: no pass-through, momentum conserved, energy does not grow`, () => {
    const { world: w } = boats(2, (hull, index) => {
      hull.position_m = { x: index ? 2.2 : -2.2, y: 2, z: index ? offset : -offset };
      hull.linearVelocity_m_s = { x: (index ? -.5 : .5) * relative, y: 0, z: 0 };
    });
    const { mass, inertia } = hullMass(), e0 = kineticEnergy(w, 0, mass, inertia) + kineticEnergy(w, 9, mass, inertia);
    const p0 = add(momentum(w, 0, mass), momentum(w, 9, mass));
    let deepest = -Infinity, pairContact = false;
    for (let frame = 0; frame < 60; frame++) {
      stepRigidMirror(w, DT);
      deepest = Math.max(deepest, pairDepth(w, 0, 9));
      const e = kineticEnergy(w, 0, mass, inertia) + kineticEnergy(w, 9, mass, inertia);
      assert.ok(e <= e0 * (1 + 1e-9), `frame ${frame}: energy ${e} > ${e0}`);
      const p = add(momentum(w, 0, mass), momentum(w, 9, mass));
      // Until a wall is within reach of either compound, the pair alone exchanges momentum.
      const bound = rigidAssemblyBuild(w)[0]!, clear = [0, 9].every(i => { const c = w.bodies[i]!.positionShape;
        return Math.abs(c[0]) + bound < 6.4 && Math.abs(c[2]) + bound < 2.4 && c[1] - bound > 0; });
      if (clear && w.bodies[0]!.linearVelocity[0] < .5 * relative * .95) pairContact = true;
      if (clear) assert.ok(Math.hypot(p[0] - p0[0], p[1] - p0[1], p[2] - p0[2]) < 1e-6 * mass * relative, `frame ${frame}: pair impulses are equal and opposite: ${p} vs ${p0}`);
    }
    if (label === 'head-on') assert.ok(w.bodies[0]!.positionShape[0] < w.bodies[9]!.positionShape[0], 'the hulls cannot pass through each other');
    assert.ok(deepest <= 1e-3, `pieces interpenetrate by ${deepest} m after the contact sweeps`);
    assert.ok(pairContact, 'the hulls meet (no tunnelling) before either reaches a wall');
  });
}

test('separating-axis gaps are certificates: the pieces really are that far apart', () => {
  const { world: w } = boats(2, (hull, index) => { hull.position_m = { x: index ? 1 : -1, y: .6, z: 0 }; });
  let random = 7;
  const next = () => (random = (random * 16807) % 2147483647) / 2147483647;
  let checked = 0;
  for (let trial = 0; trial < 400; trial++) {
    const a = w.bodies[0]!, b = w.bodies[9]!;
    const q = (): V4 => { const v: V4 = [next() - .5, next() - .5, next() - .5, next() - .5]; const l = Math.hypot(...v); return v.map(x => x / l) as V4; };
    a.orientation = q(); b.orientation = q();
    b.positionShape = [a.positionShape[0] + 3 * (next() - .5), .6 + 2 * (next() - .5), 3 * (next() - .5), b.positionShape[3]];
    const contact = rigidPieceContact(w, a, 1, b, 1);
    if (contact.depth >= 0) continue;
    const world = (body: MirrorBody) => hullCloud.map(p => add(xyz(body.positionShape), qRotate(body.orientation, p)));
    const gap = Math.min(...world(b).map(p => dot(p, contact.normal))) - Math.max(...world(a).map(p => dot(p, contact.normal)));
    assert.ok(gap >= -contact.depth - 1e-6, `a reported gap of ${-contact.depth} m is only ${gap} m`);
    checked++;
  }
  assert.ok(checked > 100);
});

test('penetration is recovered by projection without injecting velocity', () => {
  // Both hulls start 0.13-0.33 m into the side wall (wheels and envelope),
  // and bow into bow by 0.16 m, at rest.
  const { world: w } = boats(2, (hull, index) => { hull.position_m = index ? { x: -.5, y: .6, z: 2 } : { x: -2, y: .6, z: 2.2 }; }, ZERO_G);
  assert.ok(pairDepth(w, 0, 9) > .1, `starts overlapped: ${pairDepth(w, 0, 9)}`);
  stepRigidMirror(w, DT);
  assert.ok(pairDepth(w, 0, 9) <= 1e-5, `pair overlap after one step ${pairDepth(w, 0, 9)}`);
  for (const root of [0, 9]) for (const p of assemblyPoints(w, root)) assert.ok(Math.abs(p[2]) <= 2.4 + 1e-5, `wall overlap ${Math.abs(p[2]) - 2.4}`);
  for (const index of [0, 9]) {
    assert.ok(Math.hypot(...xyz(w.bodies[index]!.linearVelocity)) < 1e-9, 'positional recovery leaves bodies at rest');
    assert.ok(Math.hypot(...xyz(w.bodies[index]!.angularVelocity)) < 1e-9);
  }
});

test('a dry, contact-free scene stays bit-identical: no gravity or motor thrust leaks into the hulls', () => {
  const scene = createPaddleboatsScene(); scene.fluid.gravity_m_s2 = ZERO_G;
  const w = rigidMirrorWorld(scene), roots = [0, 9];
  const before = roots.map(i => structuredClone(w.bodies[i]!));
  for (let i = 0; i < 30; i++) stepRigidMirror(w, DT);
  assert.deepEqual(roots.map(i => w.bodies[i]!), before);
  assert.ok(rigidAssemblyBuild(w)[0]! > 1, 'both hulls are assemblies');
});

test('the mirror agrees with Box3D on what a wheel-first wall impact does', async () => {
  // Box3D (TEST ORACLE ONLY) treats the same envelopes as hull shapes with
  // its own solver; the outcomes agree in kind, not bit for bit.
  for (const [yaw, speed] of [[0, 4], [Math.PI / 2, 4]] as const) {
    const scene = createPaddleboatsScene(); scene.rigidBodies = scene.rigidBodies.slice(0, 9);
    const hull = scene.rigidBodies[0]!; hull.position_m = { x: 3, y: .6, z: 0 }; const q = yawQ(yaw);
    hull.orientation = { w: q[0], x: q[1], y: q[2], z: q[3] }; hull.linearVelocity_m_s = { x: 0, y: 0, z: speed };
    const oracle = await Box3DMotorContacts.create(scene, scene.rigidBodies);
    const values = new Float32Array(24 * 32);
    values.set(encodeMirrorBodies(rigidMirrorWorld(scene).bodies));
    const w = rigidMirrorWorld({ ...scene, fluid: { ...scene.fluid, gravity_m_s2: ZERO_G } });
    try {
      let firstOracle = -1, firstMirror = -1;
      for (let frame = 0; frame < 60; frame++) {
        oracle.step(values, scene.rigidBodies, DT); stepRigidMirror(w, DT);
        if (firstOracle < 0 && values[14]! < speed * .99) firstOracle = frame;
        if (firstMirror < 0 && w.bodies[0]!.linearVelocity[2] < speed * .99) firstMirror = frame;
      }
      assert.ok(firstOracle > 0 && Math.abs(firstMirror - firstOracle) <= 1, `contact begins on the same frame: Box3D ${firstOracle}, mirror ${firstMirror}`);
      assert.ok(Math.abs(w.bodies[0]!.positionShape[2] - values[2]!) < .15, `the hull stops where Box3D stops it: ${w.bodies[0]!.positionShape[2]} vs ${values[2]}`);
      assert.ok(w.bodies[0]!.linearVelocity[2] < .5 && values[14]! < .5, 'both absorb the approach');
    } finally { oracle.destroy(); }
  }
});

test('one step a frame stays within 1.2 cm of the refined answer for a wheel-first wall impact', () => {
  // The GPU suite's impact (tests/rigid-motor-gpu.test.ts): 3 m/s at the side
  // wall in zero g, 0.75 s. Stepped at (1/60)/n the final hull z converges to
  // 1.947 (Box3D, refined, to about 1.949); one step a frame gives 1.9565.
  const final = (n: number) => {
    const scene = createPaddleboatsScene(); scene.fluid.gravity_m_s2 = { x: 0, y: 0, z: 0 };
    scene.rigidBodies = scene.rigidBodies.slice(0, 9);
    scene.rigidBodies[0]!.position_m = { x: 0, y: .6, z: 1.5 }; scene.rigidBodies[0]!.linearVelocity_m_s = { x: 0, y: 0, z: 3 };
    scene.rigidBodies[0]!.motorStrengths = { left: 0, right: -.5 };
    const w = rigidMirrorWorld(scene);
    for (let i = 0; i < 45 * n; i++) stepRigidMirror(w, DT / n);
    return w.bodies[0]!.positionShape[2];
  };
  const coarse = final(1), refined = final(64);
  assert.ok(Math.abs(refined - 1.947) < 2e-3, `the refined answer: ${refined}`);
  assert.ok(Math.abs(coarse - refined) < .012, `one step a frame ${coarse} vs refined ${refined}`);
});

test('the contact tables hold a roster\'s assemblies and pieces in piece order, and fail explicitly past their capacity', () => {
  const scene = createPaddleboatsScene(), layout = rigidContactTableLayout(GPU_RIGID_BODY_CAPACITY);
  const motors = rigidMotorRecords(scene.rigidBodies, scene.rigidBodies.map(b => b.motorAttachment ? { signature: '', strength: 1, phaseOffset_rad: 0 } : undefined));
  const count = scene.rigidBodies.length, header = rigidContactTableHeader(motors, count, GPU_RIGID_BODY_CAPACITY), w = rigidMirrorWorld(scene);
  const assemblies = rigidAssemblyBuild(w).flatMap((bound, index) => bound > 0 ? [index] : []);
  assert.deepEqual([header[0], ...assemblies.map((_, slot) => header[4 * (1 + slot)])], [assemblies.length, ...assemblies]);
  for (let index = 0; index < GPU_RIGID_BODY_CAPACITY; index++) {
    const at = 4 * (layout.slots + 2 * index), pieces = index < count ? Array.from({ length: 2 + count }, (_, piece) => piece).filter(piece => rigidPieceExists(w, index, piece)) : [];
    assert.deepEqual([...header.slice(at, at + pieces.length)], pieces, `body ${index}'s pieces`);
    assert.equal(header[at + 7], pieces.length);
    assert.equal(header[at + 6], assemblies.indexOf(index), 'the assembly slot, -1 for an ordinary body');
  }
  assert.equal(rigidContactTableThreads(GPU_RIGID_BODY_CAPACITY, 2), layout.soloThreads + 2 * layout.pairThreads);
  // Five hulls: one more assembly than the tables hold.
  const five = Array.from({ length: 5 }, () => [...motors.slice(0, 32)]).flat();
  assert.throws(() => rigidContactTableHeader(five, 5, GPU_RIGID_BODY_CAPACITY), /hold 4 motor assemblies/);
  // A fixture whose parent is itself a fixture has no owner thread.
  const chained = [...motors.slice(0, 64), ...motors.slice(32, 64)]; chained[64 + 3] = 2;
  assert.throws(() => rigidContactTableHeader(chained, 3, GPU_RIGID_BODY_CAPACITY), /not a free body/);
});

test('the tabulated axis reaches bound a piece\'s reach along any direction', () => {
  // The kernel skips a static surface the bound proves unreachable this step.
  const { world: w } = boats(1, hull => { hull.position_m = { x: 0, y: .6, z: 0 }; });
  let random = 11; const next = () => (random = (random * 16807) % 2147483647) / 2147483647;
  const axes: V3[] = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  for (let trial = 0; trial < 300; trial++) {
    const q: V4 = [next() - .5, next() - .5, next() - .5, next() - .5], l = Math.hypot(...q), d: V3 = [next() - .5, next() - .5, next() - .5], dl = Math.hypot(...d);
    const body = { ...w.bodies[0]!, orientation: q.map(v => v / l) as V4 }, direction = d.map(v => v / dl) as V3;
    for (let piece = 0; piece < 2 + w.bodies.length; piece++) {
      if (!rigidPieceExists(w, 0, piece)) continue;
      const bound = axes.reduce((sum, e, a) => sum + Math.abs(direction[a]!) * rigidPieceReach(w, body, piece, e.map(v => v * Math.sign(direction[a]!)) as V3), 0);
      assert.ok(rigidPieceReach(w, body, piece, direction) <= bound + 1e-9, `piece ${piece}: ${rigidPieceReach(w, body, piece, direction)} > ${bound}`);
    }
  }
});

test('unknown kinds fail explicitly', () => {
  assert.throws(() => mirrorShapeSupportRadius(99, [1, 1, 1], [1, 0, 0]), /Unknown rigid shape code 99/);
  assert.equal(mirrorShapeSupportRadius(-2, [1, 1, 1], [1, 0, 0]), 0, 'fluid-only interior boundaries reach no surface');
  assert.throws(() => rigidAxleEnvelope(createPaddleboatsScene().rigidBodies[0]!), /no motor attachment/);
});

test('every kernel function has its TypeScript twin of the same name', () => {
  const kernel = rigidAssemblyContactWGSL(24), source = readFileSync(new URL('../lib/core/rigid-assembly-contact.ts', import.meta.url), 'utf8');
  const names = [...kernel.matchAll(/\bfn (\w+)\(/g)].map(m => m[1]!);
  assert.ok(names.length > 20);
  const missing = names.filter(name => !new RegExp(`(function ${name}\\(|const ${name} = )`).test(source));
  assert.deepEqual(missing.filter(name => !['rigidIsAssembly', 'rigidContactRadius', 'rigidPairUsesAssembly', 'contactReach', 'staticSurfaceContact'].includes(name)), [],
    'a WGSL function without a mirror is untested arithmetic');
  const solid = rigidAssemblySolidWGSL({ cells: [512, 32, 192], origin_m: [-6.4, 0, -2.4], cellSize_m: [.025, .025, .025],
    walls: { policy: 2, origin: [1, 1, 1], pages: [16, 1, 6] }, floor: false }, 'rigidBoundary', 24);
  const solidNames = [...solid.matchAll(/\bfn (\w+)\(/g)].map(m => m[1]!);
  assert.ok(solidNames.includes('assemblyVoxelContact'));
  // The mask readers are solidOccupancyMaskWGSL's, twinned by solidOccupancyMaskPage/Voxel.
  assert.deepEqual(solidNames.filter(name => !new RegExp(`(function ${name}\\(|const ${name} = )`).test(source))
    .filter(name => !['rigidMaskPageWall', 'rigidMaskPage', 'rigidMaskVoxel'].includes(name)), [], 'the assembly solid contact has its mirror');
});
