import assert from 'node:assert/strict';
import test from 'node:test';
import { sampleSolidNodes, validateProceduralSolid } from '../lib/core/procedural-solid';
import { sampleRigidPoseTrack } from '../lib/core/prescribed-rigid-motion';
import { prepareClothTopology, validateClothDocument } from '../lib/cloth/topology';
import {
  BRIDSON_2002_FIGURE_4, BRIDSON_2002_PARAMETERS, BRIDSON_2002_SCENES, BRIDSON_2002_STEP_S, BRIDSON_2002_VIDEO_FRAME_RATE, CLOTH, CURTAIN, SPHERE, TABLE, VOXEL,
  assertBridsonSolidResolved, bakeBridsonStaticSolids, bridson2002Curtain, bridson2002RotatingSphere, bridson2002Tablecloth,
  bridsonLatticeSurfaceError, bridsonShapeDistance, bridsonSolidDistance, bridsonSolidPose, bridsonStaticLattice, herringBoneSheet,
  type BridsonSolid, type BridsonSolidShape, type Bridson2002Scene,
} from '../lib/cloth/scenes/bridson-2002';

const curtain = bridson2002Curtain(), sphere = bridson2002RotatingSphere(), table = bridson2002Tablecloth();
const scenes = [curtain, sphere, table];
const solid = (s: Bridson2002Scene, id: string) => s.solids.find(b => b.id === id)!;
const near = (a: number, b: number, tol: number, what: string) => assert.ok(Math.abs(a - b) <= tol, `${what}: ${a} vs ${b}`);
const positions = (s: Bridson2002Scene) => s.cloth.geometry.data.restPositions_m;

test('every Bridson 2002 parameter names its source', () => {
  const sources = new Set(['paper', 'thesis', 'video', 'physbam', 'assumption']);
  for (const [group, table] of Object.entries(BRIDSON_2002_PARAMETERS)) for (const [name, p] of Object.entries(table)) {
    assert.ok(sources.has(p.source), `${group}.${name} source`);
    assert.ok(p.ref.length > 10 && Number.isFinite(p.value) && p.unit.length > 0, `${group}.${name} provenance`);
  }
  assert.equal(BRIDSON_2002_VIDEO_FRAME_RATE.value, 24);
  assert.equal(BRIDSON_2002_STEP_S, 1 / 60);
  assert.equal(BRIDSON_2002_FIGURE_4.referenceOnly, true);
});

test('Bridson 2002 cloth documents are valid and carry the paper material', () => {
  for (const s of scenes) {
    assert.deepEqual(validateClothDocument(s.cloth), [], s.id);
    const m = s.cloth.materials[0]!;
    assert.equal(m.strainLimit, 0.1); assert.equal(m.thickness_m, 0.001);
    assert.equal(s.cloth.materials.length, 1);
    assert.equal(s.step_s, 1 / 60);
    assert.equal(s.cloth.scripts, undefined, 'pins hold still: no attachment scripts');
  }
});

test('Bridson 2002 sheets: node counts and herring-bone layout', () => {
  assert.deepEqual([curtain.sheet.columns, curtain.sheet.rows], [254, 150]);
  for (const s of [sphere, table]) assert.deepEqual([s.sheet.columns, s.sheet.rows], [150, 150]);
  for (const s of scenes) {
    const { columns: m, rows: n, width_m, depth_m } = s.sheet, data = s.cloth.geometry.data, p = positions(s);
    assert.equal(p.length, 3 * m * n); assert.equal(data.triangles.length, 6 * (m - 1) * (n - 1));
    // Vertex (i, j) is index i + m j, on a flat lattice of nearly square cells.
    const dx = width_m / (m - 1), dz = depth_m / (n - 1);
    near(dx / dz, 1, 0.002, `${s.id} cell aspect`);
    for (const [i, j] of [[0, 0], [m - 1, 0], [0, n - 1], [m - 1, n - 1], [7, 11]]) {
      const v = i! + m * j!;
      near(p[3 * v]! - p[0]!, i! * dx, 1e-9, 'x'); near(p[3 * v + 2]! - p[2]!, j! * dz, 1e-9, 'z'); near(p[3 * v + 1]!, p[1]!, 0, 'flat');
    }
    // Herring-bone: six triangles at every interior vertex.
    const valence = new Uint8Array(m * n); for (const v of data.triangles) valence[v]!++;
    for (let j = 1; j < n - 1; j++) for (let i = 1; i < m - 1; i++) assert.equal(valence[i + m * j], 6);
  }
  const groups = curtain.cloth.geometry.data.groups!, m = curtain.sheet.columns, n = curtain.sheet.rows;
  assert.deepEqual(groups.flatMap((g, v) => g ? [v] : []), [0, m * (n - 1)], 'curtain pins its two x = 0 corners');
  assert.deepEqual([positions(curtain)[0], positions(curtain)[3 * m * (n - 1) + 2]], [0, 1]);
  for (const s of [sphere, table]) assert.ok(s.cloth.geometry.data.groups!.every(g => g === 0), `${s.id} is unpinned`);
});

test('Bridson 2002 masses: equal per node, as the paper states', () => {
  for (const s of scenes) {
    const { columns: m, rows: n, width_m, depth_m } = s.sheet;
    near(s.nodeMass_kg * m * n, width_m * depth_m * CLOTH.arealDensity.value, 1e-12, `${s.id} total mass`);
  }
  // Herring-bone cells give the area-lumped solver equal interior masses
  // (an alternating-diagonal grid gives a 2:1 checkerboard). The paper's
  // heavier hems still need C1's per-vertex mass.
  const small = herringBoneSheet({ columns: 9, rows: 7, width_m: 0.8, depth_m: 0.6, origin_m: [0, 0, 0] });
  const mass = prepareClothTopology(small, curtain.cloth.materials).mass_kg;
  const interior: number[] = []; for (let j = 1; j < 6; j++) for (let i = 1; i < 8; i++) interior.push(mass[i + 9 * j]!);
  for (const v of interior) near(v, 0.1 * 0.1, 1e-12, 'interior lumped mass');
  assert.ok(mass[0]! < interior[0]! / 2, 'area lumping lightens the hem the paper makes heavy');
});

test('Bridson 2002 geometry matches its provenance', () => {
  // Fig 1: 1.7 x 1 m curtain at 0.5 m; a 0.25 m dome centred on the ground under the pinned edge.
  assert.deepEqual([curtain.sheet.width_m, curtain.sheet.depth_m, positions(curtain)[1]], [1.7, 1, 0.5]);
  const ball = solid(curtain, 'ball');
  assert.deepEqual(ball.shape, { kind: 'sphere', radius_m: 0.25 });
  assert.deepEqual(bridsonSolidPose(ball, 0).position_m, { x: 0, y: 0, z: 0.5 });
  assert.deepEqual([solid(curtain, 'ground').friction, ball.friction], [0.3, 0]);
  // Fig 2: 1.9 m sheet centred 1 cm above a 0.25 m sphere floating 5 cm clear; frustum standing on the ground.
  const s = solid(sphere, 'sphere'), f = solid(sphere, 'frustum');
  near(positions(sphere)[0]!, -0.95, 1e-12, 'sheet centred'); near(positions(sphere)[1]! - (0.3 + 0.25), 0.01, 1e-12, 'sheet gap');
  near(bridsonSolidPose(s, 0).position_m.y - 0.25, 0.05, 1e-12, 'sphere clearance');
  near(bridsonSolidDistance(f, { x: -0.65, y: 0, z: 0.65 }), 0, 1e-12, 'frustum base on the ground');
  near(bridsonSolidDistance(f, { x: -0.65, y: 0.24, z: 0.65 }), 0, 1e-12, 'frustum height');
  assert.deepEqual([s.friction, f.friction, solid(sphere, 'ground').friction], [1, 0.1, 0.1]);
  // Fig 3: square table of four 0.75 m posts, 0.48 m apart; the cloth starts 1 cm above them, its corners overhanging.
  const legs = table.solids.filter(b => b.id.startsWith('leg-'));
  assert.equal(legs.length, 4);
  const centres = legs.map(l => bridsonSolidPose(l, 0).position_m);
  assert.deepEqual(new Set(centres.map(c => `${Math.abs(c.x)},${Math.abs(c.z)},${c.y}`)), new Set(['0.24,0.24,0']));
  for (const l of legs) {
    const c = bridsonSolidPose(l, 0).position_m;
    near(bridsonSolidDistance(l, { x: c.x, y: TABLE.legHeight.value, z: c.z }), 0, 1e-12, 'leg top');
    // The bevel removes the 22 mm corner of the top edge.
    assert.ok(bridsonSolidDistance(l, { x: c.x + 0.055 - 0.005, y: 0.75 - 0.005, z: c.z }) > 0, 'chamfered edge');
    assert.ok(bridsonSolidDistance(l, { x: c.x + 0.055 - 0.005, y: 0.75 - 0.03, z: c.z }) < 0, 'below the chamfer');
  }
  near(positions(table)[1]! - TABLE.legHeight.value, 0.01, 1e-12, 'cloth gap');
  near(table.sheet.width_m, 1.59, 0, 'cloth side');
  // Video f0: the corners rest on the ground. From a leg's outer corner the diagonal overhang, stretched by at most 10%, spans the leg height.
  const overhang = Math.SQRT2 * (table.sheet.width_m - TABLE.legSpacing.value - TABLE.legWidth.value) / 2;
  assert.ok(overhang < TABLE.legHeight.value && overhang * 1.1 > TABLE.legHeight.value, `corner overhang ${overhang}`);
  assert.equal(TABLE.legSpacing.value - TABLE.legWidth.value > 2 * TABLE.sphereRadius.value, true, 'the sphere fits between two legs');
});

test('Bridson 2002 voxel solids resolve their smallest features at 1 cm', () => {
  const cell = VOXEL.cell.value, shapes = new Map<string, string>();
  for (const s of scenes) {
    assert.equal(s.voxelCell_m, cell);
    for (const b of s.solids) {
      assertBridsonSolidResolved(b, cell, VOXEL.minimumFeatureCells.value);
      if (b.shape.kind !== 'box') shapes.set(JSON.stringify(b.shape), `${s.id}/${b.id}`);
    }
  }
  // Sharp edges round off inside the cell they cross; nothing is lost beyond half a cell.
  for (const [shape, name] of shapes) {
    const error = bridsonLatticeSurfaceError(JSON.parse(shape) as BridsonSolidShape, cell, 2);
    assert.ok(error < 0.5 * cell, `${name}: stored surface within half a cell (${error})`);
  }
  // A 3 cm lattice loses the 22 mm bevel: authoring refuses it rather than dropping the feature.
  assert.throws(() => assertBridsonSolidResolved(solid(table, 'leg-0'), 0.03, 2), /narrower than 2 cells/);
  assert.throws(() => bridsonShapeDistance({ kind: 'cone' } as unknown as BridsonSolidShape, { x: 0, y: 0, z: 0 }), /Unknown Bridson 2002 solid shape kind cone/);
  const moving = { id: 'x', friction: 0, shape: { kind: 'sphere', radius_m: 1 }, motion: { kind: 'orbit' } } as unknown as BridsonSolid;
  assert.throws(() => bridsonSolidPose(moving, 0), /Unknown Bridson 2002 solid motion kind orbit/);
});

test('Bridson 2002 static solids bake into the shared voxel solid', () => {
  // The lattice spans the static solids; the ground slab occupies its lowest bricks.
  const lattice = bridsonStaticLattice(table.solids, table.voxelCell_m);
  assert.ok(lattice.dimensions.every(d => d % 8 === 0));
  assert.ok(lattice.origin_m.y <= -VOXEL.groundDepth.value && lattice.origin_m.y > -VOXEL.groundDepth.value - 0.08 - 1e-9, 'ground slab sits in the lowest bricks');
  near(-lattice.origin_m.y / 0.08, Math.round(-lattice.origin_m.y / 0.08), 1e-9, 'brick-aligned origin');
  const legsOnly = table.solids.filter(b => b.id.startsWith('leg-'));
  const { solidVoxels, solidBoundary } = bakeBridsonStaticSolids(legsOnly, bridsonStaticLattice(legsOnly, table.voxelCell_m));
  validateProceduralSolid(solidBoundary);
  assert.ok(solidVoxels.length > 0 && solidBoundary.bricks.length > 0);
  // The stored field puts each leg top at 0.75 m and the bevel within half a cell.
  const legLattice = bridsonStaticLattice(legsOnly, table.voxelCell_m), h = legLattice.cell_m;
  const bricks = new Map(solidBoundary.bricks.map(b => [b.origin.join(), b.values]));
  const phi = (x: number, y: number, z: number) => {
    const q = [(x - legLattice.origin_m.x) / h, (y - legLattice.origin_m.y) / h, (z - legLattice.origin_m.z) / h];
    const origin = q.map(v => 8 * Math.floor(v / 8)), values = bricks.get(origin.join());
    return values ? sampleSolidNodes(values, q.map((v, a) => v - origin[a]!) as [number, number, number])[3] * h : undefined;
  };
  near(phi(0.24, 0.75 - 0.005, 0.24)!, -0.005, 0.5 * h, 'inside the top');
  near(phi(0.24, 0.75 + 0.005, 0.24)!, 0.005, 0.5 * h, 'above the top');
});

test('Bridson 2002 motion scripts keep the published timings', () => {
  // Fig 1, PhysBAM test 5 Set_Kinematic_Positions, sampled exactly.
  const ball = solid(curtain, 'ball'), at = (t: number) => bridsonSolidPose(ball, t).position_m;
  const physbam = (t: number) => t < 2 ? [0, 0, 0.5] : t < 3.5 ? [t - 2, 0.5 * (t - 2), 0.5] : t < 4 ? [1.5, 0.75 - 1.5 * (t - 3.5), 0.5] : [1.5 - 1.5 * (t - 4), 0, 0.5];
  for (let t = 0; t <= curtain.duration_s; t += 1 / 24) {
    const p = at(t), q = physbam(t);
    near(p.x, q[0]!, 1e-9, `ball x at ${t}`); near(p.y, q[1]!, 1e-9, `ball y at ${t}`); near(p.z, q[2]!, 1e-12, 'ball z');
  }
  near(curtain.duration_s, 160 / 24, 1e-12, 'curtain duration');
  // Fig 2: still until 0.75 s, then -pi/4 rad/s about +y.
  const spin = solid(sphere, 'sphere');
  const yaw = (t: number) => { const q = bridsonSolidPose(spin, t).orientation; return 2 * Math.atan2(q.y, q.w); };
  near(yaw(0.5), 0, 1e-12, 'no spin before 0.75 s');
  for (const t of [1, 2.3, 4.75, 8]) near(yaw(t + 0.1) - yaw(t), -Math.PI / 40, 1e-9, `spin rate at ${t}`);
  assert.ok(Math.abs(bridsonSolidPose(spin, 5).position_m.y - 0.3) < 1e-12);
  near(sphere.duration_s, 213 / 24, 1e-12, 'sphere duration');
  // Fig 3: descends at 0.82 m/s through y = 2.10 at video frame 0, lands at f57, then leaves along +x at 0.62 m/s.
  const drop = solid(table, 'sphere'), y = (t: number) => bridsonSolidPose(drop, t).position_m;
  near(y(table.videoStart_s).y, 2.1, 1e-9, 'height at video f0');
  near((y(table.videoStart_s + 0.5).y - y(table.videoStart_s + 1).y) / 0.5, 0.82, 1e-9, 'descent speed');
  const land = drop.motion.kind === 'pose-track' ? drop.motion.track.keyframes[1]!.time_s : NaN;
  near((land - table.videoStart_s) * 24, 57, 1, 'lands at video frame 57');
  near(y(land).y, 0.15, 1e-12, 'resting on the ground');
  near((y(land + 1).x - y(land).x), 0.62, 1e-9, 'rolls out at 0.62 m/s');
  near(table.duration_s - table.videoStart_s, 93 / 24, 1e-12, 'table video duration');
  // Moving solids are prescribed by RigidPoseTrack and agree with the scene's own sampler.
  for (const s of scenes) for (const b of s.solids) if (b.motion.kind === 'pose-track')
    assert.deepEqual(sampleRigidPoseTrack(b.motion.track, 1.25).position_m, bridsonSolidPose(b, 1.25).position_m);
});

test('Bridson 2002 key moments sit on video frames inside each scene', () => {
  const frames = { 'bridson-2002-curtain': 160, 'bridson-2002-rotating-sphere': 213, 'bridson-2002-tablecloth': 93 } as Record<string, number>;
  for (const s of scenes) {
    let last = -1;
    for (const k of s.keyMoments) {
      assert.ok(k.videoFrame >= 0 && k.videoFrame < frames[s.id]!, `${s.id} f${k.videoFrame}`);
      near(k.time_s, s.videoStart_s + k.videoFrame / 24, 1e-12, 'video time');
      assert.ok(k.time_s > last && k.time_s <= s.duration_s);
      last = k.time_s;
    }
  }
  assert.equal(BRIDSON_2002_SCENES.length, 3);
  assert.equal(CURTAIN.ballRadius.source, 'physbam'); assert.equal(SPHERE.frustumBase.source, 'video'); assert.equal(TABLE.legChamfer.source, 'video');
});
