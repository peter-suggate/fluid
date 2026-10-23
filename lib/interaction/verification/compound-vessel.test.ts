import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createEmptyScene } from '../../core/empty-scene';
import { interactionCatalog } from '../installed';
import { sealedVessel, vesselRuntime } from '../features/sealed-vessel/definition';
import { Box3DWorld } from '../adapters/box3d/world';
import type { FluidExchange } from '../contracts';

async function createVessel(panelProxies = false) {
  const scene = createEmptyScene({ extents_m: { x: 8, y: 6, z: 8 }, finestCellSize_m: .25 });
  scene.scenery = undefined;
  const entity = interactionCatalog.createEntity({ id: 'vessel', kind: 'fluid.sealed-vessel',
    transform: { position_m: { x: 0, y: 3, z: 0 }, orientation: { w: 1, x: 0, y: 0, z: 0 } } });
  const recipe = vesselRuntime.body!(entity);
  if (panelProxies) delete recipe.fluidInterior_m;
  return await Box3DWorld.create(scene, [{ entity, recipe }], 7);
}
const close = (actual: number, expected: number) => assert.ok(Math.abs(actual-expected) < 1e-5, `${actual} != ${expected}`);

test('sealed vessel is one native six-shape body with hollow mass and an interior lid', async () => {
  const world = await createVessel();
  try {
    assert.equal(world.bodies.size, 1);
    const native = world.bodies.get('vessel')!.native;
    assert.equal(native.getShapeCount(), 6);
    const outer = 1.6*1.44*1.6, inner = 1.28*1.12*1.28;
    assert.ok(Math.abs(native.getMass()-600*(outer-inner)) < .001, 'mass comes from the six walls, not a solid bounding box');
    assert.deepEqual(native.getWorldCenterOfMass(), native.getPosition());
    const origin = native.getPosition();
    for (const [direction, distance] of [
      [{ x: 1, y: 0, z: 0 }, .64], [{ x: -1, y: 0, z: 0 }, .64],
      [{ x: 0, y: 1, z: 0 }, .56], [{ x: 0, y: -1, z: 0 }, .56],
      [{ x: 0, y: 0, z: 1 }, .64], [{ x: 0, y: 0, z: -1 }, .64],
    ] as const) {
      const hit = world.ray(origin, direction, 2);
      assert.equal(hit?.entityId, 'vessel'); close(hit!.distance, distance);
    }
    assert.equal(world.snapshots().length, 1);
    assert.equal(world.renderBodies().length, 1);
    assert.equal(world.renderBodies()[0]!.description.fluidBoundary, "interior");
  } finally { world.destroy(); }
});

test('wall pressure impulses include their moment arms and use the compound inertia exactly once', async () => {
  const world = await createVessel(true);
  const identity = { generation: 7, tick: 1, solidRevision: 1 };
  const packets: FluidExchange[] = world.renderBodies().map(b => ({ ...identity, bodyId: b.description.id,
    bodyGeneration: world.fluidBodyGenerations()[b.description.id]!,
    linearImpulse_N_s: { x: b.description.id.endsWith('/lid') ? 100 : b.description.id.endsWith('/floor') ? -100 : 0, y: 0, z: 0 },
    angularImpulse_N_m_s: { x: b.description.id.endsWith('/front') ? 40 : 0, y: 0, z: 0 },
    displacedVolume_m3: 0, fluidVelocity_m_s: { x: 0, y: 0, z: 0 }, policy: 'pressure-traction' }));
  try {
    for (const invalid of [packets.slice(0, 5), [...packets.slice(0, 5), packets[0]!],
      [...packets.slice(0, 5), { ...packets[5]!, bodyGeneration: 6 }]]) {
      assert.throws(() => world.applyFluid(invalid, identity), /Incomplete|duplicate|stale/);
      assert.deepEqual(world.snapshots()[0]!.angularVelocity_rad_s, { x: 0, y: 0, z: 0 });
    }
    world.applyFluid(packets, identity);
    const actual = world.snapshots()[0]!;
    assert.deepEqual(actual.linearVelocity_m_s, { x: 0, y: 0, z: 0 }, 'a force couple has no net linear impulse');
    const outer = 1.6*1.44*1.6, inner = 1.28*1.12*1.28;
    const inertia = 600/12*(outer*(1.44**2+1.6**2)-inner*(1.12**2+1.28**2));
    close(actual.angularVelocity_rad_s.x, 40/inertia);
    close(actual.angularVelocity_rad_s.z, -128/inertia);
    assert.throws(() => world.applyFluid(packets, identity), /already applied/);
    assert.deepEqual(world.snapshots()[0]!.angularVelocity_rad_s, actual.angularVelocity_rad_s);
  } finally { world.destroy(); }
});

test('vessel configuration rejects a wall thickness that closes its cavity', () => {
  assert.throws(() => interactionCatalog.createEntity({ id: 'invalid', kind: 'fluid.sealed-vessel',
    config: { ...sealedVessel.schema.defaults, wallThickness: .72 }, transform: { position_m: { x: 0, y: 3, z: 0 }, orientation: { w: 1, x: 0, y: 0, z: 0 } } }), /interior/);
});
