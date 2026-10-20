import assert from 'node:assert/strict';
import test from 'node:test';
import { createBackyardPoolPartyScene, createBackyardPoolDriftScene } from '../../core/backyard-pool-party-scene';
import { parseScene, validateScene } from '../../core/model';
import { SceneInteractionHost } from '../runtime/host';
import { InteractionInputRouter } from '../runtime/input';
import { interactionScene } from '../adapters/scene';
import { interactionCatalog } from '../installed';
import { BUB_PARTS, robotRenderExtent } from '../features/robot/render';

const steps = async (host: SceneInteractionHost, n: number) => { for (let i = 0; i < n; i++) await host.advance(1 / 60); };
async function possess(host: SceneInteractionHost) {
  const view = host.getView();
  const result = await host.dispatch({ id: 'control-bub', generation: view.generation, documentRevision: view.documentRevision,
    command: { kind: 'possess', entityId: 'pool-robot' } });
  assert.equal(result.accepted, true, JSON.stringify(result));
}
test('Party Pool ships a serializable Blender robot and a bounded ladder; drift stays passive', () => {
  const scene = createBackyardPoolPartyScene();
  assert.deepEqual(validateScene(scene), []);
  assert.deepEqual(parseScene(JSON.stringify(scene)).interaction, scene.interaction);
  assert.equal(scene.interaction!.entities[0].kind, 'fluid.robot');
  const capsule = interactionScene(scene).rigidBodies.find(b => b.appearance === 'bub')!;
  assert.equal(capsule.shape, 'capsule');
  assert.equal(capsule.dimensions_m.x, .38);
  assert.ok(BUB_PARTS.length > 15);
  assert.equal(createBackyardPoolDriftScene().interaction, undefined);
  const bad = structuredClone(scene.interaction!);
  assert.throws(() => interactionCatalog.inspectDocument({ ...bad, ladders: [{ ...bad.ladders![0], top_m: bad.ladders![0].bottom_m }] }), /ladder dimensions/);
});

test('Bub climbs the actual pool ladder, walks the board, flips once and falls into the basin', async () => {
  const scene = createBackyardPoolPartyScene();
  scene.systems = { fluid: false }; scene.scenery = undefined;
  const host = new SceneInteractionHost(scene);
  let input: InteractionInputRouter | undefined;
  const body = () => host.getView().snapshot!.bodies.find(b => b.entityId === 'pool-robot')!;
  try {
    await host.ready; await steps(host, 20); await possess(host);
    input = new InteractionInputRouter(host);
    // Bub spawns facing +x beside the ladder: a quarter turn left faces it at the ladder's foot.
    input.key('ArrowLeft', true); await input.settled(); await steps(host, 36);
    input.key('ArrowLeft', false); await input.settled();
    input.key('ArrowUp', true); await input.settled(); await steps(host, 25);
    input.key('ArrowUp', false); await input.settled(); await steps(host, 15);
    assert.ok(Math.abs(body().transform.position_m.z) < .5, JSON.stringify(body().transform.position_m));
    input.key('Space', true); await input.settled(); await steps(host, 170);
    input.key('Space', false); await input.settled(); await steps(host, 35);
    assert.ok(body().transform.position_m.y > 5.8, `ladder landing: ${JSON.stringify(body().transform.position_m)}`);
    assert.ok(body().grounded, 'lands on the actual board');
    // The ladder leaves Bub facing along the board.
    input.key('ArrowUp', true); await input.settled(); await steps(host, 82);
    const takeoff = body().transform.position_m;
    assert.ok(takeoff.x > -3.4 && takeoff.x < -2.5, JSON.stringify(takeoff));
    input.key('KeyF', true); await input.settled(); await steps(host, 12);
    const render = host.currentBodies().find(b => b.description.appearance === 'bub')!;
    assert.ok(Math.abs(render.renderOrientation!.x) + Math.abs(render.renderOrientation!.z) > .3, 'visible somersault');
    assert.equal(body().transform.orientation.x, 0, 'physical capsule stays upright');
    assert.equal(body().transform.orientation.z, 0);
    assert.deepEqual(robotRenderExtent(render), [.38, .5, .38]);
    assert.ok(body().transform.position_m.y > takeoff.y + .3, 'F launches from board');
    await steps(host, 75);
    assert.ok(body().transform.position_m.x > -2.1, 'clears board tip');
    assert.ok(body().transform.position_m.y < 2, `enters pool: ${JSON.stringify(body().transform.position_m)}`);
    const finished = host.currentBodies().find(b => b.description.appearance === 'bub')!.renderOrientation!;
    assert.equal(finished.x, 0); assert.equal(finished.z, 0, 'held F cannot repeatedly flip');
  } finally { input?.dispose(); await host.dispose(); }
});

test('releasing control cancels a ladder assist and reset restores the robot pose', async () => {
  const scene = createBackyardPoolPartyScene(); scene.systems = { fluid: false }; scene.scenery = undefined;
  scene.interaction = { ...scene.interaction!, entities: scene.interaction!.entities.map(e => ({ ...e,
    transform: { ...e.transform, position_m: { x: -7.4, y: 2.52, z: 0 } } })) };
  const host = new SceneInteractionHost(scene);
  try {
    await host.ready; await possess(host); const input = new InteractionInputRouter(host);
    input.key('Space', true); await input.settled(); await steps(host, 40);
    const before = host.currentBodies().find(b => b.description.appearance === 'bub')!.position_m.y;
    input.reset(); await input.settled(); await steps(host, 70);
    assert.ok(host.currentBodies().find(b => b.description.appearance === 'bub')!.position_m.y < before);
    const view = host.getView(); await host.dispatch({ id: 'reset', generation: view.generation, documentRevision: view.documentRevision, command: { kind: 'transport', action: 'reset' } });
    assert.ok(Math.abs(host.currentBodies().find(b => b.description.appearance === 'bub')!.position_m.y - 2.52) < 1e-6);
    input.dispose();
  } finally { await host.dispose(); }
});
