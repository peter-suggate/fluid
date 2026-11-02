import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createEmptyScene } from '../../core/empty-scene';
import { interactionCatalog } from '../installed';
import { SceneInteractionHost } from '../runtime/host';
import { InteractionInputRouter } from '../runtime/input';
import type { CommandEnvelope, CommandResult, EntityRecord, InteractionCommand, InteractionView } from '../contracts';
import { createLiveInteractionBinding } from '../../features/interactive/live-binding';
import { InteractionPane } from '../../features/interactive/pane';
import { createSimulatedInteractionHost } from '../../features/interactive/simulated-host';
import { createDemoInteractionCatalog, DEMO_HOST_CAPABILITIES } from '../../features/interactive/demo/catalog';
import { DEMO_RUNTIME_FACTORIES } from '../../features/interactive/demo/runtime';
import type { SimulationController } from '../../core/simulation/controller';

/**
 * Who owns the interactive run state: the host's queue orders every write to
 * it, the context is derived from it, and nothing outlives a dispose.
 */
const transform = (x: number, y: number, z: number) => ({ position_m: { x, y, z }, orientation: { x: 0, y: 0, z: 0, w: 1 } });
const entity = (kind: string, id: string): EntityRecord =>
  interactionCatalog.createEntity({ kind, id, name: id, transform: transform(0, 1.5, 0) });
function scene(entities: EntityRecord[], fluid = false) {
  const s = createEmptyScene({ extents_m: { x: 8, y: 4, z: 8 }, finestCellSize_m: .25 });
  s.scenery = undefined; s.interaction = { schemaVersion: 1, entities, connections: [] };
  s.systems = { ...s.systems, fluid };
  return s;
}
let seq = 0;
async function send(h: SceneInteractionHost, command: InteractionCommand) {
  const v = h.getView();
  return h.dispatch({ id: `run-state:${++seq}`, generation: v.generation, documentRevision: v.documentRevision, command });
}
const readyFluid = { step: async (request: { bodies: unknown[] }) => { void request; return []; }, isReady: () => true };

test('a failed step while possessed ends play: control released, context derived', async () => {
  const h = new SceneInteractionHost(scene([entity('fluid.character', 'hero')], true), { fluid: readyFluid as never });
  try {
    await h.ready;
    assert.equal((await send(h, { kind: 'possess', entityId: 'hero' })).accepted, true);
    assert.equal(h.getView().context, 'play');
    await h.setFluidDriver({ ...readyFluid, step: async () => { throw new Error('fluid exchange failed'); } } as never);
    await assert.rejects(h.advance(1 / 60), /fluid exchange failed/);
    const view = h.getView();
    assert.equal(view.running, false);
    assert.equal(view.controlledEntityId, undefined);
    assert.equal(view.context, 'edit');
  } finally { await h.dispose(); }
});

test('play pressed during a rebuild lands after it rather than being undone', async () => {
  const h = new SceneInteractionHost(scene([entity('fluid.character', 'hero')]), { fluid: readyFluid as never });
  try {
    await h.ready;
    const reset = send(h, { kind: 'transport', action: 'reset' });
    const play = h.setRunning(true);
    await Promise.all([reset, play]);
    assert.equal(h.getView().running, true);
    assert.equal(h.getView().context, 'observe');
  } finally { await h.dispose(); }
});

test('nothing is published once the host is disposed', async () => {
  const h = new SceneInteractionHost(scene([entity('fluid.character', 'hero')]), { fluid: readyFluid as never });
  await h.ready;
  let published = 0;
  h.subscribe(() => { published += 1; });
  const rebuilding = h.replaceScene(scene([entity('fluid.character', 'hero')]), true).catch(() => undefined);
  const disposing = h.dispose();
  const before = published;
  await Promise.all([rebuilding, disposing]);
  assert.equal(published, before);
});

test('a refusal that answers after a newer reset leaves the newer input alone', async () => {
  let refuse!: (result: CommandResult) => void;
  let cleared = 0;
  const view: InteractionView = {
    generation: 1, documentRevision: 0, running: true, context: 'play', controlledEntityId: 'hero',
    readiness: { state: 'ready', reasons: [] }, controlCandidates: [],
    document: { schemaVersion: 1, entities: [entity('fluid.character', 'hero')], connections: [] },
  };
  const host = {
    getView: () => view,
    subscribe: () => () => {},
    dispatch: (_envelope: CommandEnvelope) => new Promise<CommandResult>(resolve => { refuse = resolve; }),
    clearInput: () => { cleared += 1; },
    releaseInput: async () => {},
  } as unknown as SceneInteractionHost;
  const router = new InteractionInputRouter(host);
  assert.equal(router.key('KeyW', true), true);
  await Promise.resolve(); await Promise.resolve();
  router.reset(false);
  assert.equal(cleared, 1);
  refuse({ accepted: false, reason: 'stale' });
  await router.settled();
  assert.equal(cleared, 1);
});

test('every command sent while the live host comes up is restamped; dispose does not reattach', async () => {
  let ready!: () => void;
  let generation = 0;
  const stamped: number[] = [];
  let subscriptions = 0;
  const fake = {
    ready: new Promise<void>(resolve => { ready = () => { generation = 3; resolve(); }; }),
    getView: () => ({ generation, documentRevision: 0 }) as InteractionView,
    subscribe: () => { subscriptions += 1; return () => {}; },
    dispatch: async (envelope: CommandEnvelope) => { stamped.push(envelope.generation); return { accepted: true, documentRevision: 0, impact: 'live' } as const; },
  };
  const controller = {
    session: () => ({ scene: { getState: () => ({ scene: { interaction: { entities: [{}] } } }) } }),
    interaction: () => fake,
  } as unknown as SimulationController;
  const binding = createLiveInteractionBinding('a', controller);
  const envelope = (id: string): CommandEnvelope => ({ id, generation: 0, documentRevision: 0, command: { kind: 'release-control' } });
  const first = binding.host.dispatch(envelope('first'));
  const second = binding.host.dispatch(envelope('second'));
  ready();
  await Promise.all([first, second]);
  assert.deepEqual(stamped, [3, 3]);
  binding.dispose!();
  const attached = subscriptions;
  binding.host.getView();
  assert.equal(subscriptions, attached);
});

test('busy holds while any request is in flight', async () => {
  const gates: (() => void)[] = [];
  const view = { generation: 1, documentRevision: 0, running: false, context: 'edit', controlCandidates: [],
    readiness: { state: 'ready', reasons: [] }, document: { schemaVersion: 1, entities: [], connections: [] } } as InteractionView;
  const host = {
    getView: () => view,
    subscribe: () => () => {},
    dispatch: () => new Promise<CommandResult>(resolve => gates.push(() => resolve({ accepted: false, reason: 'no such entity' }))),
  };
  const pane = new InteractionPane('a', { catalog: interactionCatalog, host });
  try {
    const first = pane.control('nobody');
    const second = pane.control('nobody');
    await new Promise(resolve => setTimeout(resolve));
    assert.equal(gates.length, 2);
    gates[0]!();
    await first;
    assert.equal(pane.state.busy, true);
    gates[1]!();
    await second;
    assert.equal(pane.state.busy, false);
  } finally { pane.dispose(); }
});

test('the simulated host gives every new world a new generation', async () => {
  const catalog = createDemoInteractionCatalog();
  const host = createSimulatedInteractionHost({ catalog, capabilities: DEMO_HOST_CAPABILITIES, runtimes: DEMO_RUNTIME_FACTORIES, autoTick: false });
  try {
    const kind = catalog.entities[0]!;
    const placed = catalog.createEntity({ id: 'one', kind: kind.id, name: 'one', transform: transform(0, 1, 0), config: kind.schema.defaults });
    const view = host.getView();
    const document = { schemaVersion: 1 as const, entities: [placed], connections: [] };
    const result = await host.dispatch({ id: 'import', generation: view.generation, documentRevision: view.documentRevision, command: { kind: 'import', document } });
    assert.equal(result.accepted, true, JSON.stringify(result));
    assert.equal(result.accepted && result.impact, 'reset');
    assert.equal(host.getView().generation, view.generation + 1);
  } finally { host.dispose(); }
});
