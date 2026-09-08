import "../lib/methods";
import assert from "node:assert/strict";
import test from "node:test";
import { simulation } from "../lib/core/simulation/controller";
import { cloneScene, defaultScene } from "../lib/core/model";
const settled = () => new Promise<void>(resolve => setImmediate(resolve));

test("voxel history awaits atomic acceptance and wet rejection preserves document and both stacks", async () => {
  const session = simulation.session();
  const base = cloneScene(defaultScene);
  const edited = { ...base, solidVoxels: [...base.solidVoxels,
    { operation: "fill" as const, minimum: [1, 0, 1] as [number, number, number], maximumExclusive: [2, 1, 2] as [number, number, number] }] };
  session.method.getState().setMethodId("adaptive-mass");
  session.ui.setState({ voxelStrokePending: false });
  session.scene.getState().setScene(edited, "test");
  session.history.getState().clear();
  session.history.getState().record({ scene: base, presetId: "test", label: "Wall" });
  const beforeMount = session.scene.getState().scene;
  assert.equal(simulation.undo(), false, "without a renderer history fails closed");
  assert.equal(session.scene.getState().scene, beforeMount);
  assert.equal(session.history.getState().past.length, 1);
  let resolve!: () => void, reject!: (error: Error) => void;
  const proposals: unknown[] = [];
  const dispose = simulation.registerLiveSolidEditAcceptance(session.id, async (next, current) => {
    proposals.push([next, current]);
    await new Promise<void>((yes, no) => { resolve = yes; reject = no; });
  });
  try {
    const initial = session.scene.getState().scene;
    assert.equal(simulation.undo(), true);
    assert.equal(session.ui.getState().voxelStrokePending, true);
    assert.equal(simulation.redo(), false, "history operations serialize with tool gestures");
    assert.equal(session.scene.getState().scene, initial);
    assert.equal(session.history.getState().past.length, 1);
    resolve(); await settled();
    assert.deepEqual(session.scene.getState().scene.solidVoxels, base.solidVoxels);
    assert.equal(session.history.getState().past.length, 0);
    assert.equal(session.history.getState().future.length, 1);
    const beforeRedo = session.scene.getState().scene;
    const history = session.history.getState();
    assert.equal(simulation.redo(), true);
    reject(new Error("Solid insertion overlaps moving water")); await settled();
    assert.equal(session.scene.getState().scene, beforeRedo);
    assert.equal(session.history.getState().past, history.past);
    assert.equal(session.history.getState().future, history.future);
    assert.equal(session.ui.getState().voxelStrokePending, false);
    assert.equal(proposals.length, 2);
    assert.equal(simulation.redo(), true);
    resolve(); await settled();
    assert.deepEqual(session.scene.getState().scene.solidVoxels, edited.solidVoxels);
    assert.equal(session.history.getState().future.length, 0);
  } finally { dispose(); session.ui.setState({ voxelStrokePending: false }); }
});

test("mirrored voxel publication restores the accepted document while the receiving pane checks water", async () => {
  const session = simulation.session();
  const previous = cloneScene(defaultScene);
  const next = { ...previous, solidVoxels: [...previous.solidVoxels,
    { operation: "fill" as const, minimum: [1, 0, 1] as [number, number, number], maximumExclusive: [2, 1, 2] as [number, number, number] }] };
  session.method.getState().setMethodId("adaptive-mass");
  session.ui.setState({ voxelStrokePending: false });
  let reject!: (error: Error) => void;
  const dispose = simulation.registerLiveSolidEditAcceptance(session.id, () => new Promise<void>((_yes, no) => { reject = no; }));
  try {
    session.scene.getState().setScene(next);
    assert.equal(simulation.adoptSceneEdit(previous), true);
    assert.equal(session.scene.getState().scene, previous);
    assert.equal(session.ui.getState().voxelStrokePending, true);
    reject(new Error("Solid insertion overlaps moving water")); await settled();
    assert.equal(session.scene.getState().scene, previous);
    assert.equal(session.ui.getState().voxelStrokePending, false);
  } finally { dispose(); }
});
