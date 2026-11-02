import assert from "node:assert/strict";
import test from "node:test";

import { OctreeSparseBrickWorld } from "../lib/svo/features/construction/webgpu-svo-sparse-bricks";

/** The private maintenance state these tests drive, seen through a prototype-built world. */
type World = Record<string, unknown> & {
  encodeSceneMaintenance(encoder: GPUCommandEncoder, physics: boolean): boolean;
  encodeLiveDerivedMaintenance(encoder: GPUCommandEncoder, seam: undefined, physics: boolean): boolean;
  setRadianceEnabled(enabled: boolean): void;
};
const pass = { end() {} };
const encoder = { beginComputePass: () => pass } as unknown as GPUCommandEncoder;

test("a scene publication the voxelizer refuses is not handed over again every frame", () => {
  let publishes = 0;
  const world = Object.assign(Object.create(OctreeSparseBrickWorld.prototype), {
    destroyed: false, topologyPublished: true, pendingTopologyRecords: 0,
    pendingScenePublication: { dirtyRegions: [] },
    proxyVoxelizer: { publish: () => { publishes += 1; throw new Error("refused"); }, encodeMaintenance: () => false },
  }) as unknown as World;
  assert.throws(() => world.encodeSceneMaintenance(encoder, true), /refused/);
  assert.equal(world.pendingScenePublication, undefined);
  assert.equal(world.encodeSceneMaintenance(encoder, true), false);
  assert.equal(publishes, 1);
});

test("live derived lighting never owes fewer frames than drain frames, so every drain resets its seed bound", () => {
  let activity = 0;
  const world = Object.assign(Object.create(OctreeSparseBrickWorld.prototype), {
    liveDerivedAddressPlanValid: true, radianceEnabled: true, liveDerivedInitial: true,
    liveDerivedFramesRemaining: 0, liveDerivedDrainFramesRemaining: 0, liveDerivedSeedBound: 0,
    liveDerivedChunkSeedBound: 64, liveDerivedSceneActivity: 0, liveDerivedRelight: false,
    liveDerivedPlanner: {
      drainFrames: (seeds: number) => Math.ceil(seeds / 32), stepDrainFrames: (seeds: number) => Math.ceil((seeds + 16) / 32),
      encodeReset() {}, encodeFrame() {}, encodeRound() {}, initialRounds: 1, roundsPerFrame: 1,
    },
    liveDerivedBuilder: { radianceFeedbackEnabled: true, encodeCertifyEmpty() {}, encodeRound() {} },
    proxyVoxelizer: { maintenanceBinding: { get activity() { return activity; } }, indexStatus: { regionCells: 0 } },
    tetrahedralRadiance: { storageEnabled: true, visibleGeneration: () => ({}) },
    sceneSource: { derivedLightingContent: { revision: 0, settled: false }, tetrahedralRadiance: {} },
    republishLiveDerivedOpacity() {},
  }) as unknown as World;
  const frame = (physics = false) => {
    world.encodeLiveDerivedMaintenance(encoder, undefined, physics);
    const frames = world.liveDerivedFramesRemaining as number, drain = world.liveDerivedDrainFramesRemaining as number;
    assert.ok(frames >= drain, `frames ${frames} < drain ${drain}`);
    if (drain === 0) assert.equal(world.liveDerivedSeedBound, 0);
  };
  frame();
  // Edits journal leaves across several voxelizer frames; one toggles radiance mid-drain.
  for (let step = 0; step < 40; step += 1) {
    if (step % 7 === 0) activity += 3;
    if (step === 10) world.setRadianceEnabled(false);
    if (step === 14) world.setRadianceEnabled(true);
    if (step === 20) world.liveDerivedRelight = true;
    frame(step % 5 === 0);
  }
  for (let step = 0; step < 200 && (world.liveDerivedFramesRemaining as number) > 0; step += 1) frame();
  assert.equal(world.liveDerivedFramesRemaining, 0);
  assert.equal(world.liveDerivedDrainFramesRemaining, 0);
  assert.equal(world.liveDerivedSeedBound, 0);
});
