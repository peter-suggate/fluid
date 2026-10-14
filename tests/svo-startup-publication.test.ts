import assert from "node:assert/strict";
import test from "node:test";
import { OctreeSparseBrickWorld } from "../lib/svo/features/construction/webgpu-svo-sparse-bricks";

function startupWorld() {
  const firstFence = Promise.withResolvers<void>();
  const passes: boolean[] = [];
  let submissions = 0;
  const world = Object.assign(Object.create(OctreeSparseBrickWorld.prototype), {
    destroyed: false,
    pendingScenePublication: { budgeted: false } as { budgeted: boolean } | undefined,
    proxyVoxelizer: { budgetStatus: { plannedChunks: 2, converging: true } },
    device: {
      createCommandEncoder: () => ({ finish: () => ({}) }),
      queue: {
        submit: () => { submissions++; },
        onSubmittedWorkDone: () => submissions === 1 ? firstFence.promise : Promise.resolve(),
      },
    },
    encodeSceneMaintenance(_encoder: unknown, deferDerived = false) {
      if (this.pendingScenePublication) assert.equal(this.pendingScenePublication.budgeted, true);
      this.pendingScenePublication = undefined;
      passes.push(deferDerived);
      this.proxyVoxelizer.budgetStatus.converging = passes.length < 2;
      return true;
    },
  });
  return { world: world as OctreeSparseBrickWorld, passes, firstFence };
}

test("startup fences each voxelization batch and builds lighting after the full publication", async () => {
  const { world, passes, firstFence } = startupWorld();
  const progress: number[] = [];
  const ready = world.initializeScenePublication(undefined, completed => progress.push(completed));
  await new Promise<void>(resolve => setImmediate(resolve));
  assert.deepEqual(passes, [true], "a pending fence must prevent the next batch");
  assert.deepEqual(progress, [], "only completed batches are reported");
  firstFence.resolve();
  await ready;
  assert.deepEqual(passes, [true, true, false], "lighting must follow both voxelization batches");
  assert.deepEqual(progress, [1, 2]);
});

test("cancelling startup waits for its submitted batch without submitting more work", async () => {
  const { world, passes, firstFence } = startupWorld();
  const abort = new AbortController();
  const ready = world.initializeScenePublication(abort.signal);
  abort.abort();
  firstFence.resolve();
  await assert.rejects(ready, { name: "AbortError" });
  assert.deepEqual(passes, [true]);
});
