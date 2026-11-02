import assert from "node:assert/strict";
import test from "node:test";

/**
 * A worker that throws is retired with its tile. The regression: the pool kept
 * handing tiles to a dead worker, the tile it held never came back, and every
 * worker that saw the same failure reported it again.
 */
const workers: FakeWorker[] = [];
class FakeWorker {
  readonly listeners = new Map<string, (event: { message?: string; data?: unknown }) => void>();
  readonly posted: { type: string }[] = [];
  terminated = false;
  constructor() { workers.push(this); }
  addEventListener(type: string, listener: (event: { message?: string; data?: unknown }) => void) { this.listeners.set(type, listener); }
  postMessage(message: { type: string }) { this.posted.push(message); }
  terminate() { this.terminated = true; }
}
Object.assign(globalThis, { Worker: FakeWorker });
const { ShapeLabPool } = await import("../shape-lab/pool");

test("a worker that throws is retired, and the render fails once", () => {
  const errors: string[] = [];
  const pool = new ShapeLabPool(2, () => {}, () => {}, message => errors.push(message));
  pool.render({ sceneId: "s", descriptors: [], colors: [], width: 256, height: 256,
    camera: {} as never, shading: {} as never, showGround: false });
  const [first, second] = workers;
  first!.listeners.get("error")!({ message: "boom" });
  second!.listeners.get("error")!({ message: "boom" });
  assert.equal(first!.terminated, true);
  assert.deepEqual(errors, ["boom"]);
  const tiles = (worker: FakeWorker) => worker.posted.filter(message => message.type === "tile").length;
  const before = tiles(first!) + tiles(second!);
  pool.render({ sceneId: "s", descriptors: [], colors: [], width: 64, height: 64,
    camera: {} as never, shading: {} as never, showGround: false });
  assert.equal(tiles(first!) + tiles(second!), before);
  pool.dispose();
});
