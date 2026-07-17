import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const source = readFileSync(new URL("../lib/webgpu-uniform-eulerian.ts", import.meta.url), "utf8");

test("uniform and adaptive wrapper telemetry reuse bounded MAP_READ pools", () => {
  const statsHelper = source.slice(source.indexOf("private statsReadback()"), source.indexOf("private rigidReadback()"));
  const rigidHelper = source.slice(source.indexOf("private rigidReadback()"), source.indexOf("private retireQuadtreeProjection"));
  const stats = source.slice(source.indexOf("async readStats()"), source.indexOf("\n  destroy()"));
  const advance = source.slice(source.indexOf("advanceTo(time_s"), source.indexOf("async readStats()"));

  assert.match(statsHelper, /this\.statsReadbackBuffer \?\?=/, "statistics lazily allocate one staging buffer");
  assert.match(rigidHelper, /this\.rigidReadbackPool\.find/, "rigid loads first reuse an idle staging slot");
  assert.match(rigidHelper, /this\.rigidReadbackPool\.push/, "overlapping asynchronous maps grow a reusable bounded pool");
  assert.match(rigidHelper, /size: 2 \* GPU_RIGID_EXCHANGE_BYTES/, "one pooled slot carries both dense and variational rigid reactions");
  assert.doesNotMatch(stats, /createBuffer\(/, "each statistics poll must not allocate a GPU buffer");
  assert.doesNotMatch(advance, /createBuffer\(/, "each rigid exchange must not allocate a GPU buffer");
  assert.match(advance, /encodeBodyImpulseReadback\(encoder, exchangeReadback\.buffer, GPU_RIGID_EXCHANGE_BYTES\)/, "quadtree impulses use the pooled slot's second region");
  assert.match(advance, /await mapPromise\.catch/, "a rejected impulse channel must wait for the pooled mapping to settle");
  assert.match(advance, /finally\(\(\) => \{ slot\.busy = false; \}\)/, "every rigid impulse map releases its own pool slot");
  assert.match(stats, /await mapPromise\.catch/, "a rejected diagnostic channel must wait for the pooled mapping to settle");
  assert.match(stats, /finally \{[\s\S]*this\.readbackPending = false/, "failed quadtree diagnostics release the pooled readback slot");
});
