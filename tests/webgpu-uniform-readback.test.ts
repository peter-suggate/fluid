import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const source = readFileSync(new URL("../lib/webgpu-uniform-eulerian.ts", import.meta.url), "utf8");

test("uniform and adaptive rigid coupling stays resident while telemetry remains pooled", () => {
  const statsHelper = source.slice(source.indexOf("private statsReadback()"), source.indexOf("private retireQuadtreeProjection"));
  const stats = source.slice(source.indexOf("async readStats()"), source.indexOf("\n  destroy()"));
  const advance = source.slice(source.indexOf("advanceTo(time_s"), source.indexOf("async readStats()"));

  assert.match(statsHelper, /this\.statsReadbackBuffer \?\?=/, "statistics lazily allocate one staging buffer");
  assert.doesNotMatch(stats, /createBuffer\(/, "each statistics poll must not allocate a GPU buffer");
  assert.doesNotMatch(advance, /createBuffer\(/, "each rigid exchange must not allocate a GPU buffer");
  assert.doesNotMatch(advance, /mapAsync|getMappedRange|rigidReadbackPool|encodeBodyImpulseReadback/, "the physics path must not map rigid feedback");
  assert.match(advance, /encodeBodyImpulseExchange\(encoder, this\.rigidExchangeBuffer\)/, "quadtree pressure impulses remain in GPU storage");
  assert.match(advance, /this\.rigidSystem\.encode\(encoder, delta/, "the resident rigid solver consumes the exchange in the same command stream");
  assert.match(stats, /await mapPromise\.catch/, "a rejected diagnostic channel must wait for the pooled mapping to settle");
  assert.match(stats, /finally \{[\s\S]*this\.readbackPending = false/, "failed quadtree diagnostics release the pooled readback slot");
});
