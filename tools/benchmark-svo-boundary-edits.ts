/** CPU/upload-work microbenchmark; no GPU execution or end-to-end latency claim.
 * node --import tsx tools/benchmark-svo-boundary-edits.ts
 */
import assert from "node:assert/strict";
import { poolPartyBoardGeometry } from "../lib/core/backyard-pool-party-scene";
import { ProceduralSolidField } from "../lib/core/procedural-solid";
import { ProceduralSolidPublication } from "../lib/core/procedural-solid-upload";

const geometry = poolPartyBoardGeometry();
const base = new ProceduralSolidField(geometry.solidBoundary, []);
const origin = geometry.solidBoundary.bricks[0]!.origin;
const edited = base.withEdits([{ operation: "clear", minimum: origin,
  maximumExclusive: origin.map(value => value + 1) as [number, number, number] }]);
const image = base.pack();
let bytes = 0;
const queue = { writeBuffer(_buffer: GPUBuffer, _offset: number, data: Uint32Array, _from = 0, count = data.length) {
  bytes += count * 4;
} } as Pick<GPUQueue, "writeBuffer">;
const publication = new ProceduralSolidPublication(queue, {} as GPUBuffer, 0, base, image);
const runs = 100, batches = 20;
const sample = (incremental: boolean) => {
  bytes = 0;
  const start = performance.now();
  for (let index = 0; index < runs; index++) {
    const field = index % 2 ? base : edited;
    if (incremental) publication.publish(field);
    else queue.writeBuffer({} as GPUBuffer, 0, field.pack());
  }
  return { ms: performance.now() - start, bytes };
};
sample(false); sample(true);
const before = [], after = [];
for (let batch = 0; batch < batches; batch++) {
  if (batch % 2) { after.push(sample(true)); before.push(sample(false)); }
  else { before.push(sample(false)); after.push(sample(true)); }
}
assert.ok(before.every(result => result.bytes === image.byteLength * runs));
assert.ok(after.every(result => result.bytes === 4 * runs));
const median = (values: number[]) => {
  values.sort((left, right) => left - right);
  return (values[values.length / 2 - 1]! + values[values.length / 2]!) / 2;
};
console.log(JSON.stringify({ scene: "Pool Party diving board", bricks: base.bricks.size,
  packedBytes: image.byteLength, runsPerBatch: runs, batches,
  note: "CPU pack/publication only, recording queue; no GPU or end-to-end timing",
  before: { medianMs: median(before.map(result => result.ms)), bytesPerEdit: before[0]!.bytes / runs },
  after: { medianMs: median(after.map(result => result.ms)), bytesPerEdit: after[0]!.bytes / runs },
}, null, 2));
