import assert from "node:assert/strict";
import test from "node:test";
import { ProceduralSolidField, type ProceduralSolidBoundary } from "../lib/core/procedural-solid";
import { ProceduralSolidPublication } from "../lib/core/procedural-solid-upload";

const source: ProceduralSolidBoundary = {
  version: 1, dimensions: [16, 8, 8], cellSize_m: [1, 1, 1], patchCount: 0, patchesKey: "",
  bricks: [0, 8].map(x => ({ origin: [x, 0, 0], values: Array.from({ length: 729 }, (_, i) => i % 9 - 3.5) })),
};

function fixture() {
  const base = new ProceduralSolidField(source, []), image = base.pack();
  const prefix = 12, gpu = new Uint32Array(prefix + image.length).fill(0x12345678);
  gpu.set(image, prefix);
  const writes: { offset: number; bytes: number }[] = [];
  const queue = { writeBuffer(_buffer: GPUBuffer, at: number, data: Uint32Array, from = 0, count = data.length) {
    writes.push({ offset: at, bytes: count * 4 });
    gpu.set(data.subarray(from, from + count), at / 4);
  } } as Pick<GPUQueue, "writeBuffer">;
  return { base, gpu, prefix, writes,
    publication: new ProceduralSolidPublication(queue, {} as GPUBuffer, prefix * 4, base, image) };
}

test("sampled boundary uploads only changed validity words, with exact full-image parity through edits and undo", () => {
  const { base, gpu, prefix, writes, publication } = fixture();
  const outside = base.withEdits([{ operation: "fill", minimum: [24, 0, 0], maximumExclusive: [25, 1, 1] }]);
  // The fast path must not repack the immutable scalar samples.
  outside.pack = () => { throw new Error("unchanged source must not be packed"); };
  assert.equal(publication.publish(outside), 0);
  assert.equal(writes.length, 0);
  const edited = base.withEdits([{ operation: "clear", minimum: [7, 2, 1], maximumExclusive: [9, 3, 2] }]);
  const reference = edited.pack();
  edited.pack = () => { throw new Error("an edit must upload masks without packing samples"); };
  assert.equal(publication.publish(edited), 8, "one word in each of the two seam bricks");
  assert.deepEqual(gpu.subarray(prefix), reference);
  assert.equal(publication.publish(edited), 0);
  assert.equal(publication.publish(base), 8, "undo restores exactly the invalidated words");
  assert.deepEqual(gpu.subarray(prefix), base.pack());
  assert.ok(gpu.subarray(0, prefix).every(word => word === 0x12345678));
});

test("provenance loss, restore and source replacement publish the exact packed boundary", () => {
  const { base, gpu, prefix, publication } = fixture();
  assert.ok(publication.publish(undefined) > 0);
  assert.ok(gpu.subarray(prefix).every(word => word === 0));
  assert.equal(publication.publish(undefined), 0);
  assert.ok(publication.publish(base) > 0);
  assert.deepEqual(gpu.subarray(prefix), base.pack());
  const reordered = new ProceduralSolidField({ ...structuredClone(source), bricks: [...source.bricks].reverse() }, []);
  publication.publish(reordered);
  assert.deepEqual(gpu.subarray(prefix), reordered.pack(), "equal-size sources may have different record order");
  const changed = reordered.withEdits([{ operation: "fill", minimum: [9, 0, 0], maximumExclusive: [10, 1, 1] }]);
  assert.equal(publication.publish(changed), 4);
  assert.deepEqual(gpu.subarray(prefix), changed.pack());
  const smaller = new ProceduralSolidField({ ...source, bricks: source.bricks.slice(0, 1) }, []);
  assert.throws(() => publication.publish(smaller), /rebuild/);
  assert.deepEqual(gpu.subarray(prefix), changed.pack(), "refusal must leave the publication intact");
});
