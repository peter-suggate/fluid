/** Observe the two surface searches around adaptive redistance. Readbacks
 * happen after each frame and are deliberately excluded from performance
 * claims. Equality of masks certifies equality of the nearest-cell search;
 * it does not certify equality of the nodal metric rebuilt from phi. */
import assert from "node:assert/strict";

type Stage = {
  bins: GPUBuffer;
  dispatchBatch(encoder: GPUCommandEncoder, entries: readonly string[], group: string | undefined, label: string, shared?: GPUComputePassEncoder): void;
};
const popcount = (input: number) => {
  let value = input >>> 0;
  value -= (value >>> 1) & 0x55555555;
  value = (value & 0x33333333) + ((value >>> 2) & 0x33333333);
  return (((value + (value >>> 4)) & 0x0f0f0f0f) * 0x01010101) >>> 24;
};

export class CrossingMaskObserver {
  private readonly before: GPUBuffer;
  private readonly after: GPUBuffer;
  private readonly original: Stage["dispatchBatch"];
  private pending = false;
  private pairs = 0;
  readonly tiles: number;
  constructor(private readonly stage: Stage, device: GPUDevice, dimensions: readonly number[]) {
    const cells = dimensions.reduce((a, b) => a * b, 1), vertices = dimensions.reduce((a, b) => a * (b + 1), 1);
    this.tiles = cells / 64;
    const offset = 4 * (3 * cells + vertices + 2 * this.tiles), size = 4 * (4 + 3 * this.tiles);
    this.before = device.createBuffer({ label: "Crossing masks before redistance", size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    this.after = device.createBuffer({ label: "Crossing masks after redistance", size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    this.original = stage.dispatchBatch;
    stage.dispatchBatch = (encoder, entries, group, label, shared) => {
      this.original.call(stage, encoder, entries, group, label, shared);
      if (label !== "crossing search") return;
      assert.equal(shared, undefined, "Crossing observer copies must sit outside a compute pass");
      if (group === "redistance") {
        assert.ok(!this.pending, "Each redistance search must precede one update search");
        encoder.copyBufferToBuffer(stage.bins, offset, this.before, 0, size); this.pending = true;
      } else if (this.pending) {
        encoder.copyBufferToBuffer(stage.bins, offset, this.after, 0, size); this.pending = false; this.pairs++;
      }
    };
  }
  async take(frame: number): Promise<unknown | undefined> {
    if (!this.pairs) {
      const skippedSecondSearch = this.pending; this.pending = false;
      return skippedSecondSearch ? { frame, skippedSecondSearch: true } : undefined;
    }
    assert.ok(!this.pending, "A captured frame must finish its paired search");
    assert.equal(this.pairs, 1, "Observe one simulation step per frame to count every search pair");
    this.pairs = 0;
    await Promise.all([this.before.mapAsync(GPUMapMode.READ), this.after.mapAsync(GPUMapMode.READ)]);
    const a = new Uint32Array(this.before.getMappedRange()), b = new Uint32Array(this.after.getMappedRange());
    let changedMasks = 0, changedCrossingBits = 0, changedReach = 0, oldCrossings = 0, newCrossings = 0;
    for (let tile = 0; tile < this.tiles; tile++) {
      const at = 4 + 2 * tile, x = a[at]! ^ b[at]!, y = a[at + 1]! ^ b[at + 1]!;
      if ((x | y) !== 0) changedMasks++;
      changedCrossingBits += popcount(x) + popcount(y);
      oldCrossings += popcount(a[at]!) + popcount(a[at + 1]!); newCrossings += popcount(b[at]!) + popcount(b[at + 1]!);
      if (a[4 + 2 * this.tiles + tile] !== b[4 + 2 * this.tiles + tile]) changedReach++;
    }
    const result = { frame, tiles: this.tiles, changedMasks, changedCrossingBits, changedReach, oldCrossings, newCrossings, oldCounts: [...a.slice(0, 3)], newCounts: [...b.slice(0, 3)], identicalMasks: changedMasks === 0, identicalReach: changedReach === 0 };
    this.before.unmap(); this.after.unmap(); return result;
  }
  destroy(): void { this.stage.dispatchBatch = this.original; this.before.destroy(); this.after.destroy(); }
}
