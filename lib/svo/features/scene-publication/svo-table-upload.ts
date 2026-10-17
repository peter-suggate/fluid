/**
 * Upload the words of a table that differ from what its buffer already holds.
 *
 * `published` is the host's record of the buffer region starting at
 * `offsetBytes`, and is brought up to date as each run is written. `next` is
 * read as zero past its end, so a table that shrank clears its own tail.
 * Differences closer than `gapWords` share one write: a queue write has a fixed
 * cost that a few unchanged words in the middle of a run do not.
 *
 * Returns the bytes written, which is zero when the table did not change.
 */
export function uploadChangedWords(
  queue: Pick<GPUQueue, "writeBuffer">,
  buffer: GPUBuffer,
  offsetBytes: number,
  published: Uint32Array,
  next: Uint32Array,
  gapWords = 32,
): number {
  if (next.length > published.length) throw new RangeError("Published table exceeds the region reserved for it");
  const length = published.length, held = next.length;
  let written = 0, index = 0;
  while (index < length) {
    if (published[index] === (index < held ? next[index] : 0)) { index += 1; continue; }
    let end = index + 1, last = end;
    while (end < length && end - last <= gapWords) {
      if (published[end] !== (end < held ? next[end] : 0)) last = end + 1;
      end += 1;
    }
    const within = Math.min(last, held);
    if (within > index) {
      // Bound the driver's staging reservation for large sampled asset images.
      // Offsets and sizes below are words because the source is a Uint32Array.
      const chunkWords = 256 * 1024;
      for (let from = index; from < within; from += chunkWords) {
        queue.writeBuffer(buffer, offsetBytes + from * 4, next as Uint32Array<ArrayBuffer>, from, Math.min(chunkWords, within - from));
      }
      published.set(next.subarray(index, within), index);
    }
    if (last > within) {
      const from = Math.max(index, held);
      queue.writeBuffer(buffer, offsetBytes + from * 4, new Uint32Array(last - from));
      published.fill(0, from, last);
    }
    written += (last - index) * 4;
    index = last;
  }
  return written;
}

/**
 * A buffer region and the host's copy of what was last published into it.
 *
 * The copy is as long as the longest table the region has held; the buffer
 * starts zeroed and a shorter table clears its tail, so the words past any
 * table are zero on both sides and a longer one is diffed against them.
 */
export class PublishedWords {
  private words = new Uint32Array(0);

  /** Upload the words of `next` that differ from the last table published here. Returns the bytes written. */
  publish(queue: GPUQueue, buffer: GPUBuffer, offsetBytes: number, next: Uint32Array | Float32Array): number {
    const view = next instanceof Uint32Array ? next : new Uint32Array(next.buffer, next.byteOffset, next.length);
    if (view.length > this.words.length) {
      const grown = new Uint32Array(view.length);
      grown.set(this.words);
      this.words = grown;
    }
    return uploadChangedWords(queue, buffer, offsetBytes, this.words, view);
  }
}
