import { uploadChangedWords } from "../svo/features/scene-publication/svo-table-upload";
import { PROCEDURAL_SOLID_RECORD_WORDS, PROCEDURAL_SOLID_VALIDITY_WORDS,
  type ProceduralSolidField } from "./procedural-solid";

/** The uploaded sampled boundary. Edits change validity masks, never source samples. */
export class ProceduralSolidPublication {
  private source: ProceduralSolidField["source"] | undefined;

  /** `image` is the exact image already uploaded at `offsetBytes`; this owns its host copy. */
  constructor(private readonly queue: Pick<GPUQueue, "writeBuffer">,
    private readonly buffer: GPUBuffer, private readonly offsetBytes: number,
    field: ProceduralSolidField, private readonly image: Uint32Array<ArrayBuffer>) {
    this.source = field.source;
  }

  /** Returns actual uploaded bytes, including restoring a previously invalidated boundary. */
  publish(field: ProceduralSolidField | undefined): number {
    if (field && field.source === this.source) {
      // The immutable source fixes hash slots, record order and sampled values.
      // Walk just the 16 mask words per brick, leaving its 729 samples alone.
      let offset = 4 + (this.image[0]! + 1) * 4, written = 0;
      for (const brick of field.bricks.values()) {
        written += uploadChangedWords(this.queue, this.buffer, this.offsetBytes + offset * 4,
          this.image.subarray(offset, offset + PROCEDURAL_SOLID_VALIDITY_WORDS), brick.valid);
        offset += PROCEDURAL_SOLID_RECORD_WORDS;
      }
      return written;
    }
    // A different source (including a structured clone) cannot rely on record
    // order. Loss of provenance clears the entire field; undo restores it.
    const next = field?.pack();
    if (next && next.length !== this.image.length) throw new RangeError("Procedural boundary changed; rebuild the renderer");
    const written = uploadChangedWords(this.queue, this.buffer, this.offsetBytes, this.image, next ?? new Uint32Array(0));
    this.source = field?.source;
    return written;
  }
}
