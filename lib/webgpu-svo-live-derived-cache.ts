export type LiveSvoDerivedPageAvailability = "current" | "last-known-good" | "unavailable";

export interface LiveSvoDerivedUpdate {
  generation: number;
  /** Fixed-slot worklist derived from dirty leaves and their affected ancestors/aprons. */
  dirtySlots: readonly number[];
}

export interface LiveSvoDerivedPublishDecision {
  published: boolean;
  generation: number;
  pendingDirtyPages: number;
  unavailableDirtyPages: number;
}

export interface LiveSvoDerivedPageValidityBinding {
  /** Sampled uint texture avoids consuming a scarce renderer storage binding. */
  texture: GPUTexture;
  view: GPUTextureView;
  format: "r32uint";
  capacity: number;
}

function validGeneration(value: number, previous: number): void {
  if (!Number.isSafeInteger(value) || value <= previous || value > 0xffffffff) {
    throw new RangeError("Derived-cache generation must be a strictly increasing nonzero uint32");
  }
}

function coalescedRanges(slots: readonly number[]): Array<readonly [number, number]> {
  if (slots.length === 0) return [];
  const result: Array<readonly [number, number]> = [];
  let first = slots[0], previous = first;
  for (let index = 1; index < slots.length; index += 1) {
    const slot = slots[index];
    if (slot === previous + 1) previous = slot;
    else { result.push([first, previous + 1]); first = previous = slot; }
  }
  result.push([first, previous + 1]);
  return result;
}

/**
 * Fixed-capacity page-generation owner shared by opacity and radiance atlases.
 * Dirty pages become unavailable immediately. Clean pages remain valid because
 * the scene delta did not touch them; no old value is ever sampled in a dirty
 * region. Publication advances only after every dirty slot is complete.
 */
export class WebGpuLiveSvoDerivedPageState {
  readonly validity: LiveSvoDerivedPageValidityBinding;
  readonly allocatedBytes: number;

  private readonly generations: Uint32Array<ArrayBuffer>;
  private readonly uploadScratch: Uint32Array<ArrayBuffer>;
  private readonly dirty = new Set<number>();
  private readonly completed = new Set<number>();
  private requestedGeneration = 0;
  private completeGeneration = 0;
  private destroyed = false;

  constructor(private readonly device: GPUDevice, capacity: number, label: string) {
    if (!Number.isSafeInteger(capacity) || capacity <= 0) throw new RangeError("Derived page capacity must be a positive integer");
    const texture = device.createTexture({
      label: `${label} page validity generations`, size: [capacity, 1], format: "r32uint",
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_DST,
    });
    this.generations = new Uint32Array(capacity);
    this.uploadScratch = new Uint32Array(capacity);
    this.allocatedBytes = this.generations.byteLength;
    this.validity = Object.freeze({ texture, view: texture.createView(), format: "r32uint" as const, capacity });
  }

  begin(update: LiveSvoDerivedUpdate): void {
    this.assertAlive();
    if (this.requestedGeneration !== this.completeGeneration) throw new Error("Complete or abandon the pending derived-cache generation first");
    validGeneration(update.generation, this.requestedGeneration);
    const slots = [...new Set(update.dirtySlots)].sort((a, b) => a - b);
    for (const slot of slots) {
      if (!Number.isSafeInteger(slot) || slot < 0 || slot >= this.generations.length) throw new RangeError("Dirty derived page slot exceeds fixed capacity");
    }
    this.requestedGeneration = update.generation;
    this.dirty.clear(); this.completed.clear();
    slots.forEach((slot) => { this.dirty.add(slot); this.generations[slot] = 0; });
    for (const [first, end] of coalescedRanges(slots)) {
      this.uploadScratch.fill(0, first, end);
      this.device.queue.writeTexture({ texture: this.validity.texture, origin: [first, 0] }, this.uploadScratch.subarray(first, end),
        { bytesPerRow: (end - first) * 4 }, [end - first, 1]);
    }
  }

  markComplete(slot: number): void {
    this.assertAlive();
    if (!this.dirty.has(slot)) throw new Error("Completed derived page is not in the current dirty worklist");
    this.completed.add(slot);
  }

  publish(): LiveSvoDerivedPublishDecision {
    this.assertAlive();
    if (this.requestedGeneration === this.completeGeneration) throw new Error("No derived-cache generation is pending");
    const pendingDirtyPages = this.dirty.size - this.completed.size;
    if (pendingDirtyPages !== 0) return {
      published: false, generation: this.completeGeneration, pendingDirtyPages,
      unavailableDirtyPages: this.dirty.size,
    };
    const slots = [...this.dirty].sort((a, b) => a - b);
    slots.forEach((slot) => { this.generations[slot] = this.requestedGeneration; });
    for (const [first, end] of coalescedRanges(slots)) {
      this.device.queue.writeTexture({ texture: this.validity.texture, origin: [first, 0] }, this.generations.subarray(first, end),
        { bytesPerRow: (end - first) * 4 }, [end - first, 1]);
    }
    this.completeGeneration = this.requestedGeneration;
    this.dirty.clear(); this.completed.clear();
    return { published: true, generation: this.completeGeneration, pendingDirtyPages: 0, unavailableDirtyPages: 0 };
  }

  availability(slot: number): LiveSvoDerivedPageAvailability {
    if (!Number.isSafeInteger(slot) || slot < 0 || slot >= this.generations.length) return "unavailable";
    if (this.dirty.has(slot)) return "unavailable";
    const generation = this.generations[slot];
    if (generation === 0) return "unavailable";
    return generation === this.completeGeneration ? "current" : "last-known-good";
  }

  telemetry() {
    let availablePages = 0;
    for (const generation of this.generations) if (generation !== 0) availablePages += 1;
    return {
      completeGeneration: this.completeGeneration,
      requestedGeneration: this.requestedGeneration,
      dirtyPages: this.dirty.size,
      completedDirtyPages: this.completed.size,
      availablePages,
      allocatedBytes: this.allocatedBytes,
    };
  }

  destroy(): void {
    if (this.destroyed) return;
    this.validity.texture.destroy();
    this.destroyed = true;
  }

  private assertAlive(): void {
    if (this.destroyed) throw new Error("Live SVO derived page state is destroyed");
  }
}
