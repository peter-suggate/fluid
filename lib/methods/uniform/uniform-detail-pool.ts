/**
 * Uniform Geometric h patch pool bookkeeping (docs/plans/uniform-4h-first-implementation-handoff-2026-10-03.md,
 * "Compact base and optional detail" and "Dynamic transactions and temporal reuse").
 *
 * No GPU calls: this module decides the directory and the copy/transfer plan
 * a GPU layer executes, and when retired storage may be reused.
 *  - Identity: a patch's logical id IS its world patch key (px + PX(py + PY pz)),
 *    stable for the life of the lattice. Physical slots carry a generation that
 *    changes on every assignment; directory entries name both. The execution
 *    list's packed rank is never identity.
 *  - Capacity grows geometrically within a byte budget, planning the old+new
 *    peak (plus old pools still awaiting release); a resize copies live slots
 *    into the candidate pool and the old one is released only after readers
 *    finish its last generation. Capacity also shrinks (compaction) once live
 *    slots fall to 1/growth² of it, and to zero with no detail.
 *  - Deferred reclamation: a slot retired by generation G was last referenced
 *    by G-1 and is reusable only after EVERY registered reader acknowledges
 *    completing G-1. Two frames in flight is the ordinary case; nothing is
 *    reused after an assumed delay.
 *  - prepare() never mutates; commit() adopts a prepared generation, so a
 *    failed candidate leaves the last accepted generation intact.
 */
import { UNIFORM_DETAIL_POLICY } from "./uniform-detail-policy";

export const UNIFORM_DETAIL_NO_SLOT = 0xffffffff;

export interface UniformDetailPoolOptions {
  /** Domain patches; directory length and key range. */
  readonly domainPatches: number;
  /** Bytes one slot holds across every h field and scratch texture. */
  readonly slotBytes: number;
  /** Device-derived ceiling for all slot storage alive at once, including a resize peak. */
  readonly byteBudget: number;
  /** Frame consumers that must acknowledge generations (e.g. simulation, render). */
  readonly readers: readonly string[];
  readonly growth?: number;
}
export interface UniformDetailPoolEntry { readonly key: number; readonly slot: number; readonly slotGeneration: number }
export interface UniformDetailPoolTransaction {
  readonly generation: number;
  readonly baseGeneration: number;
  readonly capacity: number;
  /** A resize: allocate `toSlots`, copy live slots `from → to`, release the old pool after readers finish `baseGeneration`. */
  readonly resize?: { readonly fromSlots: number; readonly toSlots: number; readonly copies: readonly { readonly key: number; readonly from: number; readonly to: number }[] };
  /** New residents: prolong from the base (restricted coarse data). */
  readonly promotions: readonly UniformDetailPoolEntry[];
  /** Restrict to the base; the old slot is reclaimed only after readers finish baseGeneration. */
  readonly retirements: readonly { readonly key: number; readonly slot: number }[];
  readonly retained: readonly UniformDetailPoolEntry[];
  /** Keys refused for lack of storage within the byte budget, in request order. */
  readonly deferred: readonly number[];
  /** Per domain patch: slot, or UNIFORM_DETAIL_NO_SLOT. */
  readonly directory: Uint32Array;
  /** Per directory entry's slot: its generation (0 for a free slot). */
  readonly slotGenerations: Uint32Array;
  /** Live slots in world-key order: a dispatch order, not identity. */
  readonly execution: Uint32Array;
  readonly bytes: number;
  /** Slot bytes alive at once while this generation is adopted (old pools awaiting release included). */
  readonly peakBytes: number;
}

interface Slot { key: number; generation: number; retiredAfter: number }
const FREE = -1, LIVE = Infinity;
interface Pool { readonly capacity: number; readonly lastGeneration: number }

export class UniformDetailPool {
  readonly slotBytes: number;
  readonly byteBudget: number;
  readonly growth: number;
  private capacityValue = 0;
  private slots: Slot[] = [];
  private live = new Map<number, number>();
  private accepted = 0;
  private slotGeneration = 0;
  private readonly acknowledged = new Map<string, number>();
  /** Old pools a resize replaced, released once readers finish their last generation. */
  private pending: Pool[] = [];
  constructor(private readonly options: UniformDetailPoolOptions) {
    if (!Number.isSafeInteger(options.domainPatches) || options.domainPatches < 1) throw new Error("Detail pool needs a positive patch count");
    if (!(options.slotBytes > 0) || !(options.byteBudget >= 0)) throw new Error("Detail pool needs positive slot bytes and a byte budget");
    if (!options.readers.length || new Set(options.readers).size !== options.readers.length) throw new Error("Detail pool needs distinct readers");
    this.slotBytes = options.slotBytes;this.byteBudget = options.byteBudget;
    this.growth = Math.max(1.25, options.growth ?? UNIFORM_DETAIL_POLICY.poolGrowth);
    for (const reader of options.readers) this.acknowledged.set(reader, 0);
  }
  get capacity(): number { return this.capacityValue; }
  get acceptedGeneration(): number { return this.accepted; }
  get liveSlots(): number { return this.live.size; }
  get bytes(): number { return this.capacityValue * this.slotBytes; }
  /** Slot bytes alive now, including old pools still awaiting release. */
  get residentBytes(): number { return this.bytes + this.pending.reduce((s, p) => s + p.capacity * this.slotBytes, 0); }
  /** The oldest generation some reader may still be using: everything ≤ it is complete everywhere. */
  get completedGeneration(): number { return Math.min(...this.acknowledged.values()); }
  /** Most patches the next generation can hold: one resize from now within the budget. */
  get admissionLimitPatches(): number {
    const pending = this.pending.reduce((s, p) => s + p.capacity, 0) * this.slotBytes;
    return Math.min(this.options.domainPatches, Math.max(this.capacityValue, Math.floor((this.byteBudget - pending - this.bytes) / this.slotBytes)));
  }
  slotOf(key: number): number { return this.live.get(key) ?? UNIFORM_DETAIL_NO_SLOT; }

  /** A reader finished every frame of generations ≤ generation. Monotonic; never ahead of acceptance. */
  acknowledge(reader: string, generation: number): void {
    const previous = this.acknowledged.get(reader);
    if (previous === undefined) throw new Error(`Unknown detail pool reader ${reader}`);
    if (generation > this.accepted) throw new Error(`${reader} acknowledged generation ${generation} past accepted ${this.accepted}`);
    this.acknowledged.set(reader, Math.max(previous, generation));
  }
  /** Old pools every reader has finished with; the caller destroys them. */
  takeReleasable(): { capacity: number; bytes: number }[] {
    const done = this.completedGeneration, out = this.pending.filter(p => p.lastGeneration <= done);
    this.pending = this.pending.filter(p => p.lastGeneration > done);
    return out.map(p => ({ capacity: p.capacity, bytes: p.capacity * this.slotBytes }));
  }

  /** The next generation's directory for `resident` keys (priority order: earlier keys win storage). */
  prepare(resident: Iterable<number>): UniformDetailPoolTransaction {
    const keys: number[] = [], seen = new Set<number>();
    for (const key of resident) {
      if (!Number.isSafeInteger(key) || key < 0 || key >= this.options.domainPatches) throw new Error(`Detail patch key ${key} is outside the domain`);
      if (!seen.has(key)) { seen.add(key);keys.push(key); }
    }
    const done = this.completedGeneration, retained = keys.filter(k => this.live.has(k)), fresh = keys.filter(k => !this.live.has(k));
    const retirements = [...this.live].filter(([k]) => !seen.has(k)).sort((a, b) => a[0] - b[0]).map(([key, slot]) => ({ key, slot }));
    // Slots usable without a resize: free, or retired and finished by every reader.
    const reusable: number[] = [];
    for (let s = 0; s < this.capacityValue; s++) { const r = this.slots[s]!.retiredAfter; if (r === FREE || (r !== LIVE && r <= done)) reusable.push(s); }
    const pendingSlots = this.pending.reduce((s, p) => s + p.capacity, 0);
    const fits = (capacity: number) => (pendingSlots + this.capacityValue + capacity) * this.slotBytes <= this.byteBudget;
    let capacity = this.capacityValue, admitted = fresh, deferred: number[] = [];
    const need = retained.length + fresh.length;
    if (fresh.length > reusable.length) {
      // Grow geometrically; shrink the step toward the need, then admit what fits.
      let target = Math.min(this.options.domainPatches, Math.max(need, Math.ceil(Math.max(1, this.capacityValue) * this.growth)));
      while (target > need && !fits(target)) target = Math.max(need, Math.floor(target / this.growth));
      while (target > Math.max(this.capacityValue, retained.length) && !fits(target)) target--;
      if (target > this.capacityValue && fits(target)) capacity = target;
      const room = capacity === this.capacityValue ? reusable.length : capacity - retained.length;
      admitted = fresh.slice(0, room);deferred = fresh.slice(room);
    } else if (need === 0 && this.capacityValue > 0) capacity = 0;
    else if (need * this.growth * this.growth <= this.capacityValue && fits(Math.ceil(need * this.growth))) capacity = Math.ceil(need * this.growth);
    const generation = this.accepted + 1, directory = new Uint32Array(this.options.domainPatches).fill(UNIFORM_DETAIL_NO_SLOT);
    const promotions: UniformDetailPoolEntry[] = [], kept: UniformDetailPoolEntry[] = [];
    let nextGeneration = this.slotGeneration;
    let resize: UniformDetailPoolTransaction["resize"];
    let slotGenerations: Uint32Array;
    if (capacity !== this.capacityValue) {
      // A new pool: old generations read the old one, so every index not
      // copied into is free at once. Growth keeps indices; shrink compacts.
      slotGenerations = new Uint32Array(capacity);
      const copies: { key: number; from: number; to: number }[] = [], used = new Uint8Array(capacity);
      const ordered = [...retained].sort((a, b) => a - b), grow = capacity > this.capacityValue;
      let cursor = 0;
      for (const key of ordered) {
        const from = this.live.get(key)!;let to = from;
        if (!grow) { while (used[cursor]) cursor++;to = cursor; }
        used[to] = 1;copies.push({ key, from, to });
        directory[key] = to;slotGenerations[to] = ++nextGeneration;kept.push({ key, slot: to, slotGeneration: nextGeneration });
      }
      let free = 0;
      for (const key of admitted) {
        while (used[free]) free++;
        used[free] = 1;directory[key] = free;slotGenerations[free] = ++nextGeneration;promotions.push({ key, slot: free, slotGeneration: nextGeneration });
      }
      resize = { fromSlots: this.capacityValue, toSlots: capacity, copies };
    } else {
      slotGenerations = Uint32Array.from(this.slots, s => s.retiredAfter === LIVE ? s.generation : 0);
      for (const key of retained) { const slot = this.live.get(key)!;directory[key] = slot;kept.push({ key, slot, slotGeneration: this.slots[slot]!.generation }); }
      for (const { slot } of retirements) slotGenerations[slot] = 0;
      admitted.forEach((key, i) => {
        const slot = reusable[i]!;directory[key] = slot;slotGenerations[slot] = ++nextGeneration;
        promotions.push({ key, slot, slotGeneration: nextGeneration });
      });
    }
    const execution: number[] = [];
    for (let key = 0; key < directory.length; key++) if (directory[key] !== UNIFORM_DETAIL_NO_SLOT) execution.push(directory[key]!);
    const bytes = capacity * this.slotBytes, peakBytes = (pendingSlots + (resize ? this.capacityValue : 0)) * this.slotBytes + bytes;
    if (peakBytes > this.byteBudget && capacity > 0) throw new Error(`Detail pool plan peaks at ${peakBytes} bytes, above the ${this.byteBudget}-byte budget`);
    return { generation, baseGeneration: this.accepted, capacity, resize, promotions, retirements, retained: kept, deferred,
      directory, slotGenerations, execution: Uint32Array.from(execution), bytes, peakBytes };
  }

  /** Adopt a prepared generation once its transfers completed and the frame was accepted. */
  commit(tx: UniformDetailPoolTransaction): void {
    if (tx.baseGeneration !== this.accepted) throw new Error(`Detail pool generation ${tx.generation} was prepared on ${tx.baseGeneration}, not the accepted ${this.accepted}`);
    if (tx.resize) {
      if (this.capacityValue) this.pending.push({ capacity: this.capacityValue, lastGeneration: this.accepted });
      this.slots = Array.from({ length: tx.capacity }, () => ({ key: -1, generation: 0, retiredAfter: FREE }));
    } else for (const { slot } of tx.retirements) Object.assign(this.slots[slot]!, { key: -1, retiredAfter: this.accepted });
    this.capacityValue = tx.capacity;this.live = new Map();
    for (const entry of [...tx.retained, ...tx.promotions]) {
      Object.assign(this.slots[entry.slot]!, { key: entry.key, generation: entry.slotGeneration, retiredAfter: LIVE });
      this.live.set(entry.key, entry.slot);this.slotGeneration = Math.max(this.slotGeneration, entry.slotGeneration);
    }
    this.accepted = tx.generation;
  }
}
