/**
 * The banded leaf payload: store the surface, reconstruct the rest.
 *
 * A leaf's 512-voxel payload is the whole arena at refinement depth. At depth 3
 * the hero garden's 227 168 leaves hold 116.3 M voxels, so the arena is 1.40 GB
 * before anything else is allocated. Three questions can shrink it and only two
 * of them are separate: how *wide* a geometry voxel is
 * ({@link SparseBrickSceneGeometryFormat}, 8 bytes toward 2), and how *many*
 * voxels need storing at all, which is this file. The third — how wide the
 * material-owner word is — turns out not to be a separate question, and folding
 * it in here is why this file carries the identity palette rather than leaving
 * it to the owner lane.
 *
 * ## The identity word splits, because its two halves are different populations
 *
 * The identity word used to be `(ownerId << 16) | materialId`, and its high half
 * was a per-*object* handle. It is now `(oct8 normal << 16) | materialId`
 * ({@link sparseBrickSceneIdentityWordCodecWGSL}), and the normal is per-*voxel*:
 * the voxeliser evaluates the winning primitive's outward normal at every
 * occupied voxel centre, deep interior included, so the two halves no longer vary
 * on the same scale at all. Measured on `hero-garden-hose` at depth 0 — 12 724
 * published leaves, 6.51 M voxels, censused off the device by replaying the dense
 * identity lane the voxeliser wrote:
 *
 * | interned                     | mean a leaf | max | leaves at 1 | leaves > 16 |
 * |------------------------------|------------:|----:|------------:|------------:|
 * | whole word `normal\|material` |     29.05  | 474 |      12.3 % |      51.9 % |
 * | normal half alone            |      29.04  | 474 |      12.3 % |      51.9 % |
 * | **material half alone**      |     **1.026** |  28 |  **94.8 %** |   **0.5 %** |
 *
 * The first two rows are the same number twice, which is the finding: interning
 * the whole word interns the *normal alphabet*, and the resulting per-voxel index
 * is a normal store wearing a palette's clothes — an 8-bit one on half the leaves.
 * The material half is what a palette is for, and it is a better palette than the
 * owner id ever was (1.026 against the 1.457 SP21 measured).
 *
 * So the palette interns **materials**, and the normal gets its own lane: one
 * `u16` per voxel, in a fixed 256-word lane at the head of the leaf's own blob.
 * {@link SvoBandedLeaf.normals}.
 *
 * ### Why the normal is not stored beside the geometry records
 *
 * Records are the band dilated by the normal stencil — 6.63 % of hero voxels — so
 * putting normals there would cost nothing per interior voxel. It is only legal if
 * every voxel that can be a primary *first hit* is in the record set, and it is
 * not, for a reason that is structural rather than statistical: **the stencil
 * dilation clamps at the leaf boundary** (`encodeBandedLeaf` below, and the GPU
 * encoder it mirrors), so an occupied voxel on a leaf face whose only band
 * neighbour lies in the *adjacent* leaf gets no record. A ray entering that leaf
 * through that face hits it first. The census counts **2 733 158 of 6 514 688
 * voxels (42.0 %)** occupied, unrecorded and on a leaf face, plus 7 680 that are
 * occupied, unrecorded and touch air *inside* their own leaf. The LOD tier
 * compounds it: `dryLodCellStride > 1` samples every Nth cell, so its first solid
 * sample is routinely a deep interior voxel.
 *
 * A wrong normal on a first hit is a visible artifact, so the normal lane spans
 * every voxel. That costs 2 bytes where a record costs 4, and it is the largest
 * single term in the layout — see {@link bandedLeafBytes}.
 *
 * ## Two lanes, one structure
 *
 * Most voxels in a leaf are not surface. Measured on the two dry garden scenes
 * with `FLUID_SVO_REFINEMENT_MODE=surface` (`tmp/sp22/band-census.ts`, a CPU
 * replay of `rebuildDirtyBrickPayload` over a uniform sample of planned leaves):
 *
 * | scene / depth              | band  | interior | exterior | all-interior leaves |
 * |----------------------------|-------|----------|----------|---------------------|
 * | `hero-garden-hose` d0      |  3.2 %|   79.1 % |   17.7 % | 68 %                |
 * | `hero-garden-hose` d1      |  4.6 %|   71.3 % |   24.1 % | 56 %                |
 * | `hero-garden-hose` d2      |  4.2 %|   74.7 % |   21.1 % | 62 %                |
 * | `garden-svo-lighting` d0   | 12.7 %|   42.7 % |   44.5 % | 18 %                |
 * | `garden-svo-lighting` d1   | 11.0 %|   48.3 % |   40.8 % | 22 %                |
 * | `garden-svo-lighting` d2   |  8.4 %|   58.2 % |   33.4 % | 35 %                |
 *
 * "Band" is a voxel whose `solidFraction` is strictly between 0 and 1, which
 * under the planar coverage law is exactly a voxel whose nearest surface is
 * within one cell radius. Everything else is a constant: fraction 1 with an
 * owner, or fraction 0 with none.
 *
 * The material lane, read off the device rather than replayed, describes the
 * *same* leaves through the other lane: **94.8 % of hero leaves at depth 0 hold
 * exactly one material across all 512 voxels**, with a mean of 1.026 distinct
 * materials a leaf. Those are not two savings to multiply. They are one fact — a
 * leaf is usually one solid — seen twice, and a design that encoded them
 * separately would spend per-voxel material bits inside the very records whose
 * whole leaf shares one material.
 *
 * So the material is *not* a field of a record. It is a per-leaf palette of
 * slots into a scene-global table, with an index array over the leaf's occupied
 * voxels whose width is chosen per leaf — **zero bits when the leaf has one
 * material**, which is nineteen leaves in twenty. Records carry geometry only.
 *
 * ## What an interior voxel costs
 *
 * Two bits and a normal: one occupancy bit and one zero record bit, both already
 * inside the fixed masks, plus the 16-bit baked normal every occupied voxel
 * carries. In a single-material leaf that is the whole cost — no record, no index,
 * no palette entry. In a multi-material leaf it is those two bits plus the leaf's
 * index width, one to eight, plus the same normal.
 *
 * ## What must still be stored, and why the band is not the record set
 *
 * `safeNormal` (lib/webgpu-svo-live-derived-builder.ts) central-differences
 * `solidDistance` at `local +/- 1` on each axis, clamped inside the leaf. A band
 * voxel's normal is therefore exact only if its six axis neighbours also carry a
 * distance. The record set is consequently the band *dilated by that stencil* —
 * 6.8-9.4 % of a hero leaf and 15.7-23.1 % of a lighting-study leaf. On the hero
 * the *median* leaf still stores nothing at all.
 *
 * Voxels outside the record set keep a reconstructed distance: the saturated
 * band value, signed by occupancy, sharing
 * `SPARSE_BRICK_SCENE_DISTANCE_BAND_RADII` with the `snorm8` geometry format
 * rather than copying it, so past the band the reconstruction *is* the value
 * that format would have stored. That is lossy in exactly one place — the normal
 * of an occupied voxel with no surface within one cell, and only at level 0,
 * since the radiance build selects `pageGradientNormal` above it. Every such
 * voxel is buried inside solid, where `incidentAlong` and `directIncident` are
 * both occluded. The frame hash is the arbiter, on the same terms as
 * `octreeLiveSceneDryPayloadProfile`, and it is a real gate rather than a
 * formality: this changes how every voxel in the world is read.
 *
 * ## The predicate move
 *
 * Occupancy is defined today by `fraction > 0`: the voxeliser writes a material
 * identity only where the fraction is positive (lib/webgpu-sparse-scene-
 * proxies.ts), so `material != 0` and `fraction > 0` are one predicate. An
 * interior voxel that stops carrying an explicit fraction would stop being
 * occupied. {@link bandedLeafOccupied} is where occupancy moves to: an explicit
 * bit per voxel, written by the encoder and read by every consumer, so a voxel
 * can be solid without being stored.
 *
 * ## What it delivers
 *
 * Bytes a leaf by {@link bandedLeafBytes}, with size-class rounding, leaf
 * palette, material index, normal lane and dense escapes all charged. Measured on
 * `hero-garden-hose` at depth 0 (12 724 published leaves, 77.28 % occupancy,
 * 6.63 % stencil records, 1.026 materials a leaf) against the shipped dense arm's
 * 8 bytes a voxel — 4 of geometry at `f16-unorm8` plus a 4-byte identity word:
 *
 * | term                     | bytes a leaf | share |
 * |--------------------------|-------------:|------:|
 * | occupancy mask           |           64 |  4.9% |
 * | record mask              |           64 |  4.9% |
 * | header (4 words)         |           16 |  1.2% |
 * | material palette + index |          ~ 6 |  0.5% |
 * | **normals**              |    **1 024** | 78.2% |
 * | geometry records         |          136 | 10.4% |
 * | **total**                |    **1 310** |       |
 *
 * 1 310 bytes against the dense arm's 4 096 is **3.13x**, and the shape of the
 * table is the whole story: the normal lane is four fifths of it. The pre-cutover
 * layout quoted 13x, and the difference is not a regression in this file — it is
 * the price of the render path no longer reading any analytic object. A per-voxel
 * normal is 16 bits of genuinely new information the old arena did not carry; what
 * it bought was the deletion of the primitive-record fetch and field
 * re-evaluation the renderer used to do per pixel.
 *
 * **Rank-compacting the normal lane is a real alternative and it is close.**
 * Storing one `u16` per *occupied* voxel instead of per voxel is 792 bytes rather
 * than 1 024 — **1 081 a leaf, 3.79x**, 17 % less arena — at the cost of a
 * sixteen-word prefix popcount in front of every read. Both were measured the same
 * way (interleaved rounds, `hero-garden-hose` depth 1, GPU pass timestamps):
 * compacted **+4.10 %** against `dense`, voxel-indexed **+3.54 %**. So the
 * compaction costs 0.56 pp of frame and buys 51 MB at depth 3. This file ships the
 * voxel-indexed arm because it is also the simpler producer — 256 lanes store 256
 * words with no rank and no reduction — but the trade is close enough that a
 * memory-bound depth-3 lane could legitimately take the other one.
 *
 * **Neither arrangement is where the frame cost is.** `occupancy` — the mask
 * predicate over the *flat* identity lane — measures **+0.64 %**, so the predicate
 * move is free and the ~3 % is the indirection: a `banded` hit resolves from the
 * leaf header, the blob's palette and the blob's normal lane where `dense` read one
 * word. That is the term a further cull has to attack.
 *
 * **The named next cull is a per-leaf normal palette.** The census says a leaf
 * holds 29.05 distinct normals of ~396 occupied voxels, so an 8-bit index over a
 * 29-entry palette is 347 bytes a leaf against 1 024 — 1.3 B a voxel, half the
 * whole arena — and it stays voxel-indexed, so it costs one extra dependent load
 * rather than a popcount. It needs a raw escape for the 14 leaves in 12 724
 * (0.11 %) that exceed 256 distinct normals, and that escape is exactly why it is
 * not in this change: a clamped palette entry is a *wrong normal on a real
 * surface*, which this layout is now free of by construction.
 *
 * The plan for landing it, and the sequencing behind SP20, is
 * `docs/svo-banded-leaf-payload-handoff.md`.
 */
import { SPARSE_BRICK_SCENE_DISTANCE_BAND_RADII } from "./sparse-brick-octree";

/** The lever's variable, named once so a test and a doc cannot drift from it. */
export const SVO_BANDED_RECONSTRUCTION_ENVIRONMENT_VARIABLE = "FLUID_SVO_BANDED_RECONSTRUCTION";

/**
 * Whether the derived builder reconstructs `solidDistance` outside the record set.
 *
 * **Off until the storage lands**, then on. `FLUID_SVO_BANDED_RECONSTRUCTION=1`
 * selects it today.
 *
 * The fidelity question is already settled and the answer is favourable: measured
 * on `hero-garden-hose` it moves the frame by mean 0.54/255, max **7/255**, with
 * **zero pixels over 16/255** — strictly less visible than the `f16-unorm8`
 * geometry lane already shipping at max 11/255. The cause of even that much is a
 * pre-existing lighting bug that predates this work (see the two durable findings
 * in the handoff), not the reconstruction being wrong.
 *
 * So the reason it is off is sequencing, not doubt. Turning it on changes *only*
 * what `solidDistance` returns — the arena, the masks and the palette are separate
 * work (B1/B2), and **until those land this arm carries the diff without carrying
 * any of the saving**. Paying a fidelity cost for a projected benefit is the wrong
 * order; the ~11.6x is real but it does not exist until the storage exists. Flip
 * this default on in the same change that lands B2, and the frame moves once,
 * against a measured arena rather than a forecast one.
 */
export function svoBandedReconstructionEnabled(
  environment: Record<string, string | undefined> | undefined
    = typeof process !== "undefined" ? process.env : undefined,
): boolean {
  const raw = environment?.[SVO_BANDED_RECONSTRUCTION_ENVIRONMENT_VARIABLE];
  if (raw === undefined || raw === "") return false;
  if (raw === "0" || raw === "off") return false;
  if (raw !== "1" && raw !== "on") {
    throw new RangeError(`${SVO_BANDED_RECONSTRUCTION_ENVIRONMENT_VARIABLE} must be "on" or "off"`);
  }
  return true;
}

/** Voxels in an 8-cubed leaf. The only brick size the live scene path builds. */
export const SVO_BANDED_LEAF_VOXELS = 512;

/** 32-bit words in one 512-bit per-voxel mask. */
export const SVO_BANDED_LEAF_MASK_WORDS = SVO_BANDED_LEAF_VOXELS / 32;

/**
 * How far from the surface a stored distance still means anything.
 *
 * `SPARSE_BRICK_SCENE_DISTANCE_BAND_RADII` itself, imported rather than copied,
 * in the same units — cell radii — and for the same reason: beyond it the
 * `snorm8` scene geometry format already saturates, so a reconstructed saturated
 * value and a stored one are the same number.
 */
export const SVO_BANDED_LEAF_SATURATION_RADII = SPARSE_BRICK_SCENE_DISTANCE_BAND_RADII;

/**
 * Distinct material ids a scene may hold.
 *
 * `2 + OCTREE_LIVE_SCENE_PRIMITIVE_CAPACITY`: every live primitive may publish
 * one material, plus the ownerless ground and the air sentinel. 16 386 fits
 * fifteen bits, so a global slot is a `u16` with a bit to spare — which is also
 * the width of the material half of the lane word, so the palette is exact by
 * construction rather than by fitting.
 *
 * Only the *material* half is interned. The high half of the lane word is a
 * per-voxel baked oct8 normal and belongs to {@link SvoBandedLeaf.normals}; a
 * palette over it would be a palette over 29.05 entries a leaf, which is a
 * per-voxel store with an indirection in front of it. See the module comment.
 *
 * Not imported from `lib/webgpu-octree-sparse-bricks.ts` because that module
 * pulls the whole GPU world in behind it and this one is arithmetic. A test pins
 * the two together instead.
 */
export const SVO_IDENTITY_PALETTE_CAPACITY = 2 + 16_384;

/** The identity word the voxeliser writes where nothing is solid. */
export const SVO_BANDED_NO_MATERIAL_OWNER = 0xffff0000;

/**
 * The scene-wide alphabet of material ids.
 *
 * A scene-global table exists so a leaf palette entry can be narrower than the
 * value it names. With the normal split out that is no longer buying anything —
 * a material id is already a `u16` — so the *device* stores the material inline
 * in the leaf palette and never builds this table at all
 * ({@link sparseBrickBandedLeafCodecWGSL}). It is retained here because the
 * reference encoder is the layout's spec and the indirection is the shape a
 * wider identity would need again.
 */
export class SvoIdentityPalette {
  private readonly slots = new Map<number, number>();
  private readonly words: number[] = [];

  /** Intern one material id, returning its global slot. */
  slotFor(materialId: number): number {
    const existing = this.slots.get(materialId);
    if (existing !== undefined) return existing;
    if (this.words.length >= SVO_IDENTITY_PALETTE_CAPACITY) {
      throw new RangeError(`Scene needs more than ${SVO_IDENTITY_PALETTE_CAPACITY} distinct materials`);
    }
    const slot = this.words.length;
    this.slots.set(materialId, slot);
    this.words.push(materialId);
    return slot;
  }

  /** The material id one global slot names. */
  wordAt(slot: number): number {
    const word = this.words[slot];
    if (word === undefined) throw new RangeError(`Identity palette has no slot ${slot}`);
    return word;
  }

  get size(): number { return this.words.length; }
  /** Bytes the global table costs once, for the whole scene. */
  get bytes(): number { return this.words.length * 2; }
}

/**
 * Index widths a leaf palette may use, in bits.
 *
 * Zero is the common case and the reason the ladder starts there: a leaf with
 * one identity spends nothing per voxel. The top is eight bits because the
 * widest leaf measured held 45 distinct identities, so a 256-entry leaf palette
 * has never been approached.
 */
export const SVO_BANDED_LEAF_INDEX_BITS: readonly number[] = Object.freeze([0, 1, 2, 4, 8]);

/** The narrowest index width that addresses `entries` leaf palette slots. */
export function bandedLeafIndexBits(entries: number): number {
  if (!Number.isInteger(entries) || entries < 0) throw new RangeError("Leaf palette size must be a non-negative integer");
  for (const bits of SVO_BANDED_LEAF_INDEX_BITS) if (entries <= (bits === 0 ? 1 : 1 << bits)) return bits;
  return Number.POSITIVE_INFINITY;
}

/**
 * Record counts a leaf may be allocated.
 *
 * A variable-length leaf record needs a free list and a free list needs size
 * classes or it fragments. The ladder is finer in the middle than a doubling one
 * because that is where the leaves are: on `garden-svo-lighting` at depth 2 the
 * sampled stencil counts have p50 10, p90 247 and p99 356, so a doubling ladder
 * would round 247 and 130 alike up to 256. The rounding waste is charged to the
 * layout in {@link bandedLeafBytes} rather than hidden.
 */
export const SVO_BANDED_LEAF_SIZE_CLASSES: readonly number[] = Object.freeze([
  0, 8, 16, 24, 32, 48, 64, 96, 128, 160, 192, 224, 256, 320, 384, 448, 512,
]);

/** The smallest class that holds `records`. */
export function bandedLeafSizeClass(records: number): number {
  if (!Number.isInteger(records) || records < 0 || records > SVO_BANDED_LEAF_VOXELS) {
    throw new RangeError("Banded leaf record count must be an integer in [0, 512]");
  }
  for (const size of SVO_BANDED_LEAF_SIZE_CLASSES) if (records <= size) return size;
  return SVO_BANDED_LEAF_VOXELS;
}

/** One voxel of the dense scene payload, in the lane vocabulary the voxeliser writes. */
export interface SvoDenseVoxel {
  /** `solidSignedDistance`, metres, the field's own value at the voxel centre. */
  readonly distance_m: number;
  /** `solidFraction`, clamped to [0, 1]. */
  readonly fraction: number;
  /**
   * The identity word, `(oct8 normal << 16) | materialId`.
   *
   * Named for the pair it used to be — `(ownerId << 16) | materialId` — because
   * every producer and consumer still calls the channel `materialOwner`. The high
   * half is opaque to this file: it is stored and returned verbatim, so nothing
   * here depends on it being a normal rather than an owner.
   */
  readonly materialOwner: number;
}

/**
 * A leaf's payload in banded form.
 *
 * Against the 6 144 bytes a dense 12-byte leaf costs: two 64-byte masks and a
 * 20-byte header are unconditional, the leaf palette and its index array are
 * empty on a single-material leaf, the normal lane spans the occupied voxels,
 * and the records are the variable part and carry geometry only.
 */
export interface SvoBandedLeaf {
  /** One bit a voxel: is this voxel solid at all. The occupancy predicate. */
  readonly occupancy: Uint32Array;
  /** One bit a voxel: does this voxel carry an explicit geometry record. */
  readonly records: Uint32Array;
  /** `distance_m` for each record voxel, in record order. */
  readonly recordDistance: Float32Array;
  /** `fraction` for each record voxel, in record order. */
  readonly recordFraction: Float32Array;
  /**
   * Global palette slots this leaf's *materials* use, in leaf-index order.
   *
   * One entry on 94.8 % of hero leaves at depth 0, in which case
   * {@link SvoBandedLeaf.identityIndex} is empty and every occupied voxel
   * resolves to `leafPalette[0]`.
   */
  readonly leafPalette: Uint16Array;
  /** Bits an index. Zero when the leaf palette holds one entry or none. */
  readonly indexBits: number;
  /**
   * Leaf-palette index per *occupied* voxel, compacted by occupancy popcount.
   *
   * Compacted rather than laid over all 512 slots because occupancy is 42-79 %
   * of a leaf, so the dense form would spend a fifth to a half of the array on
   * air. The reader already popcounts the record mask; this is the same
   * instruction against the occupancy mask.
   */
  readonly identityIndex: Uint8Array;
  /**
   * The baked oct8 normal per voxel, all 512 of them, indexed by voxel.
   *
   * The high half of the identity word, stored raw rather than interned. Every
   * occupied voxel needs one, because the record set cannot bound the first-hit
   * population — see the module comment.
   *
   * **Indexed by voxel, not by occupancy rank, and that is a measured choice — a
   * close one.** Rank-compacting it saves `1 - occupancy` of the array (232 bytes
   * a leaf on the hero garden, 17 % of the whole layout) and makes every read pay
   * the sixteen-word prefix popcount that addresses it: **+4.10 %** against `dense`
   * on the depth-1 frame, against **+3.54 %** laid out by voxel. 0.56 pp of frame
   * for 51 MB at depth 3. Voxel-indexed ships because it is also the simpler
   * producer — 256 lanes store 256 words with no rank and no reduction — not
   * because the margin is decisive.
   *
   * An air voxel's slot holds whatever the producer wrote there and is never
   * read: `bandedOccupied` gates every consumer.
   */
  readonly normals: Uint16Array;
  /** The saturated distance magnitude this leaf reconstructs with, metres. */
  readonly saturation_m: number;
}

function readBit(mask: Uint32Array, index: number): boolean {
  return ((mask[index >> 5] >>> (index & 31)) & 1) === 1;
}

function writeBit(mask: Uint32Array, index: number): void {
  mask[index >> 5] |= 1 << (index & 31);
}

function popcount(value: number): number {
  let v = value - ((value >>> 1) & 0x55555555);
  v = (v & 0x33333333) + ((v >>> 2) & 0x33333333);
  v = (v + (v >>> 4)) & 0x0f0f0f0f;
  return (v * 0x01010101) >>> 24;
}

/**
 * Set bits before `index`, by prefix popcount.
 *
 * Sixteen words means at most fifteen whole-word popcounts and one masked one,
 * which is what the GPU decode costs per read. It is deliberately not a stored
 * prefix table: sixteen popcounts are cheaper than the sixteen extra bytes a
 * per-leaf table would add to every leaf, including the 47-73 % that carry no
 * records at all — on the hero the median leaf has none.
 */
export function bandedLeafPrefixCount(mask: Uint32Array, index: number): number {
  let count = 0;
  const word = index >> 5;
  for (let i = 0; i < word; i += 1) count += popcount(mask[i]);
  return count + popcount(mask[word] & ((1 << (index & 31)) - 1));
}

/**
 * Whether a voxel is solid.
 *
 * The predicate that used to be `materialOwner != 0`, which the voxeliser made
 * equivalent to `fraction > 0`. Moving it here is the prerequisite for storing a
 * voxel's fraction nowhere: solidity is now a bit the encoder writes, not an
 * inference from a value that may not exist.
 */
export function bandedLeafOccupied(leaf: SvoBandedLeaf, index: number): boolean {
  if (!Number.isInteger(index) || index < 0 || index >= SVO_BANDED_LEAF_VOXELS) {
    throw new RangeError("Banded leaf voxel index must be an integer in [0, 512)");
  }
  return readBit(leaf.occupancy, index);
}

/** The global palette slot one occupied voxel's material resolves to. */
function leafMaterialSlot(leaf: SvoBandedLeaf, rank: number): number {
  if (leaf.indexBits === 0) return leaf.leafPalette[0];
  const bit = rank * leaf.indexBits;
  const entry = (leaf.identityIndex[bit >> 3] >>> (bit & 7)) & ((1 << leaf.indexBits) - 1);
  return leaf.leafPalette[entry];
}

/**
 * One voxel, decoded.
 *
 * Exact in `fraction` and `materialOwner` for every voxel by construction: a
 * voxel with no record is not in the band, so its fraction is exactly 0 or
 * exactly 1; its material is a palette slot and its normal is a raw `u16`, so the
 * word reassembles bit for bit. `distance_m` is exact for record voxels and
 * saturated elsewhere — the one lossy channel, and the reason this ships behind a
 * lever with a frame hash on both arms.
 */
export function bandedLeafVoxel(
  leaf: SvoBandedLeaf,
  index: number,
  palette: SvoIdentityPalette,
): SvoDenseVoxel {
  const occupied = bandedLeafOccupied(leaf, index);
  // The normal is addressed by the voxel; only the material *index* needs a rank,
  // and only on the one leaf in twenty that holds more than one material.
  const rank = occupied && leaf.indexBits > 0 ? bandedLeafPrefixCount(leaf.occupancy, index) : 0;
  const materialOwner = occupied
    ? ((palette.wordAt(leafMaterialSlot(leaf, rank)) & 0xffff) | (leaf.normals[index] << 16)) >>> 0
    : SVO_BANDED_NO_MATERIAL_OWNER;
  if (readBit(leaf.records, index)) {
    const record = bandedLeafPrefixCount(leaf.records, index);
    return { distance_m: leaf.recordDistance[record], fraction: leaf.recordFraction[record], materialOwner };
  }
  return {
    distance_m: occupied ? -leaf.saturation_m : leaf.saturation_m,
    fraction: occupied ? 1 : 0,
    materialOwner,
  };
}

const STENCIL_OFFSETS: readonly (readonly [number, number, number])[] = Object.freeze([
  [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1],
]);

/** Two masks and a four-word header, before any record, palette, index or normal. */
export const SVO_BANDED_LEAF_FIXED_BYTES = 64 + 64 + 16;

/** What a dense voxel costs today: 8 bytes of geometry plus a 4-byte identity word. */
export const SVO_BANDED_DENSE_VOXEL_BYTES = 12;
/** What a dense geometry record costs today, `f32x2`. */
export const SVO_BANDED_DENSE_GEOMETRY_BYTES = 8;

/**
 * What one banded leaf costs, size class, palette, index and normals included.
 *
 * `recordBytes` is the width of one *geometry* record under the world's scene
 * geometry format — 8 for `f32x2` and 4 for `f16-unorm8`, the only arms that
 * survive (the tables above also quote a since-removed 2-byte arm).
 * The identity word is deliberately not part of it; its material half lives in the
 * leaf palette and its normal half in the normal lane, which is why the width cull
 * and this cull compose without counting the same bytes twice.
 *
 * The scene-global palette table is not charged here because it is charged once
 * for the whole scene: a hero garden's whole material alphabet is tens of bytes.
 */
export function bandedLeafBytes(leaf: SvoBandedLeaf, recordBytes: number): number {
  return SVO_BANDED_LEAF_FIXED_BYTES
    + leaf.leafPalette.length * 2
    + leaf.identityIndex.length
    + leaf.normals.length * 2
    + bandedLeafSizeClass(leaf.recordDistance.length) * recordBytes;
}

/**
 * Which voxels get an explicit geometry record.
 *
 * `stencil` is the narrow rule: the band dilated by the normal stencil, so every
 * *band* voxel's normal is exact and everything else is reconstructed. It is the
 * cheapest rule and the one exposed to SP20's finding — snorm8 was rejected on
 * device because flattening the central difference over the deep interior cost
 * 3 299 normal flips and 28.5 % of pixels, and this rule flattens a strictly
 * larger share of the same population.
 *
 * `{ radii }` is the wider rule: store the distance wherever `|d|` is within
 * `radii` cell radii of the surface, which at
 * `SPARSE_BRICK_SCENE_DISTANCE_BAND_RADII` is exactly where SP20 established the
 * gradient still carries information. More records, and a reconstruction that
 * only ever replaces values the `snorm8` lane would itself have saturated.
 *
 * A parameter rather than a constant because the choice between them is a
 * device measurement (`tmp/sp22/banded-normal-fidelity.ts`), and an A/B that
 * needs a constant edited between arms cannot be interleaved.
 */
export type SvoBandedRecordSet = "stencil" | { readonly radii: number };

/**
 * The rule the device chose: `stencil`.
 *
 * Measured on the real hero arena by `tmp/sp22/banded-normal-fidelity.ts`, which
 * replays `safeNormal` per arm over three populations. Angular error against the
 * shipped f16 lane, and the share of voxels each rule stores:
 *
 * | rule    | stored | band error / flips | occupied flips of 4 384 719 |
 * |---------|-------:|-------------------:|----------------------------:|
 * | stencil |  6.64 %|   **0.0000° / 0**  |                   3 789 718 |
 * | 1R      |  4.35 %|      8.6411° / 15  |                   3 846 086 |
 * | 2R      |  8.42 %|      0.3186° / 0   |                   3 739 679 |
 * | 4R      | 15.55 %|      0.0000° / 0   |                   3 567 349 |
 * | 6R      | 22.24 %|      0.0000° / 0   |                   3 355 523 |
 *
 * Two things fall out, and both were the opposite of the expectation.
 *
 * `stencil` is *exact* on the band — zero error, zero flips — because it is
 * defined as the closure of the band under the stencil `safeNormal` actually
 * reads, which no distance-radius rule respects: 1R stores *less* than stencil
 * and is 8.6° wrong. So the band population needs no radius at all.
 *
 * And widening the record set does almost nothing for the interior. Going from
 * 6.64 % stored to 15.55 % — 2.3x the records — moves occupied flips from 86 % to
 * 81 %. There is no radius that rescues it, because the hero is mostly *deep*
 * interior, which is SP20's own structural finding restated: a rule keyed on
 * distance-to-surface cannot help a population that is nowhere near the surface.
 *
 * So the radius arm is not the fallback it was drafted as. If the interior turns
 * out to matter, the fix is not more records — it is to stop reconstructing the
 * *distance* and reconstruct the *normal*, from the occupancy mask's own
 * gradient, which this layout already stores and which degenerates only where the
 * field it replaces is itself arbitrary.
 */
export const SVO_BANDED_DEFAULT_RECORD_SET: SvoBandedRecordSet = "stencil";

/**
 * Encode one leaf, interning its identities into the scene-global palette.
 *
 * `cellRadius_m` is half the diagonal of this leaf's own voxel — the same
 * quantity `rebuildDirtyBrickPayload` divides by to turn a coverage distance
 * into a fraction — so both the record radius and the saturation are in the
 * leaf's own units.
 *
 * Returns `undefined` when the leaf cannot be banded, which is the dense escape:
 * a leaf needing more than 256 distinct *materials*, or one whose banded form
 * would cost more than the dense one. A layout that can only win is a layout
 * that has not been asked the awkward question. The escape is much rarer than it
 * was when the palette interned whole identity words: the hero garden's widest
 * leaf holds 28 materials against 474 distinct identities.
 */
export function encodeBandedLeaf(
  voxels: readonly SvoDenseVoxel[],
  cellRadius_m: number,
  palette: SvoIdentityPalette,
  recordSet: SvoBandedRecordSet = SVO_BANDED_DEFAULT_RECORD_SET,
): SvoBandedLeaf | undefined {
  if (voxels.length !== SVO_BANDED_LEAF_VOXELS) {
    throw new RangeError(`Banded leaf needs exactly ${SVO_BANDED_LEAF_VOXELS} voxels`);
  }
  if (!(cellRadius_m > 0) || !Number.isFinite(cellRadius_m)) {
    throw new RangeError("Banded leaf cell radius must be a positive finite length");
  }
  const occupancy = new Uint32Array(SVO_BANDED_LEAF_MASK_WORDS);
  const band = new Uint32Array(SVO_BANDED_LEAF_MASK_WORDS);
  for (let index = 0; index < SVO_BANDED_LEAF_VOXELS; index += 1) {
    const { fraction } = voxels[index];
    if (fraction > 0) writeBit(occupancy, index);
    if (fraction > 0 && fraction < 1) writeBit(band, index);
  }
  const records = new Uint32Array(SVO_BANDED_LEAF_MASK_WORDS);
  if (recordSet === "stencil") {
    // The normal stencil's own dilation. `safeNormal` clamps `local +/- 1` into
    // the leaf, so a voxel is read for its distance exactly when a band voxel
    // sits within one cell of it along an axis.
    for (let index = 0; index < SVO_BANDED_LEAF_VOXELS; index += 1) {
      if (!readBit(band, index)) continue;
      const x = index & 7, y = (index >> 3) & 7, z = index >> 6;
      writeBit(records, index);
      for (const [dx, dy, dz] of STENCIL_OFFSETS) {
        const nx = Math.min(7, Math.max(0, x + dx));
        const ny = Math.min(7, Math.max(0, y + dy));
        const nz = Math.min(7, Math.max(0, z + dz));
        writeBit(records, nx + ny * 8 + nz * 64);
      }
    }
  } else {
    // Every voxel whose distance the `snorm8` lane would not have saturated,
    // which is where SP20 established the central difference still carries. The
    // band is unioned in unconditionally: its *fraction* is needed whatever its
    // distance does.
    const radius = recordSet.radii * cellRadius_m;
    if (!(radius > 0) || !Number.isFinite(radius)) {
      throw new RangeError("Banded record radius must be a positive finite multiple of the cell radius");
    }
    for (let index = 0; index < SVO_BANDED_LEAF_VOXELS; index += 1) {
      if (readBit(band, index) || Math.abs(voxels[index].distance_m) <= radius) writeBit(records, index);
    }
  }
  // The leaf palette spans *every* occupied voxel, record and resident alike, and
  // it interns the *material* half alone. Splitting it the other way — a material
  // field inside records, a palette outside them — was the first shape tried and
  // encodes one fact twice: 94.8 % of hero leaves hold a single material across
  // all 512 voxels, so a record's material field would be a copy of the leaf's own
  // in nineteen leaves out of twenty.
  const leafSlots = new Map<number, number>();
  const leafPaletteList: number[] = [];
  for (let index = 0; index < SVO_BANDED_LEAF_VOXELS; index += 1) {
    if (!readBit(occupancy, index)) continue;
    const material = voxels[index].materialOwner & 0xffff;
    if (leafSlots.has(material)) continue;
    leafSlots.set(material, leafPaletteList.length);
    leafPaletteList.push(palette.slotFor(material));
  }
  const indexBits = bandedLeafIndexBits(leafPaletteList.length);
  if (!Number.isFinite(indexBits)) return undefined;
  let occupied = 0;
  for (const word of occupancy) occupied += popcount(word);
  const identityIndex = indexBits === 0
    ? new Uint8Array(0)
    : new Uint8Array(Math.ceil(occupied * indexBits / 8));
  // The normal lane spans every voxel and is indexed by voxel; only the material
  // index is rank-compacted, and only when the leaf holds more than one material.
  const normals = new Uint16Array(SVO_BANDED_LEAF_VOXELS);
  {
    let rank = 0;
    for (let index = 0; index < SVO_BANDED_LEAF_VOXELS; index += 1) {
      normals[index] = (voxels[index].materialOwner >>> 16) & 0xffff;
      if (!readBit(occupancy, index)) continue;
      if (indexBits > 0) {
        const entry = leafSlots.get(voxels[index].materialOwner & 0xffff) ?? 0;
        const bit = rank * indexBits;
        identityIndex[bit >> 3] |= entry << (bit & 7);
      }
      rank += 1;
    }
  }
  let recordCount = 0;
  for (const word of records) recordCount += popcount(word);
  const recordDistance = new Float32Array(recordCount);
  const recordFraction = new Float32Array(recordCount);
  let slot = 0;
  for (let index = 0; index < SVO_BANDED_LEAF_VOXELS; index += 1) {
    if (!readBit(records, index)) continue;
    recordDistance[slot] = voxels[index].distance_m;
    recordFraction[slot] = voxels[index].fraction;
    slot += 1;
  }
  const leaf: SvoBandedLeaf = {
    occupancy, records, recordDistance, recordFraction,
    leafPalette: Uint16Array.from(leafPaletteList),
    indexBits,
    identityIndex,
    normals,
    saturation_m: SVO_BANDED_LEAF_SATURATION_RADII * cellRadius_m,
  };
  const dense = SVO_BANDED_LEAF_VOXELS * SVO_BANDED_DENSE_VOXEL_BYTES;
  return bandedLeafBytes(leaf, SVO_BANDED_DENSE_GEOMETRY_BYTES) >= dense ? undefined : leaf;
}

/** The lever's variable, named once so a test and a doc cannot drift from it. */
export const SVO_SOLID_MARCH_OFFSET_FIX_ENVIRONMENT_VARIABLE = "FLUID_SVO_SOLID_MARCH_OFFSET";

/**
 * Whether a fully solid voxel's radiance gather starts at the voxel itself.
 *
 * **On by default**, with `FLUID_SVO_SOLID_MARCH_OFFSET=0` as the rollback. It is
 * a correctness fix at source and it stands on its own: mean 1.16/255, 119 pixels
 * of 368 000 over 16/255, no self-shadow acne, every oracle clean including
 * `terrain-coverage-solid`, and the frame gets faster (17.57 -> 17.19 ms).
 *
 * The shipped radiance seed offsets every gather origin by `1.5 * scale` finest
 * cells along the normal. At a coarse level that is tens of cells, so a voxel
 * metres inside the ground begins its visibility march *above* the terrain and
 * receives direct sun. That is the root cause of the buried-radiance leak this
 * work measured at 22.22/255 mean, and correcting where the gather starts is
 * preferable to omitting the radiance downstream of it — which is the route this
 * replaced, and the reason no skip lever survives.
 *
 * The offset cannot merely shrink: a band voxel needs an offset on the order of
 * its own extent or its gather self-intersects the voxel it started in, which is
 * self-shadow acne and is worse at coarse levels. What separates the two cases is
 * not size but whether there is anywhere outside to move to — a fully solid voxel
 * has none. So this arm removes the offset for fully solid voxels only.
 */
export function svoSolidMarchOffsetFixEnabled(
  environment: Record<string, string | undefined> | undefined
    = typeof process !== "undefined" ? process.env : undefined,
): boolean {
  const raw = environment?.[SVO_SOLID_MARCH_OFFSET_FIX_ENVIRONMENT_VARIABLE];
  if (raw === undefined || raw === "") return true;
  if (raw === "0" || raw === "off") return false;
  if (raw !== "1" && raw !== "on") {
    throw new RangeError(`${SVO_SOLID_MARCH_OFFSET_FIX_ENVIRONMENT_VARIABLE} must be "on" or "off"`);
  }
  return true;
}

/** The lever's variable, named once so a test and a doc cannot drift from it. */
export const SVO_SOLID_DIRECT_OCCLUSION_ENVIRONMENT_VARIABLE = "FLUID_SVO_SOLID_DIRECT_OCCLUSION";

/**
 * Whether a fully solid receiver is opaque to direct light.
 *
 * **Off by default** until its own hash, pixel diff and PNG are blessed; `on` is
 * the arm under test. Measured on `hero-garden-hose` at 800x460 against the
 * shipped default arm (`0x1ab2434b`): `0x5b393c3d`, 70.10 % of pixels differ,
 * whole-frame mean 12.30/255, max channel 91/255, mean luma 116.99 -> 105.70, and
 * the change is a darkening in 69.99 % of pixels against 0.11 % brighter. Sky is
 * byte-identical, no pixel goes black that was not, and `terrain-coverage-solid`
 * (4 187 294 of 4 187 294), `radiance-black-pages` (0 of 44) and
 * `frame-carries-radiance` (256/256) all stay clean.
 *
 * **It scores worse against the reference plate, and that is the reason it is
 * still off.** `npm run gate:hero-fidelity`, with the grade re-solved separately
 * for each arm so the comparison is like-for-like: off 13.46 mean ΔE₀₀ / 2.456
 * octaves at its own solved exposure 0.3980, on **14.77 / 2.651** at its own
 * solved 0.4138. The lever loses by 1.31 ΔE₀₀ and 0.195 octaves, far above the
 * 0.25 / 0.02 measurement floor, and it loses in *every* region: coping +0.80,
 * stones +1.85, canopy +3.45, pond +0.47.
 *
 * That is a finding about the scene, not a defect in this guard. Removing light
 * the ground was never entitled to makes the frame match the plate *uniformly
 * worse*, which is what it looks like when the leak was doing the job of a term
 * the scene does not have: global fill and bounce. The plate is 55 % water this
 * frame does not render, and `lib/hero-garden-scene.ts`'s grade note already
 * records canopy fill-and-bounce as open. So this guard should land *with* a
 * legitimate fill source, not before one — physical correctness and plate score
 * move together only once the light it removes has somewhere real to come from.
 *
 * `directVisibility` marches four samples at `cellScale * {2.5, 6, 18, 54}` where
 * `cellScale = max(mappingCellSize) * exp2(radianceFloorLevel())`. On
 * `hero-garden-hose` the floor level is 3, so the *first* sample is 2.5 x 8 = 20
 * finest cells — 12.5 cm at a 6.25 mm cell — while the hero's terrain is only
 * 0.14-0.38 m thick. For a voxel just under the surface all four samples are
 * already in open air above the ground, the march never samples the ground
 * between the voxel and the sun, `1 - coverage` is 1 at every step, and solid
 * earth receives *full* direct sun. `docs/svo-banded-leaf-payload-handoff.md`
 * measures that leak at roughly 21 of the 22.22/255 mean it attributes to buried
 * radiance overall.
 *
 * It is a mis-scaled trade rather than an oversight, and the gather comment in
 * `lib/webgpu-svo-live-derived-builder.ts` says so: the steps are expressed in
 * *receiver* widths so that a step from an eight-cell block does not land inside
 * the block that emitted it. Preserving the step-to-sample-width ratio is right
 * for **radiance gathering**, where the hazard is self-sampling. It is wrong for
 * **occlusion**, where the hazard is skipping the occluder. One scaling cannot
 * serve both.
 *
 * Rescaling the occlusion march is not the fix, and that was **measured** rather
 * than assumed. An arm taking the same four steps in *finest* cells instead of
 * receiver widths — 8x shorter at this floor level, so the first sample is 1.6 cm
 * rather than 12.5 cm — recovers only **1.42/255 of the 12.30/255**, and once this
 * guard is in it adds **0.013/255** on top of it. The coarse pyramid holds one
 * average coverage per floor-level block, so it cannot tell "just below the
 * surface" from "just above" at any step length in that family: a sample short
 * enough to find the ground under a buried voxel reads the receiver's own
 * straddling block for an exposed one, which is the same self-shadow acne the
 * receiver-width scaling buys, arriving as a uniform dimming instead of speckle.
 *
 * So the question is answered where the answer exists — in the *fine* fraction. A
 * fully solid voxel sees no sun, exactly and independently of any step length, so
 * this arm returns zero transmittance before taking any step, on the same
 * `bandedFractionAt >= 1.0` test `FLUID_SVO_SOLID_MARCH_OFFSET` uses. Partially
 * solid voxels are exactly the voxels the surface passes through, are entitled to
 * sun, and keep the shipped march untouched.
 *
 * A stricter receiver test was measured and **rejected**: also requiring the
 * floor-level texel's own opacity to be full (`coverage >= 1.0`), on the theory
 * that a block straddling the surface should not be zeroed because its centre cell
 * happens to be solid. It closes only 5.26/255 of the leak, leaves the hero's
 * contact shadow at luma 71.0 where both independent routes put it at 50.8 and
 * 51.2, and converges with neither — the level-3 opacity texel is a filtered
 * average that a genuinely buried block does not reach 1.0 on. Under-correcting,
 * not safer.
 */
export function svoSolidDirectOcclusionEnabled(
  environment: Record<string, string | undefined> | undefined
    = typeof process !== "undefined" ? process.env : undefined,
): boolean {
  const raw = environment?.[SVO_SOLID_DIRECT_OCCLUSION_ENVIRONMENT_VARIABLE];
  if (raw === undefined || raw === "" || raw === "0" || raw === "off") return false;
  if (raw !== "1" && raw !== "on") {
    throw new RangeError(`${SVO_SOLID_DIRECT_OCCLUSION_ENVIRONMENT_VARIABLE} must be "on" or "off"`);
  }
  return true;
}
