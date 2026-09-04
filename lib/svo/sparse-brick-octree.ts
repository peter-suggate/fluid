/**
 * Sparse-brick octree primitives shared by scene, rendering, and fluid work.
 *
 * The CPU helpers in this file only produce deterministic addresses and testable
 * topology plans. Runtime publication is device-to-device: counts can be
 * authored by an earlier GPU pass and no map/readback is needed to publish the
 * tree or size later compute/draw work.
 */

import { SVO_BRICK_LIFECYCLE } from "./svo-brick-occupancy";
import { SVO_GBUFFER_NORMAL_OCT8_ABSENT } from "./svo-gbuffer";

export type SparseBrickSize = 4 | 8;

export interface SparseBrickCoordinate {
  x: number;
  y: number;
  z: number;
}

export interface SparseBrickNodePlan {
  index: number;
  level: number;
  morton: bigint;
  coordinate: SparseBrickCoordinate;
  childMask: number;
  firstChild: number;
  childCount: number;
  leafIndex: number;
}

export interface SparseBrickLeafPlan {
  index: number;
  nodeIndex: number;
  morton: bigint;
  coordinate: SparseBrickCoordinate;
  voxelOffset: number;
  /** Terminal representation. Zero is a conventional voxel brick. */
  terminalKind: SparseBrickLeafTerminalKind;
  /** Kind-specific accepted-record index; invalid for a voxel terminal. */
  terminalIndex: number;
}

export const SPARSE_BRICK_LEAF_TERMINAL = Object.freeze({
  voxels: 0,
  planarBoundary: 1,
} as const);

export type SparseBrickLeafTerminalKind =
  typeof SPARSE_BRICK_LEAF_TERMINAL[keyof typeof SPARSE_BRICK_LEAF_TERMINAL];

export interface SparseBrickLeafTerminal {
  readonly kind: SparseBrickLeafTerminalKind;
  readonly index: number;
}

export const SPARSE_BRICK_VOXEL_TERMINAL: SparseBrickLeafTerminal = Object.freeze({
  kind: SPARSE_BRICK_LEAF_TERMINAL.voxels,
  index: 0xffff_ffff,
});

export interface SparseBrickPlan {
  brickSize: SparseBrickSize;
  maximumDepth: number;
  /** Global node offsets for levels 0..maximumDepth, followed by the end. */
  levelOffsets: readonly number[];
  nodes: readonly SparseBrickNodePlan[];
  leaves: readonly SparseBrickLeafPlan[];
  voxelCount: number;
}

export interface SparseBrickPlanOptions {
  brickSize: SparseBrickSize;
  /** If omitted, the smallest depth containing every non-negative coordinate is used. */
  maximumDepth?: number;
}

export const SPARSE_BRICK_INVALID_INDEX = 0xffffffff;
export const SPARSE_BRICK_NO_OWNER = 0xffff;
export const SPARSE_BRICK_MAX_MORTON_BITS = 21;

/**
 * Per-voxel payload lanes. Every lane is a byte range of the one payload arena.
 *
 * The five original lanes are per-voxel. The four the banded leaf payload adds
 * are not all per-voxel — see {@link SparseBrickPayloadLaneLayout.elements} —
 * but they are still byte ranges of the same arena, and deliberately so: the dry
 * render path binds one buffer and resolves every lane from uniform offsets, so
 * a lane that lived in its own buffer would cost a binding in five pipelines.
 */
export type SparseBrickPayloadLaneName =
  | "geometry"
  | "velocity"
  | "materialOwners"
  | "sceneGeometry"
  | "sceneMaterialOwners"
  | "sceneOccupancy"
  | "sceneRecordMask"
  | "sceneBandedHeader"
  | "sceneBandedBlob"
  | "sceneBandedRecords";

/**
 * How one channel of a lane is stored. `f32` is the historical default and the
 * only one a lane needs to declare nothing for.
 *
 * A narrowed channel is a *storage* decision, never a meaning one: the value a
 * consumer reads still carries the channel's documented units, with the single
 * exception called out on {@link SparseBrickSceneGeometryFormat}.
 *
 * `u1` is the occupancy predicate's own width, and the reason
 * {@link sparseBrickLaneStrideBytes} has to express a fraction of a byte: an
 * occupancy mask is one bit a voxel and 512 voxels of it are the 64-byte per-leaf
 * mask the banded layout reads, laid out by the same `offset + voxel * stride`
 * arithmetic every other lane uses. `u16` is a global identity-palette slot.
 */
export type SparseBrickChannelFormat = "f32" | "f16" | "snorm8" | "unorm8" | "u32" | "u16" | "u1";

const CHANNEL_FORMAT_BITS: Readonly<Record<SparseBrickChannelFormat, number>> = Object.freeze({
  f32: 32, f16: 16, snorm8: 8, unorm8: 8, u32: 32, u16: 16, u1: 1,
});

/** What a lane allocates one element for. Everything that predates banding is per-voxel. */
export type SparseBrickLaneElements = "voxels" | "leaves";

export interface SparseBrickPayloadLaneLayout {
  readonly name: SparseBrickPayloadLaneName;
  readonly strideBytes: number;
  /** Channel names in storage order. */
  readonly channels: readonly string[];
  /**
   * Storage format per channel, in channel order. Omitted means every channel
   * is `f32`, which is what makes `channels.length * 4 === strideBytes` hold on
   * every lane that predates narrowing.
   */
  readonly channelFormats?: readonly SparseBrickChannelFormat[];
  /**
   * What one element of this lane is. Defaults to `voxels`, which every lane
   * that predates the banded layout is.
   *
   * `leaves` is how a *per-leaf* structure lives in the per-voxel arena without
   * the arithmetic changing shape: `voxelOffset` is exactly `leafIndex * 512`
   * (verified 693/693 aligned; sole GPU writer
   * `lib/webgpu-sparse-brick-topology-mutation.ts:200`), so a reader recovers the
   * leaf slot as `voxelOffset >> 9` and indexes at `offset + slot * stride`.
   */
  readonly elements?: SparseBrickLaneElements;
  /**
   * Multiplier on the element count, for a lane that is *suballocated* rather
   * than indexed one-to-one.
   *
   * The banded record arena is the only such lane: records are 6.6-23 % of the
   * voxels a scene holds, so it is sized at a fraction of voxel capacity and
   * bump-allocated per leaf. That fraction is a capacity decision with a measured
   * high-water mark, which is the honest way to size it — a per-leaf reservation
   * at the p99 size class would cost more than the dense lane it replaces.
   */
  readonly elementScale?: number;
}

/**
 * The stride a lane's declared channels round up to.
 *
 * WGSL storage buffers are word-addressed, so a lane is only ever addressable at
 * 1, 2 or 4 bytes a voxel (2 or 4 voxels to the word) or at a whole number of
 * words. Three bytes straddles a word boundary and would cost more in unpack ALU
 * than it saves in bandwidth, so it rounds up to four and the slack is declared
 * spare rather than sold as a third channel.
 *
 * Below a byte the same argument runs the other way and the answer is a
 * *fraction* of a byte, because a mask lane's whole point is that 8, 16 or 32
 * voxels share one word. Only powers of two are expressible: the bit position of
 * a voxel inside its word has to be a shift of `voxel & mask`, so three declared
 * bits round up to four (two voxels a byte) and the spare bit is slack, exactly
 * as three declared bytes round up to four above.
 */
export function sparseBrickLaneStrideBytes(
  channelFormats: readonly SparseBrickChannelFormat[],
): number {
  const bits = channelFormats.reduce((sum, format) => sum + CHANNEL_FORMAT_BITS[format], 0);
  if (bits < 1) throw new RangeError("A lane must declare at least one bit");
  if (bits < 8) return bits === 1 ? 1 / 8 : bits === 2 ? 1 / 4 : 1 / 2;
  const bytes = Math.ceil(bits / 8);
  if (bytes > 4) return Math.ceil(bytes / 4) * 4;
  return bytes > 2 ? 4 : bytes > 1 ? 2 : 1;
}

/** Voxels that share one 32-bit word of a lane. Always 1 for a word-or-wider lane. */
export function sparseBrickLaneVoxelsPerWord(strideBytes: number): number {
  return strideBytes >= 4 ? 1 : 4 / strideBytes;
}

/**
 * Which lanes a world allocates. The lane set is a property of the *world*, not
 * of the renderer, so turning water on is this one string and nothing else.
 *
 * `full` is the five-lane layout every existing world uses. The duplication is
 * deliberate: the solver owns `geometry`/`velocity`/`materialOwners` and the
 * live voxeliser owns the two `scene*` lanes, so an advance cannot clobber
 * authored geometry.
 *
 * `dry` is for a world with no solver. Nothing writes the three dynamic lanes
 * on such a world — `encodeFromDenseFields` and `encodePublish` are the only
 * writers and neither is ever encoded — so 36 of 56 bytes per voxel are
 * allocated and never touched. See the comment at
 * lib/webgpu-octree-sparse-bricks.ts:693-698, which already records that the
 * dynamic lane "stays zero-filled for the whole session" on the hero garden.
 *
 * `dry` also narrows the surviving geometry lane from four channels to two,
 * which is a *pruning* of provably dead channels and not a fidelity choice. The
 * scene voxeliser writes exactly two of the four (lib/webgpu-sparse-scene-
 * proxies.ts:1972-1973); `fluidSignedDistance` and `pressure` are solver state
 * with no writer on this lane at all, and no reader either. Both survivors are
 * load-bearing and neither may be dropped without a visible regression:
 *
 *   - `solidSignedDistance` is the *only* smooth-normal source in the world. It
 *     is central-differenced by `safeNormal`
 *     (lib/webgpu-svo-live-derived-builder.ts:246-252, and again at :359 for
 *     the feedback solve), and the voxeliser deliberately keeps the field's own
 *     value at the voxel centre for exactly that reason — see its own comment
 *     at proxies:1911-1914. The primary meanwhile shades from
 *     `dryVoxelFaceNormal` (lib/webgpu-svo-dry-scene.ts:4869), one of six axis
 *     directions, which is why every surface in the frame terraces at every
 *     lattice. Deleting this channel would make that permanent.
 *   - `solidFraction` is the opacity-pyramid source (`sceneCoverage`, derived
 *     builder :291) and is also what *defines* occupancy: the voxeliser writes
 *     an identity only where the fraction is positive (proxies:1974), so
 *     `material != 0` and `fraction > 0` are one predicate today.
 */
export type SparseBrickPayloadProfileName = "full" | "dry";

/** Select payload lanes from ownership, independently of scene content. */
export function sparseBrickPayloadProfileForOwnership(
  fluidEnabled: boolean,
  rendererOnly: boolean,
  compactProfile: SparseBrickPayloadProfileName,
): SparseBrickPayloadProfileName {
  return !fluidEnabled || rendererOnly ? compactProfile : "full";
}

const GEOMETRY_CHANNELS_FULL = ["fluidSignedDistance", "solidSignedDistance", "solidFraction", "pressure"] as const;
/** Storage order is preserved from the full lane so channel *meaning* is stable. */
const GEOMETRY_CHANNELS_SCENE = ["solidSignedDistance", "solidFraction"] as const;

export const SPARSE_BRICK_PAYLOAD_PROFILES: Readonly<Record<
  SparseBrickPayloadProfileName, readonly SparseBrickPayloadLaneLayout[]
>> = Object.freeze({
  full: Object.freeze([
    { name: "geometry", strideBytes: 16, channels: GEOMETRY_CHANNELS_FULL },
    { name: "velocity", strideBytes: 16, channels: ["velocityX", "velocityY", "velocityZ", "liquidFraction"] },
    { name: "materialOwners", strideBytes: 4, channels: ["materialOwner"] },
    { name: "sceneGeometry", strideBytes: 16, channels: GEOMETRY_CHANNELS_FULL },
    { name: "sceneMaterialOwners", strideBytes: 4, channels: ["materialOwner"] },
  ] as const),
  dry: Object.freeze([
    { name: "sceneGeometry", strideBytes: 8, channels: GEOMETRY_CHANNELS_SCENE },
    { name: "sceneMaterialOwners", strideBytes: 4, channels: ["materialOwner"] },
  ] as const),
});

/**
 * How a dry world stores a leaf's 512-voxel payload.
 *
 * The lever behind `lib/svo-banded-leaf-payload.ts`, whose module comment is the
 * layout's own derivation and whose encoder is its executable spec. Three rungs,
 * one lever, because a `banded` arm without the occupancy mask is not a legal
 * state — occupancy is what makes a voxel storable nowhere — and two independent
 * levers can express it.
 *
 *   - `dense` is what shipped: every voxel carries its own geometry word and its
 *     own 4-byte material-owner word, so a solid leaf spends 6 144 bytes saying
 *     "solid" 512 times. **The default.**
 *   - `occupancy` adds the 1-bit-a-voxel occupancy mask and moves the occupancy
 *     *predicate* onto it, and changes nothing else. It exists to be bisected
 *     against: it is hash-identical by construction, because
 *     `bandedLeafOccupied` and the `material != 0` test it replaces are the same
 *     predicate over the same data, and it costs 64 bytes a leaf (~1 %). A hash
 *     that moves here is a bug in the predicate move and nothing else.
 *   - `banded` is the layout: masks, a per-leaf header, a suballocated record
 *     arena carrying geometry only, the material as a per-leaf palette with an
 *     index array over occupied voxels whose width is zero on the 94.8 % of hero
 *     leaves that hold one material, and the baked normal as a raw `u16` per
 *     voxel, in a fixed 256-word lane at the head of the leaf's blob. An
 *     interior voxel costs two bits — both already inside the fixed masks — plus
 *     its normal.
 *
 *     **The identity word's two halves are different populations, so they are
 *     stored differently.** The high half is now a per-voxel baked normal
 *     ({@link sparseBrickSceneIdentityWordCodecWGSL}) rather than a per-object
 *     owner id. Measured on `hero-garden-hose` at depth 0: interning the whole
 *     word gives **29.05** entries a leaf (max 474, 51.9 % of leaves past a 4-bit
 *     index, 0.11 % past the 256-entry palette entirely), and interning the
 *     material half alone gives **1.026** (max 28, 94.8 % of leaves at exactly
 *     one). So the palette interns materials and the normal gets a lane. See
 *     `lib/svo-banded-leaf-payload.ts` for why that lane spans every occupied
 *     voxel rather than riding the geometry records.
 *
 * The staging is why `occupancy` is a rung rather than a footnote. `banded` moves
 * occupancy, fraction, distance *and* identity to a new decode path at once, so a
 * hash that moves has four suspects; landing the predicate first leaves one.
 */
export type SparseBrickLeafPayloadMode = "dense" | "occupancy" | "banded";

export const SPARSE_BRICK_LEAF_PAYLOAD_MODES: readonly SparseBrickLeafPayloadMode[]
  = Object.freeze(["dense", "occupancy", "banded"]);

/** Voxels in a leaf of the only brick size the live scene path builds. */
const BANDED_VOXELS_PER_LEAF = 512;

/**
 * Record slots the banded arena reserves, as a fraction of voxel capacity.
 *
 * Measured, not guessed: the stencil record set is **6.63 %** of the hero
 * garden's voxels (re-measured 2026-08-07 on the published depth-0 arena,
 * 12 724 leaves / 6.51 M voxels; the pre-cutover figure was 6.64 % and it has not
 * moved, because the record set is a function of `solidFraction` and the cutover
 * touched only the identity word), and 15.7-23.1 % of a `garden-svo-lighting`
 * leaf — that study is the crowded case and the one this has to hold. 25 %
 * carries it with room, costs 1 byte a voxel of *capacity* against the 4 the
 * dense lane spends on every voxel, and the allocator reports its high-water mark
 * so the reservation is checkable rather than assumed.
 *
 * A shared arena rather than a per-leaf reservation because the size-class
 * distribution is long-tailed: the p99 hero leaf holds 301 records where the
 * median holds none, so reserving the p99 class per leaf would cost 1 204 bytes a
 * leaf on a scene whose median leaf needs zero.
 */
export const SPARSE_BRICK_BANDED_RECORD_CAPACITY_FRACTION = 0.25;

/**
 * Bytes a leaf reserved for its material palette, its index and its normal lane.
 *
 * A global bump allocator hands the blob out, so this is a *mean* budget over the
 * published leaves rather than a per-leaf cap — a leaf may take more as long as
 * the scene's total fits, and the allocator raises
 * {@link SPARSE_BRICK_BANDED_OVERFLOW}`.blob` rather than corrupting anything if
 * it does not.
 *
 * Three terms, measured on `hero-garden-hose` at depth 0 (77.28 % occupancy,
 * 1.026 materials a leaf, 94.8 % of leaves at exactly one material):
 *
 *   - **normals**: a flat 256 words — one `u16` a voxel, all 512 of them.
 *     **1 024 bytes**, and 78 % of the whole layout.
 *   - **palette**: `ceil(materials / 2)` words, two `u16` to a word. 4 bytes on
 *     nineteen leaves in twenty.
 *   - **index**: `ceil(occupied * indexBits / 32)` words, and `indexBits` is zero
 *     on those same nineteen leaves. Mean under 1 byte.
 *
 * Measured mean is therefore ~1 029 bytes. The reservation is **1 152**: the
 * normal lane is fixed at 1 024 and can never overflow, and the remaining 128
 * covers palette and index. So the common case — a single-material leaf, whatever
 * its occupancy — is exactly 1 028 bytes, and the 12 % headroom absorbs the
 * multi-material tail against a global bump allocator.
 *
 * This constant carried 96 before the unified-voxel cutover, when the blob held a
 * palette of whole identity words and nothing else. That budget is no longer
 * expressible: with the normal in the word, interning it needed 29.05 entries a
 * leaf and an 8-bit index on half of them, which is 380 bytes a leaf of palette
 * and index — four times the old reservation to store the same information *less*
 * directly than the raw 792-byte lane does. See `lib/svo-banded-leaf-payload.ts`.
 */
export const SPARSE_BRICK_BANDED_BLOB_BYTES_PER_LEAF = 1_152;

/**
 * The banded layout's per-leaf header, in `u32` words.
 *
 * Four words, 16 bytes, which with the two 64-byte masks is the layout's fixed
 * 144 bytes a leaf — 11 % of the measured 1 308-byte depth-0 leaf, against 41 %
 * before the normal lane existed.
 *
 * It stays at four because the blob's three arrays are ordered so that every one
 * of them has a derivable start: normals first at a fixed 256 words, then the
 * material palette, then the index. An earlier arrangement put normals last and
 * needed a fifth word to address them; ordering is cheaper than an offset.
 */
export const SPARSE_BRICK_BANDED_HEADER_WORDS = 4;

/** Words of the banded arena reserved before the first leaf's blob, for allocator counters. */
export const SPARSE_BRICK_BANDED_ALLOCATOR_WORDS = 4;

/**
 * The lanes a leaf payload mode adds on top of the profile's own.
 *
 * Appended after the profile's lanes so a `dense` world's offsets are unchanged
 * to the byte and the two shipped profiles stay exactly where they were.
 */
const LEAF_PAYLOAD_MODE_LANES: Readonly<Record<
  SparseBrickLeafPayloadMode, readonly SparseBrickPayloadLaneLayout[]
>> = Object.freeze({
  dense: Object.freeze([] as const),
  occupancy: Object.freeze([
    { name: "sceneOccupancy", strideBytes: 1 / 8, channels: ["occupancy"], channelFormats: ["u1"] },
  ] as const),
  banded: Object.freeze([
    { name: "sceneOccupancy", strideBytes: 1 / 8, channels: ["occupancy"], channelFormats: ["u1"] },
    { name: "sceneRecordMask", strideBytes: 1 / 8, channels: ["record"], channelFormats: ["u1"] },
    // Placeholder width: `resolveSparseBrickPayloadLayout` replaces this lane with
    // the world's own record width and capacity fraction, because both are
    // resolved arguments rather than table constants.
    {
      name: "sceneBandedRecords", strideBytes: 4, channels: GEOMETRY_CHANNELS_SCENE,
      channelFormats: ["f16", "unorm8"], elements: "voxels", elementScale: 1,
    },
    {
      name: "sceneBandedHeader", strideBytes: SPARSE_BRICK_BANDED_HEADER_WORDS * 4, elements: "leaves",
      channels: ["recordBase", "blobBase", "counts", "scale"],
      channelFormats: ["u32", "u32", "u32", "u32"],
    },
    {
      name: "sceneBandedBlob", strideBytes: SPARSE_BRICK_BANDED_BLOB_BYTES_PER_LEAF, elements: "leaves",
      channels: Object.freeze(
        Array.from({ length: SPARSE_BRICK_BANDED_BLOB_BYTES_PER_LEAF / 4 }, (_, i) => `word${i}`)),
      channelFormats: Object.freeze(
        Array.from({ length: SPARSE_BRICK_BANDED_BLOB_BYTES_PER_LEAF / 4 },
          (): SparseBrickChannelFormat => "u32")),
    },
  ] as const),
});

/**
 * How a dry world stores the two surviving scene-geometry channels.
 *
 * The lane is the whole arena at refinement depth: at depth 3 the hero garden's
 * 227 168 leaves hold 116.3 M voxels, so every byte a voxel costs is 116 MB, and
 * the leaf, node and state records together are rounding error against it. This
 * enum is therefore the depth-3 memory knob, and it is a *storage* choice made
 * one channel at a time:
 *
 *   - `f32x2` (8 B) is what shipped: two plain f32 words a voxel.
 *   - `f16-unorm8` (4 B) keeps the distance in metres at half precision and the
 *     fraction at 8 bits, in one word a voxel with 8 bits spare. Word-aligned,
 *     so no invocation ever shares a word with another. **This is the default.**
 * A third arm, `snorm8-unorm8` (2 B), stored the distance as a signed byte over a
 * band scaled to the voxel's own cell, two voxels to the word. It was measured on
 * device, **rejected, and then removed** — the numbers are worth keeping even
 * though the code is not. A solid scene is mostly deep *interior*: 4.5 M of the
 * hero garden's 5.3 M voxels saturate whatever the band, and the resulting zero
 * gradients flipped 3 299 normals to the `vec3f(0,1,0)` fallback against 5 for
 * f16. Mean `safeNormal` angular error ran 0.386-6.95 deg across bands against
 * f16's 0.008 deg. Nothing narrower than a byte is worth trying on this channel
 * without first changing *which voxels are stored at all* — which is what the
 * banded leaf payload does, and why that is the live line of work.
 *
 * `solidFraction` is 8-bit under the narrowed format, and that is the
 * strongest case in the set rather than a compromise: the node-mip opacity
 * pyramid derived *from* this channel has stored it as `rg8unorm` since SP12,
 * so the payload lane was keeping at f32 what its own consumer immediately
 * rounded to 8 bits. The one thing that had to be preserved by hand is the
 * occupancy predicate — the voxeliser writes a material identity exactly where
 * `fraction > 0` — so the encoder floors any positive fraction at 1/255 rather
 * than letting round-to-nearest retire a voxel the material lane still owns.
 *
 * `solidSignedDistance` is the only smooth-normal source in the world and the
 * reason this is a lever rather than an edit. Both surviving arms store it in
 * metres. A narrowed arm need not: `safeNormal` central-differences six samples
 * drawn from one leaf — one level, hence one cell size — and then normalises, so a
 * per-leaf scale factor cancels exactly, which is what let the removed
 * `snorm8-unorm8` arm store cell-band units. Any arm that does that again must say
 * so on {@link OctreeSparseBrickWorld}'s published field units, because a consumer
 * reading a scaled value as if it were metres is silent and wrong.
 */
export type SparseBrickSceneGeometryFormat = "f32x2" | "f16-unorm8";

export const SPARSE_BRICK_SCENE_GEOMETRY_FORMATS: readonly SparseBrickSceneGeometryFormat[]
  = Object.freeze(["f32x2", "f16-unorm8"]);

/**
 * Which arm a live dry scene stores its scene-geometry lane in.
 *
 * `f16-unorm8` is the default and the reasoning for it — including the device
 * table that rejected and removed a 2-byte third arm — is on
 * {@link SparseBrickSceneGeometryFormat} above.
 *
 * It lives here rather than with the producer because it is now read by both
 * ends: the voxeliser picks the arm, and the dry primary's shading normal is a
 * gradient of the very lane it picked, so the renderer has to emit the matching
 * decoder. `lib/webgpu-octree-sparse-bricks.ts` re-exports it for the callers
 * that have always imported it from there.
 */
export function octreeLiveSceneSceneGeometryFormat(
  environment: Record<string, string | undefined> | undefined
    = typeof process !== "undefined" ? process.env : undefined,
): SparseBrickSceneGeometryFormat {
  const raw = environment?.FLUID_SVO_SCENE_GEOMETRY;
  if (raw === undefined || raw === "") return "f16-unorm8";
  if (!SPARSE_BRICK_SCENE_GEOMETRY_FORMATS.includes(raw as SparseBrickSceneGeometryFormat)) {
    throw new RangeError(`FLUID_SVO_SCENE_GEOMETRY must be one of ${SPARSE_BRICK_SCENE_GEOMETRY_FORMATS.join(", ")}`);
  }
  return raw as SparseBrickSceneGeometryFormat;
}

/**
 * How a dry world stores a leaf's 512-voxel payload.
 *
 * `FLUID_SVO_LEAF_PAYLOAD=dense|occupancy|banded`, **defaulting to `dense`**,
 * which is the arm that shipped. See {@link SparseBrickLeafPayloadMode} for what
 * each rung stores and `lib/svo-banded-leaf-payload.ts` for the layout's own
 * derivation and its executable encoder.
 *
 * Every identity reader is now cut over — the dry primary, its shadow walk, the
 * LOD tier, both probes, the derived builder, the brick-occupancy summary and the
 * two readback oracles all resolve identity through
 * {@link sparseBrickSceneIdentityCodecWGSL} rather than the flat lane — so
 * `banded` is a *correct* arm rather than a producer-only one. The default stays
 * on the shipped arm for two reasons that are arithmetic, not caution:
 *
 * 1. **The saving is not available yet.** `encodeBandedLeaves` builds the palette
 *    and the normal lane by reading the dense identity lane the voxeliser has just
 *    written, so a banded world still allocates it (see
 *    {@link SPARSE_BRICK_BANDED_PRODUCER_DENSE_LANES}). Until that producer stops
 *    needing a whole-arena staging lane, `banded` *adds* the mask, header, blob
 *    and record lanes — 3.53 bytes a voxel of reserved capacity — on top of the 8
 *    the dry profile already spends. Retiring the two staging lanes is what turns
 *    that into the 8 → 2.56 B/voxel the measured occupancy supports.
 * 2. A lever that defaults to an arm nobody has cleared *on device* is how one
 *    recorded "baseline" hash became the `snorm8` arm under test, and it cost
 *    three agents time. The banded arm is cleared for *fidelity* — mean 0.54/255,
 *    max 7/255, zero pixels over 16/255, less visible than the `f16-unorm8`
 *    geometry lane already shipping at 11/255.
 *
 * So the arm to measure is `FLUID_SVO_LEAF_PAYLOAD=banded`, and what it buys
 * today is the per-cell predicate in `traceLeafPayload` — one occupancy bit
 * instead of a strided 4-byte identity load for every cell the walk *rejects* —
 * rather than footprint.
 *
 * `occupancy` is the bisection rung, not a product state: it costs 64 bytes a
 * leaf and saves no storage, and exists so that a hash which moves under `banded`
 * can be attributed to the storage rather than to the predicate move. It is the
 * arm that carries the predicate move *alone*: same dense identity lane, same
 * dense geometry lane, `bandedOccupied` as the only new question.
 */
export function octreeLiveSceneLeafPayloadMode(
  environment: Record<string, string | undefined> | undefined
    = typeof process !== "undefined" ? process.env : undefined,
): SparseBrickLeafPayloadMode {
  const raw = environment?.FLUID_SVO_LEAF_PAYLOAD;
  if (raw === undefined || raw === "") return "dense";
  if (!SPARSE_BRICK_LEAF_PAYLOAD_MODES.includes(raw as SparseBrickLeafPayloadMode)) {
    throw new RangeError(`FLUID_SVO_LEAF_PAYLOAD must be one of ${SPARSE_BRICK_LEAF_PAYLOAD_MODES.join(", ")}`);
  }
  return raw as SparseBrickLeafPayloadMode;
}

/**
 * Half-width of a stored distance band, in multiples of the voxel's own cell
 * radius (`0.5 * |cellExtent|`, the same quantity the voxeliser already computes
 * to invert its coverage law).
 *
 * Originally the quantisation band of a removed 2-byte geometry arm. Its live
 * consumer is now {@link SVO_BANDED_LEAF_SATURATION_RADII} — the banded leaf
 * payload's saturation, the value a voxel outside the record set reconstructs to —
 * which is why the constant outlived the arm that motivated it. The study below is
 * retained because it is the reason the number is 4 rather than 1.
 *
 * Both ends of the band cost something. Narrow bands quantise finely but
 * saturate, and a saturated neighbour flattens one component of the central
 * difference; wide bands never saturate but step coarsely. Against the f32 arm
 * over 11 528 shaded voxels of analytic planes, spheres, boxes and terrain ramps,
 * the mean/p99/max angular error of `safeNormal` looked like a clean optimum at
 * 4R:
 *
 *   1R  4.020 / 23.579 / 45.803 deg
 *   2R  0.270 /  1.311 / 14.571 deg
 *   3R  0.335 /  1.387 /  5.166 deg
 *   4R  0.395 /  1.342 /  2.112 deg
 *
 * That study was wrong about the scene because of what it sampled. It scored only
 * voxels the renderer shades — coverage strictly between 0 and 1 — which is a
 * *surface* population, where every distance is within a cell of zero and no band
 * saturates. A real scene is mostly deep interior: on the hero garden 4.5 M of
 * 5.3 M voxels sit outside even the 4R band, clamp to +/-1, and hand `safeNormal`
 * a zero gradient. The surface voxels the study measured are the minority whose
 * *neighbours* are those interior voxels, so the saturation reaches them too.
 *
 * The lesson is the sampling, not the number: a quantisation study must be run
 * over the population the field actually holds, not over the population whose
 * answers are consumed.
 */
export const SPARSE_BRICK_SCENE_DISTANCE_BAND_RADII = 4;

const SCENE_GEOMETRY_LANES: Readonly<Record<SparseBrickSceneGeometryFormat, SparseBrickPayloadLaneLayout>>
  = Object.freeze({
    "f32x2": { name: "sceneGeometry", strideBytes: 8, channels: GEOMETRY_CHANNELS_SCENE },
    "f16-unorm8": {
      name: "sceneGeometry", strideBytes: 4, channels: GEOMETRY_CHANNELS_SCENE,
      channelFormats: ["f16", "unorm8"],
    },
  });

/**
 * The one definition of the narrowed scene-geometry encoding, shared by the
 * voxeliser that writes it (`sparseSceneProxyVoxelizationShaderFor`) and the
 * derived builder that reads it (`liveSvoDerivedBuildWGSLFor`).
 *
 * Emitted as WGSL rather than duplicated as two hand-matched bit expressions,
 * because a producer and a consumer that disagree about a shift produce a scene
 * that renders — wrongly — instead of one that fails to compile.
 *
 * `f32x2` emits nothing: that arm addresses the lane directly, exactly as it did
 * before formats existed, and its shader text is unchanged to the character.
 */
export function sparseBrickSceneGeometryCodecWGSL(format: SparseBrickSceneGeometryFormat): string {
  if (format === "f32x2") return "";
  return /* wgsl */ `
// scene geometry lane, f16-unorm8: one word a voxel, 8 bits spare.
${SCENE_FRACTION_WGSL}
fn sceneGeometryWord(base:u32,voxel:u32)->u32{return base+voxel;}
// A voxel owns its whole word here, so the shift is zero and the mask is total.
// Naming them anyway is what lets the writer be one expression across formats.
fn sceneGeometryShift(voxel:u32)->u32{return 0u;}
fn sceneGeometryMask(voxel:u32)->u32{return 0xffffffffu;}
fn sceneDistanceOf(word:u32,voxel:u32)->f32{return unpack2x16float(word).x;}
fn sceneFractionOf(word:u32,voxel:u32)->f32{return f32((word>>16u)&0xffu)*(1.0/255.0);}
// The band argument is unused here; the distance stays in metres, clamped to the
// f16 finite range so a never-written voxel's 1e20 sentinel cannot become an
// infinity that turns a central difference into a NaN.
fn packSceneGeometry(signedDistance:f32,fraction:f32,cellRadius:f32)->u32{
  let d=pack2x16float(vec2f(clamp(signedDistance,-1024.0,1024.0),0.0))&0xffffu;
  return d|(packSceneFraction(fraction)<<16u);
}`;
}

/**
 * The one definition of the scene identity word, shared by the voxeliser that
 * writes it and every path that shades from it.
 *
 * The low half is unchanged and always was: a material id, and `!= 0` is what
 * makes a voxel solid. The high half used to be an owner id — a back-pointer to
 * the authored primitive record — and now carries that primitive's **surface
 * normal, baked at voxelisation time as oct8**.
 *
 * The trade is the whole point of the representation. The producer knows exactly
 * what it is writing and can evaluate the analytic normal for free at init;
 * the renderer, asking the same question per pixel, had to load a 64-byte record
 * through the owner id and re-evaluate a field program or a smooth-union cluster
 * to answer it. Baking makes the render path depend on the voxel and on nothing
 * else — no primitive records, no owner ids, no heightfield — and it costs
 * nothing in memory, because the owner id it replaces was the same two bytes.
 *
 * What is lost with the owner is real and was accepted: hover, picking, per-owner
 * visibility suppression of voxels, and the hard-feature id (a voxel surface now
 * reports `smooth`, which moves only the contact-visibility ray bias, 0.025 vs
 * 0.05 cells). Analytic hits — rigid bodies, glass — keep their own owners.
 *
 * Emitted as WGSL rather than duplicated as hand-matched shifts for the reason
 * {@link sparseBrickSceneGeometryCodecWGSL} is: a producer and a consumer that
 * disagree about a shift render a wrong scene instead of failing to compile.
 *
 * The *word*, not its address: {@link sparseBrickSceneIdentityCodecWGSL} decides
 * which word a voxel's identity lives in under each leaf payload mode, and this
 * decides what is in it. The two are separable because no reader needs both —
 * the voxeliser addresses the flat lane directly and every marcher hoists a
 * `SceneIdentitySource` before it unpacks anything.
 *
 * Requires `svoGBufferPackNormalOct8`/`svoGBufferUnpackNormalOct8` in scope —
 * `SVO_GBUFFER_NORMAL_OCT8_WGSL`, which is where the encoding itself lives.
 */
export function sparseBrickSceneIdentityWordCodecWGSL(): string {
  return /* wgsl */ `
// scene identity word: material id in the low half, baked oct8 normal in the high.
const SCENE_IDENTITY_NO_NORMAL:u32=${SVO_GBUFFER_NORMAL_OCT8_ABSENT}u;
fn sceneIdentityMaterial(word:u32)->u32{return word&0xffffu;}
// Air is \`packMaterialOwner(0, SPARSE_BRICK_NO_OWNER)\` and an unvoxelised brick
// is all zeroes; both have a zero material, which is why this one test covers
// both and why the normal half is free to hold anything.
fn sceneIdentitySolid(word:u32)->bool{return (word&0xffffu)!=0u;}
fn sceneIdentityHasNormal(word:u32)->bool{return (word>>16u)!=SCENE_IDENTITY_NO_NORMAL;}
fn sceneIdentityNormal(word:u32)->vec3f{return svoGBufferUnpackNormalOct8(word>>16u);}
// A degenerate normal is stored as the absent sentinel rather than as a
// direction, so a consumer can fall back to the voxel face instead of shading
// from an arbitrary axis. The producer is the only place that knows the
// difference between "the field has no gradient here" and "the gradient is +Z".
fn packSceneIdentity(materialId:u32,normal:vec3f)->u32{
  let magnitude=length(normal);
  let packed=select(SCENE_IDENTITY_NO_NORMAL,svoGBufferPackNormalOct8(normal),magnitude>1e-8);
  return (materialId&0xffffu)|(packed<<16u);
}`;
}

/** How a shader addresses the banded lanes, and how it loads a payload word. */
export interface SparseBrickBandedCodecOptions {
  /** WGSL expression for the occupancy mask lane's first word. */
  readonly occupancyBase: string;
  /** WGSL expression for the record mask lane's first word. */
  readonly recordMaskBase: string;
  /** WGSL expression for the per-leaf header lane's first word. */
  readonly headerBase: string;
  /** WGSL expression for the blob lane's first word. */
  readonly blobBase: string;
  /** WGSL expression for the record lane's first word. */
  readonly recordsBase: string;
  /** Wrap a payload word index into the load form this shader's binding needs. */
  readonly load: (index: string) => string;
  /** Which rungs to declare. `occupancy` emits the mask alone. */
  readonly mode: Exclude<SparseBrickLeafPayloadMode, "dense">;
  /**
   * Declare the geometry-record accessors. Default `true`, which is the shape
   * the voxeliser needs and the text every existing caller already emits.
   *
   * The pair is separable because it is the *only* part of the codec that reaches
   * outside itself: `bandedRecordWord` calls `sceneGeometryWord`, so a module that
   * wants identity alone would otherwise have to emit the whole scene-geometry
   * codec — and therefore pick a geometry format — to compile a function it never
   * calls. Identity consumers pass `false`.
   */
  readonly records?: boolean;
}

/**
 * The one definition of the banded leaf payload's addressing, shared by the
 * voxeliser that writes it and every consumer that reads it.
 *
 * Emitted as WGSL rather than duplicated at each of the eleven read sites, for
 * the same reason {@link sparseBrickSceneGeometryCodecWGSL} is: a producer and a
 * consumer that disagree about a shift render a wrong scene instead of failing to
 * compile. `lib/svo-banded-leaf-payload.ts` is the executable CPU spec these
 * functions mirror, and `tests/svo-banded-leaf-storage.test.ts` holds the two
 * against each other over the same bytes.
 *
 * One deliberate departure from the reference encoder, and it now costs nothing.
 * The encoder interns materials into a scene-global `u16` table and stores slots;
 * the device stores the material **inline** in the leaf palette. Since the split
 * both are `u16`, so the indirection buys no width at all and the device simply
 * skips the scene-global GPU hash map — the interning was the only part of the
 * layout that needed one. Leaf palette entries are packed two to a word.
 *
 * The blob a leaf owns is three arrays, in this order:
 *
 *   `[ceil(palette/2) material words][index words][ceil(occupied/2) normal words]`
 *
 * and the header carries the normal array's blob-relative start, so a reader
 * never re-derives the index width's contribution.
 */
export function sparseBrickBandedLeafCodecWGSL(options: SparseBrickBandedCodecOptions): string {
  const { load, mode } = options;
  const occupancy = /* wgsl */ `
// banded leaf payload — occupancy is an explicit bit, not an inference.
//
// The predicate this replaces was \`fraction > 0\`, which the voxeliser made
// identical to \`materialOwner != 0\` by writing an identity only where the
// fraction is positive. An interior voxel that stops carrying an explicit
// fraction stops being occupied under that rule, so occupancy has to become a bit
// before anything can be stored nowhere.
// Functions, not module-scope \`const\`s: the bases come from a uniform and WGSL
// refuses to reference \`var<uniform>\` at module scope.
fn bandedOccupancyBase()->u32{return ${options.occupancyBase};}
const BANDED_VOXELS_PER_LEAF:u32=${BANDED_VOXELS_PER_LEAF}u;
const BANDED_MASK_WORDS_PER_LEAF:u32=${BANDED_VOXELS_PER_LEAF / 32}u;
fn bandedVoxelBit(voxel:u32)->u32{return 1u<<(voxel&31u);}
fn bandedOccupancyWord(voxel:u32)->u32{return bandedOccupancyBase()+(voxel>>5u);}
fn bandedOccupied(voxel:u32)->bool{
  return (${load("bandedOccupancyWord(voxel)")}&bandedVoxelBit(voxel))!=0u;
}`;
  if (mode === "occupancy") return occupancy;
  return /* wgsl */ `${occupancy}
fn bandedRecordMaskBase()->u32{return ${options.recordMaskBase};}
fn bandedHeaderBase()->u32{return ${options.headerBase};}
fn bandedBlobBase()->u32{return ${options.blobBase};}
fn bandedRecordsBase()->u32{return ${options.recordsBase};}
const BANDED_HEADER_WORDS:u32=${SPARSE_BRICK_BANDED_HEADER_WORDS}u;
const BANDED_ALLOCATOR_WORDS:u32=${SPARSE_BRICK_BANDED_ALLOCATOR_WORDS}u;
const BANDED_NO_MATERIAL_OWNER:u32=0xffff0000u;
// \`voxelOffset\` is exactly \`leafIndex * 512\` — verified 693/693 aligned, sole GPU
// writer lib/webgpu-sparse-brick-topology-mutation.ts:200 — so a global voxel
// index carries its own leaf slot and no signature has to change to pass one.
fn bandedLeafSlot(voxel:u32)->u32{return voxel/BANDED_VOXELS_PER_LEAF;}
fn bandedLeafLocal(voxel:u32)->u32{return voxel%BANDED_VOXELS_PER_LEAF;}
fn bandedRecordMaskWord(voxel:u32)->u32{return bandedRecordMaskBase()+(voxel>>5u);}
fn bandedRecorded(voxel:u32)->bool{
  return (${load("bandedRecordMaskWord(voxel)")}&bandedVoxelBit(voxel))!=0u;
}
fn bandedHeader(slot:u32,word:u32)->u32{return ${load("bandedHeaderBase()+slot*BANDED_HEADER_WORDS+word")};}
fn bandedHeaderRecordBase(slot:u32)->u32{return bandedHeader(slot,0u);}
fn bandedHeaderBlobBase(slot:u32)->u32{return bandedHeader(slot,1u);}
fn bandedHeaderRecordCount(slot:u32)->u32{return bandedHeader(slot,2u)&0xffffu;}
fn bandedHeaderPaletteCount(slot:u32)->u32{return (bandedHeader(slot,2u)>>16u)&0xffu;}
fn bandedHeaderIndexBits(slot:u32)->u32{return (bandedHeader(slot,2u)>>24u)&0xffu;}
// The blob's three arrays, in order:
//
//   [BANDED_NORMAL_WORDS normal words][ceil(palette/2) material words][index words]
//
// The normal lane is first and fixed-size *so that both of the other two have a
// derivable start and the header needs no offset word for either*. It spans all
// 512 voxels rather than the occupied ones for the same reason it is first: a
// rank-compacted lane saves 22 % of its own bytes and makes every read pay a
// sixteen-word prefix popcount, which measured +4.1 % on the frame.
const BANDED_NORMAL_WORDS:u32=BANDED_VOXELS_PER_LEAF/2u;
/** One \`u16\` of a two-per-word array, by element index. */
fn bandedHalfWord(base:u32,element:u32)->u32{
  return (${load("base+(element>>1u)")}>>((element&1u)*16u))&0xffffu;
}
// Set bits before \`local\` inside this leaf's own mask, by prefix popcount.
//
// Sixteen words means at most fifteen whole-word popcounts and one masked one.
// Deliberately not a stored prefix table: sixteen \`countOneBits\` are cheaper than
// the sixteen extra bytes a per-leaf table would add to *every* leaf, including
// the 47-73 % that carry no records at all — on the hero the median leaf has none.
fn bandedPrefixCount(maskBase:u32,slot:u32,local:u32)->u32{
  let base=maskBase+slot*BANDED_MASK_WORDS_PER_LEAF;
  let word=local>>5u;
  var count=0u;
  for(var i=0u;i<word;i+=1u){count+=countOneBits(${load("base+i")});}
  // \`local & 31\` never reaches 32, so the shift is always defined and the mask is
  // empty at a word boundary rather than total.
  return count+countOneBits(${load("base+word")}&((1u<<(local&31u))-1u));
}
${options.records === false ? "" : /* wgsl */ `/** Where this voxel's geometry record lives, valid only when \`bandedRecorded\`. */
fn bandedRecordSlot(voxel:u32)->u32{
  let slot=bandedLeafSlot(voxel);
  return bandedHeaderRecordBase(slot)+bandedPrefixCount(bandedRecordMaskBase(),slot,bandedLeafLocal(voxel));
}
/** The record's own packed geometry word, through the shared scene-geometry codec. */
fn bandedRecordWord(voxel:u32)->u32{
  return ${load("sceneGeometryWord(bandedRecordsBase(),bandedRecordSlot(voxel))")};
}`}
/**
 * The whole packed \`(normal << 16) | materialId\` word, reassembled.
 *
 * Zero index bits is the common case and the reason the ladder starts there: a
 * leaf with one material resolves every occupied voxel to its single palette
 * entry with no index read at all. 94.8 % of hero leaves are that leaf. The
 * normal is always a read, because it is per voxel — the same occupancy rank the
 * index would have used addresses it.
 */
fn bandedIdentity(voxel:u32)->u32{
  if(!bandedOccupied(voxel)){return BANDED_NO_MATERIAL_OWNER;}
  let slot=bandedLeafSlot(voxel);
  let local=bandedLeafLocal(voxel);
  let blob=bandedBlobBase()+BANDED_ALLOCATOR_WORDS+bandedHeaderBlobBase(slot);
  let bits=bandedHeaderIndexBits(slot);
  let normal=bandedHalfWord(blob,local);
  if(bits==0u){return bandedHalfWord(blob+BANDED_NORMAL_WORDS,0u)|(normal<<16u);}
  let palette=bandedHeaderPaletteCount(slot);
  let bitIndex=bandedPrefixCount(bandedOccupancyBase(),slot,local)*bits;
  let indexWord=${load("blob+BANDED_NORMAL_WORDS+((palette+1u)>>1u)+(bitIndex>>5u)")};
  // An entry never straddles a word: the widths are 1/2/4/8 bits, so
  // \`bitIndex % 32 + bits <= 32\` for every rank.
  let entry=(indexWord>>(bitIndex&31u))&((1u<<bits)-1u);
  // Out of range means the header and the index disagree, which is a bug in the
  // encoder rather than a state to tolerate. Returning the air sentinel makes it
  // a visible hole; clamping into the palette would make it a plausible identity
  // and hide it.
  if(entry>=palette){return BANDED_NO_MATERIAL_OWNER;}
  return bandedHalfWord(blob+BANDED_NORMAL_WORDS,entry)|(normal<<16u);
}`;
}

/** How a shader reaches one voxel's scene identity, whatever the leaf payload mode. */
export interface SparseBrickSceneIdentityCodecOptions {
  readonly mode: SparseBrickLeafPayloadMode;
  /**
   * WGSL expression for the flat owner lane's first word. Read only by the
   * `dense` and `occupancy` arms; a banded world has no such lane.
   */
  readonly materialOwnerBase: string;
  /** Wrap a payload word index into the load form this shader's binding needs. */
  readonly load: (index: string) => string;
}

/**
 * Scene identity, in the shape a per-cell DDA can actually afford.
 *
 * `bandedIdentity` is the correct answer and the wrong *shape* for a marcher: per
 * cell it costs an occupancy word, three header words, a sixteen-word prefix
 * popcount and two blob loads, where the flat lane cost one load. Substituting it
 * into `traceLeafPayload` unchanged is slower than the lane it replaces, and by a
 * lot.
 *
 * None of those loads depend on the *cell* — only on the leaf — so the obvious
 * repair is to resolve the header once into {@link SceneIdentitySource} before the
 * walk. **The repair is measured and it is not where the cost is.** Once the
 * per-cell test is `sceneIdentitySolidAt`, which reads the occupancy mask and
 * needs no source at all, a hoisted source is spent on every leaf the ray crosses
 * — including the ones it misses — to save nothing on the at most one cell per ray
 * that hits; and removing it moved `banded` from **+4.27 % to +4.10 %** against
 * `dense` on `hero-garden-hose` at depth 1, which is inside that lane's noise.
 *
 * The marcher therefore takes one bit per cell from
 * {@link SCENE_IDENTITY_SOLID_AT_WGSL} and calls `sceneIdentityAt` once, on the
 * hit — kept because it is the honest structure rather than because it was worth
 * 0.17 pp. {@link SceneIdentitySource} survives for the readers that resolve many
 * voxels of one leaf in a row — the derived builder and the readback oracles —
 * where the amortisation it was designed for is real.
 *
 * **What the cost actually is:** the indirection itself. A `banded` hit resolves
 * from three arena regions — the leaf header, the blob's material palette, the
 * blob's normal lane — where `dense` read one flat word, and no rearrangement of
 * *when* those loads happen removes them. The `occupancy` rung isolates it: mask
 * predicate, flat identity lane, **+0.64 %**. So the predicate move is free and
 * roughly 3 % is the price of identity not being a lane.
 *
 * There is no `full`-brick rung here and the shipped header has no such kind: a
 * fully solid leaf still carries its 64-byte occupancy mask and still answers the
 * first cell the DDA enters. That is not a loss, because the DDA returns at the
 * first solid cell anyway, and the `full` kind would only remove a bit test that
 * has already been paid for by the load it shares with 31 neighbours.
 *
 * The `dense` arm reuses the same two functions so no reader carries two shapes:
 * the source degenerates to the lane base and the per-cell cost is the one load
 * it always was, plus an add.
 */
export function sparseBrickSceneIdentityCodecWGSL(
  options: SparseBrickSceneIdentityCodecOptions,
): string {
  const { load } = options;
  const source = /* wgsl */ `
// Everything about a leaf's identity storage that does not vary per cell.
//
// \`base\` is the flat lane's first word under \`dense\` and the leaf's blob under
// \`banded\`; the two arms never share a reader, so one field carries both.
// \`normals\` is the leaf's normal lane, already offset off the blob.
struct SceneIdentitySource{base:u32,bits:u32,palette:u32,single:u32,normals:u32}`;
  if (options.mode !== "banded") {
    return /* wgsl */ `${source}
fn sceneIdentitySourceAt(voxel:u32)->SceneIdentitySource{
  return SceneIdentitySource(${options.materialOwnerBase},0u,0u,0u,0u);
}
fn sceneIdentityOf(source:SceneIdentitySource,voxel:u32)->u32{
  return ${load("source.base+voxel")};
}
fn sceneIdentityAt(voxel:u32)->u32{return ${load(`${options.materialOwnerBase}+voxel`)};}
${options.mode === "dense" ? "" : SCENE_IDENTITY_SOLID_AT_WGSL}`;
  }
  return /* wgsl */ `${source}
// The leaf slot is \`voxel >> 9\` for any voxel of the leaf, so a caller with only
// \`leaf.voxelOffset\` and a caller with a global cell index pass the same argument.
fn sceneIdentitySourceAt(voxel:u32)->SceneIdentitySource{
  let slot=bandedLeafSlot(voxel);
  let counts=bandedHeader(slot,2u);
  let blob=bandedBlobBase()+BANDED_ALLOCATOR_WORDS+bandedHeaderBlobBase(slot);
  // Palette entry zero, loaded unconditionally: it is the answer for every
  // occupied voxel of a single-material leaf, and a branch around one load that
  // is already in flight buys nothing on a leaf that has any other kind.
  return SceneIdentitySource(blob+BANDED_NORMAL_WORDS,(counts>>24u)&0xffu,(counts>>16u)&0xffu,
    bandedHalfWord(blob+BANDED_NORMAL_WORDS,0u),blob);
}
fn sceneIdentityOf(source:SceneIdentitySource,voxel:u32)->u32{
  if(!bandedOccupied(voxel)){return BANDED_NO_MATERIAL_OWNER;}
  // The normal is addressed by the voxel — one load, no rank. Only the material
  // index is rank-compacted, and only on the one leaf in twenty that needs one.
  let normal=bandedHalfWord(source.normals,bandedLeafLocal(voxel));
  if(source.bits==0u){return source.single|(normal<<16u);}
  let bitIndex=bandedPrefixCount(bandedOccupancyBase(),bandedLeafSlot(voxel),bandedLeafLocal(voxel))*source.bits;
  let entry=(${load("source.base+((source.palette+1u)>>1u)+(bitIndex>>5u)")}>>(bitIndex&31u))&((1u<<source.bits)-1u);
  // Same contract as \`bandedIdentity\`: an out-of-range entry is an encoder bug,
  // and the air sentinel makes it a visible hole rather than a plausible material.
  if(entry>=source.palette){return BANDED_NO_MATERIAL_OWNER;}
  return bandedHalfWord(source.base,entry)|(normal<<16u);
}
fn sceneIdentityAt(voxel:u32)->u32{return sceneIdentityOf(sceneIdentitySourceAt(voxel),voxel);}
${SCENE_IDENTITY_SOLID_AT_WGSL}`;
}

/**
 * The solidity predicate, in the shape a DDA can afford.
 *
 * **This is what the occupancy mask is for, and it takes no source.** Solidity
 * used to mean loading a 4-byte identity word and testing its low half, so a
 * marcher paid a strided 4-byte load for every cell it *rejected*. Here it is one
 * bit of a word shared with 31 neighbours, and the identity — header, palette
 * entry, prefix popcount, normal — is resolved only for the one cell that answers
 * yes.
 *
 * That it takes no {@link SceneIdentitySource} is the load-bearing part, and it
 * retires the advice on {@link sparseBrickSceneIdentityCodecWGSL} above. That
 * advice — hoist the header out of the walk, keep one bit test per cell — was
 * written when the per-cell *test* still went through the identity, and once the
 * test is the mask a hoisted source is spent on every leaf the ray crosses,
 * including the ones it misses, to save nothing on the one cell that hits. Worth
 * 0.17 pp when measured, which is inside the lane's noise; kept because it is the
 * honest structure, not because it paid.
 *
 * The predicate is the same one over the same data: the voxeliser writes a
 * material exactly where `solidFraction > 0`, and the encoder sets the occupancy
 * bit from that same stored fraction. That equality is what makes the `occupancy`
 * rung hash-identical to `dense` and therefore a bisector.
 *
 * Deliberately not declared on the `dense` arm. There is no mask to ask, so the
 * only honest implementation is `sceneIdentitySolid` on a loaded word — which is
 * what every dense caller already writes, and which would drag this module's
 * output into a dependency on the *word* codec that three of its five consumers
 * do not emit.
 */
const SCENE_IDENTITY_SOLID_AT_WGSL = /* wgsl */ `
fn sceneIdentitySolidAt(voxel:u32)->bool{return bandedOccupied(voxel);}`;

/**
 * One leaf-payload arena, viewed off the device.
 *
 * Word offsets rather than byte offsets, and the whole payload as one `Uint32Array`,
 * because that is exactly how every shader sees it — so a CPU decode that agrees
 * with the WGSL is agreeing about the same arithmetic rather than about a
 * translation of it.
 */
export interface SparseBrickBandedArenaView {
  readonly words: Uint32Array;
  readonly occupancyBase: number;
  readonly recordMaskBase: number;
  readonly headerBase: number;
  readonly blobBase: number;
  readonly recordsBase: number;
}

/** The arena view a resolved layout implies, so a reader never re-derives an offset. */
export function sparseBrickBandedArenaView(
  words: Uint32Array, layout: SparseBrickResolvedPayloadLayout,
): SparseBrickBandedArenaView {
  // `occupancy` carries the mask alone, so its view resolves the mask base and
  // leaves the rest at the absent lanes' own offset. A reader that asks a mask-only
  // arena for a header is a bug in the reader, and the `present` flags say so.
  if (layout.leafPayloadMode === "dense") {
    throw new RangeError("A dense payload layout has no banded arena");
  }
  return {
    words,
    occupancyBase: layout.lanes.sceneOccupancy.offsetBytes / 4,
    recordMaskBase: layout.lanes.sceneRecordMask.offsetBytes / 4,
    headerBase: layout.lanes.sceneBandedHeader.offsetBytes / 4,
    blobBase: layout.lanes.sceneBandedBlob.offsetBytes / 4,
    recordsBase: layout.lanes.sceneBandedRecords.offsetBytes / 4,
  };
}

/** The allocator's own counters: record cursor, blob cursor, flags, leaves encoded. */
export function sparseBrickBandedAllocatorState(view: SparseBrickBandedArenaView): {
  records: number; blobWords: number; flags: number; leaves: number;
} {
  return {
    records: view.words[view.blobBase],
    blobWords: view.words[view.blobBase + 1],
    flags: view.words[view.blobBase + 2],
    leaves: view.words[view.blobBase + 3],
  };
}

/** Overflow bits the encoder raises. Any of them makes an arena figure unreportable. */
export const SPARSE_BRICK_BANDED_OVERFLOW = Object.freeze({
  records: 1, blob: 2, palette: 4,
});

function bandedMaskBit(view: SparseBrickBandedArenaView, base: number, voxel: number): boolean {
  return ((view.words[base + (voxel >>> 5)] >>> (voxel & 31)) & 1) === 1;
}

/** Set bits of one leaf's mask before `local`, the WGSL prefix popcount exactly. */
function bandedPrefix(view: SparseBrickBandedArenaView, base: number, slot: number, local: number): number {
  let count = 0;
  const first = base + slot * (BANDED_VOXELS_PER_LEAF / 32);
  const word = local >>> 5;
  for (let i = 0; i < word; i += 1) count += popcount32(view.words[first + i]);
  return count + popcount32(view.words[first + word] & ((1 << (local & 31)) - 1));
}

function popcount32(value: number): number {
  let v = value - ((value >>> 1) & 0x55555555);
  v = (v & 0x33333333) + ((v >>> 2) & 0x33333333);
  v = (v + (v >>> 4)) & 0x0f0f0f0f;
  return (v * 0x01010101) >>> 24;
}

/** The occupancy predicate, off the device. */
export function sparseBrickBandedOccupiedAt(view: SparseBrickBandedArenaView, voxel: number): boolean {
  return bandedMaskBit(view, view.occupancyBase, voxel);
}

/** Whether this voxel carries an explicit geometry record. */
export function sparseBrickBandedRecordedAt(view: SparseBrickBandedArenaView, voxel: number): boolean {
  return bandedMaskBit(view, view.recordMaskBase, voxel);
}

/** One leaf's header, unpacked. */
export function sparseBrickBandedHeaderAt(view: SparseBrickBandedArenaView, slot: number): {
  recordBase: number; blobBase: number; recordCount: number;
  paletteCount: number; indexBits: number; scale: number;
} {
  const base = view.headerBase + slot * SPARSE_BRICK_BANDED_HEADER_WORDS;
  const counts = view.words[base + 2];
  return {
    recordBase: view.words[base], blobBase: view.words[base + 1],
    recordCount: counts & 0xffff, paletteCount: (counts >>> 16) & 0xff,
    indexBits: (counts >>> 24) & 0xff, scale: view.words[base + 3],
  };
}

/** Words the blob's fixed-size normal lane occupies: one `u16` for each voxel. */
const BANDED_NORMAL_WORDS_PER_LEAF = BANDED_VOXELS_PER_LEAF / 2;

/** Where a recorded voxel's geometry word lives, in record slots. */
export function sparseBrickBandedRecordSlotAt(view: SparseBrickBandedArenaView, voxel: number): number {
  const slot = Math.floor(voxel / BANDED_VOXELS_PER_LEAF);
  return sparseBrickBandedHeaderAt(view, slot).recordBase
    + bandedPrefix(view, view.recordMaskBase, slot, voxel % BANDED_VOXELS_PER_LEAF);
}

/** One `u16` of a two-per-word array, the WGSL `bandedHalfWord` exactly. */
function bandedHalfWord(view: SparseBrickBandedArenaView, base: number, element: number): number {
  return (view.words[base + (element >>> 1)] >>> ((element & 1) * 16)) & 0xffff;
}

/**
 * The whole packed `(normal << 16) | materialId` word, reassembled.
 *
 * The same loads the WGSL does, in the same order, so a disagreement between this
 * and the device is a disagreement about the *bytes* rather than about two
 * independent implementations of a spec.
 */
export function sparseBrickBandedIdentityAt(view: SparseBrickBandedArenaView, voxel: number): number {
  if (!sparseBrickBandedOccupiedAt(view, voxel)) return 0xffff0000;
  const slot = Math.floor(voxel / BANDED_VOXELS_PER_LEAF);
  const local = voxel % BANDED_VOXELS_PER_LEAF;
  const header = sparseBrickBandedHeaderAt(view, slot);
  const blob = view.blobBase + SPARSE_BRICK_BANDED_ALLOCATOR_WORDS + header.blobBase;
  const palette = blob + BANDED_NORMAL_WORDS_PER_LEAF;
  const normal = bandedHalfWord(view, blob, local);
  if (header.indexBits === 0) return ((normal << 16) | bandedHalfWord(view, palette, 0)) >>> 0;
  const bit = bandedPrefix(view, view.occupancyBase, slot, local) * header.indexBits;
  const entry = (view.words[palette + ((header.paletteCount + 1) >>> 1) + (bit >>> 5)] >>> (bit & 31))
    & ((1 << header.indexBits) - 1);
  if (entry >= header.paletteCount) return 0xffff0000;
  return ((normal << 16) | bandedHalfWord(view, palette, entry)) >>> 0;
}

/**
 * Scene identity off the device, whatever the leaf payload mode.
 *
 * The CPU twin of {@link sparseBrickSceneIdentityCodecWGSL}, for the readback
 * oracles: they hold the whole payload arena and the same word offsets the
 * shaders are handed, so a mode switch moves the decode in one place instead of
 * at every oracle. `words` is the arena from word zero, not a lane slice.
 */
export function sparseBrickScenePayloadIdentityAt(
  words: Uint32Array, lanes: SparseBrickScenePayloadLanes, voxel: number,
): number {
  if (lanes.mode !== "banded") return words[lanes.materialOwnerWords + voxel] >>> 0;
  return sparseBrickBandedIdentityAt({
    words,
    occupancyBase: lanes.occupancyWords, recordMaskBase: lanes.recordMaskWords,
    headerBase: lanes.headerWords, blobBase: lanes.blobWords, recordsBase: lanes.recordWords,
  }, voxel);
}

/**
 * The fraction encoder both narrowed formats share.
 *
 * `max(1u, ...)` is not a rounding preference. The voxeliser publishes a
 * material identity exactly where `primitiveFraction > 0`, and the derived
 * builder's opacity pyramid stores `select(0., 1., solid > 0.)` as its maximum
 * channel, so a fraction that rounds to zero retires a voxel that the material
 * lane still says is occupied — a hard flip of the occupancy bit, not a least
 * significant bit of coverage. Flooring positive coverage at 1/255 keeps
 * `fraction > 0` the same predicate on both sides of the encoding.
 */
const SCENE_FRACTION_WGSL = /* wgsl */ `
fn packSceneFraction(fraction:f32)->u32{
  let clamped=clamp(fraction,0.0,1.0);
  return select(0u,max(1u,u32(round(clamped*255.0))),clamped>0.0);
}`;

/**
 * CPU readers of the same lane.
 *
 * The scene geometry lane has a reader off the device — the dry render smoke's
 * ground oracle reads coverage back to prove the terrain is solid — and that
 * reader used to be the place the 8-byte stride was hard-coded. Decoding it
 * here rather than at the call site keeps the number of definitions of this
 * encoding at two (one WGSL, one TypeScript) instead of one per consumer, and
 * `tests/sparse-brick-scene-geometry-format.test.ts` holds the two against each
 * other on the device rather than by inspection.
 *
 * `words` is the lane's own byte range viewed as `u32`, so index 0 is the first
 * voxel's first word whatever the lane's offset in the arena was.
 */
export function sparseBrickSceneFractionAt(
  words: Uint32Array, format: SparseBrickSceneGeometryFormat, voxel: number,
): number {
  if (format === "f32x2") return new Float32Array(words.buffer, words.byteOffset, words.length)[voxel * 2 + 1];
  if (format === "f16-unorm8") return ((words[voxel] >>> 16) & 0xff) / 255;
  return ((words[voxel >>> 1] >>> ((voxel & 1) * 16 + 8)) & 0xff) / 255;
}

/**
 * The distance channel, in metres under every surviving format. A future narrowed
 * arm storing anything else must declare it — see
 * {@link SparseBrickSceneGeometryFormat} — because a caller reading scaled units as
 * metres fails silently.
 */
export function sparseBrickSceneDistanceAt(
  words: Uint32Array, format: SparseBrickSceneGeometryFormat, voxel: number,
): number {
  if (format === "f32x2") return new Float32Array(words.buffer, words.byteOffset, words.length)[voxel * 2];
  if (format === "f16-unorm8") return halfToFloat(words[voxel] & 0xffff);
  const byte = (words[voxel >>> 1] >>> ((voxel & 1) * 16)) & 0xff;
  return Math.max(-1, (byte > 127 ? byte - 256 : byte) / 127);
}

/** IEEE 754 binary16 decode, matching WGSL `unpack2x16float`'s low component. */
function halfToFloat(bits: number): number {
  const sign = bits & 0x8000 ? -1 : 1;
  const exponent = (bits >>> 10) & 0x1f;
  const mantissa = bits & 0x3ff;
  if (exponent === 0) return sign * mantissa * 2 ** -24;
  if (exponent === 0x1f) return mantissa === 0 ? sign * Infinity : NaN;
  return sign * (1 + mantissa / 1024) * 2 ** (exponent - 15);
}

/**
 * Absent lanes report this offset and a 256-byte zeroed page rather than a null
 * binding, so a consumer that binds a lane and bounds it with `arrayLength`
 * keeps working unchanged and reads material zero — air — instead of failing
 * validation. A consumer that indexes an absent lane *without* a bound is a bug
 * and must compile the lane out; the enumerated set is in the dry-profile
 * handoff and every one of them is a templated WGSL fragment today.
 */
export const SPARSE_BRICK_ABSENT_LANE_PAGE_BYTES = 256;

export const SPARSE_BRICK_GPU_LAYOUT = Object.freeze({
  nodeStrideBytes: 32,
  leafStrideBytes: 16,
  geometryStrideBytes: 16,
  velocityStrideBytes: 16,
  materialOwnerStrideBytes: 4,
  controlStrideBytes: 128,
  /** Structural publication state is colocated after control in the shared arena. */
  publicationStrideBytes: 32,
  /** Publication is independently bindable at the portable 256-byte alignment. */
  publicationOffsetBytes: 256,
  /** Control/publication prefix rounded up for storage-buffer binding alignment. */
  topologyOffsetBytes: 512,
  /** geometry = fluid SDF, solid SDF estimate, solid fraction, pressure */
  geometryChannels: ["fluidSignedDistance", "solidSignedDistance", "solidFraction", "pressure"] as const,
  /** velocity = world velocity xyz, reconstructed liquid volume fraction */
  velocityChannels: ["velocityX", "velocityY", "velocityZ", "liquidFraction"] as const,
  controlWords: {
    publishedNodes: 0,
    publishedLeaves: 1,
    publishedVoxels: 2,
    generation: 3,
    requestedNodes: 4,
    requestedLeaves: 5,
    requestedVoxels: 6,
    requestedGeneration: 7,
    nodeCapacity: 8,
    leafCapacity: 9,
    voxelCapacity: 10,
    brickSize: 11,
    overflowFlags: 12,
    droppedNodes: 13,
    droppedLeaves: 14,
    droppedVoxels: 15,
    leafWordOffset: 16,
    velocityWordOffset: 17,
    materialOwnerWordOffset: 18,
    /** Fixed-arena allocator high-water marks; no scene rebuild allocation is required. */
    allocatedNodes: 19,
    allocatedLeaves: 23,
    activeLeaves: 28,
    dirtyLeaves: 29,
    queuedLeaves: 30,
    mutationGeneration: 31,
  } as const,
  dispatchIndirectOffsetBytes: 80,
  drawIndirectOffsetBytes: 96,
});

function integerCoordinate(value: number, name: string): number {
  if (!Number.isSafeInteger(value) || value < 0 || value >= 2 ** SPARSE_BRICK_MAX_MORTON_BITS) {
    throw new RangeError(`${name} must be a non-negative integer below 2^${SPARSE_BRICK_MAX_MORTON_BITS}`);
  }
  return value;
}

/**
 * Spread 21 bits so that bit `i` lands at position `3 * i`.
 *
 * The five masked shifts are the standard closed form, and they are here
 * because the bit-at-a-time loop this replaces was the most expensive single
 * symbol in a large scene build: 20.7 % of `hero-garden-hose` at environment
 * refinement depth 3, because it ran 63 iterations of `Math.floor(v / 2 ** bit)`
 * plus a `BigInt` shift and a `BigInt` allocation *per bit* — 126 bigints to
 * produce one. Every caller is a per-brick call over millions of bricks.
 *
 * The halves are interleaved as `number`s and joined once, so the whole
 * encode allocates two bigints rather than a hundred and twenty-six. Both
 * halves stay inside 32 bits, where `|`, `<<` and `&` are exact.
 */
function spreadBits3(value: number): bigint {
  // Source bits 0..9 land in result bits 0..27, 10..19 in 30..57, and bit 20 —
  // the one that does not fit a ten-bit chain — is placed directly at 60. The
  // three ranges are disjoint, so the joins are `|` rather than addition.
  return BigInt(spreadTenBits(value))
    | (BigInt(spreadTenBits(value >>> 10)) << 30n)
    | (BigInt((value >>> 20) & 1) << 60n);
}

/** Insert two zero bits after each of the ten low bits, in 32-bit arithmetic. */
function spreadTenBits(value: number): number {
  let x = value & 0x0000_03ff;
  x = (x | (x << 16)) & 0xff00_00ff;
  x = (x | (x << 8)) & 0x0300_f00f;
  x = (x | (x << 4)) & 0x030c_30c3;
  x = (x | (x << 2)) & 0x0924_9249;
  return x >>> 0;
}

/** Interleave 21 bits from x/y/z into one deterministic 63-bit address. */
export function mortonEncode3D(x: number, y: number, z: number): bigint {
  return spreadBits3(integerCoordinate(x, "x"))
    | (spreadBits3(integerCoordinate(y, "y")) << 1n)
    | (spreadBits3(integerCoordinate(z, "z")) << 2n);
}

export function mortonDecode3D(key: bigint): SparseBrickCoordinate {
  if (key < 0n || key >= 1n << 63n) throw new RangeError("Morton key must be an unsigned 63-bit integer");
  const values = [0, 0, 0];
  for (let bit = 0; bit < SPARSE_BRICK_MAX_MORTON_BITS; bit += 1) {
    for (let axis = 0; axis < 3; axis += 1) {
      if ((key & (1n << BigInt(3 * bit + axis))) !== 0n) values[axis] += 2 ** bit;
    }
  }
  return { x: values[0], y: values[1], z: values[2] };
}

/** Append one xyz child octant (x | y<<1 | z<<2) to a prefix. */
export function mortonChild(parent: bigint, childOctant: number): bigint {
  if (parent < 0n || parent >= 1n << 60n) throw new RangeError("Parent Morton prefix is too deep");
  if (!Number.isInteger(childOctant) || childOctant < 0 || childOctant > 7) throw new RangeError("Child octant must be 0..7");
  return (parent << 3n) | BigInt(childOctant);
}

export function mortonParent(child: bigint): bigint {
  if (child < 0n || child >= 1n << 63n) throw new RangeError("Morton key must be an unsigned 63-bit integer");
  return child >> 3n;
}

export function packMaterialOwner(materialId: number, ownerId: number = SPARSE_BRICK_NO_OWNER): number {
  if (!Number.isInteger(materialId) || materialId < 0 || materialId > 0xffff) throw new RangeError("Material ID must fit uint16");
  if (!Number.isInteger(ownerId) || ownerId < 0 || ownerId > 0xffff) throw new RangeError("Owner ID must fit uint16");
  return ((ownerId << 16) | materialId) >>> 0;
}

export function unpackMaterialOwner(value: number): { materialId: number; ownerId: number } {
  const word = value >>> 0;
  return { materialId: word & 0xffff, ownerId: word >>> 16 };
}

function depthForCoordinates(coordinates: readonly SparseBrickCoordinate[]): number {
  let maximum = 0;
  for (const coordinate of coordinates) maximum = Math.max(maximum, coordinate.x, coordinate.y, coordinate.z);
  return maximum === 0 ? 0 : Math.ceil(Math.log2(maximum + 1));
}

function popcount8(value: number): number {
  let count = 0;
  for (let word = value & 0xff; word !== 0; word >>>= 1) count += word & 1;
  return count;
}

/**
 * Build a canonical level-major, Morton-sorted pointerless topology.
 * Input coordinates address finest-level bricks, not individual voxels.
 */
export function planSparseBrickOctree(
  input: readonly SparseBrickCoordinate[],
  options: SparseBrickPlanOptions,
): SparseBrickPlan {
  if (options.brickSize !== 4 && options.brickSize !== 8) throw new RangeError("Sparse brick size must be 4 or 8");
  const unique = new Map<bigint, SparseBrickCoordinate>();
  for (const value of input) {
    const coordinate = {
      x: integerCoordinate(value.x, "brick x"),
      y: integerCoordinate(value.y, "brick y"),
      z: integerCoordinate(value.z, "brick z"),
    };
    unique.set(mortonEncode3D(coordinate.x, coordinate.y, coordinate.z), coordinate);
  }
  const coordinates = [...unique.values()];
  const requiredDepth = depthForCoordinates(coordinates);
  const maximumDepth = options.maximumDepth ?? requiredDepth;
  if (!Number.isInteger(maximumDepth) || maximumDepth < 0 || maximumDepth > SPARSE_BRICK_MAX_MORTON_BITS) {
    throw new RangeError(`Maximum depth must be 0..${SPARSE_BRICK_MAX_MORTON_BITS}`);
  }
  if (maximumDepth < requiredDepth) throw new RangeError("Maximum depth cannot contain all brick coordinates");

  const levelCoordinates: SparseBrickCoordinate[][] = [];
  const levelOffsets: number[] = [];
  const nodes: SparseBrickNodePlan[] = [];
  const indexByLevelAndMorton = new Map<string, number>();
  for (let level = 0; level <= maximumDepth; level += 1) {
    levelOffsets.push(nodes.length);
    const divisor = 2 ** (maximumDepth - level);
    const levelMap = new Map<bigint, SparseBrickCoordinate>();
    for (const coordinate of coordinates) {
      const ancestor = {
        x: Math.floor(coordinate.x / divisor),
        y: Math.floor(coordinate.y / divisor),
        z: Math.floor(coordinate.z / divisor),
      };
      levelMap.set(mortonEncode3D(ancestor.x, ancestor.y, ancestor.z), ancestor);
    }
    const sorted = [...levelMap.entries()].sort(([a], [b]) => a < b ? -1 : a > b ? 1 : 0);
    levelCoordinates.push(sorted.map(([, coordinate]) => coordinate));
    for (const [morton, coordinate] of sorted) {
      const index = nodes.length;
      indexByLevelAndMorton.set(`${level}:${morton}`, index);
      nodes.push({
        index, level, morton, coordinate,
        childMask: 0, firstChild: SPARSE_BRICK_INVALID_INDEX, childCount: 0,
        leafIndex: SPARSE_BRICK_INVALID_INDEX,
      });
    }
  }
  levelOffsets.push(nodes.length);

  for (let level = 0; level < maximumDepth; level += 1) {
    for (let localIndex = 0; localIndex < levelCoordinates[level].length; localIndex += 1) {
      const node = nodes[levelOffsets[level] + localIndex];
      let firstChild = SPARSE_BRICK_INVALID_INDEX;
      let mask = 0;
      for (let childOctant = 0; childOctant < 8; childOctant += 1) {
        const childKey = mortonChild(node.morton, childOctant);
        const childIndex = indexByLevelAndMorton.get(`${level + 1}:${childKey}`);
        if (childIndex === undefined) continue;
        firstChild = Math.min(firstChild, childIndex);
        mask |= 1 << childOctant;
      }
      node.childMask = mask;
      node.childCount = popcount8(mask);
      node.firstChild = firstChild;
    }
  }

  const voxelCountPerBrick = options.brickSize ** 3;
  const leaves: SparseBrickLeafPlan[] = [];
  if (coordinates.length > 0) {
    for (let nodeIndex = levelOffsets[maximumDepth]; nodeIndex < levelOffsets[maximumDepth + 1]; nodeIndex += 1) {
      const node = nodes[nodeIndex];
      const index = leaves.length;
      node.leafIndex = index;
      leaves.push({
        index, nodeIndex, morton: node.morton, coordinate: node.coordinate,
        voxelOffset: index * voxelCountPerBrick,
        terminalKind: SPARSE_BRICK_VOXEL_TERMINAL.kind,
        terminalIndex: SPARSE_BRICK_VOXEL_TERMINAL.index,
      });
    }
  }
  return {
    brickSize: options.brickSize,
    maximumDepth,
    levelOffsets,
    nodes,
    leaves,
    voxelCount: leaves.length * voxelCountPerBrick,
  };
}

export interface PackedSparseBrickPlan {
  /** Eight u32 words per node: key lo/hi, level, child mask, first child, child count, leaf, flags. */
  nodes: Uint32Array<ArrayBuffer>;
  /** Four u32 words per leaf: node, voxel offset, terminal kind, terminal record index. */
  leaves: Uint32Array<ArrayBuffer>;
  /** Nodes followed immediately by leaves; preferred by the portable eight-storage-binding publication path. */
  topology: Uint32Array<ArrayBuffer>;
  counts: Uint32Array<ArrayBuffer>;
}

function splitMorton(key: bigint): [number, number] {
  return [Number(key & 0xffffffffn) >>> 0, Number((key >> 32n) & 0xffffffffn) >>> 0];
}

export function packSparseBrickPlan(plan: SparseBrickPlan, generation = 0): PackedSparseBrickPlan {
  const nodeWords = new Uint32Array(plan.nodes.length * 8);
  for (const node of plan.nodes) {
    const [low, high] = splitMorton(node.morton);
    const lifecycle = node.leafIndex === SPARSE_BRICK_INVALID_INDEX ? 0 : SVO_BRICK_LIFECYCLE.activeBit;
    nodeWords.set([
      low, high, node.level, node.childMask, node.firstChild, node.childCount, node.leafIndex, lifecycle,
    ], node.index * 8);
  }
  const leafWords = new Uint32Array(plan.leaves.length * 4);
  for (const leaf of plan.leaves) {
    if ((leaf.terminalKind === SPARSE_BRICK_LEAF_TERMINAL.voxels
      && leaf.terminalIndex !== SPARSE_BRICK_INVALID_INDEX)
      || (leaf.terminalKind === SPARSE_BRICK_LEAF_TERMINAL.planarBoundary
        && leaf.terminalIndex === SPARSE_BRICK_INVALID_INDEX)) {
      throw new RangeError(`Sparse brick leaf ${leaf.index} has an invalid terminal record`);
    }
    leafWords.set([
      leaf.nodeIndex, leaf.voxelOffset, leaf.terminalKind, leaf.terminalIndex,
    ], leaf.index * 4);
  }
  const topology = new Uint32Array(nodeWords.length + leafWords.length);
  topology.set(nodeWords); topology.set(leafWords, nodeWords.length);
  return {
    nodes: nodeWords, leaves: leafWords, topology,
    counts: new Uint32Array([plan.nodes.length, plan.leaves.length, plan.voxelCount, generation >>> 0, 0, nodeWords.length]),
  };
}

export interface SparseBrickPublicationSource {
  /** First four u32 words are node, leaf, voxel, and generation counts. */
  counts: GPUBuffer;
  /** Raw u32 topology arena. Count words 4 and 5 give source node/leaf word offsets. */
  topology: GPUBuffer;
  geometry: GPUBuffer;
  velocity: GPUBuffer;
  materialOwners: GPUBuffer;
  /** Allocated source bounds used only to size a conservative GPU dispatch. */
  capacities: { nodes: number; leaves: number; voxels: number };
}

export interface SparseBrickDenseFieldSource {
  levelSet: GPUTextureView;
  velocity: GPUTextureView;
  /** Dense array of `{ fraction: f32, owner: i32 }`, x-major then y then z. */
  solidCells: GPUBuffer;
  dimensions: readonly [number, number, number];
  gridOriginCells?: readonly [number, number, number];
  /** Maximum topology level. Omit for legacy fixed-level brick plans. */
  finestLevel?: number;
  cellSize: readonly [number, number, number];
  fluidMaterialId: number;
  solidMaterialId: number;
  /**
   * Optional GPU fluid-brick worklist. Header words 0..15 are followed by
   * active `(solverBrick, leafIndex)` pairs and retired pairs. When present,
   * dense publication dispatches only resident brick payloads and separately
   * clears payloads for bricks that have just retired.
   */
  activeBrickWorklist?: GPUBuffer;
}

export interface SparseBrickOctreeGPUOptions {
  brickSize: SparseBrickSize;
  nodeCapacity: number;
  leafCapacity: number;
  label?: string;
  /**
   * Which per-voxel lanes to allocate. Defaults to `full` so every existing
   * world is byte-identical; a renderer-only world passes `dry`. This is the
   * single switch that turns water back on.
   */
  payloadProfile?: SparseBrickPayloadProfileName;
  /**
   * How the dry world stores its scene-geometry channels. Defaults to `f32x2`,
   * which is what shipped; only a `dry` profile may narrow it.
   */
  sceneGeometryFormat?: SparseBrickSceneGeometryFormat;
  /**
   * How a leaf's 512 voxels are stored. Defaults to `dense`, which is what
   * shipped; only a `dry` profile may band. See {@link SparseBrickLeafPayloadMode}.
   */
  leafPayloadMode?: SparseBrickLeafPayloadMode;
  /** Record slots the banded arena reserves, as a fraction of voxel capacity. */
  bandedRecordCapacityFraction?: number;
  /** Dense scene lanes a banded world keeps. See {@link SparseBrickPayloadLayoutOptions.retainDenseLanes}. */
  retainDenseLanes?: readonly SparseBrickDenseSceneLane[];
}

/** One resolved lane: where it starts, how wide a voxel is, and its channels. */
export interface SparseBrickResolvedLane extends SparseBrickPayloadLaneLayout {
  readonly offsetBytes: number;
  /** Word-aligned extent of the whole lane. Zero when the lane is absent. */
  readonly bytes: number;
  readonly present: boolean;
}

export interface SparseBrickResolvedPayloadLayout {
  readonly profile: SparseBrickPayloadProfileName;
  readonly sceneGeometryFormat: SparseBrickSceneGeometryFormat;
  readonly leafPayloadMode: SparseBrickLeafPayloadMode;
  readonly lanes: Readonly<Record<SparseBrickPayloadLaneName, SparseBrickResolvedLane>>;
  readonly totalBytes: number;
  readonly bytesPerVoxel: number;
  /** Record slots the banded arena reserves. Zero unless the mode is `banded`. */
  readonly bandedRecordCapacity: number;
  /** Blob words the banded arena reserves, allocator counters included. Zero unless `banded`. */
  readonly bandedBlobWords: number;
  /**
   * Bytes the arena costs with *no* dense scene lane retained. Equal to
   * {@link totalBytes} unless
   * {@link SparseBrickPayloadLayoutOptions.retainDenseLanes} names one.
   */
  readonly productBytes: number;
}

/**
 * Lane order, and therefore lane offsets.
 *
 * Append-only: the five original lanes keep the offsets they had, so a `dense`
 * world's arena is byte-identical to the one that shipped and any banded lane a
 * profile does not carry costs nothing.
 */
const ALL_LANE_NAMES: readonly SparseBrickPayloadLaneName[] = [
  "geometry", "velocity", "materialOwners", "sceneGeometry", "sceneMaterialOwners",
  "sceneOccupancy", "sceneRecordMask", "sceneBandedHeader", "sceneBandedBlob",
  "sceneBandedRecords",
];

/** Everything about a payload arena that is not the profile or the geometry width. */
export interface SparseBrickPayloadLayoutOptions {
  /** How a leaf's 512-voxel payload is stored. Defaults to `dense`, which is what shipped. */
  readonly leafPayloadMode?: SparseBrickLeafPayloadMode;
  /** Voxels a leaf holds. 512 for every world the live scene path builds. */
  readonly voxelsPerLeaf?: number;
  /** Record slots as a fraction of voxel capacity, for the `banded` mode. */
  readonly bandedRecordCapacityFraction?: number;
  /**
   * Dense per-voxel scene lanes to keep allocated alongside the banded ones.
   *
   * Named one lane at a time rather than switched as a pair, because the two
   * lanes are retained for *different* reasons and are retired by different
   * pieces of work:
   *
   *   - `sceneMaterialOwners` is the banded encoder's own **input**. The
   *     voxeliser writes an identity word a voxel and `encodeBandedLeaves`
   *     reads that lane back to intern the leaf's material palette and to fill
   *     its normal lane (lib/webgpu-sparse-scene-proxies.ts, "after the dense
   *     rebuild, because it reads the lane that pass just wrote"). Retiring it is
   *     a *producer* change — both halves of the identity have to reach the blob
   *     without a whole-arena staging lane — not a reader cutover, and until that
   *     lands the lane is load-bearing on every banded world.
   *   - `sceneGeometry` is still read directly by consumers the banded record
   *     arena has not replaced: the dry primary's smooth-normal reconstruction
   *     and the derived builder's `solidDistance`/`sceneCoverage`. That is the
   *     B2 reader cutover.
   *
   * Retention also costs the whole saving while it is on, which is why the
   * layout reports {@link SparseBrickResolvedPayloadLayout.productBytes}
   * separately: the arena a banded world allocates once nothing retains a dense
   * lane is a function of the same measured leaf count, not a forecast.
   */
  readonly retainDenseLanes?: readonly SparseBrickDenseSceneLane[];
}

/** A dense per-voxel scene lane the banded layout is meant to replace. */
export type SparseBrickDenseSceneLane = "sceneGeometry" | "sceneMaterialOwners";

/**
 * The dense lanes a banded world must keep for its own *producer* to work.
 *
 * Both, today: `encodeBandedLeaves` stages a leaf from the dense identity and
 * geometry words the voxeliser has just written and compacts them in place — the
 * identity's low half into the material palette, its high half into the normal
 * lane, the geometry into the records. A banded world that drops either publishes
 * an arena built from the absent-lane page.
 */
export const SPARSE_BRICK_BANDED_PRODUCER_DENSE_LANES: readonly SparseBrickDenseSceneLane[]
  = Object.freeze(["sceneGeometry", "sceneMaterialOwners"]);

/**
 * Resolve one profile into byte offsets for a given voxel capacity.
 *
 * Absent lanes all share the zeroed page reserved at the head of the arena, so
 * their offset is a real, bindable, 256-aligned address that reads as air. That
 * is a fallback for bound-and-guarded consumers, never a licence to index them:
 * the page is 256 bytes and a lane read is `offset + voxel * stride`, so a
 * consumer must be compiled without the lane or it walks straight into the
 * first present lane.
 *
 * Exported without a device so allocation size is a unit-testable pure function.
 */
export function resolveSparseBrickPayloadLayout(
  profile: SparseBrickPayloadProfileName,
  voxelCapacity: number,
  sceneGeometryFormat: SparseBrickSceneGeometryFormat = "f32x2",
  options: SparseBrickPayloadLayoutOptions = {},
): SparseBrickResolvedPayloadLayout {
  const declared = SPARSE_BRICK_PAYLOAD_PROFILES[profile];
  if (!declared) throw new RangeError(`Unknown sparse brick payload profile "${profile}"`);
  if (!SCENE_GEOMETRY_LANES[sceneGeometryFormat]) {
    throw new RangeError(`Unknown sparse brick scene geometry format "${sceneGeometryFormat}"`);
  }
  const leafPayloadMode = options.leafPayloadMode ?? "dense";
  if (!LEAF_PAYLOAD_MODE_LANES[leafPayloadMode]) {
    throw new RangeError(`Unknown sparse brick leaf payload mode "${leafPayloadMode}"`);
  }
  // The banded layout replaces the per-voxel owner word with a per-leaf palette
  // and reconstructs the distance outside the record set. Both are dry-world
  // decisions for the same reason narrowing is: a solver writes the owner lane
  // itself and mins its own metres against the scene distance.
  if (profile !== "dry" && leafPayloadMode !== "dense") {
    throw new RangeError(`The "${profile}" payload profile cannot band its leaf payload`);
  }
  // The banded arm takes one geometry width. A sub-word record would put two
  // records in one word, written by two threads of one workgroup, and so would need
  // atomics no other arm pays for.
  if (leafPayloadMode === "banded" && sceneGeometryFormat !== "f16-unorm8") {
    // `f32x2` is the *geometry* axis's rollback and composes with `dense`; the
    // banded arm's record width is the one its projection is quoted at, and carrying
    // a second would double the arms to verify for a rollback nothing needs.
    throw new RangeError("The banded leaf payload requires the f16-unorm8 record width");
  }
  const voxelsPerLeaf = options.voxelsPerLeaf ?? BANDED_VOXELS_PER_LEAF;
  if (!Number.isSafeInteger(voxelsPerLeaf) || voxelsPerLeaf < 1) {
    throw new RangeError("Voxels per leaf must be a positive safe integer");
  }
  const recordFraction = options.bandedRecordCapacityFraction
    ?? SPARSE_BRICK_BANDED_RECORD_CAPACITY_FRACTION;
  if (!(recordFraction > 0) || recordFraction > 1) {
    throw new RangeError("Banded record capacity fraction must be in (0, 1]");
  }
  // A solver writes `solidSignedDistance` in metres on the shared four-channel
  // geometry lane and the derived builder takes the *minimum* of the two, so a
  // cell-band value from a narrowed scene lane would be compared against metres.
  // Narrowing is a dry-world property for that reason, not merely by convention.
  if (profile !== "dry" && sceneGeometryFormat !== "f32x2") {
    throw new RangeError(`The "${profile}" payload profile cannot narrow its scene geometry lane`);
  }
  if (!Number.isSafeInteger(voxelCapacity) || voxelCapacity < 1) {
    throw new RangeError("Voxel capacity must be a positive safe integer");
  }
  const banded = leafPayloadMode === "banded";
  const retained = new Set<SparseBrickDenseSceneLane>(banded ? options.retainDenseLanes ?? [] : []);
  const leafCapacity = Math.ceil(voxelCapacity / voxelsPerLeaf);
  const recordCapacity = banded ? Math.ceil(voxelCapacity * recordFraction) : 0;
  const blobWords = banded
    ? SPARSE_BRICK_BANDED_ALLOCATOR_WORDS
      + leafCapacity * (SPARSE_BRICK_BANDED_BLOB_BYTES_PER_LEAF / 4)
    : 0;
  const present = new Map<SparseBrickPayloadLaneName, SparseBrickPayloadLaneLayout>();
  for (const lane of [...declared, ...LEAF_PAYLOAD_MODE_LANES[leafPayloadMode]]) {
    // Identity moves into the per-leaf palette under `banded`, so the per-voxel
    // owner word is not narrowed — it is gone. Absent rather than zero-width, so
    // a consumer that still indexes it fails its own `present` check instead of
    // silently reading the lane behind it.
    if (banded && !retained.has("sceneMaterialOwners") && lane.name === "sceneMaterialOwners") continue;
    // The dense geometry lane goes the same way as the owner lane: records carry
    // geometry for the 6.6 % of voxels that need one, in their own lane, at the
    // same width and through the same codec.
    if (banded && !retained.has("sceneGeometry") && lane.name === "sceneGeometry") continue;
    if (lane.name === "sceneBandedRecords") {
      present.set(lane.name, {
        ...lane, ...SCENE_GEOMETRY_LANES[sceneGeometryFormat], name: "sceneBandedRecords",
        elements: "voxels", elementScale: recordFraction,
      });
      continue;
    }
    if (lane.name === "sceneGeometry" && profile === "dry") {
      present.set(lane.name, SCENE_GEOMETRY_LANES[sceneGeometryFormat]);
      continue;
    }
    present.set(lane.name, lane);
  }
  // The full profile keeps `geometry` at offset zero, which several shaders
  // address implicitly as `payload[voxel * 4 + channel]`. Only a profile that
  // omits it can afford to spend offset zero on the absent page.
  const reservesAbsentPage = !present.has("geometry");
  let cursor = reservesAbsentPage ? SPARSE_BRICK_ABSENT_LANE_PAGE_BYTES : 0;
  const lanes = {} as Record<SparseBrickPayloadLaneName, SparseBrickResolvedLane>;
  // Only the *gaps* between lanes need 256-byte alignment; the arena's own end
  // does not, so the tail is tracked separately from the padded cursor.
  let end = cursor;
  for (const name of ALL_LANE_NAMES) {
    const lane = present.get(name);
    if (!lane) {
      lanes[name] = { name, strideBytes: 0, channels: [], offsetBytes: 0, bytes: 0, present: false };
      continue;
    }
    const formats = lane.channelFormats ?? lane.channels.map((): SparseBrickChannelFormat => "f32");
    if (formats.length !== lane.channels.length) {
      throw new RangeError(`Lane ${name} declares ${lane.channels.length} channels and ${formats.length} formats`);
    }
    if (sparseBrickLaneStrideBytes(formats) !== lane.strideBytes) {
      throw new RangeError(
        `Lane ${name} declares ${formats.join("+")} for ${lane.strideBytes} bytes`
        + ` (channels round up to ${sparseBrickLaneStrideBytes(formats)})`);
    }
    const elements = lane.elements === "leaves"
      ? leafCapacity
      : Math.ceil(voxelCapacity * (lane.elementScale ?? 1));
    // A sub-word lane rounds its own extent up to a word: storage bindings are
    // word-addressed and the last voxel of an odd capacity shares one with
    // nothing.
    const declaredBytes = Math.ceil(checkedBytes(elements, lane.strideBytes, name) / 4) * 4;
    // The blob lane carries the allocator's own counters ahead of the first
    // leaf's words, so the bump cursor lives in the arena it hands out rather
    // than in a buffer a consumer would have to bind separately.
    const bytes = name === "sceneBandedBlob" ? blobWords * 4 : declaredBytes;
    lanes[name] = { ...lane, offsetBytes: cursor, bytes, present: true };
    end = cursor + bytes;
    cursor += alignBytes(bytes);
  }
  return {
    profile,
    sceneGeometryFormat,
    leafPayloadMode,
    lanes,
    totalBytes: Math.max(SPARSE_BRICK_ABSENT_LANE_PAGE_BYTES, end),
    // Bytes a *voxel*, so a per-leaf or suballocated lane is charged at the rate
    // it actually costs one — 16 bytes a leaf is 1/32 of a byte a voxel, and a
    // record arena at a quarter of capacity is a quarter of its own stride.
    bytesPerVoxel: ALL_LANE_NAMES.reduce((sum, name) => {
      const lane = lanes[name];
      if (!lane.present) return sum;
      if (lane.elements === "leaves") return sum + lane.strideBytes / voxelsPerLeaf;
      return sum + lane.strideBytes * (lane.elementScale ?? 1);
    }, 0),
    bandedRecordCapacity: recordCapacity,
    bandedBlobWords: blobWords,
    // What the same measured scene costs once the dense lanes are gone.
    //
    // Resolved by *this function* with the provenance flag cleared, rather than by
    // subtracting lane sizes here: the second form would have to re-derive the
    // 256-byte gap padding and would drift from the allocator the moment a lane
    // moved. One code path, two answers.
    productBytes: retained.size > 0
      ? resolveSparseBrickPayloadLayout(profile, voxelCapacity, sceneGeometryFormat,
        { ...options, retainDenseLanes: [] }).totalBytes
      : Math.max(SPARSE_BRICK_ABSENT_LANE_PAGE_BYTES, end),
  };
}

const publicationShader = /* wgsl */ `
struct StructuralArena {
  control: array<atomic<u32>, 128>,
  topology: array<u32>,
}
@group(0) @binding(0) var<storage, read> sourceCounts: array<u32>;
@group(0) @binding(1) var<storage, read> sourceTopology: array<u32>;
@group(0) @binding(2) var<storage, read> sourceGeometry: array<vec4f>;
@group(0) @binding(3) var<storage, read> sourceVelocity: array<vec4f>;
@group(0) @binding(4) var<storage, read> sourceMaterialOwners: array<u32>;
@group(0) @binding(5) var<storage, read_write> structure: StructuralArena;
@group(0) @binding(7) var<storage, read_write> payload: array<u32>;

fn linearIndex(gid: vec3u, groups: vec3u) -> u32 {
  return gid.x + gid.y * groups.x * 256u + gid.z * groups.x * groups.y * 256u;
}
@compute @workgroup_size(256)
fn publishStructure(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) groups: vec3u) {
  let index = linearIndex(gid, groups);
  let requested = vec3u(sourceCounts[0], sourceCounts[1], sourceCounts[2]);
  let capacities = vec3u(
    atomicLoad(&structure.control[8]),
    atomicLoad(&structure.control[9]),
    atomicLoad(&structure.control[10]),
  );
  let overflow = requested > capacities;
  let valid = !any(overflow);
  if (index == 0u) {
    atomicStore(&structure.control[0], select(0u, requested.x, valid));
    atomicStore(&structure.control[1], select(0u, requested.y, valid));
    atomicStore(&structure.control[2], select(0u, requested.z, valid));
    atomicStore(&structure.control[3], select(0u, sourceCounts[3], valid));
    atomicStore(&structure.control[4], requested.x); atomicStore(&structure.control[5], requested.y);
    atomicStore(&structure.control[6], requested.z); atomicStore(&structure.control[7], sourceCounts[3]);
    let flags = select(0u, 1u, overflow.x) | select(0u, 2u, overflow.y) | select(0u, 4u, overflow.z);
    atomicStore(&structure.control[12], flags);
    atomicStore(&structure.control[13], select(0u, requested.x - capacities.x, overflow.x));
    atomicStore(&structure.control[14], select(0u, requested.y - capacities.y, overflow.y));
    atomicStore(&structure.control[15], select(0u, requested.z - capacities.z, overflow.z));
    // Initialize the fixed arenas as high-water allocators. Later mutations
    // advance these counters in place rather than replacing scene buffers.
    atomicStore(&structure.control[19], select(0u, requested.x, valid));
    atomicStore(&structure.control[23], select(0u, requested.y, valid));
    atomicStore(&structure.control[28], select(0u, requested.y, valid));
    atomicStore(&structure.control[29], 0u); atomicStore(&structure.control[30], 0u);
    atomicStore(&structure.control[31], select(0u, sourceCounts[3], valid));
    let blocks = select(0u, (requested.z + 255u) / 256u, valid && requested.z > 0u);
    let dispatchX = min(blocks, 65535u);
    atomicStore(&structure.control[20], dispatchX);
    if (dispatchX > 0u) { atomicStore(&structure.control[21], (blocks + dispatchX - 1u) / dispatchX); }
    else { atomicStore(&structure.control[21], 0u); }
    atomicStore(&structure.control[22], 1u);
    atomicStore(&structure.control[24], select(0u, 36u, valid && requested.y > 0u));
    atomicStore(&structure.control[25], select(0u, requested.y, valid));
    atomicStore(&structure.control[26], 0u); atomicStore(&structure.control[27], 0u);
  }
  if (!valid) { return; }
  if (index < requested.x) {
    let sourceBase = sourceCounts[4] + index * 8u;
    let destinationBase = index * 8u;
    for (var word = 0u; word < 8u; word += 1u) {
      structure.topology[destinationBase + word] = sourceTopology[sourceBase + word];
    }
  }
  if (index < requested.y) {
    let sourceBase = sourceCounts[5] + index * 4u;
    let destinationBase = atomicLoad(&structure.control[16]) + index * 4u;
    for (var word = 0u; word < 4u; word += 1u) {
      structure.topology[destinationBase + word] = sourceTopology[sourceBase + word];
    }
  }
}

@compute @workgroup_size(256)
fn publishGeometry(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) groups: vec3u) {
  let index = linearIndex(gid, groups);
  if (index >= atomicLoad(&structure.control[2])) { return; }
  let destinationBase = index * 4u;
  payload[destinationBase] = bitcast<u32>(sourceGeometry[index].x);
  payload[destinationBase + 1u] = bitcast<u32>(sourceGeometry[index].y);
  payload[destinationBase + 2u] = bitcast<u32>(sourceGeometry[index].z);
  payload[destinationBase + 3u] = bitcast<u32>(sourceGeometry[index].w);
}

@compute @workgroup_size(256)
fn publishVelocity(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) groups: vec3u) {
  let index = linearIndex(gid, groups);
  if (index >= atomicLoad(&structure.control[2])) { return; }
  let destinationBase = atomicLoad(&structure.control[17]) + index * 4u;
  payload[destinationBase] = bitcast<u32>(sourceVelocity[index].x);
  payload[destinationBase + 1u] = bitcast<u32>(sourceVelocity[index].y);
  payload[destinationBase + 2u] = bitcast<u32>(sourceVelocity[index].z);
  payload[destinationBase + 3u] = bitcast<u32>(sourceVelocity[index].w);
}

@compute @workgroup_size(256)
fn publishMaterialOwners(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) groups: vec3u) {
  let index = linearIndex(gid, groups);
  if (index >= atomicLoad(&structure.control[2])) { return; }
  payload[atomicLoad(&structure.control[18]) + index] = sourceMaterialOwners[index];
}
`;

const denseFieldShader = /* wgsl */ `
struct SolidCell { fraction: f32, owner: i32 }
struct Params { dims: vec4u, origin: vec4i, cell: vec4f, materials: vec4u }
@group(0) @binding(0) var<storage, read> control: array<u32>;
@group(0) @binding(1) var<storage, read> topology: array<u32>;
@group(0) @binding(2) var levelSet: texture_3d<f32>;
@group(0) @binding(3) var velocityField: texture_3d<f32>;
@group(0) @binding(4) var<storage, read> solidCells: array<SolidCell>;
@group(0) @binding(5) var<uniform> params: Params;
@group(0) @binding(6) var<storage, read_write> payload: array<u32>;
@group(0) @binding(7) var<storage, read> brickWorklist: array<u32>;

const ACTIVE_WORKLIST_BIT: u32 = 0x80000000u;
const WORKLIST_HEADER_WORDS: u32 = 16u;

fn keyBit(low: u32, high: u32, bit: u32) -> u32 {
  if (bit >= 32u) { return (high >> (bit - 32u)) & 1u; }
  return (low >> bit) & 1u;
}
fn decodeMorton(low: u32, high: u32, level: u32) -> vec3u {
  var result = vec3u(0u);
  for (var bit = 0u; bit < level; bit += 1u) {
    let scale = 1u << bit;
    result.x += keyBit(low, high, 3u * bit) * scale;
    result.y += keyBit(low, high, 3u * bit + 1u) * scale;
    result.z += keyBit(low, high, 3u * bit + 2u) * scale;
  }
  return result;
}
fn linearIndex(gid: vec3u, groups: vec3u) -> u32 {
  return gid.x + gid.y * groups.x * 256u + gid.z * groups.x * groups.y * 256u;
}
fn usesActiveWorklist() -> bool { return (params.dims.w & ACTIVE_WORKLIST_BIT) != 0u; }
fn finestLevel() -> u32 { return params.dims.w & ~ACTIVE_WORKLIST_BIT; }
fn solverBrickCapacity(brickSize: u32) -> u32 {
  let bricks = (params.dims.xyz + vec3u(brickSize - 1u)) / vec3u(brickSize);
  return bricks.x * bricks.y * bricks.z;
}
@compute @workgroup_size(256)
fn materializeDenseFields(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) groups: vec3u) {
  let index = linearIndex(gid, groups);
  let brickSize = control[11];
  let voxelsPerBrick = brickSize * brickSize * brickSize;
  let streamBrick = index / voxelsPerBrick;
  if (usesActiveWorklist() && streamBrick >= brickWorklist[0]) { return; }
  var leafIndex = streamBrick;
  if (usesActiveWorklist()) { leafIndex = brickWorklist[WORKLIST_HEADER_WORDS + streamBrick * 2u + 1u]; }
  if (leafIndex >= control[1]) { return; }
  // In active mode index addresses the compact work stream, not the sparse
  // tree's leaf order. Keep voxel-local addressing tied to the stream slot;
  // only payload lookup uses the mapped leaf index.
  let localIndex = index - streamBrick * voxelsPerBrick;
  let local = vec3u(localIndex % brickSize, (localIndex / brickSize) % brickSize, localIndex / (brickSize * brickSize));
  let leafBase = control[16] + leafIndex * 4u;
  let nodeIndex = topology[leafBase];
  let voxelOffset = topology[leafBase + 1u];
  let nodeBase = nodeIndex * 8u;
  let level = topology[nodeBase + 2u];
  let brick = decodeMorton(topology[nodeBase], topology[nodeBase + 1u], level);
  var scale = 1u;
  let finest = finestLevel();
  if (finest != 0x7fffffffu && finest > level) { scale = 1u << (finest - level); }
  let worldCell = vec3i((brick * brickSize + local) * scale);
  let q = worldCell - params.origin.xyz;
  let output = voxelOffset + localIndex;
  let geometryBase = output * 4u;
  let velocityBase = control[17] + output * 4u;
  if (scale != 1u || any(q < vec3i(0)) || any(q >= vec3i(params.dims.xyz))) {
    // Leaves outside the current dense fluid window are maintained by their
    // own scene mutation streams and are not part of this field update.
    return;
  }
  let dense = u32(q.x) + params.dims.x * (u32(q.y) + params.dims.y * u32(q.z));
  let phi = textureLoad(levelSet, q, 0).x;
  var solid = SolidCell(0.0, -1);
  if (dense < arrayLength(&solidCells)) { solid = solidCells[dense]; }
  let h = min(params.cell.x, min(params.cell.y, params.cell.z));
  let solidPhi = (0.5 - clamp(solid.fraction, 0.0, 1.0)) * 2.0 * h;
  let materialOffset = control[18] + output;
  payload[geometryBase] = bitcast<u32>(phi); payload[geometryBase + 1u] = bitcast<u32>(solidPhi);
  payload[geometryBase + 2u] = bitcast<u32>(solid.fraction); payload[geometryBase + 3u] = 0u;
  let fieldVelocity = textureLoad(velocityField, q, 0).xyz;
  let liquidFraction = clamp(0.5 - phi / max(h, 1e-8), 0.0, 1.0);
  payload[velocityBase] = bitcast<u32>(fieldVelocity.x); payload[velocityBase + 1u] = bitcast<u32>(fieldVelocity.y);
  payload[velocityBase + 2u] = bitcast<u32>(fieldVelocity.z); payload[velocityBase + 3u] = bitcast<u32>(liquidFraction);
  var material = select(0u, params.materials.x, phi < 0.0);
  material = select(material, params.materials.y, solid.fraction > 0.0);
  // Empty solid cells may be zero-initialized before the first raster pass.
  // Never treat their default owner 0 as rigid body 0 unless occupancy is
  // actually present.
  let owner = select(0xffffu, min(u32(max(solid.owner, 0)), 0xfffeu), solid.fraction > 0.0 && solid.owner >= 0);
  payload[materialOffset] = (owner << 16u) | (material & 0xffffu);
}

@compute @workgroup_size(256)
fn clearRetiredDenseFields(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) groups: vec3u) {
  let index = linearIndex(gid, groups);
  let brickSize = control[11];
  let voxelsPerBrick = brickSize * brickSize * brickSize;
  let retiredSlot = index / voxelsPerBrick;
  if (retiredSlot >= brickWorklist[4]) { return; }
  let capacity = solverBrickCapacity(brickSize);
  let retiredBase = WORKLIST_HEADER_WORDS + capacity * 2u;
  let leafIndex = brickWorklist[retiredBase + retiredSlot * 2u + 1u];
  if (leafIndex >= control[1]) { return; }
  let localIndex = index - retiredSlot * voxelsPerBrick;
  let leafBase = control[16] + leafIndex * 4u;
  let output = topology[leafBase + 1u] + localIndex;
  let geometryBase = output * 4u;
  let velocityBase = control[17] + output * 4u;
  let materialOffset = control[18] + output;
  payload[geometryBase] = bitcast<u32>(3.402823e38);
  payload[geometryBase + 1u] = bitcast<u32>(3.402823e38);
  payload[geometryBase + 2u] = 0u;
  payload[geometryBase + 3u] = 0u;
  payload[velocityBase] = 0u; payload[velocityBase + 1u] = 0u;
  payload[velocityBase + 2u] = 0u; payload[velocityBase + 3u] = 0u;
  payload[materialOffset] = 0xffff0000u;
}
`;

export const sparseBrickPublicationShader = publicationShader;
export const sparseBrickDenseFieldShader = denseFieldShader;

function positiveCapacity(value: number, name: string): number {
  if (!Number.isSafeInteger(value) || value < 1) throw new RangeError(`${name} must be a positive safe integer`);
  return value;
}

/** Portable 2D dispatch sizing for streams larger than WebGPU's x dimension. */
export function sparseBrickDispatchDimensions(itemCount: number, workgroupSize = 256): [number, number, number] {
  if (!Number.isSafeInteger(itemCount) || itemCount < 0) throw new RangeError("Item count must be a non-negative safe integer");
  const blocks = Math.ceil(itemCount / workgroupSize);
  if (blocks === 0) return [0, 1, 1];
  const x = Math.min(blocks, 65_535);
  const y = Math.ceil(blocks / x);
  if (y > 65_535) throw new RangeError("Sparse brick dispatch exceeds portable WebGPU dimensions");
  return [x, y, 1];
}

/**
 * Bytes `items` of a lane cost, with the sub-byte case rounded up.
 *
 * A mask lane's stride is a fraction of a byte, so the product is fractional
 * whenever the item count is not a multiple of the packing — `ceil` there is not
 * a tolerance, it is the last partial byte the lane genuinely occupies. The
 * integer path is unchanged: `ceil` of an integer is itself, so every lane that
 * predates sub-byte strides sizes to the same number it always did.
 */
function checkedBytes(items: number, stride: number, name: string): number {
  const bytes = Math.ceil(items * stride);
  if (!Number.isSafeInteger(bytes) || bytes > 0xffffffff) throw new RangeError(`${name} allocation is too large`);
  return Math.max(Math.ceil(stride), bytes);
}

function alignBytes(value: number, alignment = 256): number {
  return Math.ceil(value / alignment) * alignment;
}

/**
 * Where a consumer finds scene identity inside one payload arena.
 *
 * One block rather than a lane binding, because under `banded` the identity of a
 * voxel is not a lane read at all: it is an occupancy bit, a per-leaf header and
 * a palette entry, in four different lanes of the same buffer. A consumer binds
 * the whole arena once and addresses it through these word offsets, which is the
 * same shape `structureOffsetsWords` already uses for the structural arena.
 *
 * `materialOwnerWords` is the flat lane's own base and is meaningful only while
 * `mode` is `dense` or `occupancy`; a banded world has no such lane and reports
 * the absent page's offset, which is why `mode` is carried with the offsets
 * rather than resolved separately by each consumer.
 */
export interface SparseBrickScenePayloadLanes {
  readonly mode: SparseBrickLeafPayloadMode;
  /** Dense scene-geometry base. Retained while the voxelizer stages banded records from it. */
  readonly geometryWords: number;
  /** Dense geometry record stride, in u32 words. */
  readonly geometryStrideWords: number;
  /** Solid-fraction word inside an unpacked dense geometry record. */
  readonly geometryFractionWord: number;
  /** Packed f16-distance/unorm8-fraction record, rather than plain f32 channels. */
  readonly geometryPacked: boolean;
  readonly materialOwnerWords: number;
  readonly occupancyWords: number;
  readonly recordMaskWords: number;
  readonly headerWords: number;
  readonly blobWords: number;
  readonly recordWords: number;
}

export class SparseBrickOctreeGPU {
  readonly brickSize: SparseBrickSize;
  readonly nodeCapacity: number;
  readonly leafCapacity: number;
  readonly voxelCapacity: number;
  /** Total bytes owned by this class, including control/indirect uniforms. */
  readonly allocatedBytes: number;
  /** Control, publication state, nodes, and leaves share one bindable arena. */
  readonly structure: GPUBuffer;
  readonly topology: GPUBuffer;
  readonly topologyOffsetBytes = SPARSE_BRICK_GPU_LAYOUT.topologyOffsetBytes;
  readonly payload: GPUBuffer;
  readonly controlAndIndirect: GPUBuffer;
  readonly nodes: GPUBuffer;
  readonly nodeOffsetBytes = SPARSE_BRICK_GPU_LAYOUT.topologyOffsetBytes;
  readonly leaves: GPUBuffer;
  /** Leaf offset relative to the topology slice used by legacy producer shaders. */
  readonly leafTopologyOffsetBytes: number;
  readonly leafOffsetBytes: number;
  readonly geometry: GPUBuffer;
  readonly geometryOffsetBytes: number;
  readonly velocity: GPUBuffer;
  readonly velocityOffsetBytes: number;
  readonly materialOwners: GPUBuffer;
  readonly materialOwnerOffsetBytes: number;
  /** Authored/live scene lanes share topology but never alias evolving fluid writes. */
  readonly sceneGeometry: GPUBuffer;
  readonly sceneGeometryOffsetBytes: number;
  readonly sceneMaterialOwners: GPUBuffer;
  readonly sceneMaterialOwnerOffsetBytes: number;
  /**
   * The resolved lane table. Consumers that must survive both profiles read
   * `lanes.<name>.present` and `.strideBytes` rather than assuming the full
   * five-lane layout; the `*OffsetBytes` aliases above stay for the many call
   * sites that only ever see a `full` world.
   */
  readonly payloadLayout: SparseBrickResolvedPayloadLayout;
  readonly payloadProfile: SparseBrickPayloadProfileName;
  /** How the dry world stores its two scene geometry channels. */
  readonly sceneGeometryFormat: SparseBrickSceneGeometryFormat;
  /** Voxel stride of the scene geometry lane: 16 bytes on `full`, 8/4/2 on `dry`. */
  readonly sceneGeometryStrideBytes: number;
  /** Word-aligned extent of the scene geometry lane, which a sub-word stride rounds up. */
  readonly sceneGeometryBytes: number;
  /** How a leaf's 512 voxels are stored. See {@link SparseBrickLeafPayloadMode}. */
  readonly leafPayloadMode: SparseBrickLeafPayloadMode;
  /**
   * Word offsets of the banded lanes, in the order the shaders' uniform block
   * carries them: occupancy mask, record mask, per-leaf header, blob arena.
   *
   * Word offsets rather than byte offsets because every consumer binds the
   * payload as `array<u32>` and the masks are addressed by word — and because a
   * 1/8-byte stride has no byte offset per voxel to speak of.
   */
  readonly bandedLaneWordOffsets: readonly [number, number, number, number, number];
  /** Every address a consumer needs to read scene identity out of {@link payload}. */
  readonly scenePayloadLanes: SparseBrickScenePayloadLanes;
  /** Record slots the banded arena reserves. Zero unless the mode is `banded`. */
  readonly bandedRecordCapacity: number;
  /**
   * Payload bytes a banded world costs without the dense lanes it keeps only so
   * the banded bytes can be cross-decoded against them. Equal to the payload
   * buffer's own size otherwise, and the figure the reader cutover realises.
   */
  readonly payloadProductBytes: number;
  readonly control: GPUBuffer;
  readonly controlOffsetBytes = 0;
  readonly structuralPublication: GPUBuffer;
  readonly structuralPublicationOffsetBytes = SPARSE_BRICK_GPU_LAYOUT.publicationOffsetBytes;
  readonly dispatchIndirect: GPUBuffer;
  readonly dispatchIndirectOffsetBytes = SPARSE_BRICK_GPU_LAYOUT.dispatchIndirectOffsetBytes;
  readonly drawIndirect: GPUBuffer;
  readonly drawIndirectOffsetBytes = SPARSE_BRICK_GPU_LAYOUT.drawIndirectOffsetBytes;

  private readonly device: GPUDevice;
  private publicationPipelines: readonly GPUComputePipeline[] = [];
  private denseFieldPipeline!: GPUComputePipeline;
  private denseFieldCleanupPipeline!: GPUComputePipeline;
  private publicationModule!: GPUShaderModule;
  private denseFieldModule!: GPUShaderModule;
  private readonly label: string;
  private readonly denseFieldParams: GPUBuffer;
  private destroyed = false;

  constructor(device: GPUDevice, options: SparseBrickOctreeGPUOptions) {
    if (options.brickSize !== 4 && options.brickSize !== 8) throw new RangeError("Sparse brick size must be 4 or 8");
    this.device = device;
    this.brickSize = options.brickSize;
    this.nodeCapacity = positiveCapacity(options.nodeCapacity, "Node capacity");
    this.leafCapacity = positiveCapacity(options.leafCapacity, "Leaf capacity");
    this.voxelCapacity = this.leafCapacity * this.brickSize ** 3;
    const label = options.label ?? "Sparse brick octree";
    this.label = label;
    const storageUsage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    const indirectUsage = storageUsage | GPUBufferUsage.INDIRECT;
    const nodeBytes = checkedBytes(this.nodeCapacity, SPARSE_BRICK_GPU_LAYOUT.nodeStrideBytes, "Node");
    const leafBytes = checkedBytes(this.leafCapacity, SPARSE_BRICK_GPU_LAYOUT.leafStrideBytes, "Leaf");
    this.leafTopologyOffsetBytes = alignBytes(nodeBytes);
    this.leafOffsetBytes = this.topologyOffsetBytes + this.leafTopologyOffsetBytes;
    // Lane offsets come from the profile table rather than a fixed chain, so a
    // dry world simply never reaches the three lanes nothing writes.
    this.payloadProfile = options.payloadProfile ?? "full";
    // A `full` world may not narrow, and says so by resolving to `f32x2` here
    // rather than by throwing at a caller that only knows the world's lever.
    this.sceneGeometryFormat = this.payloadProfile === "dry" ? options.sceneGeometryFormat ?? "f32x2" : "f32x2";
    this.leafPayloadMode = this.payloadProfile === "dry" ? options.leafPayloadMode ?? "dense" : "dense";
    const layout = resolveSparseBrickPayloadLayout(
      this.payloadProfile, this.voxelCapacity, this.sceneGeometryFormat, {
        leafPayloadMode: this.leafPayloadMode,
        voxelsPerLeaf: this.brickSize ** 3,
        bandedRecordCapacityFraction: options.bandedRecordCapacityFraction,
        retainDenseLanes: options.retainDenseLanes,
      });
    this.payloadLayout = layout;
    this.bandedLaneWordOffsets = [
      layout.lanes.sceneOccupancy.offsetBytes / 4, layout.lanes.sceneRecordMask.offsetBytes / 4,
      layout.lanes.sceneBandedHeader.offsetBytes / 4, layout.lanes.sceneBandedBlob.offsetBytes / 4,
      layout.lanes.sceneBandedRecords.offsetBytes / 4,
    ];
    this.scenePayloadLanes = Object.freeze({
      mode: this.leafPayloadMode,
      geometryWords: layout.lanes.sceneGeometry.offsetBytes / 4,
      geometryStrideWords: layout.lanes.sceneGeometry.strideBytes / 4,
      geometryFractionWord: this.payloadProfile === "full" ? 2
        : this.sceneGeometryFormat === "f32x2" ? 1 : 0,
      geometryPacked: this.payloadProfile === "dry" && this.sceneGeometryFormat === "f16-unorm8",
      materialOwnerWords: layout.lanes.sceneMaterialOwners.offsetBytes / 4,
      occupancyWords: this.bandedLaneWordOffsets[0],
      recordMaskWords: this.bandedLaneWordOffsets[1],
      headerWords: this.bandedLaneWordOffsets[2],
      blobWords: this.bandedLaneWordOffsets[3],
      recordWords: this.bandedLaneWordOffsets[4],
    });
    this.bandedRecordCapacity = layout.bandedRecordCapacity;
    this.payloadProductBytes = layout.productBytes;
    this.geometryOffsetBytes = layout.lanes.geometry.offsetBytes;
    this.velocityOffsetBytes = layout.lanes.velocity.offsetBytes;
    this.materialOwnerOffsetBytes = layout.lanes.materialOwners.offsetBytes;
    this.sceneGeometryOffsetBytes = layout.lanes.sceneGeometry.offsetBytes;
    this.sceneMaterialOwnerOffsetBytes = layout.lanes.sceneMaterialOwners.offsetBytes;
    this.sceneGeometryStrideBytes = layout.lanes.sceneGeometry.strideBytes;
    this.sceneGeometryBytes = layout.lanes.sceneGeometry.bytes;
    const topologyBytes = this.leafTopologyOffsetBytes + leafBytes;
    const payloadBytes = layout.totalBytes;
    const structureBytes = this.topologyOffsetBytes + topologyBytes;
    this.allocatedBytes = structureBytes + payloadBytes + 64;
    this.structure = device.createBuffer({ label: `${label} structural arena`, size: structureBytes, usage: indirectUsage });
    this.topology = this.structure;
    this.payload = device.createBuffer({ label: `${label} payload arena`, size: payloadBytes, usage: storageUsage });
    this.controlAndIndirect = this.structure;
    // Aliases plus explicit offsets keep producer APIs ergonomic while render
    // consumers bind the complete structural arena exactly once.
    this.nodes = this.structure; this.leaves = this.structure;
    this.geometry = this.payload; this.velocity = this.payload; this.materialOwners = this.payload;
    this.sceneGeometry = this.payload; this.sceneMaterialOwners = this.payload;
    this.control = this.structure;
    this.structuralPublication = this.structure;
    this.dispatchIndirect = this.controlAndIndirect; this.drawIndirect = this.controlAndIndirect;
    device.queue.writeBuffer(this.control, 8 * 4, new Uint32Array([
      this.nodeCapacity, this.leafCapacity, this.voxelCapacity, this.brickSize,
    ]));
    device.queue.writeBuffer(this.control, 16 * 4, new Uint32Array([
      this.leafTopologyOffsetBytes / 4, this.velocityOffsetBytes / 4, this.materialOwnerOffsetBytes / 4,
    ]));
    this.denseFieldParams = device.createBuffer({ label: `${label} dense publication parameters`, size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  }

  async initializePipelines(): Promise<void> {
    if (this.publicationPipelines.length > 0) return;
    this.publicationModule = this.device.createShaderModule({
      label: `${this.label} publication shader`, code: publicationShader,
    });
    this.denseFieldModule = this.device.createShaderModule({
      label: `${this.label} dense-field shader`, code: denseFieldShader,
    });
    this.publicationPipelines = [];
    for (const [stage, entryPoint] of [
      ["structure", "publishStructure"],
      ["geometry", "publishGeometry"],
      ["velocity", "publishVelocity"],
      ["material-owner", "publishMaterialOwners"],
    ] as const) {
      this.publicationPipelines = [...this.publicationPipelines,
        await this.device.createComputePipelineAsync({
          label: `${this.label} ${stage} publication pipeline`, layout: "auto",
          compute: { module: this.publicationModule, entryPoint },
        })];
    }
    this.denseFieldPipeline = await this.device.createComputePipelineAsync({
      label: `${this.label} dense-field pipeline`, layout: "auto",
      compute: { module: this.denseFieldModule, entryPoint: "materializeDenseFields" },
    });
    this.denseFieldCleanupPipeline = await this.device.createComputePipelineAsync({
      label: `${this.label} retired dense-field cleanup pipeline`, layout: "auto",
      compute: { module: this.denseFieldModule, entryPoint: "clearRetiredDenseFields" },
    });
  }

  /** Reset only authoritative counts/arguments; stale payload is unreachable. */
  encodeReset(encoder: GPUCommandEncoder): void {
    encoder.clearBuffer(this.control, 0, 8 * 4);
    encoder.clearBuffer(this.control, 12 * 4, 4 * 4);
    // Mutable allocator, work counts, indirect arguments, and mutation
    // generation occupy the remaining non-capacity tail of the control arena.
    encoder.clearBuffer(this.control, 19 * 4, 13 * 4);
  }

  /** Publish a GPU-authored topology and payload without any CPU count readback. */
  encodePublish(encoder: GPUCommandEncoder, source: SparseBrickPublicationSource): void {
    this.encodeReset(encoder);
    const [structurePipeline, geometryPipeline, velocityPipeline, materialOwnerPipeline] = this.publicationPipelines;
    // The three dynamic stages copy from source buffers that a renderer-only
    // world never fills — they write zeros over lanes it does not allocate. On
    // `dry` they would land in the absent-lane page and then in the scene
    // geometry behind it, so they are skipped outright rather than clamped.
    const dynamic = this.payloadLayout.lanes.geometry.present;
    const stages: readonly [string, GPUComputePipeline, readonly GPUBindGroupEntry[]][] = [
      ["structure", structurePipeline, [
        { binding: 0, resource: { buffer: source.counts } },
        { binding: 1, resource: { buffer: source.topology } },
        { binding: 5, resource: { buffer: this.structure } },
      ]],
      ...(dynamic ? [
        ["geometry", geometryPipeline, [
          { binding: 2, resource: { buffer: source.geometry } },
          { binding: 5, resource: { buffer: this.structure } },
          { binding: 7, resource: { buffer: this.payload } },
        ]],
        ["velocity", velocityPipeline, [
          { binding: 3, resource: { buffer: source.velocity } },
          { binding: 5, resource: { buffer: this.structure } },
          { binding: 7, resource: { buffer: this.payload } },
        ]],
        ["material-owner", materialOwnerPipeline, [
          { binding: 4, resource: { buffer: source.materialOwners } },
          { binding: 5, resource: { buffer: this.structure } },
          { binding: 7, resource: { buffer: this.payload } },
        ]],
      ] as const : []),
    ] as readonly [string, GPUComputePipeline, readonly GPUBindGroupEntry[]][];
    const maximum = Math.max(source.capacities.nodes, source.capacities.leaves, source.capacities.voxels, 1);
    const dispatch = sparseBrickDispatchDimensions(maximum);
    for (const [stage, pipeline, entries] of stages) {
      const bindGroup = this.device.createBindGroup({
        label: `Sparse brick ${stage} publication bind group`,
        layout: pipeline.getBindGroupLayout(0),
        entries,
      });
      const pass = encoder.beginComputePass({ label: `Publish sparse brick octree ${stage}` });
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(...dispatch);
      pass.end();
    }
  }

  /**
   * Fill the current leaves from the octree solver's resident dense fields.
   * Topology must already have been published earlier in this command stream.
   */
  encodeFromDenseFields(encoder: GPUCommandEncoder, source: SparseBrickDenseFieldSource): void {
    // This is the solver's own publication path and it writes all three dynamic
    // lanes. A world built `dry` has none of them; reaching here means a solver
    // was attached to a renderer-only world, which is a construction bug rather
    // than something to degrade past.
    if (!this.payloadLayout.lanes.geometry.present) {
      throw new Error(`${this.label} was built with the "${this.payloadProfile}" payload profile and has no dynamic lanes to publish into`);
    }
    const [nx, ny, nz] = source.dimensions;
    const origin = source.gridOriginCells ?? [0, 0, 0];
    for (const [value, name] of [[nx, "nx"], [ny, "ny"], [nz, "nz"]] as const) positiveCapacity(value, name);
    const materials = [source.fluidMaterialId, source.solidMaterialId];
    for (const material of materials) if (!Number.isInteger(material) || material < 0 || material > 0xffff) throw new RangeError("Material IDs must fit uint16");
    const words = new ArrayBuffer(64);
    const uints = new Uint32Array(words);
    const ints = new Int32Array(words);
    const floats = new Float32Array(words);
    const finestLevel = source.finestLevel;
    if (finestLevel !== undefined && (!Number.isInteger(finestLevel) || finestLevel < 0 || finestLevel > SPARSE_BRICK_MAX_MORTON_BITS)) throw new RangeError("Finest topology level is invalid");
    const activeWorklist = source.activeBrickWorklist;
    const encodedFinestLevel = finestLevel ?? 0x7fffffff;
    uints.set([nx, ny, nz, encodedFinestLevel | (activeWorklist ? 0x80000000 : 0)], 0);
    ints.set([origin[0], origin[1], origin[2], 0], 4);
    floats.set([source.cellSize[0], source.cellSize[1], source.cellSize[2], 0], 8);
    uints.set([source.fluidMaterialId, source.solidMaterialId, 0, 0], 12);
    this.device.queue.writeBuffer(this.denseFieldParams, 0, words);
    const bindGroup = this.device.createBindGroup({
      label: "Sparse brick dense-field bind group",
      layout: this.denseFieldPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.control, size: SPARSE_BRICK_GPU_LAYOUT.controlStrideBytes } },
        { binding: 1, resource: { buffer: this.topology, offset: this.topologyOffsetBytes } },
        { binding: 2, resource: source.levelSet },
        { binding: 3, resource: source.velocity },
        { binding: 4, resource: { buffer: source.solidCells } },
        { binding: 5, resource: { buffer: this.denseFieldParams } },
        { binding: 6, resource: { buffer: this.payload } },
        { binding: 7, resource: { buffer: activeWorklist ?? this.control } },
      ],
    });
    const pass = encoder.beginComputePass({ label: "Materialize octree dense fields into sparse bricks" });
    pass.setPipeline(this.denseFieldPipeline);
    pass.setBindGroup(0, bindGroup);
    if (activeWorklist) pass.dispatchWorkgroupsIndirect(activeWorklist, 4);
    else pass.dispatchWorkgroups(...sparseBrickDispatchDimensions(this.voxelCapacity));
    pass.end();
    if (activeWorklist) {
      const cleanupBindGroup = this.device.createBindGroup({
        label: "Retired sparse brick cleanup bind group",
        layout: this.denseFieldCleanupPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.control, size: SPARSE_BRICK_GPU_LAYOUT.controlStrideBytes } },
          { binding: 1, resource: { buffer: this.topology, offset: this.topologyOffsetBytes } },
          { binding: 5, resource: { buffer: this.denseFieldParams } },
          { binding: 6, resource: { buffer: this.payload } },
          { binding: 7, resource: { buffer: activeWorklist } },
        ],
      });
      const cleanup = encoder.beginComputePass({ label: "Clear retired sparse brick payloads" });
      cleanup.setPipeline(this.denseFieldCleanupPipeline);
      cleanup.setBindGroup(0, cleanupBindGroup);
      cleanup.dispatchWorkgroupsIndirect(activeWorklist, 20);
      cleanup.end();
    }
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.structure.destroy(); this.payload.destroy(); this.denseFieldParams.destroy();
  }
}
