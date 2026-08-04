import { SVO_BRICK_OCCUPANCY } from "./svo-brick-occupancy";
import { cameraApertureShaderLibrary } from "./webgpu-camera";

/**
 * Raster-assisted primary visibility: GPU-driven emission, frustum culling and
 * front-to-back ordering of one box instance per resident, occupied SVO leaf
 * brick (docs/SVO_RASTER_PRIMARY_HANDOFF.md).
 *
 * The instance box is the brick's *occupied sub-AABB*, decoded from the
 * occupancy word the sparse-brick producer already publishes in the terminal
 * node's `links.w`. Sub-boxes of disjoint boxes stay disjoint, so a ray still
 * meets each instance over one interval and those intervals remain totally
 * ordered — which is what makes depth resolution exact and makes ordering a
 * pure performance lever rather than a correctness one.
 *
 * Ordering is a bucket counting sort on the instance's minimum view depth.
 * It is deliberately approximate within a bucket: the depth test is the
 * authority, and the sort exists only so tile-based hidden-surface removal
 * gets to reject occluded bricks before shading them.
 */
export const SVO_BRICK_RASTER_CONTRACT = Object.freeze({
  /** vec3f proxy minimum + voxel offset, vec3f proxy maximum + node/key word. */
  instanceStrideBytes: 32,
  /** Two triangles per box face, drawn non-indexed like the rigid impostors. */
  verticesPerInstance: 36,
  /**
   * The box's far side is drawn, so a camera inside a brick still shades it.
   * The table below winds outward-CCW in world space; projection flips that to
   * clockwise in framebuffer coordinates for the near-facing triangles, which
   * WebGPU's default `ccw` front face then classifies as back faces.
   */
  cullMode: "back" as GPUCullMode,
  sortBuckets: 1024,
  /**
   * The instance word is `nodeIndex | (sortKey << 22)`: 22 bits of node index
   * under 10 bits of bucket. Ten bits is exactly `sortBuckets`, so the split is
   * not a free parameter — widening the node index costs sort resolution and
   * vice versa. `svoBrickRasterAddressableNodes` is the derived ceiling and
   * `assertSvoBrickRasterNodeAddressable` is the loud check; see both for what
   * happens past it.
   */
  sortKeyShift: 22,
  nodeIndexMask: (1 << 22) - 1,
  emitWorkgroupSize: 64,
  scanWorkgroupSize: 256,
  scatterWorkgroupSize: 64,
  /**
   * Fixed per-pixel conservative candidate arena. Garden's measured proxy
   * crossings are p90=9 and max=18 at 1500 square; twenty-four covers that
   * measured tail while an exact direct fallback handles every overflow.
   */
  coverageCandidatesPerPixel: 24,
  /**
   * packedSurface(16) + identityMedia(8) + splitGeometry(16) + splitIdentity(8).
   * Every plane the deferred lighting pass consumes is a depth-tested colour
   * attachment; nothing in this pass writes an untested storage texture.
   */
  colorAttachmentBytesPerSample: 48,
  colorAttachmentCount: 4,
  /** Group-zero bindings of the standalone emission/sort module. */
  bindings: Object.freeze({
    uniforms: 0,
    mapping: 1,
    structure: 2,
    candidates: 3,
    rasterPublication: 4,
  }),
  /** The dry-scene split group carries the sorted list into the vertex stage. */
  instanceDrawBinding: 2,
  coverageCountBinding: 3,
  coverageCandidateBinding: 4,
  /** Nearest sub-pixel proxy key published by the depth-tested coverage tier. */
  lodKeyBinding: 6,
  /** Current primary geometry, sampled by later coverage tiers for occlusion. */
  primaryGeometryBinding: 7,
  /** Current primary owner/field metadata for suppressing redundant upgrades. */
  primaryIdentityBinding: 8,
  /** `SvoMapping` prefix of `DryParams`; bound directly so the two cannot drift. */
  mappingBindingBytes: 48,
  sortStateHeaderWords: 8,
  entryPoints: Object.freeze({
    emit: "svoBrickEmitMain" as const,
    scan: "svoBrickScanMain" as const,
    scatter: "svoBrickScatterMain" as const,
    vertex: "svoBrickRasterVertex" as const,
    fragment: "svoBrickRasterFragment" as const,
    coverage: "svoBrickCoverageFragment" as const,
    resolve: "svoBrickCoverageResolveFragment" as const,
    overflowResolve: "svoBrickCoverageOverflowFragment" as const,
    background: "dryRasterPrimaryBackgroundMain" as const,
  }),
});

export function svoBrickRasterSortStateBytes(): number {
  return (SVO_BRICK_RASTER_CONTRACT.sortStateHeaderWords + SVO_BRICK_RASTER_CONTRACT.sortBuckets)
    * Uint32Array.BYTES_PER_ELEMENT;
}

export function svoBrickRasterPublicationInstanceOffsetBytes(): number {
  return Math.ceil(svoBrickRasterSortStateBytes() / 256) * 256;
}

/**
 * How many octree nodes the instance word can name: 2^22 = 4,194,304.
 *
 * Every drawn instance carries `nodeIndex | (sortKey << sortKeyShift)` in one
 * u32, and the resolve reads the node index back with `nodeIndexMask`. A node
 * index at or above this ceiling cannot survive that round trip.
 */
export function svoBrickRasterAddressableNodes(): number {
  return SVO_BRICK_RASTER_CONTRACT.nodeIndexMask + 1;
}

/**
 * The loud half of the node-index ceiling.
 *
 * The quiet half is in `svoBrickEmitMain`, which refuses to emit an instance
 * whose node index does not fit and so *drops the brick* rather than drawing
 * some other brick's payload in its place. That is the right degrade — there is
 * no coarser proxy to fall back to for a single brick — but on its own it is a
 * scene that renders with holes and says nothing, which is precisely the
 * failure `docs/svo-raster-visibility-handoff.md` §5/W4 exists to remove.
 *
 * (The handoff's table calls this row "silent aliasing past 4.19 M leaves". At
 * HEAD it is a silent *drop*, not aliasing: the `nodeIndex > MASK` clause in
 * the emit kernel already prevents the corruption. The remaining defect is the
 * silence, and the count of dropped leaves — see
 * `svoBrickRasterSortStateDiagnostics().unaddressable`, which this kernel now
 * makes derivable from the published counters.)
 *
 * Call this wherever a published node count first becomes known — it is a
 * scalar compare, and it turns "the far half of the world stopped drawing" into
 * a message that names the ceiling, the overage, and the two ways past it.
 */
export function assertSvoBrickRasterNodeAddressable(nodeCount: number, context = "published octree"): void {
  const addressable = svoBrickRasterAddressableNodes();
  if (!Number.isSafeInteger(nodeCount) || nodeCount < 0) {
    throw new RangeError(`Brick-raster node count for ${context} must be a non-negative safe integer`);
  }
  if (nodeCount > addressable) {
    throw new RangeError(
      `Brick-raster node index overflow: ${context} publishes ${nodeCount} nodes and the instance word addresses `
      + `${addressable} (${SVO_BRICK_RASTER_CONTRACT.sortKeyShift} bits under a `
      + `${32 - SVO_BRICK_RASTER_CONTRACT.sortKeyShift}-bit sort key). Every node past the ceiling is dropped by `
      + `svoBrickEmitMain and never drawn. Fix by lowering SVO_BRICK_RASTER_CONTRACT.sortBuckets to 512 or 256 and `
      + `raising sortKeyShift to 23 or 24 (coarser front-to-back bucketing, which is a performance lever only), or `
      + `by carrying the sort key in a second instance word.`);
  }
}

/**
 * Instance arena bytes for a published leaf capacity.
 *
 * `nodeCapacity` is optional only because the leaf count alone already proves
 * a violation — every leaf names a distinct node, so leaves > addressable nodes
 * cannot be drawn — while the node count is what the shader actually masks.
 * Pass it wherever it is known and the check becomes exact instead of necessary.
 */
export function svoBrickRasterInstanceBytes(leafCapacity: number, nodeCapacity?: number): number {
  if (!Number.isSafeInteger(leafCapacity) || leafCapacity < 1) {
    throw new RangeError("Brick-raster leaf capacity must be a positive safe integer");
  }
  assertSvoBrickRasterNodeAddressable(nodeCapacity ?? leafCapacity,
    nodeCapacity === undefined ? "published leaf capacity" : "published node capacity");
  return leafCapacity * SVO_BRICK_RASTER_CONTRACT.instanceStrideBytes;
}

/**
 * The published sort state, decoded — including the count the kernel cannot
 * print.
 *
 * `resident` is incremented for every leaf the emit kernel looked at, before
 * any rejection, and each of the three rejections has its own counter, so
 * whatever is left over is the number of leaves whose node index was stale or
 * past `nodeIndexMask`. That residual is the GPU-side tripwire for the row
 * above: it is zero on every scene that fits, and a capacity sweep that reads
 * it can fail on the first rung where a brick stops being drawable.
 */
export function svoBrickRasterSortStateDiagnostics(words: ArrayLike<number>): {
  vertexCount: number;
  drawn: number;
  candidates: number;
  culled: number;
  empty: number;
  resident: number;
  unaddressable: number;
} {
  const value = (index: number) => Number(words[index] ?? 0);
  const drawn = value(1);
  const candidates = value(4);
  const culled = value(5);
  const empty = value(6);
  const resident = value(7);
  return {
    vertexCount: value(0),
    drawn,
    candidates,
    culled,
    empty,
    resident,
    unaddressable: Math.max(0, resident - candidates - culled - empty),
  };
}

export function svoBrickRasterCoverageCountBytes(width: number, height: number): number {
  if (!Number.isSafeInteger(width) || width < 1 || !Number.isSafeInteger(height) || height < 1) {
    throw new RangeError("Brick-raster coverage dimensions must be positive safe integers");
  }
  return width * height * Uint32Array.BYTES_PER_ELEMENT;
}

export function svoBrickRasterCoverageCandidateBytes(width: number, height: number): number {
  return svoBrickRasterCoverageCountBytes(width, height)
    * SVO_BRICK_RASTER_CONTRACT.coverageCandidatesPerPixel;
}

/**
 * One arena, two disjoint coverage passes, plus the depth seed between them.
 *
 * Bricks and authored SDF proxies are rasterized into per-pixel candidate lists
 * by passes that never overlap in time — the brick coverage/resolve/overflow
 * triple completes before the scene-primitive coverage pass clears the counters
 * — so a second allocation would only duplicate the largest buffer in the
 * renderer. Each pass addresses the arena with its own stride and the wider of
 * the two sizes the allocation.
 *
 * The tail beyond `pixels * candidatesPerPixel` holds one word per pixel: the
 * nearest opaque distance the brick resolve published, which is what lets the
 * scene-primitive resolve reject a candidate that starts behind an already
 * resolved surface. It lives here rather than in a fifth binding so the shipped
 * coverage bind-group layout is unchanged, and it sits *after* every candidate
 * slot so neither stride can reach it.
 */
export function svoRasterCoverageArenaBytes(
  width: number, height: number, candidatesPerPixel: number,
): number {
  if (!Number.isSafeInteger(candidatesPerPixel) || candidatesPerPixel < SVO_BRICK_RASTER_CONTRACT.coverageCandidatesPerPixel) {
    throw new RangeError("Shared coverage arena must be at least the brick candidate capacity");
  }
  return svoBrickRasterCoverageCountBytes(width, height) * (candidatesPerPixel + 1);
}

/**
 * Overflow-driven indirect draw for the conservative coverage arms.
 *
 * Both overflow passes — the brick one and the authored-SDF one — exist purely
 * so that arena capacity is a performance parameter and never an image change.
 * They almost never write: garden's measured maximum crossing is 18 against a
 * 24-entry arena, so the overflow fragment discards on every pixel. Yet each
 * still runs its *vertex* stage over the whole instance list every frame —
 * 0.841 ms measured today, and vertex work is linear in instances, so ~8 ms at
 * a 10x brick count for a pass that draws nothing.
 *
 * The fix is an indirect draw whose instance count is zero unless some pixel
 * actually overflowed. Three facts decide where the state lives:
 *
 *   - only the coverage *fragment* knows a pixel overflowed, and the one
 *     read-write buffer it binds is the per-pixel count buffer;
 *   - only the cull *compute* pass knows how many instances survived culling,
 *     and it cannot see the count buffer;
 *   - the draw is issued between them.
 *
 * So the count buffer carries a short tail past its per-pixel range: the
 * fragment raises the flag there, a one-lane compute pass folds the flag and
 * the published instance count into a draw-args block in the same tail, and the
 * overflow pass draws indirectly from it. Nothing about the per-pixel range
 * moves, so every existing index into it stays valid, and a frame that clears
 * only `svoBrickRasterCoverageCountBytes` leaves the tail intact by
 * construction.
 */
/**
 * Integration, for the owner of `lib/webgpu-svo-dry-scene.ts`.
 *
 * Six edits, none of which touch a shipped layout or a pinned constant. Line
 * numbers are against the tree this was written on and will have moved.
 *
 * 1. `ensureBrickCoverageBuffers` (~:5347) — size the count buffer with the
 *    tail and let it be drawn from:
 *      -  size: svoBrickRasterCoverageCountBytes(width, height),
 *      -  usage: STORAGE | COPY_DST | COPY_SRC,
 *      +  size: svoRasterCoverageCountAllocationBytes(width, height),
 *      +  usage: STORAGE | COPY_DST | COPY_SRC | INDIRECT,
 *
 * 2. Both `encoder.clearBuffer(this.brickCoverageCountBuffer!)` calls (~:5571,
 *    ~:5648) — clear the pixel range only, so the tail survives:
 *      +  encoder.clearBuffer(this.brickCoverageCountBuffer!, 0,
 *      +    svoBrickRasterCoverageCountBytes(this.brickCoverageWidth!, this.brickCoverageHeight!));
 *
 * 3. Coverage fragments (~:2687 for bricks, and the authored-SDF one) — raise
 *    the flag on the fragment that takes the *first* slot past capacity, so the
 *    counter counts overflowing pixels rather than surplus fragments:
 *        if(slot<CAPACITY){ ...append...; discard; }
 *      +  if(slot==CAPACITY){svoRasterCoverageOverflowSignal();}
 *        return 1u;
 *    with `${svoRasterCoverageOverflowSignalWGSL}` included beside them. It
 *    reads `svoBrickCoverageCounts`, which both fragments already bind.
 *
 * 4. A publisher pipeline from `createSvoRasterCoverageOverflowArgsWGSL()` over
 *    `svoRasterCoverageOverflowArgsBindGroupLayoutEntries()`, bound to
 *    {0: brickSortStateBuffer at brickSortStateOffsetBytes, 1: the count
 *    buffer}. For the authored-SDF arm pass
 *    `{ verticesPerInstance: SVO_SCENE_PRIMITIVE_RASTER_CONTRACT.verticesPerProxy,
 *       overflowInstanceCount: this.primitiveCount }` — that arm's instance
 *    count is a host number, not a GPU-culled one, so it needs no publication.
 *
 * 5. One `beginComputePass` dispatching it (1 workgroup) between the resolve and
 *    the overflow pass of each arm.
 *
 * 6. The overflow draws:
 *      -  overflow.drawIndirect(this.brickSortStateBuffer!, this.brickSortStateOffsetBytes);
 *      +  overflow.drawIndirect(this.brickCoverageCountBuffer!,
 *      +    svoRasterCoverageOverflowDrawArgsOffsetBytes(this.brickCoverageWidth!, this.brickCoverageHeight!));
 *    and the authored-SDF arm's `overflow.draw(verticesPerProxy, primitiveCount)`
 *    becomes the same `drawIndirect`.
 *
 * 7. Once (3) is in, the overflow *rate* is one word instead of a
 *    width-by-height readback, which is what makes it cheap enough to report
 *    every frame. Add a `copyCoverageOverflowCount(encoder, target)` beside the
 *    existing `copyCoverageCounts` and pass the result through
 *    `svoRasterCoverageOverflowStatus`; a `console.warn` on the first
 *    `over-budget` frame is the whole of the loud half. Today that row is
 *    observable only by scanning every pixel, which is why nothing does it —
 *    `tools/run-svo-capacity-sweep-smoke.ts` scans, and measures 0 overflowing
 *    pixels at every record rung to 10x at 800x460, busiest pixel 37 of 40.
 *    That zero is not reassurance: it means the overflow pass currently draws
 *    its entire instance list, rasterizes every proxy and discards every
 *    fragment, and edit (6) removes all of it.
 *
 * The guards that read `arrayLength(&svoBrickCoverageCounts)` as a pixel bound
 * stay correct — `pixel < width*height <= arrayLength - tailWords` — but they
 * become permissive by `tailWords`, so tightening them to an explicit pixel
 * count is worth doing in the same edit.
 *
 * Verification without the renderer: `tools/run-svo-capacity-sweep-smoke.ts`
 * compiles this module and asserts both arms of the publisher — zero instances
 * with the flag clear, the published count with it raised.
 */
export const SVO_RASTER_COVERAGE_OVERFLOW_CONTRACT = Object.freeze({
  /** Words appended past `width * height`. */
  tailWords: 8,
  /** Tail word raised once per overflowing pixel; also the "did anything overflow" flag. */
  overflowPixelWord: 0,
  /** `{vertexCount, instanceCount, firstVertex, firstInstance}` for `drawIndirect`. */
  drawArgsWord: 4,
  entryPoints: Object.freeze({
    publishArgs: "svoRasterCoverageOverflowArgsMain" as const,
  }),
  bindings: Object.freeze({
    publication: 0,
    coverageCounts: 1,
  }),
  /** Word index of `drawInstanceCount` inside the published sort state. */
  publicationInstanceCountWord: 1,
});

/**
 * The fraction of covered pixels allowed to exceed the per-pixel arena.
 *
 * W1's gate ("overflow rate < 1 % of covered pixels",
 * `docs/svo-raster-visibility-handoff.md` §5/W1) as a number the code can
 * enforce rather than a sentence in a document. Overflow is image-preserving by
 * construction — the overflow pass re-runs the exact fragment for the pixels
 * that exceeded capacity — which is exactly why nothing has ever noticed it
 * moving: the frame stays correct and only the cost changes.
 */
export const SVO_RASTER_COVERAGE_OVERFLOW_BUDGET = 0.01;

/**
 * The arena's overflow rate, classified, with the fix named.
 *
 * This is the row the capacity table calls "silent-by-design overflow", and
 * silent-by-design is the correct design for the *image* and the wrong one for
 * the *frame*: on this path an overflowing arena moves the cost from one
 * bounded per-pixel walk to a second full re-march of every covering proxy, and
 * at 10x records that has been measured as the single largest item in the
 * frame. Capacity remaining a performance parameter is the guarantee; nobody
 * ever promised the performance would be good, and nothing was reporting when
 * it stopped being.
 *
 * Deliberately a classifier and not a throw: unlike the node index, this
 * capacity is *meant* to be crossed occasionally and the degrade is exact.
 * The tripwire is that crossing it in bulk becomes a named, measured statement
 * instead of a slow frame nobody can attribute.
 */
export function svoRasterCoverageOverflowStatus(input: {
  coveredPixels: number;
  overflowPixels: number;
  candidatesPerPixel: number;
  arm: string;
  budget?: number;
}): { state: "clear" | "within-budget" | "over-budget"; fraction: number; message: string } {
  const budget = input.budget ?? SVO_RASTER_COVERAGE_OVERFLOW_BUDGET;
  const fraction = input.coveredPixels > 0 ? input.overflowPixels / input.coveredPixels : 0;
  const percent = (100 * fraction).toFixed(3);
  if (input.overflowPixels === 0) {
    return {
      state: "clear",
      fraction,
      message: `${input.arm} coverage arena: no pixel exceeded ${input.candidatesPerPixel} candidates`
        + ` over ${input.coveredPixels} covered pixels`,
    };
  }
  if (fraction <= budget) {
    return {
      state: "within-budget",
      fraction,
      message: `${input.arm} coverage arena: ${input.overflowPixels} of ${input.coveredPixels} covered pixels`
        + ` (${percent} %) exceeded ${input.candidatesPerPixel} candidates and were re-marched by the overflow pass`,
    };
  }
  return {
    state: "over-budget",
    fraction,
    message: `${input.arm} coverage arena overflow: ${input.overflowPixels} of ${input.coveredPixels} covered pixels`
      + ` (${percent} %) exceeded ${input.candidatesPerPixel} candidates, against a ${(100 * budget).toFixed(1)} %`
      + ` budget. The image is still exact — the overflow pass re-marches every covering proxy for those pixels —`
      + ` but that pass is the exact per-fragment cost the arena exists to avoid, so this is the frame going`
      + ` quadratic in covering proxies again. Fix by raising coverageCandidatesPerPixel (linear arena memory:`
      + ` width * height * 4 B per candidate), by shrinking the covering set (W3's near-field band), or by`
      + ` accepting it and giving the overflow pass an overflow-driven indirect count so at least the frames`
      + ` that do not overflow stop paying for it.`,
  };
}

/**
 * Bytes to allocate for the per-pixel count buffer: the pixel range plus the
 * overflow tail. `svoBrickRasterCoverageCountBytes` remains the *pixel* range
 * and is what a per-frame clear must use, so the tail survives the clear.
 *
 * The buffer needs `GPUBufferUsage.INDIRECT` in addition to its existing usage,
 * because the tail is what the overflow pass draws from.
 */
export function svoRasterCoverageCountAllocationBytes(width: number, height: number): number {
  return svoBrickRasterCoverageCountBytes(width, height)
    + SVO_RASTER_COVERAGE_OVERFLOW_CONTRACT.tailWords * Uint32Array.BYTES_PER_ELEMENT;
}

/** Byte offset of the draw-args block, for `drawIndirect`. */
export function svoRasterCoverageOverflowDrawArgsOffsetBytes(width: number, height: number): number {
  return svoBrickRasterCoverageCountBytes(width, height)
    + SVO_RASTER_COVERAGE_OVERFLOW_CONTRACT.drawArgsWord * Uint32Array.BYTES_PER_ELEMENT;
}

/**
 * Raise the overflow flag. Included by the dry scene beside its coverage
 * fragments, which declare the count buffer as `array<atomic<u32>>`.
 *
 * Called only on the fragment that takes the first slot past capacity, so the
 * counter is a count of overflowing *pixels* rather than of surplus fragments —
 * which is the number the W1 gate ("overflow rate < 1 % of covered pixels")
 * asks for, and a diagnostic worth having on its own.
 */
export const svoRasterCoverageOverflowSignalWGSL = /* wgsl */ `
const SVO_RASTER_COVERAGE_TAIL_WORDS:u32=${SVO_RASTER_COVERAGE_OVERFLOW_CONTRACT.tailWords}u;
fn svoRasterCoverageOverflowSignal(){
  let words=arrayLength(&svoBrickCoverageCounts);
  if(words<SVO_RASTER_COVERAGE_TAIL_WORDS){return;}
  atomicAdd(&svoBrickCoverageCounts[words-SVO_RASTER_COVERAGE_TAIL_WORDS+${SVO_RASTER_COVERAGE_OVERFLOW_CONTRACT.overflowPixelWord}u],1u);
}
`;

/**
 * Group-zero layout of the one-lane publisher.
 *
 * Deliberately its own layout and its own module rather than a fourth entry
 * point on the cull module: the cull module already binds the publication as
 * read-write at group 0, and one module cannot declare the same slot twice with
 * two access modes. Keeping it separate also leaves every shipped layout in
 * this file byte-identical.
 */
export function svoRasterCoverageOverflowArgsBindGroupLayoutEntries(): GPUBindGroupLayoutEntry[] {
  const { bindings } = SVO_RASTER_COVERAGE_OVERFLOW_CONTRACT;
  const visibility = GPUShaderStage.COMPUTE;
  return [
    { binding: bindings.publication, visibility, buffer: { type: "read-only-storage" } },
    { binding: bindings.coverageCounts, visibility, buffer: { type: "storage" } },
  ];
}

/**
 * The publisher. One lane, no loop: it reads the flag the coverage fragment
 * raised and the instance count the scan published, and writes the draw-args
 * block. `instanceCount` is the *culled* count, not the arena capacity, because
 * the whole point is to stop paying vertex work for instances that exist.
 *
 * `overflowInstanceCount` overrides that source for the authored-SDF arm, whose
 * coverage pass draws a host-known record count rather than a GPU-culled one.
 */
export function createSvoRasterCoverageOverflowArgsWGSL(options: {
  verticesPerInstance?: number;
  overflowInstanceCount?: number;
} = {}): string {
  const contract = SVO_RASTER_COVERAGE_OVERFLOW_CONTRACT;
  const { bindings } = contract;
  const vertices = options.verticesPerInstance ?? SVO_BRICK_RASTER_CONTRACT.verticesPerInstance;
  const instances = options.overflowInstanceCount === undefined
    ? `svoRasterOverflowPublication[${contract.publicationInstanceCountWord}u]`
    : `${options.overflowInstanceCount}u`;
  return /* wgsl */ `
@group(0) @binding(${bindings.publication}) var<storage,read> svoRasterOverflowPublication:array<u32>;
@group(0) @binding(${bindings.coverageCounts}) var<storage,read_write> svoRasterOverflowCounts:array<u32>;
const SVO_RASTER_COVERAGE_TAIL_WORDS:u32=${contract.tailWords}u;
@compute @workgroup_size(1)
fn ${contract.entryPoints.publishArgs}(){
  let words=arrayLength(&svoRasterOverflowCounts);
  if(words<SVO_RASTER_COVERAGE_TAIL_WORDS){return;}
  let tail=words-SVO_RASTER_COVERAGE_TAIL_WORDS;
  let overflowed=svoRasterOverflowCounts[tail+${contract.overflowPixelWord}u];
  let args=tail+${contract.drawArgsWord}u;
  // The vertex count is written every frame rather than once per allocation:
  // the frame's own clear of the pixel range must be allowed to grow into the
  // tail without silently turning the overflow pass into a no-op.
  svoRasterOverflowCounts[args]=${vertices}u;
  svoRasterOverflowCounts[args+1u]=select(0u,${instances},overflowed!=0u);
  svoRasterOverflowCounts[args+2u]=0u;
  svoRasterOverflowCounts[args+3u]=0u;
}
`;
}

/** Standalone group-zero layout for the emission, scan and scatter passes. */
export function svoBrickRasterCullBindGroupLayoutEntries(): GPUBindGroupLayoutEntry[] {
  const { bindings } = SVO_BRICK_RASTER_CONTRACT;
  const visibility = GPUShaderStage.COMPUTE;
  return [
    { binding: bindings.uniforms, visibility, buffer: { type: "uniform" } },
    { binding: bindings.mapping, visibility, buffer: { type: "uniform" } },
    { binding: bindings.structure, visibility, buffer: { type: "read-only-storage" } },
    { binding: bindings.candidates, visibility, buffer: { type: "storage" } },
    { binding: bindings.rasterPublication, visibility, buffer: { type: "storage" } },
  ];
}

/**
 * The direct control reads only the sorted instances. The production coverage
 * arm additionally appends instance indices per pixel, then reads the same
 * arena from its one-fragment-per-pixel resolve. Counts remain atomic because
 * overlapping proxy fragments are unordered by the rasterizer.
 */
export function svoBrickRasterDrawBindGroupLayoutEntries(): GPUBindGroupLayoutEntry[] {
  return [{
    binding: SVO_BRICK_RASTER_CONTRACT.instanceDrawBinding,
    visibility: GPUShaderStage.VERTEX,
    buffer: { type: "read-only-storage" },
  }];
}

export function svoBrickRasterCoverageBindGroupLayoutEntries(): GPUBindGroupLayoutEntry[] {
  return [
    { binding: SVO_BRICK_RASTER_CONTRACT.instanceDrawBinding,
      visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT | GPUShaderStage.COMPUTE,
      buffer: { type: "read-only-storage" } },
    {
      binding: SVO_BRICK_RASTER_CONTRACT.coverageCountBinding,
      visibility: GPUShaderStage.FRAGMENT | GPUShaderStage.COMPUTE,
      buffer: { type: "storage" },
    },
    {
      binding: SVO_BRICK_RASTER_CONTRACT.coverageCandidateBinding,
      visibility: GPUShaderStage.FRAGMENT | GPUShaderStage.COMPUTE,
      buffer: { type: "storage" },
    },
  ];
}

/**
 * Outward, counter-clockwise unit-box corner indices. Corner `i` is
 * `(i & 1, (i >> 1) & 1, (i >> 2) & 1)`; the pipeline culls front faces, so
 * these wind so that the *back* faces survive.
 */
export const SVO_BRICK_RASTER_BOX_CORNERS: readonly number[] = Object.freeze([
  0, 4, 6, 0, 6, 2, // -X
  1, 3, 7, 1, 7, 5, // +X
  0, 1, 5, 0, 5, 4, // -Y
  2, 6, 7, 2, 7, 3, // +Y
  0, 2, 3, 0, 3, 1, // -Z
  4, 5, 7, 4, 7, 6, // +Z
]);

/** Shared declarations: the instance record, the sort state, and the box table. */
export const svoBrickRasterSharedWGSL = /* wgsl */ `
struct SvoBrickInstance{proxyMinimum:vec3f,voxelOffset:u32,proxyMaximum:vec3f,nodeIndexKey:u32}
const SVO_BRICK_SORT_BUCKETS:u32=${SVO_BRICK_RASTER_CONTRACT.sortBuckets}u;
const SVO_BRICK_SORT_KEY_SHIFT:u32=${SVO_BRICK_RASTER_CONTRACT.sortKeyShift}u;
const SVO_BRICK_NODE_INDEX_MASK:u32=${SVO_BRICK_RASTER_CONTRACT.nodeIndexMask}u;
fn svoBrickBoxCorner(vertexIndex:u32)->vec3f{
  var table=array<u32,${SVO_BRICK_RASTER_CONTRACT.verticesPerInstance}>(${SVO_BRICK_RASTER_BOX_CORNERS.map((corner) => `${corner}u`).join(",")});
  let corner=table[vertexIndex];
  return vec3f(f32(corner&1u),f32((corner>>1u)&1u),f32((corner>>2u)&1u));
}
`;

/**
 * Emission, prefix scan and scatter. Deliberately standalone: it needs the
 * camera uniform, the published topology and the `SvoMapping` prefix of
 * `DryParams`, none of the renderer's fragment-only shading bindings.
 */
export function createSvoBrickRasterCullWGSL(options: { reversedZNear_m: number }): string {
  const { bindings } = SVO_BRICK_RASTER_CONTRACT;
  return /* wgsl */ `
struct Uniforms{viewport:vec4f,cameraPosition:vec4f,cameraTarget:vec4f,container:vec4f,options:vec4f,gridInfo:vec4f,debug:vec4f,environment:vec4f,terrainMeta:vec4f,terrainFeatures:array<vec4f,16>}
struct SvoNode{address:vec4u,links:vec4u}
struct SvoLeaf{topology:vec4u}
struct SvoMapping{worldOrigin:vec3f,brickSize:u32,cellSize:vec3f,maximumDepth:u32,nodeCount:u32,leafCount:u32,maxVisits:u32,_padding:u32}
struct SvoBrickSortState{
  drawVertexCount:u32,
  drawInstanceCount:atomic<u32>,
  drawFirstVertex:u32,
  drawFirstInstance:u32,
  candidateCount:atomic<u32>,
  culled:atomic<u32>,
  empty:atomic<u32>,
  resident:atomic<u32>,
  buckets:array<atomic<u32>,${SVO_BRICK_RASTER_CONTRACT.sortBuckets}>,
}
${svoBrickRasterSharedWGSL}
${svoBrickOccupancyDecodeWGSL}
struct SvoBrickRasterPublication{
  sort:SvoBrickSortState,
  _instanceAlignment:array<u32,${(svoBrickRasterPublicationInstanceOffsetBytes() - svoBrickRasterSortStateBytes()) / 4}>,
  instances:array<SvoBrickInstance>,
}

@group(0) @binding(${bindings.uniforms}) var<uniform> uniforms:Uniforms;
${cameraApertureShaderLibrary()}
@group(0) @binding(${bindings.mapping}) var<uniform> mapping:SvoMapping;
@group(0) @binding(${bindings.structure}) var<storage,read> svoBrickStructure:array<u32>;
@group(0) @binding(${bindings.candidates}) var<storage,read_write> svoBrickCandidates:array<SvoBrickInstance>;
@group(0) @binding(${bindings.rasterPublication}) var<storage,read_write> svoBrickRaster:SvoBrickRasterPublication;

fn svoBrickStructureWords4(offset:u32)->vec4u{return vec4u(svoBrickStructure[offset],svoBrickStructure[offset+1u],svoBrickStructure[offset+2u],svoBrickStructure[offset+3u]);}
fn svoBrickControl(index:u32)->u32{return svoBrickStructure[index];}
fn svoBrickNode(index:u32)->SvoNode{let base=128u+index*8u;return SvoNode(svoBrickStructureWords4(base),svoBrickStructureWords4(base+4u));}
fn svoBrickLeaf(index:u32)->SvoLeaf{let base=128u+svoBrickControl(16u)+index*4u;return SvoLeaf(svoBrickStructureWords4(base));}

const SVO_BRICK_NEAR_M:f32=${options.reversedZNear_m};

fn svoBrickCompactMortonBits(value:vec3u)->vec3u{
  var compact=value&vec3u(0x49249249u);
  compact=(compact^(compact>>vec3u(2u)))&vec3u(0xc30c30c3u);
  compact=(compact^(compact>>vec3u(4u)))&vec3u(0x0f00f00fu);
  compact=(compact^(compact>>vec3u(8u)))&vec3u(0xff0000ffu);
  return (compact^(compact>>vec3u(16u)))&vec3u(0x0000ffffu);
}
fn svoBrickDecodeMorton(low:u32,high:u32,level:u32)->vec3u{
  let levelMask=(1u<<level)-1u;
  let lowBits=svoBrickCompactMortonBits(vec3u(low,low>>1u,low>>2u));
  let highBits=svoBrickCompactMortonBits(vec3u(high>>1u,high>>2u,high));
  return (lowBits|(highBits<<vec3u(11u,11u,10u)))&vec3u(levelMask);
}
fn svoBrickNodeBounds(node:SvoNode)->mat2x3f{
  let coordinate=vec3f(svoBrickDecodeMorton(node.address.x,node.address.y,node.address.z));
  let scale=f32((1u<<(mapping.maximumDepth-node.address.z))*mapping.brickSize);
  let minimum=mapping.worldOrigin+coordinate*scale*mapping.cellSize;
  return mat2x3f(minimum,minimum+scale*mapping.cellSize);
}
struct SvoBrickCamera{origin:vec3f,forward:vec3f,right:vec3f,up:vec3f,aspect:f32}
fn svoBrickCamera()->SvoBrickCamera{
  let origin=uniforms.cameraPosition.xyz;
  let forward=normalize(uniforms.cameraTarget.xyz-origin);
  let right=normalize(cross(forward,vec3f(0.0,1.0,0.0)));
  return SvoBrickCamera(origin,forward,right,normalize(cross(right,forward)),
    uniforms.viewport.x/max(uniforms.viewport.y,1.0));
}
/** Conservative plane rejection: a box survives unless every corner is outside one plane. */
fn svoBrickFrustumVisible(camera:SvoBrickCamera,bounds:mat2x3f)->bool{
  var outside=vec4<bool>(true,true,true,true);
  var outsideNear=true;
  for(var corner=0u;corner<8u;corner+=1u){
    let point=vec3f(
      select(bounds[0].x,bounds[1].x,(corner&1u)!=0u),
      select(bounds[0].y,bounds[1].y,(corner&2u)!=0u),
      select(bounds[0].z,bounds[1].z,(corner&4u)!=0u));
    let relative=point-camera.origin;
    let viewDepth=dot(relative,camera.forward);
    let lateral=viewDepth*cameraTanHalfFov();
    let x=dot(relative,camera.right);
    let y=dot(relative,camera.up);
    outsideNear=outsideNear&&(viewDepth<SVO_BRICK_NEAR_M);
    outside=vec4<bool>(
      outside.x&&(x+lateral*camera.aspect<0.0),
      outside.y&&(lateral*camera.aspect-x<0.0),
      outside.z&&(y+lateral<0.0),
      outside.w&&(lateral-y<0.0));
  }
  return !(outsideNear||any(outside));
}
/** Minimum view depth of the box; the counting-sort key is a quantization of it. */
fn svoBrickSortKey(camera:SvoBrickCamera,bounds:mat2x3f)->u32{
  let center=0.5*(bounds[0]+bounds[1]);
  let halfExtent=0.5*(bounds[1]-bounds[0]);
  let nearDepth=dot(center-camera.origin,camera.forward)-dot(abs(camera.forward),halfExtent);
  let rootScale=f32((1u<<mapping.maximumDepth)*mapping.brickSize);
  let rootHalf=0.5*rootScale*mapping.cellSize;
  let rootCenter=mapping.worldOrigin+rootHalf;
  let farDepth=dot(rootCenter-camera.origin,camera.forward)+dot(abs(camera.forward),rootHalf);
  let normalized=max(nearDepth,0.0)/max(farDepth,1e-3);
  return min(u32(f32(SVO_BRICK_SORT_BUCKETS)*clamp(normalized,0.0,1.0)),SVO_BRICK_SORT_BUCKETS-1u);
}
fn svoBrickTopologyPublished()->bool{
  return svoBrickStructure[64u]!=0u;
}

@compute @workgroup_size(${SVO_BRICK_RASTER_CONTRACT.emitWorkgroupSize})
fn ${SVO_BRICK_RASTER_CONTRACT.entryPoints.emit}(@builtin(global_invocation_id) globalId:vec3u){
  let leafIndex=globalId.x;
  if(!svoBrickTopologyPublished()){return;}
  if(leafIndex>=svoBrickControl(1u)||leafIndex>=arrayLength(&svoBrickCandidates)){return;}
  let leaf=svoBrickLeaf(leafIndex).topology;
  let nodeIndex=leaf.x;
  // Counted before any rejection, so that resident - candidates - culled -
  // empty is exactly the number of leaves the line below dropped for having an
  // unaddressable or stale node index. Nothing in the emit kernel can print;
  // this residual is what makes that drop visible to a reader of the published
  // state (svoBrickRasterSortStateDiagnostics) instead of silent.
  atomicAdd(&svoBrickRaster.sort.resident,1u);
  if(nodeIndex>=svoBrickControl(0u)||nodeIndex>SVO_BRICK_NODE_INDEX_MASK){return;}
  let node=svoBrickNode(nodeIndex);
  let occupancy=svoBrickOccupancyDecode(node.links.w);
  // An empty brick can never produce a primary hit, so it is never drawn. The
  // producer publishes this word every topology change, which is why the
  // instance list can be rebuilt from scratch each frame instead of cached.
  if(occupancy.ready!=0u&&occupancy.occupied==0u){atomicAdd(&svoBrickRaster.sort.empty,1u);return;}
  let bounds=svoBrickNodeBounds(node);
  let cellSize=(bounds[1]-bounds[0])/f32(mapping.brickSize);
  var proxy=bounds;
  if(occupancy.ready!=0u){proxy=svoBrickOccupiedBounds(occupancy,bounds[0],cellSize);}
  let camera=svoBrickCamera();
  if(!svoBrickFrustumVisible(camera,proxy)){atomicAdd(&svoBrickRaster.sort.culled,1u);return;}
  let key=svoBrickSortKey(camera,proxy);
  let slot=atomicAdd(&svoBrickRaster.sort.candidateCount,1u);
  if(slot>=arrayLength(&svoBrickCandidates)){return;}
  svoBrickCandidates[slot]=SvoBrickInstance(proxy[0],leaf.y,proxy[1],nodeIndex|(key<<SVO_BRICK_SORT_KEY_SHIFT));
  atomicAdd(&svoBrickRaster.sort.buckets[key],1u);
}

var<workgroup> svoBrickBucketScan:array<u32,${SVO_BRICK_RASTER_CONTRACT.sortBuckets}>;
var<workgroup> svoBrickLaneTotals:array<u32,${SVO_BRICK_RASTER_CONTRACT.scanWorkgroupSize}>;
const SVO_BRICK_BUCKETS_PER_LANE:u32=${SVO_BRICK_RASTER_CONTRACT.sortBuckets / SVO_BRICK_RASTER_CONTRACT.scanWorkgroupSize}u;

/** Exclusive prefix sum over the bucket histogram; also publishes the draw count. */
@compute @workgroup_size(${SVO_BRICK_RASTER_CONTRACT.scanWorkgroupSize})
fn ${SVO_BRICK_RASTER_CONTRACT.entryPoints.scan}(@builtin(local_invocation_id) localId:vec3u){
  let lane=localId.x;
  let base=lane*SVO_BRICK_BUCKETS_PER_LANE;
  var laneTotal=0u;
  for(var offset=0u;offset<SVO_BRICK_BUCKETS_PER_LANE;offset+=1u){
    let value=atomicLoad(&svoBrickRaster.sort.buckets[base+offset]);
    svoBrickBucketScan[base+offset]=laneTotal;
    laneTotal+=value;
  }
  svoBrickLaneTotals[lane]=laneTotal;
  workgroupBarrier();
  // Hillis-Steele over the lane totals; ${SVO_BRICK_RASTER_CONTRACT.scanWorkgroupSize} lanes is a fixed log2 rounds.
  var stride=1u;
  loop{
    if(stride>=${SVO_BRICK_RASTER_CONTRACT.scanWorkgroupSize}u){break;}
    var added=svoBrickLaneTotals[lane];
    if(lane>=stride){added+=svoBrickLaneTotals[lane-stride];}
    workgroupBarrier();
    svoBrickLaneTotals[lane]=added;
    workgroupBarrier();
    stride*=2u;
  }
  let laneExclusive=svoBrickLaneTotals[lane]-laneTotal;
  for(var offset=0u;offset<SVO_BRICK_BUCKETS_PER_LANE;offset+=1u){
    atomicStore(&svoBrickRaster.sort.buckets[base+offset],svoBrickBucketScan[base+offset]+laneExclusive);
  }
  if(lane==${SVO_BRICK_RASTER_CONTRACT.scanWorkgroupSize - 1}u){
    atomicStore(&svoBrickRaster.sort.drawInstanceCount,min(svoBrickLaneTotals[lane],arrayLength(&svoBrickRaster.instances)));
  }
}

@compute @workgroup_size(${SVO_BRICK_RASTER_CONTRACT.scatterWorkgroupSize})
fn ${SVO_BRICK_RASTER_CONTRACT.entryPoints.scatter}(@builtin(global_invocation_id) globalId:vec3u){
  let candidate=globalId.x;
  if(candidate>=atomicLoad(&svoBrickRaster.sort.candidateCount)||candidate>=arrayLength(&svoBrickCandidates)){return;}
  let record=svoBrickCandidates[candidate];
  let key=record.nodeIndexKey>>SVO_BRICK_SORT_KEY_SHIFT;
  let slot=atomicAdd(&svoBrickRaster.sort.buckets[key],1u);
  if(slot>=arrayLength(&svoBrickRaster.instances)){return;}
  svoBrickRaster.instances[slot]=record;
}
`;
}

/** Local copy of the occupancy decoder so the cull module stays self-contained. */
const svoBrickOccupancyDecodeWGSL = /* wgsl */ `
struct SvoBrickOccupancy{ready:u32,occupied:u32,macroMask:u32,minInclusive:vec3u,maxInclusive:vec3u}
fn svoBrickOccupancyDecode(packed:u32)->SvoBrickOccupancy{
  return SvoBrickOccupancy(
    select(0u,1u,(packed&${SVO_BRICK_OCCUPANCY.readyBit}u)!=0u),
    select(0u,1u,(packed&${SVO_BRICK_OCCUPANCY.occupiedBit}u)!=0u),
    packed&0xffu,
    vec3u((packed>>8u)&7u,(packed>>11u)&7u,(packed>>14u)&7u),
    vec3u((packed>>17u)&7u,(packed>>20u)&7u,(packed>>23u)&7u));
}
fn svoBrickOccupiedBounds(summary:SvoBrickOccupancy,brickMinimum:vec3f,cellSize:vec3f)->mat2x3f{
  let minimum=brickMinimum+vec3f(summary.minInclusive)*cellSize;
  let maximum=brickMinimum+vec3f(summary.maxInclusive+vec3u(1u))*cellSize;
  return mat2x3f(minimum,maximum);
}
`;
