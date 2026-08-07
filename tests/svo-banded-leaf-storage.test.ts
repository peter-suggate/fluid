/**
 * The banded leaf payload's *storage*, as distinct from its encoding.
 *
 * `tests/svo-banded-leaf-payload.test.ts` holds the reference encoder in
 * `lib/svo-banded-leaf-payload.ts` — the layout's executable spec. This file holds
 * the arena that has to carry it: the lane widths, the fixed 144 bytes a leaf, the
 * shared WGSL codec's addressing, and the three prerequisite edits without which
 * none of it is expressible.
 *
 * What it deliberately does not assert is fidelity. That question is closed
 * elsewhere and favourably (B0.5b: mean 0.54/255, max 7/255, zero pixels over
 * 16/255, against the `f16-unorm8` geometry lane already shipping at 11/255).
 */
import test from "node:test";
import assert from "node:assert/strict";
import {
  SPARSE_BRICK_BANDED_ALLOCATOR_WORDS,
  SPARSE_BRICK_BANDED_BLOB_BYTES_PER_LEAF,
  SPARSE_BRICK_BANDED_HEADER_WORDS,
  SPARSE_BRICK_BANDED_PRODUCER_DENSE_LANES,
  SPARSE_BRICK_BANDED_RECORD_CAPACITY_FRACTION,
  SPARSE_BRICK_LEAF_PAYLOAD_MODES,
  resolveSparseBrickPayloadLayout,
  sparseBrickBandedLeafCodecWGSL,
  sparseBrickLaneStrideBytes,
  sparseBrickLaneVoxelsPerWord,
  sparseBrickSceneIdentityCodecWGSL,
  type SparseBrickLeafPayloadMode,
} from "../lib/sparse-brick-octree";
import { createSvoDrySceneFragmentWGSL } from "../lib/webgpu-svo-dry-scene";
import {
  SVO_BANDED_LEAF_FIXED_BYTES,
  SVO_BANDED_LEAF_INDEX_BITS,
  SVO_BANDED_LEAF_VOXELS,
} from "../lib/svo-banded-leaf-payload";
import { sparseSceneProxyVoxelizationShaderFor } from "../lib/webgpu-sparse-scene-proxies";
import { octreeLiveSceneLeafPayloadMode } from "../lib/webgpu-octree-sparse-bricks";

const VOXELS = 1 << 20;
const LEAVES = VOXELS / SVO_BANDED_LEAF_VOXELS;

test("a mask lane costs a fraction of a byte a voxel, not a whole one", () => {
  // The first of the three prerequisite edits SP21 identified. A one-bit lane that
  // floors at a byte costs 512 bytes a leaf instead of 64, which is eight times the
  // fixed part of the whole layout — the occupancy mask would not have been worth
  // writing.
  assert.equal(sparseBrickLaneStrideBytes(["u1"]), 1 / 8);
  assert.equal(sparseBrickLaneStrideBytes(["u1", "u1"]), 1 / 4);
  // Three declared bits round up to four for the same reason three bytes round up
  // to four: a voxel's position inside its word has to be a shift.
  assert.equal(sparseBrickLaneStrideBytes(["u1", "u1", "u1"]), 1 / 2);
  assert.equal(sparseBrickLaneStrideBytes(["u1", "u1", "u1", "u1"]), 1 / 2);
  assert.equal(sparseBrickLaneVoxelsPerWord(1 / 8), 32);
  assert.equal(sparseBrickLaneVoxelsPerWord(1 / 4), 16);
  // The second prerequisite: a global palette slot is a `u16`.
  assert.equal(sparseBrickLaneStrideBytes(["u16"]), 2);
  assert.equal(sparseBrickLaneStrideBytes(["u16", "u16"]), 4);
  // And nothing below a bit is a lane.
  assert.throws(() => sparseBrickLaneStrideBytes([]), RangeError);
  // Every stride that predates sub-byte lanes is unchanged, or the arena moved.
  assert.equal(sparseBrickLaneStrideBytes(["f32", "f32", "f32", "f32"]), 16);
  assert.equal(sparseBrickLaneStrideBytes(["f16", "unorm8"]), 4);
  assert.equal(sparseBrickLaneStrideBytes(["snorm8", "unorm8"]), 2);
  assert.equal(sparseBrickLaneStrideBytes(["unorm8"]), 1);
});

test("the dense arm's arena is byte-identical to the one that shipped", () => {
  // The banded lanes are appended, never inserted, so a `dense` world's offsets
  // cannot move. This is the assertion that would catch the lane order drifting.
  const shipped = resolveSparseBrickPayloadLayout("dry", VOXELS, "f16-unorm8");
  const explicit = resolveSparseBrickPayloadLayout("dry", VOXELS, "f16-unorm8",
    { leafPayloadMode: "dense" });
  assert.equal(shipped.leafPayloadMode, "dense");
  assert.deepEqual(shipped.lanes, explicit.lanes);
  assert.equal(shipped.totalBytes, explicit.totalBytes);
  assert.equal(shipped.productBytes, shipped.totalBytes);
  assert.equal(shipped.bytesPerVoxel, 8);
  for (const name of ["sceneOccupancy", "sceneRecordMask", "sceneBandedHeader",
    "sceneBandedBlob", "sceneBandedRecords"] as const) {
    assert.equal(shipped.lanes[name].present, false, name);
    assert.equal(shipped.lanes[name].bytes, 0, name);
  }
});

test("the occupancy rung costs 64 bytes a leaf and saves nothing", () => {
  const dense = resolveSparseBrickPayloadLayout("dry", VOXELS, "f16-unorm8");
  const masked = resolveSparseBrickPayloadLayout("dry", VOXELS, "f16-unorm8",
    { leafPayloadMode: "occupancy" });
  assert.equal(masked.lanes.sceneOccupancy.present, true);
  assert.equal(masked.lanes.sceneOccupancy.bytes, VOXELS / 8);
  assert.equal(masked.lanes.sceneOccupancy.bytes / LEAVES, 64);
  assert.equal(masked.lanes.sceneRecordMask.present, false);
  // Both dense lanes survive: this rung is a bisection step, not a saving.
  assert.equal(masked.lanes.sceneMaterialOwners.bytes, dense.lanes.sceneMaterialOwners.bytes);
  assert.equal(masked.lanes.sceneGeometry.bytes, dense.lanes.sceneGeometry.bytes);
  assert.equal(masked.bytesPerVoxel, 8.125);
  // ~1.6 % on a 4 096-byte leaf, which is the price of being able to bisect a hash.
  assert.ok(masked.totalBytes > dense.totalBytes);
  assert.ok(masked.totalBytes < dense.totalBytes * 1.02, `${masked.totalBytes} vs ${dense.totalBytes}`);
});

test("the banded arm's fixed part is the layout's own 144 bytes a leaf", () => {
  const banded = resolveSparseBrickPayloadLayout("dry", VOXELS, "f16-unorm8",
    { leafPayloadMode: "banded" });
  const fixed = banded.lanes.sceneOccupancy.bytes
    + banded.lanes.sceneRecordMask.bytes
    + banded.lanes.sceneBandedHeader.bytes;
  assert.equal(fixed / LEAVES, SVO_BANDED_LEAF_FIXED_BYTES);
  // Still 144. The normal lane needs no header offset because it is *first* in the
  // blob at a fixed 256 words, which leaves the palette and the index derivable.
  assert.equal(fixed / LEAVES, 144);
  assert.equal(SPARSE_BRICK_BANDED_HEADER_WORDS * 4, 16);
  // The per-voxel owner word and the per-voxel geometry word are both *gone*, not
  // narrowed. Absent rather than zero-width, so a consumer that still indexes one
  // fails its own `present` check instead of reading the lane behind it.
  assert.equal(banded.lanes.sceneMaterialOwners.present, false);
  assert.equal(banded.lanes.sceneGeometry.present, false);
  // Records are suballocated from a shared arena at a measured fraction of voxels.
  assert.equal(banded.lanes.sceneBandedRecords.present, true);
  assert.equal(banded.lanes.sceneBandedRecords.strideBytes, 4);
  assert.equal(banded.bandedRecordCapacity,
    Math.ceil(VOXELS * SPARSE_BRICK_BANDED_RECORD_CAPACITY_FRACTION));
  assert.equal(banded.lanes.sceneBandedRecords.bytes, banded.bandedRecordCapacity * 4);
  assert.equal(banded.bandedBlobWords,
    SPARSE_BRICK_BANDED_ALLOCATOR_WORDS + LEAVES * SPARSE_BRICK_BANDED_BLOB_BYTES_PER_LEAF / 4);
  // 144 fixed + a quarter of 4 bytes of records + 1 152/512 of blob = 3.53 B a
  // voxel of *reserved* capacity against the dense arm's 8. The measured occupancy
  // is lower — 1 308 bytes a leaf, 2.55 B a voxel, on the hero at depth 0 — because
  // the record arena reserves for the crowded lighting study, not for this scene.
  assert.ok(banded.bytesPerVoxel < 3.6, `${banded.bytesPerVoxel} B/voxel`);
  assert.ok(banded.bytesPerVoxel < 0.5 * 8, "reserved capacity must still halve the dense arm");
  // The blob is where the normal lane lives, and it is the layout's largest term.
  // The lane is a *fixed* 2 bytes a voxel, so a budget below that would raise
  // `SPARSE_BRICK_BANDED_OVERFLOW.blob` on every leaf rather than on some of them.
  assert.ok(SPARSE_BRICK_BANDED_BLOB_BYTES_PER_LEAF >= SVO_BANDED_LEAF_VOXELS * 2,
    "the blob must hold a whole leaf's normal lane");
});

test("a retained dense lane is named, and the product arena is still reported", () => {
  // Retention is per lane because the two lanes are held by different consumers:
  // `sceneMaterialOwners` by the banded encoder itself (it interns each leaf's
  // palette from the dense words the rebuild pass just wrote), `sceneGeometry` by
  // the B2 readers. The figure that matters must therefore be reported separately,
  // and it has to be the *same resolver's* answer rather than a subtraction.
  const product = resolveSparseBrickPayloadLayout("dry", VOXELS, "f16-unorm8",
    { leafPayloadMode: "banded" });
  const shadow = resolveSparseBrickPayloadLayout("dry", VOXELS, "f16-unorm8",
    { leafPayloadMode: "banded", retainDenseLanes: SPARSE_BRICK_BANDED_PRODUCER_DENSE_LANES });
  assert.equal(shadow.lanes.sceneMaterialOwners.present, true);
  assert.equal(shadow.lanes.sceneGeometry.present, true);
  assert.equal(shadow.productBytes, product.totalBytes);
  assert.ok(shadow.totalBytes > product.totalBytes * 3, "the shadow shape costs the whole saving");
  // Naming one lane keeps one lane. That is the shape a partial reader cutover
  // lands in, and a boolean could not express it.
  const geometryOnly = resolveSparseBrickPayloadLayout("dry", VOXELS, "f16-unorm8",
    { leafPayloadMode: "banded", retainDenseLanes: ["sceneGeometry"] });
  assert.equal(geometryOnly.lanes.sceneGeometry.present, true);
  assert.equal(geometryOnly.lanes.sceneMaterialOwners.present, false);
  assert.equal(geometryOnly.productBytes, product.totalBytes);
  assert.ok(geometryOnly.totalBytes < shadow.totalBytes);
});

test("only a dry world may band, and only at the record width the figure is quoted at", () => {
  for (const profile of ["full"] as const) {
    for (const mode of ["occupancy", "banded"] as const) {
      assert.throws(() => resolveSparseBrickPayloadLayout(profile, VOXELS, "f32x2",
        { leafPayloadMode: mode }), RangeError, `${profile}/${mode}`);
    }
  }
  // `f32x2` is the geometry axis's rollback and composes with `dense`, but the
  // banded arm takes exactly one record width, so it is refused here.
  assert.throws(() => resolveSparseBrickPayloadLayout("dry", VOXELS, "f32x2",
    { leafPayloadMode: "banded" }), RangeError);
  assert.doesNotThrow(() => resolveSparseBrickPayloadLayout("dry", VOXELS, "f32x2",
    { leafPayloadMode: "occupancy" }));
  assert.throws(() => resolveSparseBrickPayloadLayout("dry", VOXELS, "f16-unorm8",
    { leafPayloadMode: "packed" as never }), RangeError);
});

test("the lever defaults to the arm that shipped", () => {
  // The rule that cost three agents time when it was broken: a lever whose default
  // is an arm nobody has cleared turns one recorded baseline into a measurement of
  // the arm under test. `banded` is cleared for *fidelity* and not for storage, and
  // this lever selects storage.
  assert.equal(octreeLiveSceneLeafPayloadMode({}), "dense");
  assert.equal(octreeLiveSceneLeafPayloadMode({ FLUID_SVO_LEAF_PAYLOAD: "" }), "dense");
  for (const mode of SPARSE_BRICK_LEAF_PAYLOAD_MODES) {
    assert.equal(octreeLiveSceneLeafPayloadMode({ FLUID_SVO_LEAF_PAYLOAD: mode }), mode);
  }
  assert.throws(() => octreeLiveSceneLeafPayloadMode({ FLUID_SVO_LEAF_PAYLOAD: "band" }), RangeError);
});

test("the codec addresses through functions, because the bases come from a uniform", () => {
  const options = {
    occupancyBase: "params.banded.x", recordMaskBase: "params.banded.y",
    headerBase: "params.banded.z", blobBase: "params.banded.w",
    recordsBase: "params.bandedCapacities.w",
    load: (index: string) => `atomicLoad(&payload[${index}])`,
  } as const;
  const mask = sparseBrickBandedLeafCodecWGSL({ ...options, mode: "occupancy" });
  const full = sparseBrickBandedLeafCodecWGSL({ ...options, mode: "banded" });
  // WGSL refuses to reference `var<uniform>` at module scope, so a `const` base does
  // not compile — and the failure is at module creation, not at a call site.
  assert.ok(!/const BANDED_\w*BASE/.test(mask), "a base must not be a module-scope const");
  assert.ok(!/const BANDED_\w*BASE/.test(full), "a base must not be a module-scope const");
  assert.match(mask, /fn bandedOccupancyBase\(\)->u32\{return params\.banded\.x;\}/);
  // The mask rung declares the predicate and nothing else: a shader that cannot
  // reach a header must not compile a function that reads one.
  for (const symbol of ["bandedHeader", "bandedIdentity", "bandedRecorded", "bandedPrefixCount"]) {
    assert.ok(!mask.includes(`fn ${symbol}`), `occupancy rung must not declare ${symbol}`);
    assert.ok(full.includes(`fn ${symbol}`), `banded rung must declare ${symbol}`);
  }
  // One definition of each, or the module does not compile.
  for (const symbol of ["bandedOccupied", "bandedIdentity", "bandedPrefixCount", "bandedRecordSlot"]) {
    assert.equal((full.match(new RegExp(`fn ${symbol}\\(`, "g")) ?? []).length, 1, symbol);
  }
  // An index entry never straddles a word, which is what lets the reader unpack with
  // one shift: the widths are the reference encoder's own ladder.
  assert.deepEqual([...SVO_BANDED_LEAF_INDEX_BITS], [0, 1, 2, 4, 8]);
  for (const bits of SVO_BANDED_LEAF_INDEX_BITS) {
    if (bits === 0) continue;
    for (let rank = 0; rank < SVO_BANDED_LEAF_VOXELS; rank += 1) {
      assert.ok((rank * bits) % 32 + bits <= 32, `${bits} bits at rank ${rank} straddles a word`);
    }
  }
  // Out of range reads as air rather than as a plausible identity, so an encoder bug
  // is a visible hole instead of the wrong material.
  assert.match(full, /if\(entry>=palette\)\{return BANDED_NO_MATERIAL_OWNER;\}/);
  // The normal lane is first in the blob at a fixed width, which is what lets the
  // header stay at four words: the palette and the index both have derivable
  // starts, and the normal's start is the blob itself.
  assert.match(full, /const BANDED_NORMAL_WORDS:u32=BANDED_VOXELS_PER_LEAF\/2u;/);
  assert.equal((full.match(/fn bandedHalfWord\(/g) ?? []).length, 1);
  assert.equal(SPARSE_BRICK_BANDED_HEADER_WORDS, 4);
  assert.ok(!full.includes("bandedHeaderNormalBase"), "the normal lane needs no header offset");
});

test("the DDA's solidity question is one occupancy bit, not an identity word", () => {
  // The point of the whole layout, and the thing a storage-only landing would
  // miss. Under `dense` a marcher pays a strided 4-byte identity load for every
  // cell it *rejects*; under the two mask arms it pays one bit of a word it shares
  // with 31 neighbours, and resolves the identity only for the cell that answers
  // yes — at most one per ray, because the walk returns on it.
  const identityCodec = (mode: SparseBrickLeafPayloadMode) => sparseBrickSceneIdentityCodecWGSL({
    mode, materialOwnerBase: "dry.payloadLanes1.y", load: (index) => `scenePayload[${index}]`,
  });
  assert.ok(!identityCodec("dense").includes("sceneIdentitySolidAt"),
    "the dense arm has no mask to ask, so it must not offer the predicate");
  for (const mode of ["occupancy", "banded"] as const) {
    // It takes a voxel and *not* a `SceneIdentitySource`. That signature is the
    // finding: a source hoisted out of the walk costs header loads on every leaf
    // the ray crosses and saves nothing on the one cell that hits, which measured
    // +4.3 % against `dense` before the predicate stopped needing it.
    assert.match(identityCodec(mode),
      /fn sceneIdentitySolidAt\(voxel:u32\)->bool\{return bandedOccupied\(voxel\);\}/, mode);
  }
  // And the primary actually calls it. `dense` keeps the shipped expression, so
  // the arm every hash baseline was recorded against is untouched.
  const shaderFor = (mode: SparseBrickLeafPayloadMode) => {
    const previous = process.env.FLUID_SVO_LEAF_PAYLOAD;
    process.env.FLUID_SVO_LEAF_PAYLOAD = mode;
    try {
      return createSvoDrySceneFragmentWGSL(1, "canonical");
    } finally {
      if (previous === undefined) delete process.env.FLUID_SVO_LEAF_PAYLOAD;
      else process.env.FLUID_SVO_LEAF_PAYLOAD = previous;
    }
  };
  const dense = shaderFor("dense");
  assert.ok(dense.includes(
    "let identity=sceneIdentityOf(identitySource,payloadIndex);"
    + "if(sceneIdentitySolid(identity)){cellSolid=true;cellIdentity=identity;}"),
  "the dense primary keeps the identity-word test it shipped with");
  assert.ok(!dense.includes("sceneIdentitySolidAt"), "and never reaches for a mask it has not got");
  for (const mode of ["occupancy", "banded"] as const) {
    const shader = shaderFor(mode);
    assert.ok(shader.includes(
      "if(sceneIdentitySolidAt(payloadIndex)){let identity=sceneIdentityAt(payloadIndex);"
      + "cellSolid=true;cellIdentity=identity;}"),
    `the ${mode} primary must reject a cell without loading its identity`);
    // And it must not hoist a source it no longer needs. `traceLeafPayload`'s body
    // touching identity storage *outside* the gate is the +4.3 % regression this
    // arrangement replaced.
    const loop = shader.slice(shader.indexOf("fn traceLeafPayload("));
    const body = loop.slice(0, loop.indexOf("\n}"));
    assert.ok(!body.includes("sceneIdentitySourceAt("), `${mode} must not hoist an identity source`);
    assert.equal((body.match(/sceneIdentityAt\(/g) ?? []).length, 1, mode);
  }
});

test("the payload binding is atomic whenever any lane shares a word", () => {
  // The third prerequisite. Atomicity was keyed on `snorm8-unorm8` alone, which was
  // sound only while that format was the only sharer; an occupancy mask puts 32
  // voxels in one word at *every* geometry width, and 31 of them are other
  // invocations of the same dispatch.
  for (const format of ["f32x2", "f16-unorm8"] as const) {
    const dense = sparseSceneProxyVoxelizationShaderFor("dry", format, "dense");
    assert.match(dense, /var<storage, read_write> payload: array<u32>;/, format);
    assert.ok(!dense.includes("atomicStore(&payload"), format);
    const masked = sparseSceneProxyVoxelizationShaderFor("dry", format, "occupancy");
    assert.match(masked, /var<storage, read_write> payload: array<atomic<u32>>;/, format);
    // Clear-then-set with bit-disjoint atomics: idempotent across rebuilds, and it
    // cannot drop the 31 neighbours a read-modify-write would.
    assert.match(masked, /atomicAnd\(&payload\[occupancyWord\],~bandedVoxelBit\(output\)\);/);
    assert.match(masked, /if\(primitiveFraction>0\.0\)\{atomicOr\(&payload\[occupancyWord\],bandedVoxelBit\(output\)\);\}/);
  }
  // The shipped expansions are untouched, to the character.
  assert.equal(sparseSceneProxyVoxelizationShaderFor("dry", "f16-unorm8"),
    sparseSceneProxyVoxelizationShaderFor("dry", "f16-unorm8", "dense"));
  assert.equal(sparseSceneProxyVoxelizationShaderFor("full"),
    sparseSceneProxyVoxelizationShaderFor("full", "f32x2", "banded"),
    "a full world can never band, so its shader must ignore the mode entirely");
});

test("the predicate move lands at the one site whose answer reaches the frame", () => {
  // `finalizeDirtyBricks` summarises a leaf's occupancy for the macro-HDDA tier, and
  // the primary reads that summary. Moving the predicate here is what makes a frame
  // hash a proof that the occupancy bit and the `material != 0` test it replaces are
  // the same predicate over the same data.
  const dense = sparseSceneProxyVoxelizationShaderFor("dry", "f16-unorm8", "dense");
  assert.match(dense, /let sceneMaterial=atomicLoad\(&payload\[sceneMaterialOffset\(\)\+voxelOffset\+localIndex\]\)&0xffffu;|let sceneMaterial=payload\[sceneMaterialOffset\(\)\+voxelOffset\+localIndex\]&0xffffu;/);
  for (const mode of ["occupancy", "banded"] as const) {
    const moved = sparseSceneProxyVoxelizationShaderFor("dry", "f16-unorm8", mode);
    assert.match(moved, /let sceneMaterial=select\(0u,1u,bandedOccupied\(voxelOffset\+localIndex\)\);/, mode);
  }
});

test("the banded encoder owns a whole leaf and reduces in workgroup storage", () => {
  const banded = sparseSceneProxyVoxelizationShaderFor("dry", "f16-unorm8", "banded");
  assert.equal((banded.match(/fn encodeBandedLeaves/g) ?? []).length, 1);
  // One workgroup a leaf: the band's stencil dilation, the leaf palette, the index
  // ranks and the record ranks are all whole-leaf reductions, and a leaf split
  // across two workgroups can express none of them.
  assert.match(banded, /@workgroup_size\(256\)\nfn encodeBandedLeaves/);
  // Contiguous after the other three dispatch triples, because all four are copied
  // to the argument buffer in one `copyBufferToBuffer` whose offsets are differences
  // from `binDispatch`.
  assert.match(banded, /writeDispatch\(stateOffset\(\)\+17u,chunk,1u\);/);
  // No early return before a barrier. WGSL cannot prove a storage load is uniform,
  // so a `return` on the chunk bound makes every later barrier non-uniform control
  // flow and the module will not compile — the flag is load-bearing, not a style.
  const body = banded.slice(banded.indexOf("fn encodeBandedLeaves"));
  const guarded = body.slice(0, body.indexOf("workgroupBarrier"));
  assert.ok(!guarded.includes("return;"), "the encoder must not return before a barrier");
  assert.match(banded, /let encoding=dirtyIndex<chunkEnd\(\)&&controlLoad\(11u\)==8u;/);
  // The masks are stored as whole words by the workgroup that owns them, so the mask
  // writes need no atomic ordering of their own.
  assert.match(banded, /atomicStore\(&payload\[params\.banded\.x\+maskWord\],atomicLoad\(&ldsOcc\[lane\]\)\);/);
  // The allocator resets only when a revision rebuilds every leaf, because only then
  // is every allocation it holds about to be reissued.
  assert.match(banded, /if\(cursor==0u&&dirtyCount>=controlLoad\(1u\)\)\{/);
  // And the dense arm compiles none of it.
  assert.ok(!sparseSceneProxyVoxelizationShaderFor("dry", "f16-unorm8", "dense")
    .includes("encodeBandedLeaves"));
  assert.ok(!sparseSceneProxyVoxelizationShaderFor("dry", "f16-unorm8", "occupancy")
    .includes("encodeBandedLeaves"));
});
