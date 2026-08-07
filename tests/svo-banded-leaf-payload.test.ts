import test from "node:test";
import assert from "node:assert/strict";
import { SPARSE_BRICK_SCENE_DISTANCE_BAND_RADII, SPARSE_BRICK_NO_OWNER } from "../lib/sparse-brick-octree";
import { OCTREE_LIVE_SCENE_PRIMITIVE_CAPACITY } from "../lib/webgpu-octree-sparse-bricks";
import { liveSvoDerivedBuildWGSLFor, liveSvoRadianceFeedbackWGSLFor } from "../lib/webgpu-svo-live-derived-builder";
import {
  SVO_BANDED_DENSE_GEOMETRY_BYTES,
  SVO_BANDED_LEAF_FIXED_BYTES,
  SVO_BANDED_LEAF_SATURATION_RADII,
  SVO_BANDED_LEAF_VOXELS,
  SVO_BANDED_NO_MATERIAL_OWNER,
  SVO_BANDED_DEFAULT_RECORD_SET,
  SVO_IDENTITY_PALETTE_CAPACITY,
  SvoIdentityPalette,
  bandedLeafBytes,
  bandedLeafIndexBits,
  bandedLeafOccupied,
  bandedLeafPrefixCount,
  bandedLeafSizeClass,
  bandedLeafVoxel,
  encodeBandedLeaf,
  svoBandedReconstructionEnabled,
  svoSolidDirectOcclusionEnabled,
  svoSolidMarchOffsetFixEnabled,
  type SvoDenseVoxel,
} from "../lib/svo-banded-leaf-payload";

const CELL_SIZE_M = [0.00625, 0.00625, 0.00625] as const;
const CELL_RADIUS_M = 0.5 * Math.sqrt(3) * 0.00625;

/**
 * The lane's own precision.
 *
 * Records land in `Float32Array` because the payload lane is f32 — narrower
 * still under `f16-unorm8` — so "exactly" means exactly at f32, and comparing
 * the decode against a f64 reference would be asserting something the dense
 * layout does not deliver either.
 */
const asStored = (value: number) => Math.fround(value);

/** A packed lane word, in the form the voxeliser writes. */
const identity = (owner: number, material: number) => (((owner << 16) >>> 0) | material) >>> 0;

/**
 * A leaf voxelized the way `rebuildDirtyBrickPayload` does it: a signed distance
 * at the voxel centre and the planar coverage law's fraction from it.
 */
function leafFromField(
  field: (x: number, y: number, z: number) => number,
  materialOwner: (x: number, y: number, z: number) => number = () => identity(7, 2),
): SvoDenseVoxel[] {
  const voxels: SvoDenseVoxel[] = [];
  for (let index = 0; index < SVO_BANDED_LEAF_VOXELS; index += 1) {
    const x = index & 7, y = (index >> 3) & 7, z = index >> 6;
    const distance_m = field(x, y, z);
    const fraction = Math.max(0, Math.min(1, 0.5 - distance_m / (2 * CELL_RADIUS_M)));
    voxels.push({
      distance_m,
      fraction,
      materialOwner: fraction > 0 ? materialOwner(x, y, z) : SVO_BANDED_NO_MATERIAL_OWNER,
    });
  }
  return voxels;
}

/** A slab filling the lower half, so the leaf holds interior, band and exterior. */
const slab = (_x: number, y: number) => (y - 3.5) * CELL_RADIUS_M * 1.2;
/** A leaf entirely inside solid. */
const buried = () => -40 * CELL_RADIUS_M;
/** A leaf entirely in air. */
const empty = () => 40 * CELL_RADIUS_M;

test("a banded leaf reproduces every fraction and owner exactly", () => {
  for (const field of [slab, buried, empty]) {
    const palette = new SvoIdentityPalette();
    const dense = leafFromField(field);
    const leaf = encodeBandedLeaf(dense, CELL_RADIUS_M, palette, "stencil");
    assert.ok(leaf, "every measured leaf shape must band");
    for (let index = 0; index < SVO_BANDED_LEAF_VOXELS; index += 1) {
      const decoded = bandedLeafVoxel(leaf, index, palette);
      assert.equal(decoded.fraction, asStored(dense[index].fraction), `fraction at ${index}`);
      assert.equal(decoded.materialOwner, dense[index].materialOwner, `owner at ${index}`);
    }
  }
});

test("occupancy survives a voxel that carries no record", () => {
  const palette = new SvoIdentityPalette();
  const dense = leafFromField(buried);
  const leaf = encodeBandedLeaf(dense, CELL_RADIUS_M, palette);
  assert.ok(leaf);
  // The whole point: a buried leaf stores nothing per voxel and is solid anyway.
  assert.equal(leaf.recordDistance.length, 0);
  assert.equal(leaf.identityIndex.length, 0);
  assert.equal(leaf.indexBits, 0);
  for (let index = 0; index < SVO_BANDED_LEAF_VOXELS; index += 1) {
    assert.equal(bandedLeafOccupied(leaf, index), true, `voxel ${index} must stay solid`);
  }
});

test("the record set covers the normal stencil of every band voxel", () => {
  const palette = new SvoIdentityPalette();
  const dense = leafFromField(slab);
  const leaf = encodeBandedLeaf(dense, CELL_RADIUS_M, palette, "stencil");
  assert.ok(leaf);
  const stored = (index: number) =>
    bandedLeafVoxel(leaf, index, palette).distance_m === asStored(dense[index].distance_m);
  for (let index = 0; index < SVO_BANDED_LEAF_VOXELS; index += 1) {
    const { fraction } = dense[index];
    if (!(fraction > 0 && fraction < 1)) continue;
    const x = index & 7, y = (index >> 3) & 7, z = index >> 6;
    assert.ok(stored(index), `band voxel ${index} must store its own distance`);
    // `safeNormal` clamps `local +/- 1` inside the leaf; every sample it takes
    // at a band voxel has to be the exact field value or the normal moves.
    for (const [dx, dy, dz] of [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]] as const) {
      const nx = Math.min(7, Math.max(0, x + dx));
      const ny = Math.min(7, Math.max(0, y + dy));
      const nz = Math.min(7, Math.max(0, z + dz));
      assert.ok(stored(nx + ny * 8 + nz * 64), `stencil neighbour of ${index} must store its distance`);
    }
  }
});

test("the packed high half survives the palette round trip", () => {
  // SP21's finding, and the trap a u16 identity would have walked into: the lane
  // word is `(high << 16) | materialId`, and the high half is read at five
  // dry-path sites — it was an owner id, it is now a baked oct8 normal, and either
  // way truncating to the material half would silently discard every one of them.
  const palette = new SvoIdentityPalette();
  const owners = [0, 1, 2, 4095, 16_383, SPARSE_BRICK_NO_OWNER];
  const dense = leafFromField(buried, (x) => identity(owners[x % owners.length], 1 + (x % 3)));
  const leaf = encodeBandedLeaf(dense, CELL_RADIUS_M, palette);
  assert.ok(leaf);
  for (let index = 0; index < SVO_BANDED_LEAF_VOXELS; index += 1) {
    const word: number = bandedLeafVoxel(leaf, index, palette).materialOwner;
    assert.equal(word, dense[index].materialOwner, `whole word at ${index}`);
    assert.equal(word >>> 16, dense[index].materialOwner >>> 16, `high half at ${index}`);
  }
});

test("a distinct normal in every voxel costs a lane, not a palette", () => {
  // The regression the unified-voxel cutover left behind, pinned. The identity
  // word's high half is now a per-voxel baked normal, so interning whole words
  // gave a leaf as many palette entries as it has occupied voxels — measured 29.05
  // a leaf on the hero garden against 1.026 for the material half alone, with
  // 0.11 % of leaves past the 256-entry limit entirely. This leaf is that case at
  // its extreme: 512 distinct high halves, one material.
  const palette = new SvoIdentityPalette();
  const dense = leafFromField(buried, (x, y, z) => identity(1 + x + y * 8 + z * 64, 3));
  const leaf = encodeBandedLeaf(dense, CELL_RADIUS_M, palette);
  assert.ok(leaf, "a leaf with one material must band however many normals it holds");
  assert.equal(leaf.leafPalette.length, 1, "the palette interns materials, not identities");
  assert.equal(leaf.indexBits, 0);
  assert.equal(leaf.identityIndex.length, 0);
  // One u16 per *occupied* voxel, in occupancy-rank order.
  assert.equal(leaf.normals.length, SVO_BANDED_LEAF_VOXELS);
  for (let index = 0; index < SVO_BANDED_LEAF_VOXELS; index += 1) {
    assert.equal(bandedLeafVoxel(leaf, index, palette).materialOwner, dense[index].materialOwner,
      `whole word at ${index}`);
  }
});

test("the normal lane is indexed by voxel, and rank-compacting it was a loss", () => {
  // A half-empty leaf spends 2 bytes on every air voxel it holds, deliberately.
  // Compacting by occupancy rank saves 22 % of the lane on the hero garden and
  // puts a sixteen-word prefix popcount in front of every read; measured on
  // device, that cost `banded` +4.1 % against `dense` on the depth-1 frame. The
  // normal is read once per shaded pixel and written once per publication.
  const palette = new SvoIdentityPalette();
  const dense = leafFromField(slab, (x, y, z) => identity(1 + x + y * 8 + z * 64, 3));
  const leaf = encodeBandedLeaf(dense, CELL_RADIUS_M, palette, "stencil");
  assert.ok(leaf);
  let occupied = 0;
  for (let index = 0; index < SVO_BANDED_LEAF_VOXELS; index += 1) {
    if (dense[index].fraction > 0) occupied += 1;
  }
  assert.ok(occupied > 0 && occupied < SVO_BANDED_LEAF_VOXELS, `slab must be partly solid, got ${occupied}`);
  assert.equal(leaf.normals.length, SVO_BANDED_LEAF_VOXELS);
  for (let index = 0; index < SVO_BANDED_LEAF_VOXELS; index += 1) {
    const decoded = bandedLeafVoxel(leaf, index, palette);
    if (dense[index].fraction > 0) {
      assert.equal(decoded.materialOwner, dense[index].materialOwner, `word at ${index}`);
    } else {
      assert.equal(decoded.materialOwner, SVO_BANDED_NO_MATERIAL_OWNER, `air at ${index}`);
    }
  }
});

test("a single-identity leaf spends no bits on identity", () => {
  const palette = new SvoIdentityPalette();
  const leaf = encodeBandedLeaf(leafFromField(slab), CELL_RADIUS_M, palette, "stencil");
  assert.ok(leaf);
  assert.equal(leaf.leafPalette.length, 1);
  assert.equal(leaf.indexBits, 0);
  assert.equal(leaf.identityIndex.length, 0);
});

test("a leaf holding two solids indexes them at one bit a voxel", () => {
  const palette = new SvoIdentityPalette();
  const dense = leafFromField(buried, (x) => (x < 4 ? identity(7, 2) : identity(9, 5)));
  const leaf = encodeBandedLeaf(dense, CELL_RADIUS_M, palette);
  assert.ok(leaf);
  assert.equal(leaf.recordDistance.length, 0);
  assert.equal(leaf.leafPalette.length, 2);
  assert.equal(leaf.indexBits, 1);
  assert.equal(leaf.identityIndex.length, SVO_BANDED_LEAF_VOXELS / 8);
  for (let index = 0; index < SVO_BANDED_LEAF_VOXELS; index += 1) {
    assert.equal(bandedLeafVoxel(leaf, index, palette).materialOwner, dense[index].materialOwner);
  }
});

test("the global palette interns materials across leaves and never truncates", () => {
  const palette = new SvoIdentityPalette();
  // Two leaves of material 2 and one of material 5, with *different* high halves
  // throughout: the high half must not reach the global table at all now.
  encodeBandedLeaf(leafFromField(buried, () => identity(7, 2)), CELL_RADIUS_M, palette);
  encodeBandedLeaf(leafFromField(buried, (x) => identity(7 + x, 2)), CELL_RADIUS_M, palette);
  encodeBandedLeaf(leafFromField(buried, () => identity(9, 5)), CELL_RADIUS_M, palette);
  assert.equal(palette.size, 2, "the same material in two leaves is one global slot");
  assert.equal(palette.wordAt(0), 2);
  assert.equal(palette.wordAt(1), 5);
  // The capacity is the scene alphabet SP21 bounded, not a width this file chose.
  assert.equal(SVO_IDENTITY_PALETTE_CAPACITY, 2 + OCTREE_LIVE_SCENE_PRIMITIVE_CAPACITY);
  assert.ok(SVO_IDENTITY_PALETTE_CAPACITY <= 2 ** 15, "a global slot must fit fifteen bits");
});

test("a leaf needing more materials than the index can address escapes to dense", () => {
  const palette = new SvoIdentityPalette();
  const dense = leafFromField(buried, (x, y, z) => identity(1, 1 + x + y * 8 + z * 64));
  assert.equal(encodeBandedLeaf(dense, CELL_RADIUS_M, palette), undefined);
});

test("the reconstruction saturates at the scene distance band, not a sentinel", () => {
  const palette = new SvoIdentityPalette();
  const leaf = encodeBandedLeaf(leafFromField(buried), CELL_RADIUS_M, palette);
  assert.ok(leaf);
  // Sharing `SPARSE_BRICK_SCENE_DISTANCE_BAND_RADII` is what makes the
  // reconstruction bit-exact under `snorm8-unorm8`, where the lane saturates at
  // the same radius. A copy of the number here would let the two drift apart
  // silently, and the drift would show up as a normal, not as a failure.
  assert.equal(SVO_BANDED_LEAF_SATURATION_RADII, SPARSE_BRICK_SCENE_DISTANCE_BAND_RADII);
  assert.equal(leaf.saturation_m, SPARSE_BRICK_SCENE_DISTANCE_BAND_RADII * CELL_RADIUS_M);
  assert.equal(bandedLeafVoxel(leaf, 0, palette).distance_m, -leaf.saturation_m);
});

test("randomised fields keep fraction and owner exact and the stencil covered", () => {
  // Three hand-built shapes cannot exercise a leaf clipped by two solids at an
  // angle, which is what the hero's ground under a stone actually is.
  let state = 0x2b22a1;
  const random = () => {
    state = (state * 1664525 + 1013904223) >>> 0;
    return state / 0x100000000;
  };
  for (let trial = 0; trial < 64; trial += 1) {
    const palette = new SvoIdentityPalette();
    const centre = [random() * 8, random() * 8, random() * 8];
    const radius = 1 + random() * 6;
    const normal = [random() - 0.5, random() - 0.5, random() - 0.5];
    const offset = (random() - 0.5) * 8;
    const dense = leafFromField(
      (x, y, z) => Math.min(
        (Math.hypot(x - centre[0], y - centre[1], z - centre[2]) - radius) * CELL_RADIUS_M * 1.15,
        ((x - 3.5) * normal[0] + (y - 3.5) * normal[1] + (z - 3.5) * normal[2] - offset) * CELL_RADIUS_M * 1.15,
      ),
      (x, y) => identity(7 + (x < 4 ? 0 : 1) + (y < 4 ? 0 : 2), 2),
    );
    const leaf = encodeBandedLeaf(dense, CELL_RADIUS_M, palette, "stencil");
    if (!leaf) continue; // A dense escape is a legal answer, not a failure.
    for (let index = 0; index < SVO_BANDED_LEAF_VOXELS; index += 1) {
      const decoded = bandedLeafVoxel(leaf, index, palette);
      assert.equal(decoded.fraction, asStored(dense[index].fraction), `trial ${trial} fraction ${index}`);
      assert.equal(decoded.materialOwner, dense[index].materialOwner, `trial ${trial} owner ${index}`);
      assert.equal(bandedLeafOccupied(leaf, index), dense[index].fraction > 0, `trial ${trial} occupancy ${index}`);
      const { fraction } = dense[index];
      if (!(fraction > 0 && fraction < 1)) continue;
      const x = index & 7, y = (index >> 3) & 7, z = index >> 6;
      for (const [dx, dy, dz] of [[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]] as const) {
        const nx = Math.min(7, Math.max(0, x + dx));
        const ny = Math.min(7, Math.max(0, y + dy));
        const nz = Math.min(7, Math.max(0, z + dz));
        const at = nx + ny * 8 + nz * 64;
        assert.equal(bandedLeafVoxel(leaf, at, palette).distance_m, asStored(dense[at].distance_m),
          `trial ${trial} stencil sample ${at} of band voxel ${index}`);
      }
    }
  }
});

test("prefix popcount indexes records in mask order", () => {
  const palette = new SvoIdentityPalette();
  const leaf = encodeBandedLeaf(leafFromField(slab), CELL_RADIUS_M, palette, "stencil");
  assert.ok(leaf);
  let expected = 0;
  for (let index = 0; index < SVO_BANDED_LEAF_VOXELS; index += 1) {
    assert.equal(bandedLeafPrefixCount(leaf.records, index), expected);
    if (((leaf.records[index >> 5] >>> (index & 31)) & 1) === 1) expected += 1;
  }
  assert.equal(expected, leaf.recordDistance.length);
});

test("index widths and size classes round up and never exceed the leaf", () => {
  assert.equal(bandedLeafIndexBits(0), 0);
  assert.equal(bandedLeafIndexBits(1), 0);
  assert.equal(bandedLeafIndexBits(2), 1);
  assert.equal(bandedLeafIndexBits(3), 2);
  assert.equal(bandedLeafIndexBits(45), 8);
  assert.equal(bandedLeafIndexBits(257), Number.POSITIVE_INFINITY);
  assert.equal(bandedLeafSizeClass(0), 0);
  assert.equal(bandedLeafSizeClass(1), 8);
  assert.equal(bandedLeafSizeClass(97), 128);
  assert.equal(bandedLeafSizeClass(512), 512);
  assert.throws(() => bandedLeafSizeClass(513), RangeError);
});

test("a buried leaf costs the fixed part, one palette slot and its normals", () => {
  const palette = new SvoIdentityPalette();
  const leaf = encodeBandedLeaf(leafFromField(buried), CELL_RADIUS_M, palette);
  assert.ok(leaf);
  // No record, no index, one palette entry — and 512 normals, because every
  // occupied voxel carries one and the record set cannot bound the first-hit
  // population. That last term is 1 024 of the 1 174 bytes, which is the shape of
  // the whole layout: see `bandedLeafBytes`.
  assert.equal(bandedLeafBytes(leaf, SVO_BANDED_DENSE_GEOMETRY_BYTES),
    SVO_BANDED_LEAF_FIXED_BYTES + 2 + SVO_BANDED_LEAF_VOXELS * 2);
  assert.equal(leaf.normals.length, SVO_BANDED_LEAF_VOXELS);
  // 1 174 bytes against 6 144 is still 5.2x on the leaf that stores least, and the
  // geometry width no longer moves it at all — the normals dominate.
  assert.ok(bandedLeafBytes(leaf, 2) * 5 < SVO_BANDED_LEAF_VOXELS * 12);
});

test("the radius record set stores everything the snorm8 lane would not saturate", () => {
  // SP20's finding is that a flattened central difference over the deep interior
  // is what cost 28.5 % of pixels, so the wider rule stores the distance exactly
  // where that lane still carries a gradient — and reconstructs only what it
  // would itself have clamped.
  const palette = new SvoIdentityPalette();
  const dense = leafFromField(slab);
  const leaf = encodeBandedLeaf(dense, CELL_RADIUS_M, palette,
    { radii: SPARSE_BRICK_SCENE_DISTANCE_BAND_RADII });
  assert.ok(leaf);
  const radius = SPARSE_BRICK_SCENE_DISTANCE_BAND_RADII * CELL_RADIUS_M;
  for (let index = 0; index < SVO_BANDED_LEAF_VOXELS; index += 1) {
    const exact = bandedLeafVoxel(leaf, index, palette).distance_m === asStored(dense[index].distance_m);
    if (Math.abs(dense[index].distance_m) <= radius) {
      assert.ok(exact, `voxel ${index} inside the band must store its distance`);
    }
  }
});

test("the default rule is the one the device chose, and it stores less", () => {
  // The device measurement inverted the drafted expectation. `stencil` is exact
  // on the band *and* cheaper than every radius rule, and no radius rescues the
  // interior — 2.3x the records moves occupied flips from 86 % to 81 %. So the
  // default is the narrow rule, on evidence, not the wide one on caution.
  assert.equal(SVO_BANDED_DEFAULT_RECORD_SET, "stencil");
  const byDefault = encodeBandedLeaf(leafFromField(slab), CELL_RADIUS_M, new SvoIdentityPalette());
  const byRadius = encodeBandedLeaf(leafFromField(slab), CELL_RADIUS_M, new SvoIdentityPalette(),
    { radii: SPARSE_BRICK_SCENE_DISTANCE_BAND_RADII });
  assert.ok(byDefault && byRadius);
  assert.ok(byDefault.recordDistance.length < byRadius.recordDistance.length,
    "the chosen rule must be the cheaper one");
});

test("the stencil rule is exact for every band normal and the 1R rule is not", () => {
  // The mechanism behind the table in `SVO_BANDED_DEFAULT_RECORD_SET`: a rule
  // keyed on distance does not respect the stencil `safeNormal` reads, so 1R
  // stores less than stencil and still misses samples the normal needs.
  const dense = leafFromField(slab);
  const stencilPalette = new SvoIdentityPalette();
  const radiusPalette = new SvoIdentityPalette();
  const byStencil = encodeBandedLeaf(dense, CELL_RADIUS_M, stencilPalette, "stencil");
  const byRadius = encodeBandedLeaf(dense, CELL_RADIUS_M, radiusPalette, { radii: 1 });
  assert.ok(byStencil && byRadius);
  let radiusMisses = 0;
  for (let index = 0; index < SVO_BANDED_LEAF_VOXELS; index += 1) {
    const { fraction } = dense[index];
    if (!(fraction > 0 && fraction < 1)) continue;
    const x = index & 7, y = (index >> 3) & 7, z = index >> 6;
    for (const [dx, dy, dz] of [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]] as const) {
      const at = Math.min(7, Math.max(0, x + dx)) + Math.min(7, Math.max(0, y + dy)) * 8
        + Math.min(7, Math.max(0, z + dz)) * 64;
      const want = asStored(dense[at].distance_m);
      assert.equal(bandedLeafVoxel(byStencil, at, stencilPalette).distance_m, want,
        `stencil must store neighbour ${at}`);
      if (bandedLeafVoxel(byRadius, at, radiusPalette).distance_m !== want) radiusMisses += 1;
    }
  }
  assert.ok(radiusMisses > 0, "a 1R rule is expected to miss stencil samples the band needs");
});

test("the shipped arms are the measured ones, with a one-word rollback each", () => {
  // The offset fix ships: mean 1.16/255, every oracle clean, and it is a
  // correctness bug fixed at source that stands on its own.
  assert.equal(svoSolidMarchOffsetFixEnabled({}), true);
  // The reconstruction is *off*, and the reason is sequencing rather than doubt.
  // Its fidelity question is settled favourably — mean 0.54/255, zero pixels over
  // 16/255, less visible than the f16 geometry lane already shipping at 11/255 —
  // but it changes only what `solidDistance` returns, so until the arena, masks
  // and palette land (B1/B2) it carries the diff and none of the ~11.6x saving.
  // Paying a fidelity cost for a projected benefit is the wrong order. Flip this
  // to `true` in the same change that lands B2, so the frame moves once against a
  // measured arena rather than a forecast one.
  assert.equal(svoBandedReconstructionEnabled({}), false);
  // The direct-occlusion fix is the second mechanism of that same leak, and it
  // stays off until its own hash, diff and PNG are blessed. It reproduces most of
  // the skip's 22.22/255 by construction, so shipping it silently would relight
  // the hero.
  assert.equal(svoSolidDirectOcclusionEnabled({}), false);
  for (const off of ["0", "off"]) {
    assert.equal(svoSolidMarchOffsetFixEnabled({ FLUID_SVO_SOLID_MARCH_OFFSET: off }), false);
    assert.equal(svoBandedReconstructionEnabled({ FLUID_SVO_BANDED_RECONSTRUCTION: off }), false);
  }
  for (const on of ["1", "on"]) {
    assert.equal(svoBandedReconstructionEnabled({ FLUID_SVO_BANDED_RECONSTRUCTION: on }), true);
  }
  assert.throws(() => svoBandedReconstructionEnabled({ FLUID_SVO_BANDED_RECONSTRUCTION: "yes" }), RangeError);
  assert.throws(() => svoSolidMarchOffsetFixEnabled({ FLUID_SVO_SOLID_MARCH_OFFSET: "yes" }), RangeError);
});

test("the off arm leaves the build shader's distance read untouched", () => {
  // The gate is only a gate if its off arm is the shipped shader. Anything the
  // banded arm adds must be absent here, or a "baseline" hash is measuring the
  // arm under test.
  const off = liveSvoDerivedBuildWGSLFor("dry", "rgba16float", "rg8unorm", "f16-unorm8");
  const on = liveSvoDerivedBuildWGSLFor("dry", "rgba16float", "rg8unorm", "f16-unorm8", CELL_SIZE_M);
  for (const symbol of ["bandedRecordAt", "bandedSaturation", "BANDED_FINEST_CELL", "storedSolidDistance"]) {
    assert.ok(!off.includes(symbol), `off arm must not declare ${symbol}`);
    assert.ok(on.includes(symbol), `on arm must declare ${symbol}`);
  }
  assert.ok(off.includes("fn solidDistance(leafIndex:u32,local:vec3u)->f32{let voxel=leafVoxel(leafIndex,local);"),
    "the off arm keeps the shipped one-line accessor");
  // Exactly one definition either way, or the module does not compile.
  assert.equal((off.match(/fn solidDistance/g) ?? []).length, 1);
  assert.equal((on.match(/fn solidDistance/g) ?? []).length, 1);
});

test("the banded arm reconstructs at the same radius the encoder saturates at", () => {
  // The shader and `encodeBandedLeaf` must agree about the saturation or the gate
  // is testing a different reconstruction than the layout would ship.
  const on = liveSvoDerivedBuildWGSLFor("dry", "rgba16float", "rg8unorm", "f16-unorm8", CELL_SIZE_M);
  assert.ok(on.includes(`${SPARSE_BRICK_SCENE_DISTANCE_BAND_RADII}.0*0.5*length(BANDED_FINEST_CELL*f32(bandedLeafScale(leafIndex)))`),
    "the shader must saturate at SPARSE_BRICK_SCENE_DISTANCE_BAND_RADII cell radii of the leaf's own voxel");
  assert.equal(SVO_BANDED_LEAF_SATURATION_RADII, SPARSE_BRICK_SCENE_DISTANCE_BAND_RADII);
  // And the record predicate is the encoder's clamped 6-neighbourhood, which at a
  // face folds the outward neighbour onto the voxel itself.
  assert.ok(on.includes("let lo=max(local,vec3u(1u))-1u;let hi=min(local+1u,vec3u(control[11]-1u));"),
    "the shader must clamp the neighbourhood exactly as safeNormal does");
});

test("the solid direct-occlusion lever defaults to the shipped arm", () => {
  assert.equal(svoSolidDirectOcclusionEnabled({}), false);
  assert.equal(svoSolidDirectOcclusionEnabled({ FLUID_SVO_SOLID_DIRECT_OCCLUSION: "" }), false);
  assert.equal(svoSolidDirectOcclusionEnabled({ FLUID_SVO_SOLID_DIRECT_OCCLUSION: "0" }), false);
  assert.equal(svoSolidDirectOcclusionEnabled({ FLUID_SVO_SOLID_DIRECT_OCCLUSION: "off" }), false);
  for (const on of ["1", "on"]) {
    assert.equal(svoSolidDirectOcclusionEnabled({ FLUID_SVO_SOLID_DIRECT_OCCLUSION: on }), true);
  }
  assert.throws(() => svoSolidDirectOcclusionEnabled({ FLUID_SVO_SOLID_DIRECT_OCCLUSION: "yes" }), RangeError);
});

test("the direct-occlusion off arm is the shipped feedback shader", () => {
  // The gate is only a gate if its off arm is what ships. `receiverSolid` appearing
  // in a "baseline" shader means the baseline hash measured the arm under test.
  for (const offsetFix of [false, true]) {
    const off = liveSvoRadianceFeedbackWGSLFor("dry", "rgba16float", "f16-unorm8", undefined, offsetFix);
    const explicit = liveSvoRadianceFeedbackWGSLFor(
      "dry", "rgba16float", "f16-unorm8", undefined, offsetFix, false);
    assert.equal(off, explicit, "the default must be the off arm");
    assert.ok(!off.includes("receiverSolid"), "the off arm must not mention the predicate");
    assert.ok(off.includes("fn directVisibility(originCells:vec3f,towardLight:vec3f,maximumDistance_m:f32)->f32{let cellScale="),
      "the off arm keeps the shipped signature and takes its first step immediately");
  }
});

test("a fully solid receiver is opaque to direct light before any step", () => {
  const on = liveSvoRadianceFeedbackWGSLFor("dry", "rgba16float", "f16-unorm8", undefined, true, true);
  // The guard is the *first* statement, ahead of `cellScale` and every sample: the
  // whole point is that no step length can answer this question, so none is taken.
  assert.ok(on.includes("fn directVisibility(originCells:vec3f,towardLight:vec3f,maximumDistance_m:f32,receiverSolid:bool)->f32{if(receiverSolid){return 0.0;}let cellScale="),
    "the on arm must return zero transmittance before computing a step");
  // Threaded from the seed, and on the *same* fraction test the march-offset rule
  // uses. Two predicates would be two definitions of solid.
  assert.ok(on.includes("incident+=directIncident(worldPosition,originCells,normal,bandedFractionAt(leaf,sampleLocal)>=1.0);"),
    "the receiver's own fraction decides, at the seed");
  assert.ok(on.includes("directVisibility(originCells,toward,maximumDistance_m,receiverSolid)>0.0"),
    "every light in the loop sees the same receiver classification");
  assert.equal((on.match(/receiverSolid/g) ?? []).length, 4, "declared, guarded, forwarded, consumed — once each");
  // And the step schedule itself is untouched, so a band voxel's march — the one
  // the receiver-width scaling exists to protect — cannot pick up acne from this.
  const off = liveSvoRadianceFeedbackWGSLFor("dry", "rgba16float", "f16-unorm8", undefined, true);
  for (const schedule of ["var distance_m=cellScale*2.5;", "distance_m=cellScale*6.0;", "distance_m=cellScale*18.0;", "distance_m=cellScale*54.0;"]) {
    assert.ok(off.includes(schedule) && on.includes(schedule), `both arms keep ${schedule}`);
  }
});

test("the direct-occlusion arm brings its own predicate into scope", () => {
  // `bandedFractionAt` is declared by the shared helper block, which used to be
  // gated on banding, the skip or the march offset. Without the widening the on
  // arm calls an undeclared function and the module fails to compile.
  const alone = liveSvoRadianceFeedbackWGSLFor("dry", "rgba16float", "f16-unorm8", undefined, false, true);
  assert.equal((alone.match(/fn bandedFractionAt/g) ?? []).length, 1, "declared exactly once");
  assert.ok(alone.includes("receiverSolid"), "and used");
  // The offset rule is off in that arm, so its own predicate must be absent.
  assert.ok(alone.includes("normal*(1.5*f32(scale))"), "the march offset stays shipped when its lever is off");
});
