import assert from "node:assert/strict";
import test from "node:test";
import {
  effectiveSvoScreenSpaceThresholdPixels,
  projectedSvoNodeFootprintPixels,
  selectSvoLodCellStride,
  SVO_SCREEN_SPACE_TERMINATION_CONTRACT,
  svoLodBrickSubLevels,
  svoLodCellStrideForLevel,
  svoLodDescentWGSL,
} from "../lib/svo-screen-space-termination";
import { createSvoDrySceneFragmentWGSL } from "../lib/webgpu-svo-dry-scene";

/**
 * The hero garden's lattice after the 25 mm -> 6.25 mm refinement: an 8-cell
 * brick is 50 mm, so a brick is exactly the geometry that made an intermediate
 * rung necessary rather than optional.
 */
const CELL_M = 0.00625;
const BRICK_CELLS = 8;
const BRICK_M = CELL_M * BRICK_CELLS;
const brickAt = (distance_m: number) => ({
  minimum: [-BRICK_M / 2, -BRICK_M / 2, distance_m - BRICK_M / 2] as [number, number, number],
  maximum: [BRICK_M / 2, BRICK_M / 2, distance_m + BRICK_M / 2] as [number, number, number],
});
// The hero is a 50 mm lens; the contract's 0.72 default is a much wider one and
// puts every rung four times nearer, which is the wrong scene to reason about.
const HERO_TAN_HALF_FOV = 0.24;
const options = (overrides: Partial<Parameters<typeof selectSvoLodCellStride>[3]> = {}) => ({
  thresholdPixels: SVO_SCREEN_SPACE_TERMINATION_CONTRACT.defaultThresholdPixels,
  viewportHeightPixels: SVO_SCREEN_SPACE_TERMINATION_CONTRACT.referenceViewportHeightPixels,
  tanHalfVerticalFov: HERO_TAN_HALF_FOV,
  ...overrides,
});

test("an eight-cell brick offers three rungs below its leaf", () => {
  assert.equal(svoLodBrickSubLevels(8), 3);
  assert.equal(svoLodBrickSubLevels(4), 2);
  assert.throws(() => svoLodBrickSubLevels(16), RangeError);
});

test("fixed level walks the ladder from whole brick to individual cell", () => {
  const leaf = 5;
  // At or above the leaf's own level there is no brick left to subdivide.
  assert.equal(svoLodCellStrideForLevel(leaf, leaf, 8), 8);
  assert.equal(svoLodCellStrideForLevel(leaf, leaf - 3, 8), 8);
  assert.equal(svoLodCellStrideForLevel(leaf, leaf + 1, 8), 4);
  assert.equal(svoLodCellStrideForLevel(leaf, leaf + 2, 8), 2);
  assert.equal(svoLodCellStrideForLevel(leaf, leaf + 3, 8), 1);
  // Past the last sub-level the walk is already exact and stays exact, which is
  // what makes the panel's default of 21 a true no-op on any real scene.
  assert.equal(svoLodCellStrideForLevel(leaf, leaf + 4, 8), 1);
  assert.equal(svoLodCellStrideForLevel(leaf, 21, 8), 1);
});

test("a zero threshold is exact traversal and not nearly-exact", () => {
  for (const distance of [0.5, 4, 40, 400]) {
    assert.equal(selectSvoLodCellStride(brickAt(distance), [0, 0, 0], BRICK_CELLS,
      options({ thresholdPixels: 0 })), 1);
  }
});

test("the ladder engages at doubling distances and never skips a rung", () => {
  const stride = (distance_m: number) =>
    selectSvoLodCellStride(brickAt(distance_m), [0, 0, 0], BRICK_CELLS, options());
  // Near work stays exact; each coarser aggregate needs roughly twice the
  // distance of the one before it, because footprint falls as 1/d.
  assert.equal(stride(1), 1);
  const onsets = [2, 4, 8].map((stridePower) => {
    let distance = 0.5;
    while (distance < 512 && stride(distance) < stridePower) distance *= 1.01;
    return distance;
  });
  assert.ok(onsets[0] > 3 && onsets[0] < 12, `stride 2 onset ${onsets[0]} m`);
  for (const [index, onset] of onsets.slice(1).entries()) {
    const ratio = onset / onsets[index];
    assert.ok(Math.abs(ratio - 2) < 0.15, `rung ${index + 1} onset ratio ${ratio}`);
  }
  // Monotone: a brick can only ever coarsen as it recedes.
  let previous = 1;
  for (let distance = 0.5; distance < 400; distance *= 1.2) {
    const current = stride(distance);
    assert.ok(current >= previous, `stride fell from ${previous} to ${current} at ${distance} m`);
    previous = current;
  }
});

test("the chosen aggregate never exceeds the authored threshold on screen", () => {
  // This is the whole contract: the threshold is a fidelity budget, so what
  // gets drawn is never larger than the author agreed to tolerate. The
  // predicate it replaced measured one *cell* and then collapsed all eight,
  // which is exactly this invariant violated eight-fold.
  const view = options({ thresholdPixels: 4, viewportHeightPixels: 920 });
  const threshold = effectiveSvoScreenSpaceThresholdPixels(view.thresholdPixels, view.viewportHeightPixels);
  for (let distance = 0.5; distance < 600; distance *= 1.15) {
    const bounds = brickAt(distance);
    const stride = selectSvoLodCellStride(bounds, [0, 0, 0], BRICK_CELLS, view);
    if (stride === 1) continue;
    const half = (CELL_M * stride) / 2;
    const centre: [number, number, number] = [0, 0, distance - BRICK_M / 2 + half];
    const footprint = projectedSvoNodeFootprintPixels({
      minimum: centre.map((value) => value - half) as [number, number, number],
      maximum: centre.map((value) => value + half) as [number, number, number],
    }, [0, 0, 0], view);
    assert.ok(footprint <= threshold + 1e-9,
      `stride ${stride} at ${distance.toFixed(2)} m projects ${footprint.toFixed(3)} px over ${threshold}`);
  }
});

test("the ladder is angular: resolution and DPR cannot silently buy detail", () => {
  const bounds = brickAt(30);
  const base = selectSvoLodCellStride(bounds, [0, 0, 0], BRICK_CELLS, options());
  for (const viewportHeightPixels of [230, 460, 920, 2160]) {
    assert.equal(selectSvoLodCellStride(bounds, [0, 0, 0], BRICK_CELLS,
      options({ viewportHeightPixels })), base,
    `viewport ${viewportHeightPixels} changed the stride`);
  }
});

test("a camera inside the brick is always exact", () => {
  assert.equal(selectSvoLodCellStride(brickAt(0), [0, 0, 0], BRICK_CELLS, options({ thresholdPixels: 64 })), 1);
});

test("the WGSL twin declares what the dry scene calls, and only when compiled", () => {
  for (const name of ["svoLodCellStrideForLevel", "svoLodScreenSpaceCellStride",
    "SVO_LOD_MODE_SCREEN_SPACE", "SVO_LOD_MODE_FIXED_LEVEL"]) {
    assert.ok(svoLodDescentWGSL.includes(name), `svoLodDescentWGSL is missing ${name}`);
  }
  const lod = createSvoDrySceneFragmentWGSL(0.5, "raster-primary", "off", "split", 3, false, true, true);
  for (const name of ["dryLodCellStride", "dryLodThresholdPixels", "dryLodFixedLevel",
    "dryPrimaryLeafAggregateHit", "dryLodAggregateIdentity", "dryPrimaryBrickProxySubPixel"]) {
    assert.ok(lod.includes(`fn ${name}`), `LOD build is missing ${name}`);
  }
  // The predicate that collapsed a whole brick on one cell's footprint is gone,
  // not merely unused.
  assert.ok(!lod.includes("dryPrimaryBrickCellsSubPixel"));
  // The threshold reaches the shader through the uniform, so a sweep is a
  // 16-byte write rather than a pipeline rebuild.
  assert.ok(lod.includes("dry.lod.x"));
  // A build compiled without the machinery is still the bit-exact reference:
  // nothing about level of detail appears in it at all.
  const exact = createSvoDrySceneFragmentWGSL(0.5, "raster-primary", "off", "split", 0, false, true, true);
  assert.ok(!/dryLod|svoLod|SVO_LOD_MODE/.test(exact));
});

test("the LOD tier resolves from the brick node, never from the instance sub-AABB", () => {
  const lod = createSvoDrySceneFragmentWGSL(0.5, "raster-primary", "off", "split", 3, false, true, true);
  // The emit kernel publishes each instance's *occupied* sub-AABB
  // (webgpu-svo-brick-raster.ts:727). Deriving cell extents from it and then
  // indexing the payload as full-brick coordinates rendered a squeezed copy of
  // the brick into its occupied corner, so the proxy resolve must take a node
  // index and read the real box.
  assert.ok(lod.includes("fn dryPrimaryLeafProxyHit(ro:vec3f,rd:vec3f,nodeIndex:u32"));
  assert.ok(!/dryPrimaryLeafProxyHit\(ro,rd,proxyBounds/.test(lod));
  assert.ok(!/dryPrimaryLeafProxyHit\(ro,rd,bounds,/.test(lod));
});

test("the LOD tier's solidity test matches the voxels-only walk", () => {
  const lod = createSvoDrySceneFragmentWGSL(0.5, "raster-primary", "off", "split", 3, false, true, true);
  // Ground is ownerless by construction (SPARSE_SCENE_TERRAIN_MATERIAL_OWNER)
  // and on the hero garden so is most of the scenery. The LOD resolve is the
  // only pass a pixel classified into that tier ever gets, so an owner-range
  // test there is a hole and not a missed upgrade.
  assert.ok(lod.includes("fn dryLodCellSolid"));
  assert.ok(/fn dryLodCellSolid\(identity:u32\)->bool\{\s*let owner=identity>>16u;\s*return \(identity&0xffffu\)!=0u/.test(lod));
  assert.ok(!/dryLodCellSolid[\s\S]{0,200}dry\.metadata\.y/.test(lod));
});
