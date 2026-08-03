import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { GARDEN_CONTAINER, gardenPoolTerrain } from "../lib/garden-scene";
import {
  createHeroGardenHoseScene,
  HERO_GARDEN_CONTAINER,
  HERO_GARDEN_VESSEL,
} from "../lib/hero-garden-scene";
import {
  sampleTerrainGrid,
  TERRAIN_CEILING_MARGIN_M,
  terrainCeiling,
  terrainGridExtent,
  terrainHeightAt,
  type TerrainGrid,
} from "../lib/terrain";
import { planVoxelScene, type VoxelTerrainSource } from "../lib/voxel-scene";
import { pondVesselHeightAt, pondVesselPlanCurve } from "../lib/voxel-scenery/pond-vessel";
import {
  packSvoPrimitiveRecords,
  sampleSvoPrimitive,
  SVO_PRIMITIVE_FLAGS,
  SVO_PRIMITIVE_KINDS,
  unpackSvoPrimitiveRecords,
} from "../lib/svo-primitive-abi";
import {
  intersectSvoTerrainHeightfield,
  packSvoDrySceneTerrainHeightfield,
  SVO_DRY_SCENE_TERRAIN_REFERENCE,
  svoDrySceneTerrainPrimitive,
  svoDrySceneTerrainResolver,
  SVO_DRY_SCENE_ARENA_LAYOUT,
  SVO_DRY_SCENE_TERRAIN_HEADER_BYTES,
  SVO_DRY_SCENE_TERRAIN_HEADER_WORDS,
  svoDrySceneArenaSizeBytes,
  svoDrySceneShader,
  svoDrySceneTerrainRegionBytes,
  unpackSvoDrySceneTerrainHeightfield,
} from "../lib/webgpu-svo-dry-scene";

/**
 * The hero pond is the fixture because it is the scene the bug was found in: a
 * vessel whose every surface — basin, inner face, coping — is a sculpted grid
 * and nothing else, so any path that quietly falls back to the analytic form
 * renders a flat plane at `baseHeight_m` and says so loudly.
 */
const scene = createHeroGardenHoseScene();
const terrain = scene.terrain!;
const grid = terrain.grid!;
const curve = pondVesselPlanCurve(HERO_GARDEN_VESSEL);
const sceneScale = Math.max(HERO_GARDEN_CONTAINER.width_m, HERO_GARDEN_CONTAINER.height_m, HERO_GARDEN_CONTAINER.depth_m);
const analytic = gardenPoolTerrain();

/** Sum of the analytic mound amounts: the whole of the old ceiling, and zero here. */
const analyticRise = (features: typeof terrain.features) =>
  features.reduce((sum, feature) => sum + (feature.kind === "mound" ? feature.amount_m : 0), 0);

const terrainSourceOf = (description = scene): VoxelTerrainSource =>
  planVoxelScene(description).sceneSources.find((source) => source.kind === "terrain-heightfield") as VoxelTerrainSource;

test("a sculpted grid survives into the voxel source instead of being dropped on the way", () => {
  const source = terrainSourceOf();
  assert.ok(source.terrain.grid, "the published terrain must carry the sculpted heights the solver reads");
  assert.deepEqual(source.terrain.grid!.size, grid.size);
  assert.equal(source.terrain.grid!.spacing_m, grid.spacing_m);
  assert.deepEqual(source.terrain.grid!.heights_m, grid.heights_m);
  // The clone exists so a published source cannot alias document state, and a
  // shared heights array would let an editor stroke rewrite a published
  // generation underneath the renderer.
  assert.notEqual(source.terrain.grid!.heights_m, grid.heights_m);
});

test("the voxel source bounds the real crest, not the base ground", () => {
  const source = terrainSourceOf();
  const crest = Math.max(...grid.heights_m);
  // The bound the analytic sum would have produced. It is `baseHeight_m` alone
  // for a grid-only terrain, so the whole coping stood outside the candidate.
  const analyticBound = terrain.baseHeight_m + analyticRise(terrain.features);
  assert.ok(crest > analyticBound + 0.05,
    `the vessel must reach materially above ${analyticBound} for this test to mean anything`);
  assert.ok(source.candidate.exact_m.max.y >= crest,
    `terrain bounds stop at ${source.candidate.exact_m.max.y}, below the ${crest} crest`);
  assert.ok(source.candidate.exact_m.max.y <= scene.container.height_m);
});

test("the ceiling is the tallest ground the terrain can reach, sculpted or analytic", () => {
  const extent = terrainGridExtent(grid);
  assert.equal(terrainCeiling(terrain), extent.maximum_m + TERRAIN_CEILING_MARGIN_M);
  // Against the generator rather than against the bake, so this pins the
  // ceiling to the vessel that was authored and not merely to its own samples.
  let tallest = 0;
  for (const [x, z] of curve) tallest = Math.max(tallest, pondVesselHeightAt(HERO_GARDEN_VESSEL, curve, x, z));
  assert.ok(terrainCeiling(terrain) > tallest,
    `the ceiling ${terrainCeiling(terrain)} must clear the crest ${tallest}`);
  // ...and the margin is a margin, not a hiding place for a wrong bound.
  assert.ok(terrainCeiling(terrain) - tallest < 2 * TERRAIN_CEILING_MARGIN_M);

  // Analytic terrain keeps exactly the bound it had before the grid existed.
  assert.equal(terrainCeiling(analytic), analytic.baseHeight_m + analyticRise(analytic.features) + TERRAIN_CEILING_MARGIN_M);
  assert.equal(terrainCeiling(undefined), TERRAIN_CEILING_MARGIN_M);
});

test("the grid extent bounds its own slope, which is what the march marches on", () => {
  const extent = terrainGridExtent(grid);
  assert.equal(extent.minimum_m, Math.min(...grid.heights_m));
  assert.equal(extent.maximum_m, Math.max(...grid.heights_m));
  assert.ok(extent.slopeBound > 1, "a vessel with a bullnose rim and an inner face is not gentle ground");
  // The bound has to hold for the interpolated surface, not only at the nodes,
  // or the march it licenses could step over the coping after all.
  const step = grid.spacing_m / 3;
  let steepest = 0;
  for (let x = -0.7; x <= 0.7; x += 0.01) for (let z = -0.5; z <= 0.5; z += 0.01) {
    const dx = (terrainHeightAt(terrain, x + step, z) - terrainHeightAt(terrain, x - step, z)) / (2 * step);
    const dz = (terrainHeightAt(terrain, x, z + step) - terrainHeightAt(terrain, x, z - step)) / (2 * step);
    steepest = Math.max(steepest, Math.hypot(dx, dz));
  }
  assert.ok(steepest <= extent.slopeBound + 1e-9,
    `the surface reaches slope ${steepest}, above the published bound ${extent.slopeBound}`);
});

test("a ray cast down at the coping hits the coping", () => {
  // This is the assertion the bug failed. The old bracket started at
  // `baseHeight_m + 0.05` and the crest stands 5.5 cm above the base ground, so
  // the ray entered *inside* the rim, never changed sign, and reported nothing
  // at all — a hole exactly where the vessel is.
  for (const [x, z] of curve) {
    const hit = intersectSvoTerrainHeightfield(terrain, { x, y: 0.58, z }, { x: 0, y: -1, z: 0 }, sceneScale);
    assert.ok(hit, `no hit on the crest at (${x.toFixed(3)}, ${z.toFixed(3)})`);
    assert.equal(hit.solver, "sculpted");
    assert.ok(hit.position_m.y > terrain.baseHeight_m + 0.7 * HERO_GARDEN_VESSEL.rimHeight_m,
      `the crest resolved at ${hit.position_m.y}, near the base ground ${terrain.baseHeight_m}`);
    assert.ok(Math.abs(hit.position_m.y - terrainHeightAt(terrain, x, z)) < 1e-4,
      `the crest resolved at ${hit.position_m.y} rather than ${terrainHeightAt(terrain, x, z)}`);
    // And the ground the ray reports is the ground the generator drew, to
    // within what a 6.25 mm bake of a hand-formed rim can hold.
    assert.ok(Math.abs(hit.position_m.y - pondVesselHeightAt(HERO_GARDEN_VESSEL, curve, x, z)) < 1e-3);
  }
});

test("a ray into the basin resolves the basin floor, not the ground it is cut into", () => {
  for (const [x, z] of [[0, 0], [0.1, -0.05], [-0.12, 0.08]] as const) {
    const hit = intersectSvoTerrainHeightfield(terrain, { x, y: 0.58, z }, { x: 0, y: -1, z: 0 }, sceneScale);
    assert.ok(hit);
    assert.ok(Math.abs(hit.position_m.y - terrainHeightAt(terrain, x, z)) < 1e-4);
    assert.ok(hit.position_m.y < terrain.baseHeight_m - 0.5 * HERO_GARDEN_VESSEL.basinDepth_m,
      `the basin floor resolved at ${hit.position_m.y}, which is not inside the vessel`);
    assert.ok(Math.abs(hit.position_m.y - pondVesselHeightAt(HERO_GARDEN_VESSEL, curve, x, z)) < 1e-3);
  }
});

test("the march crosses the coping without stepping over it", () => {
  // A profile across the rim, at the resolution the rim is expressed at. A
  // bracket tuned for smooth analytic mounds samples three points across the
  // whole interval and reports the ground beyond; every sample here has to be
  // the surface directly under it.
  const z = 0;
  let crest = 0;
  for (let x = 0.40; x <= 0.72; x += grid.spacing_m) {
    const hit = intersectSvoTerrainHeightfield(terrain, { x, y: 0.58, z }, { x: 0, y: -1, z: 0 }, sceneScale);
    assert.ok(hit, `no hit crossing the coping at x=${x.toFixed(4)}`);
    const expected = terrainHeightAt(terrain, x, z);
    assert.ok(Math.abs(hit.position_m.y - expected) < 1e-4,
      `x=${x.toFixed(4)} resolved ${hit.position_m.y} instead of ${expected}`);
    crest = Math.max(crest, hit.position_m.y);
  }
  assert.ok(crest > terrain.baseHeight_m + 0.9 * HERO_GARDEN_VESSEL.rimHeight_m,
    `the sweep never found the rim; it topped out at ${crest}`);
});

test("an angled ray reports the first crossing rather than a later one", () => {
  const origin = { x: -0.95, y: 0.55, z: 0.18 };
  const direction = { x: 0.72, y: -0.62, z: -0.14 };
  const hit = intersectSvoTerrainHeightfield(terrain, origin, direction, sceneScale);
  assert.ok(hit);
  assert.ok(Math.abs(hit.position_m.y - terrainHeightAt(terrain, hit.position_m.x, hit.position_m.z)) < 1e-4);
  const length = Math.hypot(direction.x, direction.y, direction.z);
  const rd = { x: direction.x / length, y: direction.y / length, z: direction.z / length };
  for (let t = 0.005; t < hit.t_m - 0.001; t += 0.0005) {
    const point = { x: origin.x + rd.x * t, y: origin.y + rd.y * t, z: origin.z + rd.z * t };
    assert.ok(point.y - terrainHeightAt(terrain, point.x, point.z) > 0,
      `the march passed through the ground at t=${t.toFixed(4)}`);
  }
});

test("analytic terrain still takes the analytic solvers", () => {
  const gardenScale = Math.max(GARDEN_CONTAINER.width_m, GARDEN_CONTAINER.height_m, GARDEN_CONTAINER.depth_m);
  const x = -0.2, z = 0.1;
  const hit = intersectSvoTerrainHeightfield(analytic, { x, y: 1.4, z }, { x: 0, y: -1, z: 0 }, gardenScale);
  assert.ok(hit);
  assert.equal(hit.solver, "fast", "a scene with no grid must not pay for the sculpted march");
  assert.ok(Math.abs(hit.position_m.y - terrainHeightAt(analytic, x, z)) < 5e-4);
  assert.equal(packSvoDrySceneTerrainHeightfield(undefined), undefined);
});

test("the packed heightfield region round-trips into the same bilinear surface", () => {
  const packed = packSvoDrySceneTerrainHeightfield(grid)!;
  assert.equal(packed.byteLength, svoDrySceneTerrainRegionBytes(grid.size.nx * grid.size.nz));
  const region = unpackSvoDrySceneTerrainHeightfield(packed)!;
  assert.ok(region);
  assert.deepEqual(region.size, grid.size);
  // Everything but the sample count arrives as f32, so these are equal to the
  // precision the GPU has rather than to the precision the bake had.
  assert.ok(Math.abs(region.spacing_m - grid.spacing_m) < 1e-9);
  assert.ok(Math.abs(region.origin_m.x - grid.origin_m.x) < 1e-6);
  assert.ok(Math.abs(region.origin_m.z - grid.origin_m.z) < 1e-6);
  const extent = terrainGridExtent(grid);
  assert.ok(Math.abs(region.minimum_m - extent.minimum_m) < 1e-6);
  assert.ok(Math.abs(region.maximum_m - extent.maximum_m) < 1e-6);
  assert.ok(Math.abs(region.slopeBound - extent.slopeBound) < 1e-6);

  // The header has to be enough on its own: the shader reconstructs the grid
  // from it and nothing else, so a mirror built only from what was packed must
  // sample identically to the grid the solver holds.
  const mirrored: TerrainGrid = {
    kind: "grid",
    origin_m: region.origin_m,
    spacing_m: region.spacing_m,
    size: region.size,
    heights_m: [...region.heights_m],
  };
  let worst = 0;
  for (let x = -1.0; x <= 1.0; x += 0.017) for (let z = -0.7; z <= 0.7; z += 0.013) {
    worst = Math.max(worst, Math.abs(sampleTerrainGrid(mirrored, x, z) - sampleTerrainGrid(grid, x, z)));
  }
  assert.ok(worst < 1e-6, `the packed surface differs from the authored one by ${worst}`);
  // Beyond the footprint too, because the sampler clamps rather than wrapping
  // and a shader that clamped differently would show it as a skirt at the wall.
  for (const [x, z] of [[-9, -9], [9, 9], [0, -9], [9, 0]] as const) {
    assert.ok(Math.abs(sampleTerrainGrid(mirrored, x, z) - sampleTerrainGrid(grid, x, z)) < 1e-6,
      `the clamped skirt at (${x}, ${z}) does not survive packing`);
  }
});

test("a zeroed region is the encoding for analytic terrain", () => {
  assert.equal(unpackSvoDrySceneTerrainHeightfield(new Uint32Array(SVO_DRY_SCENE_TERRAIN_HEADER_WORDS)), undefined);
  const truncated = packSvoDrySceneTerrainHeightfield(grid)!.slice(0, SVO_DRY_SCENE_TERRAIN_HEADER_WORDS + 4);
  assert.throws(() => unpackSvoDrySceneTerrainHeightfield(truncated), /Sculpted terrain region declares/);
});

test("the arena sizes its terrain region per scene and keeps the header always addressable", () => {
  // The header is reserved even with no grid anywhere, because a storage read
  // past the end of the binding is not required to return zero — and zero is
  // exactly what tells the shader that this scene is analytic.
  assert.equal(SVO_DRY_SCENE_ARENA_LAYOUT.sizeBytes - SVO_DRY_SCENE_ARENA_LAYOUT.terrainOffsetBytes >= SVO_DRY_SCENE_TERRAIN_HEADER_BYTES, true);
  assert.equal(svoDrySceneArenaSizeBytes(0), SVO_DRY_SCENE_ARENA_LAYOUT.sizeBytes);
  const packed = packSvoDrySceneTerrainHeightfield(grid)!;
  assert.ok(svoDrySceneArenaSizeBytes(packed.byteLength)
    >= SVO_DRY_SCENE_ARENA_LAYOUT.terrainOffsetBytes + packed.byteLength);
  // ...and materially under the authored 262 144-sample ceiling, which is the
  // whole reason the region is scene-sized rather than reserved.
  assert.ok(packed.byteLength < 0.3 * svoDrySceneTerrainRegionBytes(262_144));
  assert.equal(SVO_DRY_SCENE_ARENA_LAYOUT.terrainOffsetBytes % 256, 0, "regions stay 256-byte aligned for writeBuffer");
});

test("the terrain heightfield primitive kind finally has a producer, and it names the arena region", () => {
  // The kind, the `externalTerrain` flag and the `terrainReference` word have
  // all been in the ABI from the start with nothing constructing one. The
  // reference is the arena region's own word offset, so a consumer that is
  // handed the record needs no side table to find the samples.
  const primitive = svoDrySceneTerrainPrimitive({ primitiveId: 1, materialId: 2, ownerId: 3 });
  assert.equal(primitive.terrainReference, SVO_DRY_SCENE_TERRAIN_REFERENCE);
  assert.equal(SVO_DRY_SCENE_TERRAIN_REFERENCE, SVO_DRY_SCENE_ARENA_LAYOUT.terrainOffsetBytes / 4);

  const [record] = unpackSvoPrimitiveRecords(packSvoPrimitiveRecords([primitive]));
  assert.equal(record.kind, "terrain-heightfield");
  assert.equal(record.kind === "terrain-heightfield" && record.terrainReference, SVO_DRY_SCENE_TERRAIN_REFERENCE);
  assert.equal(record.materialId, 2);
  assert.equal(record.ownerId, 3);
  assert.equal(SVO_PRIMITIVE_KINDS.terrainHeightfield, 6);
  assert.equal(SVO_PRIMITIVE_FLAGS.externalTerrain, 4);
});

test("a record and its arena region evaluate the same ground the terrain does", () => {
  const primitive = svoDrySceneTerrainPrimitive({ primitiveId: 1, materialId: 2 });
  const resolve = svoDrySceneTerrainResolver(packSvoDrySceneTerrainHeightfield(grid));
  for (const [x, z] of [...curve.slice(0, 8), [0, 0], [0.8, 0.5]] as const) {
    const sample = sampleSvoPrimitive(primitive, { x, y: 0.4, z }, resolve);
    assert.ok(Math.abs(sample.signedDistance_m - (0.4 - terrainHeightAt(terrain, x, z))) < 1e-6,
      `the record disagrees with the terrain at (${x}, ${z})`);
  }
  // A reference that names something else resolves to nothing rather than to a
  // flat plane that would render as plausible, wrong ground.
  assert.equal(resolve(SVO_DRY_SCENE_TERRAIN_REFERENCE + 1), undefined);
  assert.equal(svoDrySceneTerrainResolver(undefined)(SVO_DRY_SCENE_TERRAIN_REFERENCE), undefined);
});

test("the WGSL samples the arena heightfield with the exact form the CPU sampler uses", () => {
  assert.match(svoDrySceneShader, /fn terrainGridHeightAt\(grid:DryTerrainGrid,x:f32,z:f32\)->f32\{/);
  assert.match(svoDrySceneShader, /let fx=clamp\(\(x-grid\.origin\.x\)\/grid\.spacing,0\.0,f32\(grid\.size\.x-1\)\);/);
  assert.match(svoDrySceneShader, /let fz=clamp\(\(z-grid\.origin\.y\)\/grid\.spacing,0\.0,f32\(grid\.size\.y-1\)\);/);
  assert.match(svoDrySceneShader, /let x0=min\(grid\.size\.x-2,i32\(floor\(fx\)\)\);let z0=min\(grid\.size\.y-2,i32\(floor\(fz\)\)\);/);
  assert.match(svoDrySceneShader, /return max\(0\.0,lower\*\(1\.0-tz\)\+upper\*tz\);/);
  // The height query has to route through it, or the analytic form is still
  // what every ray, normal and shadow in the frame is evaluating.
  assert.match(svoDrySceneShader, /fn terrainHeightAt\(x:f32,z:f32\)->f32\{[^]*if\(terrainSculpted\(grid\)\)\{return terrainGridHeightAt\(grid,x,z\);\}/);
  assert.match(svoDrySceneShader, /fn terrainCeiling\(\)->f32\{[^]*if\(terrainSculpted\(grid\)\)\{return grid\.maximum\+0\.05;\}/);
  assert.match(svoDrySceneShader, /fn traceTerrain\(ro:vec3f,rd:vec3f\)->DryHit\{[^]*if\(terrainSculpted\(sculpted\)\)\{return traceTerrainHeightfield\(ro,rd,sculpted\);\}/);
  // The march's safety property is the step, so pin the step.
  assert.match(svoDrySceneShader, /let rate=max\(1e-4,abs\(rd\.y\)\+grid\.slopeBound\*length\(rd\.xz\)\);/);
  assert.match(svoDrySceneShader, /t=t\+abs\(field\)\/rate;/);
  assert.equal(svoDrySceneShader.includes(`const DRY_SCENE_TERRAIN_WORD_OFFSET:u32=${SVO_DRY_SCENE_ARENA_LAYOUT.terrainOffsetBytes / 4}u;`), true);

  const drySceneSource = readFileSync(new URL("../lib/webgpu-svo-dry-scene.ts", import.meta.url), "utf8");
  assert.match(drySceneSource, /writeBuffer\(\s*this\.sceneArenaBuffer,\s*SVO_DRY_SCENE_ARENA_LAYOUT\.terrainOffsetBytes,\s*scene\.terrainHeightfield \?\? new Uint32Array\(SVO_DRY_SCENE_TERRAIN_HEADER_WORDS\),/,
    "the header must be republished unconditionally so a flat scene cannot inherit the previous vessel");
  const rendererSource = readFileSync(new URL("../lib/webgpu-renderer.ts", import.meta.url), "utf8");
  assert.match(rendererSource, /terrainHeightfield: packSvoDrySceneTerrainHeightfield\(scene\.terrain\?\.grid\)/,
    "the sculpted grid must reach the presentation publication, which is the hop it was missing");
});
