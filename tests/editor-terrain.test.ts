import assert from "node:assert/strict";
import test from "node:test";
import {
  applyTerrainFeatureDrag,
  canAddTerrainFeature,
  createTerrainFeature,
  terrainFeatureAt,
  terrainFeatureHandles,
  terrainFeatureIndex,
  terrainFeatureSelectionId,
  TERRAIN_MINIMUM_RADIUS_M,
} from "../lib/editor-terrain";
import {
  applyTerrainBrush,
  bakeTerrainGrid,
  sampleTerrainGrid,
  sceneHasTerrain,
  terrainColumnHeights,
  terrainHeightAt,
  terrainIsSculpted,
  terrainNormalAt,
  validateTerrain,
  MAX_TERRAIN_FEATURES,
  type TerrainDescription,
} from "../lib/terrain";

const container = { width_m: 2.4, height_m: 1.2, depth_m: 1.6 };

function analyticTerrain(): TerrainDescription {
  return {
    baseHeight_m: 0.3,
    features: [
      { kind: "mound", center_m: { x: 0.4, z: -0.2 }, radius_m: { x: 0.3, z: 0.25 }, amount_m: 0.2, flat: 0.2 },
      { kind: "basin", center_m: { x: -0.5, z: 0.3 }, radius_m: { x: 0.5, z: 0.45 }, amount_m: 0.15, flat: 0.3 },
    ],
  };
}

test("terrain feature hit test prefers the tightest containing footprint", () => {
  const terrain = analyticTerrain();
  assert.equal(terrainFeatureAt(terrain, 0.4, -0.2), 0);
  assert.equal(terrainFeatureAt(terrain, -0.5, 0.3), 1);
  assert.equal(terrainFeatureAt(terrain, 1.1, 0.7), undefined, "outside every footprint selects nothing");

  // A small mound sitting inside a broad basin must still be selectable.
  const nested: TerrainDescription = {
    baseHeight_m: 0.4,
    features: [
      { kind: "basin", center_m: { x: 0, z: 0 }, radius_m: { x: 0.9, z: 0.9 }, amount_m: 0.2 },
      { kind: "mound", center_m: { x: 0, z: 0 }, radius_m: { x: 0.15, z: 0.15 }, amount_m: 0.1 },
    ],
  };
  assert.equal(terrainFeatureAt(nested, 0.02, 0.01), 1);
});

test("selection ids survive round-trip and reject stale indices", () => {
  const terrain = analyticTerrain();
  assert.equal(terrainFeatureIndex(terrainFeatureSelectionId(1), terrain), 1);
  assert.equal(terrainFeatureIndex(terrainFeatureSelectionId(7), terrain), undefined, "a removed feature deselects");
  assert.equal(terrainFeatureIndex("body-sphere-1", terrain), undefined);
  assert.equal(terrainFeatureIndex(terrainFeatureSelectionId(0), undefined), undefined);
});

test("feature handles sit on the ground and above it for amount", () => {
  const terrain = analyticTerrain();
  const handles = terrainFeatureHandles(terrain, 0);
  assert.deepEqual(handles.map((handle) => handle.kind), ["center", "radius-x", "radius-z", "amount"]);
  const center = handles[0]!, amount = handles[3]!;
  assert.ok(Math.abs(center.position_m.y - terrainHeightAt(terrain, 0.4, -0.2)) < 1e-12);
  assert.ok(amount.position_m.y > center.position_m.y, "the amount handle rises from the centre");
  assert.equal(terrainFeatureHandles(terrain, 9).length, 0);
});

test("dragging handles stays inside the authoring constraints", () => {
  const terrain = analyticTerrain();
  const moved = applyTerrainFeatureDrag(terrain, 0, "center", { x: 99, y: 0.3, z: -99 }, container);
  assert.equal(moved.features[0]!.center_m.x, container.width_m / 2, "centre clamps to the container");
  assert.equal(moved.features[0]!.center_m.z, -container.depth_m / 2);

  const shrunk = applyTerrainFeatureDrag(terrain, 0, "radius-x", { x: 0.4, y: 0.3, z: -0.2 }, container);
  assert.equal(shrunk.features[0]!.radius_m.x, TERRAIN_MINIMUM_RADIUS_M, "a radius cannot collapse to zero");
  assert.equal(shrunk.features[0]!.radius_m.z, terrain.features[0]!.radius_m.z, "the other axis is untouched");

  // validateTerrain rejects a basin deeper than the base height, so the drag
  // must clamp there rather than produce an unloadable scene.
  const deepened = applyTerrainFeatureDrag(terrain, 1, "amount", { x: -0.5, y: 99, z: 0.3 }, container);
  assert.ok(deepened.features[1]!.amount_m <= terrain.baseHeight_m + 1e-9);
  assert.deepEqual(validateTerrain(deepened, container), []);

  assert.deepEqual(applyTerrainFeatureDrag(terrain, 9, "center", { x: 0, y: 0, z: 0 }, container), terrain);
});

test("new features are valid and the eight-feature cap is enforced", () => {
  const terrain = analyticTerrain();
  const mound = createTerrainFeature("mound", 0.1, 0.1, container, terrain.baseHeight_m);
  const basin = createTerrainFeature("basin", -0.1, -0.1, container, terrain.baseHeight_m);
  const grown = { ...terrain, features: [...terrain.features, mound, basin] };
  assert.deepEqual(validateTerrain(grown, container), []);
  assert.equal(canAddTerrainFeature(grown), true);
  const full = { ...terrain, features: Array.from({ length: MAX_TERRAIN_FEATURES }, () => mound) };
  assert.equal(canAddTerrainFeature(full), false);
});

test("a baked grid reproduces the analytic ground it came from", () => {
  const terrain = analyticTerrain();
  const grid = bakeTerrainGrid(terrain, container, 0.02);
  assert.equal(grid.heights_m.length, grid.size.nx * grid.size.nz);
  // Exact at sample nodes; bilinear between them, so only the node check is tight.
  for (let j = 0; j < grid.size.nz; j += 7) for (let i = 0; i < grid.size.nx; i += 7) {
    const x = grid.origin_m.x + i * grid.spacing_m, z = grid.origin_m.z + j * grid.spacing_m;
    assert.ok(Math.abs(sampleTerrainGrid(grid, x, z) - terrainHeightAt(terrain, x, z)) < 1e-9, `node ${i},${j}`);
  }
  const sculpted: TerrainDescription = { ...terrain, grid };
  assert.equal(terrainIsSculpted(sculpted), true);
  assert.equal(sceneHasTerrain({ terrain: { baseHeight_m: 0, features: [], grid } }), true);
  for (const [x, z] of [[0.13, -0.07], [-0.42, 0.31], [0.8, 0.5]] as const) {
    assert.ok(Math.abs(terrainHeightAt(sculpted, x, z) - terrainHeightAt(terrain, x, z)) < 5e-3, `interior ${x},${z}`);
  }
});

test("grid sampling clamps outside the lattice instead of extrapolating", () => {
  const grid = bakeTerrainGrid(analyticTerrain(), container, 0.05);
  const edge = sampleTerrainGrid(grid, grid.origin_m.x, grid.origin_m.z);
  assert.equal(sampleTerrainGrid(grid, grid.origin_m.x - 50, grid.origin_m.z - 50), edge);
  assert.ok(sampleTerrainGrid(grid, 1e6, 1e6) >= 0, "far samples stay non-negative");
});

test("the brush raises and lowers within a bounded footprint", () => {
  const grid = bakeTerrainGrid({ baseHeight_m: 0.3, features: [] }, container, 0.05);
  const before = sampleTerrainGrid(grid, 0, 0);
  const raised = applyTerrainBrush(grid, 0, 0, 0.3, 0.2, container.height_m);
  assert.ok(sampleTerrainGrid(raised, 0, 0) > before + 0.15, "the brush centre takes the full delta");
  assert.ok(Math.abs(sampleTerrainGrid(raised, 0.9, 0) - before) < 1e-9, "outside the radius is untouched");

  const lowered = applyTerrainBrush(raised, 0, 0, 0.3, -5, container.height_m);
  assert.ok(lowered.heights_m.every((height) => height >= 0), "heights never go negative");
  const overfilled = applyTerrainBrush(grid, 0, 0, 0.3, 99, container.height_m);
  assert.ok(overfilled.heights_m.every((height) => height <= container.height_m), "heights never exceed the container");
  assert.deepEqual(applyTerrainBrush(grid, 0, 0, 0, 1, container.height_m), grid, "a zero radius is inert");
});

test("grid terrain validates and flows through the solver column bake", () => {
  const grid = bakeTerrainGrid(analyticTerrain(), container, 0.05);
  const sculpted: TerrainDescription = { baseHeight_m: 0.3, features: [], grid };
  assert.deepEqual(validateTerrain(sculpted, container), []);

  assert.ok(validateTerrain({ ...sculpted, grid: { ...grid, heights_m: grid.heights_m.slice(1) } }, container)
    .some((error) => error.includes("row-major")));
  assert.ok(validateTerrain({ ...sculpted, grid: { ...grid, spacing_m: 0 } }, container)
    .some((error) => error.includes("spacing")));
  assert.ok(validateTerrain({ ...sculpted, grid: { ...grid, size: { nx: 1, nz: 1 } } }, container)
    .some((error) => error.includes("at least")));
  assert.ok(validateTerrain({ ...sculpted, grid: { ...grid, heights_m: grid.heights_m.map(() => 99) } }, container)
    .some((error) => error.includes("inside [0, container height]")));

  // Solvers consume terrain only as baked column heights, so a sculpted
  // ground needs no kernel change to take effect.
  const columns = terrainColumnHeights({ terrain: sculpted, container }, 24, 16);
  assert.equal(columns.length, 24 * 16);
  assert.ok(columns.some((height) => height > 0.3), "the mound survives the column bake");
  const normal = terrainNormalAt(sculpted, 0.4, -0.2);
  assert.ok(Math.abs(Math.hypot(normal.x, normal.y, normal.z) - 1) < 1e-9, "grid normals stay unit length");
});
