import assert from "node:assert/strict";
import test from "node:test";
import type { SceneDescription } from "../lib/model";
import type { SparseSceneDomainPlan } from "../lib/sparse-scene-domain";
import { SVO_STATIC_NODE_MIP_DEFAULT_CAPACITY, buildSvoStaticNodeMipPublication } from "../lib/svo-static-node-mips";
import type { EnvironmentProxyPrimitive } from "../lib/voxel-environments";

function scene(terrain = false): SceneDescription {
  return {
    schemaVersion: "1.0.0", sceneId: "static-node-mip-test", environment: "garden", randomSeed: 1, duration_s: 1,
    container: { width_m: 16, height_m: 8, depth_m: 16, fillFraction: 0, top: "open", fluidWallMode: "free-slip" },
    voxelDomain: { finestCellSize_m: 1, brickSize_cells: 8 },
    terrain: terrain ? { baseHeight_m: .5, features: [] } : undefined,
    fluid: { density_kg_m3: 1_000, dynamicViscosity_Pa_s: .001, surfaceTension_N_m: .07, gravity_m_s2: { x: 0, y: -9.81, z: 0 }, initialCondition: "tank-fill" },
    nominalResolution: { length_m: 1 },
    numerics: { fixedDt_s: .01, maxDt_s: .01, pressureRelativeTolerance: 1e-5, pressureMaxIterations: 10 },
    rigidBodies: [],
  };
}

function domain(options: { origin?: readonly [number, number, number]; dimensions?: readonly [number, number, number]; cell?: number } = {}): SparseSceneDomainPlan {
  const origin = options.origin ?? [0, 0, 0], dimensions = options.dimensions ?? [16, 16, 16], cell = options.cell ?? 1;
  const solverGridOriginCells = [0, Math.max(0, Math.round(-origin[1] / cell)), 0] as const;
  const solverDimensionsCells = [dimensions[0], Math.min(8, dimensions[1] - solverGridOriginCells[1]), dimensions[2]] as const;
  const point = (values: readonly [number, number, number]) => ({ x: values[0], y: values[1], z: values[2] });
  const worldMinimum = point(origin), worldMaximum = point(origin.map((value, axis) => value + dimensions[axis] * cell) as [number, number, number]);
  const solverMinimum = point(origin.map((value, axis) => value + solverGridOriginCells[axis] * cell) as [number, number, number]);
  const solverMaximum = point(([solverMinimum.x + solverDimensionsCells[0] * cell, solverMinimum.y + solverDimensionsCells[1] * cell, solverMinimum.z + solverDimensionsCells[2] * cell]));
  return {
    brickSize: 8, cellSize_m: [cell, cell, cell], worldOrigin_m: worldMinimum,
    solverGridOriginCells, solverDimensionsCells, sceneDimensionsCells: dimensions,
    brickDimensions: dimensions.map((value) => Math.ceil(value / 8)) as [number, number, number],
    solverBounds_m: { min: solverMinimum, max: solverMaximum }, worldBounds_m: { min: worldMinimum, max: worldMaximum },
    solverBrickCoordinates: [], environmentBrickCoordinates: [], proxyBrickCoordinates: [], coordinates: [],
  };
}

function box(
  key: string,
  center: readonly [number, number, number],
  half: readonly [number, number, number],
  group = "stone",
  tags: readonly string[] = [],
): EnvironmentProxyPrimitive {
  return {
    kind: "box", key, ownerIndex: 0, group, tags,
    center_m: { x: center[0], y: center[1], z: center[2] },
    halfSize_m: { x: half[0], y: half[1], z: half[2] },
    material: { colorLinear: [1, 1, 1], emission: 0, roughness: .8 },
    aabb_m: {
      min: { x: center[0] - half[0], y: center[1] - half[1], z: center[2] - half[2] },
      max: { x: center[0] + half[0], y: center[1] + half[1], z: center[2] + half[2] },
    },
  };
}

function texel(interior: Uint8Array, x: number, y: number, z: number): readonly number[] {
  const offset = ((z * 8 + y) * 8 + x) * 4;
  return [...interior.slice(offset, offset + 4)];
}

test("static proxy publication builds base interiors and recursively reduced parents", () => {
  const publication = buildSvoStaticNodeMipPublication(scene(), domain(), [box("solid", [4, 4, 4], [4, 4, 4])], {
    generation: 12, levelCount: 2, samplesPerAxis: 2,
  });
  assert.equal(publication.plan.complete, true);
  assert.equal(publication.interiors.length, 2);
  assert.equal(publication.selectedBasePageCount, 1);
  const base = publication.interiors.find(({ key }) => key.level === 0)!;
  const parent = publication.interiors.find(({ key }) => key.level === 1)!;
  assert.deepEqual(texel(base.interior, 0, 0, 0), [255, 255, 0, 0]);
  assert.deepEqual(texel(base.interior, 7, 7, 7), [255, 255, 0, 0]);
  assert.deepEqual(texel(parent.interior, 0, 0, 0), [255, 255, 0, 0]);
  assert.deepEqual(texel(parent.interior, 4, 0, 0), [0, 0, 0, 0]);
});

test("terrain selects surface pages without allocating deep solid volume", () => {
  const publication = buildSvoStaticNodeMipPublication(scene(true), domain({ origin: [0, -16, 0], dimensions: [16, 24, 16] }), [], {
    generation: 2, levelCount: 1,
  });
  assert.ok(publication.terrainCandidatePageCount > 0);
  const baseYs = new Set(publication.plan.pages.map(({ key }) => key.coordinate[1]));
  assert.deepEqual([...baseYs], [1, 2], "only the two pages adjacent to y=.5 are selected, not deep page y=0");
  assert.ok(publication.interiors.some(({ interior }) => interior.some((value, index) => index % 4 === 0 && value > 0)));
  for (const { interior } of publication.interiors) {
    for (let index = 0; index < interior.length; index += 4) assert.deepEqual([interior[index + 2], interior[index + 3]], [0, 0]);
  }
});

test("capacity selection omits base pages before planning so the returned plan stays complete", () => {
  const proxies = [
    box("a", [1, 1, 1], [.4, .4, .4]), box("b", [9, 1, 1], [.4, .4, .4]),
    box("c", [1, 1, 9], [.4, .4, .4]), box("d", [9, 1, 9], [.4, .4, .4]),
  ];
  const publication = buildSvoStaticNodeMipPublication(scene(), domain(), proxies, { generation: 3, levelCount: 2, capacity: 2 });
  assert.equal(publication.plan.complete, true);
  assert.ok(publication.plan.pages.length <= 2);
  assert.equal(publication.selectedBasePageCount, 1);
  assert.equal(publication.omittedBasePageCount, 3);
});

test("default static policy excludes glass and the open front shell", () => {
  const primitives = [
    box("room/shell/wall-front", [1, 1, 1], [.4, .4, .4], "shell-wall", ["shell", "wall"]),
    box("glass", [9, 1, 1], [.4, .4, .4], "glass-pane", ["glass"]),
    box("opaque", [1, 1, 9], [.4, .4, .4]),
  ];
  const publication = buildSvoStaticNodeMipPublication(scene(), domain(), primitives, { generation: 4, levelCount: 1 });
  assert.equal(publication.proxyCandidatePageCount, 1);
  assert.equal(publication.selectedBasePageCount, 1);
});

test("default capacity fits the guaranteed WebGPU sampled-directory height", () => {
  assert.equal(SVO_STATIC_NODE_MIP_DEFAULT_CAPACITY, 8_192);
});
