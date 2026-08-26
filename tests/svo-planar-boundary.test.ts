import assert from "node:assert/strict";
import test from "node:test";
import { planAdaptiveSparseBrickOctree } from "../lib/core/adaptive-sparse-brick-plan";
import { packSparseBrickPlan, SPARSE_BRICK_INVALID_INDEX,
  SPARSE_BRICK_LEAF_TERMINAL } from "../lib/svo/sparse-brick-octree";
import { traversePackedSvo } from "../lib/svo/webgpu-svo-traversal";
import {
  buildSvoPlanarBoundaryCatalog,
  buildSvoSolidWorldPlanarBoundaryCatalog,
  createSvoPlanarLeafClassifier,
  isSvoPlanarBoundaryProxy,
  svoPlanarResidualEnvironmentPrimitives,
  svoPlanarResidualSolidWorld,
  svoPlanarBoundaryForProxy,
} from "../lib/svo/svo-planar-boundary";
import { buildEnvironmentProxyCatalog, environmentProxyPrimitives,
  type EnvironmentProxyPrimitive } from "../lib/core/voxel-environments";
import {
  createSvoDrySceneFragmentWGSL,
  SVO_DRY_SCENE_BINDING_CONTRACT,
} from "../lib/svo/webgpu-svo-dry-scene";
import { boxSolidVoxelShell, createSolidWorld, sampleSolidWorld } from "../lib/core/solid-world";
import { getScenePreset } from "../lib/core/scenes";

const plate = {
  kind: "box", key: "test/plate", ownerIndex: 0, group: "test", tags: [],
  center_m: { x: 4, y: 4, z: 4 },
  halfSize_m: { x: 4, y: 0.05, z: 4 },
  material: { colorLinear: [1, 1, 1], roughness: 1, emission: 0 },
  aabb_m: { min: { x: 0, y: 3.95, z: 0 }, max: { x: 8, y: 4.05, z: 8 } },
} as EnvironmentProxyPrimitive;

test("thin authored boxes become oriented finite planar boundary records", () => {
  assert.equal(isSvoPlanarBoundaryProxy(plate), true);
  const patch = svoPlanarBoundaryForProxy(plate, { materialId: 32, ownerId: 12 });
  assert.ok(patch);
  assert.deepEqual(patch.normal, [0, 1, 0]);
  assert.equal(patch.halfThickness_m, 0.05);
  assert.deepEqual([patch.halfExtentU_m, patch.halfExtentV_m], [4, 4]);
});

test("planar boundary promotion preserves arbitrary box orientation", () => {
  const angle = Math.PI / 3;
  const oriented = {
    ...plate,
    orientation: { x: Math.sin(angle / 2), y: 0, z: 0, w: Math.cos(angle / 2) },
  } as EnvironmentProxyPrimitive;
  const patch = svoPlanarBoundaryForProxy(oriented, { materialId: 32, ownerId: 12 });
  assert.ok(patch);
  assert.ok(Math.abs(patch.normal[0]) < 1e-12);
  assert.ok(Math.abs(patch.normal[1] - Math.cos(angle)) < 1e-12);
  assert.ok(Math.abs(patch.normal[2] - Math.sin(angle)) < 1e-12);
});

test("the deliberately open front shell is never promoted", () => {
  const openFront = {
    ...plate,
    key: "test/shell/wall-front",
    tags: ["shell", "wall"],
  } as EnvironmentProxyPrimitive;
  assert.equal(isSvoPlanarBoundaryProxy(openFront), false);
});

test("accepted patch indices are compact and independent of owner order", () => {
  const owners = [
    { ...plate, key: "test/high-owner", ownerIndex: 12 },
    { ...plate, key: "test/not-planar", ownerIndex: 7,
      halfSize_m: { x: 1, y: 1, z: 1 } },
    { ...plate, key: "test/low-owner", ownerIndex: 3 },
  ] as EnvironmentProxyPrimitive[];
  const catalog = buildSvoPlanarBoundaryCatalog(owners, (primitive) => ({
    materialId: primitive.ownerIndex + 32,
    ownerId: primitive.ownerIndex + 64,
  }));
  assert.deepEqual(catalog.sources.map(({ sourceIndex }) => sourceIndex), [0, 1]);
  assert.equal(catalog.patchIndexByOwner.get(12), 0);
  assert.equal(catalog.patchIndexByOwner.get(7), undefined);
  assert.equal(catalog.patchIndexByOwner.get(3), 1);
  assert.deepEqual(catalog.sources.map(({ patch }) => patch.ownerId), [76, 67]);
});

test("accepted environment planes cannot also enter the residual voxel field", () => {
  const solid = { ...plate, key: "test/solid", ownerIndex: 1,
    halfSize_m: { x: 1, y: 1, z: 1 } } as EnvironmentProxyPrimitive;
  const primitives = [plate, solid];
  const catalog = buildSvoPlanarBoundaryCatalog(primitives, (primitive) => ({
    materialId: 32 + primitive.ownerIndex, ownerId: 12 + primitive.ownerIndex,
  }));

  assert.deepEqual(svoPlanarResidualEnvironmentPrimitives(primitives, catalog),
    [solid]);
  assert.ok(catalog.patchIndexByOwner.has(plate.ownerIndex));
  assert.ok(!svoPlanarResidualEnvironmentPrimitives(primitives, catalog)
    .some((primitive) => catalog.patchIndexByOwner.has(primitive.ownerIndex)));
});

test("symmetric expansion stage floor has one analytic owner and no voxel owner", () => {
  const scene = getScenePreset("sparse-cm12-symmetric-expansion").create();
  const primitives = environmentProxyPrimitives(buildEnvironmentProxyCatalog(
    scene, scene.environment ?? "default"), true);
  const floor = primitives.find((primitive) => primitive.key === "stage/stage/floor");
  assert.ok(floor, "the Dawn reproduction scene must contain the studio floor");
  const catalog = buildSvoPlanarBoundaryCatalog(primitives, (primitive) => ({
    materialId: 32 + primitive.ownerIndex, ownerId: 12 + primitive.ownerIndex,
  }));
  const floorRecord = catalog.patchIndexByOwner.get(floor.ownerIndex);
  assert.notEqual(floorRecord, undefined, "the visible floor must be admitted analytically");
  const residual = svoPlanarResidualEnvironmentPrimitives(primitives, catalog);
  assert.equal(residual.some((primitive) => primitive.key === floor.key), false,
    "an admitted floor cannot reach conservative voxel coverage");
  assert.equal(residual.some((primitive) =>
    catalog.patchIndexByOwner.has(primitive.ownerIndex)), false,
  "the residual publication must be disjoint from the entire planar catalogue");
});

test("SolidWorld tank faces append to the same compact planar catalogue", () => {
  const scene = getScenePreset("bounded-pool-transfer").create();
  scene.container.width_m = 16;
  scene.container.height_m = 8.4;
  scene.container.depth_m = 16.4;
  scene.voxelDomain.finestCellSize_m = 1;
  const patches = [
    ...boxSolidVoxelShell([16, 8, 16]),
    { operation: "clear" as const, minimum: [4, -1, 4] as const,
      maximumExclusive: [5, 0, 5] as const },
  ];
  const catalog = buildSvoSolidWorldPlanarBoundaryCatalog(scene, patches, 4);

  assert.deepEqual(catalog.sources.map(({ sourceIndex }) => sourceIndex),
    [4, 5, 6, 7, 8]);
  assert.deepEqual([...catalog.patchIndexByPatch],
    [[0, 4], [1, 5], [2, 6], [3, 7], [4, 8]]);
  assert.equal(catalog.patchIndexByPatch.has(5), false,
    "a clear edit has no analytic record");
  assert.deepEqual(catalog.sources[0]!.bounds_m, {
    minimum: [-8, -1.05, -8.2], maximum: [8, 0, 8.2],
  });
});

test("accepted SolidWorld planes are absent from the render voxel residual", () => {
  const scene = getScenePreset("bounded-pool-transfer").create();
  scene.container.width_m = 16;
  scene.container.height_m = 8;
  scene.container.depth_m = 16;
  scene.voxelDomain.finestCellSize_m = 1;
  const patches = [
    ...boxSolidVoxelShell([16, 8, 16]),
    { operation: "fill" as const, minimum: [4, 2, 4] as const,
      maximumExclusive: [6, 4, 6] as const, materialId: 9 },
  ];
  const world = createSolidWorld(patches);
  const catalog = buildSvoSolidWorldPlanarBoundaryCatalog(scene, patches);
  const residual = svoPlanarResidualSolidWorld(world, catalog);

  assert.equal(sampleSolidWorld(world, [2, -1, 2]).materialId, 1,
    "the canonical world retains the tank floor for fluid boundary sampling");
  assert.equal(sampleSolidWorld(residual, [2, -1, 2]).materialId, 0,
    "the render voxel residual cannot contain the accepted floor");
  assert.equal(sampleSolidWorld(residual, [4, 2, 4]).materialId, 9,
    "non-planar solids remain voxel-owned");
  for (const patchIndex of catalog.patchIndexByPatch.keys()) {
    assert.ok(!residual.patches.includes(patches[patchIndex]!));
  }
});

test("transparent symmetric-expansion tank shell is physical-only, never opaque SVO geometry", () => {
  const scene = getScenePreset("sparse-cm12-symmetric-expansion").create();
  const world = createSolidWorld(scene.solidVoxels);
  const catalog = buildSvoSolidWorldPlanarBoundaryCatalog(scene, world.patches);
  const residual = svoPlanarResidualSolidWorld(world, catalog);

  assert.equal(catalog.sources.length, 0,
    "glass owns presentation of all six canonical shell faces");
  assert.deepEqual([...catalog.residualExcludedPatchIndices], [0, 1, 2, 3, 4, 5]);
  assert.equal(sampleSolidWorld(world, [8, -1, 8]).materialId, 1,
    "the canonical tank floor remains solid for CM12");
  assert.equal(sampleSolidWorld(residual, [8, -1, 8]).materialId, 0,
    "the opaque renderer cannot publish a second floor below the glass");
});

test("explicit legacy glass keeps its dielectric shell in the voxel residual", () => {
  const scene = getScenePreset("sparse-cm12-symmetric-expansion").create();
  scene.container.vessel = "glass";
  const world = createSolidWorld(scene.solidVoxels);
  const catalog = buildSvoSolidWorldPlanarBoundaryCatalog(scene, world.patches);
  const residual = svoPlanarResidualSolidWorld(world, catalog);

  assert.equal(catalog.sources.length, 0,
    "container glass must never be promoted into the opaque planar catalogue");
  assert.equal(catalog.residualExcludedPatchIndices.size, 0);
  assert.equal(sampleSolidWorld(residual, [8, -1, 8]).materialId, 1,
    "the explicit glass option retains the thin-dielectric voxel shell");
});

test("a cut tank face stays voxel-owned instead of receiving a false outline owner", () => {
  const scene = getScenePreset("sparse-cm12-symmetric-expansion").create();
  scene.container.vessel = "outline";
  scene.solidVoxels.push({ operation: "clear", minimum: [4, -1, 4],
    maximumExclusive: [5, 0, 5] });
  const world = createSolidWorld(scene.solidVoxels);
  const catalog = buildSvoSolidWorldPlanarBoundaryCatalog(scene, world.patches);
  const residual = svoPlanarResidualSolidWorld(world, catalog);

  assert.equal(catalog.residualExcludedPatchIndices.has(0), false,
    "the edited canonical floor must not be removed from the render residual");
  assert.equal(sampleSolidWorld(residual, [3, -1, 3]).materialId, 1,
    "the uncut portion of the rejected face remains visible voxel volume");
  assert.equal(sampleSolidWorld(residual, [4, -1, 4]).materialId, 0,
    "the authored hole survives residual composition");
});

test("only isolated planar leaves become macro terminals", () => {
  const patch = svoPlanarBoundaryForProxy(plate, { materialId: 32, ownerId: 12 })!;
  const source = { patch, sourceIndex: 0, bounds_m: {
    minimum: [0, 3.95, 0] as const, maximum: [8, 4.05, 8] as const,
  } };
  const classifier = createSvoPlanarLeafClassifier({
    sources: [source],
    blockers: [{ ...source.bounds_m, planarSourceIndex: 0 }],
    worldOrigin_m: [0, 0, 0], nodeEdge_m: [[8, 8, 8]],
  });
  assert.deepEqual(classifier(0, { x: 0, y: 0, z: 0 }), {
    kind: SPARSE_BRICK_LEAF_TERMINAL.planarBoundary, index: 0,
  });

  const residual = createSvoPlanarLeafClassifier({
    sources: [source],
    blockers: [
      { ...source.bounds_m, planarSourceIndex: 0 },
      { minimum: [3, 3, 3], maximum: [5, 5, 5] },
    ],
    worldOrigin_m: [0, 0, 0], nodeEdge_m: [[8, 8, 8]],
  });
  assert.equal(residual(0, { x: 0, y: 0, z: 0 }).kind,
    SPARSE_BRICK_LEAF_TERMINAL.voxels);
  assert.equal(residual.requiresFineVoxelResidual(0, { x: 0, y: 0, z: 0 }), true);
  assert.equal(residual.requiresFineVoxelResidual(0, { x: 2, y: 2, z: 2 }), false);
});

test("the cutover leaf ABI packs terminal kind and accepted record index", () => {
  const plan = planAdaptiveSparseBrickOctree({
    brickSize: 8,
    solverBricks: [], proxyBricks: [{ x: 0, y: 0, z: 0 }],
    maximumDepth: 0, maximumEnvironmentCoarseningPower: 0,
    classifyEnvironmentLeaf: () => ({
      kind: SPARSE_BRICK_LEAF_TERMINAL.planarBoundary, index: 7,
    }),
  });
  const packed = packSparseBrickPlan(plan, 1);
  assert.deepEqual([...packed.leaves], [0, 0,
    SPARSE_BRICK_LEAF_TERMINAL.planarBoundary, 7]);
  assert.notEqual(packed.leaves[3], SPARSE_BRICK_INVALID_INDEX);
  const traversed = traversePackedSvo({ origin: [4, 10, 4], direction: [0, -1, 0] }, {
    nodes: packed.nodes, leaves: packed.leaves,
  }, { origin: [0, 0, 0], cellSize: [1, 1, 1], brickSize: 8, maximumDepth: 0 });
  assert.equal(traversed.status, "hit");
  if (traversed.status === "hit") {
    assert.equal(traversed.hit.terminalKind, SPARSE_BRICK_LEAF_TERMINAL.planarBoundary);
    assert.equal(traversed.hit.terminalIndex, 7);
  }
});

test("the cutover rejects legacy or malformed terminal lanes", () => {
  assert.throws(() => planAdaptiveSparseBrickOctree({
    brickSize: 8,
    solverBricks: [], proxyBricks: [{ x: 0, y: 0, z: 0 }],
    maximumDepth: 0, maximumEnvironmentCoarseningPower: 0,
    classifyEnvironmentLeaf: () => ({
      kind: SPARSE_BRICK_LEAF_TERMINAL.voxels,
      index: 0,
    }),
  }), /does not match its kind/);
});

test("the production shader binds and resolves the accepted planar catalogue", () => {
  const planarBinding = SVO_DRY_SCENE_BINDING_CONTRACT
    .find(({ binding }) => binding === 6);
  assert.equal(planarBinding?.type, "read-only-storage");
  const shader = createSvoDrySceneFragmentWGSL(1, "canonical", "bounds", "inline");
  assert.match(shader, /@group\(0\) @binding\(6\) var<storage,read> dryPlanarBoundaries/);
  const start = shader.indexOf("fn dryPlanarTerminalHit(");
  const end = shader.indexOf("// Exact live-scene acceleration", start);
  const terminalFunction = shader.slice(start, end);
  assert.match(terminalFunction, /intersectPlanarBoundary\(boundary/);
  assert.doesNotMatch(terminalFunction, /dryPrimitive\(/);
  assert.match(shader, /fn dryLeafStructuralPlanar\(hit:SvoTraversalHit\)->bool/);
  assert.match(shader, /return dryLeafStructuralPlanar\(hit\)\s*\|\|svoBrickLifecycleCurrent/);
  assert.match(shader, /fn dryPlanarCatalogHit\(/);
  assert.match(shader, /var seeded=dryPlanarCatalogHit\(/);
});

test("flat-voxel presentation never redraws a lattice over an exact planar terminal", () => {
  const shader = createSvoDrySceneFragmentWGSL(1, "canonical", "bounds", "inline");
  const start = shader.indexOf("fn dryVoxelFaceEdgeFactor(");
  const end = shader.indexOf("/**\n * The baked normal", start);
  const edgeFactor = shader.slice(start, end);

  assert.match(edgeFactor, /fieldSource:u32/);
  assert.match(edgeFactor, /fieldSource==DRY_GBUFFER_FIELD_ANALYTIC/);
  assert.match(shader,
    /dryVoxelFaceEdgeFactor\(position,hit\.normal,hit\.t,hit\.fieldSource\)/);
});
