import assert from "node:assert/strict";
import test from "node:test";
import { buildSparseAtlasCompositeGrid } from
  "../lib/methods/adaptive-volume/sparse-atlas-composite-projection";
import { compileSparseCM12FactoredAEIPackedTemplateCatalog } from
  "../lib/methods/adaptive-volume/sparse-cm12-factored-aei-packed-template";
import {
  SPARSE_CM12_FACTORED_AEI_INVALID,
  type SparseCM12FactoredAEIPatchDescriptor,
} from "../lib/methods/adaptive-volume/sparse-cm12-factored-aei-topology";
import {
  createSparseAdaptiveMassAtlas,
  sparseBrickKey,
  type SparseAdaptiveMassBrick,
} from "../lib/methods/adaptive-volume/sparse-brick-atlas";
import { compareSparseCM12IBOSemanticAuthority } from
  "../lib/methods/adaptive-volume/sparse-cm12-ibo-semantic-authority";
import {
  createSparseCM12InternedBoundaryImage,
  prepareSparseCM12InternedBoundaryShadow,
  validateSparseCM12InternedBoundaryPreflip,
} from
  "../lib/methods/adaptive-volume/sparse-cm12-interned-boundary-image";
import {
  compileSparseCM12InternedBoundaryOperators,
  selectSparseCM12InternedBoundaryPatches,
  sparseCM12InternedBoundaryFullFaceExteriorPatch,
} from "../lib/methods/adaptive-volume/sparse-cm12-interned-boundary-operators";
import { packSparseCM12ResidentTopologyTemplatesForQA } from
  "../lib/methods/adaptive-volume/webgpu-sparse-cm12-resident";

const SOURCE = 0;
const POSITIVE_X = 1;
const FACE_RESOLUTION = 8;

const fixture = () => {
  const dimensions = [32, 16, 16] as const;
  const logical = [4, 2, 2] as const;
  const specifications = [
    { coordinate: [0, 0, 0] as const, span: 2, resolution: 8 as const },
    { coordinate: [2, 0, 0] as const, span: 1, resolution: 4 as const },
    { coordinate: [2, 1, 0] as const, span: 1, resolution: 4 as const },
    { coordinate: [2, 0, 1] as const, span: 1, resolution: 4 as const },
    { coordinate: [2, 1, 1] as const, span: 1, resolution: 4 as const },
  ];
  const bricks: SparseAdaptiveMassBrick[] = specifications.map((value) => ({
    key: sparseBrickKey(value.coordinate, logical),
    coordinate: value.coordinate,
    spanBricks: value.span,
    resolution: value.resolution,
    density: new Float64Array(value.resolution ** 3),
    gamma: new Float64Array(value.resolution ** 3).fill(1),
  }));
  const atlas = createSparseAdaptiveMassAtlas(dimensions, bricks, 1, 8);
  const grid = buildSparseAtlasCompositeGrid(atlas);
  const packed = packSparseCM12ResidentTopologyTemplatesForQA(atlas, grid);
  const catalog = compileSparseCM12FactoredAEIPackedTemplateCatalog({
    words: packed.words,
    brickFineResolution: 8,
    brickKeyByLeafId: atlas.bricks.map((brick) => brick.key),
    validDimensions: (_leaf, resolution) => [resolution, resolution, resolution],
    scaleLog2: (leaf, resolution) => Math.log2(
      8 * (atlas.bricks[leaf]!.spanBricks ?? 1) / resolution,
    ),
    selectedResolution: (leaf) => atlas.bricks[leaf]!.resolution,
  });
  return { packed, catalog };
};

const faceSites = (patch: SparseCM12FactoredAEIPatchDescriptor): Set<string> => {
  const result = new Set<string>();
  const [originU, originV] = patch.sourceFaceOrigin;
  const [width, height] = patch.faceDimensions;
  assert.equal(patch.rowCount, width * height,
    `fixture patch ${patch.id} must have a rectangular row footprint`);
  for (let v = originV; v < originV + height; v += 1) {
    for (let u = originU; u < originU + width; u += 1) result.add(`${u},${v}`);
  }
  return result;
};

const sourceTermFaceSites = (
  patch: SparseCM12FactoredAEIPatchDescriptor,
): Set<string> => {
  assert.equal(patch.sourceTermFaceCertified, true);
  assert.equal(patch.sourceTermFaceCount,
    patch.sourceTermFaceDimensions[0] * patch.sourceTermFaceDimensions[1]);
  const result = new Set<string>();
  for (let v = patch.sourceTermFaceOrigin[1];
    v < patch.sourceTermFaceOrigin[1] + patch.sourceTermFaceDimensions[1]; v += 1) {
    for (let u = patch.sourceTermFaceOrigin[0];
      u < patch.sourceTermFaceOrigin[0] + patch.sourceTermFaceDimensions[0]; u += 1) {
      result.add(`${u},${v}`);
    }
  }
  return result;
};

const expectedFaceSites = (): Set<string> => new Set(Array.from(
  { length: FACE_RESOLUTION ** 2 }, (_, ordinal) =>
    `${ordinal % FACE_RESOLUTION},${Math.floor(ordinal / FACE_RESOLUTION)}`,
));

const selected = (activeLeaves: readonly number[]) => {
  const { packed, catalog } = fixture();
  const active = new Set(activeLeaves);
  const patches = selectSparseCM12InternedBoundaryPatches({
    catalog,
    sourceDescriptorId: catalog.descriptorIdByLeaf[SOURCE]!,
    side: POSITIVE_X,
    activeLeaves: active,
    descriptorIdByLeaf: catalog.descriptorIdByLeaf,
  });
  const compilation = compileSparseCM12InternedBoundaryOperators({
    catalog, packedWords: packed.words, activeLeaves,
  });
  const image = createSparseCM12InternedBoundaryImage(compilation,
    activeLeaves, 1, catalog.descriptorIdByLeaf);
  const authority = compareSparseCM12IBOSemanticAuthority({
    image,
    packedWords: packed.words,
    activeLeaves,
    descriptorIdByLeaf: catalog.descriptorIdByLeaf,
    leaves: [SOURCE],
    slot: 0,
  });
  assert.equal(authority.exact, true);
  assert.equal(authority.duplicateCandidateRows, 0);
  return { patches, fallback: sparseCM12InternedBoundaryFullFaceExteriorPatch({
    catalog, sourceDescriptorId: catalog.descriptorIdByLeaf[SOURCE]!,
    side: POSITIVE_X,
  }) };
};

const classify = (patches: readonly SparseCM12FactoredAEIPatchDescriptor[]) => ({
  internal: patches.filter((patch) =>
    patch.targetLeaf !== SPARSE_CM12_FACTORED_AEI_INVALID),
  exterior: patches.filter((patch) =>
    patch.targetLeaf === SPARSE_CM12_FACTORED_AEI_INVALID),
});

test("IBO full four-quadrant cover has one internal authority per face site", () => {
  const { patches, fallback } = selected([0, 1, 2, 3, 4]);
  const { internal, exterior } = classify(patches);
  assert.equal(exterior.length, 0,
    "complete internal coverage supersedes the full-face fallback");
  assert.ok(fallback);
  assert.deepEqual(sourceTermFaceSites(fallback), expectedFaceSites());
  assert.equal(internal.length, 4);
  const visits = new Map<string, number>();
  for (const patch of internal) for (const site of faceSites(patch)) {
    visits.set(site, (visits.get(site) ?? 0) + 1);
  }
  assert.deepEqual(new Set(visits.keys()), expectedFaceSites());
  assert.ok([...visits.values()].every((count) => count === 1),
    "the four active quadrants must tile the face without holes or overlap");
});

test("IBO partial cover retains the fallback and one effective exterior quadrant", () => {
  const { patches, fallback } = selected([0, 1, 2, 3]);
  const { internal, exterior } = classify(patches);
  assert.equal(internal.length, 3);
  assert.equal(exterior.length, 1);
  assert.ok(fallback);
  assert.equal(exterior[0]!.id, fallback.id);
  const fallbackSites = sourceTermFaceSites(fallback);
  assert.deepEqual(fallbackSites, expectedFaceSites());

  const internalVisits = new Map<string, number>();
  for (const patch of internal) for (const site of faceSites(patch)) {
    internalVisits.set(site, (internalVisits.get(site) ?? 0) + 1);
  }
  assert.equal(internalVisits.size, 48);
  assert.ok([...internalVisits.values()].every((count) => count === 1));

  const effectiveInternal = new Set(internalVisits.keys());
  const effectiveExterior = new Set([...fallbackSites].filter((site) =>
    !effectiveInternal.has(site)));
  assert.equal(effectiveExterior.size, 16,
    "row acceptance must leave fallback rows only in the inactive quadrant");
  assert.deepEqual(new Set([...effectiveInternal, ...effectiveExterior]),
    expectedFaceSites());
  assert.equal([...effectiveInternal].filter((site) => effectiveExterior.has(site)).length, 0);
});

test("IBO inactive neighbours retain exactly the complete exterior authority", () => {
  const { patches, fallback } = selected([SOURCE]);
  const { internal, exterior } = classify(patches);
  assert.equal(internal.length, 0);
  assert.equal(exterior.length, 1);
  assert.ok(fallback);
  assert.equal(exterior[0]!.id, fallback.id);
  assert.deepEqual(sourceTermFaceSites(fallback), expectedFaceSites());
});

const sparsePartialFaceFixture = () => {
  const logical = [2, 3, 2] as const;
  const specifications = [
    { coordinate: [0, 0, 0] as const, span: 2, resolution: 4 as const },
    { coordinate: [0, 2, 0] as const, span: 1, resolution: 1 as const },
    { coordinate: [1, 2, 0] as const, span: 1, resolution: 1 as const },
    { coordinate: [0, 2, 1] as const, span: 1, resolution: 1 as const },
    { coordinate: [1, 2, 1] as const, span: 1, resolution: 2 as const },
  ];
  const bricks: SparseAdaptiveMassBrick[] = specifications.map((value) => ({
    key: sparseBrickKey(value.coordinate, logical),
    coordinate: value.coordinate,
    spanBricks: value.span,
    resolution: value.resolution,
    density: new Float64Array(value.resolution ** 3),
    gamma: new Float64Array(value.resolution ** 3).fill(1),
  }));
  const atlas = createSparseAdaptiveMassAtlas([16, 24, 16], bricks, 1, 8);
  const grid = buildSparseAtlasCompositeGrid(atlas);
  const packed = packSparseCM12ResidentTopologyTemplatesForQA(atlas, grid);
  const catalog = compileSparseCM12FactoredAEIPackedTemplateCatalog({
    words: packed.words,
    brickFineResolution: 8,
    brickKeyByLeafId: atlas.bricks.map((brick) => brick.key),
    validDimensions: (_leaf, resolution) => [resolution, resolution, resolution],
    scaleLog2: (leaf, resolution) => Math.log2(
      8 * (atlas.bricks[leaf]!.spanBricks ?? 1) / resolution,
    ),
    selectedResolution: (leaf) => atlas.bricks[leaf]!.resolution,
  });
  return { grid, packed, catalog };
};

test("IBO B4 mixed face keeps four refs while their terms cover all 16 samples", () => {
  const { grid, packed, catalog } = sparsePartialFaceFixture();
  const descriptorId = catalog.descriptorIdByLeaf[SOURCE]!;
  const active = [0, 1, 2, 3, 4];
  const sourceKey = grid.atlas.bricks[SOURCE]!.key;
  const physicalRows = grid.gradientRows.filter((row) => row.axis === 1
    && row.centerFine[1] === 16 && row.negativeBrickKey === sourceKey);
  assert.equal(physicalRows.length, 7);
  assert.equal(physicalRows.filter((row) => row.kind === "brick-face").length, 4);
  assert.equal(physicalRows.filter((row) => row.kind === "mixed-seam").length, 3);
  assert.equal(physicalRows.filter((row) => row.kind === "sparse-air").length, 0);
  const physicalVisits = new Map<string, number>();
  for (const row of physicalRows) for (const term of row.terms) {
    const cell = grid.cells[term.cellId]!;
    if (cell.brickKey !== sourceKey || cell.local[1] !== 3) continue;
    const site = `${cell.local[0]},${cell.local[2]}`;
    physicalVisits.set(site, (physicalVisits.get(site) ?? 0) + 1);
  }
  const expectedSites = new Set(Array.from(
    { length: 16 }, (_, ordinal) => `${ordinal % 4},${Math.floor(ordinal / 4)}`,
  ));
  assert.deepEqual(new Set(physicalVisits.keys()), expectedSites);
  assert.ok([...physicalVisits.values()].every((count) => count === 1),
    "mixed-row source terms must give every B4 sample one physical authority");
  const explicit = selectSparseCM12InternedBoundaryPatches({
    catalog, sourceDescriptorId: descriptorId, side: 3,
    activeLeaves: new Set(active), descriptorIdByLeaf: catalog.descriptorIdByLeaf,
  });
  assert.deepEqual(explicit.map((patch) => [
    ...patch.sourceFaceOrigin, ...patch.faceDimensions,
  ]).sort((a, b) => a[1]! - b[1]! || a[0]! - b[0]!), [
    [1, 1, 1, 1],
    [3, 1, 1, 1],
    [2, 2, 2, 2],
    [1, 3, 1, 1],
  ]);
  assert.ok(explicit.every((patch) =>
    patch.targetLeaf !== SPARSE_CM12_FACTORED_AEI_INVALID));
  assert.deepEqual(explicit.map((patch) => [
    ...patch.sourceTermFaceOrigin, ...patch.sourceTermFaceDimensions,
    patch.sourceTermFaceCount,
  ]).sort((a, b) => a[1]! - b[1]! || a[0]! - b[0]!), [
    [0, 0, 2, 2, 4],
    [2, 0, 2, 2, 4],
    [0, 2, 2, 2, 4],
    [2, 2, 2, 2, 4],
  ]);
  const certifiedSites = new Map<string, number>();
  for (const patch of explicit) for (const site of sourceTermFaceSites(patch)) {
    certifiedSites.set(site, (certifiedSites.get(site) ?? 0) + 1);
  }
  assert.deepEqual(new Set(certifiedSites.keys()), expectedSites);
  assert.ok([...certifiedSites.values()].every((count) => count === 1));

  const fallback = sparseCM12InternedBoundaryFullFaceExteriorPatch({
    catalog, sourceDescriptorId: descriptorId, side: 3,
  });
  assert.ok(fallback);
  assert.deepEqual([...fallback.sourceTermFaceOrigin,
    ...fallback.sourceTermFaceDimensions],
    [0, 0, 4, 4]);
  assert.equal(fallback.sourceTermFaceCount, 16);
  const descriptorFootprints = new Set(explicit.flatMap((patch) =>
    [...faceSites(patch)]));
  assert.equal(descriptorFootprints.size, 7,
    "mixed patch rectangles describe rows, not every source term they cover");
  const effectiveAir = new Set([...sourceTermFaceSites(fallback)].filter((site) =>
    !physicalVisits.has(site)));
  assert.equal(effectiveAir.size, 0);

  const compilation = compileSparseCM12InternedBoundaryOperators({
    catalog, packedWords: packed.words, activeLeaves: [SOURCE],
  });
  assert.equal(compilation.exactStableRowSet, true);
  const image = createSparseCM12InternedBoundaryImage(compilation, [SOURCE], 1,
    catalog.descriptorIdByLeaf);
  assert.equal(compareSparseCM12IBOSemanticAuthority({ image,
    packedWords: packed.words, activeLeaves: [SOURCE],
    descriptorIdByLeaf: catalog.descriptorIdByLeaf, leaves: [SOURCE], slot: 0,
  }).exact, true);

  const prepared = prepareSparseCM12InternedBoundaryShadow({ image,
    targetActiveLeaves: active, targetDescriptorIdByLeaf: catalog.descriptorIdByLeaf,
    changedLeaves: [1, 2, 3, 4], candidateGeneration: 2,
  });
  const sourceAt = image.layout.slotLeafBaseWords[prepared.shadowSlot]
    + SOURCE * 8;
  assert.equal(image.words[sourceAt + 5]! >>> (3 * 3) & 7, 4,
    "the four internal patches exhaust the explicit +Y face slots");
  const receipt = validateSparseCM12InternedBoundaryPreflip({ image,
    targetActiveLeaves: active, targetDescriptorIdByLeaf: catalog.descriptorIdByLeaf,
    changedLeaves: [1, 2, 3, 4],
  });
  assert.equal(receipt.passed, true);
  const authority = compareSparseCM12IBOSemanticAuthority({ image,
    packedWords: packed.words, activeLeaves: active,
    descriptorIdByLeaf: catalog.descriptorIdByLeaf,
    leaves: prepared.deltaClosure, slot: prepared.shadowSlot,
  });
  assert.equal(authority.exact, true);
  assert.equal(authority.duplicateCandidateRows, 0);
});
