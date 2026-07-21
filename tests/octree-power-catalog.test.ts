import assert from "node:assert/strict";
import test from "node:test";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { constructOctreePowerCell, createOctreePowerSite, type OctreePowerSite } from "../lib/octree-power-geometry";
import {
  OCTREE_POWER_CATALOG_FACE_FLOATS,
  OCTREE_POWER_CATALOG_TARGET_BYTES,
  OCTREE_POWER_CATALOG_WARNING_BYTES,
  buildOctreePowerCatalog,
  canonicalizeOctreePowerConfiguration,
  resolveOctreePowerCatalogDescriptor,
} from "../lib/octree-power-catalog";
import { OCTREE_CUBE_TRANSFORMS, inverseCubeTransform, transformPowerVector } from "../lib/octree-power-topology";
import {
  sitesForSameOrCoarserPowerDescriptor,
  sitesForSameOrFinerPowerDescriptor,
} from "../lib/octree-power-descriptor";
import { describeOctreePowerRow } from "../lib/webgpu-octree-power-descriptor";
import {
  decodeGeneratedOctreePowerCatalog,
  fetchGeneratedOctreePowerCatalog,
  OCTREE_GENERATED_POWER_CATALOG_MANIFEST,
} from "../lib/generated/octree-power-catalog";

function uniformSites(): OctreePowerSite[] {
  return [
    createOctreePowerSite("a", [0, 0, 0], 1),
    createOctreePowerSite("nx", [-1, 0, 0], 1), createOctreePowerSite("px", [1, 0, 0], 1),
    createOctreePowerSite("ny", [0, -1, 0], 1), createOctreePowerSite("py", [0, 1, 0], 1),
    createOctreePowerSite("nz", [0, 0, -1], 1), createOctreePowerSite("pz", [0, 0, 1], 1),
  ];
}

function transitionSites(): OctreePowerSite[] {
  return [
    createOctreePowerSite("a", [0, 0, 0], 2),
    createOctreePowerSite("nx", [-2, 0, 0], 2), createOctreePowerSite("ny", [0, -2, 0], 2),
    createOctreePowerSite("nz", [0, 0, -2], 2),
    createOctreePowerSite("px", [2, 0, 0], 1), createOctreePowerSite("py", [0, 2, 0], 1),
    createOctreePowerSite("pz", [0, 0, 2], 1),
  ];
}

function transformedSites(sites: readonly OctreePowerSite[], transformCode: number): OctreePowerSite[] {
  const transform = OCTREE_CUBE_TRANSFORMS[transformCode];
  const anchor = sites.find((site) => site.key === "a")!;
  return sites.map((site) => {
    const relative = site.center.map((value, axis) => value - anchor.center[axis]) as [number, number, number];
    const center = transformPowerVector(relative, transform).map((value, axis) => value + anchor.center[axis]);
    return createOctreePowerSite(site.key, center.map((value) => value - site.size / 2) as [number, number, number], site.size);
  });
}

test("catalog canonicalizes rotations and safe reflections into one entry", () => {
  const original = transitionSites();
  const rotated = transformedSites(original, 17);
  assert.equal(canonicalizeOctreePowerConfiguration(original[0], original).key,
    canonicalizeOctreePowerConfiguration(rotated[0], rotated).key);
  const catalog = buildOctreePowerCatalog([
    { descriptor: 100, anchorKey: "a", sites: original },
    { descriptor: 101, anchorKey: "a", sites: rotated },
  ]);
  assert.equal(catalog.entries.length, 1);
  assert.equal(catalog.lookup.length, 2);
  assert.equal(resolveOctreePowerCatalogDescriptor(catalog, 100)?.entry, 0);
  assert.equal(resolveOctreePowerCatalogDescriptor(catalog, 101)?.entry, 0);
});

test("catalog emits deterministic compact typed arrays and geometry manifest", () => {
  const configurations = [
    { descriptor: 0, anchorKey: "a", sites: uniformSites() },
    { descriptor: 1, anchorKey: "a", sites: transitionSites() },
  ];
  const a = buildOctreePowerCatalog(configurations);
  const b = buildOctreePowerCatalog([...configurations].reverse());
  assert.deepEqual([...a.entryHeaders], [...b.entryHeaders]);
  assert.deepEqual([...a.entryVolumes], [...b.entryVolumes]);
  assert.deepEqual([...a.faceData], [...b.faceData]);
  assert.deepEqual([...a.tetrahedronHeaders], [...b.tetrahedronHeaders]);
  assert.deepEqual([...a.tetrahedronData], [...b.tetrahedronData]);
  assert.deepEqual(a.manifest, b.manifest);
  assert.equal(a.faceData.length % OCTREE_POWER_CATALOG_FACE_FLOATS, 0);
  assert.ok(a.manifest.maximumFaceIncidence >= 6);
  assert.ok(a.manifest.maximumTetrahedra > 0);
  assert.ok(a.manifest.worstFloat32GeometryError < 1e-6);
  assert.ok(a.manifest.byteCount < OCTREE_POWER_CATALOG_TARGET_BYTES);
  assert.equal(resolveOctreePowerCatalogDescriptor(a, 0)?.descriptor, 0);
  assert.equal(resolveOctreePowerCatalogDescriptor(a, 0xdead_beef), undefined);
});

test("catalog rejects duplicate descriptors and unbounded configurations", () => {
  const uniform = { descriptor: 7, anchorKey: "a", sites: uniformSites() };
  assert.throws(() => buildOctreePowerCatalog([uniform, uniform]), /Duplicate/);
  const lonely = createOctreePowerSite("a", [0, 0, 0], 1);
  assert.throws(() => buildOctreePowerCatalog([{ descriptor: 8, anchorKey: "a", sites: [lonely] }]), /empty or unbounded/);
});

test("generated exhaustive catalog decodes within the fixed budget and proven bounds", () => {
  const bytes = readFileSync(join(process.cwd(), "lib/generated/octree-power-catalog.bin"));
  const data = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
  const views = decodeGeneratedOctreePowerCatalog(data);
  assert.equal(OCTREE_GENERATED_POWER_CATALOG_MANIFEST.descriptorCount, 1_608);
  assert.equal(OCTREE_GENERATED_POWER_CATALOG_MANIFEST.maximumFaceIncidence, 30);
  assert.equal(OCTREE_GENERATED_POWER_CATALOG_MANIFEST.maximumNeighborRows, 30);
  assert.equal(OCTREE_GENERATED_POWER_CATALOG_MANIFEST.maximumTetrahedra, 56);
  assert.ok(OCTREE_GENERATED_POWER_CATALOG_MANIFEST.byteCount < OCTREE_POWER_CATALOG_WARNING_BYTES);
  assert.equal(views.entryHeaders.length, OCTREE_GENERATED_POWER_CATALOG_MANIFEST.configurationCount * 2);
  assert.equal(views.lookup.length, OCTREE_GENERATED_POWER_CATALOG_MANIFEST.descriptorCount * 3);
  assert.equal(views.sameOrFinerDirect.length, 1 << 18);
  assert.equal(views.sameOrCoarserDirect.length, 1 << 9);
  assert.equal(views.tetrahedronHeaders.length, OCTREE_GENERATED_POWER_CATALOG_MANIFEST.configurationCount * 3);
  assert.ok(views.tetrahedronData.length > 0);
  assert.equal(views.tetrahedronVertexData.length, 75 * 4);
  for (const direct of [views.sameOrFinerDirect, views.sameOrCoarserDirect]) {
    assert.ok([...direct].every((packed) => packed !== 0xffff_ffff
      && (packed & 0xffff) < OCTREE_GENERATED_POWER_CATALOG_MANIFEST.configurationCount
      && (packed >>> 16) < 48));
  }
  assert.ok([...views.entryVolumes].every((volume) => volume > 0 && Number.isFinite(volume)));
  assert.equal(typeof fetchGeneratedOctreePowerCatalog, "function");
  for (let offset = 0; offset < views.faceData.length; offset += OCTREE_POWER_CATALOG_FACE_FLOATS) {
    assert.ok(views.faceData[offset + 4] > 0, "normalized area must be positive");
    assert.ok(views.faceData[offset + 11] > 0, "inverse dual distance must be positive");
  }
  for (let entry = 0; entry < OCTREE_GENERATED_POWER_CATALOG_MANIFEST.configurationCount; entry += 1) {
    const first = views.tetrahedronHeaders[entry * 3], count = views.tetrahedronHeaders[entry * 3 + 1];
    assert.ok(first <= views.tetrahedronData.length && count <= views.tetrahedronData.length - first);
    assert.ok(count <= OCTREE_GENERATED_POWER_CATALOG_MANIFEST.maximumTetrahedra);
    for (let local = 0; local < count; local += 1) {
      const packed = views.tetrahedronData[first + local];
      const selectors = [packed & 255, (packed >> 8) & 255, (packed >> 16) & 255];
      assert.equal(new Set(selectors).size, 3);
      assert.ok(selectors.every((selector) => selector < views.tetrahedronVertexData.length / 4));
      const [a, b, c] = selectors.map((selector) => [...views.tetrahedronVertexData.slice(selector * 4, selector * 4 + 3)]);
      const determinant = a[0] * (b[1] * c[2] - b[2] * c[1])
        - a[1] * (b[0] * c[2] - b[2] * c[0]) + a[2] * (b[0] * c[1] - b[1] * c[0]);
      assert.ok(Math.abs(determinant) > 1e-6, `entry ${entry} tetrahedron ${local} must have nonzero volume`);
      const lengths = [a, b, c].map((value) => Math.hypot(...value));
      const dot = (left: number[], right: number[]) => left.reduce((sum, value, axis) => sum + value * right[axis], 0);
      const denominator = lengths[0] * lengths[1] * lengths[2] + dot(a, b) * lengths[2]
        + dot(a, c) * lengths[1] + dot(b, c) * lengths[0];
      const angle = 2 * Math.atan2(Math.abs(determinant), denominator);
      assert.ok(angle > 0 && angle < 4 * Math.PI, `entry ${entry} tetrahedron ${local} has an invalid solid angle`);
    }
  }
});

test("every nonuniform tetra selector set contains exactly every power-face neighbor", () => {
  const bytes = readFileSync(join(process.cwd(), "lib/generated/octree-power-catalog.bin"));
  const views = decodeGeneratedOctreePowerCatalog(
    bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
  const key = (values: ArrayLike<number>, offset: number) => Array.from({ length: 4 },
    (_, component) => Math.round(Number(values[offset + component]) * 1e6)).join(",");
  const boundaryEntries = new Set(Array.from({ length: views.lookup.length / 3 }, (_, index) => views.lookup[index * 3 + 1]));
  let checked = 0;
  for (let entry = 0; entry < OCTREE_GENERATED_POWER_CATALOG_MANIFEST.configurationCount; entry += 1) {
    // Boundary entries intentionally replace exterior virtual-site endpoints
    // with exact size-zero world planes while retaining the base entry's full
    // Delaunay stencil for Section 5 interpolation.
    if (boundaryEntries.has(entry)) continue;
    const tetraFirst = views.tetrahedronHeaders[entry * 3];
    const tetraCount = views.tetrahedronHeaders[entry * 3 + 1];
    const tetraFlags = views.tetrahedronHeaders[entry * 3 + 2];
    if ((tetraFlags & 1) !== 0) continue;
    const selectorGeometry = new Set<string>();
    for (let local = 0; local < tetraCount; local += 1) {
      const packed = views.tetrahedronData[tetraFirst + local];
      for (const selector of [packed & 255, (packed >> 8) & 255, (packed >> 16) & 255]) {
        selectorGeometry.add(key(views.tetrahedronVertexData, selector * 4));
      }
    }
    const faceFirst = views.entryHeaders[entry * 2];
    const faceCount = views.entryHeaders[entry * 2 + 1];
    const faceGeometry = new Set(Array.from({ length: faceCount }, (_, local) =>
      key(views.faceData, (faceFirst + local) * OCTREE_POWER_CATALOG_FACE_FLOATS)));
    assert.deepEqual(selectorGeometry, faceGeometry,
      `entry ${entry} selector geometry must exactly close its face endpoints`);
    assert.ok(selectorGeometry.size <= OCTREE_GENERATED_POWER_CATALOG_MANIFEST.maximumNeighborRows);
    checked += 1;
  }
  assert.equal(checked, 6_471,
    "the exhaustive nonuniform catalog is the authority for terminal support closure");
});

test("direct quotient lookup reconstructs non-canonical world geometry", () => {
  const bytes = readFileSync(join(process.cwd(), "lib/generated/octree-power-catalog.bin"));
  const views = decodeGeneratedOctreePowerCatalog(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
  for (const descriptor of [0x00002, 0x12345, 0x2abcd]) {
    const sites = sitesForSameOrFinerPowerDescriptor(descriptor);
    const anchor = sites.find((site) => site.key === "0,0,0/2")!;
    const direct = constructOctreePowerCell(anchor, sites).faces.map((face) => ({
      neighbor: sites.find((site) => site.key === face.incidentSiteKey)!.center.map((value, axis) => (value - anchor.center[axis]) / anchor.size),
      centroid: face.centroid.map((value, axis) => (value - anchor.center[axis]) / anchor.size),
      normal: face.normal,
      area: face.area / anchor.size ** 2,
    })).sort((a, b) => a.neighbor[0] - b.neighbor[0] || a.neighbor[1] - b.neighbor[1] || a.neighbor[2] - b.neighbor[2]);
    const packed = views.sameOrFinerDirect[descriptor];
    const entry = packed & 0xffff;
    const inverse = inverseCubeTransform(OCTREE_CUBE_TRANSFORMS[packed >>> 16]);
    const first = views.entryHeaders[entry * 2], count = views.entryHeaders[entry * 2 + 1];
    const reconstructed = Array.from({ length: count }, (_, localFace) => {
      const offset = (first + localFace) * OCTREE_POWER_CATALOG_FACE_FLOATS;
      return {
        neighbor: transformPowerVector([...views.faceData.slice(offset, offset + 3)] as [number, number, number], inverse),
        area: views.faceData[offset + 4],
        centroid: transformPowerVector([...views.faceData.slice(offset + 5, offset + 8)] as [number, number, number], inverse),
        normal: transformPowerVector([...views.faceData.slice(offset + 8, offset + 11)] as [number, number, number], inverse),
      };
    }).sort((a, b) => a.neighbor[0] - b.neighbor[0] || a.neighbor[1] - b.neighbor[1] || a.neighbor[2] - b.neighbor[2]);
    assert.equal(reconstructed.length, direct.length);
    reconstructed.forEach((face, index) => {
      assert.ok(Math.abs(face.area - direct[index].area) < 1e-6);
      [...face.neighbor, ...face.centroid, ...face.normal].forEach((value, component) => {
        const expected = [...direct[index].neighbor, ...direct[index].centroid, ...direct[index].normal][component];
        assert.ok(Math.abs(value - expected) < 1e-6, `descriptor ${descriptor}, face ${index}, component ${component}`);
      });
    });
  }
});

test("one shared dyadic tiling produces reciprocal coarse/fine edge descriptors", () => {
  // This is the exact descriptor geometry that exposed anchor-relative owner
  // synthesis in the production GPU audit.  Complete the coarse row's local
  // neighborhood into one non-overlapping dyadic tiling, then query both rows
  // from that same owner function.
  const coarseDescriptor = 51_577;
  const shift = [6, 6, 6] as const;
  const dimensions = [16, 16, 16] as const;
  const leaves: { origin: [number, number, number]; size: number }[] =
    sitesForSameOrFinerPowerDescriptor(coarseDescriptor).map((site) => ({
      origin: site.origin.map((value, axis) => value + shift[axis]) as [number, number, number],
      size: site.size,
    }));
  const contains = (leaf: { origin: [number, number, number]; size: number }, cell: readonly number[]) =>
    cell.every((value, axis) => value >= leaf.origin[axis] && value < leaf.origin[axis] + leaf.size);
  for (let z = 0; z < dimensions[2]; z += 1) for (let y = 0; y < dimensions[1]; y += 1) {
    for (let x = 0; x < dimensions[0]; x += 1) {
      if (!leaves.some((leaf) => contains(leaf, [x, y, z]))) leaves.push({ origin: [x, y, z], size: 1 });
    }
  }
  const ownerAt = (cell: readonly [number, number, number]) => {
    const matches = leaves.filter((leaf) => contains(leaf, cell));
    assert.equal(matches.length, 1, `cell ${cell.join(",")} must have one shared owner`);
    return matches[0];
  };
  const linear = (origin: readonly [number, number, number]) =>
    origin[0] + dimensions[0] * (origin[1] + dimensions[1] * origin[2]);
  const coarse = describeOctreePowerRow({ cell: linear(shift), size: 2 }, dimensions, 2, ownerAt);
  const fineOrigin = [8, 8, 6] as const;
  const fine = describeOctreePowerRow({ cell: linear(fineOrigin), size: 1 }, dimensions, 2, ownerAt);
  assert.deepEqual(coarse, { descriptor: coarseDescriptor, flags: 0, kind: "same-or-finer" });
  assert.deepEqual(fine, { descriptor: 0x8000_0158, flags: 0, kind: "same-or-coarser" });

  const coarseSites = sitesForSameOrFinerPowerDescriptor(coarse.descriptor);
  const coarseAnchor = coarseSites.find((site) => site.key === "0,0,0/2")!;
  const coarseFace = constructOctreePowerCell(coarseAnchor, coarseSites).faces
    .find((face) => face.incidentSiteKey === "2,2,0/1")!;
  const fineSites = sitesForSameOrCoarserPowerDescriptor(fine.descriptor);
  const fineAnchor = fineSites.find((site) => site.key === "anchor")!;
  const fineFace = constructOctreePowerCell(fineAnchor, fineSites).faces
    .find((face) => face.incidentSiteKey === "coarse:3")!;
  assert.ok(coarseFace && fineFace, "both catalog configurations must retain the shared edge face");
  coarseFace.normal.forEach((value, axis) => assert.ok(Math.abs(value + fineFace.normal[axis]) < 1e-9));
  const coarseNeighbor = coarseSites.find((site) => site.key === coarseFace.incidentSiteKey)!;
  const fineNeighbor = fineSites.find((site) => site.key === fineFace.incidentSiteKey)!;
  coarseNeighbor.center.forEach((value, axis) => {
    const coarseOffset = value - coarseAnchor.center[axis];
    const fineOffset = fineNeighbor.center[axis] - fineAnchor.center[axis];
    assert.ok(Math.abs(coarseOffset + fineOffset) < 1e-9,
      "reciprocal catalog selectors must encode opposite world-space center offsets");
  });
});
