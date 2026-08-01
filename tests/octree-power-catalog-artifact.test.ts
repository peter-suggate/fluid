import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import test from "node:test";
import {
  OCTREE_POWER_CATALOG_FACE_FLOATS,
  OCTREE_POWER_RECONSTRUCTION_FLOATS,
  OCTREE_POWER_ROW_TEMPLATE_FLAGS,
  OCTREE_POWER_ROW_TEMPLATE_FLOATS,
  OCTREE_POWER_ROW_TEMPLATE_HEADER_WORDS,
  OCTREE_POWER_ROW_TEMPLATE_INVALID_SELECTOR,
  OCTREE_POWER_ROW_TEMPLATE_VERSION,
  OCTREE_POWER_TRANSFER_RELATION,
  unpackOctreePowerRowTemplateSlot,
} from "../lib/octree-power-catalog";
import {
  decodeGeneratedOctreePowerCatalog,
  OCTREE_GENERATED_POWER_CATALOG_MAGIC,
  OCTREE_GENERATED_POWER_CATALOG_MANIFEST,
} from "../lib/generated/octree-power-catalog";
import {
  OCTREE_POWER_NEIGHBOR_DIRECTIONS,
  OCTREE_POWER_SAME_OR_FINER_MASK,
  sitesForSameOrCoarserPowerDescriptor,
} from "../lib/octree-power-descriptor";
import { constructOctreePowerCell } from "../lib/octree-power-geometry";
import { OCTREE_CUBE_TRANSFORMS, transformPowerVector } from "../lib/octree-power-topology";

const catalogUrl = new URL("../lib/generated/octree-power-catalog.bin", import.meta.url);

test("normal development and validation consume the committed catalog", () => {
  const packageJson = JSON.parse(readFileSync(new URL("../package.json", import.meta.url), "utf8")) as {
    scripts: Record<string, string>;
  };
  assert.equal(Object.keys(packageJson.scripts).some((name) => name.startsWith("pre")
    && packageJson.scripts[name].includes("generate:octree-power-catalog")), false);
  assert.equal(packageJson.scripts["verify:octree-power-catalog"].startsWith("npm run check:octree-power-catalog"), true);
});

test("generated power catalog carries a verified format version and content hash", () => {
  const bytes = readFileSync(catalogUrl);
  const hash = createHash("sha256").update(bytes).digest("hex");
  const data = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
  const header = new Uint32Array(data, 0, 40);

  assert.equal(OCTREE_GENERATED_POWER_CATALOG_MANIFEST.generatorVersion, 9);
  assert.match(OCTREE_GENERATED_POWER_CATALOG_MANIFEST.generatorHash, /^[0-9a-f]{64}$/);
  assert.equal(OCTREE_GENERATED_POWER_CATALOG_MANIFEST.binarySha256, hash);
  assert.equal(header[0], OCTREE_GENERATED_POWER_CATALOG_MAGIC);
  assert.equal(header[1], OCTREE_GENERATED_POWER_CATALOG_MANIFEST.version);
  assert.equal(header[7], bytes.byteLength);
  assert.equal(header[20], 40);
  assert.equal(header[25], 19);
  assert.equal(header[26], OCTREE_POWER_ROW_TEMPLATE_VERSION);
  assert.equal(header[28], OCTREE_POWER_ROW_TEMPLATE_HEADER_WORDS);
  assert.equal(header[32], OCTREE_POWER_ROW_TEMPLATE_FLOATS);
  assert.equal(header[35], OCTREE_POWER_RECONSTRUCTION_FLOATS);
  assert.equal(header[36], 0);
  assert.equal(header[38], 1);
  assert.equal(header[39], 1);
  assert.doesNotThrow(() => decodeGeneratedOctreePowerCatalog(data));
});

test("generated Section 6.3 coefficient channels exactly match every canonical stencil", () => {
  const bytes = readFileSync(catalogUrl);
  const catalog = decodeGeneratedOctreePowerCatalog(
    bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
  const bits = (value: number) => new Uint32Array(Float32Array.of(value).buffer)[0];
  assert.equal(catalog.coefficientData.length, catalog.entryVolumes.length * 19);
  const directions = [
    [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1],
    [1, 1, 0], [1, -1, 0], [-1, 1, 0], [-1, -1, 0],
    [1, 0, 1], [1, 0, -1], [-1, 0, 1], [-1, 0, -1],
    [0, 1, 1], [0, 1, -1], [0, -1, 1], [0, -1, -1],
  ] as const;
  let wideEntries = 0;
  for (let entry = 0; entry < catalog.entryVolumes.length; entry += 1) {
    const firstFace = catalog.entryHeaders[entry * 2];
    const faceCount = catalog.entryHeaders[entry * 2 + 1];
    if (faceCount > 18) wideEntries += 1;
    let diagonal = Math.fround(0);
    const expectedChannels = new Float32Array(18);
    for (let slot = 0; slot < faceCount; slot += 1) {
      const face = (firstFace + slot) * 12;
      const expected = Math.fround(Math.fround(catalog.faceData[face + 4])
        * Math.fround(catalog.faceData[face + 11]));
      diagonal = Math.fround(diagonal + expected);
      if (catalog.faceData[face + 3] !== 0) {
        const direction = [0, 1, 2].map((axis) => Math.round(
          catalog.faceData[face + 5 + axis] + 0.5 * catalog.faceData[face + 8 + axis]));
        const channel = directions.findIndex((candidate) => candidate.every(
          (value, axis) => value === direction[axis]));
        assert.notEqual(channel, -1, `entry ${entry}, face ${slot} direction`);
        expectedChannels[channel] = Math.fround(expectedChannels[channel]! + expected);
      }
    }
    assert.equal(bits(catalog.coefficientData[entry * 19]), bits(diagonal), `entry ${entry} diagonal`);
    for (let channel = 0; channel < 18; channel += 1) {
      assert.equal(bits(catalog.coefficientData[entry * 19 + 1 + channel]), bits(expectedChannels[channel]!),
        `entry ${entry}, canonical direction ${channel}`);
    }
  }
  assert.ok(wideEntries > 0, "the exhaustive catalog must exercise semantic cut rows wider than eighteen slots");
});

test("versioned dense row templates exhaustively match geometry and reconstruct xyz", () => {
  const bytes = readFileSync(catalogUrl);
  const catalog = decodeGeneratedOctreePowerCatalog(
    bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
  const bits = (value: number) => new Uint32Array(Float32Array.of(value).buffer)[0];
  assert.equal(OCTREE_GENERATED_POWER_CATALOG_MANIFEST.rowTemplateVersion,
    OCTREE_POWER_ROW_TEMPLATE_VERSION);
  assert.equal(OCTREE_GENERATED_POWER_CATALOG_MANIFEST.regularCaseId, 0);
  assert.ok(OCTREE_GENERATED_POWER_CATALOG_MANIFEST.worstReconstructionResidual < 2e-6);
  assert.equal(catalog.sameOrFinerDirect[OCTREE_POWER_SAME_OR_FINER_MASK] & 0xffff, 0);
  let extraFamilySlots = 0;
  let boundarySlots = 0;
  let finerSlots = 0;
  let coarserSlots = 0;
  for (let entry = 0; entry < catalog.entryVolumes.length; entry += 1) {
    const faceFirst = catalog.entryHeaders[entry * 2];
    const faceCount = catalog.entryHeaders[entry * 2 + 1];
    const headerOffset = entry * OCTREE_POWER_ROW_TEMPLATE_HEADER_WORDS;
    const templateFirst = catalog.rowTemplateHeaders[headerOffset];
    const templateCount = catalog.rowTemplateHeaders[headerOffset + 1];
    const flags = catalog.rowTemplateHeaders[headerOffset + 2];
    const dynamicCount = catalog.rowTemplateHeaders[headerOffset + 3];
    assert.equal(templateFirst, faceFirst, `dense case ${entry} first slot`);
    assert.equal(templateCount, faceCount, `dense case ${entry} slot count`);
    assert.equal(dynamicCount, faceCount, `dense case ${entry} dynamic slots`);
    let diagonal = Math.fround(0);
    let entryBoundarySlots = 0;
    const identity = new Array<number>(9).fill(0);
    for (let local = 0; local < faceCount; local += 1) {
      const slot = templateFirst + local;
      const packed = unpackOctreePowerRowTemplateSlot(catalog.rowTemplateSlots[slot]);
      const face = (faceFirst + local) * OCTREE_POWER_CATALOG_FACE_FLOATS;
      const sizeRatio = catalog.faceData[face + 3];
      const area = catalog.faceData[face + 4];
      const inverseDistance = catalog.faceData[face + 11];
      const coefficient = Math.fround(Math.fround(area) * Math.fround(inverseDistance));
      assert.equal(bits(catalog.rowTemplateData[slot * OCTREE_POWER_ROW_TEMPLATE_FLOATS]), bits(coefficient));
      assert.equal(bits(catalog.rowTemplateData[slot * OCTREE_POWER_ROW_TEMPLATE_FLOATS + 1]), bits(area));
      assert.equal(bits(catalog.rowTemplateData[slot * OCTREE_POWER_ROW_TEMPLATE_FLOATS + 2]), bits(inverseDistance));
      diagonal = Math.fround(diagonal + coefficient);
      assert.equal(packed.dynamicBoundarySlot, local);
      assert.ok(packed.family < 6);
      if (packed.family >= 3) extraFamilySlots += 1;
      if (sizeRatio === 0) {
        boundarySlots += 1;
        entryBoundarySlots += 1;
        assert.equal(packed.worldBoundary, true);
        assert.equal(packed.transferRelation, OCTREE_POWER_TRANSFER_RELATION.boundary);
        assert.equal(packed.neighborSelector, OCTREE_POWER_ROW_TEMPLATE_INVALID_SELECTOR);
      } else {
        assert.equal(packed.worldBoundary, false);
        assert.ok(packed.neighborSelector < catalog.tetrahedronVertexData.length / 4);
        const selector = packed.neighborSelector * 4;
        for (let axis = 0; axis < 3; axis += 1) {
          assert.equal(bits(catalog.tetrahedronVertexData[selector + axis]),
            bits(catalog.faceData[face + axis]));
        }
        assert.equal(bits(catalog.tetrahedronVertexData[selector + 3]), bits(sizeRatio));
        const expectedRelation = sizeRatio < 1 ? OCTREE_POWER_TRANSFER_RELATION.finer
          : sizeRatio > 1 ? OCTREE_POWER_TRANSFER_RELATION.coarser
            : OCTREE_POWER_TRANSFER_RELATION.same;
        assert.equal(packed.transferRelation, expectedRelation);
        if (expectedRelation === OCTREE_POWER_TRANSFER_RELATION.finer) finerSlots += 1;
        if (expectedRelation === OCTREE_POWER_TRANSFER_RELATION.coarser) coarserSlots += 1;
      }
      const normal = catalog.faceData.slice(face + 8, face + 11);
      const reconstruction = catalog.reconstructionData.slice(
        slot * OCTREE_POWER_RECONSTRUCTION_FLOATS,
        (slot + 1) * OCTREE_POWER_RECONSTRUCTION_FLOATS,
      );
      for (let row = 0; row < 3; row += 1) for (let column = 0; column < 3; column += 1) {
        identity[row * 3 + column] += reconstruction[row] * normal[column];
      }
    }
    assert.equal(bits(catalog.rowTemplateDiagonals[entry]), bits(diagonal));
    for (let row = 0; row < 3; row += 1) for (let column = 0; column < 3; column += 1) {
      assert.ok(Math.abs(identity[row * 3 + column] - (row === column ? 1 : 0)) < 2e-6,
        `dense case ${entry} reconstruction ${row},${column}`);
    }
    assert.equal(
      (flags & OCTREE_POWER_ROW_TEMPLATE_FLAGS.physicalBoundary) !== 0,
      entryBoundarySlots > 0,
    );
    if (entry === 0) assert.equal(flags, OCTREE_POWER_ROW_TEMPLATE_FLAGS.regularInterior);
  }
  assert.ok(extraFamilySlots > 0);
  assert.ok(boundarySlots > 0);
  assert.ok(finerSlots > 0);
  assert.ok(coarserSlots > 0);
});

function transformBoundaryMask(mask: number, transform: number): number {
  const directions = OCTREE_POWER_NEIGHBOR_DIRECTIONS.slice(0, 6);
  const index = new Map(directions.map((direction, bit) => [direction.join(","), bit]));
  return directions.reduce((result, direction, bit) => (mask & (1 << bit)) === 0 ? result
    : result | (1 << index.get(transformPowerVector(direction, OCTREE_CUBE_TRANSFORMS[transform]).join(","))!), 0);
}

test("boundary quotient contains exact axis planes and preserves the interpolation tetrahedra", () => {
  const bytes = readFileSync(catalogUrl);
  const catalog = decodeGeneratedOctreePowerCatalog(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
  const faceDirections = OCTREE_POWER_NEIGHBOR_DIRECTIONS.slice(0, 6);
  for (let lookupIndex = 0; lookupIndex < catalog.lookup.length / 3; lookupIndex += 1) {
    const key = catalog.lookup[lookupIndex * 3];
    const baseEntry = Math.floor(key / 64);
    const boundaryMask = key & 63;
    const boundaryEntry = catalog.lookup[lookupIndex * 3 + 1];
    assert.ok(boundaryMask > 0);
    const firstFace = catalog.entryHeaders[boundaryEntry * 2];
    const faceCount = catalog.entryHeaders[boundaryEntry * 2 + 1];
    const worldFaces: number[] = [];
    for (let localFace = 0; localFace < faceCount; localFace += 1) {
      const offset = (firstFace + localFace) * 12;
      const neighbor = [...catalog.faceData.slice(offset, offset + 3)];
      const sizeRatio = catalog.faceData[offset + 3];
      const centroid = [...catalog.faceData.slice(offset + 5, offset + 8)];
      const normal = [...catalog.faceData.slice(offset + 8, offset + 11)];
      if (sizeRatio === 0) {
        const bit = faceDirections.findIndex((direction) => direction.every((value, axis) => value === normal[axis]));
        assert.ok(bit >= 0 && (boundaryMask & (1 << bit)) !== 0, `entry ${boundaryEntry} has undeclared world normal`);
        assert.ok(Math.abs(normal.reduce((sum, value, axis) => sum + value * centroid[axis], 0) - 0.5) < 2e-6);
        assert.ok(Math.abs(catalog.faceData[offset + 11] - 2) < 2e-6);
        worldFaces.push(bit);
      } else {
        for (let bit = 0; bit < 6; bit += 1) if ((boundaryMask & (1 << bit)) !== 0) {
          const normal = faceDirections[bit];
          const furthestBoxPoint = normal.reduce((sum, value, axis) => sum + value * neighbor[axis], 0) + sizeRatio / 2;
          assert.ok(furthestBoxPoint <= 0.5 + 2e-6, `entry ${boundaryEntry} retains an exterior virtual site`);
        }
      }
    }
    assert.equal(new Set(worldFaces).size, boundaryMask.toString(2).replaceAll("0", "").length);
    const baseTetra = [...catalog.tetrahedronHeaders.slice(baseEntry * 3, baseEntry * 3 + 3)];
    const boundaryTetra = [...catalog.tetrahedronHeaders.slice(boundaryEntry * 3, boundaryEntry * 3 + 3)];
    assert.deepEqual(boundaryTetra.slice(1), baseTetra.slice(1));
    assert.deepEqual(
      [...catalog.tetrahedronData.slice(boundaryTetra[0], boundaryTetra[0] + boundaryTetra[1])],
      [...catalog.tetrahedronData.slice(baseTetra[0], baseTetra[0] + baseTetra[1])],
    );
  }
});

test("same/coarser index-16 x-minus witness uses the true plane and exact clipped volume", () => {
  const bytes = readFileSync(catalogUrl);
  const catalog = decodeGeneratedOctreePowerCatalog(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
  const rawDescriptor = 0x8000_0010;
  const packed = catalog.sameOrCoarserDirect[16];
  const baseEntry = packed & 0xffff;
  const canonicalMask = transformBoundaryMask(1, packed >>> 16);
  const key = baseEntry * 64 + canonicalMask;
  let low = 0, high = catalog.lookup.length / 3;
  while (low < high) {
    const middle = low + Math.floor((high - low) / 2);
    if (catalog.lookup[middle * 3] < key) low = middle + 1; else high = middle;
  }
  assert.equal(catalog.lookup[low * 3], key);
  const entry = catalog.lookup[low * 3 + 1];
  const sites = sitesForSameOrCoarserPowerDescriptor(rawDescriptor);
  const anchor = sites.find((site) => site.key === "anchor")!;
  const geometrySites = sites.filter((site) => site.key === anchor.key || site.origin[0] >= anchor.origin[0]);
  const oracle = constructOctreePowerCell(anchor, geometrySites, [{ key: "x-", normal: [-1, 0, 0], offset: -anchor.origin[0] }]);
  assert.ok(Math.abs(catalog.entryVolumes[entry] - oracle.volume / anchor.size ** 3) < 2e-6);
  const firstFace = catalog.entryHeaders[entry * 2];
  const faceCount = catalog.entryHeaders[entry * 2 + 1];
  const world = Array.from({ length: faceCount }, (_, localFace) => (firstFace + localFace) * 12)
    .filter((offset) => catalog.faceData[offset + 3] === 0);
  assert.equal(world.length, 1);
  const normal = [...catalog.faceData.slice(world[0] + 8, world[0] + 11)];
  const centroid = [...catalog.faceData.slice(world[0] + 5, world[0] + 8)];
  assert.equal(normal.filter((value) => Math.abs(value) > 0.9999).length, 1);
  assert.ok(Math.abs(normal.reduce((sum, value, axis) => sum + value * centroid[axis], 0) - 0.5) < 2e-6);
});
