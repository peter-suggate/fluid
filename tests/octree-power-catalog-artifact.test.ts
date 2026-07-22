import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import test from "node:test";
import {
  decodeGeneratedOctreePowerCatalog,
  OCTREE_GENERATED_POWER_CATALOG_MAGIC,
  OCTREE_GENERATED_POWER_CATALOG_MANIFEST,
} from "../lib/generated/octree-power-catalog";
import {
  OCTREE_POWER_NEIGHBOR_DIRECTIONS,
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
  const header = new Uint32Array(data, 0, 24);

  assert.equal(OCTREE_GENERATED_POWER_CATALOG_MANIFEST.generatorVersion, 4);
  assert.match(OCTREE_GENERATED_POWER_CATALOG_MANIFEST.generatorHash, /^[0-9a-f]{64}$/);
  assert.equal(OCTREE_GENERATED_POWER_CATALOG_MANIFEST.binarySha256, hash);
  assert.equal(header[0], OCTREE_GENERATED_POWER_CATALOG_MAGIC);
  assert.equal(header[1], OCTREE_GENERATED_POWER_CATALOG_MANIFEST.version);
  assert.equal(header[7], bytes.byteLength);
  assert.doesNotThrow(() => decodeGeneratedOctreePowerCatalog(data));
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
