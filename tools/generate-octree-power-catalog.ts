import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import {
  buildOctreePowerCatalog,
  OCTREE_POWER_CATALOG_FACE_FLOATS,
  OCTREE_POWER_RECONSTRUCTION_FLOATS,
  OCTREE_POWER_ROW_TEMPLATE_FLOATS,
  OCTREE_POWER_ROW_TEMPLATE_HEADER_WORDS,
  OCTREE_POWER_ROW_TEMPLATE_VERSION,
  type OctreePowerTopologyConfiguration,
} from "../lib/octree-power-catalog";
import {
  OCTREE_POWER_NEIGHBOR_DIRECTIONS,
  OCTREE_POWER_SAME_OR_COARSER_FLAG,
  OCTREE_POWER_SAME_OR_FINER_MASK,
  canonicalizeSameOrFinerPowerDescriptor,
  encodeSameOrCoarserPowerDescriptor,
  enumerateCanonicalSameOrFinerPowerDescriptors,
  sitesForSameOrCoarserPowerDescriptor,
  sitesForSameOrFinerPowerDescriptor,
} from "../lib/octree-power-descriptor";
import {
  OCTREE_CUBE_TRANSFORMS,
  composeCubeTransforms,
  transformPowerVector,
} from "../lib/octree-power-topology";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const outputDirectory = join(root, "lib", "generated");
const binaryPath = join(outputDirectory, "octree-power-catalog.bin");
const modulePath = join(outputDirectory, "octree-power-catalog.ts");
const MAGIC = 0x504f_5743;
const HEADER_WORDS = 40;
const GENERATOR_VERSION = 9;
const COEFFICIENT_CHANNELS = 19;
const INTERPOLATION_SELECTOR_VERSION = 1;
const TRANSFER_SELECTOR_VERSION = 1;

const interiorConfigurations: OctreePowerTopologyConfiguration[] = enumerateCanonicalSameOrFinerPowerDescriptors().map((descriptor) => ({
  descriptor,
  anchorKey: "0,0,0/2",
  sites: sitesForSameOrFinerPowerDescriptor(descriptor),
}));
for (let child = 0; child < 8; child += 1) for (let mask = 0; mask < 64; mask += 1) {
  const descriptor = encodeSameOrCoarserPowerDescriptor({
    child: [child & 1, (child >> 1) & 1, (child >> 2) & 1] as [0 | 1, 0 | 1, 0 | 1],
    coarseNeighbors: [0, 1, 2, 3, 4, 5].map((bit) => (mask & (1 << bit)) !== 0) as [boolean, boolean, boolean, boolean, boolean, boolean],
  });
  interiorConfigurations.push({ descriptor, anchorKey: "anchor", sites: sitesForSameOrCoarserPowerDescriptor(descriptor) });
}

// Build the interior quotient first. Boundary rows use this entry and its
// world-to-canonical transform to address only the reachable (entry,mask)
// quotient, rather than multiplying the catalog by all 64 masks.
const interiorCatalog = buildOctreePowerCatalog(interiorConfigurations);
const interiorLookup = new Map(interiorCatalog.lookup.map((record) => [record.descriptor, record]));
const faceDirections = OCTREE_POWER_NEIGHBOR_DIRECTIONS.slice(0, 6);
const directionIndex = new Map(faceDirections.map((direction, index) => [direction.join(","), index]));
const transformBoundaryMask = (mask: number, transformCode: number): number => {
  let transformed = 0;
  for (let bit = 0; bit < 6; bit += 1) if ((mask & (1 << bit)) !== 0) {
    const direction = transformPowerVector(faceDirections[bit], OCTREE_CUBE_TRANSFORMS[transformCode]);
    const target = directionIndex.get(direction.join(","));
    if (target === undefined) throw new Error("Cube transform did not preserve a boundary direction");
    transformed |= 1 << target;
  }
  return transformed;
};
const validBoundaryMasks = Array.from({ length: 63 }, (_, index) => index + 1).filter((mask) =>
  !([[0, 5], [1, 4], [2, 3]] as const).some(([negative, positive]) =>
    (mask & (1 << negative)) !== 0 && (mask & (1 << positive)) !== 0));
const requiredSameOrFinerBits = new Map(validBoundaryMasks.map((mask) => {
  let required = 0;
  OCTREE_POWER_NEIGHBOR_DIRECTIONS.forEach((direction, bit) => {
    if (faceDirections.some((face, faceBit) => (mask & (1 << faceBit)) !== 0
      && direction.some((value, axis) => value !== 0 && value === face[axis]))) required |= 1 << bit;
  });
  return [mask, required] as const;
}));
const baseRecord = (descriptor: number) => {
  if ((descriptor & OCTREE_POWER_SAME_OR_COARSER_FLAG) !== 0) {
    const record = interiorLookup.get(descriptor >>> 0);
    if (!record) throw new Error(`Missing same/coarser interior descriptor ${descriptor}`);
    return record;
  }
  const canonical = canonicalizeSameOrFinerPowerDescriptor(descriptor);
  const record = interiorLookup.get(canonical.descriptor);
  if (!record) throw new Error(`Missing same/finer interior descriptor ${canonical.descriptor}`);
  const combined = composeCubeTransforms(OCTREE_CUBE_TRANSFORMS[canonical.transform], OCTREE_CUBE_TRANSFORMS[record.transform]);
  return { descriptor, entry: record.entry, transform: combined.code };
};
const boundaryPlanes = (anchor: OctreePowerTopologyConfiguration["sites"][number], mask: number) =>
  faceDirections.flatMap((normal, bit) => {
    if ((mask & (1 << bit)) === 0) return [];
    const axis = normal.findIndex((value) => value !== 0);
    const coordinate = normal[axis] < 0 ? anchor.origin[axis] : anchor.origin[axis] + anchor.size;
    return [{ key: `world:${bit}`, normal, offset: normal[axis] * coordinate }];
  });
const boundaryConfigurations: OctreePowerTopologyConfiguration[] = [];
const boundaryPairByKey = new Map<string, { descriptor: number; rawDescriptor: number; rawMask: number }>();
const addBoundaryConfiguration = (rawDescriptor: number, rawMask: number, anchorKey: string,
  sites: OctreePowerTopologyConfiguration["sites"]) => {
  const base = baseRecord(rawDescriptor);
  const canonicalMask = transformBoundaryMask(rawMask, base.transform);
  const pairKey = `${base.entry}:${canonicalMask}`;
  if (boundaryPairByKey.has(pairKey)) return;
  const anchor = sites.find((site) => site.key === anchorKey);
  if (!anchor) throw new Error(`Boundary descriptor ${rawDescriptor} has no anchor`);
  const boundaries = boundaryPlanes(anchor, rawMask);
  const geometrySites = sites.filter((site) => site.key === anchor.key || boundaries.every((plane) => {
    const axis = plane.normal.findIndex((value) => value !== 0);
    const maximum = plane.normal[axis] < 0 ? plane.normal[axis] * site.origin[axis]
      : plane.normal[axis] * (site.origin[axis] + site.size);
    return maximum <= plane.offset;
  }));
  const descriptor = (0x4000_0000 + boundaryConfigurations.length) >>> 0;
  boundaryPairByKey.set(pairKey, { descriptor, rawDescriptor, rawMask });
  boundaryConfigurations.push({ descriptor, anchorKey, sites, geometrySites, boundaries });
};
for (let descriptor = 0; descriptor <= OCTREE_POWER_SAME_OR_FINER_MASK; descriptor += 1) {
  for (const mask of validBoundaryMasks) {
    const required = requiredSameOrFinerBits.get(mask)!;
    if ((descriptor & required) !== required) continue;
    addBoundaryConfiguration(descriptor, mask, "0,0,0/2", sitesForSameOrFinerPowerDescriptor(descriptor));
  }
}
for (let low = 0; low < 512; low += 1) {
  const descriptor = (OCTREE_POWER_SAME_OR_COARSER_FLAG | low) >>> 0;
  const child = [low & 1, (low >> 1) & 1, (low >> 2) & 1];
  const outward = child.map((bit) => bit === 0 ? -1 : 1);
  const coarseDirections = [
    [outward[0], 0, 0], [0, outward[1], 0], [0, 0, outward[2]],
    [outward[0], outward[1], 0], [outward[0], 0, outward[2]], [0, outward[1], outward[2]],
  ];
  for (const mask of validBoundaryMasks) {
    // A size-one child adjacent to a dyadic root boundary must lie on the
    // corresponding outward side of its size-two parent. Other parity/mask
    // combinations cannot occur in the graded octree and can remove the only
    // bounding neighbor when interpreted as physical geometry.
    const boundaryParityValid = faceDirections.every((face, faceBit) => (mask & (1 << faceBit)) === 0
      || face.every((value, axis) => value === 0 || value === outward[axis]));
    if (!boundaryParityValid) continue;
    const crossesBoundary = coarseDirections.some((direction, coarseBit) => (low & (1 << (coarseBit + 3))) !== 0
      && faceDirections.some((face, faceBit) => (mask & (1 << faceBit)) !== 0
        && direction.some((value, axis) => value !== 0 && value === face[axis])));
    if (crossesBoundary) continue;
    addBoundaryConfiguration(descriptor, mask, "anchor", sitesForSameOrCoarserPowerDescriptor(descriptor));
  }
}

const catalog = buildOctreePowerCatalog([...interiorConfigurations, ...boundaryConfigurations]);
if (catalog.manifest.regularCaseId !== 0) {
  throw new Error("Generated dense power case zero must be the regular Cartesian 7-point row");
}
const lookupByDescriptor = new Map(catalog.lookup.map((record) => [record.descriptor, record]));
const boundaryLookupRecords = [...boundaryPairByKey.values()].map((pair) => {
  const base = baseRecord(pair.rawDescriptor);
  const finalBaseDescriptor = (pair.rawDescriptor & OCTREE_POWER_SAME_OR_COARSER_FLAG) !== 0
    ? pair.rawDescriptor : canonicalizeSameOrFinerPowerDescriptor(pair.rawDescriptor).descriptor;
  const finalBase = lookupByDescriptor.get(finalBaseDescriptor >>> 0);
  const boundary = lookupByDescriptor.get(pair.descriptor);
  if (!finalBase || !boundary) throw new Error("Boundary quotient references a missing final catalog entry");
  return { key: (finalBase.entry * 64 + transformBoundaryMask(pair.rawMask, base.transform)) >>> 0, entry: boundary.entry };
}).sort((a, b) => a.key - b.key);
for (let index = 1; index < boundaryLookupRecords.length; index += 1) {
  if (boundaryLookupRecords[index - 1].key === boundaryLookupRecords[index].key) throw new Error("Duplicate boundary quotient lookup key");
}
const lookup = new Uint32Array(boundaryLookupRecords.length * 3);
boundaryLookupRecords.forEach((record, index) => lookup.set([record.key, record.entry, 0], index * 3));
// Rebuild descriptors are deliberately emitted in world orientation. These
// maps are publication-only: hot kernels retain the low-16-bit dense case ID
// and never perform descriptor lookup. Bits 16..21 carry the complete
// world-to-catalog transform used while stamping the resolved row.
const packLookup = (entry: number, transform: number) => {
  if (entry >= 0x1_0000 || transform >= 48) throw new Error("Power catalog direct lookup packing overflow");
  return (entry | (transform << 16)) >>> 0;
};
const sameOrFinerDirect = new Uint32Array(OCTREE_POWER_SAME_OR_FINER_MASK + 1);
sameOrFinerDirect.fill(0xffff_ffff);
for (let descriptor = 0; descriptor <= OCTREE_POWER_SAME_OR_FINER_MASK; descriptor += 1) {
  const canonical = canonicalizeSameOrFinerPowerDescriptor(descriptor);
  const record = lookupByDescriptor.get(canonical.descriptor);
  if (!record) throw new Error(`Missing canonical same/finer descriptor ${canonical.descriptor}`);
  const combined = composeCubeTransforms(OCTREE_CUBE_TRANSFORMS[canonical.transform], OCTREE_CUBE_TRANSFORMS[record.transform]);
  sameOrFinerDirect[descriptor] = packLookup(record.entry, combined.code);
}
const sameOrCoarserDirect = new Uint32Array(512);
sameOrCoarserDirect.fill(0xffff_ffff);
for (let descriptor = 0; descriptor < 512; descriptor += 1) {
  const record = lookupByDescriptor.get((OCTREE_POWER_SAME_OR_COARSER_FLAG | descriptor) >>> 0);
  if (record) sameOrCoarserDirect[descriptor] = packLookup(record.entry, record.transform);
}
// Paper Section 6.3: channel zero is the canonical diagonal and channels
// 1..18 are the six face and twelve edge directions. A power cell can expose
// as many as thirty geometric facets, but several facets address the same
// native-level neighbour/ghost. Aggregate those facets here; truncating to the
// first eighteen was a resolved-row convention and is not a valid §6.3 table.
const section63Directions = [
  [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1],
  [1, 1, 0], [1, -1, 0], [-1, 1, 0], [-1, -1, 0],
  [1, 0, 1], [1, 0, -1], [-1, 0, 1], [-1, 0, -1],
  [0, 1, 1], [0, 1, -1], [0, -1, 1], [0, -1, -1],
] as const;
const coefficientData = new Float32Array(catalog.entries.length * COEFFICIENT_CHANNELS);
for (let entry = 0; entry < catalog.entries.length; entry += 1) {
  const firstFace = catalog.entryHeaders[entry * 2];
  const faceCount = catalog.entryHeaders[entry * 2 + 1];
  let diagonal = Math.fround(0);
  for (let slot = 0; slot < faceCount; slot += 1) {
    const coefficient = catalog.rowTemplateData[
      (firstFace + slot) * OCTREE_POWER_ROW_TEMPLATE_FLOATS
    ];
    diagonal = Math.fround(diagonal + coefficient);
    const face = catalog.entries[entry]!.faces[slot]!;
    if (face.neighborSizeRatio !== 0) {
      const direction = face.centroid.map((value, axis) =>
        Math.round(value + 0.5 * face.normal[axis]!));
      const channel = section63Directions.findIndex((candidate) => candidate.every(
        (value, axis) => value === direction[axis]));
      if (channel < 0) throw new Error(`Catalog entry ${entry} face ${slot} has no Section 6.3 direction (${direction})`);
      const offset = entry * COEFFICIENT_CHANNELS + 1 + channel;
      coefficientData[offset] = Math.fround(coefficientData[offset]! + coefficient);
    }
  }
  coefficientData[entry * COEFFICIENT_CHANNELS] = diagonal;
}
const headerBytes = HEADER_WORDS * 4;
const entryHeadersOffset = headerBytes;
const entryVolumesOffset = entryHeadersOffset + catalog.entryHeaders.byteLength;
const faceDataOffset = entryVolumesOffset + catalog.entryVolumes.byteLength;
const lookupOffset = faceDataOffset + catalog.faceData.byteLength;
const sameOrFinerDirectOffset = lookupOffset + lookup.byteLength;
const sameOrCoarserDirectOffset = sameOrFinerDirectOffset + sameOrFinerDirect.byteLength;
const tetrahedronHeadersOffset = sameOrCoarserDirectOffset + sameOrCoarserDirect.byteLength;
const tetrahedronDataOffset = tetrahedronHeadersOffset + catalog.tetrahedronHeaders.byteLength;
const tetrahedronVertexDataOffset = tetrahedronDataOffset + catalog.tetrahedronData.byteLength;
const coefficientDataOffset = tetrahedronVertexDataOffset + catalog.tetrahedronVertexData.byteLength;
const rowTemplateHeadersOffset = coefficientDataOffset + coefficientData.byteLength;
const rowTemplateSlotsOffset = rowTemplateHeadersOffset + catalog.rowTemplateHeaders.byteLength;
const rowTemplateDataOffset = rowTemplateSlotsOffset + catalog.rowTemplateSlots.byteLength;
const rowTemplateDiagonalsOffset = rowTemplateDataOffset + catalog.rowTemplateData.byteLength;
const reconstructionDataOffset = rowTemplateDiagonalsOffset + catalog.rowTemplateDiagonals.byteLength;
const byteLength = reconstructionDataOffset + catalog.reconstructionData.byteLength;
const binary = new Uint8Array(byteLength);
const header = new Uint32Array(binary.buffer, 0, HEADER_WORDS);
header.set([
  MAGIC, catalog.manifest.version, catalog.entries.length, boundaryLookupRecords.length,
  catalog.faceData.length / OCTREE_POWER_CATALOG_FACE_FLOATS, catalog.manifest.maximumFaceIncidence, catalog.manifest.maximumNeighborRows, byteLength,
  entryHeadersOffset, entryVolumesOffset, faceDataOffset, lookupOffset,
  sameOrFinerDirectOffset, sameOrFinerDirect.length, sameOrCoarserDirectOffset, sameOrCoarserDirect.length,
  tetrahedronHeadersOffset, tetrahedronDataOffset, catalog.tetrahedronData.length,
  catalog.manifest.maximumTetrahedra, HEADER_WORDS,
  tetrahedronVertexDataOffset, catalog.tetrahedronVertexData.length / 4, 4,
  coefficientDataOffset, COEFFICIENT_CHANNELS,
  OCTREE_POWER_ROW_TEMPLATE_VERSION, rowTemplateHeadersOffset, OCTREE_POWER_ROW_TEMPLATE_HEADER_WORDS,
  rowTemplateSlotsOffset, catalog.rowTemplateSlots.length,
  rowTemplateDataOffset, OCTREE_POWER_ROW_TEMPLATE_FLOATS,
  rowTemplateDiagonalsOffset, reconstructionDataOffset, OCTREE_POWER_RECONSTRUCTION_FLOATS,
  catalog.manifest.regularCaseId, catalog.manifest.maximumDynamicBoundarySlots,
  INTERPOLATION_SELECTOR_VERSION, TRANSFER_SELECTOR_VERSION,
]);
binary.set(new Uint8Array(catalog.entryHeaders.buffer), entryHeadersOffset);
binary.set(new Uint8Array(catalog.entryVolumes.buffer), entryVolumesOffset);
binary.set(new Uint8Array(catalog.faceData.buffer), faceDataOffset);
binary.set(new Uint8Array(lookup.buffer), lookupOffset);
binary.set(new Uint8Array(sameOrFinerDirect.buffer), sameOrFinerDirectOffset);
binary.set(new Uint8Array(sameOrCoarserDirect.buffer), sameOrCoarserDirectOffset);
binary.set(new Uint8Array(catalog.tetrahedronHeaders.buffer), tetrahedronHeadersOffset);
binary.set(new Uint8Array(catalog.tetrahedronData.buffer), tetrahedronDataOffset);
binary.set(new Uint8Array(catalog.tetrahedronVertexData.buffer), tetrahedronVertexDataOffset);
binary.set(new Uint8Array(coefficientData.buffer), coefficientDataOffset);
binary.set(new Uint8Array(catalog.rowTemplateHeaders.buffer), rowTemplateHeadersOffset);
binary.set(new Uint8Array(catalog.rowTemplateSlots.buffer), rowTemplateSlotsOffset);
binary.set(new Uint8Array(catalog.rowTemplateData.buffer), rowTemplateDataOffset);
binary.set(new Uint8Array(catalog.rowTemplateDiagonals.buffer), rowTemplateDiagonalsOffset);
binary.set(new Uint8Array(catalog.reconstructionData.buffer), reconstructionDataOffset);

const hash = createHash("sha256");
for (const relative of [
  "lib/octree-power-geometry.ts", "lib/octree-power-topology.ts", "lib/octree-power-descriptor.ts",
  "lib/octree-power-catalog.ts", "tools/generate-octree-power-catalog.ts",
]) hash.update(readFileSync(join(root, relative)));
const generatorHash = hash.digest("hex");
const binarySha256 = createHash("sha256").update(binary).digest("hex");
const manifest = { ...catalog.manifest, descriptorCount: boundaryLookupRecords.length,
  byteCount: byteLength, generatorVersion: GENERATOR_VERSION, generatorHash, binarySha256 };
const moduleSource = `/** Generated by tools/generate-octree-power-catalog.ts; do not edit. */
export const OCTREE_GENERATED_POWER_CATALOG_MAGIC = 0x${MAGIC.toString(16)};
export const OCTREE_GENERATED_POWER_CATALOG_MANIFEST = Object.freeze(${JSON.stringify(manifest, null, 2)} as const);

export interface GeneratedOctreePowerCatalogViews {
  readonly entryHeaders: Uint32Array;
  readonly entryVolumes: Float32Array;
  readonly faceData: Float32Array;
  /** (interior entry * 64 + canonical boundary mask), boundary entry, zero. */
  readonly lookup: Uint32Array;
  /** Rebuild-only packed dense-case/transform indexed by the raw 18-bit descriptor. */
  readonly sameOrFinerDirect: Uint32Array;
  /** Rebuild-only packed dense-case/transform indexed by the low nine same/coarser bits. */
  readonly sameOrCoarserDirect: Uint32Array;
  /** first tetrahedron, count, flags triples indexed by catalog entry. */
  readonly tetrahedronHeaders: Uint32Array;
  /** Three catalog-face neighbor selectors packed into each u32. */
  readonly tetrahedronData: Uint32Array;
  /** Global byte-selector table: canonical offset xyz and size ratio. */
  readonly tetrahedronVertexData: Float32Array;
  /** Section 6.3: diagonal followed by eighteen canonical face-slot coefficients per entry. */
  readonly coefficientData: Float32Array;
  /** Versioned dense-case row templates: first slot, count, flags, dynamic-slot count. */
  readonly rowTemplateHeaders: Uint32Array;
  /** Packed neighbor/family/orientation/transfer/dynamic selectors for every template slot. */
  readonly rowTemplateSlots: Uint32Array;
  /** Normalized coefficient, area, inverse-distance triples for every template slot. */
  readonly rowTemplateData: Float32Array;
  /** Exact normalized diagonal indexed by dense case ID. */
  readonly rowTemplateDiagonals: Float32Array;
  /** xyz weighted least-squares pseudoinverse columns for every template slot. */
  readonly reconstructionData: Float32Array;
}

export function decodeGeneratedOctreePowerCatalog(data: ArrayBuffer): GeneratedOctreePowerCatalogViews {
  if (data.byteLength !== OCTREE_GENERATED_POWER_CATALOG_MANIFEST.byteCount) throw new RangeError("Generated power catalog byte count mismatch");
  const h = new Uint32Array(data, 0, ${HEADER_WORDS});
  if (h[0] !== OCTREE_GENERATED_POWER_CATALOG_MAGIC || h[1] !== OCTREE_GENERATED_POWER_CATALOG_MANIFEST.version) {
    throw new Error("Generated power catalog version mismatch");
  }
  if (h[2] !== OCTREE_GENERATED_POWER_CATALOG_MANIFEST.configurationCount
    || h[3] !== OCTREE_GENERATED_POWER_CATALOG_MANIFEST.descriptorCount || h[7] !== data.byteLength
    || h[13] !== ${OCTREE_POWER_SAME_OR_FINER_MASK + 1} || h[15] !== 512
    || h[19] !== OCTREE_GENERATED_POWER_CATALOG_MANIFEST.maximumTetrahedra || h[20] !== ${HEADER_WORDS}
    || h[22] > 256 || h[23] !== 4 || h[25] !== ${COEFFICIENT_CHANNELS}
    || h[26] !== ${OCTREE_POWER_ROW_TEMPLATE_VERSION} || h[28] !== ${OCTREE_POWER_ROW_TEMPLATE_HEADER_WORDS}
    || h[30] !== h[4] || h[32] !== ${OCTREE_POWER_ROW_TEMPLATE_FLOATS}
    || h[35] !== ${OCTREE_POWER_RECONSTRUCTION_FLOATS}
    || h[36] !== OCTREE_GENERATED_POWER_CATALOG_MANIFEST.regularCaseId
    || h[37] !== OCTREE_GENERATED_POWER_CATALOG_MANIFEST.maximumDynamicBoundarySlots
    || h[38] !== ${INTERPOLATION_SELECTOR_VERSION} || h[39] !== ${TRANSFER_SELECTOR_VERSION}) {
    throw new Error("Generated power catalog manifest mismatch");
  }
  return {
    entryHeaders: new Uint32Array(data, h[8], h[2] * 2),
    entryVolumes: new Float32Array(data, h[9], h[2]),
    faceData: new Float32Array(data, h[10], h[4] * ${OCTREE_POWER_CATALOG_FACE_FLOATS}),
    lookup: new Uint32Array(data, h[11], h[3] * 3),
    sameOrFinerDirect: new Uint32Array(data, h[12], h[13]),
    sameOrCoarserDirect: new Uint32Array(data, h[14], h[15]),
    tetrahedronHeaders: new Uint32Array(data, h[16], h[2] * 3),
    tetrahedronData: new Uint32Array(data, h[17], h[18]),
    tetrahedronVertexData: new Float32Array(data, h[21], h[22] * h[23]),
    coefficientData: new Float32Array(data, h[24], h[2] * h[25]),
    rowTemplateHeaders: new Uint32Array(data, h[27], h[2] * h[28]),
    rowTemplateSlots: new Uint32Array(data, h[29], h[30]),
    rowTemplateData: new Float32Array(data, h[31], h[30] * h[32]),
    rowTemplateDiagonals: new Float32Array(data, h[33], h[2]),
    reconstructionData: new Float32Array(data, h[34], h[30] * h[35]),
  };
}

/** Vite/Vinext rewrites this colocated URL into a fingerprinted browser asset. */
export async function fetchGeneratedOctreePowerCatalog(
  url: string | URL = new URL("./octree-power-catalog.bin", import.meta.url),
): Promise<GeneratedOctreePowerCatalogViews> {
  const response = await fetch(url);
  if (!response.ok) throw new Error(\`Failed to load generated power catalog: \${response.status}\`);
  return decodeGeneratedOctreePowerCatalog(await response.arrayBuffer());
}
`;

const check = process.argv.includes("--check");
if (check) {
  if (!existsSync(binaryPath) || !existsSync(modulePath)) {
    throw new Error("Generated octree power catalog is missing; run npm run generate:octree-power-catalog");
  }
  const existingBinary = readFileSync(binaryPath);
  const existingModule = readFileSync(modulePath, "utf8");
  if (!existingBinary.equals(binary) || existingModule !== moduleSource) {
    throw new Error("Generated octree power catalog is stale; run npm run generate:octree-power-catalog");
  }
} else {
  mkdirSync(outputDirectory, { recursive: true });
  writeFileSync(binaryPath, binary);
  writeFileSync(modulePath, moduleSource);
}

console.log(JSON.stringify(manifest, null, 2));
