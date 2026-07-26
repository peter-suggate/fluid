import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { decodeGeneratedOctreePowerCatalog } from "../lib/generated/octree-power-catalog";
import { planOctreePowerCoarseLevelSet } from "../lib/webgpu-octree-power-coarse-levelset";
import {
  OCTREE_AIR_SUPPORT_CONTROL_WORDS,
  OCTREE_AIR_SUPPORT_INVALID,
  OCTREE_AIR_SUPPORT_LAYOUT_VERSION,
  OCTREE_AIR_SUPPORT_MAXIMUM_CASE_SELECTORS,
  OCTREE_AIR_SUPPORT_TAG,
  OCTREE_AIR_SUPPORT_VALID,
  catalogMaximumOctreeAirSupportSelectors,
  decodeOctreeAirSupportControl,
  decodeOctreeVelocityTag,
  encodeOctreeAirSupportTag,
  enumerateOctreeAirSupportCandidates,
  inverseOctreePowerTransform,
  octreeAirSupportTetraWeights,
  planOctreeAirVelocitySupport,
  positiveOctreeAirSupportSelectors,
} from "../lib/webgpu-octree-air-velocity-support";
import { octreeAirVelocitySupportPublicationWGSL } from "../lib/webgpu-octree-air-velocity-support-gpu";

const bytes = readFileSync(new URL("../lib/generated/octree-power-catalog.bin", import.meta.url));
const catalog = decodeGeneratedOctreePowerCatalog(
  bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
const coarseSource = readFileSync(new URL("../lib/webgpu-octree-power-coarse-levelset.ts", import.meta.url), "utf8");
const dimensions = [24, 18, 16] as const;
const cell = (x: number, y: number, z: number) => x + dimensions[0] * (y + dimensions[1] * z);
const vertex = (selector: number) => Array.from(
  catalog.tetrahedronVertexData.slice(4 * selector, 4 * selector + 3),
) as [number, number, number];

test("air-support arena is suffix-only and its tags fail closed", () => {
  const layout = planOctreeAirVelocitySupport(4096, 4096 * 30);
  assert.equal(layout.transportMetricBytes, 4096 * 30 * 16);
  assert.equal(layout.selectorTagOffsetBytes, layout.transportMetricBytes);
  assert.equal(layout.selectorTagOffsetWords, 4096 * 30 * 4,
    "the existing selector writer retains the exact transport-prefix offset");
  assert.equal(layout.supportCapacity, 4096 * OCTREE_AIR_SUPPORT_MAXIMUM_CASE_SELECTORS);
  assert.equal(layout.regularTagBytes, 4096 * 27 * 4);
  assert.equal(layout.controlBytes, OCTREE_AIR_SUPPORT_CONTROL_WORDS * 4);
  assert.equal(layout.totalBytes % 256, 0);
  assert.ok(layout.regularTagOffsetBytes >= layout.selectorTagOffsetBytes + layout.selectorTagBytes);
  assert.ok(layout.supportVectorOffsetBytes >= layout.controlOffsetBytes + layout.controlBytes);

  const tag = encodeOctreeAirSupportTag(17);
  assert.equal(tag, (OCTREE_AIR_SUPPORT_TAG | 17) >>> 0);
  assert.deepEqual(decodeOctreeVelocityTag(7, 10, 18), { kind: "published-row", row: 7 });
  assert.deepEqual(decodeOctreeVelocityTag(tag, 10, 18), { kind: "support", support: 17 });
  assert.deepEqual(decodeOctreeVelocityTag(10, 10, 18), { kind: "invalid" });
  assert.deepEqual(decodeOctreeVelocityTag(encodeOctreeAirSupportTag(18), 10, 18), { kind: "invalid" });
  assert.deepEqual(decodeOctreeVelocityTag(OCTREE_AIR_SUPPORT_INVALID, 10, 18), { kind: "invalid" });
});

test("air-support publication exclusively owns selector and regular tags", () => {
  const layout = planOctreeAirVelocitySupport(32, 960);
  const selectorBaseBytes = layout.selectorTagOffsetBytes + layout.selectorTagBytes;
  const plan = planOctreePowerCoarseLevelSet(32, layout.transportMetricBytes,
    layout.totalBytes - selectorBaseBytes);
  assert.equal(plan.selectorOffsetWords, layout.selectorTagOffsetWords);
  assert.equal(plan.selectorRowBytes, layout.totalBytes);
  assert.match(coarseSource,
    /if \(!this\.airSupportLayout\) runRows\(this\.buildSelectorRowsPipeline, 1, 12\)/,
    "a support-enabled schedule cannot expose a mixed-generation tag table");
  assert.match(coarseSource,
    /regularTagOffsetWords\+27u\*row\+stencil[\s\S]*selectorRows\[output\]=findSite\(center\(row\)\+size\(row\)\*offset,size\(row\)\)/,
    "standalone coarse publication retains its exact direct-row adjacency path");
  assert.doesNotMatch(coarseSource, /selectorRows\[output\]=(?:row|0u);/,
    "a missing in-domain identity must never become an owner or zero-row fallback");
});

test("a rejected topology candidate refreshes support from the retained accepted epoch", () => {
  assert.doesNotMatch(octreeAirVelocitySupportPublicationWGSL,
    /accepted\.epoch!=p\.expectedEpoch/,
    "the host attempt stamp must not invalidate a clean GPU-retained topology");
  assert.match(octreeAirVelocitySupportPublicationWGSL,
    /ownerPageArena\[\d+u\]!=accepted\.epoch/,
    "owner pages must match the GPU-accepted epoch");
  assert.match(octreeAirVelocitySupportPublicationWGSL,
    /boundary!=accepted\.epoch/,
    "boundary support must match the same GPU-accepted epoch");
});

test("case 7949 selector 67 resolves to a distinct in-domain +z support row", () => {
  assert.deepEqual(vertex(67), [1, 0, 0]);
  assert.deepEqual(inverseOctreePowerTransform([1, 0, 0], 38), [0, 0, 1]);
  const candidates = enumerateOctreeAirSupportCandidates({
    catalog, caseId: 7949, transformCode: 38,
    anchor: { cell: cell(3, 3, 7), size: 1 }, dimensions,
    published: [{ cell: cell(3, 3, 7), size: 1, row: 319 }],
  });
  const selector = candidates.find((candidate) => candidate.selector === 67);
  assert.deepEqual(selector, {
    selector: 67,
    canonicalOffset: [1, 0, 0],
    worldOffset: [0, 0, 1],
    sizeRatio: 1,
    disposition: "support",
    identity: { cell: cell(3, 3, 8), size: 1 },
    publishedRow: null,
  });

  const first = catalog.tetrahedronHeaders[7949 * 3]!;
  const packed = catalog.tetrahedronData[first + 23]!;
  assert.deepEqual([packed & 0xff, (packed >>> 8) & 0xff, (packed >>> 16) & 0xff], [65, 67, 68]);
  const weights = octreeAirSupportTetraWeights([0.5, 0, 0], vertex(65), vertex(67), vertex(68));
  assert.deepEqual(weights, [0.5, 0, 0.5, 0]);
  assert.deepEqual(positiveOctreeAirSupportSelectors(packed, weights), [67],
    "the support vector is a genuine half-weight contributor, never an owner substitute");
});

test("case 7955 selector 68 is an exact zero-weight control", () => {
  assert.deepEqual(Array.from(catalog.tetrahedronVertexData.slice(4 * 68, 4 * 68 + 4)), [1, 0, 1, 1]);
  const first = catalog.tetrahedronHeaders[7955 * 3]!;
  const packed = catalog.tetrahedronData[first + 25]!;
  assert.deepEqual([packed & 0xff, (packed >>> 8) & 0xff, (packed >>> 16) & 0xff], [65, 67, 68]);
  const weights = octreeAirSupportTetraWeights([0.5, 0, 0], vertex(65), vertex(67), vertex(68));
  assert.deepEqual(weights, [0.5, 0, 0.5, 0]);
  assert.deepEqual(positiveOctreeAirSupportSelectors(packed, weights), [67]);
  assert.ok(!positiveOctreeAirSupportSelectors(packed, weights).includes(68),
    "a zero-weight selector must not require support publication");
});

test("all generated cases fit the deterministic 36-selector support bound", () => {
  assert.deepEqual(catalogMaximumOctreeAirSupportSelectors(catalog), { maximum: 36, caseId: 1073 });
});

test("physical exterior remains distinct from missing in-domain support", () => {
  const inDomain = enumerateOctreeAirSupportCandidates({
    catalog, caseId: 7949, transformCode: 38,
    anchor: { cell: cell(3, 3, 7), size: 1 }, dimensions,
  }).find((candidate) => candidate.selector === 67);
  const exterior = enumerateOctreeAirSupportCandidates({
    catalog, caseId: 7949, transformCode: 38,
    anchor: { cell: cell(3, 3, 15), size: 1 }, dimensions,
  }).find((candidate) => candidate.selector === 67);
  assert.equal(inDomain?.disposition, "support");
  assert.equal(exterior?.disposition, "exterior");
  assert.equal(exterior?.identity, null,
    "only physical exterior may use the separately defined boundary extension");
});

test("support control decoder accepts only complete generation-coherent publications", () => {
  const layout = planOctreeAirVelocitySupport(32, 960);
  const valid = new Uint32Array([
    0, OCTREE_AIR_SUPPORT_INVALID, 9, 1, 9, 20, 3, layout.supportCapacity,
    4, 8, 18, 2, 3, OCTREE_AIR_SUPPORT_VALID, OCTREE_AIR_SUPPORT_LAYOUT_VERSION, 0,
  ]);
  assert.deepEqual(decodeOctreeAirSupportControl(valid, layout, {
    structuredEpoch: 9, structuredBank: 1, boundaryEpoch: 9, pressureRowCount: 20,
  }), {
    publication: {
      structuredEpoch: 9, structuredBank: 1, boundaryEpoch: 9, pressureRowCount: 20,
      supportCount: 3, selectorReferenceCount: 4, regularReferenceCount: 8,
      faceCount: 18, seedFaceCount: 2, marchDepth: 3, fineGeneration: 0,
    },
    blocker: null,
  });

  for (const mutate of [
    (words: Uint32Array) => { words[0] = 1; },
    (words: Uint32Array) => { words[1] = 7; },
    (words: Uint32Array) => { words[2] = 8; },
    (words: Uint32Array) => { words[6] = 3; words[8] = 0; },
    (words: Uint32Array) => { words[6] = words[7]! + 1; },
    (words: Uint32Array) => { words[11] = words[10]! + 1; },
    (words: Uint32Array) => { words[13] = 0; },
    (words: Uint32Array) => { words[14] = 0; },
  ]) {
    const rejected = valid.slice(); mutate(rejected);
    assert.equal(decodeOctreeAirSupportControl(rejected, layout, {
      structuredEpoch: 9, structuredBank: 1, boundaryEpoch: 9, pressureRowCount: 20,
    }).publication, null);
  }
  assert.equal(decodeOctreeAirSupportControl(valid, layout, { fineGeneration: 1 }).publication, null,
    "diagnostics must reject an air-support transaction for a different fine generation");
  assert.equal(decodeOctreeAirSupportControl(valid.slice(0, 15), layout).publication, null);
});
