#!/usr/bin/env node
import { writeFile } from "node:fs/promises";
import { performance } from "node:perf_hooks";
import { sceneLatticeDimensions } from "../lib/core/scene-lattice";
import { createMinimalPowerDamBreak64Scene, createOceanSeicheScene,
  createSparseCM12LongDamBreakScene, createSymmetricExpansionScene } from "../lib/core/scenes";
import type { SceneDescription } from "../lib/core/model";
import { buildSparseAtlasCompositeGrid } from
  "../lib/methods/adaptive-mass/sparse-atlas-composite-projection";
import { compileSparseCM12FactoredAEIPackedTemplateCatalog } from
  "../lib/methods/adaptive-mass/sparse-cm12-factored-aei-packed-template";
import { compileSparseCM12StableLeafFaceNeighbors } from
  "../lib/methods/adaptive-mass/sparse-cm12-factored-aei-topology";
import { compileSparseCM12InternedBoundaryOperators } from
  "../lib/methods/adaptive-mass/sparse-cm12-interned-boundary-operators";
import { createSparseCM12IboTRASupplement } from
  "../lib/methods/adaptive-mass/sparse-cm12-ibo-tra-supplement";
import { createSparseCM12InternedBoundaryImage } from
  "../lib/methods/adaptive-mass/sparse-cm12-interned-boundary-image";
import { compareSparseCM12IBOSemanticAuthority,
  compileSparseCM12GeometryFaceNeighbors } from
  "../lib/methods/adaptive-mass/sparse-cm12-ibo-semantic-authority";
import { createSparseCM12InternedRefLookup, sparseCM12InternedRefLookup } from
  "../lib/methods/adaptive-mass/sparse-cm12-interned-ref-lookup";
import { initializeSparseBrickAtlasFromScene, sparseBrickSpan } from
  "../lib/methods/adaptive-mass/sparse-brick-atlas";
import { adaptiveMassReceiverScaleForScene, dormantReceiverDomain,
  residentSupportAtlas } from
  "../lib/methods/adaptive-mass/webgpu-adaptive-mass-solver";
import { packSparseCM12AcceptedTopologyTemplatesForQA,
  packSparseCM12ResidentTopologyTemplatesForQA,
  sparseCM12HostTemplateVariantsEnabled } from
  "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident";

type SceneName = "ocean-seiche" | "mini64" | "long-dam" | "symmetric-expansion";
const argument = (name: string): string | undefined => process.argv.slice(2)
  .find((value) => value.startsWith(`--${name}=`))?.slice(name.length + 3);
const name = (argument("scene") ?? "mini64") as SceneName;
interface Lane { readonly build: () => SceneDescription; readonly brick: 8 | 16;
  readonly surface: number; readonly support: number; readonly floor: "auto" | 1 }
const lane: Lane = name === "ocean-seiche" ? {
  build: createOceanSeicheScene, brick: 16, surface: 1, support: 3, floor: 1,
} : name === "mini64" ? {
  build: createMinimalPowerDamBreak64Scene, brick: 8, surface: 8,
  support: 9, floor: "auto",
} : name === "long-dam" ? {
  build: createSparseCM12LongDamBreakScene, brick: 8, surface: 1,
  support: 9, floor: "auto",
} : name === "symmetric-expansion" ? {
  build: createSymmetricExpansionScene, brick: 8, surface: 1,
  support: 9, floor: "auto",
} : (() => { throw new RangeError("invalid scene"); })();
const scene = lane.build(), dimensions = sceneLatticeDimensions(scene);
const authored = initializeSparseBrickAtlasFromScene(scene, {
  finestDimensions: dimensions, brickFineResolution: lane.brick,
  surfaceFineRings: lane.surface });
const supported = residentSupportAtlas(authored, "adaptive");
const receiver = adaptiveMassReceiverScaleForScene(scene, lane.support);
const atlas = dormantReceiverDomain(supported, "adaptive", receiver.supportRings,
  lane.floor, receiver.minimumCapacityScale);
const grid = buildSparseAtlasCompositeGrid(atlas);
const mutableBrickCount = atlas.bricks.filter((brick) => sparseBrickSpan(brick) === 1).length;
const hostTemplateVariants = sparseCM12HostTemplateVariantsEnabled(
  grid.cells.length, grid.gradientRows.length, mutableBrickCount,
  atlas.brickFineResolution,
);
const begin = performance.now();
const packed = hostTemplateVariants
  ? packSparseCM12ResidentTopologyTemplatesForQA(atlas, grid)
  : packSparseCM12AcceptedTopologyTemplatesForQA(atlas, grid);
const packedMilliseconds = performance.now() - begin;
const validDimensions = (leaf: number, resolution: number) => {
  const brick = atlas.bricks[leaf]!;
  const width = atlas.brickFineResolution * sparseBrickSpan(brick), scale = width / resolution;
  return ([0, 1, 2] as const).map((axis) => {
    const origin = atlas.brickFineResolution * brick.coordinate[axis]!;
    return Math.min(resolution, Math.ceil(Math.max(0, Math.min(width,
      atlas.dimensions[axis]! - origin)) / scale));
  }) as [number, number, number];
};
const catalogBegin = performance.now();
const stableLeafFaceNeighbors = compileSparseCM12StableLeafFaceNeighbors({
  coordinates: atlas.bricks.map((brick) => brick.coordinate),
  spans: atlas.bricks.map((brick) => sparseBrickSpan(brick)),
});
const catalog = compileSparseCM12FactoredAEIPackedTemplateCatalog({
  words: packed.words, brickFineResolution: lane.brick,
  brickKeyByLeafId: atlas.bricks.map((brick) => brick.key),
  validDimensions: (leaf, resolution) => validDimensions(leaf, resolution),
  scaleLog2: (leaf, resolution) => Math.log2(
    atlas.brickFineResolution * sparseBrickSpan(atlas.bricks[leaf]!) / resolution),
  selectedResolution: (leaf) => atlas.bricks[leaf]!.resolution,
  neighborLeavesByLeaf: stableLeafFaceNeighbors,
});
const catalogMilliseconds = performance.now() - catalogBegin;
const supportedKeys = new Set(supported.bricks.map((brick) => brick.key));
const activeLeaves = atlas.bricks.flatMap((brick, leaf) =>
  supportedKeys.has(brick.key) ? [leaf] : []);
const compileBegin = performance.now();
const compiled = compileSparseCM12InternedBoundaryOperators({ catalog,
  packedWords: packed.words, activeLeaves });
const refLookup = createSparseCM12InternedRefLookup({ ibo: compiled,
  baseWords: compiled.layout.immutableWords });
let refLookupExact = true;
for (const patch of catalog.patches) {
  const observed = sparseCM12InternedRefLookup({ lookup: refLookup,
    sourceCanonicalId: patch.sourceCanonicalId, side: patch.sourceSide,
    targetCanonicalId: patch.targetCanonicalId });
  const expected = [compiled.templateIdByPatch[patch.id]!, patch.targetLeaf,
    compiled.rowBaseByPatch[patch.id]!] as const;
  refLookupExact &&= observed?.every((value, index) => value === expected[index]) ?? false;
}
const traSupplement = createSparseCM12IboTRASupplement({ ibo: compiled,
  baseWords: refLookup.layout.totalWords });
const geometryNeighbors = compileSparseCM12GeometryFaceNeighbors({
  coordinates: atlas.bricks.map((brick) => brick.coordinate),
  spans: atlas.bricks.map((brick) => sparseBrickSpan(brick)),
});
const geometryBaseWords = Math.ceil(refLookup.layout.totalWords / 64) * 64;
const sharedImage = createSparseCM12InternedBoundaryImage(compiled, activeLeaves, 1,
  catalog.descriptorIdByLeaf, [
    { label: "IRL1", baseWords: refLookup.layout.baseWords,
      words: refLookup.words },
    { label: "IGN1", baseWords: geometryBaseWords, words: geometryNeighbors.words },
  ]);
const semanticAuthority = compareSparseCM12IBOSemanticAuthority({ image: sharedImage,
  packedWords: packed.words, activeLeaves,
  descriptorIdByLeaf: catalog.descriptorIdByLeaf,
  leaves: activeLeaves, slot: 0 });
const geometryNeighborExact = catalog.neighborLeavesByLeaf.every((expected, leaf) => {
  const observed = [...geometryNeighbors.neighbors.subarray(
    geometryNeighbors.offsets[leaf]!, geometryNeighbors.offsets[leaf + 1]!)];
  return expected.length === observed.length
    && expected.every((value, index) => value === observed[index]);
});
const compileMilliseconds = performance.now() - compileBegin;
const selectedStablePackets = activeLeaves.reduce((sum, leaf) => {
  const descriptor = catalog.canonical[catalog.descriptorIdByLeaf[leaf]!]!;
  return sum + descriptor.validDimensions.reduce((product, extent) =>
    product * Math.ceil(extent / 4), 1);
}, 0);
const boundaryVisitsByPacket = new Map<number, number>();
const boundaryVisitsByPacketLane = new Map<number, number>();
const supplementWord = (absolute: number) => traSupplement.words[
  absolute - traSupplement.layout.baseWords]!;
for (const instance of compiled.instances) {
  const template = compiled.templates[instance.templateId]!;
  const directory = traSupplement.layout.directoryBaseWords + 4 * instance.templateId;
  const offsetsBase = supplementWord(directory), resolution = supplementWord(directory + 3);
  const axis = instance.side >>> 1, dimensions = template.sourceDimensions;
  const packetAxis = Math.ceil(resolution / 4);
  for (let boundary = 0; boundary < resolution ** 2; boundary += 1) {
    const u = boundary % resolution, v = Math.floor(boundary / resolution);
    const local = axis === 0
      ? [instance.side & 1 ? dimensions[0] - 1 : 0, u, v]
      : axis === 1 ? [u, instance.side & 1 ? dimensions[1] - 1 : 0, v]
      : [u, v, instance.side & 1 ? dimensions[2] - 1 : 0];
    if (local.some((value, localAxis) => value < 0 || value >= dimensions[localAxis]!)) continue;
    const visits = supplementWord(offsetsBase + boundary + 1)
      - supplementWord(offsetsBase + boundary);
    if (visits === 0) continue;
    const localPacket = Math.floor(local[0]! / 4) + packetAxis
      * (Math.floor(local[1]! / 4) + packetAxis * Math.floor(local[2]! / 4));
    const packet = instance.sourceLeaf * 64 + localPacket;
    boundaryVisitsByPacket.set(packet, (boundaryVisitsByPacket.get(packet) ?? 0) + visits);
    const lane = local[0]! % 4 + 4 * (local[1]! % 4) + 16 * (local[2]! % 4);
    const packetLane = packet * 64 + lane;
    boundaryVisitsByPacketLane.set(packetLane,
      (boundaryVisitsByPacketLane.get(packetLane) ?? 0) + visits);
  }
}
const boundaryVisits = activeLeaves.flatMap((leaf) => {
  const descriptor = catalog.canonical[catalog.descriptorIdByLeaf[leaf]!]!;
  const packetAxis = Math.ceil(descriptor.resolution / 4), values: number[] = [];
  for (let z = 0; z < Math.ceil(descriptor.validDimensions[2] / 4); z += 1)
    for (let y = 0; y < Math.ceil(descriptor.validDimensions[1] / 4); y += 1)
      for (let x = 0; x < Math.ceil(descriptor.validDimensions[0] / 4); x += 1) {
        values.push(boundaryVisitsByPacket.get(leaf * 64
          + x + packetAxis * (y + packetAxis * z)) ?? 0);
      }
  return values;
});
const sortedBoundaryVisits = [...boundaryVisits].sort((a, b) => a - b);
const boundaryP95 = sortedBoundaryVisits[Math.max(0,
  Math.ceil(0.95 * sortedBoundaryVisits.length) - 1)] ?? 0;
const boundaryLaneVisits = [...boundaryVisitsByPacketLane.values()];
const immutableMaximumBytes = 0.5 * 2 ** 20, slotMaximumBytes = 0.25 * 2 ** 20;
const semanticCombinedImmutableBytes = sharedImage.layout.immutableBytes;
const passed = compiled.exactStableRowSet
  && refLookupExact
  && semanticAuthority.exact
  && geometryNeighborExact
  && semanticCombinedImmutableBytes <= immutableMaximumBytes
  && sharedImage.layout.bytesPerSlot <= slotMaximumBytes;
const relationCounts = Object.fromEntries([...new Set(compiled.templates.map((value) =>
  value.relation))].map((relation) => [relation, compiled.templates.filter((value) =>
  value.relation === relation).length]));
const report = { schema: "sparse-cm12-interned-boundary-gate/v1", scene: name,
  leafCapacity: atlas.bricks.length, activeLeaves: activeLeaves.length,
  productionSelection: { acceptedCells: grid.cells.length,
    acceptedRows: grid.gradientRows.length, mutableBrickCount,
    hostTemplateVariants,
    packingMode: hostTemplateVariants ? "all-rung" : "accepted-only" },
  packed: { cells: packed.cellCount, rows: packed.rowCount, words: packed.words.length },
  sourcePatchInstancesAllRungs: catalog.patches.length,
  sourceExceptionRowsAllRungs: catalog.exceptionRows.length,
  uniqueTemplates: compiled.templates.length, relationCounts,
  selectedSlotInstances: compiled.instances.length,
  templatePayloadWords: compiled.layout.templatePayloadWords,
  immutableBytes: compiled.layout.immutableBytes,
  refLookupBytes: refLookup.layout.totalBytes,
  refLookupEntries: refLookup.layout.entryCount,
  refLookupMaximumEntriesPerSide: refLookup.layout.maximumEntriesPerSide,
  refLookupExact,
  independentSemanticAuthority: { exact: semanticAuthority.exact,
    firstMismatchLeaf: semanticAuthority.firstMismatchLeaf,
    boundaryRows: semanticAuthority.totalRows,
    duplicateCandidateRows: semanticAuthority.duplicateCandidateRows },
  independentGeometryClosure: { exact: geometryNeighborExact,
    immutableBytes: geometryNeighbors.bytes,
    maximumFanout: geometryNeighbors.maximumFanout,
    combinedImmutableBytes: semanticCombinedImmutableBytes },
  semanticAuthorityDynamicMaximumBytes: 4 * (16 + 2 * atlas.bricks.length),
  traSupplementBytes: traSupplement.layout.totalBytes,
  combinedImmutableBytes: sharedImage.layout.immutableBytes,
  bytesPerSlot: sharedImage.layout.bytesPerSlot,
  totalBytes: sharedImage.layout.totalBytes,
  sharedOffsets: { refLookupBaseWords: refLookup.layout.baseWords,
    geometryBaseWords,
    slotBaseWords: sharedImage.layout.slotBaseWords },
  allSelectedSourcePacketsNonzero: { selectedStablePackets,
    traCompileWorkgroups: Math.ceil(selectedStablePackets / 4),
    targetScatterWorkgroupsPerPhase: Math.ceil(selectedStablePackets / 4),
    targetGlobalMaskReadsPerPhase: 6 * selectedStablePackets,
    targetCandidateLanesPerPhase: 64 * selectedStablePackets,
    acceptedRowInvocationsPerPhase: compiled.expectedStableRows.length,
    canonicalMaskPublishesPerSourcePacketUpperBound: 6,
    exactBoundaryRowPublishesPerSourcePacket: {
      average: boundaryVisits.reduce((sum, value) => sum + value, 0) / boundaryVisits.length,
      p95: boundaryP95, maximum: Math.max(0, ...boundaryVisits),
      maximumPerMarkedLane: boundaryLaneVisits.reduce((maximum, value) =>
        Math.max(maximum, value), 0),
      total: boundaryVisits.reduce((sum, value) => sum + value, 0) } },
  immutableMaximumBytes, slotMaximumBytes,
  exactness: { passed: compiled.exactStableRowSet,
    expectedRows: compiled.expectedStableRows.length,
    representedRows: compiled.representedStableRows.length,
    firstMissing: compiled.firstMissing, firstExtra: compiled.firstExtra },
  timings: { packedMilliseconds, catalogMilliseconds, compileMilliseconds },
  maximumRssMiB: Number((process.resourceUsage().maxRSS / 1024).toFixed(1)), passed };
const json = `${JSON.stringify(report, null, 2)}\n`;const out = argument("out");
if (out) await writeFile(out, json);console.log(json);if (!passed) process.exitCode = 1;
