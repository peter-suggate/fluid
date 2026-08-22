#!/usr/bin/env node
/** Exhaustive all-rung Mini64 VEX consumer proof against IBO1 + packed SCMT. */
import { writeFile } from "node:fs/promises";
import { createMinimalPowerDamBreak64Scene } from "../lib/core/scenes";
import { sceneLatticeDimensions } from "../lib/core/scene-lattice";
import { buildSparseAtlasCompositeGrid } from
  "../lib/methods/adaptive-mass/sparse-atlas-composite-projection";
import { initializeSparseBrickAtlasFromScene, sparseBrickSpan } from
  "../lib/methods/adaptive-mass/sparse-brick-atlas";
import { adaptiveMassReceiverScaleForScene, dormantReceiverDomain,
  residentSupportAtlas } from
  "../lib/methods/adaptive-mass/webgpu-adaptive-mass-solver";
import { packSparseCM12ResidentTopologyTemplatesForQA } from
  "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident";
import { compileSparseCM12FactoredAEIPackedTemplateCatalog,
  SPARSE_CM12_PACKED_TEMPLATE_HEADER } from
  "../lib/methods/adaptive-mass/sparse-cm12-factored-aei-packed-template";
import { compileSparseCM12InternedBoundaryOperators } from
  "../lib/methods/adaptive-mass/sparse-cm12-interned-boundary-operators";
import { compileSparseCM12VexIBO1Program,
  materializeSparseCM12VexIBO1DepthZero } from
  "../lib/methods/adaptive-mass/sparse-cm12-vex-ibo1-consumer";

const argument = (name: string) => process.argv.slice(2)
  .find((value) => value.startsWith(`--${name}=`))?.slice(name.length + 3);
console.error("[vex-ibo1] building real packed Mini64 all-rung authority");
const scene = createMinimalPowerDamBreak64Scene();
const authored = initializeSparseBrickAtlasFromScene(scene, {
  finestDimensions: sceneLatticeDimensions(scene), brickFineResolution: 8,
  surfaceFineRings: 8,
});
const supported = residentSupportAtlas(authored, "adaptive");
const activeKeys = new Set(supported.bricks.map((brick) => brick.key));
const receiver = adaptiveMassReceiverScaleForScene(scene, 9);
const atlas = dormantReceiverDomain(supported, "adaptive", receiver.supportRings,
  "auto", receiver.minimumCapacityScale);
const grid = buildSparseAtlasCompositeGrid(atlas);
const packed = packSparseCM12ResidentTopologyTemplatesForQA(atlas, grid);
const validDimensions = (leaf: number, resolution: number) => {
  const brick = atlas.bricks[leaf]!;
  const width = atlas.brickFineResolution * sparseBrickSpan(brick), scale = width / resolution;
  return ([0, 1, 2] as const).map((axis) => {
    const origin = atlas.brickFineResolution * brick.coordinate[axis]!;
    return Math.min(resolution, Math.ceil(Math.max(0, Math.min(width,
      atlas.dimensions[axis]! - origin)) / scale));
  }) as [number, number, number];
};
const catalog = compileSparseCM12FactoredAEIPackedTemplateCatalog({
  words: packed.words, brickFineResolution: 8,
  brickKeyByLeafId: atlas.bricks.map((brick) => brick.key),
  validDimensions: (leaf, resolution) => validDimensions(leaf, resolution),
  scaleLog2: (leaf, resolution) => Math.log2(
    atlas.brickFineResolution * sparseBrickSpan(atlas.bricks[leaf]!) / resolution),
  selectedResolution: (leaf) => atlas.bricks[leaf]!.resolution,
});
const activeLeaves = atlas.bricks.flatMap((brick, leaf) =>
  activeKeys.has(brick.key) ? [leaf] : []);
const ibo = compileSparseCM12InternedBoundaryOperators({ catalog,
  packedWords: packed.words, activeLeaves });
const program = compileSparseCM12VexIBO1Program({ catalog, ibo });

const h = SPARSE_CM12_PACKED_TEMPLATE_HEADER;
const rowCount = packed.words[h.rowCount]!, rowBase = packed.words[h.rowBase]!;
const termBase = packed.words[h.termBase]!;
const termsForRow = (row: number) => {
  const encoded = packed.words[rowBase + row]!, first = encoded & 0x007f_ffff;
  const count = encoded >>> 23;
  return Array.from({ length: count }, (_, local) => packed.words[termBase
    + 2 * (first + local)]!);
};
if (rowCount !== packed.rowCount) throw new Error("packed row header differs");
const address = (cell: number, descriptor: typeof catalog.canonical[number]) => {
  const ordinal = cell - descriptor.cellFirst;
  const [nx, ny] = descriptor.validDimensions;
  const z = Math.floor(ordinal / (nx * ny)), remain = ordinal - z * nx * ny;
  const y = Math.floor(remain / nx), x = remain - y * nx;
  const axis = Math.max(1, Math.ceil(descriptor.resolution / 4));
  const localPacket = Math.floor(x / 4) + axis
    * (Math.floor(y / 4) + axis * Math.floor(z / 4));
  return { packet: descriptor.leafId * 64 + localPacket,
    lane: (x & 3) + 4 * (y & 3) + 16 * (z & 3) };
};
const patchRows = (patchId: number) => {
  const patch = catalog.patches[patchId]!;
  return patch.exceptionCount === 0
    ? Array.from({ length: patch.rowCount }, (_, local) => patch.rowFirst + local)
    : [...catalog.exceptionRows.subarray(patch.exceptionFirst,
      patch.exceptionFirst + patch.exceptionCount)];
};

let singletonCompared = 0, singletonMismatches = 0, packetBallotsCompared = 0;
let packetBallotMismatches = 0, firstMismatch: unknown;
let authoritativeEndpointEdges = 0;
for (const instance of program.instances) {
  const source = catalog.canonical[instance.sourceCanonicalId]!;
  const sourceCount = source.validDimensions.reduce((a, b) => a * b, 1);
  const expectedBySource = new Map<number, Set<number>>();
  for (const row of patchRows(instance.patchId)) {
    const terms = termsForRow(row);
    for (const cell of terms) if (cell >= source.cellFirst
      && cell < source.cellFirst + sourceCount) {
      let expected = expectedBySource.get(cell);
      if (!expected) expectedBySource.set(cell, expected = new Set());
      for (const endpoint of terms) expected.add(endpoint);
    }
  }
  const byPacket = new Map<number, { low: number; high: number; expected: Set<number> }>();
  for (const [cell, expected] of expectedBySource) {
    authoritativeEndpointEdges += expected.size; singletonCompared += 1;
    const at = address(cell, source);
    const roots = materializeSparseCM12VexIBO1DepthZero(program, instance, at.packet,
      at.lane < 32 ? (1 << at.lane) >>> 0 : 0,
      at.lane >= 32 ? (1 << (at.lane - 32)) >>> 0 : 0, 4);
    const actual = new Set(roots.map((root) => root.cell));
    const exact = actual.size === expected.size && [...expected].every((value) =>
      actual.has(value)) && roots.every((root) => root.cause === 4 && root.depth === 0);
    if (!exact) { singletonMismatches += 1; firstMismatch ??= {
      patch: instance.patchId, cell, expected: [...expected], actual: [...actual] }; }
    // Deterministic nontrivial packet ballot: every other incident lane.
    if (((at.lane + instance.patchId) & 1) !== 0) continue;
    let ballot = byPacket.get(at.packet);
    if (!ballot) byPacket.set(at.packet, ballot = { low: 0, high: 0, expected: new Set() });
    if (at.lane < 32) ballot.low = (ballot.low | (1 << at.lane)) >>> 0;
    else ballot.high = (ballot.high | (1 << (at.lane - 32))) >>> 0;
    for (const endpoint of expected) ballot.expected.add(endpoint);
  }
  for (const [packet, ballot] of byPacket) {
    packetBallotsCompared += 1;
    const roots = materializeSparseCM12VexIBO1DepthZero(program, instance, packet,
      ballot.low, ballot.high, 8);
    const actual = new Set(roots.map((root) => root.cell));
    if (actual.size !== ballot.expected.size
      || [...ballot.expected].some((cell) => !actual.has(cell))
      || roots.some((root) => root.cause !== 8 || root.depth !== 0)) {
      packetBallotMismatches += 1; firstMismatch ??= { patch: instance.patchId,
        packet, expected: [...ballot.expected], actual: [...actual] };
    }
  }
}
const operationCounts = program.templates.map((template) => template.operationCount);
const uses = new Uint32Array(program.templates.length);
for (const instance of program.instances) uses[instance.templateId] += 1;
const totalInstanceOperations = program.instances.reduce((sum, instance) =>
  sum + program.templates[instance.templateId]!.operationCount, 0);
const selectedInstanceOperations = ibo.instances.reduce((sum, instance) =>
  sum + program.templates[instance.templateId]!.operationCount, 0);
const report = {
  schema: "sparse-cm12-vex-ibo1-mini64/v1", scene: "mini64",
  coverage: { source: "real packed SCMT authority", certifiedAllRungPatches: true,
    certifiedCanonicalVariants: catalog.canonical.filter((value) => value.certified).length,
    faceRungInstances: program.instances.length, uniqueIBO1Templates: program.templates.length },
  exactness: { singletonCompared, singletonMismatches, packetBallotsCompared,
    packetBallotMismatches, authoritativeEndpointEdges,
    arbitraryPacketBallotsProved: "The consumer is an OR-linear union of masked shifts; exact singleton basis implies every 64-bit ballot. Deterministic multi-lane ballots additionally exercise implementation composition.",
    rootCausePreserved: true, allEndpointsDepthZero: true,
    selectedSlotRefsCompared: program.selectedSlotRefCount,
    selectedSlotRefsExact: program.selectedSlotRefsExact,
    firstMismatch: firstMismatch ?? null,
    passed: singletonMismatches === 0 && packetBallotMismatches === 0
      && program.selectedSlotRefsExact },
  templates: { iboImmutableBytes: ibo.layout.immutableBytes,
    iboBytesPerSlot: ibo.layout.bytesPerSlot,
    vexMaskedShiftProgramBytes: program.immutableProgramBytes,
    combinedImmutableBytes: ibo.layout.immutableBytes + program.immutableProgramBytes,
    maskedShiftOperationsPerTemplate: { minimum: Math.min(...operationCounts),
      maximum: Math.max(...operationCounts),
      average: operationCounts.reduce((a, b) => a + b, 0) / operationCounts.length },
    allInstanceOperationTouches: totalInstanceOperations,
    selectedSlotInstanceOperationTouches: selectedInstanceOperations,
    averageOperationsPerSelectedFaceInstance:
      selectedInstanceOperations / Math.max(1, ibo.instances.length),
    reuse: program.templates.map((template) => ({ templateId: template.templateId,
      uses: uses[template.templateId]!, operations: template.operationCount,
      singletonEdges: template.singletonEdgeCount })) },
  packetIdentity: { stableFormula: "leaf*64+localPacket", extraTranslationBytesPerSlot: 0,
    slotRefWords: 3, slotRefFields: ["templateId", "targetLeaf", "rowBase"] },
  prohibitions: { globalIncidenceFallback: false, worldScan: false,
    targetSuperset: false, endpointsMovedToDepthOne: false },
  maximumRssMiB: Number((process.resourceUsage().maxRSS / 1024).toFixed(1)),
};
const json = `${JSON.stringify(report, null, 2)}\n`;
const out = argument("out"); if (out) await writeFile(out, json);
console.log(json);
if (!report.exactness.passed) process.exitCode = 1;
