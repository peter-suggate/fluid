#!/usr/bin/env node
import { writeFile } from "node:fs/promises";
import { sceneLatticeDimensions } from "../lib/core/scene-lattice";
import { createMinimalPowerDamBreak64Scene, createOceanSeicheScene,
  createSymmetricExpansionScene } from "../lib/core/scenes";
import type { SceneDescription } from "../lib/core/model";
import { buildSparseAtlasCompositeGrid } from
  "../lib/methods/adaptive-mass/sparse-atlas-composite-projection";
import { initializeSparseBrickAtlasFromScene, sparseBrickSpan } from
  "../lib/methods/adaptive-mass/sparse-brick-atlas";
import { compileSparseCM12FactoredAEIPackedTemplateCatalog } from
  "../lib/methods/adaptive-mass/sparse-cm12-factored-aei-packed-template";
import { compileSparseCM12InternedBoundaryOperators } from
  "../lib/methods/adaptive-mass/sparse-cm12-interned-boundary-operators";
import { compileSparseCM12VexIBO1Program } from
  "../lib/methods/adaptive-mass/sparse-cm12-vex-ibo1-consumer";
import { applySparseCM12VexIBO1ImageSelected,
  compileSparseCM12VexIBO1Image } from
  "../lib/methods/adaptive-mass/sparse-cm12-vex-ibo1-image";
import { adaptiveMassReceiverScaleForScene, dormantReceiverDomain,
  residentSupportAtlas } from
  "../lib/methods/adaptive-mass/webgpu-adaptive-mass-solver";
import { packSparseCM12AcceptedTopologyTemplatesForQA,
  packSparseCM12ResidentTopologyTemplatesForQA,
  sparseCM12HostTemplateVariantsEnabled } from
  "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident";

type SceneName = "ocean-seiche" | "mini64" | "symmetric-expansion";
const argument = (name: string) => process.argv.slice(2)
  .find((value) => value.startsWith(`--${name}=`))?.slice(name.length + 3);
const requested = argument("scene") ?? "all";
const names = (requested === "all" ? ["ocean-seiche", "mini64", "symmetric-expansion"]
  : [requested]) as SceneName[];
interface Lane { readonly build: () => SceneDescription; readonly brick: 8 | 16;
  readonly surface: number; readonly support: number; readonly floor: "auto" | 1 }
const lane = (name: SceneName): Lane => name === "ocean-seiche" ? {
  build: createOceanSeicheScene, brick: 16, surface: 1, support: 3, floor: 1,
} : name === "mini64" ? {
  build: createMinimalPowerDamBreak64Scene, brick: 8, surface: 8,
  support: 9, floor: "auto",
} : { build: createSymmetricExpansionScene, brick: 8, surface: 1,
  support: 9, floor: "auto" };

const reports = names.map((name) => {
  console.error(`[vxi1] compiling ${name}`);
  const config = lane(name), scene = config.build();
  const authored = initializeSparseBrickAtlasFromScene(scene, {
    finestDimensions: sceneLatticeDimensions(scene),
    brickFineResolution: config.brick, surfaceFineRings: config.surface });
  const supported = residentSupportAtlas(authored, "adaptive");
  const receiver = adaptiveMassReceiverScaleForScene(scene, config.support);
  const atlas = dormantReceiverDomain(supported, "adaptive", receiver.supportRings,
    config.floor, receiver.minimumCapacityScale);
  const grid = buildSparseAtlasCompositeGrid(atlas);
  const mutable = atlas.bricks.filter((brick) => sparseBrickSpan(brick) === 1).length;
  const allRung = sparseCM12HostTemplateVariantsEnabled(grid.cells.length,
    grid.gradientRows.length, mutable, atlas.brickFineResolution);
  const packed = allRung ? packSparseCM12ResidentTopologyTemplatesForQA(atlas, grid)
    : packSparseCM12AcceptedTopologyTemplatesForQA(atlas, grid);
  const validDimensions = (leaf: number, resolution: number) => {
    const brick = atlas.bricks[leaf]!, width = atlas.brickFineResolution
      * sparseBrickSpan(brick), scale = width / resolution;
    return ([0, 1, 2] as const).map((axis) => {
      const origin = atlas.brickFineResolution * brick.coordinate[axis]!;
      return Math.min(resolution, Math.ceil(Math.max(0, Math.min(width,
        atlas.dimensions[axis]! - origin)) / scale));
    }) as [number, number, number];
  };
  const catalog = compileSparseCM12FactoredAEIPackedTemplateCatalog({
    words: packed.words, brickFineResolution: config.brick,
    brickKeyByLeafId: atlas.bricks.map((brick) => brick.key), validDimensions,
    scaleLog2: (leaf, resolution) => Math.log2(atlas.brickFineResolution
      * sparseBrickSpan(atlas.bricks[leaf]!) / resolution),
    selectedResolution: (leaf) => atlas.bricks[leaf]!.resolution,
  });
  const supportedKeys = new Set(supported.bricks.map((brick) => brick.key));
  const activeLeaves = atlas.bricks.flatMap((brick, leaf) =>
    supportedKeys.has(brick.key) ? [leaf] : []);
  const ibo = compileSparseCM12InternedBoundaryOperators({ catalog,
    packedWords: packed.words, activeLeaves });
  const program = compileSparseCM12VexIBO1Program({ catalog, ibo });
  const image = compileSparseCM12VexIBO1Image(program);
  let sourcePackets = 0, operationTouches = 0, atomicPublicationGroups = 0;
  let uniqueTargetPacketsPerSource = 0;
  let sourceMaskReads = 0, selectedFaceRefs = 0;
  for (const leaf of activeLeaves) {
    const descriptor = catalog.canonical[catalog.descriptorIdByLeaf[leaf]!]!;
    const [nx, ny, nz] = descriptor.validDimensions;
    const masks = new Map<number, [number, number]>();
    const axis = Math.max(1, Math.ceil(descriptor.resolution / 4));
    for (let z = 0; z < nz; z += 1) for (let y = 0; y < ny; y += 1) {
      for (let x = 0; x < nx; x += 1) {
        const localPacket = Math.floor(x / 4) + axis * (Math.floor(y / 4)
          + axis * Math.floor(z / 4)), lane = (x & 3) + 4 * (y & 3) + 16 * (z & 3);
        const mask = masks.get(localPacket) ?? [0, 0] as [number, number];
        mask[lane >>> 5] = (mask[lane >>> 5]! | (1 << (lane & 31))) >>> 0;
        masks.set(localPacket, mask);
      }
    }
    selectedFaceRefs += ibo.instances.filter((value) => value.sourceLeaf === leaf).length;
    for (const [localPacket, mask] of masks) {
      sourcePackets += 1; sourceMaskReads += 1;
      const sourcePacket = 64 * leaf + localPacket;
      const output = applySparseCM12VexIBO1ImageSelected({ image,
        sourceStablePacket: sourcePacket, sourceLow: mask[0], sourceHigh: mask[1] });
      uniqueTargetPacketsPerSource += output.size;
      const canonicalId = catalog.descriptorIdByLeaf[leaf]!, core = image.words[
        image.layout.canonicalTemplateBaseWords + canonicalId]!;
      const recipeCost = (template: number) => {
        const directory = image.words[image.layout.directoryBaseWords
          + 64 * template + localPacket]!, first = directory & 0x000f_ffff;
        const count = directory >>> 20; let groups = 0, prior = -1;
        for (let local = 0; local < count; local += 1) {
          const key = image.words[image.layout.operationBaseWords
            + 3 * (first + local)]! & 0x8000_003f;
          if (key !== prior) { groups += 1; prior = key; }
        }
        return { count, groups };
      };
      const coreCost = recipeCost(core);
      operationTouches += coreCost.count; atomicPublicationGroups += coreCost.groups;
      for (const instance of ibo.instances) if (instance.sourceLeaf === leaf) {
        const cost = recipeCost(image.faceTemplateIds[instance.templateId]!);
        operationTouches += cost.count; atomicPublicationGroups += cost.groups;
      }
    }
  }
  const immutableBudgetBytes = 0.5 * 2 ** 20;
  const combinedImmutableBytes = ibo.layout.immutableBytes + image.layout.totalBytes;
  return { scene: name, packingMode: allRung ? "all-rung" : "accepted-only",
    leafCapacity: atlas.bricks.length, activeLeaves: activeLeaves.length,
    ibo: { immutableBytes: ibo.layout.immutableBytes,
      bytesPerSlot: ibo.layout.bytesPerSlot, templates: ibo.templates.length,
      selectedFaceRefs: ibo.instances.length },
    vxi1: { immutableSupplementBytes: image.layout.totalBytes,
      coreTemplates: image.layout.coreTemplateCount,
      faceTemplates: image.layout.faceTemplateCount,
      operationCount: image.layout.operationCount },
    combinedImmutableBytes, immutableBudgetBytes,
    immutableBudgetHeadroomBytes: immutableBudgetBytes - combinedImmutableBytes,
    runtimeAllSourcePacketsNonzero: { workgroups: Math.ceil(sourcePackets / 4),
      sourcePackets, sourceMaskReads, maskedShiftOperationTouches: operationTouches,
      targetAtomicOrPublications: atomicPublicationGroups,
      uniqueTargetPacketsPerSourceSum: uniqueTargetPacketsPerSource, selectedFaceRefs,
      averageOperationsPerSourcePacket: operationTouches / Math.max(1, sourcePackets),
      averageTargetPublicationsPerSourcePacket:
        atomicPublicationGroups / Math.max(1, sourcePackets),
      averageUniqueTargetPacketsPerSource:
        uniqueTargetPacketsPerSource / Math.max(1, sourcePackets) },
    exactStableRows: ibo.exactStableRowSet,
    passed: ibo.exactStableRowSet && combinedImmutableBytes <= immutableBudgetBytes };
});
const report = { schema: "sparse-cm12-vex-ibo1-image/v1", reports,
  passed: reports.every((value) => value.passed) };
const json = `${JSON.stringify(report, null, 2)}\n`, out = argument("out");
if (out) await writeFile(out, json); console.log(json);
if (!report.passed) process.exitCode = 1;
