#!/usr/bin/env node
/** CPU-only size and exactness census for the shared factored AEI topology ABI. */
import { writeFile } from "node:fs/promises";
import { sceneLatticeDimensions } from "../lib/core/scene-lattice";
import {
  createMinimalPowerDamBreak64Scene,
  createOceanSeicheScene,
  createSymmetricExpansionScene,
} from "../lib/core/scenes";
import type { SceneDescription } from "../lib/core/model";
import { buildSparseAtlasCompositeGrid } from
  "../lib/methods/adaptive-mass/sparse-atlas-composite-projection";
import { initializeSparseBrickAtlasFromScene } from
  "../lib/methods/adaptive-mass/sparse-brick-atlas";
import {
  adaptiveMassReceiverScaleForScene,
  dormantReceiverDomain,
  residentSupportAtlas,
} from "../lib/methods/adaptive-mass/webgpu-adaptive-mass-solver";
import {
  SPARSE_CM12_FACTORED_AEI_INVALID,
  SPARSE_CM12_FACTORED_AEI_RELATION,
  compileSparseCM12FactoredAEICatalog,
  createSparseCM12FactoredAEIImage,
  prepareSparseCM12FactoredAEIShadow,
  validateSparseCM12FactoredAEIPreflip,
} from "../lib/methods/adaptive-mass/sparse-cm12-factored-aei-topology";

type SceneName = "ocean-seiche" | "mini64" | "symmetric-expansion";
const argument = (name: string): string | undefined => process.argv.slice(2)
  .find((value) => value.startsWith(`--${name}=`))?.slice(name.length + 3);
const requested = (argument("scene") ?? "all").split(",");
const valid = new Set(["all", "ocean-seiche", "mini64", "symmetric-expansion"]);
if (requested.some((name) => !valid.has(name))) {
  throw new RangeError("--scene must be all or ocean-seiche,mini64,symmetric-expansion");
}
const scenes = (requested.includes("all")
  ? ["ocean-seiche", "mini64", "symmetric-expansion"] : requested) as SceneName[];

interface Lane {
  readonly build: () => SceneDescription;
  readonly brick: 8 | 16;
  readonly surface: number;
  readonly support: number;
  readonly floor: "auto" | 1;
}
const lane = (name: SceneName): Lane => name === "ocean-seiche" ? {
  build: createOceanSeicheScene, brick: 16, surface: 1, support: 3, floor: 1,
} : name === "mini64" ? {
  build: createMinimalPowerDamBreak64Scene, brick: 8, surface: 8,
  support: 9, floor: "auto",
} : {
  build: createSymmetricExpansionScene, brick: 8, surface: 1,
  support: 9, floor: "auto",
};

const mib = (bytes: number): number => Number((bytes / 2 ** 20).toFixed(4));

function census(name: SceneName) {
  const configuration = lane(name);
  const scene = configuration.build();
  const dimensions = sceneLatticeDimensions(scene);
  const authored = initializeSparseBrickAtlasFromScene(scene, {
    finestDimensions: dimensions,
    brickFineResolution: configuration.brick,
    surfaceFineRings: configuration.surface,
  });
  const supported = residentSupportAtlas(authored, "adaptive");
  const receiverScale = adaptiveMassReceiverScaleForScene(scene,
    configuration.support);
  const atlas = dormantReceiverDomain(supported, "adaptive",
    receiverScale.supportRings, configuration.floor,
    receiverScale.minimumCapacityScale);
  const grid = buildSparseAtlasCompositeGrid(atlas);
  const catalog = compileSparseCM12FactoredAEICatalog(grid);
  const activeLeaves = supported.bricks.map((brick) => {
    const leaf = catalog.leafIdByBrickKey.get(brick.key);
    if (leaf === undefined) throw new Error(`active brick ${brick.key} absent from AEI`);
    return leaf;
  });

  // Decode every authoritative row identity represented by the factorization.
  const represented = new Set<number>();
  for (const descriptorId of catalog.descriptorIdByLeaf) {
    const descriptor = catalog.canonical[descriptorId]!;
    for (const axis of [0, 1, 2] as const) for (let local = 0;
      local < descriptor.rowCount[axis]; local += 1) {
      represented.add(descriptor.rowBase[axis]! + local);
    }
  }
  for (const patch of catalog.patches) {
    if (patch.relation === SPARSE_CM12_FACTORED_AEI_RELATION.equalRungCanonical) {
      for (let local = 0; local < patch.rowCount; local += 1) {
        represented.add(patch.rowFirst + local);
      }
    } else {
      for (const row of catalog.exceptionRows.subarray(patch.exceptionFirst,
        patch.exceptionFirst + patch.exceptionCount)) represented.add(row);
    }
  }
  const authoritative = new Set(grid.gradientRows.map((row) => row.id));
  const missing = [...authoritative].filter((row) => !represented.has(row));
  const extra = [...represented].filter((row) => !authoritative.has(row));
  const currentDescriptors = catalog.descriptorIdByLeaf.map((id) =>
    catalog.canonical[id]!);
  const allCurrentCertified = currentDescriptors.every((value) => value.certified);

  // Exercise a real no-delta build receipt without committing the selector.
  const image = createSparseCM12FactoredAEIImage(catalog, activeLeaves, 1);
  prepareSparseCM12FactoredAEIShadow({ image, targetActiveLeaves: activeLeaves,
    changedLeaves: [], candidateGeneration: 2 });
  const receipt = validateSparseCM12FactoredAEIPreflip({ image,
    targetActiveLeaves: activeLeaves, changedLeaves: [] });

  const relationCounts = Object.fromEntries(Object.entries(
    SPARSE_CM12_FACTORED_AEI_RELATION).map(([relationName, relation]) => [
      relationName, catalog.patches.filter((patch) => patch.relation === relation).length,
    ]));
  const uniqueExceptionRows = new Set(catalog.exceptionRows).size;
  const immutableBudget = 0.5 * 2 ** 20;
  const slotBudget = 0.25 * 2 ** 20;
  const exact = missing.length === 0 && extra.length === 0
    && represented.size === authoritative.size && allCurrentCertified;
  return {
    scene: name,
    epoch: "construction topology; full dormant capacity, CPU-only",
    configuration: { finestDimensions: dimensions,
      brickFineResolution: configuration.brick,
      presentationPageResolution: configuration.brick,
      surfaceFineRings: configuration.surface,
      receiverSupportRings: configuration.support,
      receiverFloor: configuration.floor },
    capacity: {
      authoredLeaves: authored.bricks.length,
      activeLeaves: activeLeaves.length,
      leafCapacityIncludingInactive: catalog.layout.leafCapacity,
      rungLevelCount: catalog.layout.levelCount,
      canonicalCapacityAllLeavesAllRungs: catalog.layout.canonicalCapacity,
      certifiedCurrentRungDescriptors: currentDescriptors.filter((value) =>
        value.certified).length,
      absentRungCapacityInvalidByConstruction: catalog.canonical.filter((value) =>
        value.cellFirst === SPARSE_CM12_FACTORED_AEI_INVALID).length,
    },
    topology: {
      cells: grid.cells.length, rows: grid.gradientRows.length,
      canonicalInteriorRows: currentDescriptors.reduce((sum, descriptor) =>
        sum + descriptor.canonicalRowCount, 0),
      orientedPatchDescriptors: catalog.patches.length,
      relationCounts,
      orientedExceptionRowEntries: catalog.exceptionRows.length,
      uniqueExceptionRows,
      maximumPatchReferencesPerLeaf: 6 * 4,
    },
    exactness: {
      stableRowSetEquality: exact,
      authoritativeRowCount: authoritative.size,
      representedRowCount: represented.size,
      missingCount: missing.length,
      extraCount: extra.length,
      firstMissing: missing[0] ?? null,
      firstExtra: extra[0] ?? null,
      canonicalTermOrderAndF32HashCertified: allCurrentCertified,
      identityRule: "row/cell IDs and coefficient storage remain authoritative; certificates hash IDs, ordered term f32 bits and geometry",
    },
    bytes: {
      immutable: catalog.layout.immutableBytes,
      immutableMiB: mib(catalog.layout.immutableBytes),
      perSlot: catalog.layout.bytesPerSlot,
      perSlotMiB: mib(catalog.layout.bytesPerSlot),
      totalDoubleBuffered: catalog.layout.totalBytes,
      totalDoubleBufferedMiB: mib(catalog.layout.totalBytes),
      immutableBudgetMiB: 0.5,
      perSlotBudgetMiB: 0.25,
      immutableWithinBudget: catalog.layout.immutableBytes <= immutableBudget,
      perSlotWithinBudget: catalog.layout.bytesPerSlot <= slotBudget,
    },
    preflipNoDeltaReceipt: receipt,
    passed: exact && receipt.passed
      && catalog.layout.immutableBytes <= immutableBudget
      && catalog.layout.bytesPerSlot <= slotBudget,
  };
}

const receipts = [];
for (const scene of scenes) {
  console.error(`[factored-aei-census] building ${scene}`);
  const receipt = census(scene);
  receipts.push(receipt);
  console.error(`[factored-aei-census] ${scene}: immutable=${
    receipt.bytes.immutableMiB} MiB slot=${receipt.bytes.perSlotMiB} MiB exact=${
    receipt.exactness.stableRowSetEquality} passed=${receipt.passed}`);
}
const report = {
  schema: "sparse-cm12-factored-aei-topology-census/v1",
  abi: {
    immutable: "canonical descriptor per leaf/rung; oriented face patch descriptors; explicit noncanonical stable row IDs",
    slot: "fixed leaf record plus 6 faces * 4 fixed patch references for every capacity leaf",
    selector: "shared header word 3; preflip never mutates selector",
    allocation: "inactive leaves and every rung level are allocated; absent construction rungs are explicitly invalid",
  },
  receipts,
  passed: receipts.every((receipt) => receipt.passed),
};
const json = `${JSON.stringify(report, null, 2)}\n`;
const out = argument("out");
if (out) await writeFile(out, json);
console.log(json);
if (!report.passed) process.exitCode = 1;
