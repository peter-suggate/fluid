#!/usr/bin/env node
import { writeFile } from "node:fs/promises";
import { performance } from "node:perf_hooks";
import { sceneLatticeDimensions } from "../lib/core/scene-lattice";
import {
  createMinimalPowerDamBreak64Scene,
  createOceanSeicheScene,
  createSymmetricExpansionScene,
} from "../lib/core/scenes";
import type { SceneDescription } from "../lib/core/model";
import { buildSparseAtlasCompositeGrid } from
  "../lib/methods/adaptive-mass/sparse-atlas-composite-projection";
import { compileSparseCM12FactoredAEIPackedTemplateCatalog } from
  "../lib/methods/adaptive-mass/sparse-cm12-factored-aei-packed-template";
import {
  initializeSparseBrickAtlasFromScene,
  sparseBrickSpan,
} from "../lib/methods/adaptive-mass/sparse-brick-atlas";
import {
  adaptiveMassReceiverScaleForScene,
  dormantReceiverDomain,
  residentSupportAtlas,
} from "../lib/methods/adaptive-mass/webgpu-adaptive-mass-solver";
import { packSparseCM12ResidentTopologyTemplatesForQA } from
  "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident";

type SceneName = "ocean-seiche" | "mini64" | "symmetric-expansion";
const argument = (name: string): string | undefined => process.argv.slice(2)
  .find((value) => value.startsWith(`--${name}=`))?.slice(name.length + 3);
const name = (argument("scene") ?? "ocean-seiche") as SceneName;
if (!["ocean-seiche", "mini64", "symmetric-expansion"].includes(name)) {
  throw new RangeError("scene must be ocean-seiche, mini64, or symmetric-expansion");
}
interface Lane { readonly build: () => SceneDescription; readonly brick: 8 | 16;
  readonly surface: number; readonly support: number; readonly floor: "auto" | 1 }
const lane: Lane = name === "ocean-seiche" ? {
  build: createOceanSeicheScene, brick: 16, surface: 1, support: 3, floor: 1,
} : name === "mini64" ? {
  build: createMinimalPowerDamBreak64Scene, brick: 8, surface: 8,
  support: 9, floor: "auto",
} : {
  build: createSymmetricExpansionScene, brick: 8, surface: 1,
  support: 9, floor: "auto",
};
const IMMUTABLE_MAXIMUM = 0.5 * 2 ** 20;
const SLOT_MAXIMUM = 0.25 * 2 ** 20;
const scene = lane.build(), dimensions = sceneLatticeDimensions(scene);
const authored = initializeSparseBrickAtlasFromScene(scene, {
  finestDimensions: dimensions, brickFineResolution: lane.brick,
  surfaceFineRings: lane.surface,
});
const supported = residentSupportAtlas(authored, "adaptive");
const receiver = adaptiveMassReceiverScaleForScene(scene, lane.support);
const atlas = dormantReceiverDomain(supported, "adaptive", receiver.supportRings,
  lane.floor, receiver.minimumCapacityScale);
const grid = buildSparseAtlasCompositeGrid(atlas);
console.error(`[factored-aei-packed-gate] ${name}: packing ${atlas.bricks.length} leaves`);
const packBegin = performance.now();
const packed = packSparseCM12ResidentTopologyTemplatesForQA(atlas, grid);
const packMilliseconds = performance.now() - packBegin;
const validDimensions = (leaf: number, resolution: number) => {
  const brick = atlas.bricks[leaf]!;
  const width = atlas.brickFineResolution * sparseBrickSpan(brick);
  const scale = width / resolution;
  return ([0, 1, 2] as const).map((axis) => {
    const origin = atlas.brickFineResolution * brick.coordinate[axis]!;
    const extent = Math.max(0, Math.min(width, atlas.dimensions[axis]! - origin));
    return Math.min(resolution, Math.ceil(extent / scale));
  }) as [number, number, number];
};
const compileBegin = performance.now();
let report: Record<string, unknown>;
try {
  const catalog = compileSparseCM12FactoredAEIPackedTemplateCatalog({
    words: packed.words, brickFineResolution: lane.brick,
    brickKeyByLeafId: atlas.bricks.map((brick) => brick.key),
    validDimensions: (leaf, resolution) => validDimensions(leaf, resolution),
    scaleLog2: (leaf, resolution) => Math.log2(
      atlas.brickFineResolution * sparseBrickSpan(atlas.bricks[leaf]!) / resolution),
    selectedResolution: (leaf) => atlas.bricks[leaf]!.resolution,
    immutableMaximumBytes: IMMUTABLE_MAXIMUM,
  });
  const selectedCertified = catalog.descriptorIdByLeaf.every((id) =>
    catalog.canonical[id]?.certified);
  const passed = selectedCertified
    && catalog.layout.immutableBytes <= IMMUTABLE_MAXIMUM
    && catalog.layout.bytesPerSlot <= SLOT_MAXIMUM;
  report = { schema: "sparse-cm12-factored-aei-packed-gate/v1", scene: name,
    leafCapacity: atlas.bricks.length, activeLeaves: supported.bricks.length,
    packed: { words: packed.words.length, cells: packed.cellCount, rows: packed.rowCount },
    canonical: { capacity: catalog.canonical.length,
      certified: catalog.canonical.filter((value) => value.certified).length,
      explicitlyInvalid: catalog.canonical.filter((value) => !value.certified).length,
      selectedCertified },
    patches: catalog.patches.length, exceptionRows: catalog.exceptionRows.length,
    immutableBytes: catalog.layout.immutableBytes,
    bytesPerSlot: catalog.layout.bytesPerSlot,
    totalBytes: catalog.layout.totalBytes,
    immutableMaximumBytes: IMMUTABLE_MAXIMUM, slotMaximumBytes: SLOT_MAXIMUM,
    packMilliseconds, compileMilliseconds: performance.now() - compileBegin,
    maximumRssMiB: Number((process.resourceUsage().maxRSS / 1024).toFixed(1)), passed };
} catch (error) {
  report = { schema: "sparse-cm12-factored-aei-packed-gate/v1", scene: name,
    leafCapacity: atlas.bricks.length, activeLeaves: supported.bricks.length,
    packed: { words: packed.words.length, cells: packed.cellCount, rows: packed.rowCount },
    immutableMaximumBytes: IMMUTABLE_MAXIMUM, slotMaximumBytes: SLOT_MAXIMUM,
    packMilliseconds, compileMilliseconds: performance.now() - compileBegin,
    maximumRssMiB: Number((process.resourceUsage().maxRSS / 1024).toFixed(1)),
    error: error instanceof Error ? error.message : String(error), passed: false };
}
const json = `${JSON.stringify(report, null, 2)}\n`;
const out = argument("out");if (out) await writeFile(out, json);
console.log(json);
if (!report.passed) process.exitCode = 1;
