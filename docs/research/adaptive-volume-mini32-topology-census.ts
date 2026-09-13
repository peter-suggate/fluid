/**
 * CPU-only census for docs/research/adaptive-volume-topology-audit.md.
 *
 * Run from the repository root:
 *   node --import tsx docs/research/adaptive-volume-mini32-topology-census.ts
 *
 * This deliberately uses WebGPUSparseCM12Resident.recordPreparedGeneration,
 * whose GPU-shaped device records allocations and pipeline/resource operations
 * without opening a browser, Dawn, or another GPU process.
 */
import { writeFile } from "node:fs/promises";
import { fingerprintSparseCM12RepositorySources } from
  "../../tools/sparse-cm12-source-content-fingerprint";
import { createMinimalPowerDamBreak32Scene } from "../../lib/core/scenes";
import { fluidSolidWorldForScene } from "../../lib/core/solid-world";
import { SPARSE_CM12_ACTIVITY_POLICY } from
  "../../lib/methods/adaptive-volume/features/adaptivity/policy";
import {
  initializeSparseBrickAtlasFromScene,
  sparseBrickSpan,
  sparseCM12InitialActiveBrickKeys,
} from "../../lib/methods/adaptive-volume/sparse-brick-atlas";
import { buildSparseAtlasCompositeGrid } from
  "../../lib/methods/adaptive-volume/sparse-atlas-composite-projection";
import { WebGPUSparseCM12Resident } from
  "../../lib/methods/adaptive-volume/webgpu-sparse-cm12-resident";

Object.defineProperties(globalThis, {
  GPUBufferUsage: { configurable: true, value: {
    MAP_READ: 1, MAP_WRITE: 2, COPY_SRC: 4, COPY_DST: 8, INDEX: 16,
    VERTEX: 32, UNIFORM: 64, STORAGE: 128, INDIRECT: 256, QUERY_RESOLVE: 512,
  } },
  GPUShaderStage: { configurable: true, value: { VERTEX: 1, FRAGMENT: 2, COMPUTE: 4 } },
  GPUMapMode: { configurable: true, value: { READ: 1, WRITE: 2 } },
});

const scene = createMinimalPowerDamBreak32Scene();
const dimensions = [32, 32, 32] as const;
const solidWorld = fluidSolidWorldForScene(scene);
const atlas = initializeSparseBrickAtlasFromScene(scene, {
  finestDimensions: dimensions,
  brickFineResolution: 8,
  solidWorld,
  surfaceFineRings: 1,
  coarseFirstCurvatureTolerance: SPARSE_CM12_ACTIVITY_POLICY.curvatureTolerance,
  initialSurfaceCoarseningBiasRings: 1,
});
const active = sparseCM12InitialActiveBrickKeys(scene, atlas, 2);
const grid = buildSparseAtlasCompositeGrid(atlas);
const recipe = await WebGPUSparseCM12Resident.recordPreparedGeneration({
  atlas,
  active,
  finestCellSize_m: 0.025,
  solidWorld,
  maximumBytes: Number.POSITIVE_INFINITY,
  topologyPageCapacityMaximum: 512,
  symmetry: { scalar: false, face: false },
  limits: { maxComputeWorkgroupsPerDimension: 65_535,
    maxStorageBufferBindingSize: 268_435_456,
    maxBufferSize: 268_435_456 } as GPUSupportedLimits,
});

type RecordedResident = Record<string, unknown> & {
  residentAllocatedBytes: number;
  cellCount: number;
  rowCount: number;
  templateCellCount: number;
  templateRowCount: number;
  topologyPageCapacity: number;
  initialGenerationCellIds: Uint32Array;
  initialGenerationRowIds: Uint32Array;
  templateWords: Uint32Array;
  transportExecutionImageLayout?: Record<string, number>;
  faceAddressLayout: Record<string, number>;
  compiledTopologyLayout: Record<string, number>;
  pressureExecutionImageLayout: Record<string, number>;
};
const resident = (recipe.state as { resident: RecordedResident }).resident;

const referencedResources = new Set<number>();
const seen = new Set<object>();
const findReferences = (value: unknown): void => {
  if (!value || typeof value !== "object" || seen.has(value)) return;
  seen.add(value);
  if ("cm12Resource" in value) {
    referencedResources.add((value as { cm12Resource: number }).cm12Resource);
    return;
  }
  if (ArrayBuffer.isView(value) || value instanceof ArrayBuffer) return;
  if (value instanceof Map) {
    for (const [key, entry] of value) { findReferences(key); findReferences(entry); }
    return;
  }
  if (value instanceof Set) {
    for (const entry of value) findReferences(entry);
    return;
  }
  for (const entry of Object.values(value)) findReferences(entry);
};
findReferences(resident);

const buffers = recipe.operations.flatMap((operation) => {
  if (operation.method !== "createBuffer" || operation.result === undefined
    || !referencedResources.has(operation.result)) return [];
  const descriptor = operation.args[0] as { label?: string; size: number };
  return [{ id: operation.result, label: descriptor.label ?? "unlabelled", bytes: descriptor.size }];
}).sort((left, right) => right.bytes - left.bytes);

const topologyArenaId = buffers.find(buffer =>
  buffer.label === "Sparse Geometric (CM12) physical topology templates and worklists")?.id;
const initialWorklistUpload = recipe.operations.find(operation => {
  if (operation.target !== "queue" || operation.method !== "writeBuffer") return false;
  const target = operation.args[0] as { cm12Resource?: number };
  const payload = operation.args[2];
  if (target?.cm12Resource !== topologyArenaId || !ArrayBuffer.isView(payload)
    || payload.byteLength < 24) return false;
  const words = new Uint32Array(payload.buffer, payload.byteOffset, 6);
  return words[0] === 1 && words[1] === 1 && words[2] === 0 && words[3] === 0;
});
if (!initialWorklistUpload) throw new Error("Could not locate the initial accepted worklist upload");
const initialWorklistBytes = initialWorklistUpload.args[2] as ArrayBufferView;
const initialWorklistHeader = new Uint32Array(
  initialWorklistBytes.buffer, initialWorklistBytes.byteOffset, 32,
);
const acceptedCellCount = initialWorklistHeader[4]!;
const acceptedRowCount = initialWorklistHeader[5]!;

const resolutionHistogram: Record<string, number> = {};
for (const brick of atlas.bricks) {
  const key = `B${brick.resolution}`;
  resolutionHistogram[key] = (resolutionHistogram[key] ?? 0) + 1;
}
const rowKinds: Record<string, number> = {};
for (const row of grid.gradientRows) rowKinds[row.kind] = (rowKinds[row.kind] ?? 0) + 1;
const rowTerms = grid.gradientRows.reduce((sum, row) => sum + row.terms.length, 0);
const worldLeafCapacity = atlas.bricks.length + resident.topologyPageCapacity;
const activeResolutionHistogram: Record<string, number> = {};
for (const brick of atlas.bricks) if (active.has(brick.key)) {
  const key = `B${brick.resolution}`;
  activeResolutionHistogram[key] = (activeResolutionHistogram[key] ?? 0) + 1;
}

const receipt = {
  generatedAt: new Date().toISOString(),
  sourceFingerprint: await fingerprintSparseCM12RepositorySources(process.cwd()),
  method: "WebGPUSparseCM12Resident.recordPreparedGeneration (CPU resource recorder; no GPU)",
  input: {
    sceneId: scene.sceneId,
    dimensions,
    finestCellSize_m: 0.025,
    brickFineResolution: 8,
    presentationPageResolution: 8,
    selectorMode: "coarse-first",
    surfaceFineRings: 1,
    initialSurfaceCoarseningBiasRings: 1,
    curvatureTolerance: SPARSE_CM12_ACTIVITY_POLICY.curvatureTolerance,
    topologyPageCapacityMaximum: 512,
  },
  atlas: {
    generation: atlas.generation,
    leaves: atlas.bricks.length,
    activeLeaves: active.size,
    resolutionHistogram,
    activeResolutionHistogram,
    spanHistogram: Object.fromEntries([...new Set(atlas.bricks.map(sparseBrickSpan))]
      .sort((a, b) => a - b)
      .map(span => [`span${span}`, atlas.bricks.filter(brick => sparseBrickSpan(brick) === span).length])),
  },
  inputCompositeGrid: {
    cells: grid.cells.length,
    rows: grid.gradientRows.length,
    rowTerms,
    rowKinds,
    mixedSeamRows: grid.mixedSeamRowCount,
    sparseAirRows: grid.sparseAirRowCount,
  },
  resident: {
    // SCMT header words 2/3 are the host-template cell/row counts. The
    // constructor's templateCellCount/templateRowCount fields currently receive
    // the physical capacity instead, so they are deliberately not used here.
    allRungTemplateCells: resident.templateWords[2],
    allRungTemplateRows: resident.templateWords[3],
    templateWords: resident.templateWords.length,
    templateBytes: resident.templateWords.byteLength,
    topologyPageCapacity: resident.topologyPageCapacity,
    worldLeafCapacity,
    physicalCellCapacity: resident.cellCount,
    physicalRowCapacity: resident.rowCount,
    initialAcceptedCellWorklist: acceptedCellCount,
    initialAcceptedRowWorklist: acceptedRowCount,
    cellCapacityToAcceptedRatio: resident.cellCount / acceptedCellCount,
    rowCapacityToAcceptedRatio: resident.rowCount / acceptedRowCount,
    allocatedBytes: resident.residentAllocatedBytes,
    allocatedMiB: resident.residentAllocatedBytes / 2 ** 20,
    pressureExecutionImageLayout: resident.pressureExecutionImageLayout,
    compiledTopologyLayout: resident.compiledTopologyLayout,
    transportExecutionImageLayout: resident.transportExecutionImageLayout,
    faceAddressLayout: resident.faceAddressLayout,
  },
  recipe: {
    operations: recipe.operations.length,
    operationsByMethod: Object.fromEntries([...new Set(recipe.operations.map(operation => operation.method))]
      .sort().map(method => [method, recipe.operations.filter(operation => operation.method === method).length])),
  },
  retainedBuffers: buffers,
};

const output = new URL("./adaptive-volume-mini32-topology-census.json", import.meta.url);
await writeFile(output, `${JSON.stringify(receipt, null, 2)}\n`);
console.log(JSON.stringify(receipt, null, 2));
