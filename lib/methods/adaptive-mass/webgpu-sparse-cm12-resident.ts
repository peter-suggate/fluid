import type { SparseAtlasCompositeGrid } from "./sparse-atlas-composite-projection";
import { sparseAtlasScalarsHaveHorizontalD4Symmetry } from
  "./sparse-atlas-surface-conditioning";
import type { SparseAdaptiveMassAtlas } from "./sparse-brick-atlas";
import { packAdaptiveMassPresentationOwnerKey,
  type WebGPUAdaptiveMassAtlasPresentation } from "./webgpu-adaptive-mass-atlas-presentation";
import { webgpuSparseCM12ResidentWGSL } from "./webgpu-sparse-cm12-resident.wgsl";

const INVALID = 0xffff_ffff;
const WORKGROUP_SIZE = 64;
const PCG_ITERATIONS = 128;
const ACTIVITY_HEADER_WORDS = 12;
const ACTIVITY_RECORD_WORDS = 12;

interface PackedResidentTopology {
  readonly words: Uint32Array;
  readonly cellOffset: number;
  readonly rowOffset: number;
  readonly termOffset: number;
  readonly incidenceOffset: number;
  readonly incidenceRecordOffset: number;
  readonly ownerOffset: number;
  readonly brickOffset: number;
  readonly backgroundOwnerOffset: number;
  readonly brickDirectoryOffset: number;
  readonly brickCount: number;
  readonly incidenceCount: number;
}

interface ResidentStateLayout {
  readonly floatCount: number;
  readonly densityA: number; readonly densityB: number;
  readonly gammaA: number; readonly gammaB: number;
  readonly cellVelocityA: number; readonly cellVelocityB: number;
  readonly faceA: number; readonly faceB: number;
  readonly pressure: number; readonly rhs: number; readonly diagonal: number;
  readonly liquid: number; readonly theta: number; readonly residual: number;
  readonly preconditioned: number; readonly direction: number;
  readonly applied: number; readonly divergence: number;
  readonly presentationBrickWet: number;
  readonly sharpeningDelta: number; readonly sharpeningAcceptance: number;
}

export interface SparseCM12GPUActivityRecord {
  readonly scoreByte: number;
  readonly reasons: number;
  readonly hotEpochs: number;
  readonly quietEpochs: number;
  /** GPU-authored request for the next candidate topology epoch. */
  readonly plannedResolution: 1 | 2 | 4 | 8;
  readonly planReasons: number;
  readonly active: boolean;
  readonly activatedStep: number;
}

export interface SparseCM12GPUActivitySnapshot {
  readonly acceptedSteps: number;
  readonly records: readonly SparseCM12GPUActivityRecord[];
}

const align4 = (value: number): number => (value + 3) & ~3;

function residentStateLayout(
  cellCount: number,
  rowCount: number,
  brickCount: number,
): ResidentStateLayout {
  let at = 0;
  const cells = () => { const result = at; at += align4(cellCount); return result; };
  const rows = () => { const result = at; at += align4(rowCount); return result; };
  const cellVectors = () => { const result = at; at += align4(4 * cellCount); return result; };
  return {
    densityA: cells(), densityB: cells(), gammaA: cells(), gammaB: cells(),
    cellVelocityA: cellVectors(), cellVelocityB: cellVectors(),
    faceA: rows(), faceB: rows(), pressure: cells(), rhs: cells(), diagonal: cells(),
    liquid: cells(), theta: rows(), residual: cells(), preconditioned: cells(),
    direction: cells(), applied: cells(), divergence: cells(),
    presentationBrickWet: (() => { const result = at; at += align4(brickCount); return result; })(),
    sharpeningDelta: cells(), sharpeningAcceptance: cells(),
    floatCount: at,
  };
}

function setF32(words: Uint32Array, index: number, value: number): void {
  new DataView(words.buffer).setFloat32(index * 4, value, true);
}

function packResidentTopology(
  atlas: SparseAdaptiveMassAtlas,
  grid: SparseAtlasCompositeGrid,
): PackedResidentTopology {
  let termCount = 0;
  for (const row of grid.gradientRows) termCount += row.terms.length;
  const byCell: { row: number; term: number }[][] = Array.from(
    { length: grid.cells.length }, () => [],
  );
  let nextTerm = 0;
  for (const row of grid.gradientRows) {
    for (const term of row.terms) {
      byCell[term.cellId]!.push({ row: row.id, term: nextTerm++ });
    }
  }
  const incidenceCount = byCell.reduce((sum, values) => sum + values.length, 0);
  const denseCount = atlas.dimensions[0] * atlas.dimensions[1] * atlas.dimensions[2];
  const brickIndexByKey = new Map(atlas.bricks.map((brick, index) => [brick.key, index]));
  let at = 0;
  const cellOffset = at; at += 12 * grid.cells.length;
  const rowOffset = at; at += 12 * grid.gradientRows.length;
  const termOffset = at; at += 2 * termCount;
  const incidenceOffset = at; at += grid.cells.length + 1;
  const incidenceRecordOffset = at; at += 2 * incidenceCount;
  const ownerOffset = at; at += 4 * denseCount;
  const brickOffset = at; at += 4 * atlas.bricks.length;
  const backgroundOwnerOffset = at; at += 2;
  const brickDirectoryOffset = at; at += atlas.brickDimensions.reduce(
    (product, value) => product * value, 1,
  );
  const words = new Uint32Array(at);
  words.fill(INVALID, brickDirectoryOffset);

  for (const cell of grid.cells) {
    const base = cellOffset + 12 * cell.id;
    setF32(words, base, cell.centerFine[0]);
    setF32(words, base + 1, cell.centerFine[1]);
    setF32(words, base + 2, cell.centerFine[2]);
    setF32(words, base + 3, cell.volume);
    setF32(words, base + 4, cell.widthsFine[0]);
    setF32(words, base + 5, cell.widthsFine[1]);
    setF32(words, base + 6, cell.widthsFine[2]);
    words[base + 7] = cell.minimumFine[0];
    words[base + 8] = cell.minimumFine[1];
    words[base + 9] = cell.minimumFine[2];
    words[base + 10] = cell.widthsFine[0];
    const brickIndex = brickIndexByKey.get(cell.brickKey);
    if (brickIndex === undefined) throw new Error(`Sparse CM12 cell ${cell.id} has no brick`);
    words[base + 11] = brickIndex;
  }

  nextTerm = 0;
  const rowKinds = { "intra-brick": 0, "brick-face": 1, "mixed-seam": 2, "sparse-air": 3 } as const;
  for (const row of grid.gradientRows) {
    const base = rowOffset + 12 * row.id;
    words[base] = nextTerm;
    words[base + 1] = row.terms.length;
    words[base + 2] = row.axis;
    words[base + 3] = rowKinds[row.kind];
    setF32(words, base + 4, row.dualWeight);
    setF32(words, base + 5, row.area);
    setF32(words, base + 6, row.distance);
    setF32(words, base + 7, row.exteriorPhi ?? 0.5);
    setF32(words, base + 8, row.centerFine[0]);
    setF32(words, base + 9, row.centerFine[1]);
    setF32(words, base + 10, row.centerFine[2]);
    for (const term of row.terms) {
      words[termOffset + 2 * nextTerm] = term.cellId;
      setF32(words, termOffset + 2 * nextTerm + 1, term.coefficient);
      nextTerm += 1;
    }
  }

  let nextIncidence = 0;
  for (let cell = 0; cell < byCell.length; cell += 1) {
    words[incidenceOffset + cell] = nextIncidence;
    for (const incidence of byCell[cell]!) {
      words[incidenceRecordOffset + 2 * nextIncidence] = incidence.row;
      words[incidenceRecordOffset + 2 * nextIncidence + 1] = incidence.term;
      nextIncidence += 1;
    }
  }
  words[incidenceOffset + grid.cells.length] = nextIncidence;

  words.fill(INVALID, ownerOffset);
  const [nx, ny] = atlas.dimensions;
  const background = packAdaptiveMassPresentationOwnerKey(
    [0, 0, 0], Math.min(1022, Math.max(atlas.dimensions[0], atlas.dimensions[2])),
  );
  words[backgroundOwnerOffset] = background[0];
  words[backgroundOwnerOffset + 1] = background[1];
  for (let dense = 0; dense < denseCount; dense += 1) {
    words[ownerOffset + 4 * dense + 1] = background[0];
    words[ownerOffset + 4 * dense + 2] = background[1];
  }
  const owner = [0, 0] as [number, number];
  const firstCellByBrick = new Uint32Array(atlas.bricks.length).fill(INVALID);
  const cellCountByBrick = new Uint32Array(atlas.bricks.length);
  for (const cell of grid.cells) {
    const brickIndex = brickIndexByKey.get(cell.brickKey);
    if (brickIndex === undefined) {
      throw new Error(`Sparse CM12 cell ${cell.id} has no resident brick`);
    }
    if (firstCellByBrick[brickIndex] === INVALID) firstCellByBrick[brickIndex] = cell.id;
    if (cell.id !== firstCellByBrick[brickIndex] + cellCountByBrick[brickIndex]) {
      throw new Error(`Sparse CM12 brick ${cell.brickKey} cells are not contiguous`);
    }
    cellCountByBrick[brickIndex] += 1;
    packAdaptiveMassPresentationOwnerKey(cell.minimumFine, cell.widthsFine[0], owner);
    for (let z = cell.minimumFine[2]; z < cell.maximumFine[2]; z += 1)
      for (let y = cell.minimumFine[1]; y < cell.maximumFine[1]; y += 1)
        for (let x = cell.minimumFine[0]; x < cell.maximumFine[0]; x += 1) {
          const dense = x + nx * (y + ny * z);
          words[ownerOffset + 4 * dense] = cell.id;
          words[ownerOffset + 4 * dense + 1] = owner[0];
          words[ownerOffset + 4 * dense + 2] = owner[1];
          words[ownerOffset + 4 * dense + 3] = brickIndex;
        }
  }
  for (let brick = 0; brick < atlas.bricks.length; brick += 1) {
    const record = brickOffset + 4 * brick;
    words[record] = firstCellByBrick[brick];
    words[record + 1] = cellCountByBrick[brick];
    words[record + 2] = atlas.bricks[brick]!.resolution;
    words[record + 3] = atlas.bricks[brick]!.key;
    words[brickDirectoryOffset + atlas.bricks[brick]!.key] = brick;
  }
  return { words, cellOffset, rowOffset, termOffset, incidenceOffset,
    incidenceRecordOffset, ownerOffset, brickOffset, backgroundOwnerOffset,
    brickDirectoryOffset, brickCount: atlas.bricks.length, incidenceCount };
}

function uploadBuffer(
  device: GPUDevice,
  label: string,
  source: ArrayBufferView,
  usage: GPUBufferUsageFlags,
): GPUBuffer {
  const buffer = device.createBuffer({ label, size: Math.max(4, source.byteLength), usage });
  device.queue.writeBuffer(buffer, 0, source.buffer as ArrayBuffer,
    source.byteOffset, source.byteLength);
  return buffer;
}

/** Static compact topology plus fully device-resident evolving frame state. */
export class WebGPUSparseCM12Resident {
  readonly cellCount: number;
  readonly rowCount: number;
  readonly allocatedBytes: number;
  private readonly parameters: GPUBuffer;
  private readonly topology: GPUBuffer;
  private readonly state: GPUBuffer;
  private readonly partials: GPUBuffer;
  private readonly scalars: GPUBuffer;
  private readonly conditioning: GPUBuffer;
  private readonly activity: GPUBuffer;
  private readonly diagnosticsReadback: GPUBuffer;
  private readonly bindGroup: GPUBindGroup;
  private readonly pipelines: Readonly<Record<string, GPUComputePipeline>>;
  private readonly parameterWords = new ArrayBuffer(256);
  private readonly parameterU32 = new Uint32Array(this.parameterWords);
  private readonly parameterF32 = new Float32Array(this.parameterWords);
  private parity = 0;
  private destroyed = false;

  private constructor(
    private readonly device: GPUDevice,
    private readonly dimensions: readonly [number, number, number],
    private readonly layout: ResidentStateLayout,
    buffers: readonly [GPUBuffer, GPUBuffer, GPUBuffer, GPUBuffer, GPUBuffer, GPUBuffer,
      GPUBuffer],
    diagnosticsReadback: GPUBuffer,
    bindGroup: GPUBindGroup,
    pipelines: Readonly<Record<string, GPUComputePipeline>>,
    cellCount: number,
    rowCount: number,
    private horizontalD4Authority: boolean,
  ) {
    [this.parameters, this.topology, this.state, this.partials, this.scalars,
      this.conditioning, this.activity] = buffers;
    this.diagnosticsReadback = diagnosticsReadback;
    this.bindGroup = bindGroup;
    this.pipelines = pipelines;
    this.cellCount = cellCount;
    this.rowCount = rowCount;
    this.allocatedBytes = buffers.reduce((sum, buffer) => sum + buffer.size, 0)
      + diagnosticsReadback.size;
  }

  static async create(
    device: GPUDevice,
    atlas: SparseAdaptiveMassAtlas,
    grid: SparseAtlasCompositeGrid,
    presentation: WebGPUAdaptiveMassAtlasPresentation,
    initiallyActiveBrickKeys: ReadonlySet<number> = new Set(atlas.bricks.map(
      (brick) => brick.key,
    )),
  ): Promise<WebGPUSparseCM12Resident> {
    const packed = packResidentTopology(atlas, grid);
    const layout = residentStateLayout(
      grid.cells.length, grid.gradientRows.length, packed.brickCount,
    );
    const initialState = new Float32Array(layout.floatCount);
    for (const cell of grid.cells) {
      initialState[layout.densityA + cell.id] = cell.density;
      initialState[layout.densityB + cell.id] = cell.density;
      initialState[layout.gammaA + cell.id] = cell.gamma;
      initialState[layout.gammaB + cell.id] = cell.gamma;
      initialState[layout.liquid + cell.id] = cell.density >= 0.5 ? 1 : 0;
    }
    const horizontalD4Authority = sparseAtlasScalarsHaveHorizontalD4Symmetry(
      grid,
      Float64Array.from(grid.cells, (cell) => cell.density),
      Float64Array.from(grid.cells, (cell) => cell.gamma),
    );
    const cellWorkgroups = Math.ceil(grid.cells.length / WORKGROUP_SIZE);
    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    const parameters = device.createBuffer({ label: "Sparse CM12 resident parameters",
      size: 256, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const topology = uploadBuffer(device, "Sparse CM12 resident topology", packed.words, storage);
    const state = uploadBuffer(device, "Sparse CM12 resident state", initialState, storage);
    const partials = device.createBuffer({ label: "Sparse CM12 resident reductions",
      size: Math.max(8, 8 * cellWorkgroups), usage: storage });
    const scalars = device.createBuffer({ label: "Sparse CM12 resident scalar reductions",
      size: 32, usage: storage });
    const conditioning = device.createBuffer({
      label: "Sparse CM12 conservative transport and conditioning accumulators",
      size: Math.max(4, 4 * grid.cells.length * 4),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const initialActivity = new Uint32Array(ACTIVITY_HEADER_WORDS
      + ACTIVITY_RECORD_WORDS * packed.brickCount);
    initialActivity[8] = initiallyActiveBrickKeys.size;
    for (let brick = 0; brick < packed.brickCount; brick += 1) {
      const at = ACTIVITY_HEADER_WORDS + ACTIVITY_RECORD_WORDS * brick;
      initialActivity[at + 8] = atlas.bricks[brick]!.resolution;
      initialActivity[at + 9] = 32; // retained until the first GPU topology epoch
      initialActivity[at + 10] = initiallyActiveBrickKeys.has(atlas.bricks[brick]!.key)
        ? 1 : 0;
      if (initialActivity[at + 10] !== 0) {
        initialActivity[11] += packed.words[packed.brickOffset + 4 * brick + 1]!;
      }
    }
    const activity = uploadBuffer(device, "Sparse CM12 resident activity history",
      initialActivity, storage);
    const diagnosticsReadback = device.createBuffer({
      label: "Sparse CM12 resident diagnostic readback",
      size: 80,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    const bindGroupLayout = device.createBindGroupLayout({
      label: "Sparse CM12 resident layout",
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        ...[2, 3, 4].map((binding) => ({ binding, visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "storage" as const } })),
        { binding: 5, visibility: GPUShaderStage.COMPUTE,
          storageTexture: { access: "write-only", format: "r32float", viewDimension: "3d" } },
        { binding: 6, visibility: GPUShaderStage.COMPUTE,
          storageTexture: { access: "write-only", format: "r32float", viewDimension: "3d" } },
        { binding: 7, visibility: GPUShaderStage.COMPUTE,
          storageTexture: { access: "write-only", format: "rg32uint", viewDimension: "3d" } },
        { binding: 8, visibility: GPUShaderStage.COMPUTE,
          storageTexture: { access: "write-only", format: "rgba32float", viewDimension: "3d" } },
        { binding: 9, visibility: GPUShaderStage.COMPUTE,
          storageTexture: { access: "write-only", format: "r32float", viewDimension: "3d" } },
        { binding: 10, visibility: GPUShaderStage.COMPUTE,
          storageTexture: { access: "write-only", format: "r32float", viewDimension: "3d" } },
        { binding: 11, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 12, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      ],
    });
    const bindGroup = device.createBindGroup({ label: "Sparse CM12 resident bindings",
      layout: bindGroupLayout, entries: [
        { binding: 0, resource: { buffer: parameters } },
        { binding: 1, resource: { buffer: topology } },
        { binding: 2, resource: { buffer: state } },
        { binding: 3, resource: { buffer: partials } },
        { binding: 4, resource: { buffer: scalars } },
        { binding: 5, resource: presentation.densityTexture.createView() },
        { binding: 6, resource: presentation.levelSetTexture.createView() },
        { binding: 7, resource: presentation.gridCellTexture.createView() },
        { binding: 8, resource: presentation.velocityTexture.createView() },
        { binding: 9, resource: presentation.pressureTexture.createView() },
        { binding: 10, resource: presentation.divergenceTexture.createView() },
        { binding: 11, resource: { buffer: conditioning } },
        { binding: 12, resource: { buffer: activity } },
      ] });
    const shaderModule = device.createShaderModule({ label: "Sparse CM12 resident shader",
      code: webgpuSparseCM12ResidentWGSL });
    const pipelineLayout = device.createPipelineLayout({ label: "Sparse CM12 resident pipeline layout",
      bindGroupLayouts: [bindGroupLayout] });
    const names = ["injectLiquid", "initializeTransportVelocity",
      "extrapolateTransportVelocityToSource", "extrapolateTransportVelocityToDestination",
      "prepareTransportFaces", "traceGammaAndBeta", "scatterDensityDeficit",
      "gatherConservativeDensity", "diffuseGammaForwardX", "diffuseGammaForwardY",
      "diffuseGammaForwardZ", "diffuseGammaReverseZ", "diffuseGammaReverseY",
      "diffuseGammaReverseX", "averageGammaDiffusion", "scatterSharpeningMass",
      "acceptSharpeningMass", "finalizeSharpening", "preserveHorizontalD4",
      "commitHorizontalD4",
      "forceFaces", "classifyRows", "preparePressure",
      "initializePCG", "reduceInitialize", "applyDirection", "reduceCurvature",
      "updateResidual", "reduceResidual", "updateDirection", "projectFaces",
      "collocateAndDiagnose", "measureDivergenceDiagnostics",
      "reduceDivergenceDiagnostics",
      "advanceActivityClock", "measureBrickActivity",
      "planBrickResolution", "activateSweptReceivers", "closePlannedResolution",
      "retireUnsupportedEmptyBricks",
      "classifyPresentationBricks", "publishPresentation"] as const;
    const entries = await Promise.all(names.map(async (name) => [name,
      await device.createComputePipelineAsync({ label: `Sparse CM12 ${name}`,
        layout: pipelineLayout, compute: { module: shaderModule, entryPoint: name } })] as const));
    const result = new WebGPUSparseCM12Resident(device, atlas.dimensions, layout,
      [parameters, topology, state, partials, scalars, conditioning, activity],
      diagnosticsReadback,
      bindGroup,
      Object.fromEntries(entries), grid.cells.length, grid.gradientRows.length,
      horizontalD4Authority);
    result.writeParameters(packed, 0.004, 1, 1, [0, 0, 0]);
    return result;
  }

  encode(
    encoder: GPUCommandEncoder,
    dt_s: number,
    finestCellSize_m: number,
    pressureScale: number,
    accelerationFinePerSecond2: readonly [number, number, number],
  ): void {
    this.assertLive();
    const packed = this.lastPacked!;
    this.writeParameters(packed, dt_s, finestCellSize_m, pressureScale,
      accelerationFinePerSecond2);
    encoder.clearBuffer(this.conditioning);
    const pass = encoder.beginComputePass({ label: "Sparse CM12 resident frame" });
    const dispatch = (name: string, count: number) => {
      pass.setPipeline(this.pipelines[name]!); pass.setBindGroup(0, this.bindGroup);
      pass.dispatchWorkgroups(count);
    };
    const cells = Math.ceil(this.cellCount / WORKGROUP_SIZE);
    const rows = Math.ceil(this.rowCount / WORKGROUP_SIZE);
    dispatch("initializeTransportVelocity", cells);
    for (let sweep = 0; sweep < 8; sweep += 1) {
      dispatch(sweep % 2 === 0
        ? "extrapolateTransportVelocityToSource"
        : "extrapolateTransportVelocityToDestination", cells);
    }
    dispatch("prepareTransportFaces", rows);
    dispatch("traceGammaAndBeta", cells);
    dispatch("scatterDensityDeficit", cells);
    dispatch("gatherConservativeDensity", cells);
    dispatch("diffuseGammaForwardX", cells);
    dispatch("diffuseGammaForwardY", cells);
    dispatch("diffuseGammaForwardZ", cells);
    dispatch("diffuseGammaReverseZ", cells);
    dispatch("diffuseGammaReverseY", cells);
    dispatch("diffuseGammaReverseX", cells);
    dispatch("averageGammaDiffusion", cells);
    dispatch("scatterSharpeningMass", cells);
    dispatch("acceptSharpeningMass", cells);
    dispatch("finalizeSharpening", cells);
    if (this.horizontalD4Authority) {
      dispatch("preserveHorizontalD4", cells);
      dispatch("commitHorizontalD4", cells);
    }
    dispatch("forceFaces", rows);
    dispatch("preparePressure", cells);
    dispatch("classifyRows", rows);
    dispatch("preparePressure", cells);
    dispatch("initializePCG", cells);
    dispatch("reduceInitialize", 1);
    for (let iteration = 0; iteration < PCG_ITERATIONS; iteration += 1) {
      dispatch("applyDirection", cells);
      dispatch("reduceCurvature", 1);
      dispatch("updateResidual", cells);
      dispatch("reduceResidual", 1);
      dispatch("updateDirection", cells);
    }
    dispatch("projectFaces", rows);
    dispatch("collocateAndDiagnose", cells);
    dispatch("measureDivergenceDiagnostics", cells);
    dispatch("reduceDivergenceDiagnostics", 1);
    dispatch("advanceActivityClock", 1);
    dispatch("measureBrickActivity", this.lastPacked!.brickCount);
    dispatch("planBrickResolution", Math.ceil(this.lastPacked!.brickCount / WORKGROUP_SIZE));
    dispatch("activateSweptReceivers", Math.ceil(this.lastPacked!.brickCount / WORKGROUP_SIZE));
    for (let gradingPass = 0; gradingPass < 3; gradingPass += 1) {
      dispatch("closePlannedResolution", Math.ceil(
        this.lastPacked!.brickCount / WORKGROUP_SIZE,
      ));
    }
    dispatch("retireUnsupportedEmptyBricks",
      Math.ceil(this.lastPacked!.brickCount / WORKGROUP_SIZE));
    dispatch("classifyPresentationBricks", Math.ceil(this.lastPacked!.brickCount / WORKGROUP_SIZE));
    pass.setPipeline(this.pipelines.publishPresentation!);
    pass.setBindGroup(0, this.bindGroup);
    pass.dispatchWorkgroups(
      Math.ceil(this.dimensions[0] / 4),
      Math.ceil(this.dimensions[1] / 4),
      Math.ceil(this.dimensions[2] / 4),
    );
    pass.end();
    this.parity ^= 1;
  }

  /** Publish generation zero without executing a physics step or mapping state. */
  encodeInitialPresentation(encoder: GPUCommandEncoder, finestCellSize_m: number): void {
    this.assertLive();
    this.writeParameters(this.lastPacked!, 0.004, finestCellSize_m, 1, [0, 0, 0]);
    const pass = encoder.beginComputePass({ label: "Sparse CM12 resident initial presentation" });
    pass.setPipeline(this.pipelines.classifyPresentationBricks!);
    pass.setBindGroup(0, this.bindGroup);
    pass.dispatchWorkgroups(Math.ceil(this.lastPacked!.brickCount / WORKGROUP_SIZE));
    pass.setPipeline(this.pipelines.publishPresentation!);
    pass.dispatchWorkgroups(
      Math.ceil(this.dimensions[0] / 4),
      Math.ceil(this.dimensions[1] / 4),
      Math.ceil(this.dimensions[2] / 4),
    );
    pass.end();
  }

  encodeLiquidInjection(
    encoder: GPUCommandEncoder,
    finestCellSize_m: number,
    centerFine: readonly [number, number, number],
    radiusFine: readonly [number, number, number],
  ): void {
    this.assertLive();
    if (Math.abs(centerFine[0] - 0.5 * this.dimensions[0]) > 1e-6
      || Math.abs(centerFine[2] - 0.5 * this.dimensions[2]) > 1e-6
      || Math.abs(radiusFine[0] - radiusFine[2]) > 1e-6) {
      this.horizontalD4Authority = false;
    }
    this.writeParameters(this.lastPacked!, 0.004, finestCellSize_m, 1, [0, 0, 0]);
    this.parameterF32.set([...centerFine, 0], 52);
    this.parameterF32.set([...radiusFine, 0], 56);
    this.device.queue.writeBuffer(this.parameters, 0, this.parameterWords);
    const pass = encoder.beginComputePass({ label: "Sparse CM12 resident liquid injection" });
    pass.setPipeline(this.pipelines.injectLiquid!);
    pass.setBindGroup(0, this.bindGroup);
    pass.dispatchWorkgroups(Math.ceil(this.cellCount / WORKGROUP_SIZE));
    pass.setPipeline(this.pipelines.classifyPresentationBricks!);
    pass.dispatchWorkgroups(Math.ceil(this.lastPacked!.brickCount / WORKGROUP_SIZE));
    pass.setPipeline(this.pipelines.publishPresentation!);
    pass.dispatchWorkgroups(
      Math.ceil(this.dimensions[0] / 4),
      Math.ceil(this.dimensions[1] / 4),
      Math.ceil(this.dimensions[2] / 4),
    );
    pass.end();
  }

  private lastPacked?: PackedResidentTopology;

  private writeParameters(
    packed: PackedResidentTopology,
    dt_s: number,
    finestCellSize_m: number,
    pressureScale: number,
    acceleration: readonly [number, number, number],
  ): void {
    this.lastPacked = packed;
    const u = this.parameterU32, f = this.parameterF32, l = this.layout;
    u.fill(0);
    u.set([this.cellCount, this.rowCount, packed.incidenceCount,
      this.dimensions[0] * this.dimensions[1] * this.dimensions[2]], 0);
    u.set([...this.dimensions, 0], 4);
    u.set([packed.cellOffset, packed.rowOffset, packed.termOffset, packed.incidenceOffset], 8);
    u.set([packed.incidenceRecordOffset, packed.ownerOffset,
      packed.brickOffset, packed.backgroundOwnerOffset], 12);
    u.set([l.densityA, l.densityB, l.gammaA, l.gammaB], 16);
    u.set([l.cellVelocityA, l.cellVelocityB, l.faceA, l.faceB], 20);
    u.set([l.pressure, l.rhs, l.diagonal, l.liquid], 24);
    u.set([l.theta, l.residual, l.preconditioned, l.direction], 28);
    u.set([l.applied, l.divergence, l.presentationBrickWet,
      packed.brickDirectoryOffset], 32);
    u.set([l.sharpeningDelta, l.sharpeningAcceptance, 0, 0], 36);
    f.set([dt_s, finestCellSize_m, pressureScale, this.parity], 40);
    f.set([...acceleration, 0], 44);
    u.set([Math.ceil(this.cellCount / WORKGROUP_SIZE),
      Math.ceil(this.rowCount / WORKGROUP_SIZE), PCG_ITERATIONS, packed.brickCount], 48);
    f.set([0, 0, 0, 0, 1, 1, 1, 0], 52);
    this.device.queue.writeBuffer(this.parameters, 0, this.parameterWords);
  }

  async readDiagnostics(): Promise<{
    readonly pressureRelativeResidual: number;
    readonly maximumDivergence_s: number;
    readonly maximumMixedSeamDivergence_s: number;
    readonly activityMaximumScore: number;
    readonly activitySurfaceBrickCount: number;
    readonly activityHotBrickCount: number;
    readonly activityQuietBrickCount: number;
    readonly activityTopologyEpoch: boolean;
    readonly activityMeasuredBrickCount: number;
    readonly activeBrickCount: number;
    readonly newlyActivatedBrickCount: number;
    readonly residencyGeneration: number;
    readonly activeCellCount: number;
  }> {
    this.assertLive();
    const encoder = this.device.createCommandEncoder({
      label: "Sparse CM12 diagnostic scalar readback",
    });
    encoder.copyBufferToBuffer(this.scalars, 0, this.diagnosticsReadback, 0, 32);
    encoder.copyBufferToBuffer(this.activity, 0, this.diagnosticsReadback, 32, 48);
    this.device.queue.submit([encoder.finish()]);
    await this.diagnosticsReadback.mapAsync(GPUMapMode.READ);
    const mapped = this.diagnosticsReadback.getMappedRange();
    const values = new Float32Array(mapped, 0, 8);
    const activity = new Uint32Array(mapped, 32, ACTIVITY_HEADER_WORDS);
    const rhsSquared = values[1]!;
    const residualSquared = values[4]!;
    const result = {
      pressureRelativeResidual: Math.sqrt(Math.max(0, residualSquared)
        / Math.max(rhsSquared, Number.MIN_VALUE)),
      maximumDivergence_s: values[6]!,
      maximumMixedSeamDivergence_s: values[7]!,
      activityMaximumScore: activity[1]!,
      activitySurfaceBrickCount: activity[2]!,
      activityHotBrickCount: activity[3]!,
      activityQuietBrickCount: activity[4]!,
      activityTopologyEpoch: activity[5] !== 0,
      activityMeasuredBrickCount: activity[6]!,
      activeBrickCount: activity[8]!,
      newlyActivatedBrickCount: activity[9]!,
      residencyGeneration: activity[10]!,
      activeCellCount: activity[11]!,
    };
    this.diagnosticsReadback.unmap();
    return result;
  }

  /** QA-only policy readback. It is never called by frame scheduling. */
  async readActivitySnapshot(): Promise<SparseCM12GPUActivitySnapshot> {
    this.assertLive();
    const readback = this.device.createBuffer({
      label: "Sparse CM12 activity QA readback",
      size: this.activity.size,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    try {
      const encoder = this.device.createCommandEncoder({
        label: "Sparse CM12 activity QA copy",
      });
      encoder.copyBufferToBuffer(this.activity, 0, readback, 0, this.activity.size);
      this.device.queue.submit([encoder.finish()]);
      await readback.mapAsync(GPUMapMode.READ);
      const words = new Uint32Array(readback.getMappedRange());
      const records = Array.from({ length: this.lastPacked!.brickCount }, (_, brick) => {
        const at = ACTIVITY_HEADER_WORDS + ACTIVITY_RECORD_WORDS * brick;
        return {
          scoreByte: words[at]!,
          reasons: words[at + 1]!,
          hotEpochs: words[at + 2]!,
          quietEpochs: words[at + 3]!,
          plannedResolution: (words[at + 8] === 8 ? 8
            : words[at + 8] === 4 ? 4 : words[at + 8] === 2 ? 2 : 1) as
              SparseCM12GPUActivityRecord["plannedResolution"],
          planReasons: words[at + 9]!,
          active: words[at + 10] !== 0,
          activatedStep: words[at + 11]!,
        };
      });
      return { acceptedSteps: words[0]!, records };
    } finally {
      if (readback.mapState === "mapped") readback.unmap();
      readback.destroy();
    }
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    for (const buffer of [this.parameters, this.topology, this.state, this.partials,
      this.scalars, this.conditioning, this.activity]) {
      buffer.destroy();
    }
    this.diagnosticsReadback.destroy();
  }

  private assertLive(): void {
    if (this.destroyed) throw new Error("Sparse CM12 resident pipeline is destroyed");
  }
}
