import type { SparseAtlasCompositeGrid } from
  "../../methods/adaptive-mass/sparse-atlas-composite-projection";
import type { SparseAdaptiveMassAtlas } from
  "../../methods/adaptive-mass/sparse-brick-atlas";
import { sparseCM12TileClonePoolCapacity } from
  "../../methods/adaptive-mass/sparse-cm12-tile-clone-pool";
import type { SparseCM12RigidResources } from
  "../../methods/adaptive-mass/webgpu-sparse-cm12-rigid-coupling";
import {
  WebGPUSparseCM12Resident,
  type SharpeningTrace,
  type SparseCM12ActivityPolicy,
  type SparseCM12InflowControl,
  type SparseCM12PresentationPageResolution,
  type SparseCM12PressureControl,
  type SparseCM12PressureJournalCapacityRequest,
  type SparseCM12ResidentInitializationReporter,
  type SparseCM12ResidentStageSeams,
} from "../../methods/adaptive-mass/webgpu-sparse-cm12-resident";
import type {
  SparseWorld,
  SparseWorldDevice,
  SparseWorldFault,
  SparseWorldPresentation,
  SparseWorldStatus,
  SparseWorldStep,
  SparseWorldStepInput,
  SparseWorldTrace,
  SparseWorldUI,
  SparseWorldUIControl,
  SparseWorldUIDiagnostics,
  SparseWorldUIOverlays,
} from "../index";
import {
  CM12SparseWorldDevice,
  type CM12ResidentLibraryReadiness,
} from "./cm12-device-library";

export interface CM12SparseWorldStepConfiguration {
  readonly finestCellSize_m: number;
  readonly pressureScale: number;
  /** Defaults to `input.gravity / finestCellSize_m`. */
  readonly accelerationFinePerSecond2?: readonly [number, number, number];
  readonly sharpening?: SharpeningTrace;
  readonly activityPolicy?: SparseCM12ActivityPolicy;
  readonly pressureControl?: SparseCM12PressureControl;
  readonly seams?: SparseCM12ResidentStageSeams;
  readonly bodyCount?: number;
  readonly worldDimensions_m?: readonly [number, number, number];
  readonly inflow?: SparseCM12InflowControl;
}

export type CM12SparseWorldNumerics =
  | CM12SparseWorldStepConfiguration
  | ((input: SparseWorldStepInput) => CM12SparseWorldStepConfiguration);

interface AdoptCM12SparseWorldOptions {
  readonly numerics: CM12SparseWorldNumerics;
  readonly residentTiles: number;
  readonly capacityTiles: number;
  readonly trace?: SparseWorldTrace;
}

export type CM12SparseWorldResidentFactoryMode =
  | "production"
  | "presentation-publisher-qa"
  | "legacy-host-authority-qa"
  | "phase1-transport-receipt-qa";

/**
 * Temporary construction input used while atlas creation still lives above
 * the sparse-world module. New callers should depend on `SparseWorldConfig`.
 */
export interface LegacyCM12SparseWorldFactoryConfig {
  readonly device: GPUDevice;
  readonly atlas: SparseAdaptiveMassAtlas;
  readonly grid: SparseAtlasCompositeGrid;
  readonly numerics: CM12SparseWorldNumerics;
  readonly trace?: SparseWorldTrace;
  readonly initiallyActiveBrickKeys?: ReadonlySet<number>;
  readonly rigid?: SparseCM12RigidResources;
  readonly journal?: SparseCM12PressureJournalCapacityRequest;
  readonly presentationPageResolution?: SparseCM12PresentationPageResolution;
  readonly report?: SparseCM12ResidentInitializationReporter;
  readonly mode?: CM12SparseWorldResidentFactoryMode;
  /** Defaults to the authored active set. */
  readonly residentTiles?: number;
  /** Defaults to the capacity policy already used by the CM12 clone pool. */
  readonly capacityTiles?: number;
}

/**
 * Migration-only access to CM12 instruments. It intentionally contains no
 * pipeline map and no way to dispatch a named internal stage.
 */
export interface CM12SparseWorldRuntime {
  /** Deterministic harness seam; application readiness lives on `device`. */
  waitForSimulationPipelines(): Promise<void>;

  readonly allocatedBytes: number;
  readonly cellCount: number;
  readonly rowCount: number;
  readonly tracerSource: WebGPUSparseCM12Resident["tracerSource"];
  readonly faceVelocitySource: WebGPUSparseCM12Resident["faceVelocitySource"];
  readonly pressureJournalSource: WebGPUSparseCM12Resident["pressureJournalSource"];
  readonly pressureJournalLayout: WebGPUSparseCM12Resident["pressureJournalLayout"];
  readonly pressureJournalArmed: boolean;
  readonly stageLensSource: WebGPUSparseCM12Resident["stageLensSource"];

  setTracersEnabled(enabled: boolean): void;
  reseedTracers(): void;
  readTracers(): ReturnType<WebGPUSparseCM12Resident["readTracers"]>;
  armPressureJournal(armed: boolean): boolean;
  readPressureJournal(): ReturnType<WebGPUSparseCM12Resident["readPressureJournal"]>;
  setRefinementRegionParameters(parameters: ArrayBuffer): void;
  encodeInitialPresentation(encoder: GPUCommandEncoder, finestCellSize_m: number): void;
  encodeLiquidInjection(
    encoder: GPUCommandEncoder,
    finestCellSize_m: number,
    centerFine: readonly [number, number, number],
    radiusFine: readonly [number, number, number],
  ): void;
  encodeLiquidJetInjection(
    encoder: GPUCommandEncoder,
    finestCellSize_m: number,
    outletFine: readonly [number, number, number],
    radiusFine: number,
    velocityFinePerSecond: readonly [number, number, number],
    dt_s: number,
  ): void;
  encodePressureIterationReceipt(
    encoder: GPUCommandEncoder,
    destination: GPUBuffer,
  ): void;

}

/**
 * Explicit developer/QA instrumentation. Nothing in frame encoding or normal
 * presentation may depend on this surface.
 */
export interface CM12SparseWorldDeveloperTrace {
  readDiagnostics(): ReturnType<WebGPUSparseCM12Resident["readDiagnostics"]>;
  readDiagnosticFields(): ReturnType<WebGPUSparseCM12Resident["readDiagnosticFields"]>;
  readActivitySnapshot(): ReturnType<WebGPUSparseCM12Resident["readActivitySnapshot"]>;
  readPresentationPageAllocatorReceiptQA(): ReturnType<
    WebGPUSparseCM12Resident["readPresentationPageAllocatorReceiptQA"]>;
  readPhase1TransportReceiptQA(): ReturnType<
    WebGPUSparseCM12Resident["readPhase1TransportReceiptQA"]>;
  readPhase1TransportProfileQA(): ReturnType<
    WebGPUSparseCM12Resident["readPhase1TransportProfileQA"]>;
  readCandidateEffectsTransactionQA(): ReturnType<
    WebGPUSparseCM12Resident["readCandidateEffectsTransactionQA"]>;
  readFrameControlQA(): ReturnType<WebGPUSparseCM12Resident["readFrameControlQA"]>;
  readTransportPacketIndirectQA(): ReturnType<
    WebGPUSparseCM12Resident["readTransportPacketIndirectQA"]>;
  readFinalScalarMaskHeaderQA(): ReturnType<
    WebGPUSparseCM12Resident["readFinalScalarMaskHeaderQA"]>;
  readWorkShapeQA(): ReturnType<WebGPUSparseCM12Resident["readWorkShapeQA"]>;
  readAdaptiveRepresentationQA(): ReturnType<
    WebGPUSparseCM12Resident["readAdaptiveRepresentationQA"]>;
  readAcceptedIndirectQA(): ReturnType<WebGPUSparseCM12Resident["readAcceptedIndirectQA"]>;
  readFrameControlIndirectQA(): ReturnType<
    WebGPUSparseCM12Resident["readFrameControlIndirectQA"]>;
  readVelocityExtensionHeaderQA(): ReturnType<
    WebGPUSparseCM12Resident["readVelocityExtensionHeaderQA"]>;
  readVelocityExtensionQA(): ReturnType<
    WebGPUSparseCM12Resident["readVelocityExtensionQA"]>;
  readPressureCanonicalMembershipQA(): ReturnType<
    WebGPUSparseCM12Resident["readPressureCanonicalMembershipQA"]>;
}

export interface CM12SparseWorldAdoption {
  readonly device: SparseWorldDevice;
  readonly world: SparseWorld;
  readonly ui: SparseWorldUI;
  readonly runtime: CM12SparseWorldRuntime;
  readonly developerTrace: CM12SparseWorldDeveloperTrace;
}

const fault = (code: SparseWorldFault["code"], error: unknown): SparseWorldFault => ({
  code,
  message: error instanceof Error ? error.message : String(error),
  cause: error,
});

class AdoptedCM12SparseWorld implements SparseWorld {
  private generation: number;
  private lastAcceptedTime = 0;
  private state: SparseWorldStatus["state"] = "ready";
  private currentFault: SparseWorldFault | undefined;
  private destroyed = false;

  constructor(
    private readonly resident: WebGPUSparseCM12Resident,
    private readonly options: AdoptCM12SparseWorldOptions,
    private readonly device: SparseWorldDevice,
  ) {
    this.generation = resident.globalFineLevelSetSource.generation;
    options.trace?.record({
      kind: "world-created",
      residentTiles: options.residentTiles,
      capacityTiles: options.capacityTiles,
    });
  }

  encodeStep(encoder: GPUCommandEncoder, input: SparseWorldStepInput): SparseWorldStep {
    if (this.destroyed) {
      const error = new Error("Sparse world has been destroyed");
      this.publishFault("world-destroyed", error);
      throw error;
    }
    if (!(input.dt > 0) || !Number.isFinite(input.time)) {
      const error = new RangeError("Sparse world step requires a positive dt and finite time");
      this.publishFault("step-encoding", error);
      throw error;
    }
    const configuration = typeof this.options.numerics === "function"
      ? this.options.numerics(input) : this.options.numerics;
    try {
      for (const interaction of input.interactions) {
        if (interaction.kind !== "liquid-ellipsoid") {
          throw new Error(`Unsupported sparse-world interaction: ${interaction.kind}`);
        }
        const inverseCell = 1 / configuration.finestCellSize_m;
        this.resident.encodeLiquidInjection(
          encoder,
          configuration.finestCellSize_m,
          interaction.center_m.map((value) => value * inverseCell) as
            [number, number, number],
          interaction.radii_m.map((value) => value * inverseCell) as
            [number, number, number],
        );
      }
      const acceleration = configuration.accelerationFinePerSecond2 ?? [
        input.gravity[0] / configuration.finestCellSize_m,
        input.gravity[1] / configuration.finestCellSize_m,
        input.gravity[2] / configuration.finestCellSize_m,
      ] as const;
      this.resident.encode(
        encoder,
        input.dt,
        configuration.finestCellSize_m,
        configuration.pressureScale,
        acceleration,
        configuration.sharpening,
        configuration.activityPolicy,
        configuration.pressureControl,
        configuration.seams,
        configuration.bodyCount,
        configuration.worldDimensions_m,
        configuration.inflow,
      );
      this.generation += 1;
      this.lastAcceptedTime = input.time;
      this.state = "running";
      const step = Object.freeze({
        generation: this.generation,
        submittedTime: input.time,
      });
      this.options.trace?.record({ kind: "step-encoded", ...step });
      return step;
    } catch (error) {
      this.publishFault("step-encoding", error);
      throw error;
    }
  }

  presentation(): SparseWorldPresentation {
    return Object.freeze({
      acceptedGeneration: this.generation,
      fineLevelSet: this.resident.globalFineLevelSetSource,
      adaptiveGrid: this.resident.sparseAdaptiveGridSource,
    });
  }

  status(): SparseWorldStatus {
    if (this.device.fault && !this.currentFault) {
      this.currentFault = this.device.fault;
      this.state = "fault";
    }
    return Object.freeze({
      state: this.state,
      acceptedGeneration: this.generation,
      residentTiles: this.options.residentTiles,
      capacityTiles: this.options.capacityTiles,
      lastAcceptedTime: this.lastAcceptedTime,
      ...(this.currentFault ? { fault: this.currentFault } : {}),
    });
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.resident.destroy();
  }

  private publishFault(code: SparseWorldFault["code"], error: unknown): void {
    this.currentFault = fault(code, error);
    this.state = "fault";
    this.options.trace?.record({ kind: "fault", fault: this.currentFault });
  }
}

class AdoptedCM12SparseWorldRuntime implements CM12SparseWorldRuntime {
  constructor(
    private readonly resident: WebGPUSparseCM12Resident,
    private readonly readiness: CM12ResidentLibraryReadiness,
  ) {}

  waitForSimulationPipelines() { return this.readiness.ready; }
  get allocatedBytes() { return this.resident.allocatedBytes; }
  get cellCount() { return this.resident.cellCount; }
  get rowCount() { return this.resident.rowCount; }
  get tracerSource() { return this.resident.tracerSource; }
  get faceVelocitySource() { return this.resident.faceVelocitySource; }
  get pressureJournalSource() { return this.resident.pressureJournalSource; }
  get pressureJournalLayout() { return this.resident.pressureJournalLayout; }
  get pressureJournalArmed() { return this.resident.pressureJournalArmed; }
  get stageLensSource() { return this.resident.stageLensSource; }
  setTracersEnabled(enabled: boolean) { this.resident.setTracersEnabled(enabled); }
  reseedTracers() { this.resident.reseedTracers(); }
  readTracers() { return this.resident.readTracers(); }
  armPressureJournal(armed: boolean) { return this.resident.armPressureJournal(armed); }
  readPressureJournal() { return this.resident.readPressureJournal(); }
  setRefinementRegionParameters(parameters: ArrayBuffer) {
    this.resident.setRefinementRegionParameters(parameters);
  }
  encodeInitialPresentation(encoder: GPUCommandEncoder, finestCellSize_m: number) {
    this.resident.encodeInitialPresentation(encoder, finestCellSize_m);
  }
  encodeLiquidInjection(encoder: GPUCommandEncoder, finestCellSize_m: number,
    centerFine: readonly [number, number, number],
    radiusFine: readonly [number, number, number]) {
    this.resident.encodeLiquidInjection(encoder, finestCellSize_m, centerFine, radiusFine);
  }
  encodeLiquidJetInjection(encoder: GPUCommandEncoder, finestCellSize_m: number,
    outletFine: readonly [number, number, number], radiusFine: number,
    velocityFinePerSecond: readonly [number, number, number], dt_s: number) {
    this.resident.encodeLiquidJetInjection(encoder, finestCellSize_m, outletFine,
      radiusFine, velocityFinePerSecond, dt_s);
  }
  encodePressureIterationReceipt(encoder: GPUCommandEncoder, destination: GPUBuffer) {
    this.resident.encodePressureIterationReceipt(encoder, destination);
  }
}

class AdoptedCM12SparseWorldDeveloperTrace implements CM12SparseWorldDeveloperTrace {
  constructor(private readonly resident: WebGPUSparseCM12Resident) {}

  readDiagnostics() { return this.resident.readDiagnostics(); }
  readDiagnosticFields() { return this.resident.readDiagnosticFields(); }
  readActivitySnapshot() { return this.resident.readActivitySnapshot(); }
  readPresentationPageAllocatorReceiptQA() {
    return this.resident.readPresentationPageAllocatorReceiptQA();
  }
  readPhase1TransportReceiptQA() { return this.resident.readPhase1TransportReceiptQA(); }
  readPhase1TransportProfileQA() { return this.resident.readPhase1TransportProfileQA(); }
  readCandidateEffectsTransactionQA() {
    return this.resident.readCandidateEffectsTransactionQA();
  }
  readFrameControlQA() { return this.resident.readFrameControlQA(); }
  readTransportPacketIndirectQA() { return this.resident.readTransportPacketIndirectQA(); }
  readFinalScalarMaskHeaderQA() { return this.resident.readFinalScalarMaskHeaderQA(); }
  readWorkShapeQA() { return this.resident.readWorkShapeQA(); }
  readAdaptiveRepresentationQA() { return this.resident.readAdaptiveRepresentationQA(); }
  readAcceptedIndirectQA() { return this.resident.readAcceptedIndirectQA(); }
  readFrameControlIndirectQA() { return this.resident.readFrameControlIndirectQA(); }
  readVelocityExtensionHeaderQA() { return this.resident.readVelocityExtensionHeaderQA(); }
  readVelocityExtensionQA() { return this.resident.readVelocityExtensionQA(); }
  readPressureCanonicalMembershipQA() {
    return this.resident.readPressureCanonicalMembershipQA();
  }
}

class AdoptedCM12SparseWorldUIOverlays implements SparseWorldUIOverlays {
  constructor(private readonly runtime: CM12SparseWorldRuntime) {}
  get tracers() { return this.runtime.tracerSource; }
  get faceVelocity() { return this.runtime.faceVelocitySource; }
  get pressureFilm() { return this.runtime.pressureJournalSource; }
  get stageLenses() { return this.runtime.stageLensSource; }
}

class AdoptedCM12SparseWorldUIDiagnostics implements SparseWorldUIDiagnostics {
  constructor(
    private readonly world: SparseWorld,
    private readonly runtime: CM12SparseWorldRuntime,
  ) {}

  snapshot() {
    return Object.freeze({
      world: this.world.status(),
      allocatedBytes: this.runtime.allocatedBytes,
      storedCells: this.runtime.cellCount,
      constraintRows: this.runtime.rowCount,
    });
  }

  readPressureFilm() { return this.runtime.readPressureJournal(); }
  readTracers() { return this.runtime.readTracers(); }
}

function createCM12SparseWorldUI(
  world: SparseWorld,
  runtime: CM12SparseWorldRuntime,
): SparseWorldUI {
  const pressureFilm = {
    get captureEnabled() { return runtime.pressureJournalArmed; },
    get snapshotCapacity() { return runtime.pressureJournalLayout.snapshotCapacity; },
    setCaptureEnabled: (enabled: boolean) => runtime.armPressureJournal(enabled),
  };
  const control: SparseWorldUIControl = Object.freeze({
    tracers: Object.freeze({
      setEnabled: (enabled: boolean) => runtime.setTracersEnabled(enabled),
      reseed: () => runtime.reseedTracers(),
    }),
    pressureFilm: Object.freeze(pressureFilm),
  });
  return Object.freeze({
    control,
    overlays: new AdoptedCM12SparseWorldUIOverlays(runtime),
    diagnostics: new AdoptedCM12SparseWorldUIDiagnostics(world, runtime),
  });
}

export async function createCM12SparseWorldFromLegacy(
  config: LegacyCM12SparseWorldFactoryConfig,
): Promise<CM12SparseWorldAdoption> {
  const sparseDevice = new CM12SparseWorldDevice(config.device, undefined, config.trace);
  const active = config.initiallyActiveBrickKeys
    ?? new Set(config.atlas.bricks.map((brick) => brick.key));
  const args = [
    config.device,
    config.atlas,
    config.grid,
    typeof config.numerics === "function"
      ? config.numerics({ time: 0, dt: 0.004, gravity: [0, 0, 0], interactions: [] })
        .finestCellSize_m
      : config.numerics.finestCellSize_m,
    active,
    config.rigid,
    config.journal,
    config.presentationPageResolution ?? config.atlas.brickFineResolution,
    config.report,
  ] as const;
  const mode = config.mode ?? "production";
  let resident: WebGPUSparseCM12Resident;
  try {
    resident = mode === "presentation-publisher-qa"
      ? await WebGPUSparseCM12Resident.createPresentationPublisherOracleForQA(...args)
      : mode === "legacy-host-authority-qa"
        ? await WebGPUSparseCM12Resident.createLegacyHostAuthorityOracleForQA(...args)
        : mode === "phase1-transport-receipt-qa"
          ? await WebGPUSparseCM12Resident.createPhase1TransportReceiptOracleForQA(...args)
          : await WebGPUSparseCM12Resident.create(...args);
  } catch (error) {
    sparseDevice.publishLibraryFault(error);
    throw error;
  }
  const readiness = sparseDevice.adoptLegacyResident(resident, config.trace, {
    mode,
    brickFineResolution: config.atlas.brickFineResolution,
    presentationPageResolution:
      config.presentationPageResolution ?? config.atlas.brickFineResolution,
    atlasDimensions: config.atlas.dimensions,
    atlasBricks: config.atlas.bricks.length,
    rigid: config.rigid !== undefined,
    journal: config.journal !== undefined,
  });
  const world = new AdoptedCM12SparseWorld(resident, {
    numerics: config.numerics,
    residentTiles: config.residentTiles ?? active.size,
    capacityTiles: config.capacityTiles ?? sparseCM12TileClonePoolCapacity(active.size),
    trace: config.trace,
  }, sparseDevice);
  const runtime = new AdoptedCM12SparseWorldRuntime(resident, readiness);
  return Object.freeze({
    device: sparseDevice,
    world,
    ui: createCM12SparseWorldUI(world, runtime),
    runtime,
    developerTrace: new AdoptedCM12SparseWorldDeveloperTrace(resident),
  });
}

export { CM12SparseWorldDevice } from "./cm12-device-library";
