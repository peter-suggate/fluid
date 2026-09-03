import type { SparseAtlasCompositeGrid } from
  "../../methods/adaptive-mass/sparse-atlas-composite-projection";
import type { SparseAdaptiveMassAtlas } from
  "../../methods/adaptive-mass/sparse-brick-atlas";
import type { SparseCM12RigidResources } from
  "../../methods/adaptive-mass/webgpu-sparse-cm12-rigid-coupling";
import type { SolidWorld } from "../../core/solid-world";
import {
  fluidSolidWorldForScene,
  solidWorldContentStamp,
} from "../../core/solid-world";
import type { SceneDescription } from "../../core/model";
import {
  refinementRegionLattice,
  sceneRefinementRegions,
} from "../../core/refinement-regions";
import type { WebGPURigidBodySystem } from "../../core/webgpu-rigid-body";
import { packSparseCM12RefinementRegions } from
  "../../methods/adaptive-mass/sparse-cm12-refinement-regions";
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
  SparseWorldEdit,
  SparseWorldEditReceipt,
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

type SparseWorldFluidEdit = Exclude<SparseWorldEdit, { readonly kind: "set-scene" }>;
import {
  CM12SparseWorldDevice,
  type CM12ResidentLibraryReadiness,
} from "./cm12-device-library";

export interface CM12SparseWorldStepConfiguration {
  readonly finestCellSize_m: number;
  readonly pressureScale: number;
  /** World-space origin of fine cell zero. Defaults to `[0, 0, 0]`. */
  readonly origin_m?: readonly [number, number, number];
  /** Defaults to `input.gravity / finestCellSize_m`. */
  readonly accelerationFinePerSecond2?: readonly [number, number, number];
  readonly sharpening?: SharpeningTrace;
  readonly activityPolicy?: SparseCM12ActivityPolicy;
  readonly pressureControl?: SparseCM12PressureControl;
  readonly seams?: SparseCM12ResidentStageSeams;
  readonly worldDimensions_m?: readonly [number, number, number];
}

export type CM12SparseWorldNumerics =
  | CM12SparseWorldStepConfiguration
  | ((input: SparseWorldStepInput) => CM12SparseWorldStepConfiguration);

interface AdoptCM12SparseWorldOptions {
  readonly gpuDevice: GPUDevice;
  readonly numerics: CM12SparseWorldNumerics;
  readonly residentTiles: number;
  readonly capacityTiles: number;
  readonly trace?: SparseWorldTrace;
  readonly rigidSystem?: WebGPURigidBodySystem;
  readonly rigidExchange?: GPUBuffer;
  /** Construction authority used to classify later live scene edits. */
  readonly initialScene: SceneDescription;
}

export type CM12SparseWorldResidentFactoryMode =
  | "production"
  | "presentation-publisher-qa"
  | "phase1-transport-receipt-qa"
  | "velocity-extension-packet-compaction-qa"
  | "coarse-transport-cell-packing-qa"
  | "refinement-policy-leader-compaction-qa"
  | "refinement-policy-full-leaf-qa"
  | "phase1-coarse-transport-cell-packing-qa"
  | "density-capacity-early-exit-qa";

/** Construction input while atlas compilation remains solver-owned. */
export interface CM12SparseWorldFactoryConfig {
  readonly device: GPUDevice;
  /** Scene whose static world and refinement policy produced generation zero. */
  readonly scene: SceneDescription;
  readonly atlas: SparseAdaptiveMassAtlas;
  readonly grid: SparseAtlasCompositeGrid;
  readonly numerics: CM12SparseWorldNumerics;
  readonly trace?: SparseWorldTrace;
  readonly initiallyActiveBrickKeys?: ReadonlySet<number>;
  readonly rigid?: SparseCM12RigidResources;
  readonly rigidSystem?: WebGPURigidBodySystem;
  readonly journal?: SparseCM12PressureJournalCapacityRequest;
  readonly presentationPageResolution?: SparseCM12PresentationPageResolution;
  readonly report?: SparseCM12ResidentInitializationReporter;
  /** Canonical static solid authority for construction and later live edits. */
  readonly solidWorld: SolidWorld;
  /** Packed initial refinement policy, installed without publishing a live edit. */
  readonly refinementRegionParameters: ArrayBuffer;
  readonly mode?: CM12SparseWorldResidentFactoryMode;
  /** Defaults to the authored active set. */
  readonly residentTiles?: number;
  /** Defaults to the resident's physical presentation-page capacity. */
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
  readonly solidWorldCollisionSource:
    WebGPUSparseCM12Resident["solidWorldCollisionSource"];

  setTracersEnabled(enabled: boolean): void;
  reseedTracers(): void;
  readTracers(): ReturnType<WebGPUSparseCM12Resident["readTracers"]>;
  armPressureJournal(armed: boolean): boolean;
  readPressureJournal(): ReturnType<WebGPUSparseCM12Resident["readPressureJournal"]>;
  encodeInitialPresentation(encoder: GPUCommandEncoder, finestCellSize_m: number): void;
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
  setStageLimitForQA(stage: Parameters<
    WebGPUSparseCM12Resident["setStageLimitForQA"]>[0]): void;
  setActivityPhaseLimitForQA(phase: Parameters<
    WebGPUSparseCM12Resident["setActivityPhaseLimitForQA"]>[0]): void;
  setTransportPhaseLimitForQA(phase: Parameters<
    WebGPUSparseCM12Resident["setTransportPhaseLimitForQA"]>[0]): void;
  setSharpeningPhaseLimitForQA(phase: Parameters<
    WebGPUSparseCM12Resident["setSharpeningPhaseLimitForQA"]>[0]): void;
  setPressureTopologyPhaseLimitForQA(phase: Parameters<
    WebGPUSparseCM12Resident["setPressureTopologyPhaseLimitForQA"]>[0]): void;
  readDiagnostics(): ReturnType<WebGPUSparseCM12Resident["readDiagnostics"]>;
  readDiagnosticFields(includeWorldLeaves?: boolean,
    frameBank?: "accepted" | "candidate"):
    ReturnType<WebGPUSparseCM12Resident["readDiagnosticFields"]>;
  readActivitySnapshot(includeWorldLeaves?: boolean):
    ReturnType<WebGPUSparseCM12Resident["readActivitySnapshot"]>;
  readPresentationPageAllocatorReceiptQA(): ReturnType<
    WebGPUSparseCM12Resident["readPresentationPageAllocatorReceiptQA"]>;
  readWorldGrowthReceiptQA(): ReturnType<
    WebGPUSparseCM12Resident["readWorldGrowthReceiptQA"]>;
  readPhase1TransportReceiptQA(allowStageLimitedCandidate?: boolean,
    probeCells?: readonly number[]): ReturnType<
    WebGPUSparseCM12Resident["readPhase1TransportReceiptQA"]>;
  readPhase1TransportProfileQA(): ReturnType<
    WebGPUSparseCM12Resident["readPhase1TransportProfileQA"]>;
  readCandidateEffectsTransactionQA(): ReturnType<
    WebGPUSparseCM12Resident["readCandidateEffectsTransactionQA"]>;
  readFramePlanPresentationHeaderQA(): ReturnType<
    WebGPUSparseCM12Resident["readFramePlanPresentationHeaderQA"]>;
  readFramePlanPresentationFaultRecordQA(): ReturnType<
    WebGPUSparseCM12Resident["readFramePlanPresentationFaultRecordQA"]>;
  readFrameControlQA(): ReturnType<WebGPUSparseCM12Resident["readFrameControlQA"]>;
  readTransportPacketIndirectQA(): ReturnType<
    WebGPUSparseCM12Resident["readTransportPacketIndirectQA"]>;
  readCoarseTransportScheduleQA(): ReturnType<
    WebGPUSparseCM12Resident["readCoarseTransportScheduleQA"]>;
  readDynamicTransportPacketsQA(): ReturnType<
    WebGPUSparseCM12Resident["readDynamicTransportPacketsQA"]>;
  readFinalScalarMaskHeaderQA(): ReturnType<
    WebGPUSparseCM12Resident["readFinalScalarMaskHeaderQA"]>;
  readWorkShapeQA(): ReturnType<WebGPUSparseCM12Resident["readWorkShapeQA"]>;
  readAdaptiveRepresentationQA(): ReturnType<
    WebGPUSparseCM12Resident["readAdaptiveRepresentationQA"]>;
  readAcceptedIndirectQA(): ReturnType<WebGPUSparseCM12Resident["readAcceptedIndirectQA"]>;
  readFrameControlIndirectQA(): ReturnType<
    WebGPUSparseCM12Resident["readFrameControlIndirectQA"]>;
  readPersistentPressureCacheIndirectQA(): ReturnType<
    WebGPUSparseCM12Resident["readPersistentPressureCacheIndirectQA"]>;
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
  private solidWorldStamp: string;
  private solidEnvironment: SceneDescription["environment"];
  private solidScenery: SceneDescription["scenery"];
  private rigidBodies: SceneDescription["rigidBodies"];

  constructor(
    private readonly resident: WebGPUSparseCM12Resident,
    private readonly options: AdoptCM12SparseWorldOptions,
    private readonly device: SparseWorldDevice,
  ) {
    this.generation = resident.globalFineLevelSetSource.generation;
    this.solidWorldStamp = solidWorldContentStamp(options.initialScene);
    this.solidEnvironment = options.initialScene.environment;
    this.solidScenery = options.initialScene.scenery;
    this.rigidBodies = options.initialScene.rigidBodies;
    options.trace?.record({
      kind: "world-created",
      residentTiles: options.residentTiles,
      capacityTiles: options.capacityTiles,
    });
  }

  edit(edit: SparseWorldEdit): SparseWorldEditReceipt {
    if (this.destroyed) {
      const error = new Error("Sparse world has been destroyed");
      this.publishFault("world-destroyed", error);
      throw error;
    }
    if (edit.kind === "set-scene") {
      if (edit.scene.rigidBodies.length > 0 && !this.options.rigidSystem) {
        return Object.freeze({
          disposition: "rebuild-required",
          acceptedGeneration: this.generation,
          reason: "this world was created without rigid-body coupling storage",
        });
      }
      try {
        // Region/scalar edits must stay a small uniform update. Static fluid
        // colliders depend on the SolidWorld stamp plus the environment graph;
        // only a change to that authority pays the voxel upload.
        const nextSolidWorldStamp = solidWorldContentStamp(edit.scene);
        const solidWorldChanged = nextSolidWorldStamp !== this.solidWorldStamp
          || edit.scene.environment !== this.solidEnvironment
          || edit.scene.scenery !== this.solidScenery;
        if (solidWorldChanged) {
          this.resident.setSolidWorld(fluidSolidWorldForScene(edit.scene));
          this.solidWorldStamp = nextSolidWorldStamp;
          this.solidEnvironment = edit.scene.environment;
          this.solidScenery = edit.scene.scenery;
        }
        this.resident.setRefinementRegionParameters(packSparseCM12RefinementRegions(
          sceneRefinementRegions(edit.scene), refinementRegionLattice(edit.scene)));
        if (edit.scene.rigidBodies !== this.rigidBodies) {
          this.options.rigidSystem?.setScene(edit.scene);
          this.rigidBodies = edit.scene.rigidBodies;
        }
        this.generation += 1;
        this.state = "running";
        return Object.freeze({
          disposition: "applied",
          acceptedGeneration: this.generation,
        });
      } catch (error) {
        this.publishFault("internal", error);
        throw error;
      }
    }
    try {
      const interaction = this.validInteraction(edit);
      const configuration = typeof this.options.numerics === "function"
        ? this.options.numerics({
          time: this.lastAcceptedTime,
          dt: 0.004,
          gravity: [0, 0, 0],
        }) : this.options.numerics;
      const inverseCell = 1 / configuration.finestCellSize_m;
      const origin = configuration.origin_m ?? [0, 0, 0];
      const encoder = this.options.gpuDevice.createCommandEncoder({
        label: "Sparse-world liquid interaction",
      });
      if (interaction.kind === "liquid-ellipsoid") {
        this.resident.encodeLiquidInjection(
          encoder,
          configuration.finestCellSize_m,
          interaction.center_m.map((value, axis) =>
            (value - origin[axis]!) * inverseCell) as [number, number, number],
          interaction.radii_m.map((value) => value * inverseCell) as
            [number, number, number],
        );
      } else {
        this.resident.encodeLiquidJetInjection(
          encoder,
          configuration.finestCellSize_m,
          interaction.outlet_m.map((value, axis) =>
            (value - origin[axis]!) * inverseCell) as [number, number, number],
          interaction.radius_m * inverseCell,
          interaction.velocity_m_s.map((value) => value * inverseCell) as
            [number, number, number],
          interaction.dt,
        );
      }
      this.options.gpuDevice.queue.submit([encoder.finish()]);
      this.generation += 1;
      this.state = "running";
      return Object.freeze({
        disposition: "applied",
        acceptedGeneration: this.generation,
      });
    } catch (error) {
      this.publishFault("invalid-interaction", error);
      throw error;
    }
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
      const acceleration = configuration.accelerationFinePerSecond2 ?? [
        input.gravity[0] / configuration.finestCellSize_m,
        input.gravity[1] / configuration.finestCellSize_m,
        input.gravity[2] / configuration.finestCellSize_m,
      ] as const;
      const rigidBodies = input.rigidBodies ?? [];
      const inverseCell = 1 / configuration.finestCellSize_m;
      const origin = configuration.origin_m ?? [0, 0, 0];
      const inflow: SparseCM12InflowControl | undefined = input.liquidInflow
        ? {
          outletFine: input.liquidInflow.outlet_m.map((value, axis) =>
            (value - origin[axis]!) * inverseCell) as [number, number, number],
          radiusFine: input.liquidInflow.radius_m * inverseCell,
          velocityFinePerSecond: input.liquidInflow.velocity_m_s.map(
            (value) => value * inverseCell,
          ) as [number, number, number],
        } : undefined;
      this.options.rigidSystem?.syncBodies(rigidBodies);
      if (this.options.rigidExchange) encoder.clearBuffer(this.options.rigidExchange);
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
        rigidBodies.length,
        configuration.worldDimensions_m,
        inflow,
      );
      this.options.rigidSystem?.encode(
        encoder,
        input.dt,
        configuration.finestCellSize_m ** 3,
        1,
        configuration.finestCellSize_m,
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

  private validInteraction(
    interaction: SparseWorldFluidEdit,
  ): SparseWorldFluidEdit {
    if (interaction.kind === "liquid-ellipsoid") {
      if (interaction.center_m.some((value) => !Number.isFinite(value))
        || interaction.radii_m.some((value) => !(value > 0) || !Number.isFinite(value))) {
        throw new RangeError(
          "Sparse-world liquid ellipsoids require a finite centre and positive radii",
        );
      }
      return Object.freeze({
        kind: interaction.kind,
        center_m: Object.freeze([...interaction.center_m]) as
          readonly [number, number, number],
        radii_m: Object.freeze([...interaction.radii_m]) as
          readonly [number, number, number],
      });
    }
    if (interaction.kind === "liquid-jet") {
      if (interaction.outlet_m.some((value) => !Number.isFinite(value))
        || interaction.velocity_m_s.some((value) => !Number.isFinite(value))
        || !(interaction.radius_m > 0) || !Number.isFinite(interaction.radius_m)
        || !(interaction.dt > 0) || !Number.isFinite(interaction.dt)) {
        throw new RangeError(
          "Sparse-world liquid jets require finite vectors, positive radius, and positive dt",
        );
      }
      return Object.freeze({
        kind: interaction.kind,
        outlet_m: Object.freeze([...interaction.outlet_m]) as
          readonly [number, number, number],
        radius_m: interaction.radius_m,
        velocity_m_s: Object.freeze([...interaction.velocity_m_s]) as
          readonly [number, number, number],
        dt: interaction.dt,
      });
    }
    throw new Error(`Unsupported sparse-world interaction: ${
      String((interaction as { kind?: unknown }).kind)}`);
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
  get solidWorldCollisionSource() { return this.resident.solidWorldCollisionSource; }
  setTracersEnabled(enabled: boolean) { this.resident.setTracersEnabled(enabled); }
  reseedTracers() { this.resident.reseedTracers(); }
  readTracers() { return this.resident.readTracers(); }
  armPressureJournal(armed: boolean) { return this.resident.armPressureJournal(armed); }
  readPressureJournal() { return this.resident.readPressureJournal(); }
  encodeInitialPresentation(encoder: GPUCommandEncoder, finestCellSize_m: number) {
    this.resident.encodeInitialPresentation(encoder, finestCellSize_m);
  }
  encodePressureIterationReceipt(encoder: GPUCommandEncoder, destination: GPUBuffer) {
    this.resident.encodePressureIterationReceipt(encoder, destination);
  }
}

class AdoptedCM12SparseWorldDeveloperTrace implements CM12SparseWorldDeveloperTrace {
  constructor(private readonly resident: WebGPUSparseCM12Resident) {}

  setStageLimitForQA(stage: Parameters<
    WebGPUSparseCM12Resident["setStageLimitForQA"]>[0]) {
    this.resident.setStageLimitForQA(stage);
  }
  setActivityPhaseLimitForQA(phase: Parameters<
    WebGPUSparseCM12Resident["setActivityPhaseLimitForQA"]>[0]) {
    this.resident.setActivityPhaseLimitForQA(phase);
  }
  setTransportPhaseLimitForQA(phase: Parameters<
    WebGPUSparseCM12Resident["setTransportPhaseLimitForQA"]>[0]) {
    this.resident.setTransportPhaseLimitForQA(phase);
  }
  setSharpeningPhaseLimitForQA(phase: Parameters<
    WebGPUSparseCM12Resident["setSharpeningPhaseLimitForQA"]>[0]) {
    this.resident.setSharpeningPhaseLimitForQA(phase);
  }
  setPressureTopologyPhaseLimitForQA(phase: Parameters<
    WebGPUSparseCM12Resident["setPressureTopologyPhaseLimitForQA"]>[0]) {
    this.resident.setPressureTopologyPhaseLimitForQA(phase);
  }
  readDiagnostics() { return this.resident.readDiagnostics(); }
  readDiagnosticFields(includeWorldLeaves = false,
    frameBank: "accepted" | "candidate" = "accepted") {
    return this.resident.readDiagnosticFields(includeWorldLeaves, frameBank);
  }
  readActivitySnapshot(includeWorldLeaves = false) {
    return this.resident.readActivitySnapshot(includeWorldLeaves);
  }
  readPresentationPageAllocatorReceiptQA() {
    return this.resident.readPresentationPageAllocatorReceiptQA();
  }
  readWorldGrowthReceiptQA() {
    return this.resident.readWorldGrowthReceiptQA();
  }
  readPhase1TransportReceiptQA(allowStageLimitedCandidate = false,
    probeCells: readonly number[] = []) {
    return this.resident.readPhase1TransportReceiptQA(
      allowStageLimitedCandidate, probeCells,
    );
  }
  readPhase1TransportProfileQA() { return this.resident.readPhase1TransportProfileQA(); }
  readCandidateEffectsTransactionQA() {
    return this.resident.readCandidateEffectsTransactionQA();
  }
  readFramePlanPresentationHeaderQA() {
    return this.resident.readFramePlanPresentationHeaderQA();
  }
  readFramePlanPresentationFaultRecordQA() {
    return this.resident.readFramePlanPresentationFaultRecordQA();
  }
  readFrameControlQA() { return this.resident.readFrameControlQA(); }
  readTransportPacketIndirectQA() { return this.resident.readTransportPacketIndirectQA(); }
  readCoarseTransportScheduleQA() { return this.resident.readCoarseTransportScheduleQA(); }
  readDynamicTransportPacketsQA() { return this.resident.readDynamicTransportPacketsQA(); }
  readFinalScalarMaskHeaderQA() { return this.resident.readFinalScalarMaskHeaderQA(); }
  readWorkShapeQA() { return this.resident.readWorkShapeQA(); }
  readAdaptiveRepresentationQA() { return this.resident.readAdaptiveRepresentationQA(); }
  readAcceptedIndirectQA() { return this.resident.readAcceptedIndirectQA(); }
  readFrameControlIndirectQA() { return this.resident.readFrameControlIndirectQA(); }
  readPersistentPressureCacheIndirectQA() {
    return this.resident.readPersistentPressureCacheIndirectQA();
  }
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

export async function createCM12SparseWorld(
  config: CM12SparseWorldFactoryConfig,
): Promise<CM12SparseWorldAdoption> {
  const sparseDevice = new CM12SparseWorldDevice(config.device, undefined, config.trace);
  const active = config.initiallyActiveBrickKeys
    ?? new Set(config.atlas.bricks.map((brick) => brick.key));
  const args = [
    config.device,
    config.atlas,
    config.grid,
    typeof config.numerics === "function"
      ? config.numerics({ time: 0, dt: 0.004, gravity: [0, 0, 0] })
        .finestCellSize_m
      : config.numerics.finestCellSize_m,
    config.solidWorld,
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
      : mode === "phase1-transport-receipt-qa"
        ? await WebGPUSparseCM12Resident.createPhase1TransportReceiptOracleForQA(...args)
        : mode === "velocity-extension-packet-compaction-qa"
          ? await WebGPUSparseCM12Resident
            .createVelocityExtensionPacketCompactionOracleForQA(...args)
          : mode === "coarse-transport-cell-packing-qa"
            ? await WebGPUSparseCM12Resident
              .createCoarseTransportCellPackingOracleForQA(...args)
            : mode === "refinement-policy-leader-compaction-qa"
              ? await WebGPUSparseCM12Resident
                .createRefinementPolicyLeaderCompactionOracleForQA(...args)
            : mode === "refinement-policy-full-leaf-qa"
              ? await WebGPUSparseCM12Resident
                .createRefinementPolicyFullLeafOracleForQA(...args)
            : mode === "phase1-coarse-transport-cell-packing-qa"
              ? await WebGPUSparseCM12Resident
                .createPhase1CoarseTransportCellPackingOracleForQA(...args)
            : mode === "density-capacity-early-exit-qa"
              ? await WebGPUSparseCM12Resident
                .createDensityCapacityEarlyExitOracleForQA(...args)
            : await WebGPUSparseCM12Resident.create(...args);
    resident.setRefinementRegionParameters(config.refinementRegionParameters);
  } catch (error) {
    sparseDevice.publishLibraryFault(error);
    throw error;
  }
  const readiness = sparseDevice.adoptResident(resident, config.trace, {
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
    gpuDevice: config.device,
    numerics: config.numerics,
    residentTiles: config.residentTiles ?? active.size,
    capacityTiles: config.capacityTiles
      ?? resident.globalFineLevelSetSource.plan.maximumResidentBricks,
    trace: config.trace,
    rigidSystem: config.rigidSystem,
    rigidExchange: config.rigid?.exchange,
    initialScene: config.scene,
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
