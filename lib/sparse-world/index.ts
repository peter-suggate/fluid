import type {
  SparseAdaptiveGridConsumerSource,
  WebGPUFineLevelSetBrickSource,
} from "../core/levelset-consumer-abi";
import type { SceneDescription } from "../core/model";
import type { RigidBodyState } from "../core/rigid-body";

export type {
  SparseWorldUI,
  SparseWorldUIControl,
  SparseWorldUIDiagnostics,
  SparseWorldUIDiagnosticSnapshot,
  SparseWorldUIOverlays,
} from "./ui";

export type SparseWorldProfile = "cm12-b8";

export interface SparseWorldCapacity {
  /** Complete physical simulation tiles, including currently free slots. */
  readonly tiles: number;
}

export interface SparseTileSeed {
  readonly coordinate: readonly [number, number, number];
  /** Cell payload is profile-owned. Callers provide authored values, not pages. */
  readonly density?: readonly number[];
  readonly gamma?: readonly number[];
}

export interface SparseWorldBoundaries {
  readonly x: "closed" | "open";
  readonly y: "closed" | "open";
  readonly z: "closed" | "open";
}

/** Logical bounds are metadata and must never size storage or execution. */
export interface SparseWorldExtent {
  readonly minimum: readonly [number, number, number];
  readonly maximum: readonly [number, number, number];
}

export interface SparseWorldConfig {
  readonly profile: SparseWorldProfile;
  readonly capacity: SparseWorldCapacity;
  readonly seeds: readonly SparseTileSeed[];
  readonly boundaries: SparseWorldBoundaries;
  readonly extent?: SparseWorldExtent;
  readonly trace?: SparseWorldTrace;
}

/** A semantic liquid source. The implementation converts metres to tile space. */
export interface SparseWorldLiquidEllipsoidEdit {
  readonly kind: "liquid-ellipsoid";
  /** Centre in the world's metre-space coordinate frame. */
  readonly center_m: readonly [number, number, number];
  readonly radii_m: readonly [number, number, number];
}

export interface SparseWorldLiquidJet {
  readonly outlet_m: readonly [number, number, number];
  readonly radius_m: number;
  readonly velocity_m_s: readonly [number, number, number];
}

export interface SparseWorldLiquidJetEdit extends
  SparseWorldLiquidJet {
  readonly kind: "liquid-jet";
  readonly dt: number;
}

export interface SparseWorldSceneEdit {
  readonly kind: "set-scene";
  readonly scene: SceneDescription;
}

/** Every application-authored change enters the world through this union. */
export type SparseWorldEdit =
  | SparseWorldLiquidEllipsoidEdit
  | SparseWorldLiquidJetEdit
  | SparseWorldSceneEdit;

export interface SparseWorldEditReceipt {
  readonly disposition: "applied" | "rebuild-required";
  readonly acceptedGeneration: number;
  readonly reason?: string;
}

export interface SparseWorldStepInput {
  readonly time: number;
  readonly dt: number;
  readonly gravity: readonly [number, number, number];
  /** Authoritative live roster; additions and removals are ordinary world input. */
  readonly rigidBodies?: readonly RigidBodyState[];
  /** Semantic hose boundary condition for this step, in world-space SI units. */
  readonly liquidInflow?: SparseWorldLiquidJet;
}

export interface SparseWorldStep {
  readonly generation: number;
  readonly submittedTime: number;
}

/**
 * Stable renderer contract for one accepted sparse generation.
 *
 * The two resource objects are owned by the world and remain identity-stable;
 * consumers must key visibility by `acceptedGeneration`.
 */
export interface SparseWorldPresentation {
  readonly acceptedGeneration: number;
  readonly fineLevelSet: WebGPUFineLevelSetBrickSource;
  readonly adaptiveGrid: SparseAdaptiveGridConsumerSource;
}

export type SparseWorldFaultCode =
  | "device-library"
  | "capacity-reached"
  | "invalid-interaction"
  | "step-encoding"
  | "world-destroyed"
  | "internal";

export interface SparseWorldFault {
  readonly code: SparseWorldFaultCode;
  readonly message: string;
  readonly cause?: unknown;
}

export interface SparseWorldStatus {
  readonly state: "ready" | "running" | "saturated" | "fault";
  readonly acceptedGeneration: number;
  readonly residentTiles: number;
  readonly capacityTiles: number;
  readonly lastAcceptedTime: number;
  readonly fault?: SparseWorldFault;
}

export interface SparseWorld {
  /** Apply one authored edit without exposing implementation encoders or buffers. */
  edit(edit: SparseWorldEdit): SparseWorldEditReceipt;
  encodeStep(
    encoder: GPUCommandEncoder,
    input: SparseWorldStepInput,
  ): SparseWorldStep;
  presentation(): SparseWorldPresentation;
  status(): SparseWorldStatus;
  destroy(): void;
}

export interface SparseWorldDevice {
  createWorld(config: SparseWorldConfig): Promise<SparseWorld>;
  readonly status: "loading" | "ready" | "fault";
  readonly fault?: SparseWorldFault;
}

export type SparseWorldTraceEvent =
  | {
    readonly kind: "device-readiness";
    readonly state: SparseWorldDevice["status"];
    readonly detail?: string;
  }
  | {
    readonly kind: "world-created";
    readonly residentTiles: number;
    readonly capacityTiles: number;
  }
  | {
    readonly kind: "step-encoded";
    readonly generation: number;
    readonly submittedTime: number;
  }
  | {
    readonly kind: "fault";
    readonly fault: SparseWorldFault;
  }
  | {
    /** Migration-only detail. Normal application state must not inspect it. */
    readonly kind: "implementation";
    readonly subsystem: string;
    readonly detail: unknown;
  };

/** Optional developer sink; it is observational and never schedules work. */
export interface SparseWorldTrace {
  record(event: SparseWorldTraceEvent): void;
}
