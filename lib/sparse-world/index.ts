import type {
  SparseAdaptiveGridConsumerSource,
  WebGPUFineLevelSetBrickSource,
} from "../core/levelset-consumer-abi";

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

export interface SparseWorldInteractionBase {
  readonly kind: string;
}

/** A semantic liquid source. The implementation converts metres to tile space. */
export interface SparseWorldLiquidEllipsoidInteraction
  extends SparseWorldInteractionBase {
  readonly kind: "liquid-ellipsoid";
  readonly center_m: readonly [number, number, number];
  readonly radii_m: readonly [number, number, number];
}

export type SparseWorldInteraction = SparseWorldLiquidEllipsoidInteraction;

export interface SparseWorldStepInput {
  readonly time: number;
  readonly dt: number;
  readonly gravity: readonly [number, number, number];
  readonly interactions: readonly SparseWorldInteraction[];
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
