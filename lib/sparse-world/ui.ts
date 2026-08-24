import type { PressureJournal } from "../core/pressure-journal";
import type { StageLensSource } from "../core/stage-lens";
import type { GPUFluidFaceVelocitySource } from
  "../core/webgpu-face-velocity-overlay";
import type { GPUPressureJournalSource } from
  "../core/webgpu-pressure-journal-overlay";
import type { GPUFluidTracerSource } from "../core/webgpu-tracer-overlay";
import type { SparseWorldStatus } from "./index";

/** Sparse-specific controls that a normal UI may offer. */
export interface SparseWorldUIControl {
  readonly tracers?: Readonly<{
    setEnabled(enabled: boolean): void;
    reseed(): void;
  }>;
  readonly pressureFilm?: Readonly<{
    readonly captureEnabled: boolean;
    readonly snapshotCapacity: number;
    /** False means this world did not reserve film storage. */
    setCaptureEnabled(enabled: boolean): boolean;
  }>;
}

/** Read-only, method-neutral publications that renderer overlays may consume. */
export interface SparseWorldUIOverlays {
  readonly tracers?: GPUFluidTracerSource;
  readonly faceVelocity?: GPUFluidFaceVelocitySource;
  readonly pressureFilm?: GPUPressureJournalSource;
  readonly stageLenses?: StageLensSource;
}

export interface SparseWorldUIDiagnosticSnapshot {
  readonly world: SparseWorldStatus;
  readonly allocatedBytes: number;
  readonly storedCells: number;
  readonly constraintRows: number;
}

/** Bounded UI diagnostics. Internal transaction receipts live on developer trace. */
export interface SparseWorldUIDiagnostics {
  snapshot(): SparseWorldUIDiagnosticSnapshot;
  /** `[x, y, z, live]` per presentation tracer, in fine-lattice units. */
  readTracers(): Promise<Float32Array>;
  readPressureFilm(): Promise<PressureJournal | undefined>;
}

/**
 * The complete sparse-world surface available to core UI and renderer code.
 * It deliberately groups capabilities so adding one overlay does not grow the
 * generic solver interface by another CM12-shaped field.
 */
export interface SparseWorldUI {
  readonly control: SparseWorldUIControl;
  readonly overlays: SparseWorldUIOverlays;
  readonly diagnostics: SparseWorldUIDiagnostics;
}
