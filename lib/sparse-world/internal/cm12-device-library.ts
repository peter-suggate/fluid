import type { WebGPUSparseCM12Resident } from
  "../../methods/adaptive-mass/webgpu-sparse-cm12-resident";
import type {
  SparseWorld,
  SparseWorldConfig,
  SparseWorldDevice,
  SparseWorldFault,
  SparseWorldProfile,
  SparseWorldTrace,
} from "../index";

type WorldFactory = (config: SparseWorldConfig) => Promise<SparseWorld>;

export interface CM12ResidentLibraryReadiness {
  readonly ready: Promise<void>;
}

/**
 * One readiness and failure authority for a CM12 implementation on one device.
 *
 * The resident still constructs scene-shaped WGSL during the migration. Keeping
 * that debt below this owner prevents a world, solver, or UI from becoming a
 * second compilation authority. The resident's device cache ensures that an
 * identical second world adopts already-compiled programs.
 */
class CM12DeviceLibraryState {
  private currentStatus: SparseWorldDevice["status"] = "loading";
  private currentFault: SparseWorldFault | undefined;
  private readonly traces = new Set<SparseWorldTrace>();
  private readonly readiness = new WeakMap<WebGPUSparseCM12Resident,
  CM12ResidentLibraryReadiness>();
  private pendingResidents = 0;

  get status(): SparseWorldDevice["status"] { return this.currentStatus; }
  get fault(): SparseWorldFault | undefined { return this.currentFault; }

  observe(trace: SparseWorldTrace | undefined): void {
    if (!trace || this.traces.has(trace)) return;
    this.traces.add(trace);
    trace.record({ kind: "device-readiness", state: this.currentStatus,
      ...(this.currentFault ? { detail: this.currentFault.message } : {}) });
  }

  adopt(
    resident: WebGPUSparseCM12Resident,
    trace: SparseWorldTrace | undefined,
    variant: unknown,
  ): CM12ResidentLibraryReadiness {
    this.observe(trace);
    const existing = this.readiness.get(resident);
    if (existing) return existing;

    trace?.record({
      kind: "implementation",
      subsystem: "cm12-device-library",
      detail: {
        variant,
      },
    });
    this.pendingResidents += 1;
    this.publishStatus("loading");
    resident.warmSimulationPipelines();
    const ready = resident.waitForSimulationPipelines().then(() => {
      this.pendingResidents -= 1;
      if (this.currentStatus !== "fault" && this.pendingResidents === 0) {
        this.publishStatus("ready");
      }
    }).catch((error) => {
      this.pendingResidents -= 1;
      this.publishFault(error);
      throw error;
    });
    const result = Object.freeze({ ready });
    this.readiness.set(resident, result);
    // Compilation is intentionally background work after t=0 publication.
    // Observe the rejection here so a caller need not await merely to avoid an
    // unhandled rejection; deterministic callers can still await `ready`.
    void ready.catch(() => undefined);
    return result;
  }

  publishFault(error: unknown): void {
    if (this.currentFault) return;
    this.currentFault = Object.freeze({
      code: "device-library",
      message: error instanceof Error ? error.message : String(error),
      cause: error,
    });
    this.currentStatus = "fault";
    this.record({ kind: "device-readiness", state: "fault",
      detail: this.currentFault.message });
    this.record({ kind: "fault", fault: this.currentFault });
  }

  private publishStatus(status: SparseWorldDevice["status"]): void {
    if (this.currentStatus === "fault" || this.currentStatus === status) return;
    this.currentStatus = status;
    this.record({ kind: "device-readiness", state: status });
  }

  private record(event: Parameters<SparseWorldTrace["record"]>[0]): void {
    for (const trace of this.traces) trace.record(event);
  }
}

const libraries = new WeakMap<GPUDevice,
Map<SparseWorldProfile, CM12DeviceLibraryState>>();

function acquireState(device: GPUDevice, profile: SparseWorldProfile):
CM12DeviceLibraryState {
  let profiles = libraries.get(device);
  if (!profiles) {
    profiles = new Map();
    libraries.set(device, profiles);
  }
  let library = profiles.get(profile);
  if (!library) {
    library = new CM12DeviceLibraryState();
    profiles.set(profile, library);
  }
  return library;
}

/** Public CM12 device facade; all facades share one device/profile library. */
export class CM12SparseWorldDevice implements SparseWorldDevice {
  private readonly library: CM12DeviceLibraryState;

  constructor(
    device: GPUDevice,
    private readonly create: WorldFactory | undefined,
    trace?: SparseWorldTrace,
    readonly profile: SparseWorldProfile = "cm12-b8",
  ) {
    this.library = acquireState(device, profile);
    this.library.observe(trace);
  }

  get status(): SparseWorldDevice["status"] { return this.library.status; }
  get fault(): SparseWorldFault | undefined { return this.library.fault; }

  async createWorld(config: SparseWorldConfig): Promise<SparseWorld> {
    if (config.profile !== this.profile) {
      throw new RangeError(`Sparse world device ${this.profile} cannot create ${
        config.profile}`);
    }
    if (!this.create) {
      throw new Error(
        "CM12 semantic seed translation is not installed on this device facade",
      );
    }
    this.library.observe(config.trace);
    return this.create(config);
  }

  /** A world adopts device-owned readiness here. */
  adoptResident(
    resident: WebGPUSparseCM12Resident,
    trace: SparseWorldTrace | undefined,
    variant: unknown,
  ): CM12ResidentLibraryReadiness {
    return this.library.adopt(resident, trace, variant);
  }

  /** Presentation-pipeline construction faults also belong to the library. */
  publishLibraryFault(error: unknown): void { this.library.publishFault(error); }
}
