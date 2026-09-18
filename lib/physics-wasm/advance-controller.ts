import { PhysicsWasmClient, type PhysicsWasmClientOptions } from "./client";
import { detectPhysicsWasmCapabilities } from "./capabilities";
import { createAdvanceView, type AdvanceGraph, type AdvanceView } from "./advance-view";
import { decodePhysicsPublication } from "./publication";
import type { PhysicsPublication } from "./protocol";

const ADVANCE_VIEW_MASK = 0xf;

export interface AdvanceAuthoredScene {
  readonly id: string;
  readonly label: string;
  readonly document: unknown;
  readonly limitations?: readonly string[];
}
/**
 * One enforcement box, in finest cells of the run's lattice.
 *
 * The `id` is the caller's handle on a box it can still amend or delete; the
 * Rust `ResolutionRegion` does not declare it and does not deny unknown
 * fields, so it rides the wire and is ignored there.
 */
export interface AdvanceRefinementRegion {
  readonly id: string;
  readonly minimumFine: readonly [number, number];
  readonly maximumFine: readonly [number, number];
  readonly minimumCellWidth: number;
  readonly maximumCellWidth?: number;
}

export interface AdvanceLoadOptions {
  readonly pressureIterations: number;
  readonly pressureRelativeTolerance?: number;
  readonly tracerBudget?: number;
  readonly topologyPageBudget?: number;
  readonly transportExperiment?: AdvanceTransportExperimentOption;
  readonly adaptiveSdf?: boolean;
  readonly production?: Readonly<Record<string, unknown>>;
}

/** Explicit opt-in selector understood by the Rust 2-D world boundary. */
export type AdvanceTransportExperiment = "baseline" | "cellwise-remap" | "level-set-volume";
export type AdvanceTransportExperimentOption = AdvanceTransportExperiment | {
  readonly mode: "cellwise-remap";
  readonly traceSegments: number;
  readonly edgeSamples: 1 | 2 | 4;
};

/** Owns ordering and turns transferable Wasm snapshots into immutable UI views. */
export class AdvanceLabController {
  private graph?: AdvanceGraph;
  private authored?: AdvanceAuthoredScene;
  /* The enforcement boxes this run is carrying. They are a live command and
   * live nowhere else — a fresh `World` is built without them — so the run
   * that outlives a restart is the one this remembers. */
  private regions: readonly AdvanceRefinementRegion[] = [];

  private constructor(private readonly client: PhysicsWasmClient) {}

  static async create(options: PhysicsWasmClientOptions = {}): Promise<AdvanceLabController> {
    // Whole-frame M1 Max measurements favor single-worker SIMD for the 2D
    // pipeline. Keep threaded execution available to experiments explicitly;
    // the 3D client retains its capability-selected threaded default.
    const capabilities = detectPhysicsWasmCapabilities();
    return new AdvanceLabController(await PhysicsWasmClient.create({
      artifact: capabilities.simd ? "simd" : "scalar", ...options,
    }));
  }

  /** Seed a world from a document. The remembered regions are left alone: only
   * the caller knows whether this is the same run restarting or another one. */
  async load(scene: AdvanceAuthoredScene, options: AdvanceLoadOptions): Promise<AdvanceView> {
    await this.client.load(scene.document, options);
    this.authored = scene;
    this.graph = undefined;
    return this.view(await this.client.snapshot(ADVANCE_VIEW_MASK));
  }

  /**
   * Restart the loaded scene and hand the new world the regions the old one
   * was carrying, so the first view already obeys them.
   */
  async resetRun(options: AdvanceLoadOptions): Promise<AdvanceView> {
    const scene = this.authored;
    if (!scene) throw new Error("Advance controller has no loaded scene to reset");
    const seeded = await this.load(scene, options);
    return this.regions.length === 0 ? seeded : this.setRefinementRegions(this.regions);
  }

  async advance(dt_s: number): Promise<AdvanceView> {
    return this.view(await this.client.advance(dt_s, ADVANCE_VIEW_MASK));
  }

  async setTimeStep(dt_s: number): Promise<AdvanceView> {
    return this.command({ type: "set-time-step", dt_s });
  }

  async setPressureBudget(iterations: number, relativeTolerance = 1e-6): Promise<AdvanceView> {
    return this.command({ type: "set-pressure-budget", iterations, relativeTolerance });
  }

  async setTracers(enabled: boolean): Promise<AdvanceView> {
    return this.command({ type: "set-tracers", enabled });
  }

  async reseedTracers(): Promise<AdvanceView> {
    return this.command({ type: "reseed-tracers" });
  }

  async injectLiquid(centreFine: readonly [number, number], radiusFine: number): Promise<AdvanceView> {
    return this.command({ type: "inject-liquid", drop: { centreFine, radiusFine } });
  }

  async setRefinementRegions(regions: readonly AdvanceRefinementRegion[]): Promise<AdvanceView> {
    const view = await this.command({ type: "set-refinement-regions", regions });
    this.regions = [...regions];
    return view;
  }

  /** Forget the boxes without touching the world — for a run being replaced. */
  clearRefinementRegions(): void { this.regions = []; }

  destroy(): Promise<void> { return this.client.destroy(); }

  private async command(command: unknown): Promise<AdvanceView> {
    const publication = await this.client.applyCommand(command, ADVANCE_VIEW_MASK);
    if (!("bytes" in publication)) throw new TypeError("Advance command did not publish a view");
    return this.view(publication as PhysicsPublication);
  }

  private view(source: PhysicsPublication): AdvanceView {
    const decoded = decodePhysicsPublication(source);
    try {
      const view = createAdvanceView(decoded, this.graph, this.authored);
      this.graph = view.graph;
      return view;
    } finally {
      decoded.release();
    }
  }
}
