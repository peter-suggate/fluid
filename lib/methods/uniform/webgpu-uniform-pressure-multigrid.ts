import type { UniformTexturePages } from "./uniform-texture-pages";
export * from "./pressure-policy";

/** The hierarchy's levels from 4h: 4h, 8h, 16h and 32h, each a partition of a page. */
const UNIFORM_PRESSURE_PAGE_LEVELS = 4;

/**
 * What the mixed pressure solve is handed beside its own arena: the shared
 * scratch the root lies in, the residual tolerance, and whether the root's
 * cycles run from a list.
 */
export class WebGPUUniformPressureMultigrid {
  /** Levels of the whole hierarchy: the correction's and the two finer ones (h and 2h) it never visits. */
  readonly levelCount = 2 + UNIFORM_PRESSURE_PAGE_LEVELS;
  /** The published grid pressure: one texel no kernel writes. */
  readonly pressureTexture: GPUTexture;
  /** That texel's bytes: the levels' are the scratch arena's. */
  readonly allocatedBytes = 4;
  private activeResidualTolerance = 0;
  private symmetric = false;
  private isDestroyed = false;

  constructor(residualTolerance: number,
    /** List the root's jobs for the caller's cycles. The caller vouches
     * that the scene has no depth symmetry. */
    private readonly listable: boolean, private readonly scratchFields: UniformTexturePages) {
    if (!scratchFields.scratch) throw new Error("The CM11a hierarchy needs the shared scratch arena");
    this.pressureTexture = scratchFields.createTexture({ label: "Uniform CM11a L0 pressure A", size: [1, 1, 1], dimension: "3d",
      format: "r32float", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING
        | GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST });
    this.setResidualTolerance(residualTolerance);
  }

  setResidualTolerance(value: number): void { this.activeResidualTolerance = Number.isFinite(value) ? Math.max(0, value) : 0; }

  /** The pressure residual tolerance, which the solver hands the mixed pressure solve. */
  get residualTolerance(): number { return this.activeResidualTolerance; }

  /** The root's jobs are listed only without depth symmetry: a
   * continuation prepared after a live edit introduces it has no cycle list. */
  setDepthSymmetry(symmetric: boolean): void { this.symmetric = symmetric; }

  /** For a mixed solve: the arena the root lies in, and whether the root's
   * per-cycle operators run from the list of its liquid-or-bounded jobs. */
  prepareMixedContinuation() {
    if (this.isDestroyed) throw new Error("Uniform CM11a hierarchy is destroyed");
    const scratch = this.scratchFields.scratch!;
    return { arena: scratch.buffer, listed: this.listable && !this.symmetric };
  }

  destroy(): void { if (this.isDestroyed) return; this.isDestroyed = true; this.pressureTexture.destroy(); }
}
