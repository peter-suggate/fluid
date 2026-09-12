import type { SliceNumericalFault } from "./slice-stage-numerics";

/**
 * Dropping liquid into a running two-dimensional advance.
 *
 * The CPU counterpart of the resident `injectLiquid` kernel — an
 * *intervention*, not a stage. Nothing in the advance calls it; a reader does,
 * from the lab, between one frame and the next.
 *
 * Production runs a drop as a two-phase transaction outside the ordinary
 * frame, and the reason is the whole shape of this file. A drop usually
 * reaches bricks that are not resident: air the topology retired, or a coarse
 * leaf the ladder never had a reason to refine. Writing density first and
 * letting the planner catch up loses every part of the ball that landed off
 * the accepted support. So the support is prepared first — one topology
 * generation whose activation demand includes the drop — and the dose is
 * applied once, afterwards, onto a graph that can hold all of it.
 *
 * This module owns the arithmetic of both phases and none of the orchestration:
 * `injectAdvanceSliceLiquid` in the solver holds the transaction, because the
 * transaction is `transitionAdvanceSliceTopology`, and a stage module that
 * reached back for it would close an import cycle around the authority it is
 * supposed to be a part of.
 *
 * Three departures from the resident path, all of them reductions:
 *
 * - **No deferral.** `injectLiquidBall` pushes onto `deferredFrameActions`
 *   while a GPU frame is in flight. `advanceSlice` is synchronous under one
 *   rAF callback, so a pointer handler can never land mid-advance and there is
 *   no frame to wait for. The absence is deliberate, not an omission.
 * - **No page allocation.** The slice's brick roster is the whole atlas plane,
 *   so every brick a drop can reach already exists as a key and activation is
 *   the only thing ever needed. The frontier allocator is untouched.
 * - **No open world.** A 2-D drop is inside the lattice or it is nothing.
 */

/**
 * The resident kernels this file mirrors.
 *
 * Named rather than hashed: `injectLiquid` shares a source file with the whole
 * resident encoder, so a content hash would fail on every unrelated edit and be
 * re-blessed until it meant nothing. The colocated test reads these three
 * functions out of the WGSL and asserts the specific terms this port copies.
 */
export const SLICE_LIQUID_INJECTION_SOURCE = Object.freeze({
  residentFunctions: Object.freeze(["injectionCoverageAt", "injectionReachesBrick",
    "injectLiquid"] as const),
});

/** A ball of liquid asked for, in the fine lattice's own frame. */
export interface SliceLiquidDrop {
  /** Centre in finest cells, canvas frame — the coordinates cells carry. */
  readonly centreFine: readonly [number, number];
  readonly radiusFine: number;
}

/** The brick shape a demand set needs; satisfied by `SliceTopologyBrick`. */
export interface SliceInjectionBrick {
  readonly key: number;
  readonly coordinate: readonly [number, number];
  readonly spanBricks?: number;
}

/** The cell shape a dose needs; satisfied by `SliceTopologyCell`. */
export interface SliceInjectionCell {
  readonly id: number;
  readonly centerFine: readonly [number, number];
  readonly widthsFine: readonly [number, number];
  readonly volumeFineCells: number;
}

/**
 * What one drop did, in the terms the lab reports faults in.
 *
 * `areaRequestedFine` is the analytic disk; `areaAdmittedFine` is what the
 * cells actually took, after aperture clipping and after the `max` against
 * water already there. The two differ for ordinary reasons — a drop half
 * inside a wall, a drop onto a full cell — so the pair is the reading, and
 * neither number alone says whether a drop went well.
 */
export interface SliceInjectionReceipt {
  readonly accepted: boolean;
  readonly cellsWetted: number;
  readonly areaAdmittedFine: number;
  readonly areaRequestedFine: number;
  readonly bricksDemanded: number;
  readonly bricksActivated: number;
  readonly bricksPromoted: number;
  readonly acceptedGeneration: number;
  readonly candidateGeneration: number;
  /** Why a refused drop was refused; null on every accepted one. */
  readonly fault: SliceNumericalFault | null;
}

export const EMPTY_SLICE_INJECTION_RECEIPT: SliceInjectionReceipt = Object.freeze({
  accepted: false, cellsWetted: 0, areaAdmittedFine: 0, areaRequestedFine: 0,
  bricksDemanded: 0, bricksActivated: 0, bricksPromoted: 0,
  acceptedGeneration: 0, candidateGeneration: 0, fault: null,
});

/**
 * The kernel's occupancy, which is not an exact area fraction.
 *
 * `injectionCoverageAt` clamps a signed distance scaled by the radius over the
 * cell width — a one-cell smoothed indicator of the disk, soft on both sides
 * of the rim. This port has exact clipped cut-cell geometry available and does
 * not use it, because production does not: an analytically better coverage
 * here is a parity regression, and the two solvers would stop agreeing on the
 * first frame of every drop.
 *
 * Z drops out of the resident expression exactly. A drop's ellipsoid is
 * isotropic, so `min(injectionRadius)` is the radius and `length(q)` loses one
 * term; nothing else in the expression mentions the third axis.
 */
export function injectionCoverage(drop: SliceLiquidDrop,
  centre: readonly [number, number], width: number): number {
  const radius = Math.max(drop.radiusFine, 1e-6);
  const qx = (centre[0] - drop.centreFine[0]) / radius;
  const qy = (centre[1] - drop.centreFine[1]) / radius;
  const signed = Math.hypot(qx, qy) - 1;
  return Math.min(1, Math.max(0, 0.5 - signed * radius / Math.max(width, 1e-6)));
}

/**
 * Every brick whose support the drop demands, as activation keys.
 *
 * The resident test is the ellipsoid's *bounding box* against the brick's, so
 * a brick sharing only an edge with the ball is woken too. That is not
 * sloppiness: activation has to happen a generation before the dose lands, and
 * a test admitting only bricks the disk truly covers would leave the ball's rim
 * on a leaf that is still air. The dose then writes no false liquid into the
 * extra bricks, because coverage there is zero.
 *
 * Exported because the lab draws this set: a reader sizing a ball can see
 * which leaves it is about to wake before releasing the pointer.
 */
export function sliceInjectionDemandedBrickKeys(bricks: readonly SliceInjectionBrick[],
  drop: SliceLiquidDrop, brickFineResolution = 8): ReadonlySet<number> {
  const demanded = new Set<number>();
  const [cx, cy] = drop.centreFine, r = drop.radiusFine;
  for (const brick of bricks) {
    const span = (brick.spanBricks ?? 1) * brickFineResolution;
    const x0 = brick.coordinate[0] * brickFineResolution;
    const y0 = brick.coordinate[1] * brickFineResolution;
    if (cx + r >= x0 && cx - r <= x0 + span && cy + r >= y0 && cy - r <= y0 + span) {
      demanded.add(brick.key);
    }
  }
  return demanded;
}

/**
 * The analytic disk, clipped to the lattice — what the reader asked for.
 *
 * Read beside the admitted area this is the honest pair: a drop that asked for
 * 40 finest-cells² and placed 12 has not failed, it has landed mostly in a
 * wall, and the receipt should let a reader tell those two readings apart
 * without reasoning about it. Sub-sampled rather than integrated in closed
 * form because it is a reported quantity and never a transported one — no
 * stage reads it, so its error cannot enter the solve.
 */
export function sliceInjectionRequestedArea(drop: SliceLiquidDrop,
  dimensions: readonly [number, number]): number {
  const r = drop.radiusFine, [cx, cy] = drop.centreFine;
  const x0 = Math.max(0, Math.floor(cx - r)), x1 = Math.min(dimensions[0], Math.ceil(cx + r));
  const y0 = Math.max(0, Math.floor(cy - r)), y1 = Math.min(dimensions[1], Math.ceil(cy + r));
  const samples = 4, weight = 1 / (samples * samples);
  let area = 0;
  for (let y = y0; y < y1; y += 1) for (let x = x0; x < x1; x += 1) {
    for (let sy = 0; sy < samples; sy += 1) for (let sx = 0; sx < samples; sx += 1) {
      const px = x + (sx + 0.5) / samples - cx, py = y + (sy + 0.5) / samples - cy;
      if (px * px + py * py <= r * r) area += weight;
    }
  }
  return area;
}

/**
 * A drop aimed with a pointer, in the frame the solver actually holds.
 *
 * The lab draws canvas-down and the topology is source-up — the single
 * reflection `production-scene-slice` performs once at its boundary and
 * `materialize` performs back again. A drop enters from the canvas side, so
 * the flip belongs here rather than being open-coded at the call site, where
 * it would be an off-by-one nobody notices until a ball lands mirrored about
 * the waterline.
 */
export function sliceDropFromCanvas(dimensions: readonly [number, number],
  canvasFine: readonly [number, number], radiusFine: number): SliceLiquidDrop {
  return { centreFine: [canvasFine[0], dimensions[1] - canvasFine[1]], radiusFine };
}

/**
 * Whether a drop is addressable at all, before anything touches the authority.
 *
 * A drop with no radius, a drop wholly off the lattice, or a non-finite centre
 * is a pointer accident rather than an experiment. Refusing it here keeps that
 * refusal in the receipt and out of the topology transaction, which should
 * only ever be asked questions that could have been answered yes.
 */
export function sliceDropIsAddressable(drop: SliceLiquidDrop,
  dimensions: readonly [number, number]): boolean {
  const [x, y] = drop.centreFine, r = drop.radiusFine;
  return Number.isFinite(x) && Number.isFinite(y) && Number.isFinite(r) && r > 0
    && x + r >= 0 && x - r <= dimensions[0] && y + r >= 0 && y - r <= dimensions[1];
}

/**
 * Apply a prepared dose to the accepted scalars.
 *
 * The resident write, term for term: coverage clipped by the cell's open
 * fraction, then `max` against what is already there. `max`, never `+` — the
 * additive branch belongs to the hose, and adding here would let a reader
 * click twice on one spot and drive a cell past its own capacity, which the
 * `0 <= V <= K` check would then report as a fault they had caused with the
 * mouse.
 *
 * Gamma goes to 1 on any covered cell, exactly as the kernel sets both of its
 * banks. Faces are deliberately left alone: `injectLiquidFaces` returns early
 * for anything that is not a hose, so a dropped ball arrives at rest and falls
 * under the same gravity as the water it lands in.
 */
export function applySliceInjectionDose(cells: readonly SliceInjectionCell[],
  density: Float32Array, gamma: Float32Array, capacity: ArrayLike<number>,
  drop: SliceLiquidDrop): { readonly cellsWetted: number; readonly areaAdmittedFine: number } {
  let cellsWetted = 0, areaAdmittedFine = 0;
  for (const cell of cells) {
    const open = capacity[cell.id] ?? 0;
    if (open <= 1e-8) continue;
    const coverage = injectionCoverage(drop, cell.centerFine,
      Math.min(cell.widthsFine[0], cell.widthsFine[1]));
    if (coverage <= 0) continue;
    gamma[cell.id] = 1;
    const previous = density[cell.id]!;
    const next = Math.max(previous, Math.fround(coverage * open));
    if (next === previous) continue;
    density[cell.id] = next;
    cellsWetted += 1;
    areaAdmittedFine += (next - previous) * cell.volumeFineCells;
  }
  return { cellsWetted, areaAdmittedFine };
}
