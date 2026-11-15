/**
 * The fluid stage contract (brick-abi.md §6.2): where lane F's kernels sit in
 * the brick engine's recorded step, beside lane P's pressure slot (§6.1).
 *
 * A step is two submissions (§6):
 * - part 1: `head` (P's bodyStamp, E's solidStamp), F's `move` and `rebin`;
 * - part 2: E's brick set (setMark, setUpdate, setTransfer, setLists), F's
 *   `levelSet` and `velocity`, `operator`, P's solver slots, `project`, F's
 *   `tail`, then E's `stepEnd`.
 * `boot` runs once, after E's boot phi and lists: particle seeding and any
 * initial level-set work.
 *
 * Rules:
 * - Placements are BrickPlacements (brick-kernel.ts). Gated dispatches read
 *   list arguments (words 0..392) or F's region (2048..4095).
 * - F's buffers are named in `buffers` and allocated zeroed; their names may
 *   not shadow an engine buffer.
 * - F owns header words 64..127 and the class bits 16..23.
 * - A bring-up stage set (`bringUp: true`) is for E's tests only. The method
 *   refuses every scene on it (brick-method.ts).
 */
import { ARGS, type BrickLayout } from "./brick-abi";
import { ENGINE_BUFFERS, type BrickPlacement } from "./brick-kernel";
import { brickPressureKernels } from "./brick-pressure.wgsl";

/** F's args region (brick-abi.md §5.5). */
export const FLUID_ARGS = { FIRST: ARGS.LANE_F, END: ARGS.WORDS } as const;

export interface BrickStages {
  readonly label: string;
  /** True for E's bring-up stand-ins: tests only, never a gate or a timing claim. */
  readonly bringUp: boolean;
  /** True when the stages read through the tier owners (brick-owner.wgsl.ts) and write only owned storage: the
   * engine then runs the census and the transfer at the part 1 head (brick-abi.md §2.5). Default false: all-h. */
  readonly tiered?: boolean;
  /** Part 1, after the head: the particle move. */
  readonly move: readonly BrickPlacement[];
  /** Part 1: the rebin (counts, scan, scatter; raises BOUND_F from particle speeds). */
  readonly rebin: readonly BrickPlacement[];
  /** Part 2, after the brick set: advection, reconstruction and redistance of phi. */
  readonly levelSet: readonly BrickPlacement[];
  /** Part 2: p2g, momentum, forces and the extension before pressure. Writes US. */
  readonly velocity: readonly BrickPlacement[];
  /** Part 2, after `project`: the extension after pressure and the FLIP delta. */
  readonly tail: readonly BrickPlacement[];
  /** Once at boot, after E's boot phi and lists. */
  readonly boot: readonly BrickPlacement[];
  /**
   * The first frame's projected half kick, U <- P(U + g dt/2), after `boot`. The old engine's host runs it before
   * its first frame (webgpu-uniform-reference.ts kickMixedFrame, UniformMixedFrame.kick): the stored velocity is
   * the drift's, half a step ahead of the positions. Gravity alone, no transport, viscosity, capillarity or volume
   * term. `force` writes US = U + g dt/2; the engine then runs the operator, the solver and `project`; `after`
   * (F's) runs last, e.g. to give the particles the kicked U. Absent: no kick (E's bring-up tests).
   */
  readonly kick?: { readonly force: readonly BrickPlacement[]; readonly after: readonly BrickPlacement[] };
  /** Lane-owned buffers by name, in words. */
  readonly buffers: Readonly<Record<string, number>>;
}

/** What a stage factory gets: the layout and the scene's resolved stage parameters. */
export interface BrickStageContext {
  readonly layout: BrickLayout;
  /** dt (s), h (m) and gravity (m/s^2) as the param ring carries them. */
  readonly dt_s: number;
  readonly h_m: number;
  readonly gravity_mps2: readonly [number, number, number];
  /** The method's resolved values, for F's own parameters. */
  readonly values: Readonly<Record<string, unknown>>;
  /** The fluid's constants (scene.fluid); F's forces apply them or refuse the scene. Absent in E's tests. */
  readonly fluid?: { readonly density_kg_m3: number; readonly viscosity_Pa_s: number; readonly surfaceTension_N_m: number };
}

export type BrickStagesFactory = (context: BrickStageContext) => BrickStages;

const stageLists = (s: BrickStages) => [...s.move, ...s.rebin, ...s.levelSet, ...s.velocity, ...s.tail, ...s.boot, ...(s.kick?.force ?? []), ...(s.kick?.after ?? [])];

/** Checks a stage set's gated dispatches and buffer names. */
export function validateBrickStages(s: BrickStages): void {
  for (const p of stageLists(s)) {
    if ("indirect" in p.dispatch) {
      const w = p.dispatch.indirect;
      if (!(w >= 0 && w + 3 <= ARGS.FC_G64 + 3) && !(w >= FLUID_ARGS.FIRST && w + 3 <= FLUID_ARGS.END))
        throw new Error(`Brick engine: ${p.kernel.label} dispatches from args word ${w}, outside the lists and F's region`);
    }
  }
  for (const name of Object.keys(s.buffers))
    if ((ENGINE_BUFFERS as readonly string[]).includes(name) || name === "boot") throw new Error(`Brick engine: stage buffer ${name} shadows an engine buffer`);
}

/**
 * BRING-UP STAGES, FOR E'S TESTS ONLY. No particles and no level-set
 * transport. The velocity stage is uGravity, US = U + g dt on the open faces
 * with a liquid side, so a static box tests the brick set, the operator, the
 * pressure slot and the projection. Lane F's stages replace it.
 */
export function bringUpStages(context: BrickStageContext): BrickStages {
  const pres = brickPressureKernels(context.layout);
  return {
    label: "bring-up (uGravity)", bringUp: true,
    move: [], rebin: [], levelSet: [], tail: [], boot: [], buffers: {},
    velocity: [{ kernel: pres.uGravity, dispatch: { indirect: ARGS.PER_LIST * 0 + ARGS.G1 } }],
  };
}
