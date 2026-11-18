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
 *   list arguments (words 0..408: the lists, FC_* and FA_*) or F's region (2048..4095).
 * - F's buffers are named in `buffers` and allocated zeroed; their names may
 *   not shadow an engine buffer.
 * - F owns header words 64..127 and the class bits 16..23.
 * - A bring-up stage set (`bringUp: true`) is for E's tests only. The method
 *   refuses every scene on it (brick-method.ts).
 */
import type { BrickStaticWalls } from "./brick-static.wgsl";
import type { BrickSources } from "./brick-source";
import { ARGS, ARGS_LIST_END, type BrickLayout } from "./brick-abi";
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
   * The stamp stage (A6, m5-bodies-design-2026-11-17.md §8.1), after E's solidStamp: P's bodyStamp and its clean-up
   * pass. The engine places it at the end of the part 1 head (after the residency, lists(1), want, the tiered transfer
   * and the certificate; before the churn lists clear and the release port) and once at boot after the boot's lists,
   * then E's setKRelease (A5). F_ENTER and F_BODY are valid there. Absent: no body stage.
   */
  readonly stamp?: readonly BrickPlacement[];
  /**
   * The first frame's projected half kick, U <- P(U + g dt/2), after `boot`. The old engine's host runs it before
   * its first frame (webgpu-uniform-reference.ts kickMixedFrame, UniformMixedFrame.kick): the stored velocity is
   * the drift's, half a step ahead of the positions. Gravity alone, no transport, viscosity, capillarity or volume
   * term. `force` writes US = U + g dt/2; the engine then runs the operator, the solver and `project`; `after`
   * (F's) runs last, e.g. to give the particles the kicked U. Absent: no kick (E's bring-up tests).
   * `seed` (F's, optional): after `after`, the engine runs a TRIG-only census (censusFine through censusDecide on the
   * kicked state: TRIG, no WANT, no candidates, no claims, no transfer), then `seed`, so the first seeding reads the
   * triggers of the state it seeds (the old host's activity follows its first census, after the kick).
   */
  readonly kick?: { readonly force: readonly BrickPlacement[]; readonly after: readonly BrickPlacement[]; readonly seed?: readonly BrickPlacement[] };
  /** Lane-owned buffers by name, in words. */
  readonly buffers: Readonly<Record<string, number>>;
  /**
   * Where the stage set keeps its tiles' heat, for E's active class (brick-abi.md §5.5, ACT): a buffer of `buffers`,
   * the word of slot 0's tile 0, the words a tile (a slot's 8 tiles in order, slot-major), and the heat words of a
   * tile (f32; a nonzero word makes the tile ACTIVE). Absent: no heat criterion (E's bring-up stages hold none).
   */
  readonly activity?: { readonly buffer: string; readonly tile: number; readonly words: number; readonly heat: readonly number[] };
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
  /** The scene's particle settings (BrickSceneSetup.particleBandDepth); absent, the band is 3 cells deep. */
  readonly particles?: { readonly bandDepth: number };
  /** The scene's scheduled drops in source form (brick-source.ts); absent, no sources. */
  readonly sources?: BrickSources;
  /** The scene binds a static voxel mask (brick-static.wgsl.ts): F's set adds lane E's pSolid and phiSolid
   * (brick-solid-walk.wgsl.ts). */
  readonly statics?: boolean;
  /** With statics: the mask's walls and the terrain's columns in the same binding (brick-static.wgsl.ts
   * BrickStaticWalls). With terrain, pSolid and pSeed bind the static buffer (brick-terrain.wgsl.ts). */
  readonly staticWalls?: BrickStaticWalls;
}

export type BrickStagesFactory = (context: BrickStageContext) => BrickStages;

const stageLists = (s: BrickStages) => [...s.move, ...s.rebin, ...s.levelSet, ...s.velocity, ...s.tail, ...s.boot, ...(s.stamp ?? []), ...(s.kick?.force ?? []), ...(s.kick?.after ?? []), ...(s.kick?.seed ?? [])];

/** Checks a stage set's gated dispatches and buffer names. */
export function validateBrickStages(s: BrickStages): void {
  for (const p of stageLists(s)) {
    const range = p.kernel.cfgZ, z = p.z ?? 0;
    if (range && !(Number.isInteger(z) && z >= range[0] && z <= range[1]))
      throw new Error(`Brick engine: ${p.kernel.label} is placed with cfg.z ${z}, outside its range ${range[0]}..${range[1]}`);
    if ("indirect" in p.dispatch) {
      const w = p.dispatch.indirect;
      if (!(w >= 0 && w + 3 <= ARGS_LIST_END) && !(w >= FLUID_ARGS.FIRST && w + 3 <= FLUID_ARGS.END))
        throw new Error(`Brick engine: ${p.kernel.label} dispatches from args word ${w}, outside the lists and F's region`);
    }
  }
  if (s.activity) {
    const a = s.activity, n = s.buffers[a.buffer];
    if (n === undefined) throw new Error(`Brick engine: the activity buffer ${a.buffer} is not a stage buffer`);
    if (!(a.words >= 1) || a.heat.some(w => !(w >= 0 && w < a.words)) || !(a.tile >= 0 && a.tile + a.words * 8 <= n)) throw new Error(`Brick engine: the activity words of ${a.buffer} are outside its tiles`);
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
