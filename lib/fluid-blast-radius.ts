/**
 * How far one pressure cell's value reaches, stage by stage through the solve.
 *
 * The SVO pixel trace can replay one ray because a ray is one sequential shader
 * invocation. A pressure cell has no such property: its work is spread across
 * every pass of the frame and repeated once per smoother sweep, per V-cycle
 * level, per outer iteration. What can be answered exactly, and without running
 * anything, is the dependency question — which other cells entered this cell's
 * value, and when.
 *
 * The operator is symmetric, so the inbound and outbound cones are the same
 * set; this module computes it once. Each stage of the encoded schedule is a
 * reach on the multigrid hierarchy:
 *
 * - a smoother sweep or operator application expands the cone by one cell at
 *   its own level, because `applied()` reads an 18-point stencil;
 * - restriction carries the cone from level l into level l + 1 through the
 *   parent map;
 * - prolongation carries it back down into every child.
 *
 * The consequence is the point of the tool. A coarse level is a shortcut: two
 * cells at opposite ends of the domain are a few hops apart through level 3
 * even though they are dozens of cells apart at level 0. The cone therefore
 * stops being local almost immediately, and the interesting quantity is not its
 * radius but how many times the whole domain is re-touched to move one value.
 *
 * The growth below is exact for a dense hierarchy, which is the upper bound a
 * sparse resident set can only fall inside. Nothing here reads GPU buffers; it
 * is the contract the GPU cone flood is checked against.
 */
import { linearFromHex } from "./visualization-decorations";
import {
  decorationVisualization,
  hasFields,
  type Visualization,
} from "./visualization-registry";

export type BlastRadiusStageKind =
  /** One application of the Poisson operator at level 0. */
  | "operator"
  /** One smoother sweep at its level. */
  | "smooth"
  /**
   * The direct solve on the coarsest level. It couples every cell of that level
   * at once, so it is its own kind rather than a smooth whose reach happens to
   * be large: the difference is what makes the cone go global in one step.
   */
  | "coarse-solve"
  /** Residual transfer from `level` to `level + 1`. */
  | "restrict"
  /** Correction transfer from `level + 1` to `level`. */
  | "prolong";

export interface BlastRadiusStage {
  readonly kind: BlastRadiusStageKind;
  /** Level the stage acts on; transfers name their finer side. */
  readonly level: number;
  /** Zero-based outer conjugate-gradient iteration containing this stage. */
  readonly iteration: number;
  readonly label: string;
}

export interface BlastRadiusSchedule {
  readonly stages: readonly BlastRadiusStage[];
  readonly levels: number;
  readonly outerIterations: number;
  readonly smoothsPerLevel: number;
}

export interface BlastRadiusScheduleOptions {
  /** Encoded outer iterations, from `planOctreeSolveTail`. */
  readonly outerIterations: number;
  /** Levels in the SPGrid pyramid, level 0 being the finest. */
  readonly levels: number;
  /** Pre- and post-smoothing sweeps applied at each level. */
  readonly smoothsPerLevel: number;
}

const positiveInteger = (value: number, label: string): number => {
  if (!Number.isSafeInteger(value) || value < 1) throw new RangeError(`${label} must be a positive integer`);
  return value;
};

/**
 * Flatten one encoded solve into the ordered stages that move information.
 *
 * Reductions, dot products and residual tests are deliberately absent: they
 * cost passes but do not widen anyone's dependency cone, and including them
 * would make the stage axis a poor proxy for reach. The pass census is the
 * right place to account for them.
 */
export function planBlastRadiusSchedule(options: BlastRadiusScheduleOptions): BlastRadiusSchedule {
  const outerIterations = positiveInteger(options.outerIterations, "Outer iterations");
  const levels = positiveInteger(options.levels, "Levels");
  const smoothsPerLevel = positiveInteger(options.smoothsPerLevel, "Smooths per level");
  const stages: BlastRadiusStage[] = [];
  for (let iteration = 0; iteration < outerIterations; iteration += 1) {
    stages.push({ kind: "operator", level: 0, iteration, label: "operator" });
    for (let level = 0; level < levels - 1; level += 1) {
      for (let sweep = 0; sweep < smoothsPerLevel; sweep += 1) {
        stages.push({ kind: "smooth", level, iteration, label: `pre-smooth L${level}` });
      }
      stages.push({ kind: "restrict", level, iteration, label: `restrict L${level}->L${level + 1}` });
    }
    // The coarsest level is solved rather than smoothed, which couples every
    // cell on it at once. Modelling that as one sweep would understate the
    // cone; `growBlastRadius` saturates the coarsest box instead.
    stages.push({ kind: "coarse-solve", level: levels - 1, iteration, label: `coarsest solve L${levels - 1}` });
    for (let level = levels - 2; level >= 0; level -= 1) {
      stages.push({ kind: "prolong", level, iteration, label: `prolong L${level + 1}->L${level}` });
      for (let sweep = 0; sweep < smoothsPerLevel; sweep += 1) {
        stages.push({ kind: "smooth", level, iteration, label: `post-smooth L${level}` });
      }
    }
  }
  return { stages: Object.freeze(stages), levels, outerIterations, smoothsPerLevel };
}

export type BlastRadiusVec3 = readonly [number, number, number];

export interface BlastRadiusLevelBox {
  readonly empty: boolean;
  readonly lo: BlastRadiusVec3;
  /** Inclusive upper corner. */
  readonly hi: BlastRadiusVec3;
}

export interface BlastRadiusFrontier {
  /** -1 before any stage has run; otherwise the index into `schedule.stages`. */
  readonly stageIndex: number;
  readonly stage?: BlastRadiusStage;
  readonly boxes: readonly BlastRadiusLevelBox[];
  /** Level-0 cells the cone covers. */
  readonly cells: number;
  /** `cells` over the level-0 cell count. */
  readonly share: number;
}

/**
 * Levels the SPGrid hierarchy needs to reach its exact one-cell bottom.
 *
 * `webgpu-octree-spgrid-vcycle.ts` reports `bottomOperation: "exact-single-cell"`
 * and refuses a `maximumLevels` that would truncate that bottom, so the depth is
 * a property of the domain rather than a tuning knob. It is also the reason the
 * cone goes global: prolonging a full one-cell level back down doubles the box
 * at every step, so one V-cycle reaches the whole grid from anywhere in it.
 */
export function blastRadiusLevelsToSingleCell(dimensions: BlastRadiusVec3): number {
  let levels = 1;
  while (blastRadiusLevelDimensions(dimensions, levels - 1).some((extent) => extent > 1)) levels += 1;
  return levels;
}

/** `dims(l)` on the SPGrid pyramid, matching the V-cycle's own rounding. */
export function blastRadiusLevelDimensions(
  dimensions: BlastRadiusVec3,
  level: number,
): BlastRadiusVec3 {
  const stride = 2 ** level;
  return [
    Math.max(1, Math.ceil(dimensions[0] / stride)),
    Math.max(1, Math.ceil(dimensions[1] / stride)),
    Math.max(1, Math.ceil(dimensions[2] / stride)),
  ];
}

const EMPTY_BOX: BlastRadiusLevelBox = Object.freeze({
  empty: true, lo: Object.freeze([0, 0, 0]) as BlastRadiusVec3, hi: Object.freeze([0, 0, 0]) as BlastRadiusVec3,
});

function boxCells(box: BlastRadiusLevelBox): number {
  if (box.empty) return 0;
  return (box.hi[0] - box.lo[0] + 1) * (box.hi[1] - box.lo[1] + 1) * (box.hi[2] - box.lo[2] + 1);
}

function unionBox(a: BlastRadiusLevelBox, b: BlastRadiusLevelBox): BlastRadiusLevelBox {
  if (a.empty) return b;
  if (b.empty) return a;
  return {
    empty: false,
    lo: [Math.min(a.lo[0], b.lo[0]), Math.min(a.lo[1], b.lo[1]), Math.min(a.lo[2], b.lo[2])],
    hi: [Math.max(a.hi[0], b.hi[0]), Math.max(a.hi[1], b.hi[1]), Math.max(a.hi[2], b.hi[2])],
  };
}

function dilate(box: BlastRadiusLevelBox, radius: number, dims: BlastRadiusVec3): BlastRadiusLevelBox {
  if (box.empty) return box;
  return {
    empty: false,
    lo: [
      Math.max(0, box.lo[0] - radius), Math.max(0, box.lo[1] - radius), Math.max(0, box.lo[2] - radius),
    ],
    hi: [
      Math.min(dims[0] - 1, box.hi[0] + radius),
      Math.min(dims[1] - 1, box.hi[1] + radius),
      Math.min(dims[2] - 1, box.hi[2] + radius),
    ],
  };
}

export interface BlastRadiusGrowthOptions {
  readonly dimensions: BlastRadiusVec3;
  readonly schedule: BlastRadiusSchedule;
  /** Level-0 cell the cone starts from. */
  readonly cell: BlastRadiusVec3;
  /**
   * Extra level-cell radius the restriction stencil reaches beyond the parent
   * map. Zero models a plain 2:1 parent gather; raise it to match a wider
   * transfer without changing anything else about the growth.
   */
  readonly restrictionRadius?: number;
  /** Extra radius the prolongation stencil reaches beyond the child map. */
  readonly prolongationRadius?: number;
}

/**
 * Grow the cone through the schedule, reporting it after every stage.
 *
 * The cone is a per-level axis-aligned box. That is exact for a dense
 * hierarchy: each stage's reach is isotropic, and a union of boxes under
 * isotropic dilation, halving and doubling stays a box. A sparse resident set
 * can only make the true cone a subset of this, so the coverage reported here
 * is an upper bound — which is the safe direction for a claim about how quickly
 * the solve stops being local.
 */
export function growBlastRadius(options: BlastRadiusGrowthOptions): readonly BlastRadiusFrontier[] {
  const { dimensions, schedule, cell } = options;
  const restrictionRadius = options.restrictionRadius ?? 0;
  const prolongationRadius = options.prolongationRadius ?? 0;
  for (let axis = 0; axis < 3; axis += 1) {
    if (!Number.isSafeInteger(dimensions[axis]) || dimensions[axis] < 1) {
      throw new RangeError("Blast-radius dimensions must be positive integers");
    }
    if (!Number.isSafeInteger(cell[axis]) || cell[axis] < 0 || cell[axis] >= dimensions[axis]) {
      throw new RangeError("Blast-radius cell must lie inside the domain");
    }
  }
  const levelDimensions = Array.from({ length: schedule.levels },
    (_, level) => blastRadiusLevelDimensions(dimensions, level));
  const totalCells = dimensions[0] * dimensions[1] * dimensions[2];
  let boxes: BlastRadiusLevelBox[] = levelDimensions.map(() => EMPTY_BOX);
  boxes[0] = { empty: false, lo: [...cell] as unknown as BlastRadiusVec3, hi: [...cell] as unknown as BlastRadiusVec3 };

  const snapshot = (stageIndex: number, stage?: BlastRadiusStage): BlastRadiusFrontier => {
    const cells = boxCells(boxes[0]);
    return {
      stageIndex,
      ...(stage ? { stage } : {}),
      boxes: boxes.map((box) => Object.freeze({ ...box, lo: Object.freeze([...box.lo]) as BlastRadiusVec3, hi: Object.freeze([...box.hi]) as BlastRadiusVec3 })),
      cells,
      share: totalCells > 0 ? cells / totalCells : 0,
    };
  };

  const frontiers: BlastRadiusFrontier[] = [snapshot(-1)];
  for (const [stageIndex, stage] of schedule.stages.entries()) {
    const next = [...boxes];
    if (stage.kind === "coarse-solve") {
      // A direct solve couples the whole level at once, but only if the cone
      // has actually reached it; an untouched coarsest grid must stay empty
      // rather than filling from nothing.
      const level = stage.level;
      if (!next[level].empty) {
        next[level] = { empty: false, lo: [0, 0, 0], hi: [
          levelDimensions[level][0] - 1, levelDimensions[level][1] - 1, levelDimensions[level][2] - 1,
        ] };
      }
    } else if (stage.kind === "operator" || stage.kind === "smooth") {
      next[stage.level] = dilate(next[stage.level], 1, levelDimensions[stage.level]);
    } else if (stage.kind === "restrict") {
      const source = next[stage.level];
      if (!source.empty) {
        const coarse: BlastRadiusLevelBox = {
          empty: false,
          lo: [source.lo[0] >> 1, source.lo[1] >> 1, source.lo[2] >> 1],
          hi: [source.hi[0] >> 1, source.hi[1] >> 1, source.hi[2] >> 1],
        };
        next[stage.level + 1] = unionBox(
          next[stage.level + 1],
          dilate(coarse, restrictionRadius, levelDimensions[stage.level + 1]),
        );
      }
    } else {
      const source = next[stage.level + 1];
      if (!source.empty) {
        const fine: BlastRadiusLevelBox = {
          empty: false,
          lo: [source.lo[0] * 2, source.lo[1] * 2, source.lo[2] * 2],
          hi: [source.hi[0] * 2 + 1, source.hi[1] * 2 + 1, source.hi[2] * 2 + 1],
        };
        next[stage.level] = unionBox(
          next[stage.level],
          dilate(fine, prolongationRadius, levelDimensions[stage.level]),
        );
      }
    }
    boxes = next;
    frontiers.push(snapshot(stageIndex, stage));
  }
  return Object.freeze(frontiers);
}

export interface BlastRadiusSummary {
  readonly totalCells: number;
  readonly stageCount: number;
  /** Stages until the cone covers every level-0 cell, or undefined if it never does. */
  readonly stagesToGlobal?: number;
  /** Outer iteration during which the cone went global, if it did. */
  readonly iterationToGlobal?: number;
  /** Level-0 coverage after the first outer iteration completes. */
  readonly shareAfterFirstIteration: number;
  /**
   * Times the whole level-0 grid is re-touched across the encoded schedule:
   * one per operator application and one per level-0 smoother sweep. This is
   * the cost the cone's speed buys, and the number worth reducing.
   */
  readonly fineGridSweeps: number;
}

export function summarizeBlastRadius(
  frontiers: readonly BlastRadiusFrontier[],
  schedule: BlastRadiusSchedule,
  dimensions: BlastRadiusVec3,
): BlastRadiusSummary {
  if (frontiers.length === 0) throw new RangeError("Blast-radius growth must have at least the seed frontier");
  const totalCells = dimensions[0] * dimensions[1] * dimensions[2];
  const global = frontiers.find((frontier) => frontier.share >= 1);
  const lastOfFirstIteration = frontiers.filter(
    (frontier) => frontier.stage !== undefined && frontier.stage.iteration === 0).at(-1);
  const fineGridSweeps = schedule.stages.filter(
    (stage) => (stage.kind === "operator" || stage.kind === "smooth" || stage.kind === "coarse-solve")
      && stage.level === 0).length;
  return {
    totalCells,
    stageCount: schedule.stages.length,
    ...(global && global.stage ? {
      stagesToGlobal: global.stageIndex + 1,
      iterationToGlobal: global.stage.iteration,
    } : {}),
    shareAfterFirstIteration: lastOfFirstIteration?.share ?? 0,
    fineGridSweeps,
  };
}

/* ------------------------------------------------------------------------- */
/* Visualization: the cone, drawn where the cell that seeds it was picked.    */
/* ------------------------------------------------------------------------- */

/**
 * What the cone decorator needs: a domain, a seed cell, and the solve policy to
 * grow against. Structural rather than a named trace type, so the model stays
 * independent of whatever selected the cell.
 */
export interface BlastRadiusSubject {
  readonly dimensions: BlastRadiusVec3;
  readonly cell: BlastRadiusVec3;
  readonly solvePolicy: BlastRadiusScheduleOptions;
}

/** Frontiers drawn. Enough to read the growth, few enough to see through. */
export const BLAST_RADIUS_CONE_SHELLS = 6;

const CONE_SWATCH = "#1a47eb";

function acceptsBlastRadiusSubject(subject: unknown): subject is BlastRadiusSubject {
  if (!hasFields(subject, ["dimensions", "cell", "solvePolicy"])) return false;
  const candidate = subject as BlastRadiusSubject;
  const { dimensions, cell, solvePolicy } = candidate;
  if (!Array.isArray(dimensions) || !Array.isArray(cell)) return false;
  return dimensions.every((extent) => Number.isSafeInteger(extent) && extent >= 1)
    && cell.every((coordinate, axis) => Number.isSafeInteger(coordinate)
      && coordinate >= 0 && coordinate < dimensions[axis])
    && Number.isSafeInteger(solvePolicy?.levels) && solvePolicy.levels >= 1
    && Number.isSafeInteger(solvePolicy?.outerIterations) && solvePolicy.outerIterations >= 0
    && Number.isSafeInteger(solvePolicy?.smoothsPerLevel) && solvePolicy.smoothsPerLevel >= 0;
}

/**
 * Frontiers worth drawing, thinned.
 *
 * Only stages where the cone actually grew are candidates — a smoothing sweep
 * repeating a reach it already had would stack a second box on the first — and
 * the last of them is always kept, because that is the frontier that shows the
 * cone covering the domain.
 */
function drawableConeFrontiers(subject: BlastRadiusSubject): readonly BlastRadiusFrontier[] {
  const schedule = planBlastRadiusSchedule(subject.solvePolicy);
  const frontiers = growBlastRadius({
    dimensions: subject.dimensions, schedule, cell: subject.cell,
  });
  const grown: BlastRadiusFrontier[] = [];
  let previous = frontiers[0]?.cells ?? 0;
  for (const frontier of frontiers.slice(1)) {
    if (frontier.cells <= previous || frontier.boxes[0].empty) continue;
    previous = frontier.cells;
    grown.push(frontier);
  }
  if (grown.length <= BLAST_RADIUS_CONE_SHELLS) return grown;
  const picked = new Map<number, BlastRadiusFrontier>();
  for (let index = 0; index < BLAST_RADIUS_CONE_SHELLS; index += 1) {
    const at = Math.round((index * (grown.length - 1)) / (BLAST_RADIUS_CONE_SHELLS - 1));
    picked.set(at, grown[at]);
  }
  return [...picked.keys()].sort((left, right) => left - right).map((key) => picked.get(key)!);
}

/**
 * The dependency cone as nested shells around the picked cell.
 *
 * Dashed throughout, because every box here is `scheduled`: it is what the
 * encoded command graph reaches, not what a residual-gated solve necessarily
 * ran. The outermost shell is the point of the picture — it is the one that says
 * there is no locality left to exploit after a single V-cycle.
 */
export const blastRadiusVisualizations: readonly Visualization[] = Object.freeze([
  decorationVisualization<BlastRadiusSubject>({
    kind: "decoration",
    id: "pressure-solve/cone",
    pass: "Pressure solve",
    group: "cone",
    label: "Dependency cone",
    swatch: CONE_SWATCH,
    description: "How far one cell's influence reaches after each stage of the encoded solve, out to the stage where it covers the domain.",
    source: "The encoded V-cycle schedule, grown analytically over the dense hierarchy",
    legend: [
      { mark: "dashed-box", swatch: CONE_SWATCH, label: "reach after a stage (scheduled)" },
      { mark: "dashed-box", swatch: "#0a1438", label: "outermost shell — cone is global" },
    ],
    accepts: acceptsBlastRadiusSubject,
    key: (subject) => [
      subject.dimensions.join("_"), subject.cell.join("_"),
      subject.solvePolicy.levels, subject.solvePolicy.outerIterations,
      subject.solvePolicy.smoothsPerLevel,
    ].join("."),
    build(subject, _context, into) {
      const shells = drawableConeFrontiers(subject);
      if (shells.length === 0) return [];
      const colorLinear = linearFromHex(CONE_SWATCH);
      const cell_m = into.cellSize_m;
      for (const [index, frontier] of shells.entries()) {
        const fade = shells.length > 1 ? index / (shells.length - 1) : 0;
        const box = frontier.boxes[0];
        into.box(
          [box.lo[0], box.lo[1], box.lo[2]],
          [box.hi[0] + 1, box.hi[1] + 1, box.hi[2] + 1],
          {
            colorLinear, width_px: 1.2,
            intensity: 0.85 - 0.45 * fade,
            // Longer dashes further out: the shells nest, and a constant dash
            // would moiré where two of them nearly coincide.
            dash_m: cell_m * (1.5 + 4 * fade),
          },
        );
      }
      const last = shells.at(-1)!;
      const schedule = planBlastRadiusSchedule(subject.solvePolicy);
      return [{
        label: "Reach the domain",
        value: last.share >= 1 ? `${last.stageIndex + 1} stages` : "local",
        evidence: "scheduled",
        detail: last.share >= 1
          ? `after ${last.stageIndex + 1} of ${schedule.stages.length} stages this cell depends on every other cell`
          : `the cone still covers only ${(last.share * 100).toFixed(1)}% after all ${schedule.stages.length} stages`,
      }];
    },
  }),
]);
