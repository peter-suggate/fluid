import { cloneScene, defaultScene, type CameraState, type InitialLiquidSphere, type RigidBodyDescription, type SceneDescription } from "./model";
import type { MethodProfile } from "./methods";
import type { TerrainDescription } from "./terrain";
import type { SceneryGraph, SceneryNode } from "./scenery-graph";

/**
 * The scenes of Chentanez & Müller, *Mass-Conserving Eulerian Liquid
 * Simulation* (SCA 2012) — `docs/papers/massConservingLiquids.pdf`.
 *
 * What is published and what is not, because the difference decides how much of
 * each scene below is a reconstruction:
 *
 *  - **Published exactly**, in Sec. 4 and Table 2: the grid of every figure,
 *    `dx = 0.05 m`, `dt = 1/30 s`, gravity `10 m/s^2`, `D = 2.1`, `S = 1`, and
 *    per-figure CFL numbers. Those are the constants at the top of this file
 *    and the `grid` of each entry, and nothing here rounds or reinterprets
 *    them: a domain is exactly `grid * 0.05 m` on each axis.
 *  - **Published as a number we can invert**: the CFL column. `CFL = v dt/dx`
 *    with dt and dx fixed makes the paper's own table a statement about the
 *    fastest speed in each scene — `v = CFL * 1.5 m/s`. That is where the jet
 *    speed of Figure 1 and the obstacle speed of Figure 11 come from; they are
 *    derived from the paper rather than invented, which is why
 *    `cm12CharacteristicSpeed_m_s` exists instead of a literal.
 *  - **Not published at all**: the size and position of every liquid ball, the
 *    reservoir of each dam, the pool depths, and the geometry of the solids in
 *    Figures 1 and 11. Those are read off the plates and stated here in whole
 *    cells. They are explicit harness geometry, and a scene comment says so
 *    wherever a number came from a picture rather than from the text.
 *
 * The scenes run on the `uniform` method because that method *is* this paper —
 * see `lib/methods/uniform.ts`. Every entry pins the published parameter tuple
 * rather than inheriting the product defaults, so a figure keeps meaning the
 * same thing when a default moves.
 */

/** Sec. 4: "we used a time step size of 1/30s, dx = 0.05m, gravity 10m/s2". */
export const CM12_CELL_SIZE_M = 0.05;
export const CM12_TIME_STEP_S = 1 / 30;
export const CM12_GRAVITY_M_S2 = 10;
/** Sec. 4: "and D = 2.1"; Algorithm 2's gradient-trace limit, in cells. */
export const CM12_TRACE_DISTANCE_CELLS = 2.1;
/** Sec. 3.6: "We use S = 1 in all of our examples". */
export const CM12_SOLID_EXCESS_STRENGTH = 1;

/**
 * The speed a published CFL number implies, in m/s.
 *
 * `CFL = v dt / dx`, and the paper fixes dt and dx for every example, so a CFL
 * of 25 is a statement that something in Figure 1 moves at 37.5 m/s. Two scenes
 * have a single obvious carrier for that speed — the jet in Figure 1 and the
 * obstacles in Figure 11 — and both take it from here.
 */
export function cm12CharacteristicSpeed_m_s(cfl: number): number {
  return cfl * CM12_CELL_SIZE_M / CM12_TIME_STEP_S;
}

/** A figure of the paper, as the paper reports it. */
export interface Cm12Figure {
  /** Catalog id, e.g. `cm12-figure-9`. */
  readonly id: string;
  /** Figure number in the paper. */
  readonly figure: number;
  readonly name: string;
  /** Published cell counts. `undefined` z means the paper's case is 2D. */
  readonly grid: readonly [number, number, number | undefined];
  /** Table 2's CFL column, where the figure appears in it. */
  readonly cfl?: number;
  /** Table 2's per-frame simulation time on a GTX 680, in ms. */
  readonly frameTime_ms?: number;
  /** Sec. 3.8 post-processing, which is off "unless otherwise stated". */
  readonly densityPostProcessing?: boolean;
  /** What the paper says the scene is. */
  readonly blurb: string;
}

/**
 * The depth a published 2D case is simulated at, in cells.
 *
 * Figures 2 and 3 are 2D (128^2 for Figure 2; Figure 3 publishes no grid at
 * all). This solver is 3D, so they are run as a thin slab with free-slip walls
 * on the two depth faces, so that nothing in the third dimension does work. It
 * is the one place a published grid is *extended* rather than reproduced, and
 * the scene blurbs say so.
 *
 * The floor is physical rather than numerical: the depth has to comfortably
 * exceed the D = 2.1 cell trace distance of Algorithm 2, or the mass-return
 * trace reaches through the slab and out the far wall. Eight cells is a little
 * under four times that, and costs a sixteenth of a 128-deep domain.
 *
 * It used to be numerical, and much larger. The pressure hierarchy coarsened
 * all three axes in lockstep, so a thin axis capped how far the wide ones could
 * coarsen and nothing below 64 would construct at all -- see
 * `planUniformCM11aHierarchy`, which now falls back to coarsening each axis on
 * its own schedule for exactly the lattices the lockstep rule refuses.
 */
export const CM12_SLAB_DEPTH_CELLS = 8;

export const CM12_FIGURES: readonly Cm12Figure[] = Object.freeze([
  {
    id: "cm12-figure-1", figure: 1, name: "Liquid jet in a rectangular tank",
    grid: [256, 128, 128], cfl: 25, frameTime_ms: 113.2,
    blurb: "A jet with a very fast flow rate generates fast moving splashes and sheets.",
  },
  {
    id: "cm12-figure-2", figure: 2, name: "Ball into an empty box (2D)",
    grid: [128, 128, undefined],
    blurb: "The conservative-advection comparison against Lentine et al. 2011.",
  },
  {
    id: "cm12-figure-3", figure: 3, name: "Balls thrown into a pool (2D)",
    grid: [128, 128, undefined],
    blurb: "The local-sharpening case: balls whose mass must not migrate to the pool mid-air.",
  },
  {
    id: "cm12-figure-5", figure: 5, name: "Ball drop into a pool",
    grid: [128, 128, 128], cfl: 8, frameTime_ms: 54.2,
    blurb: "The scene the D parameter sweep is measured on.",
  },
  {
    id: "cm12-figure-6", figure: 6, name: "Crown splash",
    grid: [128, 128, 128], densityPostProcessing: true,
    blurb: "The density field is post-processed to bring out sub-grid detail.",
  },
  {
    id: "cm12-figure-7", figure: 7, name: "Ball drop into an empty tank",
    grid: [128, 128, 128],
    blurb: "The particle-level-set comparison; the sheet thins below a grid cell.",
  },
  {
    id: "cm12-figure-8", figure: 8, name: "Dam break inside a sphere",
    grid: [128, 128, 128], cfl: 14, frameTime_ms: 53.4,
    blurb: "A dam break in a non-axis-aligned spherical container.",
  },
  {
    id: "cm12-figure-9", figure: 9, name: "Dam break and ball drop in a glass box",
    grid: [128, 128, 64], cfl: 24, frameTime_ms: 26.7,
    blurb: "The real-time demo: simulated and ray-traced at over 30 fps on two GPUs.",
  },
  {
    id: "cm12-figure-11", figure: 11, name: "Solids moving across a tank",
    grid: [256, 128, 128], cfl: 32, frameTime_ms: 118.6,
    blurb: "One-way coupling with solids crossing the tank fast enough to throw the liquid into the air.",
  },
  {
    id: "cm12-figure-12", figure: 12, name: "Ball drop inside a sphere",
    grid: [128, 128, 128], cfl: 20, frameTime_ms: 53.8,
    blurb: "A ball dropped into an empty spherical container.",
  },
]);

export function cm12Figure(id: string): Cm12Figure {
  const figure = CM12_FIGURES.find((candidate) => candidate.id === id);
  if (!figure) throw new Error(`Unknown CM12 figure ${id}`);
  return figure;
}

/** The cell counts a figure is actually simulated at, 2D cases included. */
export function cm12Grid(figure: Cm12Figure): readonly [number, number, number] {
  return [figure.grid[0], figure.grid[1], figure.grid[2] ?? CM12_SLAB_DEPTH_CELLS];
}

/**
 * The paper's parameter tuple, pinned.
 *
 * Every value here is quoted in Sec. 3 or Sec. 4 rather than chosen: D and S
 * from Sec. 4 and Sec. 3.6, the 1/30 s step from Sec. 4, and the three
 * ablatable stages left on because the paper's results are the un-ablated
 * method. `densityPostProcessing` is the only entry that varies, because Sec. 4
 * says it "was turned off unless otherwise stated" and Figure 6 states it.
 */
export function cm12MethodProfile(figure: Cm12Figure): MethodProfile {
  return Object.freeze({
    methodId: "uniform",
    quality: "balanced",
    overrides: Object.freeze({
      timeStep: "paper",
      sharpeningDistance: CM12_TRACE_DISTANCE_CELLS,
      sharpeningStrength: CM12_SOLID_EXCESS_STRENGTH,
      densitySharpening: "on",
      sharpeningMassCorrection: "on",
      gammaDiffusion: "on",
      solidExcessCorrection: "on",
      densityPostProcessing: figure.densityPostProcessing ? "on" : "off",
    }),
  });
}

/**
 * A camera that frames a domain of this size.
 *
 * The catalog's other presets are authored for tanks about a metre across; a
 * CM12 domain is 6.4 m or 12.8 m, so a shared literal would put every one of
 * these scenes inside its own water. Framing from the container's diagonal
 * keeps the ten plates at a consistent apparent size.
 */
export function cm12Camera(scene: SceneDescription): Partial<CameraState> {
  const c = scene.container;
  return {
    azimuth_rad: 0.62,
    elevation_rad: 0.30,
    distance_m: 1.05 * Math.hypot(c.width_m, c.height_m, c.depth_m),
    target_m: { x: 0, y: 0.42 * c.height_m, z: 0 },
  };
}

/**
 * The domain, lattice and physics every figure shares, with no liquid in it.
 *
 * `top: "closed"` is a reconstruction and worth stating: the paper does not
 * name its boundary conditions, but Figure 10 plots total mass holding to
 * within 0.05% of the truth over whole simulations, which an open ceiling
 * would not do for the scenes whose splashes reach it. Free-slip walls follow
 * the variational/separating treatment of Sec. 3.7 (BBB07 with CM11a).
 *
 * Surface tension is zeroed because the paper has no surface-tension term —
 * Sec. 3.5 notes that raising D is what "visually resembles the effect of
 * surface tension" in this method.
 */
function cm12Domain(figure: Cm12Figure): SceneDescription {
  const scene = cloneScene(defaultScene);
  const [nx, ny, nz] = cm12Grid(figure);
  scene.sceneId = figure.id;
  scene.randomSeed = 2012;
  scene.duration_s = 6;
  scene.container = {
    ...scene.container,
    width_m: nx * CM12_CELL_SIZE_M,
    height_m: ny * CM12_CELL_SIZE_M,
    depth_m: nz * CM12_CELL_SIZE_M,
    fillFraction: 0,
    top: "closed",
    fluidWallMode: "free-slip",
  };
  scene.voxelDomain = { ...scene.voxelDomain, finestCellSize_m: CM12_CELL_SIZE_M };
  scene.nominalResolution = { length_m: CM12_CELL_SIZE_M };
  scene.numerics = {
    ...scene.numerics,
    fixedDt_s: CM12_TIME_STEP_S,
    maxDt_s: CM12_TIME_STEP_S,
  };
  scene.fluid = {
    ...scene.fluid,
    gravity_m_s2: { x: 0, y: -CM12_GRAVITY_M_S2, z: 0 },
    surfaceTension_N_m: 0,
    initialCondition: "tank-fill",
  };
  delete scene.fluid.initialDamBreakDimensions_m;
  delete scene.fluid.initialDamBreakOrigin_m;
  delete scene.fluid.initialBrickSeeds_m;
  delete scene.fluid.initialBrickSeedsAdditive;
  delete scene.fluid.initialLiquidVolumes;
  delete scene.fluid.inflow;
  delete scene.terrain;
  scene.rigidBodies = [];
  return scene;
}

/**
 * A bare white enclosure sized to the figure, and one overhead source.
 *
 * The catalog's authored environments are sets built at about a metre — a lab
 * bench, a stool, a pot shelf — and a CM12 domain is 6.4 m or 12.8 m across. A
 * paper figure staged in one of them would be a six-metre tank standing on a
 * desk. So these scenes carry their own graph: the studio's plain room with its
 * extents left to the scene rather than authored at the studio's size, and a
 * key light placed off the container instead of off a fixed height.
 *
 * The spherical figures use the renderer's analytic glass vessel. Their room
 * remains deliberately plain so the curved silhouette, caustics and water
 * motion read as clearly as they do in the paper plates.
 */
export function cm12SceneryGraph(scene: SceneDescription): SceneryGraph {
  const c = scene.container;
  return {
    palettes: { white: { tint: [1, 1, 1] }, daylight: { tint: [1, 1, 1] } },
    nodes: [
      {
          kind: "room-shell", id: "shell", materialModel: "room",
          floor: { palette: "white", value: .9 },
          wall: { colorLinear: [1, 1, 1] },
          ceiling: { palette: "white", value: .9 },
        },
      softbox(c),
    ],
  };
}

/**
 * The overhead panel that lights every paper figure.
 *
 * The light ABI reads a box emitter's normal off its *thinnest* axis and
 * refuses a proxy whose authored `emits-` tag disagrees. A panel sized as a
 * fraction of each container axis independently does not guarantee that: in a
 * domain much shallower than it is wide, 2% of the height can exceed 28% of the
 * depth and the emitter silently resolves to -z against a -y tag. Thinness in y
 * is therefore derived from the face the panel lights, so the tag holds for any
 * domain shape rather than for the ones that happen to be roughly cubic.
 */
function softbox(c: SceneDescription["container"]): SceneryNode {
  const halfX = 0.28 * c.width_m;
  const halfZ = 0.28 * c.depth_m;
  return {
    kind: "box", id: "light/softbox", group: "softbox",
    tags: ["softbox", "light", "emits-negative-y"],
    place: { position: { x: 0, y: 1.15 * c.height_m, z: -0.1 * c.depth_m }, anchor: "floor" },
    halfSize: {
      x: halfX,
      y: Math.min(0.02 * c.height_m, 0.25 * Math.min(halfX, halfZ)),
      z: halfZ,
    },
    material: { palette: "daylight", value: 1, emission: 1 },
  };
}
/**
 * A ball of liquid, authored in whole cells so a plate reading stays legible.
 *
 * The radii are all anchored on one measurement. Figure 7's plate — the lone
 * ball falling in the dry box — puts the ball at about a third of the domain
 * width across, a 20-cell radius at this lattice, and every other figure's
 * ball is scaled from its previous reconstruction by that same factor. They
 * were uniformly too small before: read against the plates the balls carried
 * roughly a third of the liquid they should, so the sheets and crowns they
 * make were correspondingly thin.
 */
function ball(centreCells: readonly [number, number, number], radiusCells: number): InitialLiquidSphere {
  return {
    shape: "sphere",
    center_m: {
      x: centreCells[0] * CM12_CELL_SIZE_M,
      y: centreCells[1] * CM12_CELL_SIZE_M,
      z: centreCells[2] * CM12_CELL_SIZE_M,
    },
    radius_m: radiusCells * CM12_CELL_SIZE_M,
  };
}

/**
 * A block of liquid resting on the floor, in whole cells from the -x/-z corner.
 *
 * The reservoir of a dam and the free-standing pool slab of Figures 5 and 6 are
 * the same primitive at different offsets, so they share one helper. The fill
 * fraction is written from the block because `validateScene` requires the two
 * to agree exactly.
 */
function block(
  scene: SceneDescription,
  originCells: readonly [number, number, number],
  sizeCells: readonly [number, number, number],
): void {
  const c = scene.container;
  const size = {
    x: sizeCells[0] * CM12_CELL_SIZE_M,
    y: sizeCells[1] * CM12_CELL_SIZE_M,
    z: sizeCells[2] * CM12_CELL_SIZE_M,
  };
  scene.fluid.initialCondition = "dam-break";
  scene.fluid.initialDamBreakDimensions_m = size;
  if (originCells.some((value) => value !== 0)) {
    scene.fluid.initialDamBreakOrigin_m = {
      x: originCells[0] * CM12_CELL_SIZE_M,
      y: originCells[1] * CM12_CELL_SIZE_M,
      z: originCells[2] * CM12_CELL_SIZE_M,
    };
  }
  c.fillFraction = size.x * size.y * size.z / (c.width_m * c.height_m * c.depth_m);
}

/**
 * The spherical container of Figures 8 and 12, as ground.
 *
 * A sphere is not a shape the solver takes: solids reach it as rigid primitives
 * (all of which are solid *inside*) or as a heightfield, and a heightfield is
 * single-valued in y. The bowl below the equator is therefore exact — the
 * height of the ground at (x, z) is exactly where the sphere is — and above the
 * equator, where the real vessel curves back over the liquid, the column
 * continues straight up as a wall. So this is a spherical bowl inside an open
 * cylinder, not a closed sphere: liquid climbing past the equator meets a
 * vertical wall where the paper's would meet an overhang.
 *
 * The vessel is the largest sphere the domain holds, which is what the plates
 * show: the container fills the frame in both figures.
 */
export function cm12SphericalVesselTerrain(scene: SceneDescription): TerrainDescription {
  const c = scene.container;
  const radius_m = 0.5 * Math.min(c.width_m, c.height_m, c.depth_m);
  const centre = { x: 0, y: radius_m, z: 0 };
  // One sample per finest cell, plus the closing sample on each axis, so the
  // grid spans the container exactly rather than falling a cell short.
  const nx = Math.round(c.width_m / CM12_CELL_SIZE_M) + 1;
  const nz = Math.round(c.depth_m / CM12_CELL_SIZE_M) + 1;
  const heights_m = new Array<number>(nx * nz);
  for (let j = 0; j < nz; j += 1) {
    for (let i = 0; i < nx; i += 1) {
      const x = -0.5 * c.width_m + i * CM12_CELL_SIZE_M;
      const z = -0.5 * c.depth_m + j * CM12_CELL_SIZE_M;
      const planar = Math.hypot(x - centre.x, z - centre.z);
      heights_m[i + nx * j] = planar >= radius_m
        ? c.height_m
        : centre.y - Math.sqrt(radius_m * radius_m - planar * planar);
    }
  }
  return {
    baseHeight_m: 0,
    features: [],
    grid: {
      kind: "grid",
      origin_m: { x: -0.5 * c.width_m, z: -0.5 * c.depth_m },
      spacing_m: CM12_CELL_SIZE_M,
      size: { nx, nz },
      heights_m,
    },
  };
}

/** A dense, fast solid crossing the tank; see `createCm12Figure11`. */
function obstacle(id: number, position: readonly [number, number, number], speed_m_s: number): RigidBodyDescription {
  return {
    id: `cm12-obstacle-${id}`,
    name: `Obstacle ${id}`,
    shape: "capsule",
    dimensions_m: { x: 0.5, y: 1.1, z: 0.5 },
    density_kg_m3: 5000,
    position_m: { x: position[0], y: position[1], z: position[2] },
    // Capsule axis defaults to +y; lie it along the direction of travel.
    orientation: { w: Math.SQRT1_2, x: 0, y: 0, z: Math.SQRT1_2 },
    linearVelocity_m_s: { x: speed_m_s, y: 0, z: 0 },
    angularVelocity_rad_s: { x: 0, y: 0, z: 0 },
    restitution: 0.05,
    friction: 0.4,
    motion: "dynamic",
  };
}

// ---- the figures -----------------------------------------------------------

/**
 * Figure 1 — a liquid jet with a large flow rate inside a rectangular tank.
 *
 * Published: the 256x128x128 grid and CFL 25, which fixes the jet at 37.5 m/s.
 * Reconstructed: the resting depth, and the nozzle's position and bore. The
 * plate also shows a dark tumbling solid in the jet that the caption never
 * mentions and gives no shape, mass or path for; inventing one would change the
 * solve, so it is left out.
 */
export function createCm12Figure1(): SceneDescription {
  const figure = cm12Figure("cm12-figure-1");
  const scene = cm12Domain(figure);
  const c = scene.container;
  scene.container.fillFraction = 0.3;
  scene.fluid.inflow = {
    center_m: { x: -0.5 * c.width_m + 0.4, y: 0.75 * c.height_m, z: 0 },
    radius_m: 0.4,
    length_m: 0.6,
    velocity_m_s: { x: cm12CharacteristicSpeed_m_s(figure.cfl!), y: 0, z: 0 },
    start_s: 0,
    end_s: scene.duration_s,
    ramp_s: 0,
  };
  return scene;
}

/**
 * Figure 2 — a 2D ball of liquid dropping into an empty box at 128^2.
 *
 * Published: the grid. Reconstructed: the ball, and `CM12_SLAB_DEPTH_CELLS` —
 * the depth this solver needs to run a 2D case at all, which is not small.
 */
export function createCm12Figure2(): SceneDescription {
  const scene = cm12Domain(cm12Figure("cm12-figure-2"));
  scene.fluid.initialLiquidVolumes = [ball([0, 90, 0], 14)];
  return scene;
}

/**
 * Figure 3 — liquid balls thrown into a pool.
 *
 * Published: nothing but the scene's description; the figure carries no grid,
 * so it runs at Figure 2's. Reconstructed: everything else. The paper throws
 * the balls, and a ball here has no initial velocity of its own, so they fall.
 * What the figure is *for* survives that: whether the sharpening step moves a
 * ball's mass into the pool before the ball arrives.
 */
export function createCm12Figure3(): SceneDescription {
  const scene = cm12Domain(cm12Figure("cm12-figure-3"));
  scene.container.fillFraction = 0.12;
  scene.fluid.initialLiquidVolumes = [
    ball([-40, 100, 0], 6),
    ball([-8, 112, 0], 7),
    ball([28, 96, 0], 6),
    ball([48, 108, 0], 4),
  ];
  return scene;
}

/**
 * Figure 5 — a ball dropping into a liquid pool, the D sweep's scene.
 *
 * Published: the 128^3 grid and CFL 8. Reconstructed: the ball and the pool.
 * The pool is a free-standing slab rather than a tank fill because that is what
 * the plate shows — a block of liquid on the floor with air on every side, so
 * the crown spreads without meeting a wall.
 */
export function createCm12Figure5(): SceneDescription {
  const scene = cm12Domain(cm12Figure("cm12-figure-5"));
  block(scene, [16, 0, 16], [96, 16, 96]);
  scene.fluid.initialLiquidVolumes = [ball([0, 88, 0], 17)];
  return scene;
}

/**
 * Figure 6 — a crown splash, with Sec. 3.8's density post-processing on.
 *
 * Published: the 128^3 grid, and that the density field is post-processed.
 * Reconstructed: the ball and the pool. A crown needs a pool shallow relative
 * to the ball, which is what the plate shows and what the depth below is.
 */
export function createCm12Figure6(): SceneDescription {
  const scene = cm12Domain(cm12Figure("cm12-figure-6"));
  block(scene, [16, 0, 16], [96, 8, 96]);
  scene.fluid.initialLiquidVolumes = [ball([0, 104, 0], 17)];
  return scene;
}

/**
 * Figure 7 — a liquid ball dropped inside an empty box.
 *
 * Published: the 128^3 grid. Reconstructed: the ball, whose 20-cell radius is
 * read off the plate and calibrates every other figure's (see `ball`). The
 * tank starts dry, so the ball is the entire liquid; it spreads over the floor
 * until the sheet is thinner than a cell, which is the case the figure exists
 * to show. At this radius the ball is about two cells deep spread over the
 * whole floor, so the thinning happens at the spreading front rather than
 * everywhere at once.
 */
export function createCm12Figure7(): SceneDescription {
  const scene = cm12Domain(cm12Figure("cm12-figure-7"));
  scene.fluid.initialLiquidVolumes = [ball([0, 90, 0], 20)];
  return scene;
}

/**
 * Figure 8 — a dam break inside a spherical container.
 *
 * Published: the 128^3 grid and CFL 14. Reconstructed: the vessel's radius
 * (the largest the domain holds, as the plate shows) and the dam. The dam is
 * the half-space x < 0 over the whole height, which the vessel then clips to
 * half the sphere — the lens of liquid the first frame shows.
 */
export function createCm12Figure8(): SceneDescription {
  const scene = cm12Domain(cm12Figure("cm12-figure-8"));
  scene.container.shape = "sphere";
  scene.container.vessel = "glass";
  scene.surfaceStyle = "smooth";
  scene.fluid.initialLiquidVolumes = [{
    shape: "hemisphere",
    center_m: { x: 0, y: 0.5 * scene.container.height_m, z: 0 },
    radius_m: 0.5 * Math.min(scene.container.width_m, scene.container.height_m, scene.container.depth_m),
    outwardNormal: { x: 1, y: 0, z: 0 },
  }];
  return scene;
}

/**
 * Figure 9 — a dam break and ball drop in a glass box, the real-time demo.
 *
 * Published: the 128x128x64 grid and CFL 24. Reconstructed: the reservoir and
 * the ball, both read off the first frame of the plate. The paper adds further
 * balls during the run; a scene states its initial condition only, so the one
 * ball already in the air at t = 0 is the one here.
 *
 * The catalog also carries `mass-conserving-figure-9-dam-break`, an older
 * reconstruction of this figure's dam phase alone. Its reservoir is shorter
 * (96 of 128 cells against 120 here) because it was written from the published
 * text; the column in the plate reaches all but a few cells of the lid. Both
 * are stated harness geometry, and the older one is left as it stands because
 * a conformance lane and its test are pinned to those constants.
 */
export function createCm12Figure9(): SceneDescription {
  const scene = cm12Domain(cm12Figure("cm12-figure-9"));
  block(scene, [0, 0, 0], [40, 120, 64]);
  scene.fluid.initialLiquidVolumes = [ball([8, 86, 0], 19)];
  return scene;
}

/**
 * Figure 11 — solids moving across a liquid tank at high speed.
 *
 * Published: the 256x128x128 grid and CFL 32, which fixes the obstacles at
 * 48 m/s. Reconstructed: the resting depth and the solids, which the paper
 * renders as a duck and gives no dimensions for.
 *
 * The paper couples one way — its solids are driven and the liquid never moves
 * them. This solver has no prescribed-motion body: a `static` one is frozen in
 * place (and, on the WebGPU path, would impose a moving-wall boundary while
 * never translating, which is worse than either alternative). So the obstacles
 * are dynamic and dense: at 5000 kg/m^3 and 48 m/s they cross the 12.8 m tank
 * in about a quarter of a second, during which gravity and the water they throw
 * up move them very little.
 */
export function createCm12Figure11(): SceneDescription {
  const figure = cm12Figure("cm12-figure-11");
  const scene = cm12Domain(figure);
  const c = scene.container;
  const speed_m_s = cm12CharacteristicSpeed_m_s(figure.cfl!);
  scene.container.fillFraction = 0.25;
  scene.rigidBodies = [
    obstacle(1, [-0.5 * c.width_m + 0.9, 1.9, 0], speed_m_s),
    obstacle(2, [-0.5 * c.width_m + 0.3, 2.4, -1.4], speed_m_s),
    obstacle(3, [-0.5 * c.width_m + 0.6, 1.6, 1.5], speed_m_s),
  ];
  return scene;
}

/**
 * Figure 12 — a liquid ball dropping inside an empty spherical container.
 *
 * Published: the 128^3 grid and CFL 20. Reconstructed: the vessel (shared with
 * Figure 8) and the ball. The vessel starts dry, so the ball is the whole
 * liquid, and it runs up the bowl rather than spreading on a floor.
 */
export function createCm12Figure12(): SceneDescription {
  const scene = cm12Domain(cm12Figure("cm12-figure-12"));
  scene.container.shape = "sphere";
  scene.container.vessel = "glass";
  scene.surfaceStyle = "smooth";
  scene.fluid.initialLiquidVolumes = [ball([0, 92, 0], 17)];
  return scene;
}

/**
 * The document a catalog entry publishes: the figure, then its enclosure.
 *
 * Separate from the factories because two of them assign `terrain` after the
 * domain is built, and the shell a scene needs is decided by whether it has
 * one. Running the graph last is what lets both facts be stated once.
 */
export function cm12Scene(id: string): SceneDescription {
  const scene = CM12_SCENE_FACTORIES[id]!();
  scene.scenery = cm12SceneryGraph(scene);
  return scene;
}

/** Every figure's factory, by catalog id. */
export const CM12_SCENE_FACTORIES: Readonly<Record<string, () => SceneDescription>> = Object.freeze({
  "cm12-figure-1": createCm12Figure1,
  "cm12-figure-2": createCm12Figure2,
  "cm12-figure-3": createCm12Figure3,
  "cm12-figure-5": createCm12Figure5,
  "cm12-figure-6": createCm12Figure6,
  "cm12-figure-7": createCm12Figure7,
  "cm12-figure-8": createCm12Figure8,
  "cm12-figure-9": createCm12Figure9,
  "cm12-figure-11": createCm12Figure11,
  "cm12-figure-12": createCm12Figure12,
});
