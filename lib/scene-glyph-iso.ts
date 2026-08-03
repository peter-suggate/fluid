import { editorFluidLattice, fluidBrickIndexAt, type EditorFluidLattice } from "./editor-fluid";
import { sceneDamBreakBox } from "./initial-fluid";
import type { SceneDescription } from "./model";
import { planSceneRuntime } from "./scene-runtime";
import { sceneHasTerrain, terrainHeightAt } from "./terrain";

/**
 * A scene's thumbnail as a small isometric room, derived from the document.
 *
 * The same argument as `sceneGlyph` — a mark that is a function of the document
 * is free, never stale, and correct for a scene nobody has opened — drawn from a
 * corner instead of side-on. The elevation could not tell a 3.2 m hall from a
 * 0.8 m cube once both were fitted to the same card, and a pond read as a hill
 * standing in a tank. A corner view carries depth, so proportion is visible
 * before anything in the room is.
 *
 * Everything here is in **normalized world units**: metres divided by the
 * container's longest axis, origin at the container's minimum corner. So the
 * drawing is a true view of the real room rather than three independently
 * stretched axes, and `size_m` lets several scenes be drawn to one scale — which
 * is what makes the three starters read as small, medium and large rather than
 * as three identically sized boxes.
 */

export interface IsoVec { readonly x: number; readonly y: number; readonly z: number }
export interface IsoBox { readonly min: IsoVec; readonly max: IsoVec }
export interface IsoBody extends IsoBox {
  /** Spheres, capsules and cylinders all read as one round mass at this size. */
  readonly round: boolean;
}

/** One ground profile across x at a fixed depth, back to front. */
export interface IsoTerrainRow {
  readonly z: number;
  readonly heights: readonly number[];
}

export interface SceneIsoGlyph {
  /** The room, normalized so its longest axis is exactly 1. */
  readonly extent: IsoVec;
  /** That longest axis in metres: the scene's true size, for a shared scale. */
  readonly size_m: number;
  readonly tank: {
    readonly top: "open" | "closed";
    /** Whether the room is a visible vessel or bare walls. */
    readonly glass: boolean;
  };
  /** Initial water as merged volumes; absent when the scene has no fluid. */
  readonly water?: readonly IsoBox[];
  readonly terrain?: { readonly rows: readonly IsoTerrainRow[] };
  readonly bodies: readonly IsoBody[];
  /** A hose: where it enters and which way it points, in the same units. */
  readonly inflow?: { readonly origin: IsoVec; readonly direction: IsoVec };
}

/**
 * Matches `SVO_RIGID_RASTER_CONTRACT.maximumBodies`, restated rather than
 * imported: that module carries the rasteriser's WGSL, and a front-door
 * thumbnail should not pull a shader source tree into its bundle. A scene cannot
 * show more bodies than this anyway, so the mark cannot promise more.
 */
const MAXIMUM_GLYPH_BODIES = 12;

/**
 * Odd, so the ground is sampled on the centre plane and on both walls. Nine
 * rows of seventeen is what resolves a pond's basin from the corner without
 * turning a thumbnail into a hundred-path drawing.
 */
const TERRAIN_ROWS = 9;
const TERRAIN_SAMPLES = 17;

function clamp(value: number, low: number, high: number): number {
  return Number.isFinite(value) ? Math.min(high, Math.max(low, value)) : low;
}

/** A box from normalized bounds, dropped when it trims away to nothing. */
function box(min: IsoVec, max: IsoVec, extent: IsoVec): IsoBox | undefined {
  const clamped = {
    min: { x: clamp(min.x, 0, extent.x), y: clamp(min.y, 0, extent.y), z: clamp(min.z, 0, extent.z) },
    max: { x: clamp(max.x, 0, extent.x), y: clamp(max.y, 0, extent.y), z: clamp(max.z, 0, extent.z) },
  };
  const { min: a, max: b } = clamped;
  return b.x > a.x && b.y > a.y && b.z > a.z ? clamped : undefined;
}

/**
 * Brick seeds as a few volumes rather than one box per brick.
 *
 * The ocean slab is authored as twenty seeds along depth. Twenty translucent
 * boxes stacked face to face is a grid of seams, not a body of water, so
 * contiguous bricks are merged — along depth first, then across width — and the
 * slab comes out as the one volume it is.
 */
function seedVolumes(scene: SceneDescription, lattice: EditorFluidLattice, extent: IsoVec, scale: number): IsoBox[] {
  const seeds = scene.fluid.initialBrickSeeds_m;
  if (!seeds?.length) return [];
  const occupied = new Set<string>();
  for (const seed of seeds) {
    const index = fluidBrickIndexAt(lattice, seed);
    if (index) occupied.add(`${index.x}:${index.y}:${index.z}`);
  }
  const cells = [...occupied]
    .map((key) => key.split(":").map(Number) as [number, number, number])
    .sort(([ax, ay, az], [bx, by, bz]) => ax - bx || ay - by || az - bz);

  const runs: { x: number; y: number; z0: number; z1: number }[] = [];
  for (const [x, y, z] of cells) {
    const open = runs.at(-1);
    if (open && open.x === x && open.y === y && open.z1 === z - 1) open.z1 = z;
    else runs.push({ x, y, z0: z, z1: z });
  }

  // Runs that span the same depth at the same height merge across width, which
  // is what collapses a wide reservoir seeded brick by brick into one slab.
  runs.sort((left, right) => left.y - right.y || left.z0 - right.z0 || left.z1 - right.z1 || left.x - right.x);
  const slabs: { x0: number; x1: number; y: number; z0: number; z1: number }[] = [];
  for (const run of runs) {
    const open = slabs.at(-1);
    if (open && open.y === run.y && open.z0 === run.z0 && open.z1 === run.z1 && open.x1 === run.x - 1) open.x1 = run.x;
    else slabs.push({ x0: run.x, x1: run.x, y: run.y, z0: run.z0, z1: run.z1 });
  }

  const brick = lattice.brickSize_m;
  return slabs.flatMap((slab) => {
    const volume = box(
      { x: slab.x0 * brick.x * scale, y: slab.y * brick.y * scale, z: slab.z0 * brick.z * scale },
      { x: (slab.x1 + 1) * brick.x * scale, y: (slab.y + 1) * brick.y * scale, z: (slab.z1 + 1) * brick.z * scale },
      extent,
    );
    return volume ? [volume] : [];
  });
}

function glyphWater(scene: SceneDescription, extent: IsoVec, scale: number): IsoBox[] {
  if (!planSceneRuntime(scene).fluidSolver) return [];
  const c = scene.container;
  const seeds = scene.fluid.initialBrickSeeds_m;
  const volumes: IsoBox[] = [];

  // The same resolution `combineInitialBrickWet` applies: the authored initial
  // condition is only drawn when the seeds do not own the water outright.
  if (!seeds?.length || scene.fluid.initialBrickSeedsAdditive) {
    const base = scene.fluid.initialCondition === "dam-break"
      ? (() => {
          const reservoir = sceneDamBreakBox(scene);
          return box(
            { x: reservoir.min.x * c.width_m * scale, y: reservoir.min.y * c.height_m * scale, z: reservoir.min.z * c.depth_m * scale },
            { x: reservoir.max.x * c.width_m * scale, y: reservoir.max.y * c.height_m * scale, z: reservoir.max.z * c.depth_m * scale },
            extent,
          );
        })()
      : box({ x: 0, y: 0, z: 0 }, { x: extent.x, y: c.fillFraction * c.height_m * scale, z: extent.z }, extent);
    if (base) volumes.push(base);
  }
  if (seeds?.length) volumes.push(...seedVolumes(scene, editorFluidLattice(scene), extent, scale));
  return volumes;
}

/**
 * The ground as depth-ordered profiles rather than one silhouette.
 *
 * A silhouette hides every hollow behind its own banks, which erases exactly the
 * basin that identifies a pond. Rows drawn back to front are how the corner view
 * shows a bowl as a bowl.
 */
function glyphTerrain(
  scene: SceneDescription,
  extent: IsoVec,
  scale: number,
): SceneIsoGlyph["terrain"] {
  if (!sceneHasTerrain(scene)) return undefined;
  const c = scene.container;
  const rows: IsoTerrainRow[] = Array.from({ length: TERRAIN_ROWS }, (_unused, row) => {
    const fz = row / (TERRAIN_ROWS - 1);
    const z_m = (fz - 0.5) * c.depth_m;
    return {
      z: fz * extent.z,
      heights: Array.from({ length: TERRAIN_SAMPLES }, (_unusedSample, sample) => {
        const fx = sample / (TERRAIN_SAMPLES - 1);
        return clamp(terrainHeightAt(scene.terrain, (fx - 0.5) * c.width_m, z_m) * scale, 0, extent.y);
      }),
    };
  });
  // Returning the field's own declared type, as `glyphInflow` does, rather than
  // the rows alone. A bare array assigned through the conditional spread below
  // is not checked against `SceneIsoGlyph` — object spreads widen — so this
  // shipped as a glyph whose `terrain.rows` was undefined and typechecked
  // clean. The signature is what makes that shape mismatch impossible.
  return { rows };
}

function glyphBodies(scene: SceneDescription, extent: IsoVec, scale: number): IsoBody[] {
  const c = scene.container;
  return scene.rigidBodies.slice(0, MAXIMUM_GLYPH_BODIES).flatMap((body) => {
    // Orientation is ignored on purpose, as in the elevation: at this size a
    // rotated capsule and an upright one are the same mark, and the axis-aligned
    // extent is what says where the body sits relative to the water.
    const centre = {
      x: (body.position_m.x + 0.5 * c.width_m) * scale,
      y: body.position_m.y * scale,
      z: (body.position_m.z + 0.5 * c.depth_m) * scale,
    };
    const half = {
      x: 0.5 * body.dimensions_m.x * scale,
      y: 0.5 * body.dimensions_m.y * scale,
      z: 0.5 * body.dimensions_m.z * scale,
    };
    const trimmed = box(
      { x: centre.x - half.x, y: centre.y - half.y, z: centre.z - half.z },
      { x: centre.x + half.x, y: centre.y + half.y, z: centre.z + half.z },
      extent,
    );
    return trimmed ? [{ ...trimmed, round: body.shape !== "box" }] : [];
  });
}

function glyphInflow(scene: SceneDescription, extent: IsoVec, scale: number): SceneIsoGlyph["inflow"] {
  const inflow = scene.fluid.inflow;
  if (!inflow || !planSceneRuntime(scene).fluidSolver) return undefined;
  const c = scene.container;
  const { x: vx, y: vy, z: vz } = inflow.velocity_m_s;
  const speed = Math.hypot(vx, vy, vz);
  return {
    origin: {
      x: clamp((inflow.center_m.x + 0.5 * c.width_m) * scale, 0, extent.x),
      y: clamp(inflow.center_m.y * scale, 0, extent.y),
      z: clamp((inflow.center_m.z + 0.5 * c.depth_m) * scale, 0, extent.z),
    },
    // A hose with no aim still falls, which is where its water ends up.
    direction: speed > 1e-9 ? { x: vx / speed, y: vy / speed, z: vz / speed } : { x: 0, y: -1, z: 0 },
  };
}

export function sceneIsoGlyph(scene: SceneDescription): SceneIsoGlyph {
  const c = scene.container;
  const size_m = Math.max(c.width_m, c.height_m, c.depth_m);
  const scale = size_m > 0 ? 1 / size_m : 1;
  const extent: IsoVec = { x: c.width_m * scale, y: c.height_m * scale, z: c.depth_m * scale };
  const water = glyphWater(scene, extent, scale);
  const terrain = glyphTerrain(scene, extent, scale);
  const inflow = glyphInflow(scene, extent, scale);
  return {
    extent,
    size_m,
    tank: {
      top: c.top,
      // Restated rather than imported from `lib/svo-scene-glass.ts`, which
      // carries the renderer's pane compositor: the garden's water sits in the
      // ground, so a pane around it would read as a bug on the card too.
      glass: c.vessel !== "none" && scene.environment !== "garden",
    },
    ...(water.length > 0 ? { water } : {}),
    ...(terrain ? { terrain } : {}),
    bodies: glyphBodies(scene, extent, scale),
    ...(inflow ? { inflow } : {}),
  };
}

/** Highest normalized waterline in the mark, as a fraction of room height. */
export function sceneIsoGlyphWaterline(glyph: SceneIsoGlyph): number | undefined {
  if (!glyph.water?.length) return undefined;
  return Math.max(...glyph.water.map((volume) => volume.max.y)) / glyph.extent.y;
}

/** Screen-reader summary of what the mark shows. */
export function sceneIsoGlyphLabel(glyph: SceneIsoGlyph): string {
  const waterline = sceneIsoGlyphWaterline(glyph);
  const parts = [
    `${glyph.tank.glass ? "Glass tank" : "Room"}, ${glyph.tank.top === "open" ? "open" : "closed"} on top`,
    waterline === undefined ? "no water" : `water at ${Math.round(waterline * 100)}%`,
  ];
  if (glyph.terrain) parts.push("terrain");
  if (glyph.inflow) parts.push("inflow");
  if (glyph.bodies.length > 0) parts.push(`${glyph.bodies.length} ${glyph.bodies.length === 1 ? "body" : "bodies"}`);
  return parts.join(", ");
}
