import { cloneScene, defaultScene, type SceneDescription } from "./model";
import { defaultEnvironmentId, type EnvironmentId } from "./environments";
import { solidVoxelShellForScene, latticeAxisDimension } from "./scene-lattice";
import type { SceneryGraph } from "./scenery-graph";

export interface EmptySceneOptions {
  readonly sceneId?: string;
  /** Interior extents in metres. Defaults to a comfortable 1.6 x 1.2 x 1.6 m room. */
  readonly extents_m?: { x: number; y: number; z: number };
  /** Finest lattice cell. Defaults so the domain is a modest, fast lattice. */
  readonly finestCellSize_m?: number;
  readonly environment?: EnvironmentId;
}

/**
 * 32 x 24 x 32 cells of 0.05 m: 24,576 finest cells, small enough that the SVO
 * envelope costs nothing to attach and every axis is a whole number of 8-cell
 * bricks, so no brick straddles a wall on the day the first content lands.
 */
const DEFAULT_FINEST_CELL_SIZE_M = 0.05;
const DEFAULT_EXTENTS_M = Object.freeze({ x: 1.6, y: 1.2, z: 1.6 });

/**
 * The room a new scene starts in: floor, walls, ceiling, and one softbox.
 *
 * A new scene has no subject yet, so anything beyond the enclosure and its
 * light would be scenery the author has to identify and delete before their own
 * work reads.
 *
 * Shared by every empty document, exactly as `studioSceneryGraph` is shared by
 * every scene that seeds from the studio: a document owns its graph, and an
 * edit replaces nodes rather than mutating them.
 */
export const minimalRoomSceneryGraph: SceneryGraph = {
  palettes: { white: { tint: [1, 1, 1] }, daylight: { tint: [1, 1, 1] } },
  nodes: [
    {
      kind: "room-shell", id: "shell", materialModel: "room",
      floor: { palette: "white", value: .9 },
      wall: { colorLinear: [1, 1, 1] },
      ceiling: { palette: "white", value: .9 },
    },
    {
      kind: "box", id: "light/softbox", group: "softbox", tags: ["softbox", "light", "emits-negative-y"],
      place: { position: { x: 0, y: 1.72, z: 0 }, anchor: "floor" },
      halfSize: { x: .72, y: .03, z: .72 },
      material: { palette: "daylight", value: 1, emission: 1 },
    },
  ],
};

/** A bounded document with no terrain, no bodies, no water, and no fluid solver. */
export function createEmptyScene(options: EmptySceneOptions = {}): SceneDescription {
  const finestCellSize_m = options.finestCellSize_m ?? DEFAULT_FINEST_CELL_SIZE_M;
  const requested_m = options.extents_m ?? DEFAULT_EXTENTS_M;
  // Author the integer lattice first, through the same rounding
  // `sceneLatticeDimensions` will apply, then multiply the extents back out of
  // it. A requested extent that is one part in 10^16 short of a whole number of
  // cells would otherwise round down an axis and leave the document describing
  // a lattice nobody asked for.
  const dimensions_cells = {
    x: latticeAxisDimension(requested_m.x, finestCellSize_m),
    y: latticeAxisDimension(requested_m.y, finestCellSize_m),
    z: latticeAxisDimension(requested_m.z, finestCellSize_m),
  };
  const scene = cloneScene(defaultScene);
  scene.sceneId = options.sceneId ?? "empty-scene";
  // `planSceneRuntime` reads exactly this to hand the frame to
  // `WebGPULiveSvoScene`: no fluid pipelines, no t=0 fence, no transport gate.
  // An omitted flag means "legacy scene, keep the solver", so it must be false.
  scene.systems = { ...defaultScene.systems, fluid: false };
  scene.rigidBodies = [];
  scene.container = {
    ...scene.container,
    width_m: dimensions_cells.x * finestCellSize_m,
    height_m: dimensions_cells.y * finestCellSize_m,
    depth_m: dimensions_cells.z * finestCellSize_m,
    fillFraction: 0,
    // A fresh document is about to have things dropped into it from above, and
    // a lid would be the first thing the user had to find and remove.
    top: "open",
    fluidWallMode: "no-slip",
    // The domain is still the solver's boundary; it is simply not drawn as a
    // tank. Standing an aquarium in the middle of an empty room presumes the
    // scene is about water before its author has said so.
    vessel: "none",
  };
  scene.solidVoxels = [...solidVoxelShellForScene(scene)];
  // 8-cell bricks even though a fluid-free scene may use 4: painting the first
  // water is meant to flip `systems.fluid` on this same document, and
  // `validateScene` rejects a fluid scene on a 4-cell brick.
  scene.voxelDomain = { finestCellSize_m, brickSize_cells: 8 };
  // The fluid block stays because the schema requires it, but every trace of
  // `defaultScene`'s dam break has to go: these are the fields that would
  // otherwise seed water into a document that claims to be empty.
  scene.fluid.initialCondition = "tank-fill";
  delete scene.terrain;
  delete scene.fluid.initialDamBreakDimensions_m;
  delete scene.fluid.initialDamBreakOrigin_m;
  delete scene.fluid.initialBrickSeeds_m;
  delete scene.fluid.initialBrickSeedsAdditive;
  delete scene.fluid.inflow;
  // The document owns its scenery from here, exactly as `scenePresets` does —
  // but the graph is authored rather than seeded from the environment, because
  // no environment describes an unstaged room. The environment id still names
  // the ambient light the room is lit by.
  scene.environment = options.environment ?? defaultEnvironmentId;
  scene.scenery = minimalRoomSceneryGraph;
  return scene;
}

export interface SceneStarter {
  readonly id: string;
  readonly name: string;
  readonly blurb: string;
  readonly create: () => SceneDescription;
}

/**
 * The starter strip beside "New scene".
 *
 * These differ only in how big the world is, because that is the one decision a
 * starter can honestly make on the author's behalf: the lattice is the
 * structural tier of the solver key, so growing a domain later is the single
 * edit that cannot be applied warm. Everything else — terrain, bodies, water,
 * a glass tank — is placed afterwards and costs nothing to change.
 *
 * Extents are whole 8-cell bricks on every axis, as in `createEmptyScene`.
 */
export const SCENE_STARTERS: readonly SceneStarter[] = Object.freeze([
  {
    id: "empty-room",
    name: "Room",
    blurb: "A 1.6 x 1.2 x 1.6 m room, lit and empty. The default: big enough to drop something into and watch it land.",
    create: () => createEmptyScene({ sceneId: "empty-room" }),
  },
  {
    id: "small-room",
    name: "Small room",
    blurb: "0.8 m on every side. The fastest lattice in the app — for trying an idea out before committing a domain to it.",
    create: () => createEmptyScene({
      sceneId: "small-room",
      extents_m: { x: 0.8, y: 0.8, z: 0.8 },
    }),
  },
  {
    id: "hall",
    name: "Hall",
    blurb: "3.2 x 1.6 x 3.2 m. Room for terrain with a valley in it, or a long enough run for a wave to break.",
    create: () => createEmptyScene({
      sceneId: "hall",
      extents_m: { x: 3.2, y: 1.6, z: 3.2 },
    }),
  },
] as const);
