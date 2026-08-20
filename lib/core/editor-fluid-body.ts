import { damBreakFractions, initialFluidBrickComponents } from "./initial-fluid";
import { SCENE_SHAPES_BY_CODE } from "./scene-shape";
import type { EditorAction, EditorActionTarget } from "./editor-action";
import { cellProbeAction, rayProbeAction } from "./editor-probe-actions";
import {
  boxCenter,
  boxHandles,
  boxResizeDrag,
  boxSize,
  moveBoxWithinLimits,
  moveHandles,
  pickSolidBox,
  positionFields,
  resizeBox,
  sceneContainerBox,
  WORLD_FRAME,
  pickExcluded,
  type BoxExtent,
  type BoxResizePolicy,
  type BoxSides,
  type EditorEntity,
  type EditorEntityContext,
  type EditorEntityDefinition,
} from "./editor-entity";
import { editorFluidLattice, fluidBrickCenter } from "./editor-fluid";
import {
  fluidVolumeBodies,
  fluidVolumeEntity,
  fluidVolumeVolume_m3,
  pickFluidVolume,
  type FluidVolumeBody,
} from "./editor-fluid-volume";
import { sceneCellSizes_m, sceneLatticeDimensions } from "./scene-lattice";
import type { SceneDescription, Vec3 } from "./model";

/**
 * Direct manipulation of the initial water body.
 *
 * The body is an axis-aligned box in metres with the full complement of
 * handles — six faces, twelve edges, eight corners — so it can be grown,
 * shrunk, and reshaped from whichever side is facing the camera.
 *
 * It is authored as `fluid.initialDamBreakDimensions_m` plus the optional
 * `fluid.initialDamBreakOrigin_m`, which is a continuous box rather than the
 * brick-quantized `initialBrickSeeds_m` the paint tool writes: at a typical
 * 0.4 m brick a painted box could only resize in three steps across the whole
 * tank. A tank fill is the same box spanning the footprint, so grabbing a
 * handle on a filled tank reshapes the pool it already had instead of
 * replacing it — the conversion is exact, and the water never jumps.
 *
 * Every edit lands in the solver's seed tier, so a drag re-seeds the existing
 * solver rather than building a new one.
 */

/** A water-body box is an ordinary box extent; the name survives its callers. */
export type FluidBodyBox = BoxExtent;

export const FLUID_BODY_SELECTION_ID = "fluid-body";
/** Cells of thickness the body may never drop below on any axis. */
export const FLUID_BODY_MINIMUM_CELLS = 1;

const AXES: readonly (keyof Vec3)[] = Object.freeze(["x", "y", "z"]);

/** World-space bounds the body may occupy: the container interior. */
export function fluidBodyLimits(scene: SceneDescription): FluidBodyBox {
  return sceneContainerBox(scene);
}

/**
 * Whether the base initial condition is water the scene actually has.
 *
 * Seeded bricks *replace* the dam or the tank fill unless the document asks for
 * them to be added to it — that is `combineInitialBrickWet`, and it is the whole
 * of how a scene like `twin-dam-collision` describes two reservoirs. Nothing in
 * the editor used to ask, so a seed-authored scene drew a gizmo around a base
 * body that is never seeded: on the twin dams, a 1.10 x 0.74 x 0.32 m box over
 * two real 0.8 x 0.4 x 0.4 m slabs, sharing only a corner with one of them.
 */
export function fluidBaseBodyIsSeeded(scene: SceneDescription): boolean {
  const seeds = scene.fluid.initialBrickSeeds_m;
  return !seeds?.length || scene.fluid.initialBrickSeedsAdditive === true;
}

/**
 * The base initial condition as a world-space box, or undefined when the scene
 * has no shapeable base body — a fluid-less render scene, an empty tank, or a
 * document whose brick seeds have replaced it.
 *
 * This is one body among possibly several; `fluidBodies` is what enumerates
 * them. Seeded bricks are not folded into *this* box, because a box gizmo that
 * silently swallowed a painted blob into a rectangle would destroy work — they
 * get their own entities instead, each with the handles its shape can honour.
 */
export function fluidBodyBox(scene: SceneDescription): FluidBodyBox | undefined {
  if (scene.systems?.fluid === false) return undefined;
  if (!fluidBaseBodyIsSeeded(scene)) return undefined;
  const c = scene.container;
  const limits = fluidBodyLimits(scene);
  if (scene.fluid.initialCondition === "tank-fill") {
    const height_m = Math.max(0, Math.min(1, c.fillFraction)) * c.height_m;
    if (!(height_m > 0)) return undefined;
    return { min: limits.min, max: { x: limits.max.x, y: height_m, z: limits.max.z } };
  }
  const authored = scene.fluid.initialDamBreakDimensions_m;
  const size = authored ?? (() => {
    const fractions = damBreakFractions(c.fillFraction);
    return { x: fractions.width * c.width_m, y: fractions.height * c.height_m, z: fractions.depth * c.depth_m };
  })();
  if (!(size.x > 0) || !(size.y > 0) || !(size.z > 0)) return undefined;
  const origin = scene.fluid.initialDamBreakOrigin_m ?? { x: 0, y: 0, z: 0 };
  const min = { x: limits.min.x + origin.x, y: limits.min.y + origin.y, z: limits.min.z + origin.z };
  return { min, max: { x: min.x + size.x, y: min.y + size.y, z: min.z + size.z } };
}

/**
 * How the water body reshapes: onto the finest lattice, inside the container,
 * never thinner than a cell.
 *
 * The lattice is the resolution at which the seed actually changes — an
 * unsnapped box edge lands mid-cell, where it wets nothing new and the handle
 * appears to do nothing.
 */
export function fluidBodyResizePolicy(scene: SceneDescription): BoxResizePolicy {
  const cell = sceneCellSizes_m(scene);
  return {
    snap_m: [cell[0]!, cell[1]!, cell[2]!],
    limits: fluidBodyLimits(scene),
    minimum_m: [
      FLUID_BODY_MINIMUM_CELLS * cell[0]!,
      FLUID_BODY_MINIMUM_CELLS * cell[1]!,
      FLUID_BODY_MINIMUM_CELLS * cell[2]!,
    ],
  };
}

/** Move the sides a handle owns to the dragged point. */
export function dragFluidBodyBox(
  box: FluidBodyBox,
  sides: BoxSides,
  point: Vec3,
  scene: SceneDescription,
): FluidBodyBox {
  return resizeBox(box, sides, point, fluidBodyResizePolicy(scene));
}

/**
 * Scale the body about its own centre, held inside the container.
 *
 * `factor` is per-axis, not a volume: `scaleFluidBodyVolume` is what the
 * grow/shrink control calls, because a button beside a reading in litres has
 * to mean twice the water rather than eight times it.
 */
export function scaleFluidBodyBox(
  box: FluidBodyBox,
  factor: number,
  scene: SceneDescription,
): FluidBodyBox {
  const limits = fluidBodyLimits(scene);
  const cell = sceneCellSizes_m(scene);
  const min = { ...box.min }, max = { ...box.max };
  const snap = (value: number, step: number, origin: number) =>
    step > 0 ? origin + Math.round((value - origin) / step) * step : value;
  AXES.forEach((axis, index) => {
    const step = cell[index]!;
    const minimum = FLUID_BODY_MINIMUM_CELLS * step;
    const available = limits.max[axis] - limits.min[axis];
    const centre = 0.5 * (box.min[axis] + box.max[axis]);
    const half = Math.min(0.5 * available,
      Math.max(0.5 * minimum, 0.5 * (box.max[axis] - box.min[axis]) * factor));
    // Keep the requested size and slide the body back inside rather than
    // clipping it against the wall it was pushed through.
    const shift = Math.min(limits.max[axis] - (centre + half), Math.max(limits.min[axis] - (centre - half), 0));
    min[axis] = snap(centre - half + shift, step, limits.min[axis]);
    max[axis] = snap(centre + half + shift, step, limits.min[axis]);
    if (max[axis] - min[axis] < minimum) max[axis] = min[axis] + minimum;
  });
  return { min, max };
}

/**
 * Slide the body without reshaping it, held inside the container.
 *
 * A box already touching a wall stops there rather than being squashed against
 * it: the gesture is a move, so it must never silently change the volume.
 */
export function moveFluidBodyBox(
  box: FluidBodyBox,
  centre_m: Vec3,
  scene: SceneDescription,
): FluidBodyBox {
  return moveBoxWithinLimits(box, centre_m, fluidBodyLimits(scene));
}

/**
 * Grow or shrink the body by a factor of its *volume*, which is what the
 * control's litres readout promises. The container clamps each axis, so a body
 * already against a wall grows only where it still can.
 */
export function scaleFluidBodyVolume(
  box: FluidBodyBox,
  volumeFactor: number,
  scene: SceneDescription,
): FluidBodyBox {
  return scaleFluidBodyBox(box, Math.cbrt(volumeFactor), scene);
}

export function fluidBodyBoxVolume_m3(box: FluidBodyBox): number {
  return Math.max(0, box.max.x - box.min.x)
    * Math.max(0, box.max.y - box.min.y)
    * Math.max(0, box.max.z - box.min.z);
}

/**
 * Every cubic metre of water the document describes, base and seeds together.
 *
 * Seeded bricks are counted as bricks rather than as their bounding boxes: a
 * painted L holds the volume of its three bricks, and a readout that reported
 * the box around them would be wrong by the corner that is not there.
 */
export function fluidWaterVolume_m3(scene: SceneDescription): number {
  const base = fluidBodyBox(scene);
  const lattice = editorFluidLattice(scene);
  const brick_m3 = lattice.brickSize_m.x * lattice.brickSize_m.y * lattice.brickSize_m.z;
  const bricks = fluidSeedBodies(scene)
    .reduce((total, body) => total + body.brickCount, 0);
  // Analytic volumes are counted from their own closed forms — a ball holds
  // 4/3 pi r^3, not the cube its gizmo draws — for the same reason the seeds
  // are counted as bricks rather than as the box around them.
  const analytic = fluidVolumeBodies(scene)
    .reduce((total, body) => total + fluidVolumeVolume_m3(body.volume), 0);
  return (base ? fluidBodyBoxVolume_m3(base) : 0) + bricks * brick_m3 + analytic;
}

/**
 * Author a box back into the document.
 *
 * The reservoir stays anchor-free only when it has to: a box still sitting in
 * the container's minimum corner omits the origin entirely, which keeps legacy
 * scenes byte-identical and keeps the GPU's closed-form t=0 bootstrap — an
 * authored origin costs the analytic path and rasterizes the seed on the host.
 *
 * `fillFraction` is not an independent knob here. `validateScene` requires it
 * to equal the reservoir's share of the container whenever the dimensions are
 * authored, so it is derived, and a tank fill that has just been reshaped into
 * a box carries the same water it had.
 */
export function fluidBodyBoxPatch(
  scene: SceneDescription,
  box: FluidBodyBox,
): Pick<SceneDescription, "container" | "fluid"> {
  const c = scene.container;
  const limits = fluidBodyLimits(scene);
  const size = {
    x: Math.max(0, box.max.x - box.min.x),
    y: Math.max(0, box.max.y - box.min.y),
    z: Math.max(0, box.max.z - box.min.z),
  };
  const origin = {
    x: box.min.x - limits.min.x,
    y: box.min.y - limits.min.y,
    z: box.min.z - limits.min.z,
  };
  const anchored = Math.abs(origin.x) < 1e-9 && Math.abs(origin.y) < 1e-9 && Math.abs(origin.z) < 1e-9;
  const { initialDamBreakOrigin_m: _dropped, ...fluid } = scene.fluid;
  return {
    container: {
      ...c,
      fillFraction: Math.max(0, Math.min(1,
        (size.x * size.y * size.z) / (c.width_m * c.height_m * c.depth_m))),
    },
    fluid: {
      ...fluid,
      initialCondition: "dam-break",
      initialDamBreakDimensions_m: size,
      ...(anchored ? {} : { initialDamBreakOrigin_m: origin }),
    },
  };
}

// ---- seeded bodies --------------------------------------------------------

/**
 * A body of water the seed list describes: one connected run of bricks.
 *
 * Identified by the index of its first seed rather than by its position, so a
 * selection survives the body being moved. The write-back below keeps a
 * component's seeds at the same place in the list for exactly that reason.
 */
export interface FluidSeedBody {
  readonly id: string;
  readonly box: FluidBodyBox;
  /** Indices into `fluid.initialBrickSeeds_m`, ascending. */
  readonly seedIndices: readonly number[];
  /** Distinct bricks wetted — the body's volume, in bricks. */
  readonly brickCount: number;
  /**
   * False for an L, a ring, or a hand-painted blob. Those can be moved — a
   * translation preserves the shape exactly — but not resized, because the only
   * box a resize could act on is the bounding box, and writing that back would
   * replace the author's shape with a rectangle.
   */
  readonly resizable: boolean;
}

export const FLUID_SEED_BODY_PREFIX = "fluid-seed-";

/** Every body the seed list describes, in document order. */
export function fluidSeedBodies(scene: SceneDescription): readonly FluidSeedBody[] {
  if (scene.systems?.fluid === false || !scene.fluid.initialBrickSeeds_m?.length) return [];
  return initialFluidBrickComponents(scene, sceneLatticeDimensions(scene)).map((component) => ({
    id: `${FLUID_SEED_BODY_PREFIX}${component.seedIndices[0]}`,
    box: { min: component.bounds.minimum, max: component.bounds.maximum },
    seedIndices: component.seedIndices,
    brickCount: component.brickCount,
    resizable: component.rectangular,
  }));
}

/** How a seeded body reshapes: onto the brick lattice its seeds are quantized to. */
export function fluidSeedBodyResizePolicy(scene: SceneDescription): BoxResizePolicy {
  const brick = editorFluidLattice(scene).brickSize_m;
  return {
    snap_m: [brick.x, brick.y, brick.z],
    limits: fluidBodyLimits(scene),
    minimum_m: [brick.x, brick.y, brick.z],
  };
}

/**
 * Author a seeded body back into the document as one seed per brick it covers.
 *
 * `seeds` replaces exactly this body's entries, spliced in at the position its
 * first seed held, so every other body keeps its indices and the selection
 * survives the edit. An empty replacement removes the body; emptying the list
 * entirely drops the field, because `validateScene` rejects `[]` — and, when the
 * seeds were replacing the base condition, hands the scene back to its dam.
 */
export function fluidSeedBodyPatch(
  scene: SceneDescription,
  body: FluidSeedBody,
  box: FluidBodyBox | undefined,
): SceneDescription["fluid"] {
  const lattice = editorFluidLattice(scene);
  const replaced = new Set(body.seedIndices);
  const existing = scene.fluid.initialBrickSeeds_m ?? [];
  const seeds: Vec3[] = [];
  const brickRange = (axis: "x" | "y" | "z", limit: number) => {
    if (!box) return [] as number[];
    const size = lattice.brickSize_m[axis];
    const from = Math.max(0, Math.round((box.min[axis] - lattice.origin_m[axis]) / size));
    const to = Math.min(limit, Math.round((box.max[axis] - lattice.origin_m[axis]) / size));
    return Array.from({ length: Math.max(0, to - from) }, (_unused, step) => from + step);
  };
  const written: Vec3[] = [];
  for (const z of brickRange("z", lattice.bricks.z))
    for (const y of brickRange("y", lattice.bricks.y))
      for (const x of brickRange("x", lattice.bricks.x))
        written.push(fluidBrickCenter(lattice, { x, y, z }));
  existing.forEach((seed, index) => {
    if (!replaced.has(index)) { seeds.push(seed); return; }
    // The whole replacement lands where the body's first seed was.
    if (index === body.seedIndices[0]) seeds.push(...written);
  });
  const { initialBrickSeeds_m: _dropped, ...fluid } = scene.fluid;
  return seeds.length === 0 ? fluid : { ...fluid, initialBrickSeeds_m: seeds };
}

/** Slide a seeded body by whole bricks, keeping whatever shape it has. */
export function moveFluidSeedBody(
  scene: SceneDescription,
  body: FluidSeedBody,
  centre_m: Vec3,
): SceneDescription["fluid"] {
  const lattice = editorFluidLattice(scene);
  const moved = moveBoxWithinLimits(body.box, centre_m, fluidBodyLimits(scene));
  const step = (axis: "x" | "y" | "z") =>
    Math.round((moved.min[axis] - body.box.min[axis]) / lattice.brickSize_m[axis]) * lattice.brickSize_m[axis];
  const delta = { x: step("x"), y: step("y"), z: step("z") };
  if (delta.x === 0 && delta.y === 0 && delta.z === 0) return scene.fluid;
  const replaced = new Set(body.seedIndices);
  // Translated per seed rather than regenerated from the box: a blob is not a
  // box, and a move must never be the edit that turns it into one.
  const seeds = (scene.fluid.initialBrickSeeds_m ?? []).map((seed, index) => replaced.has(index)
    ? { x: seed.x + delta.x, y: seed.y + delta.y, z: seed.z + delta.z }
    : seed);
  return { ...scene.fluid, initialBrickSeeds_m: seeds };
}

// ---- entity ---------------------------------------------------------------

function fluidSeedBodyEntityFor(
  context: EditorEntityContext,
  body: FluidSeedBody,
  ordinal: number,
): EditorEntity {
  const { scene } = context;
  const size = boxSize(body.box);
  const label = fluidBodyLabel(scene, ordinal);
  const move = (centre_m: Vec3) => ({ fluid: moveFluidSeedBody(scene, body, centre_m) });
  return {
    selection: { kind: "fluid-body", id: body.id },
    label,
    tone: "fluid",
    frame: WORLD_FRAME,
    box: body.box,
    sizeLabel: `${[size.x, size.y, size.z].map((value) => value.toFixed(2)).join(" × ")} m`,
    handles: [
      // A blob offers no resize handles at all, rather than handles that would
      // rewrite it as its own bounding box.
      ...(body.resizable
        ? boxHandles(body.box, {
          drag: boxResizeDrag(body.box, fluidSeedBodyResizePolicy(scene),
            (next) => ({ fluid: fluidSeedBodyPatch(scene, body, next) })),
        })
        : []),
      ...moveHandles(boxCenter(body.box), move),
    ],
    draftSubject: "fluid-body",
    editLabel: (handle) => handle.space === "world" ? `Moved ${label}` : `Reshaped ${label}`,
    fields: positionFields(boxCenter(body.box), move),
    remove: () => ({ ...scene, fluid: fluidSeedBodyPatch(scene, body, undefined) }),
  };
}

/**
 * What to call one body when the scene has several.
 *
 * Numbered across all of them, base included, because from the viewport they are
 * all just bodies of water: "WATER" beside "WATER 1" reads as two different
 * kinds of thing when the only difference is which field of the document happens
 * to describe them.
 */
function fluidBodyLabel(scene: SceneDescription, ordinal: number): string {
  const total = (fluidBodyBox(scene) ? 1 : 0)
    + fluidSeedBodies(scene).length
    + fluidVolumeBodies(scene).length;
  return total > 1 ? `WATER ${ordinal}` : "WATER";
}

function fluidBodyEntityFor(context: EditorEntityContext): EditorEntity | undefined {
  const scene = context.scene;
  const box = fluidBodyBox(scene);
  if (!box) return undefined;
  const size = [box.max.x - box.min.x, box.max.y - box.min.y, box.max.z - box.min.z];
  return {
    selection: { kind: "fluid-body", id: FLUID_BODY_SELECTION_ID },
    label: fluidBodyLabel(scene, 1),
    tone: "fluid",
    frame: WORLD_FRAME,
    box,
    sizeLabel: `${size.map((value) => value.toFixed(2)).join(" × ")} m`,
    handles: [
      ...boxHandles(box, {
        drag: boxResizeDrag(box, fluidBodyResizePolicy(scene),
          (next) => fluidBodyBoxPatch(scene, next)),
      }),
      // The reservoir moves because `initialDamBreakOrigin_m` exists to let it:
      // that field is the whole reason the body can leave the container's corner.
      ...moveHandles(boxCenter(box),
        (centre) => fluidBodyBoxPatch(scene, moveFluidBodyBox(box, centre, scene))),
    ],
    draftSubject: "fluid-body",
    editLabel: (handle) => handle.space === "world"
      ? "Moved the water body" : "Reshaped the water body",
    fields: positionFields(boxCenter(box),
      (centre) => fluidBodyBoxPatch(scene, moveFluidBodyBox(box, centre, scene))),
  };
}

/**
 * The water body is clicked on its seed box.
 *
 * That box is where the water will be at t=0 rather than where the solver has
 * since carried it, so a click late in a run selects the reservoir from a place
 * the water has already left. That is the honest target: the box is the thing
 * the handles move, and picking what is drawn instead would mean picking a
 * simulation result that no edit can reach.
 *
 * Three lists answer here under one selection kind: the base reservoir, the
 * painted seed bodies, and the analytic volumes of `editor-fluid-volume.ts`.
 * They are all water, they are all numbered in one `WATER n` sequence, and the
 * flyout cannot tell which field described them — which is the point. Which
 * document field a body happens to live in is the editor's problem, not the
 * user's.
 */
/**
 * What pointing at water offers: the verbs that put something at a point.
 *
 * Declared once and shared with the tank, because an empty tank and the water
 * standing in it are the same place as far as this menu is concerned: someone
 * who right-clicks either is asking what they can do to the water there, and a
 * ring that changed as the tank filled would be a ring nobody could aim at.
 *
 * These are also the whole of what a bare point in the room offers — see
 * `sceneActionsAt` — which is why the water's instruments are a separate group
 * below rather than more entries here. Every verb in this list needs a place and
 * nothing else; a ring that opened on the floor and offered to read the solver
 * would be pricing something the click never named.
 *
 * Two levels, the way every weapon wheel does it: the root ring holds a few
 * *kinds* of intention — water, a solid, a region — and choosing one opens the
 * ring of its verbs. A flat ring of every verb at once had eleven wedges over
 * the water, which is past the count a flick can remember directions in; four
 * or six wedges is not. The cost is one extra flick for the verbs that moved
 * down a level, which is the trade the second level exists to make.
 *
 * The shapes come from the shape table, so a shape declared for the solver is
 * reachable here the day it is declared and never has to be added to a menu.
 * Placing one carries it: a solid you asked for at a point is a solid you are
 * holding, and making you find it again afterwards is the step the pie removes.
 */
export function fluidPlayActions(point_m: Vec3): readonly EditorAction[] {
  return [
    {
      id: "water",
      label: "Water",
      icon: "water-ball",
      tone: "fluid",
      hint: "Add, paint, erase or pour water here",
      children: [
        {
          id: "ball",
          label: "Ball",
          icon: "water-ball",
          tone: "fluid",
          hint: "Click inside the tank to drop a ball of water \u00b7 drag out to size it",
          effect: { kind: "arm", tool: "fluid-ball" },
        },
        {
          id: "paint",
          label: "Paint",
          icon: "paint",
          tone: "fluid",
          hint: "Click to add a water brick \u00b7 drag to paint a body of water",
          effect: { kind: "arm", tool: "fluid-paint" },
        },
        {
          id: "erase",
          label: "Erase",
          icon: "erase",
          tone: "fluid",
          hint: "Click or drag to remove painted water bricks",
          effect: { kind: "arm", tool: "fluid-erase" },
        },
        {
          id: "hose",
          label: "Hose",
          icon: "hose",
          tone: "inflow",
          hint: "Click a surface to aim the hose there",
          effect: { kind: "arm", tool: "inflow" },
        },
      ],
    },
    {
      id: "carry-solid",
      label: "Solid",
      icon: "solid",
      tone: "body",
      hint: "Put a solid here and pick it straight up",
      // The shape's own name is the icon's name: the vocabulary in
      // `EditorActionIcon` covers every `SceneShapeName`, so a shape added to
      // the table draws itself in the ring without a second table to update.
      children: SCENE_SHAPES_BY_CODE.map((shape) => ({
        id: shape.name,
        label: shape.label,
        icon: shape.name,
        tone: "body" as const,
        hint: `Place a ${shape.label.toLowerCase()} here and carry it`,
        effect: { kind: "place" as const, shape: shape.name, point_m, carry: true },
      })),
    },
    {
      id: "region",
      label: "Region",
      icon: "region",
      tone: "region",
      hint: "Drag a box over the water to cap how finely it is solved there",
      effect: { kind: "arm", tool: "refinement-region" },
    },
  ];
}

/**
 * The two overlays that read the water rather than shape it.
 *
 * They are about the fluid and not about the room: "what is this advance
 * costing" and "is this solve converging" are questions asked *of a running
 * solve*, which is the same reason the ring is contextual at all. So they are
 * offered wherever the water is — on the body and on the tank that holds it —
 * and nowhere else. Their counterpart for the *picture* is `sceneDocumentActions`
 * on the empty-space ring, where the frame graph sits for the mirror reason.
 *
 * In a dry scene there is no advance for the pipeline wedge to price — the
 * tank still answers the ring, but the only pipeline the scene runs is the
 * renderer's — so the same wedge opens the frame graph instead of a sim
 * pipeline for a solver the scene never starts.
 */
export function fluidInstrumentActions(scene: SceneDescription): readonly EditorAction[] {
  const fluidEnabled = scene.systems?.fluid !== false;
  return [
    fluidEnabled
      ? {
        id: "pipeline",
        label: "Pipeline",
        icon: "pipeline",
        tone: "fluid",
        hint: "Open the advance pipeline: per-stage GPU cost, gates and solver tuning",
        effect: { kind: "open-overlay", overlay: "sim-pipeline" },
      }
      : {
        id: "pipeline",
        label: "Pipeline",
        icon: "render-pipeline",
        tone: "prop",
        hint: "Open the frame graph: per-pass GPU cost, stage ablation and node tuning",
        effect: { kind: "open-overlay", overlay: "render-pipeline" },
      },
    {
      id: "diagnostics",
      label: "Diagnostics",
      icon: "diagnostics",
      tone: "fluid",
      hint: "Open the live instrument cards: divergence, residual, CFL, mass drift",
      effect: { kind: "open-overlay", overlay: "diagnostics" },
    },
  ];
}

/**
 * The whole ring for a point on the water: what you can do to it, then the
 * instruments and probes that read it.
 *
 * Kept apart from `fluidPlayActions` because the play verbs are also what a bare
 * point in the room offers, while nothing below them is: an instrument there
 * would price a solve the click never pointed at, and a probe there would have
 * nothing to aim at — the empty-space ring is composed from a world position,
 * not from a pixel. Anything pointing at *water* gets every half, which is why
 * the tank shares this rather than the placement verbs alone.
 *
 * The instruments and probes share one INSPECT wedge for the same reason the
 * water verbs share one WATER wedge: they are one kind of intention — reading
 * the scene rather than changing it — and four reading wedges on the root ring
 * is what made the water's ring too dense to flick at. See `fluidPlayActions`
 * for the two-level doctrine.
 */
export function fluidRingActions(
  scene: SceneDescription,
  target: EditorActionTarget,
): readonly EditorAction[] {
  return [
    ...fluidPlayActions(target.point_m),
    {
      id: "inspect",
      label: "Inspect",
      icon: "diagnostics",
      tone: "fluid",
      hint: "Instruments and probes: pipelines, diagnostics, and what is behind this pixel",
      children: [
        ...fluidInstrumentActions(scene),
        cellProbeAction(target),
        // A ray through water is as real as a ray into a stone, and the water
        // is where the expensive ones are, so the ray probe is offered here too.
        rayProbeAction(target),
      ],
    },
  ];
}

export const fluidBodyEntity: EditorEntityDefinition = {
  kind: "fluid-body",
  surfacedBy: (tool) => tool === "select",
  instances: (context) => {
    const base = fluidBodyEntityFor(context);
    const seeds = fluidSeedBodies(context.scene);
    const offset = base ? 1 : 0;
    return [
      ...(base ? [base] : []),
      ...seeds.map((body, index) => fluidSeedBodyEntityFor(context, body, offset + index + 1)),
      ...fluidVolumeBodies(context.scene).map((body, index) =>
        volumeEntityFor(context.scene, body, offset + seeds.length + index + 1)),
    ];
  },
  actions: (context, target) => fluidRingActions(context.scene, target),
  find: (context, id) => {
    if (id === FLUID_BODY_SELECTION_ID) return fluidBodyEntityFor(context);
    const offset = fluidBodyBox(context.scene) ? 1 : 0;
    const seeds = fluidSeedBodies(context.scene);
    const seed = seeds.findIndex((body) => body.id === id);
    if (seed >= 0) return fluidSeedBodyEntityFor(context, seeds[seed]!, offset + seed + 1);
    const volumes = fluidVolumeBodies(context.scene);
    const volume = volumes.findIndex((body) => body.id === id);
    if (volume < 0) return undefined;
    return volumeEntityFor(context.scene, volumes[volume]!, offset + seeds.length + volume + 1);
  },
  pick: (context, ray, exclude) => {
    let nearest: { id: string; distance_m: number } | undefined;
    const consider = (id: string, distance_m: number | undefined) => {
      if (pickExcluded(exclude, "fluid-body", id)) return;
      if (distance_m !== undefined && (!nearest || distance_m < nearest.distance_m)) {
        nearest = { id, distance_m };
      }
    };
    const considerBox = (id: string, box: FluidBodyBox | undefined) =>
      consider(id, box && pickSolidBox(ray, box));
    considerBox(FLUID_BODY_SELECTION_ID, fluidBodyBox(context.scene));
    for (const body of fluidSeedBodies(context.scene)) considerBox(body.id, body.box);
    // A ball is picked on the ball, not on the box around it: the corners of
    // that box are the air a click there is trying to reach past.
    for (const body of fluidVolumeBodies(context.scene)) {
      consider(body.id, pickFluidVolume(ray, body.volume));
    }
    return nearest && {
      selection: { kind: "fluid-body", id: nearest.id },
      distance_m: nearest.distance_m,
    };
  },
};

/** The volume entity, under the label its place in the scene's water earns it. */
function volumeEntityFor(
  scene: SceneDescription,
  body: FluidVolumeBody,
  ordinal: number,
): EditorEntity {
  return fluidVolumeEntity(scene, body, fluidBodyLabel(scene, ordinal));
}
