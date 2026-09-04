import {
  boxCenter,
  boxSize,
  boxHandles,
  boxResizeDrag,
  moveBoxWithinLimits,
  moveHandles,
  pickSolidBox,
  pickSolidSphere,
  positionGroup,
  sceneContainerBox,
  WORLD_FRAME,
  type BoxExtent,
  type BoxResizePolicy,
  type EditorEntity,
  type EditorField,
  type EditorRay,
} from "./editor-entity";
import { sceneCellSizes_m } from "./scene-lattice";
import type {
  InitialLiquidCylinder,
  InitialLiquidSphere,
  InitialLiquidVolume,
  SceneDescription,
  Vec3,
} from "./model";

/**
 * Direct manipulation of the analytic liquid volumes — the balls.
 *
 * These are `fluid.initialLiquidVolumes`: balls, domes and blocks of water the
 * document states are present at t = 0, unioned with the dam or the tank fill
 * and with any painted bricks. Until this file they were the one kind of water
 * a scene could carry that the editor could not see, so a paper figure's
 * falling ball could be described but never touched: no click reached it, no
 * handle sized it, and the only way to move one was to edit the scene factory.
 *
 * They are not a separate kind of selectable thing. A volume is a body of water
 * exactly as the reservoir and the painted blobs are, so it resolves under the
 * same `fluid-body` selection kind, carries the same `WATER n` numbering, and
 * opens the same flyout — see `fluidBodyEntity`, which composes these in.
 *
 * Identity is the volume's index in that field, and every write splices back at
 * the same index, so a selection survives being moved, resized or renumbered by
 * a sibling's edit.
 */

/** Selection id prefix of a volume in `fluid.initialLiquidVolumes`. */
export const FLUID_VOLUME_PREFIX = "fluid-volume-";

/** Cells of radius, or of thickness on a block, a volume may not drop below. */
export const FLUID_VOLUME_MINIMUM_CELLS = 1;

/** A ball's default radius as a share of the container's smallest axis. */
const DEFAULT_BALL_SHARE = 1 / 12;

/** One authored volume, resolved against the document that carries it. */
export interface FluidVolumeBody {
  readonly id: string;
  /** Position in `fluid.initialLiquidVolumes`, which is what a write splices. */
  readonly index: number;
  readonly volume: InitialLiquidVolume;
  /** The volume's own bounds — tight, so a dome's box is the dome's box. */
  readonly box: BoxExtent;
  /**
   * False for a dome, whose retained half does not fill its bounds: the only
   * box a resize could act on is that bounding box, and writing it back would
   * turn the author's hemisphere into a ball. Domes move, and take their radius
   * from the numeric field instead. Same rule the painted blobs already follow.
   */
  readonly resizable: boolean;
}

/**
 * Every authored volume, in document order.
 *
 * The order matches `sceneInitialLiquidVolumes`, so the ordinal a body is
 * labelled with here is the one the seeding order would give it.
 */
export function fluidVolumeBodies(scene: SceneDescription): readonly FluidVolumeBody[] {
  if (scene.systems?.fluid === false) return [];
  return (scene.fluid.initialLiquidVolumes ?? []).map(body);
}

function body(volume: InitialLiquidVolume, index: number): FluidVolumeBody {
  return {
    id: `${FLUID_VOLUME_PREFIX}${index}`,
    index,
    volume,
    box: fluidVolumeBox(volume),
    resizable: volume.shape === "box" || volume.shape === "sphere",
  };
}

/**
 * The volume's own bounds.
 *
 * A hemisphere's are the ball's clipped by the cut plane, which is exact for
 * any normal: the support of the retained half along an axis is the full radius
 * on the side the cap points to, and the radius of the cut disc's projection on
 * the other. Reading a dome's box as the whole ball's would draw a gizmo around
 * air the volume does not contain — which is Figure 8's entire silhouette.
 */
export function fluidVolumeBox(volume: InitialLiquidVolume): BoxExtent {
  if (volume.shape === "box") return { min: volume.min_m, max: volume.max_m };
  const r = volume.radius_m;
  const centre = volume.center_m;
  if (volume.shape === "cylinder") {
    return {
      min: { x: centre.x - r, y: centre.y - r, z: centre.z - volume.halfHeight_m },
      max: { x: centre.x + r, y: centre.y + r, z: centre.z + volume.halfHeight_m },
    };
  }
  if (volume.shape === "sphere") {
    return {
      min: { x: centre.x - r, y: centre.y - r, z: centre.z - r },
      max: { x: centre.x + r, y: centre.y + r, z: centre.z + r },
    };
  }
  const normal = unitNormal(volume.outwardNormal);
  const reach = (component: number) => r * Math.sqrt(Math.max(0, 1 - component * component));
  return {
    min: {
      x: centre.x - (normal.x >= 0 ? r : reach(normal.x)),
      y: centre.y - (normal.y >= 0 ? r : reach(normal.y)),
      z: centre.z - (normal.z >= 0 ? r : reach(normal.z)),
    },
    max: {
      x: centre.x + (normal.x <= 0 ? r : reach(normal.x)),
      y: centre.y + (normal.y <= 0 ? r : reach(normal.y)),
      z: centre.z + (normal.z <= 0 ? r : reach(normal.z)),
    },
  };
}

function unitNormal(normal: Vec3): Vec3 {
  const length = Math.hypot(normal.x, normal.y, normal.z);
  return length > 1e-12
    ? { x: normal.x / length, y: normal.y / length, z: normal.z / length }
    : { x: 1, y: 0, z: 0 };
}

/** The centre a move handle drags: the ball's own centre, or the block's. */
export function fluidVolumeCentre(volume: InitialLiquidVolume): Vec3 {
  return volume.shape === "box" ? boxCenter(fluidVolumeBox(volume)) : volume.center_m;
}

/** Water the volume holds, for the litres readout. */
export function fluidVolumeVolume_m3(volume: InitialLiquidVolume): number {
  if (volume.shape === "box") {
    const size = boxSize(fluidVolumeBox(volume));
    return Math.max(0, size.x) * Math.max(0, size.y) * Math.max(0, size.z);
  }
  const ball = (4 / 3) * Math.PI * volume.radius_m ** 3;
  if (volume.shape === "sphere") return ball;
  if (volume.shape === "cylinder") return Math.PI * volume.radius_m ** 2 * (2 * volume.halfHeight_m);
  return 0.5 * ball;
}

/**
 * How a volume reshapes: onto the finest lattice, never thinner than a cell.
 *
 * A ball's three axes are one linked group because the schema stores one
 * radius for them, which is what keeps a dragged ball a ball — the same
 * mechanism a rigid sphere's gizmo uses. The container is deliberately *not* a
 * limit: a ball may hang over a wall, `validateScene` bounds only its centre,
 * and clipping is how a scene states a volume that starts half-submerged.
 */
export function fluidVolumeResizePolicy(
  scene: SceneDescription,
  volume: InitialLiquidVolume,
): BoxResizePolicy {
  const cell = sceneCellSizes_m(scene);
  const minimum_m: [number, number, number] = [
    FLUID_VOLUME_MINIMUM_CELLS * cell[0]!,
    FLUID_VOLUME_MINIMUM_CELLS * cell[1]!,
    FLUID_VOLUME_MINIMUM_CELLS * cell[2]!,
  ];
  return {
    snap_m: [cell[0]!, cell[1]!, cell[2]!],
    minimum_m: volume.shape === "box" ? minimum_m : [2 * minimum_m[0], 2 * minimum_m[1], 2 * minimum_m[2]],
    ...(volume.shape === "box" ? {} : { links: [["x", "y", "z"] as const] }),
  };
}

/**
 * The volume a reshaped box describes.
 *
 * A ball reads its radius off the box the linked resize produced, which is
 * already symmetric about the centre, so the mean of the three half-extents is
 * exact rather than a reconciliation of three disagreeing numbers.
 */
export function fluidVolumeFromBox(volume: InitialLiquidVolume, box: BoxExtent): InitialLiquidVolume {
  if (volume.shape === "box") return { ...volume, min_m: box.min, max_m: box.max };
  const size = boxSize(box);
  return { ...volume, center_m: boxCenter(box), radius_m: Math.max(0, (size.x + size.y + size.z) / 6) };
}

/** Slide a volume, holding whatever shape it has. */
export function moveFluidVolume(
  scene: SceneDescription,
  volume: InitialLiquidVolume,
  centre_m: Vec3,
): InitialLiquidVolume {
  const limits = sceneContainerBox(scene);
  if (volume.shape === "box") {
    const moved = moveBoxWithinLimits(fluidVolumeBox(volume), centre_m, limits);
    return { ...volume, min_m: moved.min, max_m: moved.max };
  }
  // The centre, not the box: a ball hanging over a wall is legal, and holding
  // the whole ball inside would make the wall unreachable for exactly the
  // volumes — a dome cut by the vessel, a ball resting in a corner — that need
  // to touch it.
  return {
    ...volume,
    center_m: {
      x: Math.min(limits.max.x, Math.max(limits.min.x, centre_m.x)),
      y: Math.min(limits.max.y, Math.max(limits.min.y, centre_m.y)),
      z: Math.min(limits.max.z, Math.max(limits.min.z, centre_m.z)),
    },
  };
}

/** Set a ball's or a dome's radius, in metres, holding its centre. */
export function resizeFluidVolumeRadius(
  scene: SceneDescription,
  volume: InitialLiquidVolume,
  radius_m: number,
): InitialLiquidVolume {
  if (volume.shape === "box") return volume;
  const cell = Math.min(...sceneCellSizes_m(scene));
  return { ...volume, radius_m: Math.max(FLUID_VOLUME_MINIMUM_CELLS * cell, radius_m) };
}

/**
 * Author a volume back into the document, or remove it.
 *
 * Removing the last entry drops the field rather than leaving `[]`, which
 * `validateScene` rejects.
 */
export function fluidVolumePatch(
  scene: SceneDescription,
  target: FluidVolumeBody,
  volume: InitialLiquidVolume | undefined,
): SceneDescription["fluid"] {
  const kept = (scene.fluid.initialLiquidVolumes ?? [])
    .map((existing, index) => index !== target.index ? existing : volume)
    .filter((existing) => existing !== undefined);
  const { initialLiquidVolumes: _dropped, ...fluid } = scene.fluid;
  return kept.length === 0 ? fluid : { ...fluid, initialLiquidVolumes: kept };
}

/**
 * The half-depth of a 2D reconstruction's slab, or undefined in a real volume.
 *
 * A 2D case is run here as a domain a few cells deep, and its liquid is
 * extruded through that depth — see `disk` in the CM12 figures. The depth is
 * therefore not a dimension the user is dropping water into: it is the
 * thickness of the plane they are looking at, so a drop has to span it rather
 * than be scaled by it.
 */
export function fluidSlabHalfDepth_m(scene: SceneDescription): number | undefined {
  const c = scene.container;
  return c.depthBoundary === "symmetry" ? 0.5 * c.depth_m : undefined;
}

/**
 * The radius a freshly dropped ball gets: a twelfth of the container's smallest
 * axis, and never less than two cells of water, so the first click always makes
 * something the solver can actually resolve.
 *
 * A slab's depth is excluded from that minimum, because it is not an axis the
 * water is being sized against — measuring against it would make the standard
 * drop into a 2D figure two cells across.
 */
export function defaultFluidBallRadius_m(scene: SceneDescription): number {
  const c = scene.container;
  const cell = Math.min(...sceneCellSizes_m(scene));
  const across = fluidSlabHalfDepth_m(scene) !== undefined
    ? Math.min(c.width_m, c.height_m)
    : Math.min(c.width_m, c.height_m, c.depth_m);
  return Math.max(2 * cell, DEFAULT_BALL_SHARE * across);
}

/**
 * The volume a drop authors: a ball, or the disk a 2D case's water is.
 *
 * Both paths a drop can take — authored into the document at t = 0, injected
 * into the running solve after it — build the shape here, so the water that
 * lands is the same water either way.
 */
function describedFluidDrop(
  scene: SceneDescription,
  centre_m: Vec3,
  radius_m: number,
): InitialLiquidSphere | InitialLiquidCylinder {
  const halfHeight_m = fluidSlabHalfDepth_m(scene);
  const radius = Math.max(radius_m, FLUID_VOLUME_MINIMUM_CELLS * Math.min(...sceneCellSizes_m(scene)));
  const seed: InitialLiquidSphere | InitialLiquidCylinder = halfHeight_m === undefined
    ? { shape: "sphere", center_m: centre_m, radius_m: radius }
    // Centred in the slab, spanning it: a 2D case has no off-plane position to
    // drop at, and water that stopped short of the walls would not be 2D.
    : { shape: "cylinder", center_m: { ...centre_m, z: 0 }, radius_m: radius, halfHeight_m };
  return seed;
}

export function fluidDropVolume(
  scene: SceneDescription,
  centre_m: Vec3,
  radius_m: number,
): InitialLiquidSphere | InitialLiquidCylinder {
  const seed = describedFluidDrop(scene, centre_m, radius_m);
  // Through the move so a drop aimed past a wall lands on it rather than
  // authoring a centre `validateScene` will reject.
  return moveFluidVolume(scene, seed, seed.center_m) as InitialLiquidSphere | InitialLiquidCylinder;
}

/** The same drop shape for a live SparseWorld edit, without a vessel clamp. */
export function fluidInteractionDropVolume(
  scene: SceneDescription,
  centre_m: Vec3,
  radius_m: number,
): InitialLiquidSphere | InitialLiquidCylinder {
  return describedFluidDrop(scene, centre_m, radius_m);
}

/** Whether a drop must create fluid authority instead of entering it live. */
export function fluidBallRequiresInitialCondition(
  scene: Pick<SceneDescription, "systems">,
  simulationTime_s: number,
  insideContainer: boolean,
): boolean {
  return scene.systems?.fluid === false && simulationTime_s <= 0 && insideContainer;
}

/** Drop a new ball into the document, and say what to select. */
export function addFluidBall(
  scene: SceneDescription,
  centre_m: Vec3,
  radius_m: number,
): { readonly fluid: SceneDescription["fluid"];
  readonly systems?: SceneDescription["systems"]; readonly id: string } {
  const existing = scene.fluid.initialLiquidVolumes ?? [];
  const fluid = {
    ...scene.fluid,
    initialLiquidVolumes: [...existing, fluidDropVolume(scene, centre_m, radius_m)],
  };
  return {
    fluid,
    // A liquid drop is an explicit request for fluid authority. Dry showcase
    // scenes otherwise accept the seed into their document while continuing to
    // run the renderer-only SVO, so the environment rebuilds and no liquid can
    // ever appear. Match the paint tool's first-wet-cell transition.
    ...(scene.systems?.fluid === false
      ? { systems: { ...scene.systems, fluid: true } }
      : {}),
    id: `${FLUID_VOLUME_PREFIX}${existing.length}`,
  };
}

/** What a volume is, for the hover chip and the history entry. */
export function fluidVolumeNoun(volume: InitialLiquidVolume): string {
  return volume.shape === "box" ? "block"
    : volume.shape === "sphere" ? "ball"
      : volume.shape === "cylinder" ? "disk" : "dome";
}

function pickZCylinder(ray: EditorRay, volume: Extract<InitialLiquidVolume, { shape: "cylinder" }>): number | undefined {
  const ox = ray.origin.x - volume.center_m.x;
  const oy = ray.origin.y - volume.center_m.y;
  const oz = ray.origin.z - volume.center_m.z;
  const { x: dx, y: dy, z: dz } = ray.direction;
  let closest = Infinity;
  const radialA = dx * dx + dy * dy;
  if (radialA > 1e-15) {
    const radialB = 2 * (ox * dx + oy * dy);
    const radialC = ox * ox + oy * oy - volume.radius_m ** 2;
    const discriminant = radialB * radialB - 4 * radialA * radialC;
    if (discriminant >= 0) {
      const root = Math.sqrt(discriminant);
      for (const hit of [(-radialB - root) / (2 * radialA), (-radialB + root) / (2 * radialA)]) {
        if (hit > 0 && Math.abs(oz + hit * dz) <= volume.halfHeight_m + 1e-9) closest = Math.min(closest, hit);
      }
    }
  }
  if (Math.abs(dz) > 1e-15) {
    for (const cap of [-volume.halfHeight_m, volume.halfHeight_m]) {
      const hit = (cap - oz) / dz;
      if (hit > 0 && Math.hypot(ox + hit * dx, oy + hit * dy) <= volume.radius_m + 1e-9) {
        closest = Math.min(closest, hit);
      }
    }
  }
  return Number.isFinite(closest) ? closest : undefined;
}

/** Where a ray meets the volume itself, rather than the box around it. */
export function pickFluidVolume(ray: EditorRay, volume: InitialLiquidVolume): number | undefined {
  if (volume.shape === "box") return pickSolidBox(ray, fluidVolumeBox(volume));
  if (volume.shape === "cylinder") return pickZCylinder(ray, volume);
  const hit = pickSolidSphere(ray, volume.center_m, volume.radius_m);
  if (hit === undefined || volume.shape === "sphere") return hit;
  // A dome is the ball intersected with a half-space, so a hit on the removed
  // side has to walk to the plane rather than being reported where the ball's
  // surface would have been.
  const normal = unitNormal(volume.outwardNormal);
  const along = ray.direction.x * normal.x + ray.direction.y * normal.y + ray.direction.z * normal.z;
  const inside = (t: number) => {
    const p = {
      x: ray.origin.x + ray.direction.x * t - volume.center_m.x,
      y: ray.origin.y + ray.direction.y * t - volume.center_m.y,
      z: ray.origin.z + ray.direction.z * t - volume.center_m.z,
    };
    return p.x * normal.x + p.y * normal.y + p.z * normal.z <= 1e-9;
  };
  if (inside(hit)) return hit;
  if (Math.abs(along) < 1e-12) return undefined;
  const offset = {
    x: ray.origin.x - volume.center_m.x,
    y: ray.origin.y - volume.center_m.y,
    z: ray.origin.z - volume.center_m.z,
  };
  const plane = -(offset.x * normal.x + offset.y * normal.y + offset.z * normal.z) / along;
  if (!(plane > 0)) return undefined;
  const radial = Math.hypot(
    ray.origin.x + ray.direction.x * plane - volume.center_m.x,
    ray.origin.y + ray.direction.y * plane - volume.center_m.y,
    ray.origin.z + ray.direction.z * plane - volume.center_m.z,
  );
  return radial <= volume.radius_m ? plane : undefined;
}

/**
 * A volume as an editable entity, under the label its ordinal earns it.
 *
 * The label is passed in rather than derived because the numbering spans every
 * body of water in the scene — reservoir, painted blobs and these — and only
 * `fluidBodyEntity` sees all three lists.
 */
export function fluidVolumeEntity(
  scene: SceneDescription,
  target: FluidVolumeBody,
  label: string,
): EditorEntity {
  const { volume } = target;
  const noun = fluidVolumeNoun(volume);
  const write = (next: InitialLiquidVolume | undefined) => ({ fluid: fluidVolumePatch(scene, target, next) });
  const move = (centre_m: Vec3) => write(moveFluidVolume(scene, volume, centre_m));
  const size = boxSize(target.box);
  const cell = Math.min(...sceneCellSizes_m(scene));
  const radiusField: readonly EditorField[] = volume.shape === "box" ? [] : [{
    id: "radius",
    label: "RADIUS",
    unit: "m",
    value: volume.radius_m,
    step: cell,
    min: FLUID_VOLUME_MINIMUM_CELLS * cell,
    apply: (value: number) => write(resizeFluidVolumeRadius(scene, volume, value)),
  }];
  return {
    selection: { kind: "fluid-body", id: target.id },
    label,
    tone: "fluid",
    frame: WORLD_FRAME,
    box: target.box,
    sizeLabel: volume.shape === "box"
      ? `${[size.x, size.y, size.z].map((value) => value.toFixed(2)).join(" × ")} m`
      : `${(2 * volume.radius_m).toFixed(2)} m ${noun}`,
    handles: [
      ...(target.resizable
        ? boxHandles(target.box, {
          drag: boxResizeDrag(target.box, fluidVolumeResizePolicy(scene, volume),
            (next) => write(fluidVolumeFromBox(volume, next))),
        })
        : []),
      ...moveHandles(fluidVolumeCentre(volume), move),
    ],
    draftSubject: "fluid-body",
    editLabel: (handle) => handle.space === "world" ? `Moved ${label}` : `Resized ${label}`,
    // A ball's size is the one thing about it that nothing else on screen
    // states, so it stays a row; where it is, is what the handles are for.
    fields: radiusField,
    groups: [positionGroup(fluidVolumeCentre(volume), move)],
    summary: `${fluidVolumeVolume_m3(volume).toFixed(3)} m³ of water, seeded as a ${noun} at t = 0`,
    remove: () => ({ ...scene, fluid: fluidVolumePatch(scene, target, undefined) }),
  };
}
