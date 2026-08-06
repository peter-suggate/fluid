import {
  HERO_GARDEN_CONTAINER,
  HERO_GARDEN_SET_SEED,
  HERO_GARDEN_VESSEL,
  HERO_GARDEN_WATERLINE_M,
} from "../hero-garden-scene";
import type { Quaternion, SceneDescription, Vec3 } from "../model";
import { quaternionMultiply } from "../rigid-body";
import type { SceneryGraph, SceneryMaterial, SceneryNode, SceneryPlacement } from "../scenery-graph";
import { SVO_PRIMITIVE_CANDIDATE_MAXIMUM_LEAVES } from "../svo-primitive-candidates";
import { terrainHeightAt } from "../terrain";
import { pondVesselPlanCurve, pondVesselPlanDistance } from "./pond-vessel";
import {
  stoneSetBoulderStations,
  stoneSetSteppingStations,
  type StoneSetSpec,
  type StoneStation,
} from "./stone-set";

/**
 * Where the objects in a set stand, how big they are, and which way they face.
 *
 * This is the *placement* half of the form/spec split the rest of this directory
 * is built on (see `README.md`). A generator's `Spec` is its `Form` plus a key, a
 * position, a ground query and a seed; a `LayoutPlacement` is exactly that spec
 * with the form taken out. Which means the same row can be handed to a crude
 * stand-in today and to `mushroomBoulderNodes` tomorrow without being rewritten —
 * moving an object is one edit here and no edit anywhere the object is drawn.
 *
 * It exists because composition is a decision that nothing else in the pipeline
 * owns. A species knows how to grow one specimen anywhere; only a layout knows
 * that the graded stone set belongs on the left bank, that the tree answers it
 * across the water, and that the discs wade between the two. Those decisions
 * change constantly while a scene is being art-directed, and they are the ones
 * every generator has to agree about, so they want a single table.
 *
 * There are two emitters over any table, and the second is the reason this exists
 * before the detail does:
 *
 *  - **blocking** puts a crude stand-in at every row — an ellipsoid on a cone for
 *    a capped boulder, a low cylinder for a stepping disc, a lobed parasol on a
 *    stick for a canopy tree. Correct in bulk, position and facing; deliberately
 *    wrong in every other way. It is what makes a composition judgeable while it
 *    is still free to change.
 *  - **budget** puts each row's *detailed* leaf allowance on the lattice as the
 *    same cheap primitives. Nothing about it is pretty. It answers the question
 *    the blocking pass cannot — what a fully populated scene costs to publish and
 *    to draw — at the point where the answer can still move the art direction
 *    rather than only the schedule. The ceiling it is measured against is
 *    `SVO_PRIMITIVE_CANDIDATE_MAXIMUM_LEAVES`, shared by every object in frame.
 *
 * Nothing above this file's second half knows about a pond. The hero table at the
 * bottom takes a rail and a ground query, like every other generator here, and is
 * the only part that has heard of `HERO_GARDEN_VESSEL`.
 */

// ---------------------------------------------------------------------------
// The placement vocabulary.
// ---------------------------------------------------------------------------

/**
 * Which generator a row is destined for.
 *
 * Named for the species in this directory rather than for what a reference image
 * calls them, because the name is a routing decision: it picks the crude stand-in
 * today and it picks the generator when the row is repointed.
 */
export type LayoutSpecies =
  | "mushroom-boulder"
  | "pebble-bed"
  | "stepping-disc"
  | "bonsai"
  | "rosette"
  | "swept-tube";

/**
 * What a row's `lift_m` is measured from.
 *
 * `ground` is the heightfield under the row's own anchor. `level` is the set's
 * one horizontal datum — a waterline, a bench top, a floor slab — which exists
 * because some objects are placed against a *surface* rather than against the
 * solid: a stepping stone's whole job is to break the water by a few
 * millimetres, and one authored against the basin floor would submerge or beach
 * itself the moment the fill changed.
 */
export type LayoutDatum = "ground" | "level";

/**
 * Who publishes a row's real geometry.
 *
 * `authored` means the scene's own graph already contains it, so the stand-in
 * emitter must publish nothing — but the row stays in the table, because
 * composition is about relationships and an object the layout cannot see is an
 * object nothing can be laid out *around*.
 */
export type LayoutAuthority = "layout" | "authored";

/**
 * Ground under a world point, in metres.
 *
 * A function, not a heightfield and never a scene: callers pass
 * `terrainHeightAt(scene.terrain, x, z)` — the baked surface the renderer draws
 * and the solver collides against. A layout that computed its own ground would
 * disagree with the ground the day someone baked the terrain differently.
 */
export type LayoutGroundQuery = (x_m: number, z_m: number) => number;

/**
 * A run of small things following a curve: a pebble bed, a gravel verge, a line
 * of setts.
 *
 * The path arrives as plain points so that a bed can hug a pond, a path, a wall
 * footing or a stream bank. `offsetRail` builds one from any closed outline; the
 * pond hands out `pondVesselPlanCurve` precisely so that this can happen without
 * the bed knowing what made the outline.
 */
export interface LayoutRun {
  readonly path_m: readonly (readonly [number, number])[];
  /** Half the spread across the path. Stone centres jitter within it. */
  readonly halfWidth_m: number;
  /** Smallest and largest stone radius in the grade. */
  readonly grade_m: readonly [number, number];
}

/**
 * One placed object: a generator's spec with the form taken out.
 *
 * `size_m` is the envelope the generator is being handed, and it is a contract in
 * both directions. The generator may fill it and must not exceed it, and the
 * layout may rely on it when deciding what stands next to what.
 * `tests/hero-layout.test.ts` holds the stand-ins to it, so a blocking pass
 * cannot quietly promise more room than the composition has.
 */
export interface LayoutPlacement {
  /** Identity, and the prefix of every node id this row publishes. */
  readonly key: string;
  readonly species: LayoutSpecies;
  /** Footprint centre in world metres. */
  readonly at_m: readonly [number, number];
  readonly datum: LayoutDatum;
  /** Signed offset from the datum to the object's own base. */
  readonly lift_m: number;
  /** Full extents about the anchor, before rotation: width, height above the base, depth. */
  readonly size_m: readonly [number, number, number];
  readonly yaw_rad: number;
  /** Lean off vertical, about the yaw-rotated X axis. Capped stones want it; most rows leave it zero. */
  readonly tilt_rad: number;
  /** The only entropy this object gets. */
  readonly seed: number;
  /** Leaves the generator may publish. Summed over a table, this is what the ceiling is spent on. */
  readonly detailBudget: number;
  readonly authority: LayoutAuthority;
  /** Present exactly when the species follows a run rather than standing at a point. */
  readonly run?: LayoutRun;
}

/** The two things a table needs resolved before it can be stood up anywhere. */
export interface LayoutWorld {
  readonly groundHeightAt: LayoutGroundQuery;
  /** What `datum: "level"` resolves to. */
  readonly level_m: number;
}

export type LayoutFidelity = "blocking" | "budget";

export interface LayoutOptions {
  /** Defaults to `blocking`: the crude stand-ins a composition is judged on. */
  readonly fidelity?: LayoutFidelity;
  /** Restrict emission to these species. Omitted emits every one. */
  readonly include?: readonly LayoutSpecies[];
}

/**
 * How many stand-ins a run gets at blocking fidelity, and how much bigger each
 * one has to be to still mass out the run.
 *
 * A pebble bed is the one family whose *count* is its look, so a bed reduced to
 * eight beads stops being a bed. Sixty-four keeps the reading and stays cheap;
 * the cube-root scale is the coarse-graining that follows — one stand-in for n
 * stones covers their volume at n^(1/3) the radius.
 */
export const LAYOUT_RUN_BLOCKING_STONES = 64;

/** Blades per rosette stand-in. Five is the fewest that reads as spiky from any bearing. */
const ROSETTE_BLADES = 5;

/** Samples a rail carries by default. Sixteen per lobe holds a metre pond inside a millimetre. */
const RAIL_SAMPLES = 64;

/** Integer avalanche hash in [0, 1). Same table in, same nodes out. */
function hash01(n: number): number {
  let h = Math.imul(n ^ 0x9e3779b9, 0x85ebca6b) >>> 0;
  h = Math.imul(h ^ (h >>> 13), 0xc2b2ae35) >>> 0;
  return ((h ^ (h >>> 16)) >>> 0) / 0x1_0000_0000;
}

const hashSigned = (n: number): number => 2 * hash01(n) - 1;

const V3 = (x: number, y: number, z: number): Vec3 => ({ x, y, z });

/**
 * The stand-ins' albedo, in the band `lib/voxel-scenery/stone-set.ts` authors
 * the finished quarry in.
 *
 * The values here used to run 0.68-0.88, which was right while the hero scene
 * shaded props through a closure with its own colour grain. It now renders every
 * surface through a flat `plaster` closure with none, and a raised key so that
 * white reads as white — so the authored value *is* the albedo, and a 0.68
 * scaffold comes back a grey lump beside a white rim. Everything below is a
 * creamy white for the same reason its detailed counterpart is.
 */
const STONE = (value: number): SceneryMaterial => ({ palette: "stone", value });
const CLAY = (value: number): SceneryMaterial => ({ palette: "clay", value });

/** Where a row's base sits, in world metres. */
export function layoutSeat_m(placement: LayoutPlacement, world: LayoutWorld): Vec3 {
  const [x, z] = placement.at_m;
  const datum = placement.datum === "level" ? world.level_m : world.groundHeightAt(x, z);
  return { x, y: datum + placement.lift_m, z };
}

/** Yaw then lean, in the repository's wxyz order. */
function orientationFor(placement: LayoutPlacement): Quaternion | undefined {
  if (placement.yaw_rad === 0 && placement.tilt_rad === 0) return undefined;
  const yaw: Quaternion = { w: Math.cos(placement.yaw_rad / 2), x: 0, y: Math.sin(placement.yaw_rad / 2), z: 0 };
  if (placement.tilt_rad === 0) return yaw;
  const tilt: Quaternion = { w: Math.cos(placement.tilt_rad / 2), x: Math.sin(placement.tilt_rad / 2), y: 0, z: 0 };
  return quaternionMultiply(yaw, tilt);
}

/** q * (0, v) * q^-1, in the repository's wxyz order. */
function rotateByQuaternion(q: Quaternion | undefined, v: Vec3): Vec3 {
  if (!q) return v;
  const tx = 2 * (q.y * v.z - q.z * v.y);
  const ty = 2 * (q.z * v.x - q.x * v.z);
  const tz = 2 * (q.x * v.y - q.y * v.x);
  return {
    x: v.x + q.w * tx + (q.y * tz - q.z * ty),
    y: v.y + q.w * ty + (q.z * tx - q.x * tz),
    z: v.z + q.w * tz + (q.x * ty - q.y * tx),
  };
}

export interface OffsetRailOptions {
  /** Start and end as fractions of one turn around the outline. May wrap past 1. */
  readonly fromTurn: number;
  readonly toTurn: number;
  /** Displacement along the outline's own outward normal. Negative runs inside it. */
  readonly offset_m: number;
  readonly samples?: number;
}

/**
 * An arc of a closed outline, displaced along its own normal.
 *
 * Offset along the polyline normal rather than radially from a centre: an outline
 * that wobbles by 8 % would ride a radial offset up and down the flank it was
 * meant to follow. The outward normal of a curve wound counter-clockwise in
 * (x, z) is `(tangent.z, -tangent.x)`, and the samples are respaced at a fixed
 * count so a run's stones stay evenly spread whatever arc it covers.
 */
export function offsetRail(
  outline: readonly (readonly [number, number])[],
  options: OffsetRailOptions,
): readonly (readonly [number, number])[] {
  const samples = options.samples ?? RAIL_SAMPLES;
  if (!Number.isInteger(samples) || samples < 2) throw new RangeError("An offset rail needs at least two samples");
  if (outline.length < 3) throw new RangeError("An offset rail needs a closed outline");
  const count = outline.length;
  const points: (readonly [number, number])[] = [];
  for (let step = 0; step < samples; step += 1) {
    const turn = options.fromTurn + (options.toTurn - options.fromTurn) * (step / (samples - 1));
    const position = (((turn % 1) + 1) % 1) * count;
    const index = Math.floor(position);
    const blend = position - index;
    const [ax, az] = outline[index % count];
    const [bx, bz] = outline[(index + 1) % count];
    const x = ax + (bx - ax) * blend;
    const z = az + (bz - az) * blend;
    const tx = bx - ax;
    const tz = bz - az;
    const length = Math.hypot(tx, tz) || 1;
    points.push([x + (tz / length) * options.offset_m, z + (-tx / length) * options.offset_m]);
  }
  return points;
}

/**
 * The largest stone any fidelity lays in a run.
 *
 * Not simply `grade_m[1]`: blocking fidelity lays a fraction of the stones and
 * coarse-grains them to keep the run's mass, so its stones are the *bigger* of
 * the two. The envelope has to be the union — a contract that only held for the
 * finished object would be broken by the stand-in that stands in for it.
 */
export function layoutRunStoneRadius_m(run: LayoutRun, detailBudget: number): number {
  return run.grade_m[1] * (detailBudget / Math.min(detailBudget, LAYOUT_RUN_BLOCKING_STONES)) ** (1 / 3);
}

/**
 * The box a row's generator may fill.
 *
 * A point row takes it from `size_m` about the anchor, carried through the row's
 * own rotation about its seat — which is exactly what the emitter does to its
 * parts, so a yaw-only bound would be wrong for the leaning stones. A run has no
 * single anchor worth the name, so its envelope is swept from its own path.
 */
export function layoutBounds(
  placement: LayoutPlacement,
  world: LayoutWorld,
): { readonly min: Vec3; readonly max: Vec3 } {
  let min = { x: Infinity, y: Infinity, z: Infinity };
  let max = { x: -Infinity, y: -Infinity, z: -Infinity };
  const cover = (x: number, y: number, z: number) => {
    min = { x: Math.min(min.x, x), y: Math.min(min.y, y), z: Math.min(min.z, z) };
    max = { x: Math.max(max.x, x), y: Math.max(max.y, y), z: Math.max(max.z, z) };
  };
  if (placement.run) {
    const stone = layoutRunStoneRadius_m(placement.run, placement.detailBudget);
    const reach = placement.run.halfWidth_m + stone;
    for (const [x, z] of placement.run.path_m) {
      const base = world.groundHeightAt(x, z) + placement.lift_m;
      cover(x - reach, base, z - reach);
      cover(x + reach, base + 2 * stone, z + reach);
    }
    return { min, max };
  }
  const seat = layoutSeat_m(placement, world);
  const [width, height, depth] = placement.size_m;
  const orientation = orientationFor(placement);
  for (const x of [-.5 * width, .5 * width]) {
    for (const y of [0, height]) {
      for (const z of [-.5 * depth, .5 * depth]) {
        const local = rotateByQuaternion(orientation, V3(x, y, z));
        cover(seat.x + local.x, seat.y + local.y, seat.z + local.z);
      }
    }
  }
  return { min, max };
}

// ---------------------------------------------------------------------------
// The crude stand-ins, one per species.
// ---------------------------------------------------------------------------

/** A crude mushroom: a flattened cap on a barely tapered stem. */
function mushroomBoulderStandIn(placement: LayoutPlacement): SceneryNode[] {
  const capRadius = .5 * placement.size_m[0];
  const height = placement.size_m[1];
  // Read off the reference's largest stone: the stem carries a little over half
  // the height and is nearly as wide as the cap, so the overhang is a lip rather
  // than a brim. Getting that ratio wrong turns a garden stone into a toadstool.
  const stemHeight = .52 * height;
  const capHalfHeight = .24 * height;
  return [
    {
      kind: "cone",
      id: `${placement.key}/stem`,
      tags: ["layout", "placeholder", placement.species],
      place: { units: "metres", position: V3(0, .5 * stemHeight, 0) },
      baseRadius: .83 * capRadius,
      topRadius: .74 * capRadius,
      halfHeight: .5 * stemHeight,
      material: STONE(.88),
    },
    {
      kind: "ellipsoid",
      id: `${placement.key}/cap`,
      tags: ["layout", "placeholder", placement.species],
      place: { units: "metres", position: V3(0, stemHeight + capHalfHeight, 0) },
      radius: V3(capRadius, capHalfHeight, capRadius),
      material: STONE(.92),
    },
  ];
}

/**
 * A disc and the thing that holds it up.
 *
 * A reference photograph cannot show what a stepping stone stands on, because the
 * water hides it. A dry render has no such cover: a disc placed at the level and
 * nothing else hangs in mid air, which reads as a bug rather than as a
 * composition. The support is therefore honest rather than decorative — these
 * stones are set on the floor of the basin — and it is one leaf. It carries a
 * `scaffold` tag because it belongs to the dry preview and not to the object the
 * generator is being handed.
 */
function steppingDiscStandIn(placement: LayoutPlacement, seat: Vec3, world: LayoutWorld): SceneryNode[] {
  const radius = .5 * placement.size_m[0];
  const thickness = placement.size_m[1];
  const floor = world.groundHeightAt(placement.at_m[0], placement.at_m[1]);
  const supportHeight = Math.max(0, seat.y - floor);
  const nodes: SceneryNode[] = [{
    kind: "cylinder",
    id: `${placement.key}/disc`,
    tags: ["layout", "placeholder", placement.species],
    place: { units: "metres", position: V3(0, .5 * thickness, 0) },
    radius,
    halfHeight: .5 * thickness,
    material: STONE(.91),
  }];
  if (supportHeight > 0) {
    nodes.push({
      kind: "cylinder",
      id: `${placement.key}/support`,
      tags: ["layout", "placeholder", placement.species, "scaffold"],
      place: { units: "metres", position: V3(0, -.5 * supportHeight, 0) },
      radius: .30 * radius,
      halfHeight: .5 * supportHeight,
      material: STONE(.82),
    });
  }
  return nodes;
}

/**
 * A lobed parasol on a flared stick, plus three roots so the flare is not a
 * mystery.
 *
 * Two things had to be learned by rendering it. The canopy's *thickness* decides
 * whether this reads as a tree at all: the first pass gave it a tenth of its own
 * width and it came back as an airship moored over the pond — a silhouette with
 * no top, no underside and nothing for the light to turn over. And a single
 * ellipsoid, however thick, is still an airship: in a set where every surface is
 * the same near-white, form comes only from the breaks between masses, so a crown
 * with no breaks in it has no form. Six overlapping lobes is still crude and
 * still cheap, and it is the least that reads as a crown.
 *
 * The trunk gets its own share of the height for the same reason. A canopy set
 * down on its own flare hides the one part of a tree that says which way is up.
 */
function bonsaiStandIn(placement: LayoutPlacement): SceneryNode[] {
  const [width, height, depth] = placement.size_m;
  const canopyHalf = .30 * height;
  const canopyY = height - canopyHalf;
  const trunkHeight = .55 * height;
  const flareRadius = .15 * width;
  const nodes: SceneryNode[] = [
    {
      kind: "cone",
      id: `${placement.key}/trunk`,
      tags: ["layout", "placeholder", placement.species],
      place: { units: "metres", position: V3(0, .5 * trunkHeight, 0) },
      baseRadius: flareRadius,
      topRadius: .055 * width,
      halfHeight: .5 * trunkHeight,
      material: CLAY(.86),
    },
    {
      kind: "ellipsoid",
      id: `${placement.key}/canopy`,
      tags: ["layout", "placeholder", placement.species, "canopy"],
      place: { units: "metres", position: V3(0, canopyY, 0) },
      radius: V3(.36 * width, canopyHalf, .36 * depth),
      material: CLAY(.78),
    },
  ];
  const lobes = 5;
  for (let index = 0; index < lobes; index += 1) {
    const bearing = 2 * Math.PI * (index / lobes) + .7;
    nodes.push({
      kind: "ellipsoid",
      id: `${placement.key}/lobe-${index}`,
      tags: ["layout", "placeholder", placement.species, "canopy"],
      place: {
        units: "metres",
        position: V3(
          .28 * width * Math.cos(bearing),
          canopyY + (hash01(placement.seed + 3 * index) - .5) * .34 * canopyHalf,
          .28 * depth * Math.sin(bearing),
        ),
      },
      radius: V3(.21 * width, .66 * canopyHalf, .21 * depth),
      material: CLAY(.80),
    });
  }
  for (let index = 0; index < 3; index += 1) {
    // Splayed on a fixed third-turn and then carried round by the row's own yaw
    // with everything else, so re-facing the tree re-faces its roots.
    const bearing = (2 * Math.PI * index) / 3 + .4;
    const reach = 1.55 * flareRadius;
    nodes.push({
      kind: "capsule",
      id: `${placement.key}/root-${index}`,
      tags: ["layout", "placeholder", placement.species],
      place: { units: "metres" },
      from: V3(0, .35 * trunkHeight, 0),
      // The root tip stops a capsule-radius clear of the seat rather than at it:
      // a swept circle's conservative bound is a rotated box, so a tip resting on
      // the ground reports itself below it and breaks the envelope contract.
      to: V3(reach * Math.cos(bearing), .042, reach * Math.sin(bearing)),
      radius: .038 * width,
      material: CLAY(.84),
    });
  }
  return nodes;
}

/**
 * A rosette as a core and five splayed blades.
 *
 * The obvious stand-in — one cone, narrow at the root and open at the top — is
 * the rosette's own envelope, and it is *wrong*, because at this size the
 * envelope is the whole of what you see: it rendered as a funnel standing on its
 * point, which reads as plumbing rather than as planting. Five blades is barely
 * more expensive and carries the one property that makes the family recognisable
 * at a glance, which is that it is spiky.
 */
function rosetteStandIn(placement: LayoutPlacement): SceneryNode[] {
  const radius = .5 * placement.size_m[0];
  const nodes: SceneryNode[] = [{
    kind: "ellipsoid",
    id: `${placement.key}/core`,
    tags: ["layout", "placeholder", placement.species],
    place: { units: "metres", position: V3(0, .31 * radius, 0) },
    radius: V3(.34 * radius, .30 * radius, .34 * radius),
    material: CLAY(.84),
  }];
  // Off the rosette radius rather than off the envelope height, so a blade can
  // never lean its way out of the box the generator was promised.
  const bladeLength = 1.18 * radius;
  for (let index = 0; index < ROSETTE_BLADES; index += 1) {
    // An irrational fraction of a turn rather than an even split, so four plants
    // grown from one species do not all show the same star.
    const bearing = 2 * Math.PI * ((index * .618034 + hash01(placement.seed + index)) % 1);
    const lean = .45 + .40 * hash01(placement.seed + 40 + index);
    const rise = .5 * bladeLength * Math.cos(lean);
    const reach = .5 * bladeLength * Math.sin(lean);
    nodes.push({
      kind: "cone",
      id: `${placement.key}/blade-${index}`,
      tags: ["layout", "placeholder", placement.species],
      place: {
        units: "metres",
        position: V3(reach * Math.cos(bearing), .22 * radius + rise, reach * Math.sin(bearing)),
        orientation: bladeOrientation(bearing, lean),
      },
      baseRadius: .17 * radius,
      topRadius: .015 * radius,
      halfHeight: .5 * bladeLength,
      material: CLAY(.90),
    });
  }
  return nodes;
}

/**
 * The rotation taking a blade's local +Y onto the direction it grows in.
 *
 * A cone is placed by a centre and an orientation; the two numbers that actually
 * describe a blade are the bearing it leaves the crown on and how far it leans
 * off vertical. This turns the second pair into the first, so the caller reads as
 * botany rather than as quaternion algebra. The tilt is about the horizontal axis
 * perpendicular to the bearing, which is `cross(+Y, bearing)`.
 */
function bladeOrientation(bearing_rad: number, lean_rad: number): Quaternion {
  const half = Math.sin(.5 * lean_rad);
  return {
    w: Math.cos(.5 * lean_rad),
    x: half * Math.sin(bearing_rad),
    y: 0,
    z: -half * Math.cos(bearing_rad),
  };
}

/**
 * A run's stones, each seated on the ground it actually lies on.
 *
 * Emitted at absolute world positions inside a group pinned to the origin, rather
 * than as offsets from a group datum, because a run has no datum: a bed covers a
 * metre and a half of ground that rises over a terrace and falls to a coping's
 * foot, and one shared y would bury half of it. This is the one species where the
 * group is a selection handle and nothing more.
 */
function runStandIn(placement: LayoutPlacement, world: LayoutWorld, count: number): SceneryNode[] {
  const run = placement.run!;
  const path = run.path_m;
  const [smallest, largest] = run.grade_m;
  const radiusScale = (placement.detailBudget / count) ** (1 / 3);
  const nodes: SceneryNode[] = [];
  for (let index = 0; index < count; index += 1) {
    // Walk the path continuously rather than per vertex, so a run described by
    // sixty-four samples and filled with nine hundred stones lays them evenly
    // instead of in sixty-four clumps. The jitter is on the path *parameter* and
    // then clamped, rather than metric along the tangent afterwards: a nudge at
    // the last vertex would walk a stone off the end of the arc and out of the
    // envelope the run was measured for.
    const step = index + .5 + hashSigned(placement.seed + 7 * index + 3) * .38;
    const position = Math.min(path.length - 1, Math.max(0, (step / count) * (path.length - 1)));
    const vertex = Math.min(path.length - 2, Math.floor(position));
    const blend = position - vertex;
    const [ax, az] = path[vertex];
    const [bx, bz] = path[vertex + 1];
    const tx = bx - ax, tz = bz - az;
    const length = Math.hypot(tx, tz) || 1;
    const across = hashSigned(placement.seed + 7 * index) * run.halfWidth_m;
    const x = ax + (bx - ax) * blend + (tz / length) * across;
    const z = az + (bz - az) * blend + (-tx / length) * across;
    // Graded by a hash rather than by position, because a laid bed is sorted only
    // loosely; a monotonic grade reads as machine-laid edging.
    const grade = hash01(placement.seed + 7 * index + 5) ** 1.6;
    const radius = radiusScale * (smallest + (largest - smallest) * grade);
    const flatten = .62 + .22 * hash01(placement.seed + 7 * index + 11);
    const spin = hash01(placement.seed + 7 * index + 13) * Math.PI;
    nodes.push({
      kind: "ellipsoid",
      id: `${placement.key}/stone-${index}`,
      tags: ["layout", "placeholder", placement.species],
      place: {
        units: "metres",
        position: V3(x, world.groundHeightAt(x, z) + placement.lift_m + flatten * radius, z),
        orientation: { w: Math.cos(spin), x: 0, y: Math.sin(spin), z: 0 },
      },
      radius: V3(radius, flatten * radius, radius * (.78 + .3 * hash01(placement.seed + 7 * index + 17))),
      material: STONE(.83 + .11 * hash01(placement.seed + 7 * index + 19)),
    });
  }
  return nodes;
}

/**
 * Budget fidelity: a row's whole detail allowance, as cheap primitives filling
 * the envelope it was given.
 *
 * This is not a preview of the detailed object and is not trying to be. It is a
 * load of the right *size*, in the right *place*, so that the publication, the
 * candidate BVH, the node-mip pyramid and the frame all see the scene they will
 * eventually be asked to hold. A canopy's fifteen hundred records land inside the
 * canopy because where the records are is most of what an acceleration structure
 * cares about; spraying them over the container would measure a different scene.
 */
function budgetFillStandIn(placement: LayoutPlacement, seat: Vec3, count: number): SceneryNode[] {
  const [width, height, depth] = placement.size_m;
  // A cell of the envelope per record, so the fill has the object's density
  // rather than its silhouette, and one record is never larger than its share.
  const cell = ((width * height * depth) / Math.max(1, count)) ** (1 / 3);
  const radius = Math.max(.0015, .42 * cell);
  const nodes: SceneryNode[] = [];
  for (let index = 0; index < count; index += 1) {
    const u = hash01(placement.seed + 31 * index + 1);
    const v = hash01(placement.seed + 31 * index + 2);
    const w = hash01(placement.seed + 31 * index + 3);
    nodes.push({
      kind: "ellipsoid",
      id: `${placement.key}/budget-${index}`,
      tags: ["layout", "placeholder", placement.species, "budget"],
      place: {
        units: "metres",
        position: V3(
          seat.x + (u - .5) * (width - 2 * radius),
          seat.y + radius + w * Math.max(0, height - 2 * radius),
          seat.z + (v - .5) * (depth - 2 * radius),
        ),
      },
      radius: V3(radius, radius, radius),
      material: STONE(.82 + .12 * hash01(placement.seed + 31 * index + 4)),
    });
  }
  return nodes;
}

/**
 * How many stones a run lays.
 *
 * The only count here that is *chosen* rather than observed: a bed's detailed
 * form genuinely is n graded ellipsoids on a curve, so its budget is its count,
 * and blocking fidelity is the same emitter coarse-grained.
 */
function runStoneCount(placement: LayoutPlacement, fidelity: LayoutFidelity): number {
  return fidelity === "budget"
    ? placement.detailBudget
    : Math.min(placement.detailBudget, LAYOUT_RUN_BLOCKING_STONES);
}

/** One row's stand-in parts, and the frame they are offset from. */
function placeholderFor(
  placement: LayoutPlacement,
  world: LayoutWorld,
  fidelity: LayoutFidelity,
): { readonly children: readonly SceneryNode[]; readonly place: SceneryPlacement } {
  const seat = layoutSeat_m(placement, world);
  if (placement.run) {
    // Runs take the same emitter at both fidelities, which makes them the one
    // species whose budget render is also an honest picture of the finished
    // object. Only the count changes.
    return {
      children: runStandIn(placement, world, runStoneCount(placement, fidelity)),
      place: { units: "metres" },
    };
  }
  if (fidelity === "budget") {
    // Absolute placement, because a budget fill is measured against the envelope
    // and not against the row's own frame; a yaw here would only rotate a cloud
    // of spheres.
    return { children: budgetFillStandIn(placement, seat, placement.detailBudget), place: { units: "metres" } };
  }
  const children = placement.species === "mushroom-boulder" ? mushroomBoulderStandIn(placement)
    : placement.species === "stepping-disc" ? steppingDiscStandIn(placement, seat, world)
      : placement.species === "bonsai" ? bonsaiStandIn(placement)
        : placement.species === "rosette" ? rosetteStandIn(placement)
          : [];
  return { children, place: { units: "metres", position: seat, orientation: orientationFor(placement) } };
}

/**
 * How many leaves one row publishes at a given fidelity.
 *
 * Counted by asking the emitter rather than by keeping a table beside it. The
 * table beside it went stale twice on the first afternoon — a rosette grew from
 * one cone to six parts and a canopy from one ellipsoid to seven, and both times
 * the statistics kept reporting the old number while the render showed the new
 * one. A count that disagrees with the scene is worse than no count, because the
 * whole reason it is published is to see the ceiling coming.
 */
export function layoutLeafCount(
  placement: LayoutPlacement,
  world: LayoutWorld,
  fidelity: LayoutFidelity = "blocking",
): number {
  if (placement.authority === "authored") return 0;
  return placeholderFor(placement, world, fidelity).children.length;
}

/**
 * The stand-ins, as one group node per row in table order.
 *
 * One group each because that is what makes a row a *thing* downstream —
 * `ScenerySpan` records a contiguous range per top-level node, which is what the
 * editor selects and the renderer outlines — and because repointing a row at its
 * real generator then replaces exactly one node.
 */
export function layoutPlaceholderNodes(
  placements: readonly LayoutPlacement[],
  world: LayoutWorld,
  options: LayoutOptions = {},
): SceneryNode[] {
  const fidelity = options.fidelity ?? "blocking";
  const include = options.include ? new Set(options.include) : undefined;
  const nodes: SceneryNode[] = [];
  for (const placement of placements) {
    if (placement.authority === "authored") continue;
    if (include && !include.has(placement.species)) continue;
    const { children, place } = placeholderFor(placement, world, fidelity);
    if (children.length === 0) continue;
    nodes.push({
      kind: "group",
      id: placement.key,
      group: placement.key,
      tags: ["layout", placement.species],
      place,
      children,
    });
  }
  return nodes;
}

export interface LayoutSpeciesStatistics {
  readonly species: LayoutSpecies;
  readonly placements: number;
  readonly placeholderLeaves: number;
  readonly detailedLeaves: number;
}

export interface LayoutStatistics {
  readonly placements: number;
  /** Leaves `layoutPlaceholderNodes` publishes at the requested fidelity. */
  readonly placeholderLeaves: number;
  /** Leaves the finished set will publish: every budget, including what is already authored. */
  readonly detailedLeaves: number;
  readonly ceiling: number;
  /** Ceiling less `detailedLeaves`. Negative means the set cannot be built as specified. */
  readonly headroom: number;
  readonly bySpecies: readonly LayoutSpeciesStatistics[];
}

/**
 * What a table costs now and what it will cost finished.
 *
 * The second number is the point. `SVO_PRIMITIVE_CANDIDATE_MAXIMUM_LEAVES` is a
 * hard 4096 — over it `buildSvoScenePrimitives` publishes no candidate BVH at all
 * and the dry-scene contract rejects the scene outright — and the moment to find
 * out that the stone set, the tree and the planting do not fit together is while
 * each of them is still a row.
 */
export function layoutStatistics(
  placements: readonly LayoutPlacement[],
  world: LayoutWorld,
  options: LayoutOptions = {},
): LayoutStatistics {
  const fidelity = options.fidelity ?? "blocking";
  const include = options.include ? new Set(options.include) : undefined;
  const bySpecies = new Map<LayoutSpecies, { placements: number; placeholderLeaves: number; detailedLeaves: number }>();
  let placeholderLeaves = 0;
  let detailedLeaves = 0;
  for (const placement of placements) {
    const emitted = include && !include.has(placement.species) ? 0 : layoutLeafCount(placement, world, fidelity);
    const row = bySpecies.get(placement.species) ?? { placements: 0, placeholderLeaves: 0, detailedLeaves: 0 };
    row.placements += 1;
    row.placeholderLeaves += emitted;
    row.detailedLeaves += placement.detailBudget;
    bySpecies.set(placement.species, row);
    placeholderLeaves += emitted;
    detailedLeaves += placement.detailBudget;
  }
  return {
    placements: placements.length,
    placeholderLeaves,
    detailedLeaves,
    ceiling: SVO_PRIMITIVE_CANDIDATE_MAXIMUM_LEAVES,
    headroom: SVO_PRIMITIVE_CANDIDATE_MAXIMUM_LEAVES - detailedLeaves,
    bySpecies: [...bySpecies].map(([species, row]) => ({ species, ...row })),
  };
}

// ---------------------------------------------------------------------------
// The hero garden's table. The only part of this file that has heard of a pond.
// ---------------------------------------------------------------------------

/**
 * What the hero table needs to know about the thing it is laid out around.
 *
 * A rail and a width, not a `PondVesselSpec`: the same table shape would lay a
 * set around a plunge pool, a path or a wall footing, and taking the outline
 * rather than the vessel is what keeps that true.
 */
export interface HeroGardenLayoutInput {
  /** The coping's crest centreline, from `pondVesselPlanCurve`. */
  readonly copingRail: readonly (readonly [number, number])[];
  /** Half-width of the coping across that crest, so beds can bank against its foot. */
  readonly copingHalfWidth_m: number;
}

/** The id every row of the hero table is prefixed with, and the handle for replacing them. */
export const HERO_LAYOUT_NODE_PREFIX = "layout/";

/**
 * The spec the stone generator is actually built with, so this table can ask it
 * where it put things instead of keeping a second opinion.
 *
 * Same three numbers `heroGardenScenery` hands its `pond-stone-set` node. If it
 * were ever to disagree, the rows below would describe a set nobody publishes —
 * which is the exact failure they were rewritten to end.
 */
const HERO_STONE_SET: StoneSetSpec = Object.freeze({
  vessel: HERO_GARDEN_VESSEL,
  waterline_m: HERO_GARDEN_WATERLINE_M,
  seed: HERO_GARDEN_SET_SEED,
});

/**
 * A boulder row, taken from where the generator actually stood the stone.
 *
 * These used to be four authored world positions and four cap radii, and by the
 * time anyone looked they were on the *far side of the pond* from the stones
 * they claimed to be — the group had been moved round the rail twice while the
 * table stayed put. Nothing caught it, because an `authored` row publishes
 * nothing and therefore cannot look wrong; it can only quietly mislead whatever
 * is placed against it, which is what put an air plant inside a pebble.
 *
 * `yaw_rad` and `tilt_rad` stay zero. A capped boulder is very nearly a solid of
 * revolution and its lean is a few degrees the generator draws from its own
 * seed, so a facing here would be a number with nothing on the other end of it.
 */
function boulderRow(station: StoneStation, seed: number): LayoutPlacement {
  return {
    key: `${HERO_LAYOUT_NODE_PREFIX}${station.key.replace(/^stone\//, "stone/cap-")}`,
    species: "mushroom-boulder",
    at_m: station.at_m,
    datum: "ground",
    // Bedded, not balanced: a cap stone set on ground sinks a little into it, and
    // a stand-in resting exactly on the heightfield reads as dropped.
    lift_m: -0.012,
    size_m: [station.width_m, station.height_m + station.base_m + 0.012, station.width_m],
    yaw_rad: 0, tilt_rad: 0, seed,
    // Four worn stones as smooth-union clusters, at the plan's own estimate.
    detailBudget: 50,
    authority: "authored",
  };
}

/**
 * A disc row, taken from where the generator's own shoreline solve put the plate.
 *
 * The path is laid on a *contour* of the pond's shelf rather than at authored
 * coordinates, so it is the one family in the set whose positions genuinely
 * cannot be written down here: change the vessel and the contour moves. The row
 * keeps its own vertical semantics — a plate is placed against the water, and
 * `lift_m` is what puts the waterline across its side rather than under it —
 * and takes only the plan position and the size from the solve.
 */
function discRow(station: StoneStation, seed: number): LayoutPlacement {
  const radius_m = 0.5 * station.width_m;
  const thickness_m = 0.17 * 2 * radius_m;
  return {
    key: `${HERO_LAYOUT_NODE_PREFIX}${station.key.replace(/^stone\/path\/step-/, "stone/disc-")}`,
    species: "stepping-disc",
    at_m: station.at_m,
    datum: "level",
    // The reference's discs break the surface by a fraction of their own
    // thickness — enough that the waterline cuts visibly across the side, not so
    // much that they read as beached.
    lift_m: -0.4 * thickness_m,
    size_m: [2 * radius_m, thickness_m, 2 * radius_m],
    yaw_rad: 0, tilt_rad: 0, seed,
    detailBudget: 20,
    authority: "authored",
  };
}

function rosetteRow(
  key: string,
  at_m: readonly [number, number],
  radius_m: number,
  yaw_rad: number,
  seed: number,
): LayoutPlacement {
  return {
    key: `${HERO_LAYOUT_NODE_PREFIX}${key}`,
    species: "rosette",
    at_m,
    datum: "ground",
    lift_m: -0.004,
    size_m: [2.05 * radius_m, 1.9 * radius_m, 2.05 * radius_m],
    yaw_rad, tilt_rad: 0, seed,
    // The planting budget, four ways.
    detailBudget: 100,
    authority: "layout",
  };
}

/**
 * A bed, with its anchor and envelope taken from the arc it follows.
 *
 * Deriving both rather than authoring them keeps one number in charge: move the
 * arc and the row's own idea of where it is moves with it, which is the property
 * the whole table exists for. Offsets are expressed in coping half-widths so a
 * bed banks against the rim's foot however wide the rim is authored.
 */
function bedRow(
  key: string,
  input: HeroGardenLayoutInput,
  arc: {
    readonly fromTurn: number;
    readonly toTurn: number;
    /** Outward displacement from the crest, in coping half-widths. */
    readonly offsetInHalfWidths: number;
    readonly halfWidth_m: number;
    readonly grade_m: readonly [number, number];
  },
  detailBudget: number,
  seed: number,
): LayoutPlacement {
  const path_m = offsetRail(input.copingRail, {
    fromTurn: arc.fromTurn,
    toTurn: arc.toTurn,
    offset_m: arc.offsetInHalfWidths * input.copingHalfWidth_m,
  });
  let minX = Infinity, maxX = -Infinity, minZ = Infinity, maxZ = -Infinity;
  for (const [x, z] of path_m) {
    minX = Math.min(minX, x); maxX = Math.max(maxX, x);
    minZ = Math.min(minZ, z); maxZ = Math.max(maxZ, z);
  }
  const run: LayoutRun = { path_m, halfWidth_m: arc.halfWidth_m, grade_m: arc.grade_m };
  const stone = layoutRunStoneRadius_m(run, detailBudget);
  const reach = arc.halfWidth_m + stone;
  return {
    key: `${HERO_LAYOUT_NODE_PREFIX}${key}`,
    species: "pebble-bed",
    at_m: [.5 * (minX + maxX), .5 * (minZ + maxZ)],
    datum: "ground",
    // One bedding depth for every stone, rather than a fraction of the largest: a
    // bed whose bedding tracked its own coarse-graining would sit at a different
    // height in the blocking pass than in the finished scene.
    lift_m: -0.008,
    size_m: [maxX - minX + 2 * reach, 2 * stone, maxZ - minZ + 2 * reach],
    yaw_rad: 0, tilt_rad: 0, seed, detailBudget, authority: "authored", run,
  };
}

/**
 * Where the reference puts things, read through the hero camera rather than
 * copied off the image.
 *
 * The two are not the same picture, and pretending otherwise is how a layout ends
 * up stacking two objects on one sightline. What the frame's axes are is a
 * two-line derivation from the camera and worth doing again whenever it moves:
 * screen right is `(sin az, 0, -cos az)` and depth is `-(x cos az + z sin az)`.
 * At the current `heroGardenCamera` — azimuth 2.4, elevation 0.40, 1.05 m — that
 * gives
 *
 *   screen right is (-x, -z)      screen left is (+x, +z)
 *   away from the eye is (-x, +z)  toward it is (+x, -z)
 *
 * and it has already been round once: the framing was a half-turn out at azimuth
 * 0.95, where the tree stood between the camera and its own pond. That is the
 * reason nothing below is authored as "left" or "behind". Every row is a world
 * position, and how the frame reads it is the camera's business — which is what
 * lets the camera be re-solved against the set without the set being re-authored.
 *
 * The container is the other constraint, and it binds harder than the reference
 * does. 1.8 x 1.2 m against a frame that is over two metres wide at the pond's
 * own distance means the outer sixth of the image is *outside the domain*: the
 * reference spreads its stones across a third of its frame and there is no honest
 * way to do that here. The set is therefore composed by the reference's
 * *relationships* — a graded stone family on one bank, the tree answering it
 * across the water, discs wading diagonally between them, planting picking out
 * four separate edges — rather than by its screen positions.
 */
export function heroGardenLayoutPlacements(input: HeroGardenLayoutInput): readonly LayoutPlacement[] {
  return Object.freeze([
    // The stone families are `authority: "authored"` — `lib/voxel-scenery/stone-set.ts`
    // now publishes them through the hero scene, so standing in for them would
    // draw every boulder, bed and disc twice. The rows stay because they are the
    // *placement*, and the generator derives its own today: repointing it at
    // these is what makes moving a stone one edit rather than two. Until then
    // they are a description with a reader pending, and they still carry the
    // stone set's share of the primitive ceiling.
    //
    // The graded set is on the left bank and recedes. The grade is the
    // composition — the reference reads as a family precisely because it is one
    // quarry at four sizes *and four proportions*, and its largest is a fifth of
    // the pond's width, big enough to be the frame's second subject after the
    // water. They recede rather than spreading across the frame because the bank
    // they stand on is a 0.2 m strip between the coping and the container wall,
    // and a line of stones laid across the view would have to stand in the pond
    // to do it.
    //
    // Read from the generator rather than restated. See `boulderRow`.
    ...stoneSetBoulderStations(HERO_STONE_SET)
      .map((station, index) => boulderRow(station, 0x51_0a01 + 0x0101 * index)),

    // The pebble beds, as three offsets of the vessel's own outline, and all three
    // *outside* the coping.
    //
    // The reference banks its main bed on a ledge between the rim and the water,
    // and that ledge no longer exists: the inner face was steepened to 35 mm of
    // run for 155 mm of drop, so the wall goes straight from the coping's inner
    // foot to the basin floor with nothing to stand a stone on. Bedding them
    // inside anyway would put them either on the bullnose's flank, perched, or
    // 110 mm under water. Outside is where this vessel has a foot to bank
    // against, and it is also what the reference's near-right corner shows.
    //
    // The grades below are the *published* ones, and they moved a long way. The
    // beds used to grade 7 mm to 21 mm of radius — a factor of three across the
    // whole set — and from a metre away that reads as one grain: an even
    // necklace of beads round the pool. They now run from a 4 mm flake to a
    // 45 mm cobble, heaped coarse against the boulder group and running out to
    // sparse shingle away from it. A row that kept the old numbers would
    // under-reserve the envelope every bed is laid inside of by half.
    // The heap at the boulders' feet, the beach where the shelf runs into the
    // water, and the fine course along the right-hand rim. Each is one of
    // `HERO_PEBBLE_DRIFTS` read back as an arc: middle turn plus and minus its
    // half length, centred at the middle of its own band.
    bedRow("stone/bed-shore", input, {
      fromTurn: 0.642, toTurn: 0.838, offsetInHalfWidths: 3.00,
      halfWidth_m: 0.059, grade_m: [0.004, 0.045],
    }, 480, 0x51_be01),
    bedRow("stone/bed-near", input, {
      fromTurn: 0.396, toTurn: 0.592, offsetInHalfWidths: 2.27,
      halfWidth_m: 0.036, grade_m: [0.004, 0.020],
    }, 300, 0x51_be02),
    bedRow("stone/bed-scatter", input, {
      fromTurn: 0.994, toTurn: 1.286, offsetInHalfWidths: 1.80,
      halfWidth_m: 0.025, grade_m: [0.004, 0.012],
    }, 120, 0x51_be03),

    // Five discs wading out from the bank along a contour of the pond's own
    // shelf, shrinking as they go — the reference's diagonal, graded its way
    // round. Each is far enough inside the outline that its whole rim clears the
    // coping's inner foot; a disc overlapping the wall would read as set into
    // it, and the assertion that pins it is `tests/hero-layout.test.ts`.
    //
    // Read from the generator rather than restated. See `discRow`.
    ...stoneSetSteppingStations(HERO_STONE_SET)
      .map((station, index) => discRow(station, 0x51_d101 + 0x0101 * index)),

    // The bonsai, on the far right and answering the stone set across the water.
    // Its canopy is a parasol, not a ball: the reference's is roughly two and a
    // half times as wide as the whole tree is tall, and that flatness is most of
    // why it reads as a trained specimen rather than as a shrub. It is also the
    // only reason a 0.56 m canopy fits under a 0.60 m container lid.
    //
    // The reference's canopy covers about three fifths of its frame; this one
    // covers a quarter, and the difference is the container rather than the art
    // direction — 0.36 m of canopy depth is all the room there is between the
    // pond's far coping and the domain wall.
    {
      key: `${HERO_LAYOUT_NODE_PREFIX}plant/bonsai`,
      species: "bonsai",
      at_m: [0.300, -0.370],
      datum: "ground",
      lift_m: -0.010,
      size_m: [0.560, 0.290, 0.360],
      yaw_rad: -0.18,
      tilt_rad: 0,
      seed: 0x51_b0_1a,
      detailBudget: 1_500,
      // `lib/voxel-scenery/bonsai.ts` now publishes it through the hero scene, on
      // its own `at_m`. Same story as the stones: the row is the placement the
      // generator wants repointing at, and until then it must stand in for
      // nothing or the tree is drawn twice.
      authority: "authored",
    },

    // Four air plants, each picking out a different edge of the frame so the eye
    // is carried round the pond: far left, far centre, the right verge and the
    // near-right corner.
    //
    // Their anchors were moved once the stone set landed, and that is worth
    // recording. The first four were placed against *this table's* beds, which
    // are not where `stoneSet` actually laid them, and one of them came back
    // inside a pebble. Two authorities placing against different ideas of the
    // same bed is the failure this file exists to remove, and until the stone
    // generator reads its rows from here, planting is chosen against what the
    // scene really publishes rather than against what the table describes.
    rosetteRow("plant/air-1", [0.800, -0.160], 0.028, 0.30, 0x51_a101),
    rosetteRow("plant/air-2", [0.380, -0.540], 0.026, -1.20, 0x51_a202),
    rosetteRow("plant/air-3", [-0.560, -0.460], 0.030, -0.55, 0x51_a303),
    rosetteRow("plant/air-4", [0.320, 0.520], 0.020, 0.90, 0x51_a404),

    // The hose. Authored in `lib/hero-garden-scene.ts` and laid out around, never
    // by, this table — but present in it, because a composition that could not see
    // the scene's subject would happily put a boulder through it. Five capsule
    // runs and a ferrule is what the scene already publishes.
    {
      key: `${HERO_LAYOUT_NODE_PREFIX}hose`,
      species: "swept-tube",
      at_m: [0.280, 0.060],
      datum: "ground",
      lift_m: 0,
      size_m: [0.700, 0.320, 0.260],
      yaw_rad: 0,
      tilt_rad: 0,
      seed: 0,
      detailBudget: 6,
      authority: "authored",
    },
  ]);
}

/** The hero table's input, from the vessel it is laid out around. */
export function heroGardenLayoutInput(): HeroGardenLayoutInput {
  return {
    copingRail: pondVesselPlanCurve(HERO_GARDEN_VESSEL),
    copingHalfWidth_m: HERO_GARDEN_VESSEL.rimHalfWidth_m,
  };
}

export function heroGardenLayout(): readonly LayoutPlacement[] {
  return heroGardenLayoutPlacements(heroGardenLayoutInput());
}

/**
 * The ground and the level the hero table stands on.
 *
 * Ground is the *baked* heightfield rather than `pondVesselHeightAt`, because the
 * bake is what the renderer draws and the solver collides against. The two agree
 * to well under a millimetre today; taking the bake means they cannot stop
 * agreeing the day it is sampled differently.
 */
export function heroGardenLayoutWorld(scene: Pick<SceneDescription, "terrain">): LayoutWorld {
  const terrain = scene.terrain;
  return {
    groundHeightAt: terrain
      ? (x, z) => terrainHeightAt(terrain, x, z)
      : () => HERO_GARDEN_VESSEL.groundHeight_m,
    level_m: HERO_GARDEN_WATERLINE_M,
  };
}

/**
 * Signed distance from a world point to the coping's crest centreline, negative
 * inside the pond. `+rimHalfWidth_m` is the outer foot, `-rimHalfWidth_m` the
 * inner one, which is where the water starts.
 */
export function heroGardenPlanDistance_m(x: number, z: number): number {
  return pondVesselPlanDistance(pondVesselPlanCurve(HERO_GARDEN_VESSEL), x, z);
}

/**
 * The still water, as a solid: inside the coping's inner foot and below the
 * level. Anything that intersects it and is not a stepping disc is a mistake the
 * composition should not be allowed to make silently.
 */
export function heroGardenInWater(x: number, y: number, z: number): boolean {
  return y < HERO_GARDEN_WATERLINE_M
    && heroGardenPlanDistance_m(x, z) < -HERO_GARDEN_VESSEL.rimHalfWidth_m;
}

/**
 * A scenery graph carrying the hero set, leaving the base untouched.
 *
 * Idempotent by construction: any node already under the layout prefix is dropped
 * before the new ones go on. The scene library composes the blocking set into the
 * hero document, so a preview that wants the budget set is asking to *replace* a
 * layout rather than to add a second one — and two sets of nodes sharing one id
 * space is a duplicate-key error at best and a doubled primitive count at worst.
 */
export function withHeroLayout(scene: SceneDescription, options: LayoutOptions = {}): SceneryGraph {
  const base = scene.scenery;
  if (!base) throw new Error("The hero layout extends a scenery graph the scene must already carry");
  /**
   * The layout is solved against one level and the document has to be at it.
   *
   * `HERO_STONE_SET` and every station read off it are module constants at
   * `HERO_GARDEN_WATERLINE_M`, and the level moves with the *solver's* lattice —
   * `heroGardenWaterBelowGround_m` is a floor in cells, so a wet document at
   * 12.5 mm sits 20 mm higher than this. Composing this layout onto that
   * document would bed stones against a shore the pond does not have, and it
   * would do it silently. Loud instead: nothing reaches this today, because the
   * only caller is a dry preset, and a caller that does should pass the level in
   * rather than have the answer guessed for it.
   */
  const documentWaterline_m = scene.container.fillFraction * scene.container.height_m;
  if (Math.abs(documentWaterline_m - HERO_GARDEN_WATERLINE_M) > 1e-9) {
    throw new Error(`The hero layout is solved at a ${(HERO_GARDEN_WATERLINE_M * 1e3).toFixed(1)} mm waterline`
      + ` and this document sits at ${(documentWaterline_m * 1e3).toFixed(1)} mm;`
      + " give the layout its own level before composing it onto a scene at another lattice");
  }
  const kept = base.nodes.filter((node) => !node.id.startsWith(HERO_LAYOUT_NODE_PREFIX));
  const nodes = layoutPlaceholderNodes(heroGardenLayout(), heroGardenLayoutWorld(scene), options);
  // Spread rather than name the fields: everything on the graph except the node
  // list belongs to the base and has to survive. Naming them cost the vessels
  // table the day generators started resolving their runs through it, and the
  // failure was a hero document whose coping no longer knew which pond it rings.
  return { ...base, nodes: [...kept, ...nodes] };
}

/** The container the hero layout must stay inside, as the AABB the scene document implies. */
export const HERO_LAYOUT_CONTAINER_BOUNDS: { readonly min: Vec3; readonly max: Vec3 } = Object.freeze({
  min: Object.freeze(V3(-.5 * HERO_GARDEN_CONTAINER.width_m, 0, -.5 * HERO_GARDEN_CONTAINER.depth_m)),
  max: Object.freeze(V3(.5 * HERO_GARDEN_CONTAINER.width_m, HERO_GARDEN_CONTAINER.height_m, .5 * HERO_GARDEN_CONTAINER.depth_m)),
});
