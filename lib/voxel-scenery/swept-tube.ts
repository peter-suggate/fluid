import type { Quaternion, Vec3 } from "../model";
import type { SceneryMaterial, SceneryNode, ScenerySweepPoint } from "../scenery-graph";
import { SVO_CLUSTER_SWEEP_MAXIMUM_POINTS } from "../svo-primitive-abi";
import { V } from "./builder";

/**
 * A tube: one continuous round-section run following a path through the scene.
 *
 * One species — a swept circle with a radius that may vary along it — and
 * therefore not only a garden hose. The same generator is a cable, a drain, a
 * handrail, a root, a vine or a length of rope; it takes stations in world
 * metres and knows nothing about what drew them.
 *
 * ## Why this is not a chain of capsules
 *
 * A capsule is already a swept circle, so a run *can* be spelled as one capsule
 * per segment, and that is what this scene did before the render ABI grew a
 * sweep field. Three things are wrong with it, and all three are consequences of
 * the representation rather than of the tuning:
 *
 *  - **A stepped normal at every joint.** Two capsules meeting at a bend are a
 *    hard union: the surface is continuous but its normal is not, so a smooth
 *    curve reads as a run of straight sausages with a crease between each. The
 *    sweep field fuses its segments with a smooth minimum, which is C¹ through
 *    the junction, so a bend is a bend.
 *  - **A seam wherever two of them cross a voxel boundary.** Each capsule is its
 *    own primitive with its own candidate entry, and the two halves of a joint
 *    can land in different bricks. The whole run is one record here, so there is
 *    nothing to disagree.
 *  - **A record each.** The scene's primitive budget is 4 096 and its per-brick
 *    candidate ceiling is 64 with a *silent* overflow drop; a run detailed
 *    enough to read as a curve spends that budget fastest exactly where it is
 *    most crowded, which is at the bends.
 *
 * ## The envelope, and why it is solved rather than authored
 *
 * A cluster record is clipped to an ellipsoid by a hard maximum, and that
 * ellipsoid is what the voxelizer, the bounds formulae and the node-mip oracle
 * read. For a long thin run it is a *terrible* fit — an ellipsoid containing a
 * metre of 13 mm hose is a large volume of nothing that still marks its bricks
 * occupied and still occludes indirect light.
 *
 * Worse, containment is checked with the envelope's own Lipschitz-1 lower bound
 * (`(1 − ‖p/R‖)·min R`, which is what the clipping `max` actually computes), and
 * that bound goes to zero at the envelope surface far faster than the true
 * distance does. A station near the end of the long axis therefore needs the
 * envelope to reach well past it: at a 40 mm cross-axis, 13 mm of tube plus its
 * blend wants the long axis around 1.7× the run's own half-length.
 *
 * So the envelope is not a parameter here. Each run is fitted with an oriented
 * frame, and its half-axes are chosen to *minimise volume* subject to every
 * station clearing that same bound — a one-dimensional search over the
 * cross-axis, because the long axis follows from it in closed form. A run whose
 * best envelope is still too loose for what it contains is split in two and each
 * half fitted again, which trades a record for a much tighter pair of ellipsoids.
 */

/** One station of a tube's path, in world metres. */
export interface SweptTubeStation {
  readonly at_m: Vec3;
  readonly radius_m: number;
}

export interface SweptTubeSpec {
  /** Prefix for the emitted node ids. A single-record tube is `${id}`, a split one `${id}/run-0`. */
  readonly id: string;
  readonly group?: string;
  readonly tags?: readonly string[];
  readonly material: SceneryMaterial;
  /** The path, in world metres, from one end to the other. At least two stations. */
  readonly stations: readonly SweptTubeStation[];
  /**
   * Radius of the smooth minimum that fuses the run's segments. Absent takes
   * half the thinnest station's radius, which rounds a bend without measurably
   * fattening the tube: a smooth minimum inflates the union by at most a quarter
   * of its own radius, so this is an eighth of the bore at the tightest point.
   */
  readonly smoothRadius_m?: number;
  readonly seed?: number;
  /**
   * How much empty envelope a record may carry, as a multiple of the tube volume
   * inside it. Exceeding it splits the run. Absent takes {@link TUBE_ENVELOPE_WASTE_BUDGET}.
   */
  readonly envelopeWasteBudget?: number;
}

/**
 * The default ceiling on envelope volume, as a multiple of the tube inside it.
 *
 * This is a straight trade of records against occupied volume and there is no
 * natural optimum, so the number is chosen at the knee. Measured on the hero
 * garden's hose — 0.75 m of 13 mm bore over six stations, 0.48 L of actual tube:
 * a budget of 40 keeps the whole path in one record and pays 19.1 L of envelope
 * for it, 20 splits it into three for 5.6 L, and 12 into four for 5.3 L. Below
 * that the count multiplies without buying anything — eight lands *fourteen*
 * records for 5.0 L — because a short run's envelope is floored by the
 * containment bound rather than by its own length. Twelve is the last split that
 * pays for itself.
 */
export const TUBE_ENVELOPE_WASTE_BUDGET = 12;

/**
 * The safety factor on every containment margin.
 *
 * The render ABI checks the same bound against the packing's f32 round trip, and
 * a margin solved *at* the limit comes back a few ulps under it. This is far
 * below anything that could change the fit and it keeps the solve from handing
 * the validator a record it computed as legal.
 */
const CONTAINMENT_SLACK = 1.02;

/** The cross-axis search, as a range of multiples of the widest station's margin. */
const CROSS_AXIS_SEARCH_FLOOR = 1.02;
const CROSS_AXIS_SEARCH_CEILING = 60;
const CROSS_AXIS_SEARCH_STEPS = 240;

/** A run is only ever split in half, so this bounds the record count at 2^n. */
const MAXIMUM_SPLIT_DEPTH = 4;

const sub = (a: Vec3, b: Vec3): Vec3 => V(a.x - b.x, a.y - b.y, a.z - b.z);
const dot = (a: Vec3, b: Vec3): number => a.x * b.x + a.y * b.y + a.z * b.z;
const cross = (a: Vec3, b: Vec3): Vec3 =>
  V(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
const scale = (a: Vec3, k: number): Vec3 => V(a.x * k, a.y * k, a.z * k);
const length = (a: Vec3): number => Math.hypot(a.x, a.y, a.z);

function normalize(a: Vec3, fallback: Vec3): Vec3 {
  const l = length(a);
  return l > 1e-9 ? scale(a, 1 / l) : fallback;
}

/** Any unit vector perpendicular to `u`, chosen off whichever axis `u` leans on least. */
function perpendicular(u: Vec3): Vec3 {
  const away = Math.abs(u.x) < 0.9 ? V(1, 0, 0) : V(0, 1, 0);
  return normalize(cross(u, away), V(0, 1, 0));
}

/** The quaternion taking local +X, +Y, +Z onto `u`, `v`, `w`. */
function basisOrientation(u: Vec3, v: Vec3, w: Vec3): Quaternion {
  const trace = u.x + v.y + w.z;
  if (trace > 0) {
    const s = Math.sqrt(trace + 1) * 2;
    return { w: 0.25 * s, x: (v.z - w.y) / s, y: (w.x - u.z) / s, z: (u.y - v.x) / s };
  }
  if (u.x > v.y && u.x > w.z) {
    const s = Math.sqrt(1 + u.x - v.y - w.z) * 2;
    return { w: (v.z - w.y) / s, x: 0.25 * s, y: (v.x + u.y) / s, z: (w.x + u.z) / s };
  }
  if (v.y > w.z) {
    const s = Math.sqrt(1 + v.y - u.x - w.z) * 2;
    return { w: (w.x - u.z) / s, x: (v.x + u.y) / s, y: 0.25 * s, z: (w.y + v.z) / s };
  }
  const s = Math.sqrt(1 + w.z - u.x - v.y) * 2;
  return { w: (u.y - v.x) / s, x: (w.x + u.z) / s, y: (w.y + v.z) / s, z: 0.25 * s };
}

interface TubeRunFit {
  readonly origin_m: Vec3;
  readonly orientation: Quaternion;
  readonly lobe_m: Vec3;
  readonly points: readonly ScenerySweepPoint[];
  readonly envelopeVolume_m3: number;
  readonly tubeVolume_m3: number;
}

/** Round-cone volume, summed over the run's segments. Two ends and a frustum between. */
function tubeVolume_m3(run: readonly SweptTubeStation[]): number {
  let volume = 0;
  for (let index = 0; index + 1 < run.length; index += 1) {
    const from = run[index];
    const to = run[index + 1];
    const span = length(sub(to.at_m, from.at_m));
    volume += (Math.PI / 3) * span * (from.radius_m ** 2 + from.radius_m * to.radius_m + to.radius_m ** 2);
  }
  const cap = (2 / 3) * Math.PI;
  return volume + cap * (run[0].radius_m ** 3 + run[run.length - 1].radius_m ** 3);
}

/**
 * Fit one run: an oriented frame from its chord, then the smallest envelope in
 * that frame every station provably clears.
 *
 * The cross-axis is searched rather than solved because the constraint couples
 * the two axes through `min R`: fixing the cross-axis at `ρ` makes each station's
 * budget `k = 1 − m/ρ` a constant, and the long axis it needs then falls out as
 * `|a| / √(k² − (lat/ρ)²)`. Volume is `Rx·ρ²` and has one interior minimum in
 * `ρ` — too thin and the long axis blows up to buy the margin back, too fat and
 * the cross-section costs more than the length saves.
 */
function fitRun(run: readonly SweptTubeStation[], smoothRadius_m: number): TubeRunFit {
  const chord = sub(run[run.length - 1].at_m, run[0].at_m);
  const u = normalize(chord, V(1, 0, 0));
  // The lateral basis is aimed at the run's own largest excursion, so a path
  // that bends in one plane spends its second axis on that bend rather than on
  // an arbitrary direction it never leaves.
  let widestLateral = V(0, 0, 0);
  let widestLateral_m = 0;
  for (const station of run) {
    const offset = sub(station.at_m, run[0].at_m);
    const lateral = sub(offset, scale(u, dot(offset, u)));
    if (length(lateral) > widestLateral_m) {
      widestLateral_m = length(lateral);
      widestLateral = lateral;
    }
  }
  const v = widestLateral_m > 1e-6 ? normalize(widestLateral, perpendicular(u)) : perpendicular(u);
  const w = cross(u, v);

  const local = run.map((station) => {
    const offset = sub(station.at_m, run[0].at_m);
    return { a: dot(offset, u), b: dot(offset, v), c: dot(offset, w), margin_m: (station.radius_m + smoothRadius_m) * CONTAINMENT_SLACK };
  });
  // Centre on the run's own extent rather than on its chord, so a path that
  // wanders to one side does not spend half its envelope on the side it never
  // reaches.
  const spanOf = (pick: (p: typeof local[number]) => number): number =>
    0.5 * (Math.min(...local.map(pick)) + Math.max(...local.map(pick)));
  const centreLocal = { a: spanOf((p) => p.a), b: spanOf((p) => p.b), c: spanOf((p) => p.c) };
  const centred = local.map((p) => ({
    a: p.a - centreLocal.a,
    b: p.b - centreLocal.b,
    c: p.c - centreLocal.c,
    margin_m: p.margin_m,
  }));
  const origin_m = V(
    run[0].at_m.x + u.x * centreLocal.a + v.x * centreLocal.b + w.x * centreLocal.c,
    run[0].at_m.y + u.y * centreLocal.a + v.y * centreLocal.b + w.y * centreLocal.c,
    run[0].at_m.z + u.z * centreLocal.a + v.z * centreLocal.b + w.z * centreLocal.c,
  );

  const widestMargin_m = Math.max(...centred.map((p) => p.margin_m));
  let best: { long_m: number; cross_m: number; volume_m3: number } | null = null;
  for (let step = 0; step <= CROSS_AXIS_SEARCH_STEPS; step += 1) {
    const t = step / CROSS_AXIS_SEARCH_STEPS;
    const cross_m = widestMargin_m
      * CROSS_AXIS_SEARCH_FLOOR * (CROSS_AXIS_SEARCH_CEILING / CROSS_AXIS_SEARCH_FLOOR) ** t;
    let long_m = cross_m;
    let feasible = true;
    for (const point of centred) {
      const budget = 1 - point.margin_m / cross_m;
      const lateral = Math.hypot(point.b, point.c) / cross_m;
      const headroom = budget * budget - lateral * lateral;
      if (budget <= 0 || headroom <= 0) {
        feasible = false;
        break;
      }
      long_m = Math.max(long_m, Math.abs(point.a) / Math.sqrt(headroom));
    }
    if (!feasible) continue;
    const volume_m3 = (4 / 3) * Math.PI * long_m * cross_m * cross_m;
    if (best === null || volume_m3 < best.volume_m3) best = { long_m, cross_m, volume_m3 };
  }
  if (best === null) {
    // Only reachable if a station's own radius exceeds the whole search range,
    // which means the caller asked for a tube fatter than it is long.
    throw new RangeError("Swept tube run has no envelope that contains it; check the station radii against the path");
  }

  const lobe_m = V(best.long_m, best.cross_m, best.cross_m);
  return {
    origin_m,
    orientation: basisOrientation(u, v, w),
    lobe_m,
    points: centred.map((point, index) => ({
      position: V(point.a, point.b, point.c),
      radius: run[index].radius_m,
    })),
    envelopeVolume_m3: best.volume_m3,
    tubeVolume_m3: tubeVolume_m3(run),
  };
}

/**
 * Where to cut a run that is too loose for one envelope.
 *
 * The straightest interior station, because a cut is the one junction the smooth
 * minimum does not cover: two records meet in a hard union, and that union is
 * invisible exactly where the two tangents already agree. Ties go to the middle,
 * so a straight run halves.
 */
function splitIndex(run: readonly SweptTubeStation[]): number {
  let bestIndex = Math.floor(run.length / 2);
  let bestScore = Infinity;
  for (let index = 1; index + 1 < run.length; index += 1) {
    const incoming = normalize(sub(run[index].at_m, run[index - 1].at_m), V(1, 0, 0));
    const outgoing = normalize(sub(run[index + 1].at_m, run[index].at_m), V(1, 0, 0));
    const turn = 1 - dot(incoming, outgoing);
    const offCentre = Math.abs(index - (run.length - 1) / 2) / run.length;
    const score = turn + 0.05 * offCentre;
    if (score < bestScore) {
      bestScore = score;
      bestIndex = index;
    }
  }
  return bestIndex;
}

/** Split at `index`, keeping that station in *both* halves so the tube stays continuous. */
function splitRun(
  run: readonly SweptTubeStation[],
  index: number,
): readonly [readonly SweptTubeStation[], readonly SweptTubeStation[]] {
  return [run.slice(0, index + 1), run.slice(index)];
}

/** A two-station run has no interior to cut, so it gains a midpoint. Exact: the taper is linear. */
function bisectSegment(run: readonly SweptTubeStation[]): readonly SweptTubeStation[] {
  const [from, to] = run;
  return [
    from,
    {
      at_m: V(0.5 * (from.at_m.x + to.at_m.x), 0.5 * (from.at_m.y + to.at_m.y), 0.5 * (from.at_m.z + to.at_m.z)),
      radius_m: 0.5 * (from.radius_m + to.radius_m),
    },
    to,
  ];
}

function fitRuns(
  run: readonly SweptTubeStation[],
  smoothRadius_m: number,
  wasteBudget: number,
  depth: number,
): TubeRunFit[] {
  if (run.length > SVO_CLUSTER_SWEEP_MAXIMUM_POINTS) {
    const [head, tail] = splitRun(run, splitIndex(run));
    return [
      ...fitRuns(head, smoothRadius_m, wasteBudget, depth),
      ...fitRuns(tail, smoothRadius_m, wasteBudget, depth),
    ];
  }
  const fit = fitRun(run, smoothRadius_m);
  if (fit.envelopeVolume_m3 <= wasteBudget * fit.tubeVolume_m3 || depth >= MAXIMUM_SPLIT_DEPTH) return [fit];
  const cuttable = run.length > 2 ? run : bisectSegment(run);
  const [head, tail] = splitRun(cuttable, splitIndex(cuttable));
  return [
    ...fitRuns(head, smoothRadius_m, wasteBudget, depth + 1),
    ...fitRuns(tail, smoothRadius_m, wasteBudget, depth + 1),
  ];
}

/**
 * The tube, as the fewest sweep records that carry it without wasting volume.
 *
 * Each record's placement is the fitted frame's origin and orientation, and its
 * control points are offsets inside that frame, which is the convention
 * `clusterPacking` expects.
 *
 * Stations are in whatever frame the caller's nodes are placed in, not
 * necessarily the world's. Nothing in the fit is world-dependent — it is
 * similarity-equivariant, and the `units: "metres"` placement it emits is
 * composed against the enclosing group by `childFrame` like any other node's.
 * So a specimen generator may pass stations in its own local frame and get a
 * tube in the right place; the hero hose passes world metres only because it is
 * authored at scene root. Do not fork this to add a frame argument.
 */
export function sweptTubeNodes(spec: SweptTubeSpec): SceneryNode[] {
  if (spec.stations.length < 2) {
    throw new RangeError(`Swept tube "${spec.id}" needs at least two stations`);
  }
  const thinnest_m = Math.min(...spec.stations.map((station) => station.radius_m));
  const smoothRadius_m = spec.smoothRadius_m ?? 0.5 * thinnest_m;
  const fits = fitRuns(
    spec.stations,
    smoothRadius_m,
    spec.envelopeWasteBudget ?? TUBE_ENVELOPE_WASTE_BUDGET,
    0,
  );
  return fits.map((fit, index) => ({
    kind: "cluster",
    field: "tapered-sweep",
    id: fits.length === 1 ? spec.id : `${spec.id}/run-${index}`,
    group: spec.group ?? spec.id,
    tags: spec.tags,
    place: { units: "metres", position: fit.origin_m, orientation: fit.orientation },
    lobe: fit.lobe_m,
    smoothRadius: smoothRadius_m,
    // One seed for the whole tube: the sweep field reads it for nothing today,
    // and a per-record seed would be a difference between halves of one object.
    seed: spec.seed ?? 1,
    points: fit.points,
    material: spec.material,
  }));
}

/** What the fit chose, for a generator that wants to report or assert on it. */
export function sweptTubeFitReport(spec: SweptTubeSpec): readonly {
  readonly lobe_m: Vec3;
  readonly envelopeVolume_m3: number;
  readonly tubeVolume_m3: number;
}[] {
  const thinnest_m = Math.min(...spec.stations.map((station) => station.radius_m));
  return fitRuns(
    spec.stations,
    spec.smoothRadius_m ?? 0.5 * thinnest_m,
    spec.envelopeWasteBudget ?? TUBE_ENVELOPE_WASTE_BUDGET,
    0,
  ).map((fit) => ({ lobe_m: fit.lobe_m, envelopeVolume_m3: fit.envelopeVolume_m3, tubeVolume_m3: fit.tubeVolume_m3 }));
}
