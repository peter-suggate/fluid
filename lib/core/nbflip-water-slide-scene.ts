import { bakeProceduralSolid } from "./procedural-solid";
import type { CameraState, SceneDescription } from "./model";
import { cellPoint, nbflipDomain } from "./nbflip-paper-scenes";
import type { SolidWorldVoxelPatch } from "./solid-world";
import { VOXEL_MATERIAL_IDS } from "./voxel-scene";

/**
 * A procedural water slide: a scene whose liquid starts as nothing, grows along a
 * long channel and ends as a pool, so the space the solver has to hold grows
 * with it.
 *
 * The slide is a channel cut into the top of a solid drum. It winds inward for
 * two turns with a continuous floor, and ends under a hood that spills
 * sideways into a pit at the drum's centre. Three things follow from the
 * liquid being inviscid and the walls free-slip, so nothing but a wall ever
 * slows the stream:
 *
 *  - **It turns one way and tightens.** A stream of speed v leans on the
 *    outside of a bend of radius R with its surface tilted v^2 / (g R). The
 *    wall between two turns is the inner wall of the higher one and the outer
 *    wall of the lower, a turn's fall taller on the low side, which is the
 *    side the stream climbs.
 *  - **Whatever jumps a wall lands on the slide.** Every wall but the drum's
 *    rim has a lower turn or the pit inside it.
 *  - **The hood is what lets the pool come to rest.** Run straight into a
 *    round pit, the stream would turn it into a whirlpool that nothing stops.
 *    The channel instead ends at a wall, under a lid, and leaves across the
 *    stream through a gap in its inner wall.
 *
 * Lengths below are in cells unless they say otherwise.
 */

export const NBFLIP_SLIDE_SCENE_ID = "nbflip-water-slide";
/** Five pages by three by five of 32 cells; the drum stands five cells clear of every wall and sixteen of the ceiling. */
export const NBFLIP_SLIDE_GRID = [160, 96, 160] as const;
export const NBFLIP_SLIDE_CELL_SIZE_M = 0.02;
export const NBFLIP_SLIDE_STEP_HZ = 60;
/** Twenty seconds: eight of stream, four for the slide to drain, eight for the pool to settle. */
export const NBFLIP_SLIDE_FRAMES = 1200;

/** The drum's axis, off the lattice's centre because a spiral's outline is. */
const AXIS = [76, 84.5] as const;
export const NBFLIP_SLIDE_TURNS = 2;
/** The centreline's radius where the channel starts, and how much of it a turn takes: the channel's width and one wall. */
const START_RADIUS = 68, PITCH = 18;
/** The channel is 14 wide between walls 4 thick. */
const HALF_WIDTH = 7;
/** The floor's height where the channel starts, and its fall to the hood. */
const TOP = 64, FALL = 32;
/** A wall's height over the floor of the turn outside it, and the drum's rim over the first turn's. */
const WALL = 8, RIM = 16;
/** Along the centreline: the wall behind the source, the hood, the gap in its inner wall, and the wall that ends it. */
const BACK = 6, HOOD = 30, GAP = 18, STOP = 5;
/** The hood's clear height and its lid's thickness. */
const HOOD_CLEARANCE = 14, HOOD_LID = 4;
const PIT_FLOOR = 4;

/** The stream's radius, its speed in cells per step, how far past the back wall it leaves, and how long it runs. */
export const NBFLIP_SLIDE_STREAM_RADIUS = 5;
export const NBFLIP_SLIDE_STREAM_CELLS_PER_STEP = 1.6;
const STREAM_AHEAD = 8;
export const NBFLIP_SLIDE_STREAM_SECONDS = 8;

const TURN = 2 * Math.PI;
/** The centreline angle the channel ends at; it starts at zero, pointing away from the camera. */
export const NBFLIP_SLIDE_END_RAD = NBFLIP_SLIDE_TURNS * TURN;
const radius = (angle: number) => START_RADIUS - PITCH * angle / TURN;
const END_RADIUS = radius(NBFLIP_SLIDE_END_RAD);
/** The floor under the centreline: whole steps down to the hood, flat under it. */
const floor = (angle: number) =>
  TOP - Math.floor(FALL * Math.min(1, Math.max(0, angle) / (NBFLIP_SLIDE_END_RAD - HOOD / END_RADIUS)));
const HOOD_FLOOR = TOP - FALL;

/** The centreline at an angle, as lattice x and z. */
export function nbflipSlideCentreline(angle: number): readonly [x: number, z: number] {
  return [AXIS[0] + radius(angle) * Math.sin(angle), AXIS[1] - radius(angle) * Math.cos(angle)];
}

/**
 * The drum under one column of cells: the height of its solid, and whether the
 * hood's lid hangs over it.
 *
 * A column is either in a turn's channel or in the wall between two, and a
 * wall is reckoned with the turn outside it, whose floor it stands on.
 */
function slideColumn(x: number, z: number): readonly [top: number, lidded: boolean] {
  const dx = x + 0.5 - AXIS[0], dz = z + 0.5 - AXIS[1];
  const bearing = (Math.atan2(dx, -dz) + TURN) % TURN;
  const turns = (START_RADIUS - Math.hypot(dx, dz)) / PITCH - bearing / TURN, turn = Math.round(turns);
  // How far outside its turn's centreline the column is.
  const outside = PITCH * (turn - turns), wall = Math.abs(outside) >= HALF_WIDTH;
  const angle = bearing + TURN * (wall && outside > 0 ? turn - 1 : turn), end = NBFLIP_SLIDE_END_RAD;
  // Past the channel's end: the wall that stops the stream, then the pit.
  if (angle > end + STOP / END_RADIUS) return [PIT_FLOOR, false];
  if (angle > end) return [HOOD_FLOOR + HOOD_CLEARANCE + HOOD_LID, false];
  if (angle >= 0) {
    return [floor(angle) + (wall && angle < end - GAP / END_RADIUS ? WALL : 0), angle >= end - HOOD / END_RADIUS];
  }
  // Before its start: the wall behind the source, and outside that only the rim around the first turn.
  if (angle >= -BACK / START_RADIUS) return [TOP + RIM, false];
  return [wall && angle + TURN >= -BACK / START_RADIUS ? floor(angle + TURN) + RIM : 0, false];
}

/**
 * Columns of solid, each from the height `span(x, z)` starts at up to the one
 * it ends at, as boxes: one for each run of equal columns along x, carried
 * along z for as long as the next row repeats it.
 */
export function columnVoxels([nx, , nz]: readonly [number, number, number], span: (x: number, z: number) => readonly [from: number, top: number]): SolidWorldVoxelPatch[] {
  type Box = { minimum: [number, number, number]; maximumExclusive: [number, number, number] };
  const boxes: Box[] = [];
  let open = new Map<string, Box>();
  for (let z = 0; z < nz; z++) {
    const row = new Map<string, Box>();
    for (let x = 0, end = 0; x < nx; x = end) {
      const [from, top] = span(x, z);
      for (end = x + 1; end < nx; end++) {
        const [nextFrom, nextTop] = span(end, z);
        if (nextFrom !== from || nextTop !== top) break;
      }
      if (top <= from) continue;
      const key = `${x}:${end}:${from}:${top}`;
      let box = open.get(key);
      if (box) box.maximumExclusive[2] = z + 1;
      else boxes.push(box = { minimum: [x, from, z], maximumExclusive: [end, top, z + 1] });
      row.set(key, box);
    }
    open = row;
  }
  return boxes.map(box => ({ operation: "fill", ...box, materialId: VOXEL_MATERIAL_IDS.container }));
}

/**
 * A stream of `radius` lying on a floor at height `floor`, leaving the lattice
 * point (x, z) along (u, w) cells a step for `seconds`.
 */
export function streamInflow(
  grid: readonly [number, number, number], h: number, dt: number,
  [x, z]: readonly [number, number], floor: number, [u, w]: readonly [number, number], radius: number, seconds: number,
): NonNullable<SceneDescription["fluid"]["inflow"]> {
  return {
    center_m: cellPoint(grid, h, [x - u / 2, floor + radius + 0.5, z - w / 2]),
    radius_m: radius * h,
    length_m: Math.hypot(u, w) * h,
    velocity_m_s: { x: u * h / dt, y: 0, z: w * h / dt },
    start_s: 0,
    end_s: seconds,
    ramp_s: 0,
  };
}

/** Continuous implicit solid in lattice coordinates. The floor, walls, hood
 * and pit are composed before sampling, so all consumers share their edges.
 * Distance-like CSG values are negative inside; no height quantization occurs. */
export function nbflipSlideSolid(p: readonly [number, number, number]): number {
  const [x,y,z]=p, dx=x-AXIS[0], dz=z-AXIS[1], r=Math.hypot(dx,dz);
  const bearing=(Math.atan2(dx,-dz)+TURN)%TURN, end=NBFLIP_SLIDE_END_RAD;
  let solid=Math.max(r-(END_RADIUS+PITCH/2-PITCH*bearing/TURN),y-PIT_FLOOR);
  for(let turn=-1;turn<=NBFLIP_SLIDE_TURNS;turn++){
    const a=bearing+TURN*turn, rad=radius(a), across=r-rad;
    const alongLow=(-BACK/START_RADIUS-a)*Math.max(1,rad), alongHigh=(a-end-STOP/END_RADIUS)*Math.max(1,rad);
    const span=Math.max(alongLow,alongHigh,Math.abs(across)-(HALF_WIDTH+WALL/2));
    const height=TOP-FALL*Math.min(1,Math.max(0,a)/(end-HOOD/END_RADIUS));
    solid=Math.min(solid,Math.max(span,y-height));
    // Walls are separate volumes: their vertical faces and the hood's
    // underside survive in the field instead of being interpreted as slopes.
    const wallBand=Math.max(span,HALF_WIDTH-Math.abs(across));
    const gap=across<0?Math.max((end-GAP/END_RADIUS-a)*END_RADIUS,(a-end)*END_RADIUS):1e6;
    const wallHeight=height+(turn===0&&across>0?RIM:WALL);
    solid=Math.min(solid,Math.max(wallBand,-gap,y-wallHeight));
    const back=Math.max(span,a*START_RADIUS,y-(TOP+RIM));
    const stop=Math.max(span,(end-a)*END_RADIUS,y-(HOOD_FLOOR+HOOD_CLEARANCE+HOOD_LID));
    solid=Math.min(solid,back,stop);
    const lidLow=HOOD_FLOOR+HOOD_CLEARANCE;
    const hood=Math.max(span,(end-HOOD/END_RADIUS-a)*END_RADIUS,(a-end)*END_RADIUS,lidLow-y,y-lidLow-HOOD_LID);
    solid=Math.min(solid,hood);
  }
  return Math.max(-y,solid);
}
let slideGeometry: ReturnType<typeof bakeProceduralSolid> | undefined;

export function createNbflipWaterSlide(): SceneDescription {
  const grid = NBFLIP_SLIDE_GRID, h = NBFLIP_SLIDE_CELL_SIZE_M, dt = 1 / NBFLIP_SLIDE_STEP_HZ;
  const scene = nbflipDomain(NBFLIP_SLIDE_SCENE_ID, grid, h, NBFLIP_SLIDE_FRAMES, 0, grid, dt);
  const geometry=slideGeometry??=bakeProceduralSolid(grid,[h,h,h],nbflipSlideSolid,VOXEL_MATERIAL_IDS.container);
  scene.solidVoxels=[...geometry.solidVoxels];
  scene.solidBoundary=geometry.solidBoundary;
  // The stream lies on the floor just past the back wall and leaves along the channel.
  const angle = STREAM_AHEAD / START_RADIUS, speed = NBFLIP_SLIDE_STREAM_CELLS_PER_STEP;
  scene.fluid.inflow = streamInflow(grid, h, dt, nbflipSlideCentreline(angle), TOP,
    [speed * Math.cos(angle), speed * Math.sin(angle)], NBFLIP_SLIDE_STREAM_RADIUS, NBFLIP_SLIDE_STREAM_SECONDS);
  // Storage starts at the stream's pages and follows the water. The water reaches every one of the drum's 5x3x5 = 75
  // pages, so the pool holds them all: 80 blocks of 4.90 MiB (two are the AIR and SOLID sentinels).
  scene.fluid.pagePool = { blocks: 80 };
  return scene;
}

/** Original whole-voxel staircase retained as a control for the inferred
 * bed-gradient bake. The registered scene uses continuous geometry above. */
export function createNbflipWaterSlideVoxelReference(): SceneDescription {
  const scene=createNbflipWaterSlide();
  delete scene.solidBoundary;
  const lid=HOOD_FLOOR+HOOD_CLEARANCE;
  scene.solidVoxels=[...columnVoxels(NBFLIP_SLIDE_GRID,(x,z)=>[0,slideColumn(x,z)[0]]),
    ...columnVoxels(NBFLIP_SLIDE_GRID,(x,z)=>[lid,slideColumn(x,z)[1]?lid+HOOD_LID:lid])];
  return scene;
}

/** From above the near rim, high enough to see every turn's floor past the wall in front of it. */
export function nbflipWaterSlideCamera(): Partial<CameraState> {
  const { x, z } = cellPoint(NBFLIP_SLIDE_GRID, NBFLIP_SLIDE_CELL_SIZE_M, [AXIS[0], 0, AXIS[1]]);
  return { azimuth_rad: 0.45, elevation_rad: 1.1, distance_m: 4.8, target_m: { x, y: 0.9, z } };
}

/**
 * The long slide: the same channel laid out as a serpentine down a box sixteen
 * pages long, so that almost all of the box is air the liquid never visits.
 *
 * It is a flume on a slab, hung in the air: a straight from the source, six
 * half-turn bends that alternate, a straight between each pair, and then the
 * drum's own ending -- one more turn, wound inward around a pit and stopped
 * under a hood that spills into it. That last turn stands on solid, as the
 * drum does, because its inner wall is the pool's.
 *
 * Nothing catches what leaves this channel, so its walls are sized to keep
 * it. A stream of speed v and flow Q, leaning on the outside of a bend of
 * radius R, climbs sqrt(2 v Q / (g R)): 12 cells at the 4 cells a step the
 * drum's stream reached. A stream thrown across the channel by a bend that
 * turns the other way overshoots that, so the wall it lands on is twice as
 * high. That is every bend's outer wall, from the straight before the bend
 * until the next bend has taken the stream off it, a quarter turn in; a wall
 * is half that height only along the inside of a bend's second half.
 */

export const NBFLIP_SLIDE_LONG_SCENE_ID = "nbflip-water-slide-long";
/** Sixteen pages by three by four of 32 cells: the cell count of the paper's Letters. */
export const NBFLIP_SLIDE_LONG_GRID = [512, 96, 128] as const;
/** Thirty seconds: thirteen of stream, nine for the slide to drain, eight for the pool to settle. */
export const NBFLIP_SLIDE_LONG_FRAMES = 1800;

/** Where the channel starts, as lattice x and z; it sets off toward +z. */
const LONG_START = [16, 48] as const;
/** Half-turn bends of this radius, alternating, with a straight of this length before each. */
export const NBFLIP_SLIDE_LONG_BENDS = 6;
const LONG_BEND_RADIUS = 32, LONG_LEG = 32;
/** The last turn's two halves: half a `PITCH` apart, so its end runs beside its start. */
const LONG_BOWL_RADII = [48, 39] as const;
/**
 * The floor's height where the channel starts, and its fall to the hood. The
 * hood's floor is the pool's rim, at a page boundary: the pool lies in the
 * bottom layer of 32-cell pages and the channel's floor in the middle one all
 * the way, the stream's own height under the top layer where it starts.
 */
const LONG_TOP = 56, LONG_FALL = 24;
/** A wall's height over the floor where a stream can land on it, and elsewhere. */
const LONG_HIGH = 24, LONG_LOW = 12;
/** The flume's thickness under its floor. */
const LONG_SLAB = 8;
export const NBFLIP_SLIDE_LONG_STREAM_RADIUS = 4;
export const NBFLIP_SLIDE_LONG_STREAM_SECONDS = 13;

/** A straight, or an arc of constant `curvature`, from a point and heading at a distance along the course. */
interface CoursePiece { readonly from: number; readonly length: number; readonly x: number; readonly z: number; readonly heading: number; readonly curvature: number }

/** Where a piece leads after `distance`: lattice x, z and heading. Positive curvature turns from +x toward +z. */
function follow({ x, z, heading, curvature }: Pick<CoursePiece, "x" | "z" | "heading" | "curvature">, distance: number): readonly [x: number, z: number, heading: number] {
  if (curvature === 0) return [x + distance * Math.cos(heading), z + distance * Math.sin(heading), heading];
  const turned = heading + curvature * distance;
  return [x + (Math.sin(turned) - Math.sin(heading)) / curvature, z - (Math.cos(turned) - Math.cos(heading)) / curvature, turned];
}

const LONG_COURSE: readonly CoursePiece[] = (() => {
  const shape: (readonly [length: number, curvature: number])[] = [];
  for (let bend = 0; bend < NBFLIP_SLIDE_LONG_BENDS; bend++) {
    shape.push([LONG_LEG, 0], [Math.PI * LONG_BEND_RADIUS, (bend % 2 ? 1 : -1) / LONG_BEND_RADIUS]);
  }
  // Half a straight brings the channel back to mid-width, where the last turn starts.
  shape.push([LONG_LEG / 2, 0], ...LONG_BOWL_RADII.map(radius => [Math.PI * radius, -1 / radius] as const));
  const pieces: CoursePiece[] = [];
  let from = 0, x: number = LONG_START[0], z: number = LONG_START[1], heading = Math.PI / 2;
  for (const [length, curvature] of shape) {
    pieces.push({ from, length, x, z, heading, curvature });
    [x, z, heading] = follow({ x, z, heading, curvature }, length);
    from += length;
  }
  return pieces;
})();
const LONG_BOWL = LONG_COURSE.slice(-LONG_BOWL_RADII.length);
/** The course's length, from the wall behind the source to the wall that stops it. */
export const NBFLIP_SLIDE_LONG_LENGTH = LONG_BOWL[1]!.from + LONG_BOWL[1]!.length;
/** Every arc of the course: where it starts and ends along it, and +1 where it turns from +x toward +z. */
export const NBFLIP_SLIDE_LONG_ARCS: readonly (readonly [from: number, to: number, turn: number])[] =
  LONG_COURSE.filter(piece => piece.curvature !== 0).map(piece => [piece.from, piece.from + piece.length, Math.sign(piece.curvature)]);

/** The long slide's centreline at a distance along it: lattice x, z and heading. */
export function nbflipSlideLongCentreline(distance: number): readonly [x: number, z: number, heading: number] {
  let piece = LONG_COURSE[0]!;
  for (const next of LONG_COURSE) if (next.from <= distance) piece = next;
  return follow(piece, distance - piece.from);
}

/** Continuous descent, level for the last thirty cells under the hood. */
const longFloor = (distance: number) =>
  LONG_TOP - LONG_FALL * Math.min(1, Math.max(0, distance) / (NBFLIP_SLIDE_LONG_LENGTH - HOOD));

/** A swept rectangular cross-section, in course coordinates. */
interface LongSweep {
  readonly piece: CoursePiece;
  readonly from: number;
  readonly to: number;
  readonly acrossLow: number;
  readonly acrossHigh: number;
  readonly joinFrom?: boolean;
  readonly joinTo?: boolean;
  /** Vertical bounds at a distance along the entire course. */
  readonly bottom: (distance: number) => number;
  readonly top: (distance: number) => number;
}

const LONG_SWEEPS: readonly LongSweep[] = (() => {
  const sweeps: LongSweep[] = [], reach = PITCH - HALF_WIDTH;
  const end = NBFLIP_SLIDE_LONG_LENGTH, lid = LONG_TOP - LONG_FALL + HOOD_CLEARANCE;
  const after = LONG_LEG + Math.PI / 2 * LONG_BEND_RADIUS;
  for (const piece of LONG_COURSE) {
    const bottom = piece.from < LONG_BOWL[0]!.from ? (s: number) => longFloor(s) - LONG_SLAB : () => 0;
    const add = (from: number, to: number, acrossLow: number, acrossHigh: number, top: LongSweep["top"], low = bottom) => {
      if (to > from) sweeps.push({ piece, from, to, acrossLow, acrossHigh, bottom: low, top });
    };
    const from = piece.from, to = from + piece.length;
    add(from, to, -reach, reach, longFloor, s => longFloor(s) - LONG_SLAB);
    if (from >= LONG_BOWL[0]!.from) add(from, to, -reach, reach, s => longFloor(s) - LONG_SLAB / 2, () => 0);
    for (const side of [-1, 1]) {
      const wallEnd = side * LONG_BOWL[1]!.curvature > 0 ? Math.min(to, end - GAP) : to;
      const high = LONG_COURSE.filter(arc => arc.curvature * side < 0)
        .map(arc => [arc.from - LONG_LEG, arc.from + arc.length + after] as const);
      // Split at height changes so each step has a real vertical end face.
      const cuts = [...new Set([from, wallEnd, ...high.flatMap(range => [...range])])]
        .filter(s => s >= from && s <= wallEnd).sort((a, b) => a - b);
      for (let i = 1; i < cuts.length; i++) {
        const a = cuts[i - 1]!, b = cuts[i]!, middle = (a + b) / 2;
        const height = high.some(([lo, hi]) => middle >= lo && middle <= hi) ? LONG_HIGH : LONG_LOW;
        add(a, b, side < 0 ? -reach : HALF_WIDTH, side < 0 ? -HALF_WIDTH : reach, s => longFloor(s) + height);
      }
    }
    if (from === 0) add(-BACK, 0, -reach, reach, () => LONG_TOP + LONG_HIGH);
    if (to === end) {
      add(end, end + STOP, -reach, reach, () => lid + HOOD_LID);
      add(end - HOOD, end, -reach, reach, () => lid + HOOD_LID, () => lid);
    }
  }
  // Continue a cross-section two cells into a neighbour that contains it.
  // At a height step only the lower wall extends, preserving the step face.
  // Merely touching CSG caps leave a zero-valued sheet inside the solid; node
  // interpolation would pull the floor down to that sheet at each bend.
  return sweeps.map(sweep => ({ ...sweep,
    joinFrom: sweeps.some(other => other.to === sweep.from && other.acrossLow === sweep.acrossLow
      && other.acrossHigh === sweep.acrossHigh && other.top(other.to) >= sweep.top(sweep.from)
      && other.bottom(other.to) <= sweep.bottom(sweep.from)),
    joinTo: sweeps.some(other => other.from === sweep.to && other.acrossLow === sweep.acrossLow
      && other.acrossHigh === sweep.acrossHigh && other.top(other.from) >= sweep.top(sweep.to)
      && other.bottom(other.from) <= sweep.bottom(sweep.to)),
  }));
})();

/** Continuous horizontal cuts through the swept slabs, walls and roof. The
 * caps use tangent planes; adjacent pieces share a cross-section exactly.
 * Contained cross-sections overlap by two sample cells so the baked field has no
 * internal cap sheet. Physical end faces and height changes keep their caps. */
function longSolidColumn(x: number, z: number): readonly (readonly [edge: number, bottom: number, top: number, cap: number, radius: number])[] {
  const spans: [number, number, number, number, number][] = [];
  for (const sweep of LONG_SWEEPS) {
    const { piece, from, to } = sweep;
    const cos = Math.cos(piece.heading), sin = Math.sin(piece.heading);
    const dx = x - piece.x, dz = z - piece.z;
    let along = dx * cos + dz * sin, across = -dx * sin + dz * cos;
    if (piece.curvature !== 0) {
      const radius = 1 / piece.curvature;
      const radialX = dx + sin * radius, radialZ = dz - cos * radius;
      // Measure about the arc's midpoint, keeping atan2's cut behind the arc.
      const middle = piece.heading + piece.curvature * piece.length / 2;
      const angle = Math.atan2((radialX * Math.cos(middle) + radialZ * Math.sin(middle)) / radius,
        (radialX * Math.sin(middle) - radialZ * Math.cos(middle)) / radius);
      along = piece.length / 2 + angle / piece.curvature;
      across = radius - Math.sign(radius) * Math.hypot(radialX, radialZ);
    }
    const [ax, az, ah] = follow(piece, from - piece.from), [bx, bz, bh] = follow(piece, to - piece.from);
    const cap = Math.max(-(x - ax) * Math.cos(ah) - (z - az) * Math.sin(ah) - (sweep.joinFrom ? 2 : 0),
      (x - bx) * Math.cos(bh) + (z - bz) * Math.sin(bh) - (sweep.joinTo ? 2 : 0));
    const edge = Math.max(sweep.acrossLow - across, across - sweep.acrossHigh);
    // Values beyond four cells are saturated; no zero or boundary gradient is
    // changed. This keeps each cached column small, including the empty world.
    const distance = piece.from + along, top = sweep.top(distance);
    if (Math.max(edge, cap) < 4) spans.push([edge, sweep.bottom(distance), top, cap, top > longFloor(distance) + 6 ? 1.5 : 0]);
  }
  for (const arc of LONG_BOWL) {
    const cx = arc.x - Math.sin(arc.heading) / arc.curvature, cz = arc.z + Math.cos(arc.heading) / arc.curvature;
    const dx = x - cx, dz = z - cz;
    const edge = Math.max(Math.hypot(dx, dz) - 1 / Math.abs(arc.curvature),
      -dx * Math.cos(arc.heading) - dz * Math.sin(arc.heading) - 2);
    if (edge < 4) spans.push([edge, 0, PIT_FLOOR, -Infinity, 0]);
  }
  return spans;
}

function longColumnValue(spans: ReturnType<typeof longSolidColumn>, y: number): number {
  let solid = 4;
  // A three-centimetre rim radius fits inside the eight-centimetre walls.
  // Flat beds and vertical faces keep their authored positions; only their
  // exposed convex intersections are rounded, like a moulded flume.
  for (const [edge, bottom, top, cap, radius] of spans) {
    if (radius === 0) { solid = Math.min(solid, Math.max(cap, edge, bottom - y, y - top)); continue; }
    const side = edge + radius, vertical = Math.max(bottom - y, y - top) + radius;
    const rounded = Math.hypot(Math.max(side, 0), Math.max(vertical, 0))
      + Math.min(Math.max(side, vertical), 0) - radius;
    solid = Math.min(solid, Math.max(cap, rounded));
  }
  return solid === 0 ? 0 : solid;
}

/** Distance-like CSG field in lattice coordinates, negative inside. The same
 * sampled boundary drives collision, smooth reconstruction and shading. */
export function nbflipSlideLongSolid([x, y, z]: readonly [number, number, number]): number {
  return longColumnValue(longSolidColumn(x, z), y);
}

let longSlideGeometry: ReturnType<typeof bakeProceduralSolid> | undefined;
function bakeLongSlide(): ReturnType<typeof bakeProceduralSolid> {
  // All y levels share the same analytic x/z projection. Cache only for this
  // bake; documents retain the sampled boundary, not the construction cache.
  const columns = new Map<number, ReturnType<typeof longSolidColumn>>();
  const h = NBFLIP_SLIDE_CELL_SIZE_M, stride = NBFLIP_SLIDE_LONG_GRID[0] + 1;
  return bakeProceduralSolid(NBFLIP_SLIDE_LONG_GRID, [h, h, h], ([x, y, z]) => {
    const key = x + stride * z;
    let spans = columns.get(key);
    if (!spans) columns.set(key, spans = longSolidColumn(x, z));
    return longColumnValue(spans, y);
  }, VOXEL_MATERIAL_IDS.container);
}

export function createNbflipWaterSlideLong(): SceneDescription {
  const grid = NBFLIP_SLIDE_LONG_GRID, h = NBFLIP_SLIDE_CELL_SIZE_M, dt = 1 / NBFLIP_SLIDE_STEP_HZ;
  const scene = nbflipDomain(NBFLIP_SLIDE_LONG_SCENE_ID, grid, h, NBFLIP_SLIDE_LONG_FRAMES, 0, grid, dt);
  const geometry = longSlideGeometry ??= bakeLongSlide();
  scene.solidVoxels = [...geometry.solidVoxels];
  scene.solidBoundary = geometry.solidBoundary;
  // The stream lies on the floor just past the back wall and leaves along the channel.
  const [x, z, heading] = nbflipSlideLongCentreline(STREAM_AHEAD), speed = NBFLIP_SLIDE_STREAM_CELLS_PER_STEP;
  scene.fluid.inflow = streamInflow(grid, h, dt, [x, z], LONG_TOP,
    [speed * Math.cos(heading), speed * Math.sin(heading)], NBFLIP_SLIDE_LONG_STREAM_RADIUS, NBFLIP_SLIDE_LONG_STREAM_SECONDS);
  // Storage starts at the stream's pages and follows the water along the course: 200 blocks of 4.90 MiB hold
  // every one of the box's 16x3x4 = 192 pages (two are the AIR and SOLID sentinels).
  scene.fluid.pagePool = { blocks: 200 };
  return scene;
}

/**
 * From nearly overhead, far enough back to hold the whole length: the walls
 * stand deeper than the channel is wide, so a lower eye sees only their tops.
 * It leans toward the source's end, where they stand nearest to it.
 */
export function nbflipWaterSlideLongCamera(): Partial<CameraState> {
  return { azimuth_rad: -0.3, elevation_rad: 1.3, distance_m: 10.5, target_m: { x: 0, y: 0.7, z: 0 } };
}
