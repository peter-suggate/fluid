import type { CameraState, SceneDescription } from "./model";
import { cellPoint, nbflipDomain } from "./nbflip-paper-scenes";
import type { SolidWorldVoxelPatch } from "./solid-world";
import { VOXEL_MATERIAL_IDS } from "./voxel-scene";

/**
 * A voxel water slide: a scene whose liquid starts as nothing, grows along a
 * long channel and ends as a pool, so the space the solver has to hold grows
 * with it.
 *
 * The slide is a channel cut into the top of a solid drum. It winds inward for
 * two turns, falling a cell at a time, and ends under a hood that spills
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
function columnVoxels([nx, , nz]: readonly [number, number, number], span: (x: number, z: number) => readonly [from: number, top: number]): SolidWorldVoxelPatch[] {
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
function streamInflow(
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

export function createNbflipWaterSlide(): SceneDescription {
  const grid = NBFLIP_SLIDE_GRID, h = NBFLIP_SLIDE_CELL_SIZE_M, dt = 1 / NBFLIP_SLIDE_STEP_HZ;
  const scene = nbflipDomain(NBFLIP_SLIDE_SCENE_ID, grid, h, NBFLIP_SLIDE_FRAMES, 0, grid, dt);
  const lid = HOOD_FLOOR + HOOD_CLEARANCE;
  scene.solidVoxels.push(
    ...columnVoxels(grid, (x, z) => [0, slideColumn(x, z)[0]]),
    ...columnVoxels(grid, (x, z) => [lid, slideColumn(x, z)[1] ? lid + HOOD_LID : lid]),
  );
  // The stream lies on the floor just past the back wall and leaves along the channel.
  const angle = STREAM_AHEAD / START_RADIUS, speed = NBFLIP_SLIDE_STREAM_CELLS_PER_STEP;
  scene.fluid.inflow = streamInflow(grid, h, dt, nbflipSlideCentreline(angle), TOP,
    [speed * Math.cos(angle), speed * Math.sin(angle)], NBFLIP_SLIDE_STREAM_RADIUS, NBFLIP_SLIDE_STREAM_SECONDS);
  // Storage starts at the stream's pages and follows the water. The water reaches every one of the drum's 5x3x5 = 75
  // pages, so the pool holds them all: 80 blocks of 4.90 MiB (two are the AIR and SOLID sentinels).
  scene.fluid.pagePool = { blocks: 80 };
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

/** The floor under the centreline: whole steps down to the hood, flat under it. */
const longFloor = (distance: number) =>
  LONG_TOP - Math.floor(LONG_FALL * Math.min(1, Math.max(0, distance) / (NBFLIP_SLIDE_LONG_LENGTH - HOOD)));

/**
 * A wall's height over the floor, on the side an arc of curvature `side`
 * turns toward. It is high wherever a stream can lean or land on it: outside
 * a bend, from the straight before it to a quarter turn into the next.
 */
function longWall(distance: number, side: number): number {
  // The hood's gap is in the wall the pit is behind.
  if (side * LONG_BOWL[1]!.curvature > 0 && distance >= NBFLIP_SLIDE_LONG_LENGTH - GAP) return 0;
  const after = LONG_LEG + Math.PI / 2 * LONG_BEND_RADIUS;
  return LONG_COURSE.some(arc => arc.curvature * side < 0 && distance >= arc.from - LONG_LEG && distance <= arc.from + arc.length + after)
    ? LONG_HIGH : LONG_LOW;
}

/**
 * The long slide's solid under each column of cells, and the hood's lid over
 * it: where each starts and ends.
 *
 * The course is walked half a cell at a time, and each step claims the columns
 * beside it: channel within the half width, wall for a wall's thickness past
 * that. A column that is channel to any step is channel; one that is wall to
 * two stretches stands as high as the higher asks and as deep as the deeper.
 */
function longSlideColumns(): readonly [body: (x: number, z: number) => readonly [number, number], lid: (x: number, z: number) => readonly [number, number]] {
  const [nx, , nz] = NBFLIP_SLIDE_LONG_GRID, reach = PITCH - HALF_WIDTH, length = NBFLIP_SLIDE_LONG_LENGTH;
  const near = new Float32Array(nx * nz).fill(reach), from = new Int16Array(nx * nz), top = new Int16Array(nx * nz), lidded = new Uint8Array(nx * nz);
  const lid = LONG_TOP - LONG_FALL + HOOD_CLEARANCE, pitSide = Math.sign(LONG_BOWL[1]!.curvature);
  for (let distance = -BACK; distance <= length + STOP; distance += 0.5) {
    const [px, pz, heading] = nbflipSlideLongCentreline(distance), cos = Math.cos(heading), sin = Math.sin(heading);
    // The flume hangs on its slab; the last turn stands on the ground, as the pit's wall has to.
    const floor = longFloor(distance), base = distance < LONG_BOWL[0]!.from ? floor - LONG_SLAB : 0;
    // The walls behind the source and at the channel's end are solid across it.
    const block = distance < 0 ? LONG_TOP + LONG_HIGH : distance > length ? lid + HOOD_LID : 0;
    const walls = [block || floor + longWall(distance, -1), block || floor + longWall(distance, 1)] as const;
    const hood = distance >= length - HOOD && distance <= length;
    for (let z = Math.max(0, Math.floor(pz - reach)); z < Math.min(nz, pz + reach); z++) {
      for (let x = Math.max(0, Math.floor(px - reach)); x < Math.min(nx, px + reach); x++) {
        const dx = x + 0.5 - px, dz = z + 0.5 - pz, across = cos * dz - sin * dx, off = Math.abs(across), c = x + nx * z;
        // Only the columns beside this step, not those ahead of it or behind.
        if (Math.abs(cos * dx + sin * dz) >= 0.75 || off >= reach) continue;
        if (off < HALF_WIDTH) {
          if (off >= near[c]!) continue;
          top[c] = block || floor; from[c] = base; lidded[c] = hood ? 1 : 0;
        } else {
          if (near[c]! < HALF_WIDTH) continue;
          const wall = walls[across > 0 ? 1 : 0];
          from[c] = near[c] === reach ? base : Math.min(from[c]!, base);
          top[c] = Math.max(top[c]!, wall);
          if (hood && across * pitSide > 0) lidded[c] = 1;
        }
        near[c] = Math.min(near[c]!, off);
      }
    }
  }
  // The pit: whatever the last turn left unclaimed inside itself, each half of the turn about its own centre.
  const pit = (x: number, z: number) => LONG_BOWL.some(arc => {
    const dx = x + 0.5 - (arc.x - Math.sin(arc.heading) / arc.curvature), dz = z + 0.5 - (arc.z + Math.cos(arc.heading) / arc.curvature);
    return dx * Math.cos(arc.heading) + dz * Math.sin(arc.heading) >= 0 && Math.hypot(dx, dz) < 1 / Math.abs(arc.curvature);
  });
  return [
    (x, z) => near[x + nx * z]! < reach ? [from[x + nx * z]!, top[x + nx * z]!] : [0, pit(x, z) ? PIT_FLOOR : 0],
    (x, z) => [lid, lidded[x + nx * z] ? lid + HOOD_LID : lid],
  ];
}

export function createNbflipWaterSlideLong(): SceneDescription {
  const grid = NBFLIP_SLIDE_LONG_GRID, h = NBFLIP_SLIDE_CELL_SIZE_M, dt = 1 / NBFLIP_SLIDE_STEP_HZ;
  const scene = nbflipDomain(NBFLIP_SLIDE_LONG_SCENE_ID, grid, h, NBFLIP_SLIDE_LONG_FRAMES, 0, grid, dt);
  const [body, lid] = longSlideColumns();
  scene.solidVoxels.push(...columnVoxels(grid, body), ...columnVoxels(grid, lid));
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
