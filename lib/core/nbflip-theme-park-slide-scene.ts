import type { CameraState, SceneDescription } from "./model";
import { nbflipDomain } from "./nbflip-paper-scenes";
import { columnVoxels, NBFLIP_SLIDE_CELL_SIZE_M, NBFLIP_SLIDE_STEP_HZ, streamInflow } from "./nbflip-water-slide-scene";
import type { SolidWorldVoxelPatch } from "./solid-world";
import { VOXEL_MATERIAL_IDS } from "./voxel-scene";
import { themeParkScenery } from "./theme-park-scenery";
import type { SweptFlumeSurface } from "./swept-flume-surface";

/** A suspended two-turn spiral, an underpass and reversing bends above a splash pool.
 * Unlike the drum and serpentine, multiple decks occupy the same x/z columns.
 * Each visit of the course owns its own vertical span, leaving air between decks
 * for the water (and the sparse page allocator) to travel through.
 * All geometry below is in cells; the solver still uses the slides' 2 cm cells.
 */
export const NBFLIP_THEME_PARK_SCENE_ID = "nbflip-water-slide-theme-park";
export const NBFLIP_THEME_PARK_GRID = [384, 256, 320] as const;
export const NBFLIP_THEME_PARK_STREAM_SECONDS = 18;
export const NBFLIP_THEME_PARK_FRAMES = 2700;
export const NBFLIP_THEME_PARK_HALF_WIDTH = 10;
export const NBFLIP_THEME_PARK_WALL = 32;
const REACH = NBFLIP_THEME_PARK_HALF_WIDTH + 4, SLAB = 6;
const TOP = 212, HELIX_RADIUS = 72, TURN = 2 * Math.PI;
const BACK = 6, STOP = 6, HOOD = 24, GAP = 20;
const SILL = 32, LID = SILL + NBFLIP_THEME_PARK_WALL;
export const NBFLIP_THEME_PARK_POOL = { minX: 288, maxX: 372, minZ: 236, maxZ: 312, floor: 4, rim: 48, wall: 4 } as const;

interface Piece {
  readonly from: number; readonly length: number;
  readonly x: number; readonly z: number; readonly heading: number;
  readonly curvature: number; readonly top: number; readonly fall: number;
}

function follow(piece: Piece, distance: number): readonly [number, number, number] {
  const { x, z, heading, curvature } = piece, angle = heading + curvature * distance;
  return curvature === 0 ? [x + distance * Math.cos(heading), z + distance * Math.sin(heading), heading]
    : [x + (Math.sin(angle) - Math.sin(heading)) / curvature, z - (Math.cos(angle) - Math.cos(heading)) / curvature, angle];
}

const COURSE: readonly Piece[] = (() => {
  // Each lap falls 60 cells: room for the lower flume's walls, its water and
  // the upper deck's slab. The straight then passes beneath the launch deck.
  const shape = [
    [TURN * HELIX_RADIUS, 1 / HELIX_RADIUS, 60],
    [TURN * HELIX_RADIUS, 1 / HELIX_RADIUS, 60],
    [128, 0, 18],
    [Math.PI / 2 * 56, 1 / 56, 8],
    [24, 0, 6],
    [Math.PI / 2 * 56, 1 / 56, 8],
    [Math.PI * 40, -1 / 40, 12],
    [48, 0, 8],
    [HOOD, 0, 0],
  ] as const;
  const pieces: Piece[] = [];
  let from = 0, x = 104, z = 64, heading = 0, top = TOP;
  for (const [length, curvature, fall] of shape) {
    const piece = { from, length, x, z, heading, curvature, top, fall };
    pieces.push(piece);
    [x, z, heading] = follow(piece, length);
    from += length; top -= fall;
  }
  return pieces;
})();
export const NBFLIP_THEME_PARK_LENGTH = COURSE.reduce((sum, piece) => sum + piece.length, 0);

/** Position, heading and stepped floor at a distance along the entire ride. */
export function nbflipThemeParkCentreline(distance: number): readonly [x: number, z: number, heading: number, floor: number] {
  const piece = COURSE.find(piece => distance < piece.from + piece.length) ?? COURSE[COURSE.length - 1]!;
  const local = distance - piece.from;
  return [...follow(piece, local), piece.top - Math.floor(piece.fall * Math.max(0, Math.min(1, local / piece.length)))];
}

interface Visit { last: number; nearest: number; base: number; floor: number; top: number; hood: boolean }

function flumeColumns(): Visit[][] {
  const [nx, , nz] = NBFLIP_THEME_PARK_GRID, length = NBFLIP_THEME_PARK_LENGTH;
  const columns: Visit[][] = Array.from({ length: nx * nz }, () => []);
  for (let distance = -BACK; distance <= length + STOP; distance += 0.5) {
    const [px, pz, heading, floor] = nbflipThemeParkCentreline(distance), cos = Math.cos(heading), sin = Math.sin(heading);
    const block = distance < 0 ? TOP + NBFLIP_THEME_PARK_WALL : distance > length ? LID + 4 : 0;
    const hood = distance >= length - HOOD;
    for (let z = Math.max(0, Math.floor(pz - REACH)); z < Math.min(nz, pz + REACH); z++) {
      for (let x = Math.max(0, Math.floor(px - REACH)); x < Math.min(nx, px + REACH); x++) {
        const dx = x + 0.5 - px, dz = z + 0.5 - pz, across = Math.abs(cos * dz - sin * dx);
        if (Math.abs(cos * dx + sin * dz) >= 0.75 || across >= REACH) continue;
        const visits = columns[x + nx * z]!;
        let visit = visits[visits.length - 1];
        // Adjacent samples belong to one deck; a later lap is a new deck.
        if (!visit || distance - visit.last > 4) {
          visits.push(visit = { last: distance, nearest: REACH, base: floor - SLAB, floor, top: floor, hood: false });
        }
        visit.last = distance;
        const channel = across < NBFLIP_THEME_PARK_HALF_WIDTH;
        if (channel) {
          if (across >= visit.nearest) continue;
          visit.base = floor - SLAB; visit.floor = floor; visit.top = block || floor; visit.hood = hood;
        } else {
          if (visit.nearest < NBFLIP_THEME_PARK_HALF_WIDTH) continue;
          visit.base = Math.min(visit.base, floor - SLAB);
          visit.floor = Math.max(visit.floor, floor);
          // Both sides open under the impact hood, venting into the pool.
          const wall = distance >= length - GAP ? 0 : NBFLIP_THEME_PARK_WALL;
          visit.top = Math.max(visit.top, block || floor + wall); visit.hood ||= hood;
        }
        visit.nearest = Math.min(visit.nearest, across);
      }
    }
  }
  return columns;
}

export function createNbflipThemeParkSlide(): SceneDescription {
  const grid = NBFLIP_THEME_PARK_GRID, [nx] = grid, h = NBFLIP_SLIDE_CELL_SIZE_M, dt = 1 / NBFLIP_SLIDE_STEP_HZ;
  const scene = nbflipDomain(NBFLIP_THEME_PARK_SCENE_ID, grid, h, NBFLIP_THEME_PARK_FRAMES, 0, grid, dt);
  const columns = flumeColumns(), decks = columns.reduce((n, visits) => Math.max(n, visits.length), 0);
  for (let deck = 0; deck < decks; deck++) {
    const at = (x: number, z: number) => columns[x + nx * z]![deck];
    scene.solidVoxels.push(
      ...columnVoxels(grid, (x, z) => { const v = at(x, z); return v ? [v.base, v.floor] : [0, 0]; }),
      ...columnVoxels(grid, (x, z) => { const v = at(x, z); return v ? [v.floor, v.top] : [0, 0]; })
        .map(patch => ({ ...patch, materialId: VOXEL_MATERIAL_IDS.sphere })),
    );
  }
  scene.solidVoxels.push(...columnVoxels(grid, (x, z) => [LID, columns[x + nx * z]!.some(v => v.hood) ? LID + 4 : LID]));

  const surfaceBoxes: { min: number[]; max: number[] }[] = [];
  const world = (p: readonly number[]) => [(p[0]! - grid[0] / 2) * h, p[1]! * h, (p[2]! - grid[2] / 2) * h];
  const box = (min: [number, number, number], max: [number, number, number], materialId: number = VOXEL_MATERIAL_IDS.box): SolidWorldVoxelPatch => {
    surfaceBoxes.push({ min: world(min), max: world(max) });
    return { operation: "fill", minimum: min, maximumExclusive: max, materialId };
  };
  // Piers terminate under the lowest deck in their footprint, including at
  // crossings. They cannot accidentally seal the lower course with a pillar.
  for (let distance = 24; distance < NBFLIP_THEME_PARK_LENGTH - 100; distance += 88) {
    const [px, pz] = nbflipThemeParkCentreline(distance), x = Math.floor(px) - 3, z = Math.floor(pz) - 3;
    let top = TOP;
    for (let dz = 0; dz < 6; dz++) for (let dx = 0; dx < 6; dx++) {
      for (const visit of columns[x + dx + nx * (z + dz)]!) top = Math.min(top, visit.base);
    }
    if (top > 0) scene.solidVoxels.push(box([x, 0, z], [x + 6, top, z + 6]));
  }
  const pool = NBFLIP_THEME_PARK_POOL;
  scene.solidVoxels.push(
    box([pool.minX, 0, pool.minZ], [pool.maxX, pool.floor, pool.maxZ]),
    box([pool.minX, pool.floor, pool.minZ], [pool.maxX, pool.rim, pool.minZ + pool.wall]),
    box([pool.minX, pool.floor, pool.maxZ - pool.wall], [pool.maxX, pool.rim, pool.maxZ]),
    box([pool.maxX - pool.wall, pool.floor, pool.minZ], [pool.maxX, pool.rim, pool.maxZ]),
    box([pool.minX, pool.floor, pool.minZ], [pool.minX + pool.wall, pool.rim, 280 - NBFLIP_THEME_PARK_HALF_WIDTH]),
    box([pool.minX, pool.floor, 280 + NBFLIP_THEME_PARK_HALF_WIDTH], [pool.minX + pool.wall, pool.rim, pool.maxZ]),
    box([pool.minX, pool.floor, 280 - NBFLIP_THEME_PARK_HALF_WIDTH], [pool.minX + pool.wall, SILL, 280 + NBFLIP_THEME_PARK_HALF_WIDTH]),
  );
  const [x, z, heading, floor] = nbflipThemeParkCentreline(10), speed = 1.6;
  scene.fluid.inflow = streamInflow(grid, h, dt, [x, z], floor, [speed * Math.cos(heading), speed * Math.sin(heading)], 4, NBFLIP_THEME_PARK_STREAM_SECONDS);
  // 960 virtual pages; enough physical blocks for the swept channel, its
  // migration halo and the pool, even if the whole course is wet at once.
  scene.fluid.pagePool = { blocks: 384 };
  scene.solidVoxels = scene.solidVoxels.map(patch => ({ ...patch, materialId: VOXEL_MATERIAL_IDS.white }));
  scene.scenery = themeParkScenery();
  scene.surfaceStyle = "smooth";
  // Continuous presentation follows exactly the course used for the collider.
  // Only the sub-cell edge fillet and the stepped fall are reconstructed; the
  // solver's course, clearance and capacity remain unchanged.
  scene.solidSurface = {
    kind: "swept-flume", halfWidth: NBFLIP_THEME_PARK_HALF_WIDTH * h,
    wallThickness: (REACH - NBFLIP_THEME_PARK_HALF_WIDTH) * h,
    wallHeight: NBFLIP_THEME_PARK_WALL * h, slab: SLAB * h, edgeRadius: .012,
    back: BACK * h, stop: STOP * h, boxes: surfaceBoxes,
    pieces: COURSE.map((p, i) => ({ x: (p.x - grid[0] / 2) * h, z: (p.z - grid[2] / 2) * h,
      heading: p.heading, curvature: p.curvature / h, length: p.length * h,
      top: p.top * h, fall: p.fall * h,
      wallEnd: Math.max(0, Math.min(p.length, NBFLIP_THEME_PARK_LENGTH - GAP - p.from)) * h,
      hood: i === COURSE.length - 1,
    })),
  } satisfies SweptFlumeSurface;
  scene.lighting = {
    directional: { colorLinear: [1, 1, 1], intensity: 3.1, direction: [-.55, .8, -.35] },
    environment: { diffuseScale: .8, specularScale: .8, lowerRadianceLinear: [.55, .55, .55],
      upperRadianceLinear: [.78, .78, .78], accentRadianceLinear: [.65, .65, .65] },
    grade: { toneCurve: "aces", exposure: .85, whiteBalance: [1, 1, 1] },
  };
  return scene;
}

export function nbflipThemeParkCamera(): Partial<CameraState> {
  return { azimuth_rad: -0.55, elevation_rad: .78, distance_m: 23.8, tanHalfFov: .48, target_m: { x: 0, y: .5, z: -.7 } };
}
