import { intersectBox, type EditorRay } from "../editor-entity";
import type { PickedSolidVoxel, SolidVoxelPickOptions } from "../editor-solid-voxel";
import type { SceneDescription } from "../model";
import { sceneCellSizes_m, sceneLatticeDimensions } from "../scene-lattice";
import { solidFractionOccupies, type VoxelBlock } from "../solid-patch/kind";
import { SOLID_WORLD_TERRAIN_MATERIAL_ID, type SolidGround, type SolidWorldCoordinate } from "../solid-world";
import type { VoxelReader } from "./voxel-access";
import { VOXEL_INDEX_CLEARED_PAGE, voxelIndexForScene, type VoxelIndex, type VoxelIndexPage } from "./voxel-index";

/** The most cells one `read` may copy: 256 cubed, 80 MB of channels. */
export const MAX_VOXEL_READ_CELLS = 256 ** 3;
/** Cells a pick walks before it gives up — the bound an editing ray has always had. */
export const VOXEL_PICK_RAY_CELLS = 4096;

const AXES = [0, 1, 2] as const;

/**
 * A scene's solids, read from the authored-voxel index and the analytic ground.
 *
 * A cell is answered by its page when a patch has touched that page, and by the
 * terrain otherwise — the rule the index and the solid world are both built on,
 * over the one `SolidGround`. The first question after a document clone costs
 * the index of its patches (or nothing, when an equal list was indexed before)
 * and the heights of the columns it reads.
 */
class SceneVoxelReader implements VoxelReader {
  private readonly index: VoxelIndex;
  private readonly ground: SolidGround | undefined;
  private readonly cell_m: readonly [number, number, number];
  private readonly origin_m: readonly [number, number, number];
  /** The lattice box a pick ray is clipped to: the tank, every patch, the ground. */
  private readonly pickMinimum: SolidWorldCoordinate;
  private readonly pickMaximum: SolidWorldCoordinate;
  /** What the scene held when it was read, so an object edited in place is noticed. */
  private readonly document: readonly unknown[];
  /** The page last asked about. Readers are asked cell by neighbouring cell. */
  private pageX = Number.NaN;
  private pageY = Number.NaN;
  private pageZ = Number.NaN;
  private page: VoxelIndexPage | undefined;

  constructor(readonly scene: SceneDescription) {
    this.document = SceneVoxelReader.documentOf(scene);
    this.index = voxelIndexForScene(scene);
    this.ground = this.index.ground;
    this.cell_m = sceneCellSizes_m(scene);
    this.origin_m = [-0.5 * scene.container.width_m, 0, -0.5 * scene.container.depth_m];
    const minimum = [0, 0, 0], maximum = [...sceneLatticeDimensions(scene)];
    for (const box of [this.index, this.ground]) for (const axis of AXES) {
      if (!box?.minimum || !box.maximumExclusive) continue;
      minimum[axis] = Math.min(minimum[axis]!, box.minimum[axis]);
      maximum[axis] = Math.max(maximum[axis]!, box.maximumExclusive[axis]);
    }
    this.pickMinimum = minimum as unknown as SolidWorldCoordinate;
    this.pickMaximum = maximum as unknown as SolidWorldCoordinate;
  }

  private static documentOf(scene: SceneDescription): readonly unknown[] {
    return [scene.solidVoxels, scene.solidVoxels.length, scene.terrain, scene.container.width_m,
      scene.container.height_m, scene.container.depth_m, scene.voxelDomain.finestCellSize_m];
  }

  /** Whether the scene object still holds the document this reader was made from. */
  current(): boolean {
    return SceneVoxelReader.documentOf(this.scene).every((value, field) => value === this.document[field]);
  }

  private pageOf(x: number, y: number, z: number): VoxelIndexPage | undefined {
    if ((x | 0) !== x || (y | 0) !== y || (z | 0) !== z) {
      throw new RangeError(`Voxel coordinate ${x}, ${y}, ${z} must be signed i32 integers`);
    }
    const px = x >> 3, py = y >> 3, pz = z >> 3;
    if (px !== this.pageX || py !== this.pageY || pz !== this.pageZ) {
      this.page = this.index.page(px, py, pz);
      this.pageX = px; this.pageY = py; this.pageZ = pz;
    }
    return this.page;
  }

  fraction(coordinate: SolidWorldCoordinate): number {
    const x = coordinate[0], y = coordinate[1], z = coordinate[2];
    const page = this.pageOf(x, y, z);
    if (page) return page.solidFraction[(x & 7) + 8 * ((y & 7) + 8 * (z & 7))]!;
    return this.ground ? this.ground.fraction(x, y, z) : 0;
  }

  solid(coordinate: SolidWorldCoordinate): boolean {
    return solidFractionOccupies(this.fraction(coordinate));
  }

  material(coordinate: SolidWorldCoordinate): number {
    const x = coordinate[0], y = coordinate[1], z = coordinate[2];
    const page = this.pageOf(x, y, z);
    if (page) return page.materialId[(x & 7) + 8 * ((y & 7) + 8 * (z & 7))]!;
    return this.ground && solidFractionOccupies(this.ground.fraction(x, y, z)) ? SOLID_WORLD_TERRAIN_MATERIAL_ID : 0;
  }

  /**
   * Walk the cells the ray crosses, in order, and stop at the first solid one.
   *
   * Empty space is crossed a page or a node at a time: a page the index has
   * cleared, a page nothing has touched that the ground cannot reach, a node
   * with neither. What it costs is the length of the ray, never the length of
   * the document — the patch list is not consulted at all.
   */
  pick(ray: EditorRay, options: SolidVoxelPickOptions = {}): PickedSolidVoxel | undefined {
    const length = Math.hypot(ray.direction.x, ray.direction.y, ray.direction.z);
    if (!(length > 1e-12)) return undefined;
    const d = [ray.direction.x / length, ray.direction.y / length, ray.direction.z / length];
    const o = [ray.origin.x, ray.origin.y, ray.origin.z];
    const normalized = { origin: ray.origin, direction: { x: d[0]!, y: d[1]!, z: d[2]! } };
    const h = this.cell_m, origin = this.origin_m, lo = this.pickMinimum, hi = this.pickMaximum;
    const span = intersectBox(normalized, {
      min: { x: origin[0] + lo[0] * h[0], y: lo[1] * h[1], z: origin[2] + lo[2] * h[2] },
      max: { x: origin[0] + hi[0] * h[0], y: hi[1] * h[1], z: origin[2] + hi[2] * h[2] } });
    if (!span || span.far_m <= 1e-6) return undefined;
    const index = this.index, ground = this.ground;
    let t = Math.max(0, span.near_m);
    const q = o.map((value, axis) => Math.floor((value + (t + 1e-8) * d[axis]! - origin[axis]!) / h[axis]!));
    const crossings = [0, 0, 0];
    let steps = 0;
    while (steps < VOXEL_PICK_RAY_CELLS && t <= span.far_m) {
      const x = q[0]!, y = q[1]!, z = q[2]!;
      // How far the ray may go from here: 1 tests this cell, 8 and 64 leave a
      // page or a node that nothing stands in.
      let stride = 1, occupied = false;
      const node = index.node(x >> 6, y >> 6, z >> 6);
      const page = node?.pages[((x >> 3) & 7) + 8 * (((y >> 3) & 7) + 8 * ((z >> 3) & 7))];
      if (page === VOXEL_INDEX_CLEARED_PAGE) stride = 8;
      else if (page) occupied = solidFractionOccupies(page.solidFraction[(x & 7) + 8 * ((y & 7) + 8 * (z & 7))]!);
      else if (ground?.reaches(x & -8, y & -8, z & -8, 8)) occupied = solidFractionOccupies(ground.fraction(x, y, z));
      else stride = node || ground?.reaches(x & -64, y & -64, z & -64, 64) ? 8 : 64;
      if (occupied) {
        const picked = this.hit(normalized, [x, y, z], options);
        if (picked) return picked;
      }
      // Where the ray leaves the cell, or the whole block of `stride` cells it stands in.
      for (const axis of AXES) {
        const base = q[axis]! & -stride;
        crossings[axis] = d[axis] === 0 ? Infinity
          : (origin[axis] + (base + (d[axis]! > 0 ? stride : 0)) * h[axis] - o[axis]!) / d[axis]!;
      }
      const next = Math.min(crossings[0]!, crossings[1]!, crossings[2]!);
      if (!Number.isFinite(next)) return undefined;
      for (const axis of AXES) {
        if (d[axis] === 0) continue;
        const base = q[axis]! & -stride, forward = d[axis]! > 0;
        let landed: number;
        if (crossings[axis]! <= next + 1e-10) landed = forward ? base + stride : base - 1;
        else if (stride === 1) continue;
        else {
          // Still inside the block on this axis: the cell is wherever the ray
          // stands as it leaves, and never one it has already passed.
          landed = Math.floor((o[axis]! + (next + 1e-10) * d[axis]! - origin[axis]) / h[axis]);
          landed = forward ? Math.min(base + stride - 1, Math.max(q[axis]!, landed))
            : Math.max(base, Math.min(q[axis]!, landed));
        }
        if (stride > 1) steps += Math.abs(landed - q[axis]!);
        q[axis] = landed;
      }
      if (stride === 1) steps += 1;
      t = next;
    }
    if (t <= span.far_m && options.rejectBudgetExhaustion) {
      throw new Error(`This pick exceeds the bounded ${VOXEL_PICK_RAY_CELLS}-cell editing ray; point closer to the surface.`);
    }
    return undefined;
  }

  /** The pick for a solid cell the ray has reached, unless the caller passes through it. */
  private hit(ray: EditorRay, coordinate: SolidWorldCoordinate,
    options: SolidVoxelPickOptions): PickedSolidVoxel | undefined {
    if (options.skip?.(coordinate)) return undefined;
    const h = this.cell_m, origin = this.origin_m;
    const lower = AXES.map((axis) => origin[axis] + coordinate[axis] * h[axis]);
    const upper = AXES.map((axis) => origin[axis] + (coordinate[axis] + 1) * h[axis]);
    const span = intersectBox(ray, { min: { x: lower[0]!, y: lower[1]!, z: lower[2]! },
      max: { x: upper[0]!, y: upper[1]!, z: upper[2]! } });
    // The walk steps diagonally through a corner it comes within a hair of, and
    // the exact test may then say the ray never entered the cell. It did not.
    if (!span) return undefined;
    const distance_m = span.near_m > 1e-6 ? span.near_m : span.far_m;
    const point = [ray.origin.x + distance_m * ray.direction.x, ray.origin.y + distance_m * ray.direction.y,
      ray.origin.z + distance_m * ray.direction.z];
    let faceAxis: 0 | 1 | 2 = 0, faceSign: -1 | 1 = -1, nearest = Infinity;
    for (const axis of AXES) for (const sign of [-1, 1] as const) {
      const distance = Math.abs(point[axis]! - (sign > 0 ? upper[axis]! : lower[axis]!));
      if (distance < nearest) { nearest = distance; faceAxis = axis; faceSign = sign; }
    }
    return { coordinate, faceAxis, faceSign, distance_m, point_m: { x: point[0]!, y: point[1]!, z: point[2]! } };
  }

  read(minimum: SolidWorldCoordinate, maximumExclusive: SolidWorldCoordinate): VoxelBlock {
    const size = AXES.map((axis) => maximumExclusive[axis] - minimum[axis]) as unknown as SolidWorldCoordinate;
    if (![...minimum, ...maximumExclusive].every((value) => (value | 0) === value) || size.some((value) => value < 1)) {
      throw new RangeError("A voxel read needs a non-empty box of signed i32 cells");
    }
    const cells = size[0] * size[1] * size[2];
    if (cells > MAX_VOXEL_READ_CELLS) {
      throw new RangeError(`A voxel read of ${size.join(" x ")} cells exceeds the `
        + `${MAX_VOXEL_READ_CELLS.toLocaleString("en-US")}-cell limit`);
    }
    const solidFraction = new Uint8Array(cells);
    const signedDistanceQ8 = new Int16Array(cells).fill(0x7fff);
    const materialId = new Uint16Array(cells);
    // The ground first, then every authored page over it: a page is the whole
    // answer for its cells, so it replaces what the ground wrote there.
    this.ground?.write(minimum, size, solidFraction, signedDistanceQ8, materialId);
    const first = minimum.map((value) => value >> 3), last = maximumExclusive.map((value) => (value - 1) >> 3);
    for (let pz = first[2]!; pz <= last[2]!; pz += 1)
      for (let py = first[1]!; py <= last[1]!; py += 1)
        for (let px = first[0]!; px <= last[0]!; px += 1) {
          const page = this.index.page(px, py, pz);
          if (!page) continue;
          const x0 = Math.max(0, minimum[0] - 8 * px), x1 = Math.min(8, maximumExclusive[0] - 8 * px);
          const y0 = Math.max(0, minimum[1] - 8 * py), y1 = Math.min(8, maximumExclusive[1] - 8 * py);
          const z0 = Math.max(0, minimum[2] - 8 * pz), z1 = Math.min(8, maximumExclusive[2] - 8 * pz);
          for (let z = z0; z < z1; z += 1) for (let y = y0; y < y1; y += 1) {
            const row = 8 * (y + 8 * z);
            const target = 8 * px - minimum[0] + size[0] * (8 * py + y - minimum[1] + size[1] * (8 * pz + z - minimum[2]));
            for (let x = x0; x < x1; x += 1) {
              solidFraction[target + x] = page.solidFraction[row + x]!;
              signedDistanceQ8[target + x] = page.signedDistanceQ8[row + x]!;
              materialId[target + x] = page.materialId[row + x]!;
            }
          }
        }
    return { size, solidFraction, signedDistanceQ8, materialId };
  }
}

const readers = new WeakMap<SceneDescription, SceneVoxelReader>();

/** The reader for a scene document. Memoized per scene object: asking twice costs nothing. */
export function voxelReaderForScene(scene: SceneDescription): VoxelReader {
  const cached = readers.get(scene);
  // A scene object edited in place is a different document under the same
  // identity, and the reader answers for the one it was made from.
  if (cached?.current()) return cached;
  const reader = new SceneVoxelReader(scene);
  readers.set(scene, reader);
  return reader;
}
