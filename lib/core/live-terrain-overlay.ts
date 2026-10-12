import { SOLID_WORLD_TERRAIN_MATERIAL_ID, solidWorldContentStamp, solidWorldPageAt, type SolidWorld,
  type SolidWorldCoordinate, type SolidWorldPage, type SolidWorldVoxelPatch } from "./solid-world";
import { composeSolidPatch, type SolidPatchPageStore } from "./solid-patch/compose";
import { appendedSolidPatches } from "./solid-patch-list";
import type { SceneDescription } from "./model";

/** Immutable heightfield/lattice identity, excluding the mutable ordered overlay. */
export function terrainFieldStamp(scene: SceneDescription): string {
  return scene.terrain ? solidWorldContentStamp({ ...scene, solidVoxels: [] }) : "no-terrain";
}

const PAGE_CELLS = 8, PAGE_VOXELS = PAGE_CELLS ** 3, MASK_BYTES = PAGE_VOXELS / 8;
const KEY_BIAS = 0x8000, KEY_SPAN = 0x1_0000;

/** One number a page coordinate, for the 65,536 pages an axis a scene can ever address. */
export function terrainEditPageKey(x: number, y: number, z: number): number {
  const bx = x + KEY_BIAS, by = y + KEY_BIAS, bz = z + KEY_BIAS;
  if (bx < 0 || by < 0 || bz < 0 || bx >= KEY_SPAN || by >= KEY_SPAN || bz >= KEY_SPAN) {
    throw new RangeError(`Terrain edit page ${x},${y},${z} is outside the addressable lattice`);
  }
  return bx + KEY_SPAN * (by + KEY_SPAN * bz);
}

/** The ground a terrain scene draws analytically: the render heightfield, as its shader samples it. */
export interface TerrainEditGround {
  readonly origin_m: readonly [number, number];
  readonly cellSize_m: readonly [number, number];
  readonly dimensions: readonly [number, number];
  readonly heights_m: Float32Array;
}

/** The solid lattice the pages are addressed on. */
export interface TerrainEditLattice {
  readonly origin_m: readonly [number, number, number];
  readonly cellSize_m: readonly [number, number, number];
}

/** Cells of one page that some clear covers, whether or not they were solid. */
interface TerrainClearMask {
  readonly coordinate: SolidWorldCoordinate;
  readonly bits: Uint8Array;
  /** Bits set. They are only ever added, so this names the mask's content. */
  count: number;
  touched: boolean;
  /** The selection that last found a world page here. */
  seen: number;
  /** What `drawn` was made from: the world's page (none where the world holds no page) and `count`. */
  source?: SolidWorldPage;
  sourceCount: number;
  drawn?: SolidWorldPage;
}

const EMPTY_FRACTION = new Uint8Array(PAGE_VOXELS);
const EMPTY_DISTANCE = new Int16Array(PAGE_VOXELS).fill(0x7fff);
const EMPTY_MATERIAL = new Uint16Array(PAGE_VOXELS);

/** Whether a page holds anything the analytic ground does not: an authored cell or a repainted one. */
function pageHoldsEdits(page: SolidWorldPage): boolean {
  if (page.authored) for (let byte = 0; byte < MASK_BYTES; byte += 1) if (page.authored[byte] !== 0) return true;
  const { solidFraction, materialId } = page;
  for (let voxel = 0; voxel < PAGE_VOXELS; voxel += 1) {
    if (solidFraction[voxel] !== 0 && materialId[voxel] !== SOLID_WORLD_TERRAIN_MATERIAL_ID) return true;
  }
  return false;
}

/**
 * The faces of a page (bit `2 axis + side`, side 1 the upper) on which an
 * authored cell is partly covered. Only such a cell reads its neighbours: a
 * whole one is its own box, and an empty one is nothing.
 */
function partialAuthoredFaces(page: SolidWorldPage): number {
  const { authored, solidFraction } = page;
  if (!authored) return 0;
  let faces = 0;
  for (let voxel = 0; voxel < PAGE_VOXELS; voxel += 1) {
    const fraction = solidFraction[voxel]!;
    if (fraction === 0 || fraction === 255 || (authored[voxel >> 3]! >> (voxel & 7) & 1) === 0) continue;
    const x = voxel & 7, y = voxel >> 3 & 7, z = voxel >> 6;
    if (x === 0) faces |= 1; else if (x === PAGE_CELLS - 1) faces |= 2;
    if (y === 0) faces |= 4; else if (y === PAGE_CELLS - 1) faces |= 8;
    if (z === 0) faces |= 16; else if (z === PAGE_CELLS - 1) faces |= 32;
  }
  return faces;
}

/**
 * The pages a terrain scene draws: its edits, and nothing of its ground.
 *
 * The ground is drawn from the render heightfield wherever no authored cell is
 * in reach, so only pages an edit shaped or repainted are uploaded, and the
 * ground pages a partly covered authored cell is shaded against. The world is
 * the same shape: it holds the pages patches wrote and its ground is analytic,
 * so this reads those pages and never the terrain.
 *
 * A cell is authored where `SolidWorldPage.authored` says so, and also
 * wherever a clear covers it. The world keeps a page a clear emptied for as
 * long as its own ground stands there, but the heightfield drawn here is a
 * different sampling of the same terrain and rises into pages the world's
 * ground does not: there a clear finds no page to write, or leaves one with no
 * solid that the world then drops, and the ground would be drawn back into the
 * hole. Those cells are recovered here by composing every clear against a page
 * that is solid throughout, which marks exactly the cells the clear covers. A
 * cleared page the world does not hold is drawn as an empty page that carries
 * those bits. Above the ground none of this is kept: an unauthored cell there
 * is already empty.
 *
 * Pages handed out keep their identity for as long as their content does, so
 * `WebgpuSolidWorldPageWriter` uploads only what an edit changed.
 */
export class TerrainEditPages {
  private patches: readonly SolidWorldVoxelPatch[] = [];
  private readonly masks = new Map<number, TerrainClearMask>();
  /** The highest ground under each page column, by `terrainEditPageKey(x, 0, z)`. */
  private readonly ceilings = new Map<number, number>();
  private readonly touched: TerrainClearMask[] = [];
  private readonly edited = new WeakMap<SolidWorldPage, boolean>();
  private readonly faces = new WeakMap<SolidWorldPage, number>();
  private generation = 0;
  private selectedWorld?: SolidWorld;
  private selected: readonly SolidWorldPage[] = [];
  private readonly solidThroughout = new Uint8Array(PAGE_VOXELS);
  private readonly scratchDistance = new Int16Array(PAGE_VOXELS);
  private readonly scratchMaterial = new Uint16Array(PAGE_VOXELS);
  private readonly store: SolidPatchPageStore = {
    page: (coordinate) => {
      // Above the ground an unauthored cell is as empty as a cleared one.
      if (!this.groundReaches(coordinate)) return undefined;
      const key = terrainEditPageKey(coordinate[0], coordinate[1], coordinate[2]);
      let mask = this.masks.get(key);
      if (!mask) {
        mask = { coordinate, bits: new Uint8Array(MASK_BYTES), count: 0, touched: false, seen: 0, sourceCount: 0 };
        this.masks.set(key, mask);
      }
      if (!mask.touched) { mask.touched = true; this.touched.push(mask); }
      this.solidThroughout.fill(255);
      return { coordinate, solidFraction: this.solidThroughout, signedDistanceQ8: this.scratchDistance,
        materialId: this.scratchMaterial, revision: 0, authored: mask.bits };
    },
  };

  constructor(private readonly ground: TerrainEditGround, private readonly lattice: TerrainEditLattice) {}

  /** Bring the clear masks to this patch list: the appended clears, or all of them after an undo. */
  private follow(patches: readonly SolidWorldVoxelPatch[]): void {
    let appended = appendedSolidPatches(this.patches, patches);
    if (!appended) { this.masks.clear(); appended = patches; }
    this.patches = patches;
    for (const patch of appended) {
      if (patch.operation !== "clear") continue;
      composeSolidPatch(patch, this.store);
      for (const mask of this.touched) {
        let count = 0;
        for (let byte = 0; byte < MASK_BYTES; byte += 1) {
          let bits = mask.bits[byte]!;
          for (; bits !== 0; bits &= bits - 1) count += 1;
        }
        mask.count = count; mask.touched = false;
      }
      this.touched.length = 0;
    }
  }

  /** Whether the heightfield, as the shader interpolates it, rises above the page's floor anywhere under it. */
  private groundReaches(coordinate: SolidWorldCoordinate): boolean {
    const { ground, lattice } = this, [nx, nz] = ground.dimensions;
    const column = terrainEditPageKey(coordinate[0], 0, coordinate[2]);
    let ceiling = this.ceilings.get(column);
    if (ceiling === undefined) {
      const edgeX = PAGE_CELLS * lattice.cellSize_m[0], edgeZ = PAGE_CELLS * lattice.cellSize_m[2];
      const minimumX = lattice.origin_m[0] + coordinate[0] * edgeX, minimumZ = lattice.origin_m[2] + coordinate[2] * edgeZ;
      // A bilinear sample reads the centre samples either side of it.
      const x0 = Math.floor((minimumX - ground.origin_m[0]) / ground.cellSize_m[0] - 0.5);
      const x1 = Math.ceil((minimumX + edgeX - ground.origin_m[0]) / ground.cellSize_m[0] + 0.5);
      const z0 = Math.floor((minimumZ - ground.origin_m[1]) / ground.cellSize_m[1] - 0.5);
      const z1 = Math.ceil((minimumZ + edgeZ - ground.origin_m[1]) / ground.cellSize_m[1] + 0.5);
      ceiling = Number.NEGATIVE_INFINITY;
      for (let z = Math.max(0, z0); z <= Math.min(nz - 1, z1); z += 1) {
        for (let x = Math.max(0, x0); x <= Math.min(nx - 1, x1); x += 1) {
          ceiling = Math.max(ceiling, ground.heights_m[x + nx * z]!);
        }
      }
      this.ceilings.set(column, ceiling);
    }
    return ceiling > lattice.origin_m[1] + coordinate[1] * PAGE_CELLS * lattice.cellSize_m[1];
  }

  /** The page drawn where a clear mask lies: the world's own when it already says as much. */
  private drawn(mask: TerrainClearMask, page: SolidWorldPage | undefined): SolidWorldPage {
    if (mask.drawn && mask.source === page && mask.sourceCount === mask.count) return mask.drawn;
    let drawn: SolidWorldPage;
    if (!page) {
      drawn = { coordinate: mask.coordinate, solidFraction: EMPTY_FRACTION, signedDistanceQ8: EMPTY_DISTANCE,
        materialId: EMPTY_MATERIAL, revision: 0, authored: mask.bits.slice() };
    } else {
      const authored = page.authored;
      let covered = authored !== undefined;
      for (let byte = 0; covered && byte < MASK_BYTES; byte += 1) {
        covered = (mask.bits[byte]! & ~authored![byte]!) === 0;
      }
      if (covered) drawn = page;
      else {
        const merged = mask.bits.slice();
        if (authored) for (let byte = 0; byte < MASK_BYTES; byte += 1) merged[byte]! |= authored[byte]!;
        drawn = { ...page, authored: merged };
      }
    }
    mask.source = page; mask.sourceCount = mask.count; mask.drawn = drawn;
    return drawn;
  }

  /** The edit pages of a world, one a coordinate, in no particular order. */
  select(world: SolidWorld): readonly SolidWorldPage[] {
    if (world === this.selectedWorld) return this.selected;
    this.follow(world.patches);
    this.generation += 1;
    const { generation, masks, edited } = this, pages: SolidWorldPage[] = [], masked = masks.size > 0;
    for (const page of world.pages) {
      const mask = masked ? masks.get(terrainEditPageKey(page.coordinate[0], page.coordinate[1], page.coordinate[2])) : undefined;
      if (mask && mask.count > 0) {
        mask.seen = generation;
        pages.push(this.drawn(mask, page));
        continue;
      }
      let holds = edited.get(page);
      if (holds === undefined) { holds = pageHoldsEdits(page); edited.set(page, holds); }
      if (holds) pages.push(page);
    }
    if (masked) for (const mask of masks.values()) {
      if (mask.seen !== generation && mask.count > 0) pages.push(this.drawn(mask, undefined));
    }
    // A partly covered authored cell is shaded from its neighbours' distances,
    // so the ground page across a face such a cell lies on is drawn with it.
    const edits = pages.length, held = new Set<number>();
    for (let index = 0; index < edits; index += 1) {
      const [x, y, z] = pages[index]!.coordinate;
      held.add(terrainEditPageKey(x, y, z));
    }
    for (let index = 0; index < edits; index += 1) {
      const page = pages[index]!;
      let faces = this.faces.get(page);
      if (faces === undefined) { faces = partialAuthoredFaces(page); this.faces.set(page, faces); }
      for (let face = 0; faces !== 0 && face < 6; face += 1) {
        if ((faces >> face & 1) === 0) continue;
        const neighbour: [number, number, number] = [page.coordinate[0], page.coordinate[1], page.coordinate[2]];
        neighbour[face >> 1]! += (face & 1) === 0 ? -1 : 1;
        const key = terrainEditPageKey(neighbour[0], neighbour[1], neighbour[2]);
        if (held.has(key)) continue;
        held.add(key);
        // The neighbour a patch wrote, or the ground's own page there, baked for this and kept.
        const beside = solidWorldPageAt(world, neighbour);
        if (beside) pages.push(beside);
      }
    }
    this.selectedWorld = world; this.selected = pages;
    return pages;
  }
}

/**
 * The edit pages a surface can cross. A page authored throughout and uniformly
 * full, or uniformly empty, among six neighbours that are the same is the
 * inside of a fill or of a dig, and no surface is drawn in it.
 */
export function terrainEditSurfacePages(pages: readonly SolidWorldPage[]): SolidWorldCoordinate[] {
  const uniform = new Map<number, number>();
  const stateOf = (page: SolidWorldPage): number => {
    const { authored, solidFraction } = page;
    if (!authored) return 0;
    for (let byte = 0; byte < MASK_BYTES; byte += 1) if (authored[byte] !== 255) return 0;
    const first = solidFraction[0]!;
    if (first !== 0 && first !== 255) return 0;
    for (let voxel = 1; voxel < PAGE_VOXELS; voxel += 1) if (solidFraction[voxel] !== first) return 0;
    return first === 255 ? 1 : 2;
  };
  for (const page of pages) {
    const state = stateOf(page);
    if (state !== 0) uniform.set(terrainEditPageKey(page.coordinate[0], page.coordinate[1], page.coordinate[2]), state);
  }
  const surface: SolidWorldCoordinate[] = [];
  for (const page of pages) {
    const [x, y, z] = page.coordinate;
    // A page only a paint touched moves no surface.
    if (!page.authored || page.authored.every((byte) => byte === 0)) continue;
    const state = uniform.get(terrainEditPageKey(x, y, z));
    if (state !== undefined
      && uniform.get(terrainEditPageKey(x - 1, y, z)) === state && uniform.get(terrainEditPageKey(x + 1, y, z)) === state
      && uniform.get(terrainEditPageKey(x, y - 1, z)) === state && uniform.get(terrainEditPageKey(x, y + 1, z)) === state
      && uniform.get(terrainEditPageKey(x, y, z - 1)) === state && uniform.get(terrainEditPageKey(x, y, z + 1)) === state) continue;
    surface.push(page.coordinate);
  }
  return surface;
}
