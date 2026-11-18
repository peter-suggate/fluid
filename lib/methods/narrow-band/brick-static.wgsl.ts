/**
 * The brick engine's static solid queries (M3, lane E; m5-bodies-design-2026-11-17.md A7): the five functions lane P's
 * bodyStamp, couple and pSolid read through `PbStaticHooks` (bodies-shapes.wgsl.ts), and solidStamp itself.
 *
 * Stage 1 (this file): the static solid is the box rule (brick-abi.md §1). A cell outside the container's box
 * `[0, N)` is solid, except above `N_y` when the top is open, which is air. Padding is part of that box: the lattice is
 * the container's own cells, and the storage rounds up to pages behind it (brick-scene.ts). The old engine closes the
 * same cells with its voxel shell (solidVoxelShellForScene), which sits at -1 and N on each closed side.
 *
 * Stage 2a: the static voxels. A scene with solid voxels beyond the shell reads them from the shared geometry path's
 * own image: the SolidOccupancyMask (lib/core/solid-occupancy-mask.ts) built from solidWorldForFluidLattice, as the
 * old engine builds it (webgpu-uniform-reference.ts), through solidOccupancyMaskWGSL's reader. The source stays the
 * scene's SolidWorld, which hit testing and editing already address; an edit is the mask's `update` (it names the
 * cells it moved), so nothing here is render-only geometry or a second copy. Procedural boundary records and terrain
 * stay a named gap (brickSceneSetup), and with them nbStaticFaceBoxed stays false and no cell is BOUNDARY.
 *
 * Semantics, as the old engine's functions with no body (uniform-mixed-solid.wgsl.ts at c95f17aa):
 * - nbStaticSolidAt(world): umSolidVoxelAtWorld || umSolidTerrainAtWorld. True outside the box.
 * - nbStaticCellOpen(c): umCellOpenGeometry with no body. 1 in the box, 0 outside.
 * - nbStaticCellKind(c): PLAIN in the box, VOXEL outside (a box wall acts as a voxel with no record). There is no
 *   BOUNDARY cell yet.
 * - nbStaticFaceOpen(c, axis): umPressureFaceV with no body, for the +axis face of c. 1 between two box cells, else 0.
 *   P's pbStampFace asks it only of a face that is not a voxel-wall face (or a boxed one), so the 0 is never read
 *   there; it is the box rule's closed face for any other caller.
 * - nbStaticFaceBoxed(c, axis): umBoundaryTouchesBox(q, 1). False: no procedural boundary yet.
 * With the voxel mask (umSolidEnabled true, no body, no terrain, no boundary record):
 * - a cell outside the box or a mask voxel is VOXEL; its open fraction is 0, any other cell's is 1 (umUnitBoxOpen
 *   with no terrain is 1);
 * - a face with a VOXEL side is umPressureFaceV's voxel-wall face, 0.5 (open(c) + open(c + e_axis)): 0.5 beside one
 *   voxel (the box walls included, where the old engine's shell voxels sit), 0 between two; any other face is 1.
 *
 * Compile rule (P, 2026-11-17): no dynamic vector-component write (`v[axis] += 1`) in these functions. A face's
 * neighbour is `c + nbStaticAxis(axis)`, built with selects, since Metal miscompiled that write inside a loop with
 * heavy calls. tests: brick-static.test.ts (the Float64 mirror below against the WGSL, bit for bit).
 */

export const NB_STATIC = { PLAIN: 0, BOUNDARY: 1, VOXEL: 2 } as const;

import { SOLID_MASK_FULL, SOLID_MASK_PAGE_CELLS, SOLID_MASK_TABLE_WORDS, SOLID_MASK_UNIT_WORDS, SolidOccupancyMask, solidOccupancyMaskVoxel, solidOccupancyMaskWGSL, type SolidOccupancyMaskWalls } from "../../core/solid-occupancy-mask";
import { solidWorldForFluidLattice } from "../../core/solid-world";
import { solidVoxelEditsForScene } from "../../core/scene-lattice";
import { sceneFluidLatticeDimensions } from "../../core/scene-lattice-dimensions";
import type { SceneDescription } from "../../core/model";
import { brickTerrainCutBricks, brickTerrainMirror, brickTerrainWGSL, type BrickTerrainColumns } from "./brick-terrain.wgsl";
import { brickBoundaryCutBricks, brickBoundaryMirror, brickBoundaryWGSL } from "./brick-boundary.wgsl";

/** The mask's walls, and where the terrain's columns ride in the same buffer after the mask's words (brick-terrain.wgsl.ts):
 * every static consumer (solidStamp, the static need, pSolid, pSeed, the body stage) takes the walls, so the terrain
 * reaches each with them. */
export type BrickStaticWalls = SolidOccupancyMaskWalls & {
  readonly terrain?: { readonly base: number; readonly columns: readonly [number, number] };
  /** The procedural boundary's image (brick-boundary.wgsl.ts), after the mask's words and the terrain's columns. */
  readonly boundary?: { readonly base: number };
};

const MASK_CELL = 32;

/**
 * The mask's walls for a lattice of `paged` cells (whole pages): the old engine's tank (uniformSolidMaskPages), the
 * box at window page 0, walled outside it (policy 1), or open above its floor under an open top (2). The shell's
 * voxels lie in wall pages and hold no slot. The hooks never read a wall: the box rule answers outside the box first.
 */
export function brickStaticMaskWalls(paged: readonly number[], openTop: boolean): SolidOccupancyMaskWalls {
  return { policy: openTop ? 2 : 1, origin: [0, 0, 0], pages: paged.map(n => Math.ceil(n / MASK_CELL)) as [number, number, number] };
}

/**
 * The scene's static voxels as the shared SolidOccupancyMask image, or undefined when no cell of the box is solid
 * (the box rule alone: no binding, no memory). Built as the old engine builds its own (webgpu-uniform-reference.ts):
 * solidWorldForFluidLattice, terrain separate, a procedural solid with voxel collision taken from half solid. Its
 * units are the two fixed tables (2 MB) and one 4 KB unit a partly solid page: memory follows the solid surface.
 * Slots: the pages partly solid now, as many again and 64 for edits. An edit past them is refused by the mask, loudly
 * (and first by the live preflight, BrickNBFlipSolver.validateLiveSolidEdit, on a clone).
 * `always` (brickSceneSetup): a mask even with no solid cell (sealed at 64 slots, no world folded), so a scene's first
 * voxel stroke is a live edit; the engine defers its static stages until then (BrickEngineOptions.staticsDeferred).
 */
export function brickStaticMask(scene: SceneDescription, options: { readonly always?: boolean; readonly padding?: boolean } = {}): SolidOccupancyMask | undefined {
  // `padding`: the container is not whole bricks, so the lattice's padding inside the engine's box is solid voxels.
  // A spherical container's shell (sphericalSolidVoxelShell) lies inside the box, so it is static voxels too: the old
  // engine's mask folds the same world, shell and all (solidVoxelEditsForScene leaves the shell out, as a box's lies
  // past the box, where the box rule closes it).
  const sphere = scene.container.shape === "sphere" && scene.container.walls !== "none" && scene.container.boundary !== "free-space";
  const voxels = solidVoxelEditsForScene(scene).length > 0 || options.padding === true || sphere;
  if (!voxels && !options.always) return undefined;
  const openTop = scene.container.top === "open";
  const walls = brickStaticMaskWalls(sceneFluidLatticeDimensions(scene), openTop), [x, y, z] = walls.pages;
  const firstSlot = Math.ceil(SOLID_MASK_TABLE_WORDS / SOLID_MASK_UNIT_WORDS);
  const mask = new SolidOccupancyMask({ walls, tileKeys: 0, slots: Math.max(0, Math.min(SOLID_MASK_FULL - firstSlot, x * y * z + (openTop ? x * z : 0))),
    separateTerrain: true, halfSolid: scene.solidBoundary?.collision === "voxels" });
  // With no voxels the mask folds no world: the first live stroke is its first fold, which makes its child tables.
  if (voxels) mask.update(solidWorldForFluidLattice(scene));
  if (mask.empty && !options.always) return undefined;
  mask.seal(Math.min(mask.layout.slots, 2 * mask.slotsHeld + 64));
  return mask;
}

/** K slots a scene keeps for live edits beyond the bricks its voxels cut at the build (BRICK_STATIC_EDIT_K). */
export const BRICK_STATIC_EDIT_K = 64;

/**
 * The bricks a mask cuts (solidStamp's rule): a brick holding a solid cell, or one whose cell just past its lower face
 * on an axis is solid. Read cell-exactly from each solid 32^3 page's rows (a wholly solid page: its 64 bricks and the
 * bricks past its upper faces). A brick wholly inside a solid counts, though it is resident only when promoted.
 */
export function brickStaticCutBricks(mask: Pick<SolidOccupancyMask, "eachSolidPage" | "words">, ground?: { readonly terrain?: BrickTerrainColumns; readonly boundary?: Uint32Array; readonly box: readonly number[] }): number {
  const bricks = new Set<number>(), add = (x: number, y: number, z: number) => bricks.add(x + 4096 * (y + 4096 * z));
  if (ground?.terrain) brickTerrainCutBricks(ground.terrain, ground.box, add);
  if (ground?.boundary) brickBoundaryCutBricks(ground.boundary, ground.box, add);
  const words = mask.words, P = SOLID_MASK_PAGE_CELLS;
  mask.eachSolidPage((px, py, pz, word) => {
    const code = word & 0xffff;
    for (let z = 0; z < P; z++) for (let y = 0; y < P; y++) {
      const bits = code === SOLID_MASK_FULL ? 0xffffffff : words[(code << 10) + y + P * z]! >>> 0;
      if (!bits) continue;
      const by = (P * py + y) >> 3, bz = (P * pz + z) >> 3;
      for (let b = 0; b < 4; b++) {
        const byte = (bits >>> (8 * b)) & 0xff;
        if (!byte) continue;
        const bx = 4 * px + b;
        add(bx, by, bz);
        if (byte & 0x80) add(bx + 1, by, bz);
        if ((y & 7) === 7) add(bx, by + 1, bz);
        if ((z & 7) === 7) add(bx, by, bz + 1);
      }
    }
  });
  return bricks.size;
}

export interface BrickStaticSource {
  /** The container's box in cells (the lattice). */
  readonly box: readonly [number, number, number];
  /** The container's top: open is air above `box[1]`. */
  readonly openTop: boolean;
  /** vec3f WGSL expression: the world position (m) of cell (0,0,0)'s low corner. */
  readonly frame: string;
  /** vec3f WGSL expression: the cell size per axis (m). */
  readonly h: string;
  /** The voxel mask (brickStaticMask): its array<u32> binding's name and its walls (with the terrain's columns, when the
   * scene has terrain: they follow the mask's words in the same binding). Absent, the box rule alone. */
  readonly mask?: { readonly buffer: string; readonly walls: BrickStaticWalls };
}

export function brickStaticWGSL(src: BrickStaticSource): string {
  const [x, y, z] = src.box, terrain = src.mask?.walls.terrain, boundary = src.mask?.walls.boundary;
  for (const n of src.box) if (!Number.isInteger(n) || n < 1) throw new Error(`Brick engine statics: box ${src.box.join("x")}`);
  return /* wgsl */`
const NB_STATIC_PLAIN:u32=${NB_STATIC.PLAIN}u;const NB_STATIC_BOUNDARY:u32=${NB_STATIC.BOUNDARY}u;const NB_STATIC_VOXEL:u32=${NB_STATIC.VOXEL}u;
const NB_STATIC_BOX:vec3i=vec3i(${x},${y},${z});
const NB_STATIC_OPEN_TOP:bool=${src.openTop};
fn nbStaticAxis(axis:u32)->vec3i{return vec3i(select(0,1,axis==0u),select(0,1,axis==1u),select(0,1,axis==2u));}
// The box rule: inside [0, N) on every axis, or above N_y under an open top.
fn nbStaticInBox(c:vec3i)->bool{
 let side=all(c>=vec3i(0))&&c.x<NB_STATIC_BOX.x&&c.z<NB_STATIC_BOX.z;
 return side&&(c.y<NB_STATIC_BOX.y||NB_STATIC_OPEN_TOP);
}
${src.mask ? maskWGSL(src.mask) : `fn nbStaticVoxel(c:vec3i)->bool{return !nbStaticInBox(c);}`}
${terrain ? brickTerrainWGSL({ buffer: src.mask!.buffer, base: terrain.base, columns: terrain.columns }) : ""}
${boundary ? brickBoundaryWGSL({ buffer: src.mask!.buffer, base: boundary.base }) : ""}
const NB_STATIC_TERRAIN:bool=${terrain !== undefined};
const NB_STATIC_FIELD:bool=${boundary !== undefined};
// The unit box's terrain lane (umUnitBoxOpen's x): 1 with no terrain.
fn nbStaticTerrainOpen(q:vec3f)->f32{${terrain ? "return nbTerrainBoxOpen(q,vec3f(1));" : "return 1.0;"}}
${boundary ? `// A record cell is BOUNDARY (the old engine's field cell), else a voxel VOXEL, else PLAIN.
fn nbStaticCellKind(c:vec3i)->u32{if(nbBoundaryRecord(c)!=0u){return NB_STATIC_BOUNDARY;}return select(NB_STATIC_PLAIN,NB_STATIC_VOXEL,nbStaticVoxel(c));}
// umCellOpenGeometry with no body: a record cell min(the field's box, the terrain's), a voxel 0, else the terrain's.
fn nbStaticCellOpen(c:vec3i)->f32{
 let record=nbBoundaryRecord(c);
 if(record==0u&&nbStaticVoxel(c)){return 0.0;}
 let q=vec3f(c)+0.5;let terrain=1.0-nbStaticTerrainOpen(q);
 if(record!=0u){return min(nbBoundaryBoxOpen(q,vec3f(1)),1.0-terrain);}
 return clamp(1.0-terrain,0.0,1.0);
}
// umPressureFaceV with no body: a face whose unit box a record reaches (boxed) is min(the field's box, the terrain's);
// else a side that is a voxel with no record gives the voxel-wall face, 0.5 (open + open); else the terrain's box.
fn nbStaticFaceOpen(c:vec3i,axis:u32)->f32{
 let n=c+nbStaticAxis(axis);let q=vec3f(c)+0.5+0.5*vec3f(nbStaticAxis(axis));
 if(nbBoundaryTouchesBox(q,vec3f(1))){return min(nbBoundaryBoxOpen(q,vec3f(1)),nbStaticTerrainOpen(q));}
 if((nbStaticVoxel(c)&&nbBoundaryRecord(c)==0u)||(nbStaticVoxel(n)&&nbBoundaryRecord(n)==0u)){return 0.5*(nbStaticCellOpen(c)+nbStaticCellOpen(n));}
 return nbStaticTerrainOpen(q);
}
fn nbStaticFaceBoxed(c:vec3i,axis:u32)->bool{return nbBoundaryTouchesBox(vec3f(c)+0.5+0.5*vec3f(nbStaticAxis(axis)),vec3f(1));}
// umSolidVoxelAtWorld || umSolidTerrainAtWorld: a record cell by the field's sign.
fn nbStaticSolidAt(world:vec3f)->bool{
 let q=(world-(${src.frame}))/(${src.h});
 return nbBoundaryBlocked(q)${terrain ? "||nbTerrainSolid(q)" : ""};
}
` : `fn nbStaticCellKind(c:vec3i)->u32{return select(NB_STATIC_PLAIN,NB_STATIC_VOXEL,nbStaticVoxel(c));}
${terrain ? `// umCellOpenGeometry with no body or record: a voxel 0, else the unit box's terrain lane (umUnitBoxOpen).
fn nbStaticCellOpen(c:vec3i)->f32{
 if(nbStaticVoxel(c)){return 0.0;}
 let terrain=1.0-nbTerrainBoxOpen(vec3f(c)+0.5,vec3f(1));
 return clamp(1.0-terrain,0.0,1.0);
}` : `fn nbStaticCellOpen(c:vec3i)->f32{return select(1.0,0.0,nbStaticVoxel(c));}`}
${src.mask ? `fn nbStaticFaceOpen(c:vec3i,axis:u32)->f32{
 let n=c+nbStaticAxis(axis);let a=nbStaticVoxel(c);let b=nbStaticVoxel(n);
 if(a||b){return 0.5*(nbStaticCellOpen(c)+nbStaticCellOpen(n));}
 ${terrain ? "// umPressureFaceV with no body: the unit box about the face centre (umUnitBoxOpen's terrain lane).\n return nbTerrainBoxOpen(vec3f(c)+0.5+0.5*vec3f(nbStaticAxis(axis)),vec3f(1));" : "return 1.0;"}
}` : `fn nbStaticFaceOpen(c:vec3i,axis:u32)->f32{return select(0.0,1.0,nbStaticInBox(c)&&nbStaticInBox(c+nbStaticAxis(axis)));}`}
fn nbStaticFaceBoxed(c:vec3i,axis:u32)->bool{return false;}
fn nbStaticSolidAt(world:vec3f)->bool{
 let q=(world-(${src.frame}))/(${src.h});
 return nbStaticVoxel(vec3i(floor(q)))${terrain ? "||nbTerrainSolid(q)" : ""};
}
`}`;
}

// The mask's reader (the fixed reader with the walls: lattice cells, both directory levels) behind the box rule: a
// cell outside the box never reads the mask.
function maskWGSL(mask: NonNullable<BrickStaticSource["mask"]>): string {
  return `${solidOccupancyMaskWGSL(mask.buffer, "nbStaticMaskPage", "nbStaticMaskVoxel", false, mask.walls)}
fn nbStaticVoxel(c:vec3i)->bool{if(!nbStaticInBox(c)){return true;}return nbStaticMaskVoxel(c);}`;
}

/** The Float64 mirror (the test oracle): the same five functions on the host. `frame` and `h` are numbers here; `mask`
 * is the image's words, read by the shared CPU reader (solidOccupancyMaskVoxel). */
export function brickStaticMirror(src: { box: readonly [number, number, number]; openTop: boolean; frame: readonly [number, number, number]; h: readonly [number, number, number]; mask?: ArrayLike<number>; terrain?: BrickTerrainColumns; boundary?: Uint32Array }) {
  const [X, Y, Z] = src.box, mask = src.mask, ground = src.terrain ? brickTerrainMirror(src.terrain) : undefined;
  const inBox = (c: readonly number[]) => c[0]! >= 0 && c[1]! >= 0 && c[2]! >= 0 && c[0]! < X && c[2]! < Z && (c[1]! < Y || src.openTop);
  const voxel = (c: readonly number[]) => !inBox(c) || (mask !== undefined && solidOccupancyMaskVoxel(mask, c[0]!, c[1]!, c[2]!));
  const field = src.boundary ? brickBoundaryMirror(src.boundary, inBox, voxel) : undefined;
  const step = (c: readonly number[], axis: number) => c.map((v, a) => v + (a === axis ? 1 : 0));
  const terrainOpen = (q: readonly number[]) => ground ? ground.boxOpen(q, [1, 1, 1]) : 1;
  const open = (c: readonly number[]) => {
    const record = field ? field.record(c) : 0, q = c.map(v => v + 0.5);
    if (record === 0 && voxel(c)) return 0;
    if (record !== 0) return Math.min(field!.boxOpen(q, [1, 1, 1]), terrainOpen(q));
    return ground ? Math.min(1, Math.max(0, terrainOpen(q))) : 1;
  };
  const faceQ = (c: readonly number[], axis: number) => c.map((v, a) => v + 0.5 + (a === axis ? 0.5 : 0));
  return {
    cellKind: (c: readonly number[]) => field && field.record(c) !== 0 ? NB_STATIC.BOUNDARY : voxel(c) ? NB_STATIC.VOXEL : NB_STATIC.PLAIN,
    cellOpen: open,
    faceOpen: (c: readonly number[], axis: number) => {
      const n = step(c, axis), q = faceQ(c, axis);
      if (mask === undefined) return inBox(c) && inBox(n) ? 1 : 0;
      if (field) {
        if (field.touchesBox(q, [1, 1, 1])) return Math.min(field.boxOpen(q, [1, 1, 1]), terrainOpen(q));
        if ((voxel(c) && field.record(c) === 0) || (voxel(n) && field.record(n) === 0)) return 0.5 * (open(c) + open(n));
        return terrainOpen(q);
      }
      if (voxel(c) || voxel(n)) return 0.5 * (open(c) + open(n));
      return terrainOpen(q);
    },
    faceBoxed: (c: readonly number[], axis: number) => field !== undefined && field.touchesBox(faceQ(c, axis), [1, 1, 1]),
    solidAt: (w: readonly number[]) => {
      const q = w.map((v, a) => Math.fround(Math.fround(v - src.frame[a]!) / src.h[a]!));
      return (field ? field.blocked(q) : voxel(q.map(Math.floor))) || (ground !== undefined && ground.solid(q));
    },
  };
}
