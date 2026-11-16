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

import { SOLID_MASK_FULL, SOLID_MASK_TABLE_WORDS, SOLID_MASK_UNIT_WORDS, SolidOccupancyMask, solidOccupancyMaskVoxel, solidOccupancyMaskWGSL, type SolidOccupancyMaskWalls } from "../../core/solid-occupancy-mask";
import { solidWorldForFluidLattice } from "../../core/solid-world";
import { solidVoxelEditsForScene } from "../../core/scene-lattice";
import { sceneFluidLatticeDimensions } from "../../core/scene-lattice-dimensions";
import type { SceneDescription } from "../../core/model";

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
 * Slots: the pages partly solid now, as many again and 64 for edits. An edit past them is refused by the mask, loudly.
 */
export function brickStaticMask(scene: SceneDescription): SolidOccupancyMask | undefined {
  if (solidVoxelEditsForScene(scene).length === 0) return undefined;
  const openTop = scene.container.top === "open";
  const walls = brickStaticMaskWalls(sceneFluidLatticeDimensions(scene), openTop), [x, y, z] = walls.pages;
  const firstSlot = Math.ceil(SOLID_MASK_TABLE_WORDS / SOLID_MASK_UNIT_WORDS);
  const mask = new SolidOccupancyMask({ walls, tileKeys: 0, slots: Math.max(0, Math.min(SOLID_MASK_FULL - firstSlot, x * y * z + (openTop ? x * z : 0))),
    separateTerrain: true, halfSolid: scene.solidBoundary?.collision === "voxels" });
  mask.update(solidWorldForFluidLattice(scene));
  if (mask.empty) return undefined;
  mask.seal(Math.min(mask.layout.slots, 2 * mask.slotsHeld + 64));
  return mask;
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
  /** The voxel mask (brickStaticMask): its array<u32> binding's name and its walls. Absent, the box rule alone. */
  readonly mask?: { readonly buffer: string; readonly walls: SolidOccupancyMaskWalls };
}

export function brickStaticWGSL(src: BrickStaticSource): string {
  const [x, y, z] = src.box;
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
fn nbStaticCellKind(c:vec3i)->u32{return select(NB_STATIC_PLAIN,NB_STATIC_VOXEL,nbStaticVoxel(c));}
fn nbStaticCellOpen(c:vec3i)->f32{return select(1.0,0.0,nbStaticVoxel(c));}
${src.mask ? `fn nbStaticFaceOpen(c:vec3i,axis:u32)->f32{
 let n=c+nbStaticAxis(axis);let a=nbStaticVoxel(c);let b=nbStaticVoxel(n);
 if(a||b){return 0.5*(select(1.0,0.0,a)+select(1.0,0.0,b));}
 return 1.0;
}` : `fn nbStaticFaceOpen(c:vec3i,axis:u32)->f32{return select(0.0,1.0,nbStaticInBox(c)&&nbStaticInBox(c+nbStaticAxis(axis)));}`}
fn nbStaticFaceBoxed(c:vec3i,axis:u32)->bool{return false;}
fn nbStaticSolidAt(world:vec3f)->bool{
 let q=(world-(${src.frame}))/(${src.h});
 return nbStaticVoxel(vec3i(floor(q)));
}
`;
}

// The mask's reader (the fixed reader with the walls: lattice cells, both directory levels) behind the box rule: a
// cell outside the box never reads the mask.
function maskWGSL(mask: NonNullable<BrickStaticSource["mask"]>): string {
  return `${solidOccupancyMaskWGSL(mask.buffer, "nbStaticMaskPage", "nbStaticMaskVoxel", false, mask.walls)}
fn nbStaticVoxel(c:vec3i)->bool{if(!nbStaticInBox(c)){return true;}return nbStaticMaskVoxel(c);}`;
}

/** The Float64 mirror (the test oracle): the same five functions on the host. `frame` and `h` are numbers here; `mask`
 * is the image's words, read by the shared CPU reader (solidOccupancyMaskVoxel). */
export function brickStaticMirror(src: { box: readonly [number, number, number]; openTop: boolean; frame: readonly [number, number, number]; h: readonly [number, number, number]; mask?: ArrayLike<number> }) {
  const [X, Y, Z] = src.box, mask = src.mask;
  const inBox = (c: readonly number[]) => c[0]! >= 0 && c[1]! >= 0 && c[2]! >= 0 && c[0]! < X && c[2]! < Z && (c[1]! < Y || src.openTop);
  const voxel = (c: readonly number[]) => !inBox(c) || (mask !== undefined && solidOccupancyMaskVoxel(mask, c[0]!, c[1]!, c[2]!));
  const step = (c: readonly number[], axis: number) => c.map((v, a) => v + (a === axis ? 1 : 0));
  const open = (c: readonly number[]) => voxel(c) ? 0 : 1;
  return {
    cellKind: (c: readonly number[]) => voxel(c) ? NB_STATIC.VOXEL : NB_STATIC.PLAIN,
    cellOpen: open,
    faceOpen: (c: readonly number[], axis: number) => {
      const n = step(c, axis);
      if (mask === undefined) return inBox(c) && inBox(n) ? 1 : 0;
      return voxel(c) || voxel(n) ? 0.5 * (open(c) + open(n)) : 1;
    },
    faceBoxed: (_c: readonly number[], _axis: number) => false,
    solidAt: (w: readonly number[]) => voxel(w.map((v, a) => Math.floor(Math.fround(Math.fround(v - src.frame[a]!) / src.h[a]!)))),
  };
}
