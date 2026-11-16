/**
 * The brick engine's static solid queries (M3, lane E; m5-bodies-design-2026-11-17.md A7): the five functions lane P's
 * bodyStamp, couple and pSolid read through `PbStaticHooks` (bodies-shapes.wgsl.ts), and solidStamp itself.
 *
 * Stage 1 (this file): the static solid is the box rule (brick-abi.md §1). A cell outside the container's box
 * `[0, N)` is solid, except above `N_y` when the top is open, which is air. Padding is part of that box: the lattice is
 * the container's own cells, and the storage rounds up to pages behind it (brick-scene.ts). The old engine closes the
 * same cells with its voxel shell (solidVoxelShellForScene), which sits at -1 and N on each closed side.
 *
 * The voxel mask and the procedural boundary records come next (solidStamp over the shared geometry source). They
 * extend these functions behind the same signatures, so P's callers stay as they are. Until then a scene with voxel
 * solids, terrain or a boundary record stays a named gap (brickSceneSetup), so nothing here is render-only geometry.
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
 *
 * Compile rule (P, 2026-11-17): no dynamic vector-component write (`v[axis] += 1`) in these functions. A face's
 * neighbour is `c + nbStaticAxis(axis)`, built with selects, since Metal miscompiled that write inside a loop with
 * heavy calls. tests: brick-static.test.ts (the Float64 mirror below against the WGSL, bit for bit).
 */

export const NB_STATIC = { PLAIN: 0, BOUNDARY: 1, VOXEL: 2 } as const;

export interface BrickStaticSource {
  /** The container's box in cells (the lattice). */
  readonly box: readonly [number, number, number];
  /** The container's top: open is air above `box[1]`. */
  readonly openTop: boolean;
  /** vec3f WGSL expression: the world position (m) of cell (0,0,0)'s low corner. */
  readonly frame: string;
  /** vec3f WGSL expression: the cell size per axis (m). */
  readonly h: string;
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
fn nbStaticCellKind(c:vec3i)->u32{return select(NB_STATIC_VOXEL,NB_STATIC_PLAIN,nbStaticInBox(c));}
fn nbStaticCellOpen(c:vec3i)->f32{return select(0.0,1.0,nbStaticInBox(c));}
fn nbStaticFaceOpen(c:vec3i,axis:u32)->f32{return select(0.0,1.0,nbStaticInBox(c)&&nbStaticInBox(c+nbStaticAxis(axis)));}
fn nbStaticFaceBoxed(c:vec3i,axis:u32)->bool{return false;}
fn nbStaticSolidAt(world:vec3f)->bool{
 let q=(world-(${src.frame}))/(${src.h});
 return !nbStaticInBox(vec3i(floor(q)));
}
`;
}

/** The Float64 mirror (the test oracle): the same five functions on the host. `frame` and `h` are numbers here. */
export function brickStaticMirror(src: { box: readonly [number, number, number]; openTop: boolean; frame: readonly [number, number, number]; h: readonly [number, number, number] }) {
  const [X, Y, Z] = src.box;
  const inBox = (c: readonly number[]) => c[0]! >= 0 && c[1]! >= 0 && c[2]! >= 0 && c[0]! < X && c[2]! < Z && (c[1]! < Y || src.openTop);
  const step = (c: readonly number[], axis: number) => c.map((v, a) => v + (a === axis ? 1 : 0));
  return {
    cellKind: (c: readonly number[]) => inBox(c) ? NB_STATIC.PLAIN : NB_STATIC.VOXEL,
    cellOpen: (c: readonly number[]) => inBox(c) ? 1 : 0,
    faceOpen: (c: readonly number[], axis: number) => inBox(c) && inBox(step(c, axis)) ? 1 : 0,
    faceBoxed: (_c: readonly number[], _axis: number) => false,
    solidAt: (w: readonly number[]) => !inBox(w.map((v, a) => Math.floor(Math.fround(Math.fround(v - src.frame[a]!) / src.h[a]!)))),
  };
}
