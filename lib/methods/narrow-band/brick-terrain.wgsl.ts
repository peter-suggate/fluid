/**
 * The brick engine's terrain (M3, the static solid's ground field): the old engine's heightfield, ported verbatim
 * from uniform-terrain-sampling.wgsl.ts (umTerrainSurface), uniform-mixed-solid.wgsl.ts (umTerrainBoundary,
 * umTerrainBoxOpen, umUnitBoxOpen's terrain lane, umSolidTerrainAtWorld) and uniform-particle-contact.wgsl.ts
 * (nbTerrainContact), for a closed scene: the frame page is 0 and there is no analytic open-world ground.
 *
 * The source is the old engine's own: terrainColumnHeights over the container's columns at cell centres, in cells
 * (metres / the cell height), row-major x + nx z (webgpu-uniform-reference.ts: the terrain texture's texels). The
 * columns ride in the static `solid` buffer after the voxel mask's words (BrickEngine: no extra binding where the mask
 * is bound), read as f32 bits; a read past the columns takes the edge column, as the texture's clamped load.
 *
 * Units: lattice cells. The brick lattice is cubic (brickSceneSetup), so the old engine's metre scalings (h.y on the
 * boundary's value and gradient, params.hDt in the contact) cancel exactly; the forms below are those with h = 1.
 *
 * Not ported, and refused explicitly where a scene would need them (brickSceneSetup, the live edit's preflight): a
 * sculpt cut (collisionSculptSample in umTerrainBoundary), an open world's analytic ground, and a rigid body's terrain
 * contact (the body stages read the voxel hooks alone).
 */
import { geometricPlaneBoxWGSL } from "../../core/geometric-plane-box.wgsl";
import { sceneLatticeDimensions } from "../../core/scene-lattice-dimensions";
import { sceneHasTerrain, terrainColumnHeights } from "../../core/terrain";
import type { SceneDescription } from "../../core/model";

/** The terrain's columns in cells (the container's nx x nz, row-major x + nx z). */
export interface BrickTerrainColumns {
  readonly columns: readonly [number, number];
  readonly heights: Float32Array;
}

/** Where the kernels read the columns: an array<u32> binding and the word its first column is at. */
export interface BrickTerrainSource {
  readonly buffer: string;
  readonly base: number;
  readonly columns: readonly [number, number];
}

/** geometricPlaneBoxFraction as nbTerrainPlaneBoxFraction (the old engine's umTerrain copy), its one dynamic vector
 * component write (spans[dimensions]) as a select: P's compile rule for the brick kernels (Metal miscompiled that write
 * in a loop beside heavy calls). The same values. */
function terrainPlaneBoxWGSL(): string {
  const text = geometricPlaneBoxWGSL.replaceAll("geometric", "nbTerrain"), write = "spans[dimensions]=value;";
  if (text.split(write).length !== 2) throw new Error("Brick engine terrain: geometricPlaneBoxWGSL changed shape");
  return text.replace(write, "spans=select(spans,vec3f(value),vec3u(dimensions)==vec3u(0u,1u,2u));");
}

/** The scene's terrain columns as the old engine bakes them, or undefined with no terrain. */
export function brickTerrainColumns(scene: SceneDescription): BrickTerrainColumns | undefined {
  if (!sceneHasTerrain(scene)) return undefined;
  const [nx, ny, nz] = sceneLatticeDimensions(scene, Number.MAX_SAFE_INTEGER);
  const metres = terrainColumnHeights(scene, nx, nz), cellHeight = scene.container.height_m / ny;
  const heights = Float32Array.from(metres, height => height / cellHeight);
  for (const v of heights) if (!Number.isFinite(v)) throw new Error("Brick engine terrain: a column height is not finite");
  return { columns: [nx, nz], heights };
}

/** The columns as the words the `solid` buffer carries after the mask's. */
export function brickTerrainWords(terrain: BrickTerrainColumns): Uint32Array {
  return new Uint32Array(terrain.heights.buffer.slice(terrain.heights.byteOffset, terrain.heights.byteOffset + terrain.heights.byteLength));
}

/**
 * The terrain's WGSL: nbTerrainSurface (height, slope), nbTerrainBoundary (gradient, signed value: positive in fluid),
 * nbTerrainBoxOpen (the open fraction of a box: four local planes), nbTerrainSolid (a point under the ground),
 * nbTerrainColumnCut (a cell below its column's height: the old engine's cut cell, uniformMixedSolidTiles) and
 * nbTerrainContact (the particle's contact on each stride of the walk).
 */
export function brickTerrainWGSL(src: BrickTerrainSource): string {
  const [nx, nz] = src.columns;
  for (const n of src.columns) if (!Number.isInteger(n) || n < 1) throw new Error(`Brick engine terrain: ${nx}x${nz} columns`);
  if (!Number.isInteger(src.base) || src.base < 0) throw new Error(`Brick engine terrain: base word ${src.base}`);
  return /* wgsl */`
const NB_TERRAIN_BASE:u32=${src.base}u;
const NB_TERRAIN_COLUMNS:vec2i=vec2i(${nx},${nz});
// umSolidTerrainHeight: a read past the columns takes the edge column.
fn nbTerrainHeight(x:i32,z:i32)->f32{
 let c=clamp(vec2i(x,z),vec2i(0),NB_TERRAIN_COLUMNS-1);
 return bitcast<f32>(${src.buffer}[NB_TERRAIN_BASE+u32(c.x)+${nx}u*u32(c.y)]);
}
// umTerrainSurface (frame page 0, no analytic ground): one continuous height and slope. The first/last bilinear
// patch continues across the outer half-cell to the wall; outside the columns the edge height stays constant.
fn nbTerrainSurface(q:vec3f)->vec3f{
 let low=vec2f(0.0);let high=vec2f(NB_TERRAIN_COLUMNS)-vec2f(1);
 let p=clamp(q.xz-vec2f(0.5),low-0.5,high+0.5);
 let a=clamp(vec2i(floor(p)),vec2i(low),vec2i(max(low,high-1.0)));
 let f=p-vec2f(a);
 var heights:array<f32,4>;for(var k=0u;k<4u;k++){heights[k]=nbTerrainHeight(a.x+i32(k&1u),a.y+i32(k>>1u));}
 let h00=heights[0];let h10=heights[1];let h01=heights[2];let h11=heights[3];
 let height=mix(mix(h00,h10,f.x),mix(h01,h11,f.x),f.y);
 let slope=vec2f(mix(h10-h00,h11-h01,f.y),mix(h01-h00,h11-h10,f.x));
 let interior=(q.xz>=low)&(q.xz<=high+1.0);
 return vec3f(height,select(vec2f(0),slope,interior));
}
// umTerrainBoundary with no sculpt cut: positive in fluid space, gradient and value in cells.
fn nbTerrainBoundary(q:vec3f)->vec4f{
 let surface=nbTerrainSurface(q);
 return vec4f(-surface.y,1.0,-surface.z,q.y-surface.x);
}
${terrainPlaneBoxWGSL()}
// umTerrainBoxOpen: four local planes, exact on a planar heightfield.
fn nbTerrainBoxOpen(q:vec3f,width:vec3f)->f32{
 var open=0.0;let part=width*vec3f(0.5,1,0.5);
 for(var k=0u;k<4u;k++){
  let at=q+vec3f(select(-0.25,0.25,(k&1u)!=0u)*width.x,0,select(-0.25,0.25,(k&2u)!=0u)*width.z);
  let s=nbTerrainBoundary(at);
  open+=0.25*nbTerrainPlaneBoxFraction(s.xyz,s.w,part);
 }
 return open;
}
// umSolidTerrainAtWorld, in cells.
fn nbTerrainSolid(q:vec3f)->bool{return nbTerrainBoundary(q).w<0.0;}
// uniformMixedSolidTiles' cut cell: a cell of the columns below its column's height (clamp(height - y, 0, 1) > 0).
fn nbTerrainColumnCut(c:vec3i)->bool{
 if(any(c.xz<vec2i(0))||any(c.xz>=NB_TERRAIN_COLUMNS)){return false;}
 return f32(c.y)<nbTerrainHeight(c.x,c.z);
}
// The highest column over columns [lo, hi] (clamped to the columns): the bound the stamp's ring and cut tests take.
fn nbTerrainTop(lo:vec2i,hi:vec2i)->f32{
 let a=clamp(lo,vec2i(0),NB_TERRAIN_COLUMNS-1);let b=clamp(hi,vec2i(0),NB_TERRAIN_COLUMNS-1);
 var top=-1e30;
 for(var z=a.y;z<=b.y;z++){for(var x=a.x;x<=b.x;x++){top=max(top,nbTerrainHeight(x,z));}}
 return top;
}
struct NbTerrainContact{position:vec3f,velocity:vec3f}
// nbTerrainContact (uniform-particle-contact.wgsl.ts), in cells: on each stride of the walk, find the contact along
// the segment, then carry the remaining tangential displacement. Stage 0 the end, 1 the start, 2 the bisection's
// twelve midpoints, 3 the hit, 4 the slide's up to four re-evaluations after it. inset: the particle radius (cells),
// kept along the boundary normal.
fn nbTerrainContact(startPoint:vec3f,endPoint:vec3f,velocity:vec3f,inset:f32)->NbTerrainContact{
 var q=endPoint;var v=velocity;
 var low=0.0;var high=1.0;var mid=0.0;var stage=0u;var i=0u;var at=endPoint;
 loop{
  let s=nbTerrainBoundary(at);let gradient=s.xyz;
  let magnitude=max(length(gradient),1e-12);let distance=s.w/magnitude;
  let skin=inset;
  if(stage==0u){if(distance>skin){return NbTerrainContact(q,v);}stage=1u;at=startPoint;continue;}
  if(stage==1u||stage==2u){
   if(stage==2u){if(distance>skin){low=mid;}else{high=mid;}i++;}
   if((stage==1u&&distance>skin)||(stage==2u&&i<12u)){stage=2u;mid=0.5*(low+high);at=mix(startPoint,endPoint,mid);continue;}
   stage=3u;at=mix(startPoint,endPoint,low);continue;
  }
  let n=gradient/magnitude;
  if(stage==3u){
   var travel=endPoint-at;
   travel-=min(dot(travel,n),0.0)*n;
   q=at+travel;
   v-=min(dot(v,n),0.0)*n;
   // A segment can cross a patch edge; re-evaluate its normal after sliding.
   stage=4u;i=0u;at=q;continue;
  }
  let penetration=skin-distance;
  if(penetration<=0.0){break;}
  q+=penetration*n;
  v-=min(dot(v,n),0.0)*n;
  i++;if(i>=4u){break;}at=q;
 }
 return NbTerrainContact(q,v);
}
`;
}

/**
 * The bricks the ground may cut (solidStamp's bound, an upper bound on its cut): every brick of the box at or below the
 * ground's reach over its cells and the cells below their lower faces (nbStampTerrainReach: their columns and one more
 * each side, the top plus the spread), wholly buried bricks included, as brickStaticCutBricks counts a voxel's.
 */
export function brickTerrainCutBricks(terrain: BrickTerrainColumns, box: readonly number[], add: (x: number, y: number, z: number) => void): void {
  const [nx, nz] = terrain.columns, H = terrain.heights, nb = box.map(n => Math.ceil(n / 8));
  for (let bz = 0; bz < nb[2]!; bz++) for (let bx = 0; bx < nb[0]!; bx++) {
    let top = -Infinity, bottom = Infinity;
    for (let z = Math.max(0, 8 * bz - 2); z <= Math.min(nz - 1, 8 * bz + 8); z++) for (let x = Math.max(0, 8 * bx - 2); x <= Math.min(nx - 1, 8 * bx + 8); x++) {
      const v = H[x + nx * z]!; top = Math.max(top, v); bottom = Math.min(bottom, v);
    }
    if (!(top > -Infinity)) continue;
    const reach = top + (top - bottom);
    for (let by = 0; by < nb[1]! && 8 * by - 0.5 < reach; by++) add(bx, by, bz);
  }
}

/** The Float64 mirror (the test oracle) of nbTerrainSurface, nbTerrainBoundary, nbTerrainBoxOpen and
 * nbTerrainColumnCut. */
export function brickTerrainMirror(terrain: BrickTerrainColumns) {
  const [nx, nz] = terrain.columns, H = terrain.heights;
  const height = (x: number, z: number) => H[Math.min(nx - 1, Math.max(0, x)) + nx * Math.min(nz - 1, Math.max(0, z))]!;
  const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));
  const surface = (q: readonly number[]): [number, number, number] => {
    const hx = nx - 1, hz = nz - 1;
    const px = clamp(q[0]! - 0.5, -0.5, hx + 0.5), pz = clamp(q[2]! - 0.5, -0.5, hz + 0.5);
    const ax = clamp(Math.floor(px), 0, Math.max(0, hx - 1)), az = clamp(Math.floor(pz), 0, Math.max(0, hz - 1));
    const fx = px - ax, fz = pz - az;
    const h00 = height(ax, az), h10 = height(ax + 1, az), h01 = height(ax, az + 1), h11 = height(ax + 1, az + 1);
    const mix = (a: number, b: number, t: number) => a * (1 - t) + b * t;
    const value = mix(mix(h00, h10, fx), mix(h01, h11, fx), fz);
    const sx = mix(h10 - h00, h11 - h01, fz), sz = mix(h01 - h00, h11 - h10, fx);
    const ix = q[0]! >= 0 && q[0]! <= hx + 1, iz = q[2]! >= 0 && q[2]! <= hz + 1;
    return [value, ix ? sx : 0, iz ? sz : 0];
  };
  const boundary = (q: readonly number[]): [number, number, number, number] => { const s = surface(q); return [-s[1], 1, -s[2], q[1]! - s[0]]; };
  const boxOpen = (q: readonly number[], width: readonly number[]) => {
    let open = 0;
    for (let k = 0; k < 4; k++) {
      const at = [q[0]! + ((k & 1) ? 0.25 : -0.25) * width[0]!, q[1]!, q[2]! + ((k & 2) ? 0.25 : -0.25) * width[2]!];
      const s = boundary(at);
      open += 0.25 * planeBoxFraction([s[0], s[1], s[2]], s[3], [0.5 * width[0]!, width[1]!, 0.5 * width[2]!]);
    }
    return open;
  };
  return {
    surface, boundary, boxOpen,
    solid: (q: readonly number[]) => boundary(q)[3] < 0,
    columnCut: (c: readonly number[]) => c[0]! >= 0 && c[2]! >= 0 && c[0]! < nx && c[2]! < nz && c[1]! < height(c[0]!, c[2]!),
  };
}

/** geometricPlaneBoxFraction in Float64: the fraction of a centred box of `widths` with dot(normal, x) <= offset. */
export function planeBoxFraction(normal: readonly number[], offset: number, widths: readonly number[]): number {
  const projected = normal.map((n, a) => Math.abs(n) * widths[a]!), dominant = Math.max(...projected);
  if (dominant <= 1e-20) return offset >= 0 ? 1 : 0;
  const spans = projected.map(p => p / dominant).filter(v => v >= 1e-6);
  const total = spans.reduce((s, v) => s + v, 0), shifted = offset / dominant + 0.5 * total;
  if (shifted <= 0) return 0;
  if (shifted >= total) return 1;
  const complement = shifted > 0.5 * total, x = complement ? total - shifted : shifted;
  let fraction: number;
  if (spans.length === 1) fraction = x / spans[0]!;
  else if (spans.length === 2) {
    const a = Math.min(spans[0]!, spans[1]!), b = Math.max(spans[0]!, spans[1]!);
    fraction = x < a ? 0.5 * (x / a) * (x / b) : (x - 0.5 * a) / b;
  } else {
    const s = [...spans].sort((p, q) => p - q), a = s[0]!, b = s[1]!, c = s[2]!;
    const prim = (t: number) => t <= 0 ? 0 : t < a ? t * (t / a) * (t / b) / 6 : t <= b ? (0.5 * t * (t - a) + a * a / 6) / b
      : t < a + b ? t - 0.5 * (a + b) + (a + b - t) * ((a + b - t) / a) * ((a + b - t) / b) / 6 : t - 0.5 * (a + b);
    fraction = (prim(x) - prim(x - c)) / c;
  }
  fraction = Math.min(1, Math.max(0, fraction));
  return complement ? 1 - fraction : fraction;
}
