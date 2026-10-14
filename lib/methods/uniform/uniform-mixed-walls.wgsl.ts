/** Walls by page kind: a wall is a face of a stored page with a SOLID page
 * across it. An AIR page across a face is no wall: nothing is held there and
 * nothing continues from it. Each fragment reads the page table by
 * coordinate (umPageWord, umPageKind of uniform-lattice-address.ts) and is
 * included only by the modules that call it; each has a host twin of the same
 * arithmetic below.
 *
 * Faces are k = 2 axis + side, side 1 the high face of the stored page (the
 * liquid's side is then below the plane); u = (axis + 1) % 3 and
 * v = (axis + 2) % 3 are the plane's axes, as the block-face families have
 * them (uniformBlockFaceWGSL). */

/** Walls through and near a lattice vertex. */
export const uniformMixedWallVertexWGSL = /* wgsl */ `
// The block of a stored page that touches vertex p and has a SOLID page
// across its face k through p, or 0 (block 0 holds no page). A vertex on a
// page face of u or v touches two pages on that axis, the upper asked first:
// one block answers for a vertex however many wall faces share it.
fn umWallBlock(p:vec3i,k:u32)->u32{
 let axis=k>>1u;if((p[axis]&31)!=0){return 0u;}
 let u=(axis+1u)%3u;let v=(axis+2u)%3u;let upper=(k&1u)!=0u;
 var page=p>>vec3u(5u);page[axis]-=select(0,1,upper);
 let both=vec2<bool>((p[u]&31)==0,(p[v]&31)==0);
 for(var j=0u;j<4u;j++){
  if(((j&1u)!=0u&&!both.x)||((j&2u)!=0u&&!both.y)){continue;}
  var q=page;q[u]-=i32(j&1u);q[v]-=i32(j>>1u);
  let b=umPageWord(q);if(!umBlockStored(b)){continue;}
  q[axis]+=select(-1,1,upper);
  if(umPageKind(q)==UM_SOLID){return b;}
 }
 return 0u;
}
// The first wall face through p, or 6.
fn umWallPlane(p:vec3i)->u32{
 if(all((p&vec3i(31))!=vec3i(0))){return 6u;}
 for(var k=0u;k<6u;k++){if(umWallBlock(p,k)!=0u){return k;}}
 return 6u;
}
// A wall across axis lies fewer than reach cells (at most 32) from p along it.
fn umWallWithin(p:vec3i,axis:u32,reach:i32)->bool{
 let l=p[axis]&31;var q=p;
 if(l<reach){q[axis]=p[axis]-l;if(umWallBlock(q,2u*axis)!=0u){return true;}}
 if(l==0||l>32-reach){q[axis]=p[axis]+((32-l)&31);if(umWallBlock(q,2u*axis+1u)!=0u){return true;}}
 return false;
}
`;

/** The lanes of a wall face of the wall-face list (umWallFace, 6 b + f):
 * needs umBlockPage. */
export const uniformMixedWallFaceWGSL = /* wgsl */ `
// Vertex l of a wall face's 33 x 33 plane, u fastest.
fn umWallFaceVertex(face:u32,l:u32)->vec3i{
 let f=face%6u;let axis=f>>1u;
 var p=vec3i(umBlockPage(face/6u)<<vec3u(5u));p[axis]+=i32(32u*(f&1u));p[(axis+1u)%3u]+=i32(l%33u);p[(axis+2u)%3u]+=i32(l/33u);
 return p;
}
// Tile lane of a wall face's 8 x 8 plane, u fastest, in the page's first or
// last layer of tiles: umFaceTile of it is 64 face + lane.
fn umWallFaceTile(face:u32,lane:u32)->vec3u{
 let f=face%6u;let axis=f>>1u;
 var t=umBlockPage(face/6u)<<vec3u(3u);t[axis]+=7u*(f&1u);t[(axis+1u)%3u]+=lane&7u;t[(axis+2u)%3u]+=lane>>3u;
 return t;
}
`;

/** The wall a face of direction k can reach a vertex from. */
export const uniformMixedWallAheadWGSL = /* wgsl */ `
// Walking out from vertex p against face k's inward normal, one page face at
// a time: the first plane at which a column of p's four probe cells (p-1 and
// p on u and on v; bit 0 of the column is u's upper cell, bit 1 v's) is a
// stored page with a SOLID page across. blocks holds that page's block for
// each such column and 0 for every other, all 0 where there is no wall: the
// walk ends where no column continues from a stored page into a stored page,
// and at the first plane too deep for reach (a length, as h) to raise bound,
// by the released wall's own skip test.
struct UMWallAhead {plane:i32,blocks:array<u32,4>}
fn umWallAhead(p:vec3i,k:u32,reach:f32,bound:f32,h:vec3f)->UMWallAhead{
 var ahead=UMWallAhead(0,array<u32,4>(0u,0u,0u,0u));
 let axis=k>>1u;let u=(axis+1u)%3u;let v=(axis+2u)%3u;let upper=(k&1u)!=0u;let inward=select(1.0,-1.0,upper);
 let slack=1e-5*max(h.x,max(h.y,h.z));
 // Columns that differ in page: only across a page face of u or v.
 let mask=u32((p[u]&31)==0)|(u32((p[v]&31)==0)<<1u);
 var plane=select(p[axis]&(-32),(p[axis]+31)&(-32),upper);
 // On a page face p has no page behind the first plane to walk out of.
 var first=(p[axis]&31)==0;
 for(var n=0u;n<64u;n++){
  if(reach-inward*f32(p[axis]-plane)*h[axis]+slack<bound){return ahead;}
  var blocks=array<u32,4>(0u,0u,0u,0u);var open=false;var wall=false;
  for(var c=0u;c<4u;c++){
   if((c&mask)!=c){blocks[c]=blocks[c&mask];continue;}
   var q=p;q[axis]=plane-select(0,1,upper);q[u]-=i32(1u-(c&1u));q[v]-=i32(1u-(c>>1u));
   q=q>>vec3u(5u);let b=umPageWord(q);q[axis]+=select(-1,1,upper);let across=umPageWord(q);
   if(umBlockStored(across)){open=open||first||umBlockStored(b);}else if(across!=0u&&umBlockStored(b)){blocks[c]=b;wall=true;}
  }
  if(wall){ahead.plane=plane;ahead.blocks=blocks;return ahead;}
  if(!open){return ahead;}
  plane+=select(-32,32,upper);first=false;
 }
 return ahead;
}
`;

/** A cell or tile held out of SOLID pages. */
export const uniformMixedSolidHoldWGSL = /* wgsl */ `
// p, a cell (bits 5) or a tile (bits 3), held out of SOLID pages. In a stored
// or an AIR page it is itself. In a SOLID page it steps back across the
// page's nearer face on each axis: on one axis alone where the page across
// that face is not SOLID (the shallowest such axis), else on two, else on
// all three, to the nearest coordinate there; itself where every such page
// is SOLID.
fn umSpanHeld(p:vec3i,bits:u32)->vec3i{
 let page=p>>vec3u(bits);if(umPageKind(page)!=UM_SOLID){return p;}
 let n=i32(1u<<bits);let low=page<<vec3u(bits);let l=p-low;
 let up=l>=vec3i(n/2);let d=select(vec3i(-1),vec3i(1),up);
 let out=low+select(vec3i(-1),vec3i(n),up);let depth=select(l+vec3i(1),vec3i(n)-l,up);
 var best=3u;
 for(var a=0u;a<3u;a++){
  var e=vec3i(0);e[a]=d[a];
  if(umPageKind(page+e)!=UM_SOLID&&(best==3u||depth[a]<depth[best])){best=a;}
 }
 if(best<3u){var q=p;q[best]=out[best];return q;}
 for(var a=0u;a<3u;a++){var e=d;e[a]=0;if(umPageKind(page+e)!=UM_SOLID){var q=out;q[a]=p[a];return q;}}
 if(umPageKind(page+d)!=UM_SOLID){return out;}
 return p;
}
fn umCellHeld(p:vec3i)->vec3i{return umSpanHeld(p,5u);}
fn umTileHeld(p:vec3i)->vec3i{return umSpanHeld(p,3u);}
`;

/** A position held out of SOLID pages. */
export const uniformMixedSolidHoldPointWGSL = /* wgsl */ `
// umSpanHeld of a position in cell units: a point in a SOLID page (its
// cell's) moves back onto the page's nearer face, by the same choice of
// axes. A point on a wall's plane stays.
fn umPointHeld(p:vec3f)->vec3f{
 let page=vec3i(floor(p))>>vec3u(5u);if(umPageKind(page)!=UM_SOLID){return p;}
 let low=vec3f(page<<vec3u(5u));let l=p-low;
 let up=l>=vec3f(16.0);let d=select(vec3i(-1),vec3i(1),up);
 let out=low+select(vec3f(0.0),vec3f(32.0),up);let depth=select(l,vec3f(32.0)-l,up);
 var best=3u;
 for(var a=0u;a<3u;a++){
  var e=vec3i(0);e[a]=d[a];
  if(umPageKind(page+e)!=UM_SOLID&&(best==3u||depth[a]<depth[best])){best=a;}
 }
 if(best<3u){var q=p;q[best]=out[best];return q;}
 for(var a=0u;a<3u;a++){var e=d;e[a]=0;if(umPageKind(page+e)!=UM_SOLID){var q=out;q[a]=p[a];return q;}}
 if(umPageKind(page+d)!=UM_SOLID){return out;}
 return p;
}
`;

type Vec3 = [number, number, number];
type Blocks4 = [number, number, number, number];
const pageWord = (table: ArrayLike<number>, x: number, y: number, z: number): number => table[(x & 63) | ((y & 63) << 6) | ((z & 63) << 12)]!;
/** umBlockStored and umPageKind's SOLID of a table word, in a pool of `blocks`. */
const storedWord = (w: number, blocks: number): boolean => w >= 1 && w < blocks - 1;
const solidWord = (w: number, blocks: number): boolean => w >= blocks - 1;
const f = Math.fround;
const scratch: Vec3 = [0, 0, 0], step: Vec3 = [0, 0, 0];

/** umWallBlock on the host, over a page table of a pool of `blocks`. */
export function uniformWallBlock(table: ArrayLike<number>, blocks: number, p: readonly number[], k: number): number {
  const axis = k >> 1, u = (axis + 1) % 3, v = (axis + 2) % 3, upper = (k & 1) !== 0;
  if ((p[axis]! & 31) !== 0) return 0;
  const bothU = (p[u]! & 31) === 0, bothV = (p[v]! & 31) === 0, q = scratch;
  for (let j = 0; j < 4; j++) {
    if (((j & 1) !== 0 && !bothU) || ((j & 2) !== 0 && !bothV)) continue;
    q[0] = p[0]! >> 5;q[1] = p[1]! >> 5;q[2] = p[2]! >> 5;
    q[axis]! -= upper ? 1 : 0;q[u]! -= j & 1;q[v]! -= j >> 1;
    const b = pageWord(table, q[0], q[1], q[2]);
    if (!storedWord(b, blocks)) continue;
    q[axis]! += upper ? 1 : -1;
    if (solidWord(pageWord(table, q[0], q[1], q[2]), blocks)) return b;
  }
  return 0;
}
/** umWallPlane on the host. */
export function uniformWallPlane(table: ArrayLike<number>, blocks: number, p: readonly number[]): number {
  if ((p[0]! & 31) !== 0 && (p[1]! & 31) !== 0 && (p[2]! & 31) !== 0) return 6;
  for (let k = 0; k < 6; k++) if (uniformWallBlock(table, blocks, p, k) !== 0) return k;
  return 6;
}
const within: Vec3 = [0, 0, 0];
/** umWallWithin on the host. */
export function uniformWallWithin(table: ArrayLike<number>, blocks: number, p: readonly number[], axis: number, reach: number): boolean {
  const l = p[axis]! & 31, q = within;
  q[0] = p[0]!;q[1] = p[1]!;q[2] = p[2]!;
  if (l < reach) { q[axis] = p[axis]! - l;if (uniformWallBlock(table, blocks, q, 2 * axis) !== 0) return true; }
  if (l === 0 || l > 32 - reach) { q[axis] = p[axis]! + ((32 - l) & 31);if (uniformWallBlock(table, blocks, q, 2 * axis + 1) !== 0) return true; }
  return false;
}
/** umWallFaceVertex on the host, over the block records. */
export function uniformWallFaceVertex(records: ArrayLike<number>, face: number, l: number, out: Vec3): Vec3 {
  const r = records[Math.floor(face / 6)]!, side = face % 6, axis = side >> 1;
  out[0] = (r & 63) << 5;out[1] = ((r >>> 6) & 63) << 5;out[2] = ((r >>> 12) & 63) << 5;
  out[axis]! += 32 * (side & 1);out[(axis + 1) % 3]! += l % 33;out[(axis + 2) % 3]! += Math.floor(l / 33);
  return out;
}
/** umWallFaceTile on the host. */
export function uniformWallFaceTile(records: ArrayLike<number>, face: number, lane: number, out: Vec3): Vec3 {
  const r = records[Math.floor(face / 6)]!, side = face % 6, axis = side >> 1;
  out[0] = (r & 63) << 3;out[1] = ((r >>> 6) & 63) << 3;out[2] = ((r >>> 12) & 63) << 3;
  out[axis]! += 7 * (side & 1);out[(axis + 1) % 3]! += lane & 7;out[(axis + 2) % 3]! += lane >> 3;
  return out;
}
/** umWallAhead on the host: the plane and the four columns' blocks, written to out. */
export function uniformWallAhead(table: ArrayLike<number>, blocks: number, p: readonly number[], k: number, reach: number, bound: number, h: readonly number[], out: { plane: number; blocks: Blocks4 }): { plane: number; blocks: Blocks4 } {
  const axis = k >> 1, u = (axis + 1) % 3, v = (axis + 2) % 3, upper = (k & 1) !== 0, inward = upper ? -1 : 1;
  const slack = f(f(1e-5) * Math.max(h[0]!, h[1]!, h[2]!)), mask = ((p[u]! & 31) === 0 ? 1 : 0) | ((p[v]! & 31) === 0 ? 2 : 0), q = scratch, found = out.blocks;
  out.plane = 0;found.fill(0);
  let plane = upper ? (p[axis]! + 31) & -32 : p[axis]! & -32, first = (p[axis]! & 31) === 0;
  for (let n = 0; n < 64; n++) {
    if (f(f(reach - f(f(inward * (p[axis]! - plane)) * h[axis]!)) + slack) < bound) return out;
    let open = false, wall = false;
    for (let c = 0; c < 4; c++) {
      if ((c & mask) !== c) { found[c] = found[c & mask]!;continue; }
      found[c] = 0;
      q[0] = p[0]!;q[1] = p[1]!;q[2] = p[2]!;
      q[axis] = plane - (upper ? 1 : 0);q[u]! -= 1 - (c & 1);q[v]! -= 1 - (c >> 1);
      q[0] >>= 5;q[1] >>= 5;q[2] >>= 5;
      const b = pageWord(table, q[0], q[1], q[2]);
      q[axis]! += upper ? 1 : -1;
      const across = pageWord(table, q[0], q[1], q[2]);
      if (storedWord(across, blocks)) open = open || first || storedWord(b, blocks);else if (across !== 0 && storedWord(b, blocks)) { found[c] = b;wall = true; }
    }
    if (wall) { out.plane = plane;return out; }
    found.fill(0);
    if (!open) return out;
    plane += upper ? 32 : -32;first = false;
  }
  return out;
}
/** The shared choice of umSpanHeld and umPointHeld: the axes (a mask) on which a coordinate of SOLID page `page`
 * steps back, from the depths to its nearer faces; 0 where it stays. */
function heldAxes(table: ArrayLike<number>, blocks: number, page: readonly number[], d: readonly number[], depth: readonly number[]): number {
  let best = 3;
  for (let a = 0; a < 3; a++) {
    step[0] = 0;step[1] = 0;step[2] = 0;step[a] = d[a]!;
    if (!solidWord(pageWord(table, page[0]! + step[0], page[1]! + step[1], page[2]! + step[2]), blocks) && (best === 3 || depth[a]! < depth[best]!)) best = a;
  }
  if (best < 3) return 1 << best;
  for (let a = 0; a < 3; a++) {
    step[0] = d[0]!;step[1] = d[1]!;step[2] = d[2]!;step[a] = 0;
    if (!solidWord(pageWord(table, page[0]! + step[0], page[1]! + step[1], page[2]! + step[2]), blocks)) return 7 & ~(1 << a);
  }
  return solidWord(pageWord(table, page[0]! + d[0]!, page[1]! + d[1]!, page[2]! + d[2]!), blocks) ? 0 : 7;
}
const heldPage: Vec3 = [0, 0, 0], heldStep: Vec3 = [0, 0, 0], heldDepth: Vec3 = [0, 0, 0];
/** umSpanHeld on the host (bits 5: umCellHeld, bits 3: umTileHeld), written to out. */
export function uniformSpanHeld(table: ArrayLike<number>, blocks: number, p: readonly number[], bits: number, out: Vec3): Vec3 {
  const n = 1 << bits;
  for (let a = 0; a < 3; a++) { out[a] = p[a]!;heldPage[a] = p[a]! >> bits; }
  if (!solidWord(pageWord(table, heldPage[0], heldPage[1], heldPage[2]), blocks)) return out;
  for (let a = 0; a < 3; a++) { const l = p[a]! - (heldPage[a]! << bits), up = l >= n / 2;heldStep[a] = up ? 1 : -1;heldDepth[a] = up ? n - l : l + 1; }
  const axes = heldAxes(table, blocks, heldPage, heldStep, heldDepth);
  for (let a = 0; a < 3; a++) if (axes & (1 << a)) out[a] = (heldPage[a]! << bits) + (heldStep[a]! > 0 ? n : -1);
  return out;
}
/** umPointHeld on the host, in f32, written to out. */
export function uniformPointHeld(table: ArrayLike<number>, blocks: number, p: readonly number[], out: Vec3): Vec3 {
  for (let a = 0; a < 3; a++) { out[a] = f(p[a]!);heldPage[a] = Math.floor(out[a]!) >> 5; }
  if (!solidWord(pageWord(table, heldPage[0], heldPage[1], heldPage[2]), blocks)) return out;
  for (let a = 0; a < 3; a++) { const l = f(out[a]! - (heldPage[a]! << 5)), up = l >= 16;heldStep[a] = up ? 1 : -1;heldDepth[a] = up ? f(32 - l) : l; }
  const axes = heldAxes(table, blocks, heldPage, heldStep, heldDepth);
  for (let a = 0; a < 3; a++) if (axes & (1 << a)) out[a] = (heldPage[a]! << 5) + (heldStep[a]! > 0 ? 32 : 0);
  return out;
}
