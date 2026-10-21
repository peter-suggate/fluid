import { UNIFORM_NEAR_SPAN, uniformBlockHeld, uniformNearFrom } from "./uniform-lattice-address";

/** The narrow band's kind-aware clamps, from a home coordinate.
 *
 * A kernel here holds a cell, a tile or a sample position, not a block: home
 * is that coordinate, and its page is the one a tap is reached from. A tap
 * inside home's page is returned as it is and reads nothing. One outside it
 * reads home's table word and then the block's neighbour row: it is held in
 * home's page on an axis only where a SOLID page lies across that face, and
 * carries on into an AIR or a stored page (uniformBlockNeighbourWGSL, whose
 * diagonal rule this is). An AIR home holds nothing; a SOLID one everything.
 * A tap lies in home's page or in one of the 26 around it: a kernel that
 * reaches farther asks again from the page it has reached.
 *
 * nbCellNear: cells. nbTileNear: tiles, home a tile. nbHomeBlock: the block
 * a stencil hands um{Cell,Tile}NearFromPage for its taps. nbPointNearIn: a
 * position in cell units from a home cell, kept `inset` inside every face a
 * wall holds; nbPointNear: the page's closed extent.
 *
 * The including module holds uniformLatticeAddressWGSL and
 * uniformBlockNeighbourWGSL. */
export const narrowBandNearWGSL = /* wgsl */`
fn nbCellNear(home:vec3i,p:vec3i)->vec3i{
 let page=home>>vec3u(5u);if(all((p>>vec3u(5u))==page)){return p;}
 return umCellNearFromPage(umBlockOf(page),page,p);
}
fn nbTileNear(home:vec3i,p:vec3i)->vec3i{
 let page=home>>vec3u(3u);if(all((p>>vec3u(3u))==page)){return p;}
 return umTileNearFromPage(umBlockOf(page),page,p);
}
// Home's block for a stencil whose corner taps lie in pages lo and hi, read
// only where one leaves home's page: the FromPage clamps then read the table
// no more. 0 where none leaves, and no clamp reads a row.
fn nbHomeBlock(page:vec3i,lo:vec3i,hi:vec3i)->u32{if(all(lo==page)&&all(hi==page)){return 0u;}return umBlockOf(page);}
fn nbPointNearIn(home:vec3i,p:vec3f,inset:f32)->vec3f{
 let page=home>>vec3u(5u);let base=vec3f(page<<vec3u(5u));let low=base+inset;let high=base+32.0-inset;
 let d=vec3i(p>high)-vec3i(p<low);if(all(d==vec3i(0))){return p;}
 return select(p,clamp(p,low,high),umBlockHeld(umBlockOf(page),page,d));
}
fn nbPointNear(home:vec3i,p:vec3f)->vec3f{return nbPointNearIn(home,p,0.0);}
// A particle trace can cross many pages in one step. The stencil clamp
// above only inspects home's immediate neighbours and cannot see a box
// wall beyond them. Fixed world walls are launch constants: clamp directly
// to those planes, preserving the full travel along every unblocked axis.
// Open axes retain umBoxWalls' unbounded extent; residency still owns them.
fn nbTraceEndpoint(home:vec3i,p:vec3f)->vec3f{
 if(umFixedBox()){let walls=umBoxWalls();return clamp(p,walls.lo+vec3f(0.01),walls.hi-vec3f(0.01));}
 return nbPointNearIn(home,p,0.01);
}
`;

const held: [boolean, boolean, boolean] = [false, false, false], step: [number, number, number] = [0, 0, 0], page: [number, number, number] = [0, 0, 0];
/** umBlockOf on the host: the block of a page of the table, a sentinel's where it holds none. */
function blockOf(table: ArrayLike<number>, blocks: number, at: readonly number[]): number {
  return Math.min(table[(at[0]! & 63) | ((at[1]! & 63) << 6) | ((at[2]! & 63) << 12)]!, blocks - 1);
}
/** nbCellNear and nbTileNear on the host, over a page table and its
 * neighbour rows (27 words a block): p from home, both in the span's units
 * (UNIFORM_NEAR_SPAN.cell or .tile), written to out. reads, where given,
 * takes the words read: none for a tap in home's page. */
export function narrowBandNear(table: ArrayLike<number>, rows: ArrayLike<number>, blocks: number, span: { readonly bits: number; readonly last: number }, home: readonly number[], p: readonly number[], out: [number, number, number], reads?: { words: number }): [number, number, number] {
  let inside = true;
  for (let a = 0; a < 3; a++) { page[a] = home[a]! >> span.bits; inside &&= p[a]! >> span.bits === page[a]; }
  if (inside) { for (let a = 0; a < 3; a++) out[a] = p[a]!; return out; }
  if (reads) reads.words += 1 + rowReads(rows, blockOf(table, blocks, page), page, span, p);
  return uniformNearFrom(rows, blockOf(table, blocks, page), page, span, p, out);
}
/** The row words umBlockHeld reads for a tap p outside home's page: one a leaving axis, and the landing page's where two or more stay loose. */
function rowReads(rows: ArrayLike<number>, home: number, at: readonly number[], span: { readonly bits: number; readonly last: number }, p: readonly number[]): number {
  let leaving = 0, loose = 0;
  for (let a = 0; a < 3; a++) {
    const low = at[a]! << span.bits; step[a] = p[a]! > low + span.last ? 1 : p[a]! < low ? -1 : 0;
    if (step[a] !== 0) { leaving++; if (rows[27 * home + 13 + step[a]! * 3 ** a] !== 0xffffffff) loose++; }
  }
  return leaving + (loose >= 2 ? 1 : 0);
}
/** nbPointNearIn on the host: a position in cell units from a home cell, in
 * single precision as the kernel computes it, written to out. */
export function narrowBandPointNearIn(table: ArrayLike<number>, rows: ArrayLike<number>, blocks: number, home: readonly number[], p: readonly number[], inset: number, out: [number, number, number]): [number, number, number] {
  const f = Math.fround;
  let inside = true;
  for (let a = 0; a < 3; a++) {
    page[a] = home[a]! >> UNIFORM_NEAR_SPAN.point.bits;
    const base = page[a]! * 32, low = f(base + f(inset)), high = f(f(base + 32) - f(inset));
    step[a] = f(p[a]!) > high ? 1 : f(p[a]!) < low ? -1 : 0; inside &&= step[a] === 0;
  }
  if (inside) { for (let a = 0; a < 3; a++) out[a] = f(p[a]!); return out; }
  uniformBlockHeld(rows, blockOf(table, blocks, page), step, held);
  for (let a = 0; a < 3; a++) {
    const base = page[a]! * 32, low = f(base + f(inset)), high = f(f(base + 32) - f(inset));
    out[a] = held[a] ? Math.min(Math.max(f(p[a]!), low), high) : f(p[a]!);
  }
  return out;
}
