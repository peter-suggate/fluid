/** Runtime frame header between the directory and packed block words.
 * Every page-table mirror carries it, including presentation. */
export const UNIFORM_WINDOW_HEADER_WORDS = 32;
export const UNIFORM_WINDOW_EDGE_FATAL = 32;
export const UNIFORM_WINDOW_HEADER = { frame: 0, revision: 3, delta: 4, boundary: 8, policy: 12, box: 13, extent: 16, forceStep: 20, forceDelta: 21, solidLow: 24, solidHigh: 27, windowLast: 30, stale: 31 } as const;

/** Words 24..31, derived once at scene build from the policy and box and never
 * written again: the world-solid box as half-open signed bounds (a page is
 * SOLID when outside them on any axis), the last in-window lattice page as
 * an unsigned bound, and the raw directory word that reads as AIR. They make
 * the shared page lookup branch free: Metal inlines it at every call site. */
export function uniformWindowDerivedWords(policy: number, origin: readonly number[], pages: readonly number[]): Uint32Array {
  const min = -0x80000000, max = 0x7fffffff, words = new Int32Array(8);
  const low = [min, min, min], high = [max, max, max];
  if (policy === 1 || policy === 2) for (let a = 0; a < 3; a++) { if (a === 1 && policy === 2) { low[a] = origin[a]!; continue; } low[a] = origin[a]!; high[a] = origin[a]! + pages[a]!; }
  if (policy === 3) low[1] = origin[1]!;
  words.set(low, 0); words.set(high, 3);
  const out = new Uint32Array(words.buffer);
  out[6] = policy === 0 ? 0xffffffff : 63; out[7] = policy === 0 ? 0xfffffffe : 0xffffffff;
  return out;
}

/** Shared classification in signed world pages. Policy 0 is the standalone
 * allocator's explicit table, 1 a closed tank, 2 its open-top column, 3 open
 * ground. Voxel/procedural solids inside stored pages remain the solid
 * sampler's responsibility. The authored box never moves with the frame. */
export function uniformWindowWorldWGSL(word: (index: string) => string, prefix: string): string {
  return /* wgsl */ `
fn ${prefix}FramePage()->vec3i{return bitcast<vec3i>(vec3u(${word("0u")},${word("1u")},${word("2u")}));}
fn ${prefix}Policy()->u32{return ${word("12u")};}
fn ${prefix}WorldSolid(q:vec3i)->bool{
 let policy=${prefix}Policy();if(policy==0u){return false;}
 let lo=bitcast<vec3i>(vec3u(${word("13u")},${word("14u")},${word("15u")}));
 let size=vec3i(vec3u(${word("16u")},${word("17u")},${word("18u")}));
 if(policy==3u){return q.y<lo.y;}
 let relative=q-lo;
 return any(relative<vec3i(0))||relative.x>=size.x||relative.z>=size.z||(policy==1u&&relative.y>=size.y);
}`;
}

/** The public block word remains a packed *lattice* page for all existing
 * decoders; only its stored representation is world-page residues. The page
 * lookup is the same rule as uniformWindowWorldWGSL's classification but
 * branch free over the derived header words (24..31): it is inlined at
 * hundreds of call sites per kernel. */
export function uniformWindowAddressWGSL(buffer: string, table: string, blocks: string, prefix = "um", entries = 64 ** 3): string {
  const w = (i: number) => `${buffer}[${table}+${entries + i}u]`;
  const v = (i: number) => `vec3u(${w(i)},${w(i + 1)},${w(i + 2)})`;
  return /* wgsl */ `
fn ${prefix}FramePage()->vec3i{return bitcast<vec3i>(${v(0)});}
// The first lattice cell above each axis's low wall (the authored box's low
// face): a face anchor below it is a low-wall face, read as the cell above.
// Never below the window's plane 0, which is all it is on an axis without a
// low wall (and a wall below the window): the fixed window's contract.
fn ${prefix}WallLow()->vec3i{let low=bitcast<vec3i>(${v(24)});return max(select(32*(low-${prefix}FramePage()),vec3i(0),low==vec3i(bitcast<i32>(0x80000000u))),vec3i(0));}
fn ${prefix}WorldSolid(world:vec3i)->bool{return any(world<bitcast<vec3i>(${v(24)}))|any(world>=bitcast<vec3i>(${v(27)}));}
fn ${prefix}PageWord(page:vec3i)->u32{
 let world=page+${prefix}FramePage();let e=bitcast<vec3u>(world)&vec3u(63u);
 let raw=${buffer}[${table}+(e.x|(e.y<<6u)|(e.z<<12u))];
 let keep=all(bitcast<vec3u>(page)<=vec3u(${w(30)}))&(raw!=${w(31)});
 return select(select(0u,raw,keep),0xffffffffu,${prefix}WorldSolid(world));
}
fn ${prefix}BlockWord(b:u32)->u32{
 let raw=${buffer}[${table}+${entries + UNIFORM_WINDOW_HEADER_WORDS}u+b];
 let q=(vec3u(raw&63u,(raw>>6u)&63u,(raw>>12u)&63u)-bitcast<vec3u>(${prefix}FramePage()))&vec3u(63u);
 return select(q.x|(q.y<<6u)|(q.z<<12u),raw,raw==0xffffffffu);
}`;
}
