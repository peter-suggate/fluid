/** The page directory (tiered-adaptivity §2.1): a top table of 64³ words over
 * super-blocks of 4³ pages (128³ cells), each AIR (0), SOLID (0xffffffff) or
 * a super-block slot, then UNIFORM_SUPER_BLOCK_SLOTS child tables of 64 words,
 * one a page of the slot's super-block: its block, AIR or SOLID. Slot 0 is all
 * AIR and the last slot all SOLID, so a page's word is always
 * child[min(top, last)·64 + local]: two dependent reads, no branch. Entry
 * (world >> 2) & 63 and local (world & 3), packed x | y<<2 | z<<4, make the
 * two levels exactly a toroidal table of 256 pages an axis (8192 cells):
 * pages 256 apart share a word, and super-blocks 64 apart share an entry.
 * After the child tables, the flat window: 64³ words, the raw directory word
 * of lattice pages -1..62 an axis (x+1 | y+1<<6 | z+1<<12) under the current
 * frame, rebuilt by every commit after its last directory write. A lookup
 * there is the one read of the 64-page table the directory replaced, so a
 * frame that fits it pays nothing for the second level. */
export const UNIFORM_DIRECTORY_EDGE = 64;
export const UNIFORM_SUPER_BLOCK_PAGES = 4;
export const UNIFORM_SUPER_BLOCK_CHILDREN = 64;
/** Child tables the directory holds, the two sentinels among them. Fixed, so
 * the tail's layout is a constant: a slot holds a resident page or a seeded
 * wall, and the pool's blocks (at most 3968, the block grid's cap) bound the
 * slots in use. */
export const UNIFORM_SUPER_BLOCK_SLOTS = 4096;
/** Pages an axis of the window: the directory's period. */
export const UNIFORM_WINDOW_PAGES = UNIFORM_DIRECTORY_EDGE * UNIFORM_SUPER_BLOCK_PAGES;
/** Words of the top table; of the child tables; the flat window's edge in pages and its words. */
export const UNIFORM_DIRECTORY_TOP_WORDS = UNIFORM_DIRECTORY_EDGE ** 3;
export const UNIFORM_DIRECTORY_CHILD_WORDS = UNIFORM_SUPER_BLOCK_CHILDREN * UNIFORM_SUPER_BLOCK_SLOTS;
export const UNIFORM_DIRECTORY_FLAT_EDGE = 64;
export const UNIFORM_DIRECTORY_FLAT_WORDS = UNIFORM_DIRECTORY_FLAT_EDGE ** 3;
/** Words of the whole directory (top, child tables, flat window): where the runtime header starts. */
export const UNIFORM_DIRECTORY_WORDS = UNIFORM_DIRECTORY_TOP_WORDS + UNIFORM_DIRECTORY_CHILD_WORDS + UNIFORM_DIRECTORY_FLAT_WORDS;
/** A page's index in its super-block's child table. */
export const uniformSuperBlockLocal = (x: number, y: number, z: number): number => (x & 3) | ((y & 3) << 2) | ((z & 3) << 4);
/** The word of window page (x, y, z) in a directory at word `at` of `tail`, a top table `edge` super-blocks an axis. */
export function uniformDirectoryWord(tail: ArrayLike<number>, x: number, y: number, z: number, edge = UNIFORM_DIRECTORY_EDGE, at = 0): number {
  const m = edge - 1, b = Math.log2(edge), top = tail[at + (((x >> 2) & m) | (((y >> 2) & m) << b) | (((z >> 2) & m) << (2 * b)))]!;
  return tail[at + edge ** 3 + UNIFORM_SUPER_BLOCK_CHILDREN * Math.min(top, UNIFORM_SUPER_BLOCK_SLOTS - 1) + uniformSuperBlockLocal(x, y, z)]!;
}
/** Write the flat window of a directory at word `at` of `tail` (a top table
 * `edge` super-blocks an axis) for frame F: what the commit's upFlat writes. */
export function uniformDirectoryFlat(tail: Uint32Array, frame: readonly number[], edge = UNIFORM_DIRECTORY_EDGE, at = 0): void {
  const E = UNIFORM_DIRECTORY_FLAT_EDGE, flat = at + edge ** 3 + UNIFORM_DIRECTORY_CHILD_WORDS;
  for (let z = 0, i = 0; z < E; z++) for (let y = 0; y < E; y++) for (let x = 0; x < E; x++, i++)
    tail[flat + i] = uniformDirectoryWord(tail, x - 1 + frame[0]!, y - 1 + frame[1]!, z - 1 + frame[2]!, edge, at);
}
/** The host twin of umPageWord: the word of world page (wx, wy, wz) in a
 * tail (directory, then the runtime header at `at`) as the GPU reads it at
 * frame page world - F, the world box's SOLID over the stored words. Host
 * readers name world pages (a seed's frame is 0, where the two agree; a
 * live tail's frame moves under an open top). A page outside the box is
 * SOLID here by the header's policy alone: a seed for a policy other than 0
 * stores no SOLID word. */
export function uniformWindowPageWord(tail: ArrayLike<number>, wx: number, wy: number, wz: number, at = UNIFORM_DIRECTORY_WORDS): number {
  const h = (i: number) => tail[at + i]! | 0, x = wx - h(0), y = wy - h(1), z = wz - h(2), last = h(30) >>> 0;
  if (wx < h(24) || wy < h(25) || wz < h(26) || wx >= h(27) || wy >= h(28) || wz >= h(29)) return 0xffffffff;
  const raw = uniformDirectoryWord(tail, wx, wy, wz);
  return (x >>> 0) <= last && (y >>> 0) <= last && (z >>> 0) <= last && raw !== (h(31) >>> 0) ? raw : 0;
}
/** A host reader's page word of a world page: umPageWord's
 * (uniformWindowPageWord) from a tail that carries its runtime header, the
 * raw directory word from a bare directory (a seed's table). */
export function uniformTablePageWord(table: ArrayLike<number>, x: number, y: number, z: number): number {
  return table.length >= UNIFORM_DIRECTORY_WORDS + UNIFORM_WINDOW_HEADER_WORDS ? uniformWindowPageWord(table, x, y, z) : uniformDirectoryWord(table, x, y, z);
}
/** A block word: the window page residue of the page a block holds, eight bits an axis. */
export const uniformBlockWordOf = (x: number, y: number, z: number): number => ((x & 255) | ((y & 255) << 8) | ((z & 255) << 16)) >>> 0;
export const uniformBlockWordPage = (w: number): [number, number, number] => [w & 255, (w >>> 8) & 255, (w >>> 16) & 255];
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
  out[6] = policy === 0 ? 0xffffffff : UNIFORM_WINDOW_PAGES - 1; out[7] = policy === 0 ? 0xfffffffe : 0xffffffff;
  return out;
}

/** A closed tank's frame never moves (the moving-window decision keeps it;
 * only probeWindowShift would), so a solver with fixed world geometry may
 * compile it as a constant: the policy word of a window header. */
export const uniformWindowFixedFrame = (policy: number): boolean => policy === 1;

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
export function uniformWindowAddressWGSL(buffer: string, table: string, blocks: string, prefix = "um", entries = UNIFORM_DIRECTORY_WORDS): string {
  const w = (i: number) => {
    const raw = `${buffer}[${table}+${entries + i}u]`;
    return i >= 24 ? `select(${raw},${prefix}WindowWord${i},${prefix}FixedWorldGeometry)` : i < 3 ? `select(${raw},${prefix}WindowWord${i},${prefix}FixedFrame)` : raw;
  };
  const v = (i: number) => `vec3u(${w(i)},${w(i + 1)},${w(i + 2)})`;
  return /* wgsl */ `
// Generic consumers read the runtime header. Only a solver promising fixed
// world geometry supplies these overrides; page residency stays live. The
// frame (words 0..2) is a constant too where the policy never moves it (a
// closed tank, uniformWindowFixedFrame): every translation then folds.
override ${prefix}FixedWorldGeometry:bool=false;
override ${prefix}FixedFrame:bool=false;
// Every stored page lies in the flat window (a narrow pool's frames, or a
// fixed frame whose world box lies in lattice pages -1..62): the lookup is
// its one read, AIR off it, the code the 64-page table compiled to.
override ${prefix}FlatWindow:bool=false;
// A solver whose ownership binds the BULK16 slot store (bulkTier "16h"):
// only there can a tagged page word exist (uniformDetailPipeline sets it).
override ${prefix}BulkSlots:bool=false;
fn ${prefix}BulkSlotsOn()->bool{return ${prefix}BulkSlots;}
${Array.from({length:8},(_,i)=>`override ${prefix}WindowWord${24+i}:u32=0u;`).join("\n")}
${Array.from({length:3},(_,i)=>`override ${prefix}WindowWord${i}:u32=0u;`).join("\n")}
fn ${prefix}FramePage()->vec3i{return bitcast<vec3i>(${v(0)});}
// The first lattice cell above each axis's low wall (the authored box's low
// face): a face anchor below it is a low-wall face, read as the cell above.
// Never below the window's plane 0, which is all it is on an axis without a
// low wall (and a wall below the window): the fixed window's contract.
fn ${prefix}WallLow()->vec3i{let low=${prefix}WorldLow();return max(select(32*(low-${prefix}FramePage()),vec3i(0),low==vec3i(bitcast<i32>(0x80000000u))),vec3i(0));}
// The world box in world pages, half open (the extreme words where an axis has no wall).
fn ${prefix}WorldLow()->vec3i{return bitcast<vec3i>(${v(24)});}
fn ${prefix}WorldHigh()->vec3i{return bitcast<vec3i>(${v(27)});}
fn ${prefix}WorldSolid(world:vec3i)->bool{return any(world<${prefix}WorldLow())|any(world>=${prefix}WorldHigh());}
// The directory's stored word of a lattice page (no keep, no world box).
fn ${prefix}PageTableWord(page:vec3i)->u32{
 let world=page+${prefix}FramePage();let f=bitcast<vec3u>(page+vec3i(1));
 // Lattice pages -1..62 read the flat window, one read; the rest the top
 // word, then the child word it names.
 let inFlat=all(f<vec3u(${UNIFORM_DIRECTORY_FLAT_EDGE}u));
 if(${prefix}FlatWindow||inFlat){let g=min(f,vec3u(${UNIFORM_DIRECTORY_FLAT_EDGE - 1}u));return select(0u,${buffer}[${table}+${UNIFORM_DIRECTORY_TOP_WORDS + UNIFORM_DIRECTORY_CHILD_WORDS}u+(g.x|(g.y<<6u)|(g.z<<12u))],inFlat);}
 let e=bitcast<vec3u>(world>>vec3u(2u))&vec3u(63u);let l=bitcast<vec3u>(world)&vec3u(3u);
 return ${buffer}[${table}+${UNIFORM_DIRECTORY_TOP_WORDS}u+64u*min(${buffer}[${table}+(e.x|(e.y<<6u)|(e.z<<12u))],${UNIFORM_SUPER_BLOCK_SLOTS - 1}u)+(l.x|(l.y<<2u)|(l.z<<4u))];
}
fn ${prefix}PageWord(page:vec3i)->u32{
 let world=page+${prefix}FramePage();let raw=${prefix}PageTableWord(page);
 let keep=all(bitcast<vec3u>(page)<=vec3u(${w(30)}))&(raw!=${w(31)});
 // A kept word is a block or AIR wherever the host's SOLID seed reads as AIR
 // (word 31 all ones): clamped below the SOLID sentinel there, a SOLID test
 // of a fixed geometry folds to the world-box compare and drops its load.
 // A tagged BULK16 word (0x80000000 | slot) is the only one above the
 // blocks there: with the slot store (BulkSlots) the clamp maps it to
 // blocks-1, which no block holds (umBlockStored). The proof that a kept
 // word is below the SOLID sentinel stands, and the kind needs no tag
 // decode (umBulkWord). Its slot is PageRawWord's.
 let word=select(raw,min(raw,select(${blocks}-2u,${blocks}-1u,${prefix}BulkSlotsOn())),${prefix}SolidIsBox());
 return select(select(0u,word,keep),0xffffffffu,${prefix}WorldSolid(world));
}
// The raw word: a BULK16 page keeps its tag, for the readers that need its
// slot (the pressure arena's).
fn ${prefix}PageRawWord(page:vec3i)->u32{
 let world=page+${prefix}FramePage();let raw=${prefix}PageTableWord(page);
 let keep=all(bitcast<vec3u>(page)<=vec3u(${w(30)}))&(raw!=${w(31)});
 return select(select(0u,raw,keep),0xffffffffu,${prefix}WorldSolid(world));
}
// The SOLID pages are exactly those outside the world box: a fixed geometry
// whose host SOLID seed reads as AIR (word 31 all ones).
fn ${prefix}SolidIsBox()->bool{return ${prefix}FixedWorldGeometry&&${w(31)}==0xffffffffu;}
fn ${prefix}BlockWord(b:u32)->u32{
 let raw=${buffer}[${table}+${entries + UNIFORM_WINDOW_HEADER_WORDS}u+b];
 let q=(vec3u(raw&255u,(raw>>8u)&255u,(raw>>16u)&255u)-bitcast<vec3u>(${prefix}FramePage()))&vec3u(255u);
 return select(q.x|(q.y<<8u)|(q.z<<16u),raw,raw==0xffffffffu);
}`;
}
