/** The one brick table of the NB-FLIP stages (surface arena, particle
 * bricks). A brick is 8³ cells of a page: a page is 4³ bricks, a brick 2³
 * tiles. Its id is 64·block + local, local = bx | by<<2 | bz<<4.
 *
 * Every stage that works brick by brick marks the bricks it needs with its
 * class bits; one build then gives every marked brick one slot of a single
 * slot space, its class bits and its 27-slot neighbour record. A core class
 * dilates by one brick into its ring class (arena core → arena, occupied →
 * particles): the build ORs a neighbour's core bits, shifted up one, into a
 * brick's class. The other classes are plain marks.
 *
 * Words (from a base in an atomic<u32> buffer; B = 64·UM_BLOCKS):
 * [0] slot count, [1..3] spare; mark (B, by brick id); table (B, slot or
 * UB_NONE by brick id); list (B, brick id by slot); class (B, by slot);
 * record (27·B, neighbour slot or UB_NONE, offset x+3y+9z of (-1..1)³).
 * The capacity is every brick of the lattice, allocated upfront: the slot
 * space cannot overflow, whatever the classes.
 *
 * Build: clear the count and marks (uniformBrickTableClear), launch the
 * markers, then brickTableList and brickTableRecord. A rebuild drops every
 * class's slots: all marks of a frame go before the one build that serves
 * them. */
export const UNIFORM_BRICK_CLASS={arenaCore:1,arena:2,particlesCore:4,particles:8,solidClear:16,blend:32,rigid:64} as const;
export const UNIFORM_BRICK_TABLE_ENTRIES=["brickTableList","brickTableRecord"] as const;
/** Words the table takes in its buffer, of a lattice of blocks blocks. */
export function uniformBrickTableWords(blocks:number):number{return 4+31*64*blocks;}
/** The byte range build clears (count and marks), of a table at word base. */
export function uniformBrickTableClear(base:number,blocks:number):{offset:number;size:number}{return {offset:4*base,size:4*(4+64*blocks)};}
/** buffer: an array<atomic<u32>> the module binds; base: a WGSL u32
 * expression for the table's first word. Needs the lattice address. */
export function uniformBrickTableWGSL(buffer:string,base:string):string{return /* wgsl */`
const UB_BRICKS:u32=64u*UM_BLOCKS;
const UB_NONE:u32=0xffffffffu;
const UB_COUNT:u32=${base};
const UB_MARK:u32=UB_COUNT+4u;
const UB_TABLE:u32=UB_MARK+UB_BRICKS;
const UB_LIST:u32=UB_TABLE+UB_BRICKS;
const UB_CLASS:u32=UB_LIST+UB_BRICKS;
const UB_RECORD:u32=UB_CLASS+UB_BRICKS;
${Object.entries(UNIFORM_BRICK_CLASS).map(([name,bit])=>`const UB_${name.replace(/[A-Z]/g,c=>`_${c}`).toUpperCase()}:u32=${bit}u;`).join("")}
// Core classes: each dilates into the class one bit above it.
const UB_CORES:u32=UB_ARENA_CORE|UB_PARTICLES_CORE;
// The brick of tile key t, and the brick holding cell c (UB_NONE where its page holds nothing).
fn ubBrickOfTile(t:u32)->u32{return 64u*(t>>9u)+(((t>>1u)&3u)|(((t>>4u)&3u)<<2u)|(((t>>7u)&3u)<<4u));}
fn ubBrickAt(c:vec3i)->u32{let b=umPageWord(c>>vec3u(5u));if(!umBlockStored(b)){return UB_NONE;}let l=(vec3u(c)>>vec3u(3u))&vec3u(3u);return 64u*b+(l.x|(l.y<<2u)|(l.z<<4u));}
// Its first cell.
fn ubBrickOrigin(id:u32)->vec3u{let l=id&63u;return (umBlockPage(id>>6u)<<vec3u(5u))+8u*vec3u(l&3u,(l>>2u)&3u,l>>4u);}
fn ubMark(id:u32,bits:u32){if((atomicLoad(&${buffer}[UB_MARK+id])&bits)!=bits){atomicOr(&${buffer}[UB_MARK+id],bits);}}
fn ubCount()->u32{return min(atomicLoad(&${buffer}[UB_COUNT]),UB_BRICKS);}
fn ubSlot(id:u32)->u32{return atomicLoad(&${buffer}[UB_TABLE+id]);}
fn ubBrick(slot:u32)->u32{return atomicLoad(&${buffer}[UB_LIST+slot]);}
fn ubClass(slot:u32)->u32{return atomicLoad(&${buffer}[UB_CLASS+slot]);}
fn ubNeighbour(slot:u32,n:u32)->u32{return atomicLoad(&${buffer}[UB_RECORD+27u*slot+n]);}
// A thread a brick id: dilate the core marks, take a slot, list it.
@compute @workgroup_size(64) fn brickTableList(@builtin(global_invocation_id) gid:vec3u,@builtin(num_workgroups) groups:vec3u){
 for(var id=gid.x;id<UB_BRICKS;id+=64u*groups.x){
  let block=id>>6u;var bits=0u;
  if(umBlockStored(block)&&umBlockWord(block)!=UM_FREE){
   bits=atomicLoad(&${buffer}[UB_MARK+id]);let origin=vec3i(ubBrickOrigin(id));
   for(var n=0u;n<27u;n++){
    let other=ubBrickAt(origin+8*(vec3i(umCorner(n,3u))-1));
    if(other!=UB_NONE){bits|=(atomicLoad(&${buffer}[UB_MARK+other])&UB_CORES)<<1u;}
   }
  }
  if(bits==0u){atomicStore(&${buffer}[UB_TABLE+id],UB_NONE);continue;}
  let slot=atomicAdd(&${buffer}[UB_COUNT],1u);
  atomicStore(&${buffer}[UB_TABLE+id],slot);atomicStore(&${buffer}[UB_LIST+slot],id);atomicStore(&${buffer}[UB_CLASS+slot],bits);
 }
}
// A thread a (slot, neighbour).
@compute @workgroup_size(64) fn brickTableRecord(@builtin(global_invocation_id) gid:vec3u,@builtin(num_workgroups) groups:vec3u){
 let n=27u*ubCount();
 for(var i=gid.x;i<n;i+=64u*groups.x){
  let slot=i/27u;let k=i-27u*slot;
  let other=ubBrickAt(vec3i(ubBrickOrigin(ubBrick(slot)))+8*(vec3i(umCorner(k,3u))-1));
  atomicStore(&${buffer}[UB_RECORD+i],select(UB_NONE,ubSlot(other),other!=UB_NONE));
 }
}
`;}
