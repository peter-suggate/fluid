/** A canonical vertex home: 26-bit physical block, then three two-bit
 * displacements (0 same, 1 positive, 2 negative). One record serves both
 * the h and 4h vertex atlases. Pool blocks are bounded well below 2^26. */
export const UNIFORM_ADDRESS_HOME_MASK = 0x03ffffff;
export const UNIFORM_ADDRESS_PLAN_WORDS = 54;

export function uniformAddressHome(rows: ArrayLike<number>, blocks: number, block: number, family: number): number {
  const side = [family % 3, Math.floor(family / 3) % 3, Math.floor(family / 9)];
  const own = side.map(s => s === 0 ? -1 : 0);
  for (let k = 0; k < 8; k++) {
    if (side.some((s, a) => s === 1 && (k & (1 << a)))) continue;
    const d = own.map((v, a) => v + ((k >> a) & 1));
    const b = rows[27 * block + d[0]! + 3 * d[1]! + 9 * d[2]! + 13]!;
    if (b > 0 && b < blocks - 1) {
      let word = b;
      for (let a = 0; a < 3; a++) word |= (d[a] === -1 ? 2 : d[a]!) << (26 + 2 * a);
      return word >>> 0;
    }
  }
  return 0;
}

/** Appended to each topology's existing page mirror; no new binding.
 * Read-only during simulation. The directory remains the general fallback
 * for queries beyond the loaded page's one-page neighborhood. */
export const uniformAddressPlanWGSL = /* wgsl */ `
override umUseAddressPlan:bool=false;
const UM_ADDRESS_HOMES:u32=UM_BLOCK_RECORD+34u*UM_BLOCKS+1u;
var<workgroup> umPlanNeighbors:array<u32,27>;
var<workgroup> umPlanHomes:array<u32,27>;
var<workgroup> umPlanPage:vec3i;
var<workgroup> umPlanBlock:u32;
var<workgroup> umPlanReady:u32;
// All lanes call this, including inactive lanes. Only the first lane's
// block is used; callers may share a plan even when jobs span several pages.
fn umLoadAddressPlan(block:u32,lane:u32){
 if(!umUseAddressPlan){return;}
 workgroupBarrier();
 if(lane==0u){
  umPlanBlock=block;umPlanReady=0u;
  if(umBlockStored(block)&&umBlockWord(block)!=UM_FREE){umPlanPage=vec3i(umBlockPage(block));umPlanReady=u32(umDirectoryPageWord(umPlanPage)==block);}
 }
 workgroupBarrier();
 if(umPlanReady!=0u){
  if(lane<27u){
   umPlanNeighbors[lane]=umTopology[UM_BLOCK_RECORD+UM_BLOCKS+27u*umPlanBlock+lane];
   umPlanHomes[lane]=umTopology[UM_ADDRESS_HOMES+27u*umPlanBlock+lane];
  }
 }
 workgroupBarrier();
}
fn umPageWord(page:vec3i)->u32{
 if(umUseAddressPlan&&umPlanReady!=0u){
  let d=page-umPlanPage;
  if(all(abs(d)<=vec3i(1))){return umPlanNeighbors[u32(d.x+3*d.y+9*d.z+13)];}
 }
 return umDirectoryPageWord(page);
}
fn umPlannedVertexSlot(v:vec3i,bits:u32)->vec4u{
 if(umUseAddressPlan&&umPlanReady!=0u){
  let edge=1<<bits;let local=v-(umPlanPage<<vec3u(bits));
  if(all(local>=vec3i(0))&&all(local<=vec3i(edge))){
   if(all(local>vec3i(0))){return vec4u(vec3u(local),umPlanBlock);}
   let family=select(select(vec3u(1),vec3u(2),local==vec3i(edge)),vec3u(0),local==vec3i(0));
   let word=umPlanHomes[family.x+3u*(family.y+3u*family.z)];
   let code=(vec3u(word)>>vec3u(26u,28u,30u))&vec3u(3u);
   let delta=select(vec3i(code),vec3i(-1),code==vec3u(2u));
   return vec4u(vec3u(local-edge*delta),word&0x03ffffffu);
  }
 }
 return vec4u(0xffffffffu);
}
`;

/** Every integrated consumer, used by the benchmark's per-stage ablations. */
export const UNIFORM_ADDRESS_PLAN_CANDIDATES = [
  "depthSeeds", "depthGeometry", "buildDistance", "surfaceSplatOccupied",
  "coupleFine", "transferProperties", "advectFine", "momentumUnitStep",
  "forcesRegular", "forcesRegularCoarse", "forces", "momentumStep", "project", "publishList",
] as const;

/** The integrated generic plan currently regresses the splash workload.
 * Keep its readers compiled out in production until a consumer demonstrates
 * a net saving including rebuild. The benchmark explicitly opts stages in;
 * keeping the same backing layout permits synchronized-state comparisons. */
export const UNIFORM_ADDRESS_PLAN_ENTRIES: ReadonlySet<string> = new Set<string>();
