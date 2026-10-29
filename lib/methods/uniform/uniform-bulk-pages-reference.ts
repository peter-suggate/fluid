/** The storage ABI of a whole 32h page represented by eight 16h cells.
 * Page words carry a slot, never a 4h block index. Slot zero is reserved;
 * the all-ones SOLID word is outside the encoding's range. */
export const BULK16_PAGE_TAG=0x80000000;
export const BULK16_PAGE_SLOT_MASK=0x3fffffff;
// A slot's record: its last field (promoteLevel) ends at word 948. With the three
// list words a slot (free, live, quarantine) a demoted page costs 3852 B.
export const BULK16_PAGE_WORDS=960;
export const BULK16_PAGE={
 // promote: nonzero when this frame asks the page back to 4h (the next
 // commit's decide): 1 the structural scan (bulkPromote), 2 the error test
 // (the arena's pBulkRows). A page the particles' closure reaches (lrAsk's
 // upPromote) leaves at that commit with the word 0. A retired slot keeps it
 // until reused.
 page:0,flags:3,generation:4,open:5,promote:6,
 // A record of the cause, read by no kernel. When the error test sets
 // promote 2: its measured error and the level it exceeded (twice the
 // member tolerance), as f32 (m/s). When the scan sets promote 1:
 // promoteError holds the BULK16_SCAN bits of the tests that failed.
 promoteError:7,
 // The pressure solve's rows (UniformPressureArena): the eight 16h
 // pressures are pressure + pressureNext (the solve's correction, folded in
 // when the next solve starts); rhs is the composite rows' (their
 // diagonal and couplings are tail4's channels). velocityNext holds the
 // face means' u*, velocity the projected ones.
 pressure:8,velocity:16,velocityNext:52,pressureNext:88,
 rhs:96,
 // The page's 4h vertex block as it stood (9^3 texels, x fastest, its
 // skirt included): deep phi is a signed distance, not a constant, so a
 // promote restores it exactly. Seam heads are hydrostatic, not stored.
 phi:104,
 // The page's rows on the arena's tail levels, slot-indexed (no pool
 // block): level 3's iterate pair and RHS for the eight 16h cells (a 16h
 // cell is its own 8h-level unknown), then the 16h and 32h rows' channels
 // (X, Y, R, D, K0..K5 in the arena's tail channel order), eight words a
 // channel at 16h, one at 32h.
 tail3:833,tail4:857,tail5:937,promoteLevel:947,
} as const;
/** The structural scan's failed tests (bulkPromote), BULK16_PAGE.promoteError's
 * bits when promote is 1: an AIR page in its 3^3 neighbourhood; within 4
 * tiles an h band tile, a HOT region, a cut tile, or a tile within
 * UNIFORM_BULK_DEPTH_CELLS of the surface; or its own face mean's speed. */
export const BULK16_SCAN={air:1,band:2,hot:4,cut:8,shallow:16,speed:32} as const;
/** Words of BULK16_PAGE.tail4 and tail5 a channel. */
export const BULK16_TAIL_CHANNELS=10;
/** Texels of a page's 4h vertex block (BULK16_PAGE.phi). */
export const BULK16_PHI_TEXELS=729;

export function bulk16PageWord(slot:number):number{
 if(!Number.isSafeInteger(slot)||slot<1||slot>BULK16_PAGE_SLOT_MASK)throw new Error("Invalid BULK16 slot");
 return (BULK16_PAGE_TAG|slot)>>>0;
}
export function bulk16PageSlot(word:number):number{
 return ((word>>>30)===2&&(word&BULK16_PAGE_SLOT_MASK)!==0)?word&BULK16_PAGE_SLOT_MASK:0;
}
/** MAC face of the page's 2x2x2 cells. The normal coordinate is 0..2;
 * tangential coordinates are 0..1. Twelve faces per component, 36 total. */
export function bulk16FaceIndex(axis:number,normal:number,u:number,v:number):number{
 if(!Number.isInteger(axis)||axis<0||axis>2||!Number.isInteger(normal)||normal<0||normal>2||
  !Number.isInteger(u)||u<0||u>1||!Number.isInteger(v)||v<0||v>1)throw new Error("Invalid BULK16 face");
 return 12*axis+4*normal+u+2*v;
}
export const bulk16FaceIndexWGSL=/* wgsl */`
fn bpFace(axis:u32,normal:u32,u:u32,v:u32)->u32{return 12u*axis+4u*normal+u+2u*v;}
fn bpSlot(word:u32)->u32{return select(0u,word&0x3fffffffu,(word>>30u)==2u);}
`;

/** The 16h transfer on the GPU (the twin of restrictBulk16Page and
 * prolongBulk16Page, in f32). The including module supplies the page's 4h
 * fields: bpFineFace(axis,n,i,j) and bpFinePressure(c) (n the face plane
 * 0..8 along axis, i and j its tangential tiles in axis order; c a tile),
 * and for a promote bpStoreFineFace and bpStoreFinePressure. Sums are
 * pairwise: equal subfaces (what a prolongation leaves on a 16h face) sum
 * exactly, so demote after promote returns every face mean bit for bit and a
 * 16h face's flux is its sixteen subfaces' sum. Needs bulk16FaceIndexWGSL. */
export const bulk16TransferWGSL=/* wgsl */`
fn bpPair16(v:ptr<function,array<f32,16>>)->f32{
 for(var s=8u;s>0u;s>>=1u){for(var i=0u;i<s;i++){(*v)[i]+=(*v)[i+s];}}
 return (*v)[0];
}
// Face mean k (bpFace order) of a page's 4h faces: its sixteen subfaces on plane 4*normal.
fn bpRestrictFace(k:u32)->f32{
 let axis=k/12u;let normal=(k/4u)%3u;let u=k&1u;let v=(k>>1u)&1u;var s:array<f32,16>;
 for(var j=0u;j<16u;j++){s[j]=bpFineFace(axis,4u*normal,4u*u+(j&3u),4u*v+(j>>2u));}
 return bpPair16(&s)/16.0;
}
// Pressure mean of 16h cell r (x + 2y + 4z): its 64 tiles, pairwise.
fn bpRestrictPressure(r:u32)->f32{
 let o=4u*vec3u(r&1u,(r>>1u)&1u,r>>2u);var q:array<f32,16>;
 for(var z=0u;z<4u;z++){
  var s:array<f32,16>;
  for(var j=0u;j<16u;j++){s[j]=bpFinePressure(o+vec3u(j&3u,j>>2u,z));}
  q[z]=bpPair16(&s);
 }
 for(var z=4u;z<16u;z++){q[z]=0.0;}
 return bpPair16(&q)/64.0;
}
// A promote's 4h face on plane n (0..8) at tangential tiles (i, j): linear
// along the normal between the 16h face means it lies between, which it
// equals on planes 0, 4 and 8 (the current umBulkRow rule).
fn bpProlongFace(face:ptr<function,array<f32,36>>,axis:u32,n:u32,i:u32,j:u32)->f32{
 let region=min(1u,n>>2u);let fraction=f32(n-4u*region)/4.0;
 let lo=(*face)[bpFace(axis,region,i>>2u,j>>2u)];let hi=(*face)[bpFace(axis,region+1u,i>>2u,j>>2u)];
 // A 16h face's planes take its mean as stored: the compiler may refactor
 // the blend (lo+f*(hi-lo)), which is not exact at f=1.
 if(n==8u){return hi;}if((n&3u)==0u){return lo;}
 return (1.0-fraction)*lo+fraction*hi;
}
// A promote's 4h pressure at tile c: its 16h cell's plus the physical
// hydrostatic gradient (per h) at the tile's offset from the cell centre.
fn bpProlongPressure(pressure:ptr<function,array<f32,8>>,c:vec3u,gradient:vec3f)->f32{
 let r=(c.x>>2u)+2u*((c.y>>2u)+2u*(c.z>>2u));let o=vec3f(c&vec3u(3u))-vec3f(1.5);
 return (*pressure)[r]+4.0*dot(gradient,o);
}
`;

/** Restrict a page's 4h MAC faces (three component-major 9x8x8 arrays,
 * normal coordinate fastest) by area, and its 8^3 pressures by volume. */
export function restrictBulk16Page(faces:ArrayLike<number>,pressure:ArrayLike<number>):{faces:Float64Array;pressure:Float64Array}{
 if(faces.length!==3*9*8*8||pressure.length!==512)throw new Error("A 4h page has 1728 faces and 512 pressures");
 const coarse=new Float64Array(36),p=new Float64Array(8),s=new Float64Array(16),q=new Float64Array(16);
 // Pairwise, as bulk16TransferWGSL: equal subfaces sum exactly.
 const pair=(v:Float64Array)=>{for(let w=8;w>0;w>>=1)for(let i=0;i<w;i++)v[i]+=v[i+w]!;return v[0]!;};
 for(let a=0;a<3;a++)for(let n=0;n<3;n++)for(let v=0;v<2;v++)for(let u=0;u<2;u++){
  for(let j=0;j<16;j++)s[j]=faces[a*576+4*n+9*(4*u+(j&3)+8*(4*v+(j>>2)))]!;
  coarse[bulk16FaceIndex(a,n,u,v)]=pair(s)/16;
 }
 for(let r=0;r<8;r++){
  const o=[4*(r&1),4*((r>>1)&1),4*(r>>2)];q.fill(0);
  for(let z=0;z<4;z++){for(let j=0;j<16;j++)s[j]=pressure[o[0]!+(j&3)+8*(o[1]!+(j>>2)+8*(o[2]!+z))]!;q[z]=pair(s);}
  p[r]=pair(q)/64;
 }
 return {faces:coarse,pressure:p};
}

/** Conservative normal-linear prolongation: each 16h face's sixteen 4h
 * subfaces retain its mean exactly. Pressure is the cell mean plus the
 * supplied physical hydrostatic gradient, sampled at each 4h centre. */
export function prolongBulk16Page(faces:ArrayLike<number>,pressure:ArrayLike<number>,gradient:readonly [number,number,number]=[0,0,0]):{faces:Float64Array;pressure:Float64Array}{
 if(faces.length!==36||pressure.length!==8)throw new Error("A BULK16 page has 36 faces and 8 pressures");
 const fine=new Float64Array(1728),p=new Float64Array(512);
 for(let a=0;a<3;a++)for(let v=0;v<8;v++)for(let u=0;u<8;u++)for(let n=0;n<9;n++){
  const region=Math.min(1,n>>2),fraction=(n-4*region)/4;
  const lo=faces[bulk16FaceIndex(a,region,u>>2,v>>2)]!,hi=faces[bulk16FaceIndex(a,region+1,u>>2,v>>2)]!;
  fine[a*576+n+9*(u+8*v)]=(1-fraction)*lo+fraction*hi;
 }
 for(let z=0;z<8;z++)for(let y=0;y<8;y++)for(let x=0;x<8;x++){
  p[x+8*(y+8*z)]=pressure[(x>>2)+2*((y>>2)+2*(z>>2))]!+
   4*(gradient[0]*((x&3)-1.5)+gradient[1]*((y&3)-1.5)+gradient[2]*((z&3)-1.5));
 }
 return {faces:fine,pressure:p};
}

/** Slot records plus free, live and quarantine lists; every capacity is
 * fixed at construction. No allocation is proportional to 4h rows. */
export function bulk16PageLayout(capacity:number){
 if(!Number.isSafeInteger(capacity)||capacity<1||capacity>BULK16_PAGE_SLOT_MASK)throw new Error("Invalid BULK16 capacity");
 const free=32,live=free+capacity,pending=live+capacity,records=Math.ceil((pending+capacity)/64)*64;
 const words=records+BULK16_PAGE_WORDS*(capacity+1);
 if(!Number.isSafeInteger(4*words))throw new Error("BULK16 storage is too large");
 return {capacity,free,live,pending,records,words,bytes:4*words};
}

/** Shared read-only binding for simulation and pressure ownership. Slot
 * identity is in the directory; no reader may use it as a pool block. */
export function bulk16ReadWGSL(layout:ReturnType<typeof bulk16PageLayout>,group:number,binding:number):string{return /* wgsl */`
@group(${group}) @binding(${binding}) var<storage,read> umBulkPages:array<u32>;
fn umBulkRead(slot:u32,word:u32)->u32{return umBulkPages[${layout.records}u+${BULK16_PAGE_WORDS}u*slot+word];}
fn umBulkPressure(slot:u32,cell:vec3u)->f32{return bitcast<f32>(umBulkRead(slot,${BULK16_PAGE.pressure}u+cell.x+2u*(cell.y+2u*cell.z)));}
fn umBulkVelocityFace(slot:u32,axis:u32,normal:u32,u:u32,v:u32)->f32{return bitcast<f32>(umBulkRead(slot,${BULK16_PAGE.velocity}u+12u*axis+4u*normal+u+2u*v));}
`;}
