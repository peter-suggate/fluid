/** The census start's BULK16 pages (docs/plans/bulk16-plan-2026-10-28.md, V2c).
 *
 * A start on the census (fluid.tiers.start "census") knows its first
 * frame's state on the host: the initial liquid at rest or in uniform
 * motion, its h tiles by the census rule (uniformCensusSeedTiles). The
 * pages the bulk tier would admit from that state (UniformBulkTier's
 * bulkSelect: every region of the page structural-clean, its speed inside
 * the reach and its restrict-prolong error under the threshold) start in
 * BULK16 slots, never in a pool block: the state decide and demote would
 * write at the first commit is written on the host instead (the oracle of
 * tests/uniform-bulk-start-gpu.test.ts). The 30-frame entry hold is a
 * guard against a transient verdict; the t=0 state has no history to be
 * transient against.
 *
 * Cost: a candidate page (stored, holding liquid) reads its regions' 12^3
 * tile neighbourhoods, each tile once (memoised), so the work is the
 * candidates' and their neighbours' tiles, never the box's. */
import {
  UNIFORM_BULK_FLOOR, UNIFORM_BULK_REACH_CELLS, UNIFORM_BULK_TOLERANCE, uniformBulkAxisError, uniformBulkRegionDeep, uniformBulkStructural,
  type BulkVec, type UniformBulkTile,
} from "./uniform-bulk-tier-reference";
import { BULK16_PAGE, BULK16_PAGE_WORDS, BULK16_PHI_TEXELS, restrictBulk16Page } from "./uniform-bulk-pages-reference";
import type { LatticePages } from "../../core/initial-liquid-lattice";

/** What the D set reads, all on box coordinates (pages of 8^3 tiles, tiles of 4^3 cells). */
export interface UniformBulkStartSource {
 /** The box in pages. */
 readonly pages:readonly [number,number,number];
 /** Above an open top a page is AIR (claimable); elsewhere off the box it is SOLID. */
 readonly openTop:boolean;
 /** Whether box page (x,y,z) is stored at build (the pool's seed). */
 readonly stored:(x:number,y:number,z:number)=>boolean;
 /** The pages to test: stored pages holding liquid (one with none fails the deep test). */
 readonly candidates:Iterable<readonly [number,number,number]>;
 /** Whether box tile (x,y,z) is at h at t=0 (the census seed and authored Fine regions). */
 readonly fine:(x:number,y:number,z:number)=>boolean;
 /** Whether the 16h region (x,y,z) is HOT at t=0 from its own heat (UniformRegionHeat: a
  * body or source contact, tile detail, live particles); undefined: its h tiles alone. */
 readonly hot?:(x:number,y:number,z:number)=>boolean;
 /** Box tile (x,y,z)'s open fraction (rootTopology.x). */
 readonly open:(x:number,y:number,z:number)=>number;
 /** Box tile (x,y,z)'s centre phi (rootPhi of a full 4h owner: its corners' mean), m. */
 readonly phi:(x:number,y:number,z:number)=>number;
 /** Box tile (x,y,z)'s +face velocity on axis a, the 4h base's (m/s). */
 readonly face:(x:number,y:number,z:number,a:number)=>number;
 /** The cell (m) and the step (s). */
 readonly h:number;readonly dt:number;
}

/** The census start's BULK16 pages (box page coordinates, x fastest): those
 * whose eight regions bulkSelect admits on the t=0 state. `deep`, when
 * given, takes the candidates' deep regions (uniformBulkRegionDeep), keyed
 * x + 2px(y + 2py z). */
export function uniformBulkStartPages(s:UniformBulkStartSource,deep?:Set<number>):[number,number,number][]{
 const [px,py,pz]=s.pages,T=[8*px,8*py,8*pz] as const;
 const inBox=(t:BulkVec)=>t.every((v,a)=>v>=0&&v<T[a]!);
 const hot=new Map<number,boolean>(),tiles=new Map<number,UniformBulkTile|undefined>();
 // Keys over the box grown by a page (the neighbourhood reaches 4 tiles out).
 const key=(t:BulkVec)=>(t[0]+8)+(T[0]+16)*((t[1]+8)+(T[1]+16)*(t[2]+8));
 const regionHot=(r:BulkVec)=>{
  const k=r[0]+2*px*(r[1]+2*py*r[2]);let held=hot.get(k);
  if(held===undefined){
   held=s.hot?s.hot(r[0],r[1],r[2]):false;
   for(let i=0;i<64&&!held;i++)held=s.fine(4*r[0]+(i&3),4*r[1]+((i>>2)&3),4*r[2]+(i>>4));
   hot.set(k,held);
  }
  return held;
 };
 const kind=(t:BulkVec):UniformBulkTile["kind"]=>{
  if(inBox(t))return s.stored(t[0]>>3,t[1]>>3,t[2]>>3)?"resident":"air";
  return s.openTop&&t[1]>=T[1]&&t[0]>=0&&t[0]<T[0]&&t[2]>=0&&t[2]<T[2]?"air":"solid";
 };
 const tile=(t:BulkVec):UniformBulkTile|undefined=>{
  const k=key(t);if(tiles.has(k))return tiles.get(k);
  const c=kind(t);let out:UniformBulkTile|undefined;
  if(c!=="resident")out={kind:c};
  else{
   const band=s.fine(t[0],t[1],t[2]),warm=band||regionHot([t[0]>>2,t[1]>>2,t[2]>>2]);
   const open=warm?1:s.open(t[0],t[1],t[2]);
   out={kind:c,band,hot:warm&&!band,open,phi:warm||open<=1e-5?undefined:s.phi(t[0],t[1],t[2])};
  }
  tiles.set(k,out);return out;
 };
 // bFace: the +face of tile t on axis a, 0 where t is not stored (a wall).
 const face=(t:BulkVec,a:number)=>tile(t)?.kind==="resident"?s.face(t[0],t[1],t[2],a):0;
 const faces=new Float64Array(80);
 if(deep)for(const [x,y,z] of s.candidates)if(s.stored(x,y,z))for(let o=0;o<8;o++){
  const r:BulkVec=[2*x+(o&1),2*y+((o>>1)&1),2*z+(o>>2)];
  if(uniformBulkRegionDeep(tile,r,s.h))deep.add(r[0]+2*px*(r[1]+2*py*r[2]));
 }
 const admitted=(r:BulkVec)=>{
  if(!uniformBulkStructural(tile,r,s.h))return false;
  let speed=0,error=0;
  for(let a=0;a<3;a++){
   for(let k=0;k<5;k++)for(let j=0;j<16;j++){
    const t:BulkVec=[4*r[0],4*r[1],4*r[2]];t[(a+1)%3]+=j&3;t[(a+2)%3]+=j>>2;t[a]+=k-1;
    const u=face(t,a);faces[16*k+j]=u;speed=Math.max(speed,Math.abs(u));
   }
   error=Math.max(error,uniformBulkAxisError(faces));
  }
  return speed*s.dt<=UNIFORM_BULK_REACH_CELLS*s.h&&error<UNIFORM_BULK_TOLERANCE*speed+UNIFORM_BULK_FLOOR*s.h/s.dt;
 };
 const out:[number,number,number][]=[];
 for(const [x,y,z] of s.candidates){
  if(!s.stored(x,y,z))continue;
  let whole=true;
  for(let o=0;o<8&&whole;o++)whole=admitted([2*x+(o&1),2*y+((o>>1)&1),2*z+(o>>2)]);
  if(whole)out.push([x,y,z]);
 }
 return out.sort((p,q)=>p[2]-q[2]||p[1]-q[1]||p[0]-q[0]);
}

/** A box tile's +face velocity on axis a as the t=0 install leaves a 4h
 * tile's base (UniformDetailStorage's coarsenFaces): the mean of its 16 h
 * faces on the tile's high plane, summed in order in f32; none (0) on the
 * box's last plane. `velocity`: the initial h faces by page (a cell's +faces,
 * uniformInitialLiquidVelocityPages), undefined at rest. */
export function uniformBulkStartFace(velocity:LatticePages|undefined,x:number,y:number,z:number,a:number):number{
 if(!velocity)return 0;
 const E=velocity.edge,o=[4*x,4*y,4*z],d=velocity.cells,[px,py]=velocity.pages;
 if(o[a]!+3>=d[a]!)return 0;
 const src=velocity.at.get(Math.floor(o[0]!/E)+px*(Math.floor(o[1]!/E)+py*Math.floor(o[2]!/E)));if(!src)return 0;
 const u=(a+1)%3,v=(a+2)%3,q=[0,0,0];let sum=0;
 for(let j=0;j<4;j++)for(let i=0;i<4;i++){q[a]=o[a]!+3;q[u]=o[u]!+i;q[v]=o[v]!+j;sum=Math.fround(sum+src[4*((q[0]!%E)+E*((q[1]!%E)+E*(q[2]!%E)))+a]!);}
 return Math.fround(sum/16);
}
/** A box tile's centre phi as the geometry pass gives a full 4h owner
 * (gcEvaluate): its eight corners' 4h vertex phi, each an eighth, summed as
 * d4Sum8 in f32. `phi`: vertex phi by page (fluidLatticePagedVertexPhi), on
 * the tile's own page. */
export function uniformBulkStartCentrePhi(phi:LatticePages,x:number,y:number,z:number):number{
 const E=phi.edge,n=E+1,[px,py]=phi.pages,p=[x>>3,y>>3,z>>3],page=phi.at.get(p[0]!+px*(p[1]!+py*p[2]!));
 if(!page)throw new Error(`BULK16 start: tile (${x}, ${y}, ${z}) has no vertex phi`);
 const c=new Array<number>(8);
 for(let j=0;j<8;j++){const i=4*(x&7)+4*(j&1),k=4*(y&7)+4*((j>>1)&1),l=4*(z&7)+4*(j>>2);c[j]=Math.fround(page[i+n*(k+n*l)]!*0.125);}
 const f=Math.fround;
 return f(f(f(c[0]!+c[5]!)+f(c[1]!+c[4]!))+f(f(c[2]!+c[7]!)+f(c[3]!+c[6]!)));
}
/** The D set's source from what a census start holds at build: the stored
 * set, the initial cell volumes, vertex phi and h faces by page, the t=0 h
 * tiles and whether a lattice cell is open. */
export function uniformBulkStartSource(o:{
 readonly pages:readonly [number,number,number];readonly openTop:boolean;
 readonly stored:(x:number,y:number,z:number)=>boolean;
 readonly volume:LatticePages;readonly phi:LatticePages;readonly velocity?:LatticePages;
 readonly fine:(x:number,y:number,z:number)=>boolean;readonly hot?:(x:number,y:number,z:number)=>boolean;
 readonly cellOpen:(x:number,y:number,z:number)=>boolean;readonly h:number;readonly dt:number;
}):UniformBulkStartSource{
 const [px,py]=o.volume.pages,candidates:[number,number,number][]=[];
 for(const i of o.volume.at.keys())candidates.push([i%px,Math.floor(i/px)%py,Math.floor(i/(px*py))]);
 return {pages:o.pages,openTop:o.openTop,stored:o.stored,candidates,fine:o.fine,hot:o.hot,h:o.h,dt:o.dt,
  open:(x,y,z)=>{let n=0;for(let k=0;k<64;k++)if(o.cellOpen(4*x+(k&3),4*y+((k>>2)&3),4*z+(k>>4)))n++;return n/64;},
  phi:(x,y,z)=>uniformBulkStartCentrePhi(o.phi,x,y,z),
  face:(x,y,z,a)=>uniformBulkStartFace(o.velocity,x,y,z,a)};
}
/** The census start's slot records (BULK16_PAGE_WORDS each, the i-th
 * page's in slot i + 1): what the first commit's claim (bpClaim) and demote
 * (btDemotePages) write into a slot from the t=0 state. Faces and pressure go
 * through restrictBulk16Page, the demote's CPU twin; the phi block is the
 * page's 4h vertex block as it stands (`phi`, texel x + 9(y + 9z)).
 * `stored`: whether a box page holds a block at that commit (the D pages
 * still do); a low neighbour that does not gives the low-face rows' 0. */
export function uniformBulkStartRecords(o:{
 readonly pages:readonly (readonly [number,number,number])[];readonly boxPage:readonly number[];
 readonly stored:(x:number,y:number,z:number)=>boolean;
 readonly face:(x:number,y:number,z:number,a:number)=>number;
 readonly phi:(page:readonly [number,number,number],texel:readonly [number,number,number])=>number;
}):Uint32Array<ArrayBuffer>{
 const W=BULK16_PAGE_WORDS,out=new Uint32Array(W*o.pages.length),f32=new Float32Array(out.buffer),i32=new Int32Array(out.buffer);
 const faces=new Float64Array(1728),pressure=new Float64Array(512),t:[number,number,number]=[0,0,0];
 o.pages.forEach((p,s)=>{
  const at=W*s;
  for(let a=0;a<3;a++)i32[at+BULK16_PAGE.page+a]=p[a]!+o.boxPage[a]!;
  out[at+BULK16_PAGE.flags]=1;out[at+BULK16_PAGE.generation]=1;out[at+BULK16_PAGE.open]=255;
  for(let a=0;a<3;a++){
   const u=(a+1)%3,v=(a+2)%3,low:[number,number,number]=[p[0],p[1],p[2]];low[a]--;
   const lowStored=low[a]>=0&&o.stored(low[0],low[1],low[2]);
   for(let j=0;j<8;j++)for(let i=0;i<8;i++)for(let n=0;n<9;n++){
    t[0]=8*p[0];t[1]=8*p[1];t[2]=8*p[2];t[u]+=i;t[v]+=j;t[a]+=n-1;
    faces[a*576+n+9*(i+8*j)]=n>0||lowStored?o.face(t[0],t[1],t[2],a):0;
   }
  }
  const r=restrictBulk16Page(faces,pressure);
  for(let k=0;k<36;k++)f32[at+BULK16_PAGE.velocity+k]=r.faces[k]!;
  for(let k=0;k<8;k++)f32[at+BULK16_PAGE.pressure+k]=r.pressure[k]!;
  for(let k=0;k<BULK16_PHI_TEXELS;k++)f32[at+BULK16_PAGE.phi+k]=o.phi(p,[k%9,Math.floor(k/9)%9,Math.floor(k/81)]);
 });
 return out;
}
/** The low-face rows a D page's demote writes into a stored high neighbour
 * (btDemotePages): one h row a 4h face of its low plane, the D page's own
 * high plane face. `block`: the block of a box page once the D pages hold
 * none (0: none). Each row goes to write(row, value). */
export function uniformBulkStartLowRows(o:{
 readonly pages:readonly (readonly [number,number,number])[];readonly box:readonly number[];
 readonly block:(x:number,y:number,z:number)=>number;
 readonly face:(x:number,y:number,z:number,a:number)=>number;
 readonly write:(row:number,value:number)=>void;
}):void{
 const t:[number,number,number]=[0,0,0],c:[number,number,number]=[0,0,0];
 for(const p of o.pages)for(let a=0;a<3;a++){
  const u=(a+1)%3,v=(a+2)%3,q:[number,number,number]=[p[0],p[1],p[2]];q[a]++;
  if(q[a]>=o.box[a]!)continue;
  const b=o.block(q[0],q[1],q[2]);if(!b)continue;
  for(let j=0;j<8;j++)for(let i=0;i<8;i++){
   t[0]=8*p[0];t[1]=8*p[1];t[2]=8*p[2];t[u]+=i;t[v]+=j;t[a]+=7;
   c[0]=c[1]=c[2]=0;c[u]=4*i;c[v]=4*j;
   const uv=a===0?[c[1],c[2]]:a===1?[c[0],c[2]]:[c[0],c[1]];
   o.write(((b*3+a)<<10)|uv[0]!|(uv[1]!<<5),o.face(t[0],t[1],t[2],a));
  }
 }
}
/** V2d, the promote reserve for closure and scan promotes (docs/plans/
 * bulk16-plan-2026-10-28.md, the pool rule): those of the start's BULK16
 * pages (`bulk`) within `reach` cells (Chebyshev, cell to cell) of the
 * liquid's surface at build (`surface`: by box page index, the 4h tile
 * bounds [x0,y0,z0,x1,y1,z1] of its wet tiles beside open dry ones;
 * uniformPagePoolResidency). The surface band or air reaches a deep page no
 * faster than the scene's speed v (uniformCensusReach), and a closure asks
 * the pages within its margin of the band, so within a hold window T those
 * within v T plus the closure margin are the pages a closure or scan promote
 * can take back before any page it promoted can demote again: finite
 * propagation, as the ballistic reserve. `edge`: a page's cells. The work
 * follows the surface pages, never the box. */
export function uniformBulkStartShell(pages:readonly [number,number,number],surface:ReadonlyMap<number,readonly number[]>,bulk:readonly (readonly [number,number,number])[],reach:number,edge:number):number{
 const [px,py,pz]=pages,size=[px,py,pz],near=new Set<number>();
 for(const b of surface.values()){
  // Per axis, the pages whose cells come within reach of the bounds' cells [4 b0, 4 b1 + 3].
  const range=[0,1,2].map(a=>{const lo=4*b[a]!,hi=4*b[a+3]!+3;return [Math.max(0,Math.ceil((lo-reach-edge+1)/edge)),Math.min(size[a]!-1,Math.floor((hi+reach)/edge))] as const;});
  for(let z=range[2]![0];z<=range[2]![1];z++)for(let y=range[1]![0];y<=range[1]![1];y++)for(let x=range[0]![0];x<=range[0]![1];x++)near.add(x+px*(y+py*z));
 }
 let held=0;for(const [x,y,z] of bulk)if(near.has(x+px*(y+py*z)))held++;
 return held;
}
