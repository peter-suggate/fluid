import { fluidLatticePagedCellVolumes, type LatticePages } from "../../core/initial-liquid-lattice";
import type { SceneDescription } from "../../core/model";
import { proceduralSolidField } from "../../core/procedural-solid";
import { sceneLatticeDimensions } from "../../core/scene-lattice-dimensions";
import { sampleSolidWorld, solidWorldForScene } from "../../core/solid-world";
import { terrainColumnHeights } from "../../core/terrain";
import { liquidExtrusionBounds, liquidExtrusionDistance, liquidExtrusionEdges } from "../../core/liquid-extrusion";
import { integratedInflowVolume } from "../../core/inflow-boundary";
import { UNIFORM_PAGE_POOL_EDGE, UNIFORM_PAGE_POOL_SEED_MARGIN, uniformPagePoolRuleBlocks } from "./uniform-page-pool";
import { UNIFORM_FINE_SLOT_GROWTH } from "./uniform-fine-slots";

/** The particle store's rule (docs/plans/bulk16-plan-2026-10-28.md, M4b):
 * the samples a band cell holds (Ferstl et al. 2016, §3.3: a cell under n = 8
 * unit samples is reseeded) x the band cells of the census liquid x
 * UNIFORM_FINE_SLOT_GROWTH, the one growth constant. Reseeding is the only
 * thing that creates records, and an overlap reseed past the rule's records
 * is deferred (reseedClipped), so the rule bounds the records exactly. */
export const UNIFORM_PARTICLE_SAMPLES_PER_CELL=8;
export function uniformParticleRecordRule(bandCells:number):number{
 if(!(Number.isSafeInteger(bandCells)&&bandCells>=0))throw new RangeError(`A particle record rule needs a count of band cells, not ${bandCells}`);
 return Math.ceil(UNIFORM_FINE_SLOT_GROWTH*UNIFORM_PARTICLE_SAMPLES_PER_CELL*bandCells);
}
/** A census band: its cells, and the h tiles (4 cells an edge) holding one. */
export interface UniformParticleBand{readonly cells:number;readonly tiles:number}
const tileKey=(cells:readonly [number,number,number],x:number,y:number,z:number)=>(x>>2)+Math.ceil(cells[0]/4)*((y>>2)+Math.ceil(cells[1]/4)*(z>>2));

/** The band cells of a liquid given by page (LatticePages of cell volumes):
 * the wet cells (volume above zero) within `depth` cells, on every axis, of a
 * dry cell of the lattice. Past the lattice counts as wet (a wall is no
 * surface); a cell inside a solid holds no volume and counts as dry, so a
 * solid's contact is counted too (an over-count, never an under-count). A
 * page that is wet throughout, with every page around it wet throughout or
 * past the lattice, holds no band cell and is not visited. `deep`, when
 * given, says whether the 16h region (cell >> 4 on each axis) is deep at t=0
 * (uniformBulkRegionDeep): its band cells are solid contact alone, where no
 * particle lives (no h tile, no free surface within the band), and are not
 * counted. */
export function uniformParticleBand(volume:LatticePages,depth:number,tiles=new Set<number>(),deep?:(x:number,y:number,z:number)=>boolean):UniformParticleBand{
 if(!(Number.isInteger(depth)&&depth>=1&&depth<volume.edge))throw new RangeError(`A particle band of ${depth} cells needs a whole depth under the page edge ${volume.edge}`);
 const E=volume.edge,[nx,ny,nz]=volume.cells,[px,py,pz]=volume.pages;
 const page=(x:number,y:number,z:number)=>x<0||y<0||z<0||x>=px||y>=py||z>=pz?undefined:volume.at.get(x+px*(y+py*z));
 // The pages wet in every lattice cell they cover.
 const full=new Set<number>();
 for(const [i,p] of volume.at){
  const x=i%px,y=Math.floor(i/px)%py,z=Math.floor(i/(px*py));let wet=true;
  for(let c=0,k=0;c<E&&wet;c++)for(let b=0;b<E&&wet;b++)for(let a=0;a<E;a++,k++){if(E*x+a<nx&&E*y+b<ny&&E*z+c<nz&&!(p[k]!>0)){wet=false;break;}}
  if(wet)full.add(i);
 }
 const fullAt=(x:number,y:number,z:number)=>x<0||y<0||z<0||x>=px||y>=py||z>=pz||full.has(x+px*(y+py*z));
 const W=E+2*depth,dry=new Uint8Array(W*W*W),near=new Uint8Array(W*W*W),pass=new Uint8Array(W*W*W);
 let band=0;
 for(const i of volume.at.keys()){
  const x=i%px,y=Math.floor(i/px)%py,z=Math.floor(i/(px*py));
  let interior=true;for(let c=-1;c<=1&&interior;c++)for(let b=-1;b<=1&&interior;b++)for(let a=-1;a<=1;a++)if(!fullAt(x+a,y+b,z+c)){interior=false;break;}
  if(interior)continue;
  // The page and its apron of `depth` cells: dry where a lattice cell holds no volume.
  for(let c=0,k=0;c<W;c++)for(let b=0;b<W;b++)for(let a=0;a<W;a++,k++){
   const gx=E*x+a-depth,gy=E*y+b-depth,gz=E*z+c-depth;
   if(gx<0||gy<0||gz<0||gx>=nx||gy>=ny||gz>=nz){dry[k]=0;continue;}
   const q=page(Math.floor(gx/E),Math.floor(gy/E),Math.floor(gz/E));
   dry[k]=q&&q[(gx%E)+E*((gy%E)+E*(gz%E))]!>0?0:1;
  }
  // A dry cell within `depth` on every axis: three separable max filters.
  for(let axis=0;axis<3;axis++){
   const s=axis===0?1:axis===1?W:W*W,src=axis===0?dry:axis===1?near:pass,dst=axis===1?pass:near;
   for(let k=0;k<W*W*W;k++){
    const along=axis===0?k%W:axis===1?Math.floor(k/W)%W:Math.floor(k/(W*W));
    let any=0;for(let d=Math.max(-depth,-along);d<=Math.min(depth,W-1-along)&&!any;d++)any=src[k+d*s]!;
    dst[k]=any;
   }
  }
  for(let c=0;c<E;c++)for(let b=0;b<E;b++)for(let a=0;a<E;a++){
   const gx=E*x+a,gy=E*y+b,gz=E*z+c;if(gx>=nx||gy>=ny||gz>=nz)continue;
   const k=(a+depth)+W*((b+depth)+W*(c+depth));
   if(dry[k]===0&&near[k]===1&&!deep?.(gx>>4,gy>>4,gz>>4)){band++;tiles.add(tileKey(volume.cells,gx,gy,gz));}
  }
 }
 return {cells:band,tiles:tiles.size};
}

/** The census liquid's band: the initial liquid's, and each scheduled
 * drop's as it is added (a drop is liquid the census meets at its step, and
 * its band is seeded then): the cells of the drop whose centre lies inside
 * it and within `depth` cells of its surface, by its signed distance (a
 * sphere's, or an extrusion's: liquidExtrusionDistance). An inflow adds its
 * whole integrated volume as band cells, and those cells packed into whole
 * tiles. `initial` is the initial liquid by page, where the caller holds
 * it; `deep`, the initial liquid's deep regions (uniformParticleBand). */
export function uniformParticleCensusBand(scene:SceneDescription,depth:number,initial?:LatticePages,deep?:(x:number,y:number,z:number)=>boolean):UniformParticleBand{
 const volume=initial??fluidLatticePagedCellVolumes(scene,UNIFORM_PAGE_POOL_EDGE).volume,tiles=new Set<number>();
 let cells=uniformParticleBand(volume,depth,tiles,deep).cells,packed=0;
 const c=scene.container,lattice=volume.cells;
 // The fluid box's cells are the container's, from its low corner (sceneFluidBox): a box past the container is padding.
 const h=scene.voxelDomain.finestCellSize_m,size=[h,h,h] as const,reach=depth*h;
 const centre=(a:number,i:number)=>(a===1?0:-0.5*(a===0?c.width_m:c.depth_m))+(i+0.5)*size[a]!;
 const cell=(a:number,m:number)=>Math.floor((m-(a===1?0:-0.5*(a===0?c.width_m:c.depth_m)))/size[a]!);
 for(const {volume:drop} of scene.fluid.scheduledDrops??[]){
  const bounds=drop.shape==="sphere"?{min_m:{x:drop.center_m.x-drop.radius_m,y:drop.center_m.y-drop.radius_m,z:drop.center_m.z-drop.radius_m},max_m:{x:drop.center_m.x+drop.radius_m,y:drop.center_m.y+drop.radius_m,z:drop.center_m.z+drop.radius_m}}:liquidExtrusionBounds(drop);
  const edges=drop.shape==="sphere"?undefined:liquidExtrusionEdges(drop);
  const lo=[bounds.min_m.x,bounds.min_m.y,bounds.min_m.z].map((m,a)=>Math.max(0,cell(a,m)));
  const hi=[bounds.max_m.x,bounds.max_m.y,bounds.max_m.z].map((m,a)=>Math.min(lattice[a]!-1,cell(a,m)));
  for(let z=lo[2]!;z<=hi[2]!;z++)for(let y=lo[1]!;y<=hi[1]!;y++)for(let x=lo[0]!;x<=hi[0]!;x++){
   const p={x:centre(0,x),y:centre(1,y),z:centre(2,z)};
   const d=drop.shape==="sphere"?Math.hypot(p.x-drop.center_m.x,p.y-drop.center_m.y,p.z-drop.center_m.z)-drop.radius_m:liquidExtrusionDistance(drop,p,edges);
   if(d<=0&&d>-reach){cells++;tiles.add(tileKey(lattice,x,y,z));}
  }
 }
 const inflow=scene.fluid.inflow;
 if(inflow){
  const volume=Math.ceil(integratedInflowVolume(inflow,inflow.start_s,inflow.end_s)/(size[0]*size[1]*size[2]));cells+=volume;packed=Math.ceil(volume/64);
 }
 return {cells,tiles:tiles.size+packed};
}

/** The census liquid's ballistic reach (M4b step 3, the pool rule): the
 * fastest speed the scene starts with or can make, v, is the largest of the
 * bodies' initial speeds, the inflow's speed, a drop's (it starts at rest,
 * so its own fall: it reaches no higher than its release) and a dam-break
 * front's 2 sqrt(g H) over the tallest census liquid column with an
 * unsupported free face, H (Ritter 1892; the front runs up a wall to
 * v²/2g = 2H). A free face is a wet cell beside an open dry cell on a
 * horizontal axis: liquid held by walls, solids or terrain on every side
 * makes no front, and a hole in a wall only the column through it. The head is
 * v²/2g plus the census's closure margin (UNIFORM_PAGE_POOL_SEED_MARGIN
 * cells) above the highest liquid, top_m (the initial liquid's, a drop's or
 * the inflow nozzle's). */
export interface UniformCensusReach{readonly v_m_s:number;readonly head_m:number;readonly top_m:number}
export function uniformCensusReach(scene:SceneDescription,volume:LatticePages):UniformCensusReach{
 const h=scene.voxelDomain.finestCellSize_m,E=volume.edge,[nx,ny,nz]=volume.cells,[px,py]=volume.pages,g=Math.hypot(scene.fluid.gravity_m_s2.x,scene.fluid.gravity_m_s2.y,scene.fluid.gravity_m_s2.z);
 const column=new Uint32Array(nx*nz),free=new Uint8Array(nx*nz);let top=0;
 const liquid=(x:number,y:number,z:number)=>{const page=volume.at.get(Math.floor(x/E)+px*(Math.floor(y/E)+py*Math.floor(z/E)));return page?page[x%E+E*(y%E+E*(z%E))]!:0;};
 const cells=sceneLatticeDimensions(scene,Number.MAX_SAFE_INTEGER),[cx,cy,cz]=cells,walled=scene.container.walls!=="none";
 const solids=solidWorldForScene(scene),boundary=proceduralSolidField(scene),ground=terrainColumnHeights(scene,cx,cz),cellHeight=scene.container.height_m/cy;
 // An open dry cell: inside the container (past it are its walls, but in an open world), no solid, no procedural boundary, above the ground.
 const openDry=(x:number,y:number,z:number)=>{
  if(x<0||z<0||x>=cx||z>=cz)return !walled;
  if(liquid(x,y,z)>=0.5||sampleSolidWorld(solids,[x,y,z],true).solidFraction>0)return false;
  return (boundary?.open([x,y,z])??1)>0.5&&(y+0.5)*cellHeight>ground[x+cx*z]!;
 };
 for(const [i,page] of volume.at){
  const ox=E*(i%px),oy=E*(Math.floor(i/px)%py),oz=E*Math.floor(i/(px*py));
  for(let z=0,j=0;z<E;z++)for(let y=0;y<E;y++)for(let x=0;x<E;x++,j++)if(page[j]!>0&&ox+x<nx&&oy+y<ny&&oz+z<nz){
   const X=ox+x,Y=oy+y,Z=oz+z,c=X+nx*Z;column[c]!++;top=Math.max(top,(Y+1)*h);
   if(!free[c]&&page[j]!>=0.5&&(openDry(X-1,Y,Z)||openDry(X+1,Y,Z)||openDry(X,Y,Z-1)||openDry(X,Y,Z+1)))free[c]=1;
  }
 }
 let tallest=0;for(let c=0;c<column.length;c++)if(free[c])tallest=Math.max(tallest,column[c]!*h);
 for(const {volume:drop} of scene.fluid.scheduledDrops??[])top=Math.max(top,drop.shape==="sphere"?drop.center_m.y+drop.radius_m:liquidExtrusionBounds(drop).max_m.y);
 if(scene.fluid.inflow)top=Math.max(top,scene.fluid.inflow.center_m.y+scene.fluid.inflow.radius_m);
 const speed=(v:{x:number;y:number;z:number}|undefined)=>v?Math.hypot(v.x,v.y,v.z):0;
 const v=Math.max(2*Math.sqrt(g*tallest),speed(scene.fluid.inflow?.velocity_m_s),...scene.rigidBodies.map(b=>speed(b.linearVelocity_m_s)));
 // Without gravity nothing at rest climbs, and nothing moving comes back down (an unbounded head).
 return {v_m_s:v,head_m:(v>0?v*v/(2*g):0)+UNIFORM_PAGE_POOL_SEED_MARGIN*h,top_m:top};
}
/** The pool rule's blocks for a walled box (M4b step 3): the census pages
 * (every box page not in `air`), each footprint column's pages above its top
 * resident one up to the page of the census liquid's top plus its head
 * (uniformCensusReach), and the two sentinels. A closed top caps the reserve
 * at the box's top page; an open top with an unbounded head (no gravity)
 * has no rule, undefined. */
export function uniformPagePoolRule(scene:SceneDescription,volume:LatticePages,air:readonly (readonly [number,number,number])[],reach=uniformPagePoolRuleReach(scene,volume)):number|undefined{
 return reach&&uniformPagePoolRuleBlocks(volume.pages,air,reach.reachPage,reach.ceilingPage);
}
/** The pool rule's reserve columns' reach (uniformPagePoolRule): the page of
 * the census liquid's top plus its head, and the page that caps it (the box's
 * top under a closed top), and the census speed v; undefined for an open top
 * with an unbounded head. */
export function uniformPagePoolRuleReach(scene:SceneDescription,volume:LatticePages):{readonly reachPage:number;readonly ceilingPage:number;readonly v_m_s:number}|undefined{
 const reach=uniformCensusReach(scene,volume),pageHeight=volume.edge*scene.voxelDomain.finestCellSize_m,open=scene.container.top==="open";
 const reachPage=Math.floor((reach.top_m+reach.head_m)/pageHeight);
 if(open&&!Number.isFinite(reachPage))return undefined;
 return {reachPage,ceilingPage:open?Number.MAX_SAFE_INTEGER:volume.pages[1]-1,v_m_s:reach.v_m_s};
}
/** A body's bound in lattice cells: its centre and bounding radius (boundingRadius). */
export interface UniformCensusBodyBall{readonly centre:readonly [number,number,number];readonly radius:number}
/** The census start's solid contact tiles (solidActive in uniform-mixed-dynamic.ts): of the
 * solid-coupled tiles (`coupled`, uniformMixedSolidTiles, by box index over `tiles`), those whose
 * geometry the 4h lattice does not hold exactly (not umSolidGeometryExact, uniform-solid-refinement.ts).
 * The census holds such a tile at h where liquid reaches it; a wall the 4h lattice carries exactly
 * (a whole tile open or closed, each low face a uniform patch) runs at 4h. Read from the cells:
 * the tile's 64 cells are one kind, and so is each 4x4 cell layer either side of its low face and
 * of its low neighbour's low face (whole cells, uniform patches). That is no weaker than the
 * census's test (equal patch codes), so the seed holds every tile the census does. `open` is whether
 * a lattice cell is open; a cell off the box is closed. */
export function uniformCensusInexactSolidTiles(tiles:readonly [number,number,number],coupled:Uint8Array,open:(x:number,y:number,z:number)=>boolean):Uint8Array{
 const [tx,ty,tz]=tiles,count=tx*ty*tz;
 if(coupled.length!==count)throw new Error("Census seed: the solid coupling does not match the box's tiles");
 const out=new Uint8Array(count),c:[number,number,number]=[0,0,0];
 // The 16 cells of layer l (a cell index on axis a) over the tile's other two axes.
 const layer=(o:readonly number[],a:number,l:number)=>{
  const b=(a+1)%3,d=(a+2)%3;let first:boolean|undefined;
  for(let j=0;j<4;j++)for(let i=0;i<4;i++){c[a]=l;c[b]=o[b]!+i;c[d]=o[d]!+j;const v=open(c[0],c[1],c[2]);if(first===undefined)first=v;else if(v!==first)return false;}
  return true;
 };
 for(let k=0;k<count;k++){
  if(!coupled[k])continue;
  const o=[4*(k%tx),4*(Math.floor(k/tx)%ty),4*Math.floor(k/(tx*ty))];
  let exact=true;const v0=open(o[0]!,o[1]!,o[2]!);
  for(let z=0;z<4&&exact;z++)for(let y=0;y<4&&exact;y++)for(let x=0;x<4;x++)if(open(o[0]!+x,o[1]!+y,o[2]!+z)!==v0){exact=false;break;}
  for(let a=0;a<3&&exact;a++)exact=layer(o,a,o[a]!-1)&&layer(o,a,o[a]!-4)&&layer(o,a,o[a]!-5);
  if(!exact)out[k]=1;
 }
 return out;
}
/** The census start's crossing tiles (classify's crossing flag, umSurfaceDisplacement in
 * uniform-mixed-dynamic.ts): a tile one of whose open h edges, between two of its vertices
 * (local v in [0,4]³, v[a] < 4 along axis a), changes the sign of vertex phi. An edge is open
 * when one of its four incident cells inside the tile is open (umCensusEdgeOpen): a solid is not
 * air, so a wall's wet face holds no crossing. `phi` holds vertex phi by page (fluidLatticePagedVertexPhi);
 * a page it leaves undefined holds none. `open` is whether a lattice cell is open (no solid). */
export function uniformCensusCrossingTiles(phi:LatticePages,tiles:readonly [number,number,number],open:(x:number,y:number,z:number)=>boolean):Uint8Array{
 const [tx,ty,tz]=tiles,E=phi.edge,n=E+1,[px,py,pz]=phi.pages,crossing=new Uint8Array(tx*ty*tz);
 if(E%4!==0)throw new Error(`Census crossing: a page edge of ${E} cells holds no whole tile`);
 const cellOpen=new Uint8Array(64);
 for(const [i,page] of phi.at){
  const x=i%px,y=Math.floor(i/px)%py,z=Math.floor(i/(px*py));
  for(let c=0;c<E/4;c++)for(let b=0;b<E/4;b++)for(let a=0;a<E/4;a++){
   const T=[(E*x>>2)+a,(E*y>>2)+b,(E*z>>2)+c] as const;if(T[0]>=tx||T[1]>=ty||T[2]>=tz)continue;
   const o=[4*a,4*b,4*c] as const,g=[4*T[0],4*T[1],4*T[2]] as const;let all=true;
   for(let k=0;k<64;k++){cellOpen[k]=open(g[0]+(k&3),g[1]+((k>>2)&3),g[2]+(k>>4))?1:0;if(!cellOpen[k])all=false;}
   const at=(i:number,j:number,k:number)=>page[o[0]+i+n*(o[1]+j+n*(o[2]+k))]!;
   const cell=(i:number,j:number,k:number)=>i>=0&&j>=0&&k>=0&&i<4&&j<4&&k<4&&cellOpen[i+4*(j+4*k)]===1;
   let found=false;
   for(let k=0;k<=4&&!found;k++)for(let j=0;j<=4&&!found;j++)for(let i=0;i<=4&&!found;i++){
    const v=[i,j,k],value=at(i,j,k);
    for(let e=0;e<3&&!found;e++){
     if(v[e]!>=4)continue;
     const w=v.slice();w[e]!++;if((value<0)===(at(w[0]!,w[1]!,w[2]!)<0))continue;
     if(!all){
      const p=(e+1)%3,q=(e+2)%3;let edgeOpen=false;
      for(let m=0;m<4&&!edgeOpen;m++){const u=v.slice();u[p]!-=m&1;u[q]!-=m>>1;edgeOpen=cell(u[0]!,u[1]!,u[2]!);}
      if(!edgeOpen)continue;
     }
     found=true;
    }
   }
   if(found)crossing[T[0]+tx*(T[1]+ty*T[2])]=1;
  }
 }
 return crossing;
}
/** The pages a census start reads vertex phi on (uniformCensusCrossingTiles): a page holding
 * liquid or beside (3³) one that does, but not one whose 3³ neighbourhood is wet in every cell,
 * which holds no crossing. */
export function uniformCensusSurfacePages(volume:LatticePages):(x:number,y:number,z:number)=>boolean{
 const E=volume.edge,[nx,ny,nz]=volume.cells,[px,py,pz]=volume.pages,full=new Set<number>();
 for(const [i,p] of volume.at){
  const x=i%px,y=Math.floor(i/px)%py,z=Math.floor(i/(px*py));let wet=true;
  for(let c=0,k=0;c<E&&wet;c++)for(let b=0;b<E&&wet;b++)for(let a=0;a<E;a++,k++){if(E*x+a<nx&&E*y+b<ny&&E*z+c<nz&&!(p[k]!>0)){wet=false;break;}}
  if(wet)full.add(i);
 }
 const inside=(x:number,y:number,z:number)=>x>=0&&y>=0&&z>=0&&x<px&&y<py&&z<pz;
 return (x,y,z)=>{
  let near=false,interior=true;
  for(let c=-1;c<=1;c++)for(let b=-1;b<=1;b++)for(let a=-1;a<=1;a++){
   const X=x+a,Y=y+b,Z=z+c;
   if(!inside(X,Y,Z))continue;
   const i=X+px*(Y+py*Z);if(volume.at.has(i))near=true;if(!full.has(i))interior=false;
  }
  return near&&!interior;
 };
}
/** The census start's h tiles (fluid.tiers.start "census", docs/plans/bulk16-plan-2026-10-28.md, V2a):
 * frame 1's census rule evaluated on the initial liquid by page, with no dense h field. A box tile
 * (index x + tx (y + ty z) over `tiles`) is at h when one of three holds.
 * - It is a crossing tile of the initial vertex phi (`crossing`: uniformCensusCrossingTiles, the
 *   census's own interface tiles, where a solid is not air). The census then keeps the ones its
 *   criteria require (shape against its tolerance, thin, impact, ...); the seed keeps them all.
 * - It is a solid contact tile (`solid`: uniformCensusInexactSolidTiles, a coupled tile whose
 *   geometry 4h does not hold exactly) beside (3³) a wet tile, or one of its 26 neighbours: the
 *   census's solidActive where liquid reaches it and solidPromote's stencil around it.
 * - It lies within a body's ball beside a wet tile: the census's body tiles, bounded by the body's
 *   sphere. The ball is grown by a cell, so a wall a cell past the bound still counts.
 * Then the 26 neighbours of every such tile: frame 1's surface moves under a tile from rest, and
 * the closure keeps the tiles it can reach at h. Every other tile starts at 4h: deep liquid, which
 * the restriction of the t=0 upload holds exactly, and air. */
export function uniformCensusSeedTiles(volume:LatticePages,crossing:Uint8Array,tiles:readonly [number,number,number],solid?:Uint8Array,bodies:readonly UniformCensusBodyBall[]=[]):Uint8Array{
 const [tx,ty,tz]=tiles,count=tx*ty*tz,E=volume.edge,[px,py]=volume.pages;
 if(volume.cells.some((n,a)=>Math.ceil(n/4)!==tiles[a]))throw new Error(`Census seed: the lattice ${volume.cells.join("x")} is not the box's tiles ${tiles.join("x")} at 4 cells a tile`);
 if(solid&&solid.length!==count)throw new Error("Census seed: the solid contact does not match the box's tiles");
 if(crossing.length!==count)throw new Error("Census seed: the crossing tiles do not match the box's tiles");
 const wet=new Uint8Array(count);
 for(const [i,page] of volume.at){
  const ox=E*(i%px),oy=E*(Math.floor(i/px)%py),oz=E*Math.floor(i/(px*py));
  for(let z=0,j=0;z<E;z++)for(let y=0;y<E;y++)for(let x=0;x<E;x++,j++){
   const X=ox+x,Y=oy+y,Z=oz+z;
   if(page[j]!>0&&X<volume.cells[0]&&Y<volume.cells[1]&&Z<volume.cells[2])wet[(X>>2)+tx*((Y>>2)+ty*(Z>>2))]=1;
  }
 }
 const dilate=(src:Uint8Array)=>{
  const out=new Uint8Array(count);
  for(let z=0,k=0;z<tz;z++)for(let y=0;y<ty;y++)for(let x=0;x<tx;x++,k++){
   if(!src[k])continue;
   for(let c=Math.max(0,z-1);c<=Math.min(tz-1,z+1);c++)for(let b=Math.max(0,y-1);b<=Math.min(ty-1,y+1);b++)for(let a=Math.max(0,x-1);a<=Math.min(tx-1,x+1);a++)out[a+tx*(b+ty*c)]=1;
  }
  return out;
 };
 const nearWet=dilate(wet),seed=new Uint8Array(count);
 for(let k=0;k<count;k++)if(crossing[k])seed[k]=1;
 if(solid){const active=new Uint8Array(count);for(let k=0;k<count;k++)if(solid[k]&&nearWet[k])active[k]=1;dilate(active).forEach((v,k)=>{if(v)seed[k]=1;});}
 for(const {centre,radius} of bodies){
  // A tile meets the ball when its nearest cell point lies within the radius (plus a cell).
  const r=radius+1,lo=centre.map(c=>Math.max(0,Math.floor((c-r)/4))),hi=centre.map((c,a)=>Math.min(tiles[a]!-1,Math.floor((c+r)/4)));
  for(let z=lo[2]!;z<=hi[2]!;z++)for(let y=lo[1]!;y<=hi[1]!;y++)for(let x=lo[0]!;x<=hi[0]!;x++){
   const k=x+tx*(y+ty*z);if(!nearWet[k])continue;
   const d=[x,y,z].map((t,a)=>Math.max(4*t-centre[a]!,0,centre[a]!-4*(t+1)));
   if(Math.hypot(d[0]!,d[1]!,d[2]!)<=r)seed[k]=1;
  }
 }
 return dilate(seed);
}
