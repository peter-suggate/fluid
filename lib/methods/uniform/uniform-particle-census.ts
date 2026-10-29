import { fluidLatticePagedCellVolumes, type LatticePages } from "../../core/initial-liquid-lattice";
import type { SceneDescription } from "../../core/model";
import { liquidExtrusionBounds, liquidExtrusionDistance, liquidExtrusionEdges } from "../../core/liquid-extrusion";
import { integratedInflowVolume } from "../../core/inflow-boundary";
import { UNIFORM_PAGE_POOL_EDGE, UNIFORM_PAGE_POOL_SEED_MARGIN } from "./uniform-page-pool";
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
 * past the lattice, holds no band cell and is not visited. */
export function uniformParticleBand(volume:LatticePages,depth:number,tiles=new Set<number>()):UniformParticleBand{
 if(!(Number.isInteger(depth)&&depth>=1&&depth<volume.edge))throw new RangeError(`A particle band of ${depth} cells needs a whole depth under the page edge ${volume.edge}`);
 const E=volume.edge,[nx,ny,nz]=volume.cells,[px,py,pz]=volume.pages;
 const page=(x:number,y:number,z:number)=>x<0||y<0||z<0||x>=px||y>=py||z>=pz?undefined:volume.at[x+px*(y+py*z)];
 // A page wet in every lattice cell it covers.
 const full=new Uint8Array(px*py*pz);
 for(let z=0,i=0;z<pz;z++)for(let y=0;y<py;y++)for(let x=0;x<px;x++,i++){
  const p=volume.at[i];if(!p)continue;let wet=true;
  for(let c=0,k=0;c<E&&wet;c++)for(let b=0;b<E&&wet;b++)for(let a=0;a<E;a++,k++){if(E*x+a<nx&&E*y+b<ny&&E*z+c<nz&&!(p[k]!>0)){wet=false;break;}}
  full[i]=wet?1:0;
 }
 const fullAt=(x:number,y:number,z:number)=>x<0||y<0||z<0||x>=px||y>=py||z>=pz||full[x+px*(y+py*z)]===1;
 const W=E+2*depth,dry=new Uint8Array(W*W*W),near=new Uint8Array(W*W*W),pass=new Uint8Array(W*W*W);
 let band=0;
 for(let z=0;z<pz;z++)for(let y=0;y<py;y++)for(let x=0;x<px;x++){
  if(!page(x,y,z))continue;
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
   if(dry[k]===0&&near[k]===1){band++;tiles.add(tileKey(volume.cells,gx,gy,gz));}
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
 * it. */
export function uniformParticleCensusBand(scene:SceneDescription,depth:number,initial?:LatticePages):UniformParticleBand{
 const volume=initial??fluidLatticePagedCellVolumes(scene,UNIFORM_PAGE_POOL_EDGE).volume,tiles=new Set<number>();
 let cells=uniformParticleBand(volume,depth,tiles).cells,packed=0;
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
 * front's 2 sqrt(g H) over the tallest liquid column of the census, H
 * (Ritter 1892; the front runs up a wall to v²/2g = 2H). The head is
 * v²/2g plus the census's closure margin (UNIFORM_PAGE_POOL_SEED_MARGIN
 * cells) above the highest liquid, top_m (the initial liquid's, a drop's or
 * the inflow nozzle's). */
export interface UniformCensusReach{readonly v_m_s:number;readonly head_m:number;readonly top_m:number}
export function uniformCensusReach(scene:SceneDescription,volume:LatticePages):UniformCensusReach{
 const h=scene.voxelDomain.finestCellSize_m,E=volume.edge,[nx,ny,nz]=volume.cells,[px,py]=volume.pages,g=Math.hypot(scene.fluid.gravity_m_s2.x,scene.fluid.gravity_m_s2.y,scene.fluid.gravity_m_s2.z);
 const column=new Uint32Array(nx*nz);let top=0;
 for(let i=0;i<volume.at.length;i++){
  const page=volume.at[i];if(!page)continue;
  const ox=E*(i%px),oy=E*(Math.floor(i/px)%py),oz=E*Math.floor(i/(px*py));
  for(let z=0,j=0;z<E;z++)for(let y=0;y<E;y++)for(let x=0;x<E;x++,j++)if(page[j]!>0&&ox+x<nx&&oy+y<ny&&oz+z<nz){column[ox+x+nx*(oz+z)]!++;top=Math.max(top,(oy+y+1)*h);}
 }
 let tallest=0;for(const n of column)tallest=Math.max(tallest,n*h);
 for(const {volume:drop} of scene.fluid.scheduledDrops??[])top=Math.max(top,drop.shape==="sphere"?drop.center_m.y+drop.radius_m:liquidExtrusionBounds(drop).max_m.y);
 if(scene.fluid.inflow)top=Math.max(top,scene.fluid.inflow.center_m.y+scene.fluid.inflow.radius_m);
 const speed=(v:{x:number;y:number;z:number}|undefined)=>v?Math.hypot(v.x,v.y,v.z):0;
 const v=Math.max(2*Math.sqrt(g*tallest),speed(scene.fluid.inflow?.velocity_m_s),...scene.rigidBodies.map(b=>speed(b.linearVelocity_m_s)));
 return {v_m_s:v,head_m:v*v/(2*g)+UNIFORM_PAGE_POOL_SEED_MARGIN*h,top_m:top};
}
