import assert from "node:assert/strict";
import test from "node:test";
import { getSceneDefinition } from "../lib/core/scenes";
import { sceneDocumentAtLattice } from "../lib/core/scene-definition";
import { createReservoirSeepScene, RESERVOIR_SEEP_ID } from "../lib/core/reservoir-seep-scene";
import { RESIDENCY_STAGES } from "../lib/methods/uniform/uniform-local-residency";
import { BULK16_PAGE, BULK16_PAGE_WORDS, BULK16_SCAN } from "../lib/methods/uniform/uniform-bulk-pages-reference";
import { BULK16_STATUS } from "../lib/methods/uniform/uniform-bulk-pages";
import { UNIFORM_BULK_REACH_CELLS } from "../lib/methods/uniform/uniform-bulk-tier-reference";
import type { UniformMixedFrame } from "../lib/methods/uniform/uniform-mixed-frame";
import { createUniformSolver, withUniformDevice } from "./helpers/uniform-geometric";

/* M2 (docs/plans/bulk16-plan-2026-10-28.md): a block that a promote claims
 * back from BULK16 holds, in every block-keyed field, the image deep resident
 * water of the same content holds. The scene is reservoir-seep's closed tank
 * cut to a one-page column (seven resident pages, five whole-16h pages
 * between the floor and the surface page), at rest with no solid. The test
 * cycle (UniformBulkPageTransfer.cycle) demotes the five and promotes them at
 * the next commit, into other blocks; each promoted block is diffed against
 * the block that held the same page in the last resident frame, at three
 * points: straight after the commit (the claim's init and the promote),
 * after the remap onto the census's layout (before any stage reads a
 * field), and at the frame's end.
 *
 * Compared: every 4h base of the frame's detail fields; the tile topology
 * (widths, stencil, blend); NB-FLIP's activity target, far-air gate and
 * crossing masks; the census's certificate records; the three low-face row
 * sets; the root's presented pressure; the surface model epochs.
 *
 * Excluded, as they legitimately differ after a claim:
 *  - heat/HOT: NB-FLIP's activity heat, the region heat tiers and state, the
 *    surface work evidence a claim promotes (its neighbours' HOT and epochs);
 *  - holds: the census's hold and budget flight words;
 *  - the bulk tier record's entry hold;
 *  - a tile word's owner index (its low 30 bits): the compact owner list's
 *    position, which follows block order, so a block that moves moves it.
 * Three compare to a bound: the presented pressure to the root solve's step
 * (rootStep: the images lie one frame's solve apart); faces to 1e-4 m/s (a demote restricts them by
 * area to 16h means, dropping sub-16h components, which at rest are noise);
 * the certificate's radii cover deep water's (the promote seeds the travel
 * bound bulk membership guarantees; the next census rewrites them). */

const COLUMN={widthScale:0.0625,depthScale:0.5,heightScale:1};
const FACE_TOLERANCE=1e-4;

// The frame's private fields are read as the test's instrument, not as an interface.
type Frame=UniformMixedFrame;
interface Image{readonly words:Map<string,Uint32Array>;readonly meta:Map<string,{texel:number;row:number}>}

/** Copies of the compared resources, encoded where a frame stands (its
 * commit's encoder, or one of the test's), mapped once the work is done. */
function encodeImage(device:GPUDevice,encoder:GPUCommandEncoder,buffers:Record<string,GPUBuffer>,textures:Record<string,GPUTexture>):()=>Promise<Image>{
 const staged:[string,GPUBuffer,{texel:number;row:number}?][]=[];
 for(const [name,buffer] of Object.entries(buffers)){
  const size=Math.floor(buffer.size/4)*4,r=device.createBuffer({size,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
  encoder.copyBufferToBuffer(buffer,0,r,0,size);staged.push([name,r]);
 }
 for(const [name,t] of Object.entries(textures)){
  const texel=t.format.startsWith("rgba32")?4:1,row=Math.ceil(t.width*texel*4/256)*256/4;
  const r=device.createBuffer({size:4*row*t.height*t.depthOrArrayLayers,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
  encoder.copyTextureToBuffer({texture:t},{buffer:r,bytesPerRow:4*row,rowsPerImage:t.height},[t.width,t.height,t.depthOrArrayLayers]);staged.push([name,r,{texel,row}]);
 }
 return async()=>{
  const words=new Map<string,Uint32Array>(),meta=new Map<string,{texel:number;row:number}>();
  for(const [name,r,m] of staged){await r.mapAsync(GPUMapMode.READ);words.set(name,new Uint32Array(r.getMappedRange().slice(0)));r.unmap();r.destroy();if(m)meta.set(name,m);}
  return {words,meta};
 };
}

test("BULK16 promote: a promoted block holds deep resident water's image in every block-keyed field",{skip:!process.env.WEBGPU_NODE_MODULE},async t=>{
 await withUniformDevice("BULK16 promote image",async device=>{
  const base=getSceneDefinition(RESERVOIR_SEEP_ID);
  const {scene}=sceneDocumentAtLattice({...base,buildAt:(l:{cellSize_m:number})=>createReservoirSeepScene(l.cellSize_m,COLUMN)},{cellSize_m:0.005});
  // The closed tank: no internal wall, the reservoir's water across the box.
  const v=scene.fluid.initialLiquidVolumes![0]! as any;scene.solidVoxels=[];
  scene.fluid.initialLiquidVolumes=[{...v,min_m:{...v.min_m,x:-scene.container.width_m/2},max_m:{...v.max_m,x:scene.container.width_m/2}}];
  scene.fluid.pagePool={...scene.fluid.pagePool,blocks:16};scene.fluid.dynamicViscosity_Pa_s=0;scene.fluid.surfaceTension_N_m=0;
  const solver=await createUniformSolver(device,scene,{...base.methodProfile!.overrides,bulkTier:"16h",volumeControlSeconds:0});
  const frame=(solver as unknown as {mixedFrame:Frame}).mixedFrame as any;
  try{
   assert.ok(frame.bulkTransfer,"bulkTier 16h builds the BULK16 transfer");
   frame.bulkTransfer.cycle=true;
   const fields=frame.fields,detail=fields.detail,address=frame.ownership.capacity.address;
   const k=address.blockBits as number,blocks=address.blocks as number,gridZ=blocks>>(2*k);
   const bases=(["volume","volumeScratch","velocity","velocityScratch","departure","phi","phiScratch","phase","centerPhi","target","correction"] as const)
    .map(name=>({name,texture:detail.baseOf(fields[name]) as GPUTexture|undefined})).filter(b=>b.texture);
   const nb=frame.narrowBandFlip as any,census=frame.pageInitialise.owners[3] as any,surface=frame.surfaceWorkControl as any;
   const tiles=frame.ownership.capacity.tiles as number,topology=frame.ownership.presentation.buffer as GPUBuffer;
   const buffers:Record<string,GPUBuffer>={topology,nb:nb.bins,census:census.work,negative:fields.negative,negativeScratch:fields.negativeScratch,negativeDeparture:fields.negativeDeparture,
    pressure:frame.presentation.pressure.buffer,surface:surface.surface};
   const pressureAt=(frame.presentation.pressure.offset??0)/4;
   const pool=frame.pagePool,table=(image:Image)=>image.words.get("pool")!.subarray(pool.layout.state.table,pool.layout.state.table+pool.layout.directory);
   buffers.pool=pool.state;
   const textures=Object.fromEntries(bases.map(b=>[`base:${b.name}`,b.texture!]));
   // Two points of each frame: straight after its commit (the claims' init
   // and the promote, BULK16's encodeAfter) and its end.
   let claim:(()=>Promise<Image>)|undefined,remapped:(()=>Promise<Image>)|undefined;
   const transfer=frame.bulkTransfer,after=transfer.encodeAfter.bind(transfer);
   transfer.encodeAfter=(encoder:GPUCommandEncoder)=>{after(encoder);claim=encodeImage(device,encoder,buffers,textures);};
   const remap=frame.remap,apply=remap.applyGpu.bind(remap);
   remap.applyGpu=(encoder:GPUCommandEncoder,...rest:unknown[])=>{apply(encoder,...rest);remapped=encodeImage(device,encoder,buffers,textures);};
   const capture=async()=>{
    await solver.awaitFrameCompletion();
    assert.ok(claim&&remapped,"every frame commits and remaps onto its layout");const atClaim=await claim(),atRemap=await remapped();claim=remapped=undefined;
    const encoder=device.createCommandEncoder(),end=encodeImage(device,encoder,buffers,textures);device.queue.submit([encoder.finish()]);
    return {claim:atClaim,remap:atRemap,end:await end()};
   };
   // Steps to the first demote (whole-16h membership takes its 30-frame entry hold), then the promote.
   const held=(w:number)=>w!==0&&w!==0xffffffff&&(w&0x80000000)===0;
   let resident:{claim:Image;remap:Image;end:Image}|undefined,promoted:typeof resident,demoted=false,step=0;
   for(step=1;step<=60&&!promoted;step++){
    assert.ok(solver.advanceTo(step*scene.numerics.fixedDt_s,[]));
    const image=await capture();assert.equal(solver.info.simulationPipelineError,undefined);
    const bulk=[...table(image.end)].filter(w=>w!==0xffffffff&&(w&0x80000000)!==0).length;
    if(bulk>0){assert.ok(resident,"a demote follows a resident frame");demoted=true;continue;}
    if(demoted)promoted=image;else resident=image;
   }
   assert.ok(resident&&promoted,`the cycle demoted and promoted whole-16h pages by step ${step}`);
   // Each promoted page: the block that held it in the last resident frame, and the one that holds it now.
   const moved=new Map<number,number>(),now=table(promoted.end);
   table(resident.end).forEach((w,p)=>{if(w!==now[p]&&held(w)){assert.ok(held(now[p]!),`page ${p} is resident again`);moved.set(w,now[p]!);}});
   const pairs=[...moved];
   // The first demote takes the pages whose regions entered first: their
   // holds start on their own evidence (region heat, not the tier's start-up
   // HOT), so the column's pages need not enter together.
   assert.ok(pairs.length>=2,`the column promotes its whole-16h pages (${pairs.length})`);
   t.diagnostic(`promoted blocks ${JSON.stringify(pairs)}`);
   const f32=(w:number)=>new Float32Array(new Uint32Array([w]).buffer)[0]!;
   const failures:string[]=[];
   type Point="claim"|"remap"|"end";
   const compare=(label:string,source:string,index:(b:number)=>number[],check:(a:number,b:number)=>boolean,points:readonly Point[]=["claim","remap","end"],keep?:(point:Point,old:number,i:number)=>boolean)=>{
    for(const point of points){
     const before=resident![point].words.get(source)!,after=promoted![point].words.get(source)!;let bad=0,first="";
     for(const [old,now] of pairs){const ia=index(old),ib=index(now);
      for(let i=0;i<ia.length;i++){if(keep&&!keep(point,old,i))continue;const a=before[ia[i]!]!,b=after[ib[i]!]!;if(!check(a,b)){if(!bad)first=`block ${old}->${now} word ${i}: ${a.toString(16)} vs ${b.toString(16)} (${f32(a)} vs ${f32(b)})`;bad++;}}}
     if(bad)failures.push(`${label} (${point}): ${bad} words differ; first ${first}`);
    }
   };
   const exact=(a:number,b:number)=>a===b;
   const face=(a:number,b:number)=>Math.abs(f32(a)-f32(b))<=FACE_TOLERANCE;
   // Deep rows are the 4h root's, which accepts a residual r up to 5 1/s h-equivalent (uniform-pressure-arena.ts, UNIFORM_PRESSURE_ARENA_RETURNS_RESIDUAL's note):
   // from u = u* - (dt/rho) dp/h, a frame's solve may move them by rho r h^2/dt (7.5 Pa at 5 mm), and the images lie one frame's solve apart.
   const rootStep=scene.fluid.density_kg_m3*5*0.005*0.005/scene.numerics.fixedDt_s!;
   const pressure=(a:number,b:number)=>Math.abs(f32(a)-f32(b))<=rootStep;
   const range=(from:number,n:number)=>Array.from({length:n},(_,i)=>from+i);
   // The 4h bases: a block's texels at its block coordinate (UniformDetailFields).
   // centerPhi's texel at an h tile holds the h owner's value (its lane 0);
   // a claimed block's tiles are 4h until the remap, and its first writer
   // after that is the geometry pass (the block is HOT), before any reader
   // (authority, momentum, forces): compared at the frame's end.
   const topologyAt=(point:Point)=>resident![point].words.get("topology")!;
   const fineTile=(point:Point,old:number,i:number)=>(topologyAt(point)[512*old+i]!&0x80000000)!==0;
   for(const b of bases){
    const tex=b.texture!,meta=resident.end.meta.get(`base:${b.name}`)!;
    const ext=[tex.width>>k,tex.height>>k,tex.depthOrArrayLayers/gridZ];
    const index=(block:number)=>{const c=[block&((1<<k)-1),(block>>k)&((1<<k)-1),block>>(2*k)],out:number[]=[];
     for(let z=0;z<ext[2]!;z++)for(let y=0;y<ext[1]!;y++)for(let x=0;x<ext[0]!;x++)for(let ch=0;ch<meta.texel;ch++)
      out.push(((c[2]!*ext[2]!+z)*tex.height+c[1]!*ext[1]!+y)*meta.row+(c[0]!*ext[0]!+x)*meta.texel+ch);return out;};
    compare(`4h base ${b.name}`,`base:${b.name}`,index,meta.texel===4?face:exact,undefined,b.name==="centerPhi"?(point,old,i)=>point==="end"||!fineTile("end",old,i):undefined);
   }
   // Tile words: widths (the owner index excluded), stencil pair and blend.
   // The remap publishes every claimed block's tile words after the commit,
   // before any stage: the claim's (4h, no owner) are its previous layout,
   // the truth for a block that holds 4h data alone.
   const rebuilt=["remap","end"] as const;
   compare("tile widths","topology",b=>range(512*b,512),(a,b)=>(a&0xc0000000)===(b&0xc0000000),rebuilt);
   compare("tile stencil","topology",b=>range(2*tiles+1024*b,1024),exact,rebuilt);
   compare("tile blend","topology",b=>range(4*tiles+512*b,512),exact,rebuilt);
   // NB-FLIP: the activity target, the far-air gate and the crossing masks (heat excluded).
   compare("NB-FLIP activity target","nb",b=>range(nb.activityWord+3*tiles+512*b,512),exact);
   compare("NB-FLIP far-air gate","nb",b=>range(nb.airGateWord+512*b,512),exact);
   if(nb.previousCrossings){compare("NB-FLIP crossings","nb",b=>range(nb.bandWord+4+2*tiles+512*b,512),exact);compare("NB-FLIP crossing pairs","nb",b=>range(nb.bandWord+4+1024*b,1024),exact);}
   // The census's certificate records: masks exact, radii covering deep water's.
   const records=(b:number)=>RESIDENCY_STAGES.flatMap((_,s)=>range(0,8).flatMap(o=>range(census.localOffset+4*(s*census.pages+8*b+o),4)));
   compare("certificate masks","census",b=>records(b).filter((_,i)=>i%4<2),exact);
   compare("certificate radii","census",b=>records(b).filter((_,i)=>i%4>=2),(a,b)=>[0,8,16].every(s=>((b>>s)&255)>=((a>>s)&255))&&(a===0)===(b===0));
   for(const rows of ["negative","negativeScratch","negativeDeparture"])compare(`low-face rows ${rows}`,rows,b=>range(3072*b,3072),face);
   compare("presented pressure","pressure",b=>range(pressureAt+512*b,512),pressure);
   compare("surface model epochs","surface",b=>range(surface.modelEpochWordOffset+8*b,8),exact);
   assert.deepEqual(failures,[],failures.join("\n"));
  }finally{solver.destroy();}
 });
});

/* M2's promote trigger (b) (UniformBulkTier's bulkPromote): at rest no
 * BULK16 page asks back, and a page whose face mean moves more than
 * UNIFORM_BULK_REACH_CELLS h a step promotes at the next commit. The same
 * column, the runtime transfer (no test cycle); the disturbance is written
 * into one live slot's face mean as the instrument. */
test("BULK16 promote: a page stays at rest and asks back when its face mean outruns the reach",{skip:!process.env.WEBGPU_NODE_MODULE},async()=>{
 await withUniformDevice("BULK16 promote trigger",async device=>{
  const base=getSceneDefinition(RESERVOIR_SEEP_ID);
  const {scene}=sceneDocumentAtLattice({...base,buildAt:(l:{cellSize_m:number})=>createReservoirSeepScene(l.cellSize_m,COLUMN)},{cellSize_m:0.005});
  const v=scene.fluid.initialLiquidVolumes![0]! as any;scene.solidVoxels=[];
  scene.fluid.initialLiquidVolumes=[{...v,min_m:{...v.min_m,x:-scene.container.width_m/2},max_m:{...v.max_m,x:scene.container.width_m/2}}];
  scene.fluid.pagePool={...scene.fluid.pagePool,blocks:16};scene.fluid.dynamicViscosity_Pa_s=0;scene.fluid.surfaceTension_N_m=0;
  const solver=await createUniformSolver(device,scene,{...base.methodProfile!.overrides,bulkTier:"16h",volumeControlSeconds:0});
  const frame=(solver as unknown as {mixedFrame:Frame}).mixedFrame as any;
  try{
   const store=frame.bulkPages,L=store.layout,offset=(store.binding.offset??0) as number;
   const read=async(at:number,n:number)=>{const r=device.createBuffer({size:4*n,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});const e=device.createCommandEncoder();e.copyBufferToBuffer(store.binding.buffer,offset+4*at,r,0,4*n);device.queue.submit([e.finish()]);await r.mapAsync(GPUMapMode.READ);const w=new Uint32Array(r.getMappedRange().slice(0));r.destroy();return w;};
   const live=async()=>(await read(BULK16_STATUS.live,1))[0]!;
   const pageOf=async(slot:number)=>[...(await read(L.records+BULK16_PAGE_WORDS*slot,3))].map(w=>w|0).join(",");
   const pages=async()=>{const n=await live(),slots=n?[...(await read(L.live,n))]:[];return Promise.all(slots.map(s=>pageOf(s)));};
   let step=0,demoted:string[]=[];
   // Every page's entry hold has run by step 60 (the pages need not enter together).
   for(step=1;step<=60;step++){assert.ok(solver.advanceTo(step*scene.numerics.fixedDt_s,[]));await solver.awaitFrameCompletion();}
   demoted=await pages();
   assert.ok(demoted.length>=3,`the column demotes its whole-16h pages by step ${step} (${demoted.length})`);
   // At rest: twenty frames, no page asks back.
   for(const end=step+20;step<end;step++){assert.ok(solver.advanceTo(step*scene.numerics.fixedDt_s,[]));await solver.awaitFrameCompletion();}
   assert.deepEqual((await pages()).sort(),[...demoted].sort(),"at rest every BULK16 page stays");
   // One slot's x face mean at five times the reach.
   const slot=(await read(L.live,1))[0]!,page=await pageOf(slot),speed=5*UNIFORM_BULK_REACH_CELLS*0.005/scene.numerics.fixedDt_s;
   device.queue.writeBuffer(store.binding.buffer,offset+4*(L.records+BULK16_PAGE_WORDS*slot+BULK16_PAGE.velocity),new Float32Array([speed]));
   for(const end=step+2;step<end;step++){assert.ok(solver.advanceTo(step*scene.numerics.fixedDt_s,[]));await solver.awaitFrameCompletion();}
   assert.equal(solver.info.simulationPipelineError,undefined);
   assert.ok(!(await pages()).includes(page),`page ${page} promoted after its face mean outran the reach`);
  }finally{solver.destroy();}
 });
});

test("BULK16 promote: one page asked back above a BULK16 page takes its vertices from the slot that holds them",{skip:!process.env.WEBGPU_NODE_MODULE},async()=>{
 await withUniformDevice("BULK16 promote above BULK16",async device=>{
  // A one-page column at 1 cm, water to 1.30 m: the three pages below the
  // surface page demote (frame 35). The top one is asked back alone, so the
  // page below it stays BULK16. The vertices of its low plane on the domain
  // walls (x or z = 0) are held, in the umWideVertexSlot rule, by the page
  // below: a promote that reads them from the asked page's own slot reads
  // texels no page held (positive phi), dries the block's wall tiles, and
  // the column's pressure collapses (-5.4 kPa, no convergence).
  const top=1.30,cell=0.01,base=getSceneDefinition(RESERVOIR_SEEP_ID);
  const {scene}=sceneDocumentAtLattice({...base,buildAt:(l:{cellSize_m:number})=>createReservoirSeepScene(l.cellSize_m,{widthScale:0.125,depthScale:1,heightScale:4})},{cellSize_m:cell});
  const v=scene.fluid.initialLiquidVolumes![0]! as any,W=scene.container.width_m,D=scene.container.depth_m;scene.solidVoxels=[];
  scene.fluid.initialLiquidVolumes=[{...v,min_m:{x:-W/2,y:0,z:-D/2},max_m:{x:W/2,y:top,z:D/2}}];
  scene.fluid.pagePool={...scene.fluid.pagePool,blocks:28};scene.fluid.dynamicViscosity_Pa_s=0;scene.fluid.surfaceTension_N_m=0;
  const solver=await createUniformSolver(device,scene,{...base.methodProfile!.overrides,bulkTier:"16h",volumeControlSeconds:0});
  const frame=(solver as unknown as {mixedFrame:Frame}).mixedFrame as any;
  try{
   const rd=async(buffer:GPUBuffer,at:number,n:number)=>{const r=device.createBuffer({size:4*n,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});const e=device.createCommandEncoder();e.copyBufferToBuffer(buffer,at,r,0,4*n);device.queue.submit([e.finish()]);await r.mapAsync(GPUMapMode.READ);const w=new Uint32Array(r.getMappedRange().slice(0));r.unmap();r.destroy();return w;};
   const store=frame.bulkPages,L=store.layout,offset=(store.binding.offset??0) as number;
   const read=(at:number,n:number)=>rd(store.binding.buffer,offset+4*at,n);
   const pages=async()=>{const n=(await read(BULK16_STATUS.live,1))[0]!,slots=n?[...(await read(L.live,n))]:[];
    return new Map(await Promise.all(slots.map(async s=>[s,[...(await read(L.records+BULK16_PAGE_WORDS*s,3))].map(w=>w|0)] as [number,number[]])));};
   let step=0;
   const advance=async()=>{step++;assert.ok(solver.advanceTo(step*scene.numerics.fixedDt_s,[]));await solver.awaitFrameCompletion();assert.equal(solver.info.simulationPipelineError,undefined);};
   for(;step<39;)await advance();
   const demoted=await pages();
   assert.equal(demoted.size,3,"the three deep pages demote");
   const [slot,asked]=[...demoted].reduce((a,b)=>b[1][1]!>a[1][1]!?b:a);
   device.queue.writeBuffer(store.binding.buffer,offset+4*(L.records+BULK16_PAGE_WORDS*slot+BULK16_PAGE.promote),new Uint32Array([1]));
   await advance();
   const pool=frame.pagePool,blocks=pool.layout.blocks as number;
   const records=()=>rd(pool.state,4*pool.layout.state.records,8*blocks);
   let rec=await records();
   const block=[...Array(blocks).keys()].find(b=>rec[8*b+3]===1&&[0,1,2].every(i=>(rec[8*b+i]|0)===asked[i]));
   assert.ok(block!==undefined,`page ${asked} is resident again`);
   const phi=frame.levels[0].phi as GPUBufferBinding,tiles=new Float32Array((await rd(phi.buffer,(phi.offset??0)+4*512*block!,512)).buffer);
   const dry=[...tiles.keys()].filter(t=>!(tiles[t]!<0));
   assert.deepEqual(dry,[],`every tile of the promoted block is liquid (${dry.slice(0,8).map(t=>`${t}:${tiles[t]!.toFixed(3)}`)})`);
   for(;step<46;)await advance();
   rec=await records();
   // The column stays hydrostatic: each resident deep block's mean pressure
   // against rho g (top - its centre), the pages that stayed BULK16 untouched.
   const rhoG=scene.fluid.density_kg_m3*-scene.fluid.gravity_m_s2.y,pr=frame.presentation.pressure;
   for(let b=0;b<blocks;b++){
    if(rec[8*b+3]!==1||(rec[8*b+1]|0)>asked[1]!)continue;
    const p=new Float32Array((await rd(pr.buffer,(pr.offset??0)+4*512*b,512)).buffer),mean=p.reduce((s,x)=>s+x,0)/512;
    const off=mean-rhoG*(top-((rec[8*b+1]|0)+0.5)*32*cell);
    assert.ok(Math.abs(off)<100,`block ${b} (page y ${rec[8*b+1]|0}) holds hydrostatic pressure (${off.toFixed(1)} Pa off)`);
   }
  }finally{solver.destroy();}
 });
});

test("BULK16 promote: a page a particle's closure reaches takes its block back before the particle arrives",{skip:!process.env.WEBGPU_NODE_MODULE},async()=>{
 await withUniformDevice("BULK16 particle closure promote",async device=>{
  // The one-page column at 1 cm, water to 1.30 m (pages y0..y2 demote) and
  // a 0.2 m cube falling from 4.4 m. Before it lands, its lowest sample is
  // made ballistic at the speed whose second step ends at 0.84 m, inside
  // BULK16 page y2: some 57 cells a step, past the scan's 4 tiles and its
  // one-commit lag, as an impact jet is (measured: 11 to 22 m/s jets under
  // this cube over the wide column reached page y2 before its scan asked it
  // back, the fatal "arrived in page ..., which holds no storage", about
  // one run in eight). Its first step ends in the air above the water, and
  // its record of that step (the census's particle closure for the next)
  // reaches y2, so that commit takes y2 back; y0 and y1, beyond the
  // closure, stay BULK16.
  const top=1.30,cell=0.01,drop=0.2,base=getSceneDefinition(RESERVOIR_SEEP_ID);
  const {scene}=sceneDocumentAtLattice({...base,buildAt:(l:{cellSize_m:number})=>createReservoirSeepScene(l.cellSize_m,{widthScale:0.125,depthScale:1,heightScale:4})},{cellSize_m:cell});
  const v=scene.fluid.initialLiquidVolumes![0]! as any,W=scene.container.width_m,D=scene.container.depth_m;scene.solidVoxels=[];
  scene.fluid.initialLiquidVolumes=[{...v,min_m:{x:-W/2,y:0,z:-D/2},max_m:{x:W/2,y:top,z:D/2}},
   {shape:"box",min_m:{x:-drop/2,y:4.4,z:-drop/2},max_m:{x:drop/2,y:4.4+drop,z:drop/2}}];
  scene.fluid.pagePool={...scene.fluid.pagePool,blocks:28};scene.fluid.dynamicViscosity_Pa_s=0;scene.fluid.surfaceTension_N_m=0;
  const solver=await createUniformSolver(device,scene,{...base.methodProfile!.overrides,bulkTier:"16h",volumeControlSeconds:0});
  const frame=(solver as unknown as {mixedFrame:Frame}).mixedFrame as any;
  try{
   const rd=async(buffer:GPUBuffer,at:number,n:number)=>{const r=device.createBuffer({size:4*n,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});const e=device.createCommandEncoder();e.copyBufferToBuffer(buffer,at,r,0,4*n);device.queue.submit([e.finish()]);await r.mapAsync(GPUMapMode.READ);const w=new Uint32Array(r.getMappedRange().slice(0));r.unmap();r.destroy();return w;};
   const store=frame.bulkPages,L=store.layout,offset=(store.binding.offset??0) as number;
   const read=(at:number,n:number)=>rd(store.binding.buffer,offset+4*at,n);
   const pages=async()=>{const n=(await read(BULK16_STATUS.live,1))[0]!,slots=n?[...(await read(L.live,n))]:[];
    return (await Promise.all(slots.map(async s=>[...(await read(L.records+BULK16_PAGE_WORDS*s,3))].map(w=>w|0).join(",")))).sort();};
   let step=0;
   const advance=async()=>{step++;assert.ok(solver.advanceTo(step*scene.numerics.fixedDt_s,[]));await solver.awaitFrameCompletion();assert.equal(solver.info.simulationPipelineError,undefined);};
   for(;step<39;)await advance();
   const deep=["0,0,0","0,1,0","0,2,0"];
   assert.deepEqual(await pages(),deep,"the three deep pages demote before the drop lands");
   // The bank's lowest live sample: its slabs fill the records below state
   // word 0, a live record's heat (position.w) is positive, and positions
   // sit 32 cells a page from the held frame page (state words 12..14).
   const nb=frame.narrowBandFlip,stride=nb.particleStride as number,st=await rd(nb.state,0,16),count=st[0]!;
   const bank=await rd(nb.activeParticles,0,stride*count),f=new Float32Array(bank.buffer);
   let pick=-1;for(let i=0;i<count;i++)if(f[stride*i+3]!>0&&(pick<0||f[stride*i+1]!<f[stride*pick+1]!))pick=i;
   assert.ok(pick>=0,"the drop holds a live sample");
   const y=f[stride*pick+1]!+32*(st[13]!|0),travel=(y-84)/2,dt=scene.numerics.fixedDt_s;
   assert.ok(y-travel>top/cell+2,`the first step ends above the water (${y.toFixed(1)} cells, ${travel.toFixed(1)} a step)`);
   const record=bank.slice(stride*pick,stride*pick+stride),fr=new Float32Array(record.buffer);
   // Ballistic: the mode bit of before.w.
   fr[4]=0;fr[5]=-travel*cell/dt;fr[6]=0;record[11]=record[11]!|0x80000000;
   device.queue.writeBuffer(nb.activeParticles,4*stride*pick,record);
   await advance();
   assert.deepEqual(await pages(),deep,"the first step's closure is the old record's");
   await advance();
   assert.deepEqual(await pages(),["0,0,0","0,1,0"],"page y2 promoted for the sample's closure; y0 and y1 stay BULK16");
   for(const end=step+3;step<end;)await advance();
  }finally{solver.destroy();}
 });
});

// A disturbed deep region: every page asked back has a physical cause,
// named by its promote word, and none is a cascade.
//  0 the closure of a live sample whose travel reaches it (the census's
//    particle stage: the sample's tile, its travel of the move rounded up
//    to tiles, a ballistic sample's speed bounding every direction, and the
//    stage's two-tile margin; within its region's fastest travel + 16 cells
//    of the sample, and the bank holds the move's end velocity, so 24);
//  1 the structural scan (its tiles +-4: surface, band, cut, AIR, region
//    heat from contact, Sato et al.'s velocity detail or live particles,
//    or its own face speed), which records the tests it failed
//    (BULK16_SCAN's bits in promoteError): a contact the kernel measured.
//    Its gap to the samples is not a bound: velocity-detail heat under the
//    impact cavity reaches the page below with no sample near (measured:
//    0,1,0 from HOT alone at gaps 33 and 40, frames 64 and 65);
//  2 the error test of its own face rows, whose measured error must exceed
//    the level it records (twice the member tolerance).
// A promote's claim heat is not a cause: chained through the connected
// bulk it reached every page. In an incompressible liquid an impact's
// pressure reaches the bottom at once, so a deep page may leave by its
// error test; only a page at rest must stay (the at-rest tests).
type Promote={page:string;y:number;step:number;why:number;gap:number;error?:number;level?:number;scan?:number};
async function followPromotes(device:GPUDevice,solver:any,scene:any,demotedCount:number,until:number):Promise<Promote[]>{
 const frame=solver.mixedFrame,store=frame.bulkPages,L=store.layout,offset=(store.binding.offset??0) as number;
 const rd=async(buffer:GPUBuffer,at:number,n:number)=>{const r=device.createBuffer({size:4*n,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});const e=device.createCommandEncoder();e.copyBufferToBuffer(buffer,at,r,0,4*n);device.queue.submit([e.finish()]);await r.mapAsync(GPUMapMode.READ);const w=new Uint32Array(r.getMappedRange().slice(0));r.unmap();r.destroy();return w;};
 const read=(at:number,n:number)=>rd(store.binding.buffer,offset+4*at,n);
 const live=async()=>{const n=(await read(BULK16_STATUS.live,1))[0]!;return n?[...(await read(L.live,n))]:[];};
 const pageOf=new Map<number,number[]>();
 const pages=async()=>{const slots=await live();for(const s of slots)if(!pageOf.has(s))pageOf.set(s,[...(await read(L.records+BULK16_PAGE_WORDS*s,3))].map(w=>w|0));return slots;};
 const dt=scene.numerics.fixedDt_s;let step=0;
 const advance=async()=>{step++;assert.ok(solver.advanceTo(step*dt,[]));await solver.awaitFrameCompletion();assert.equal(solver.info.simulationPipelineError,undefined);};
 for(;step<40;)await advance();
 const demoted=await pages();
 assert.equal(demoted.length,demotedCount,"the deep pages demote before the drop nears");
 // The live samples each step, in box cells (32 a page from the held frame
 // page, state words 12..14), each with its 16-cell region's fastest speed
 // for the move: the census keeps one radius a region and direction.
 const nb=frame.narrowBandFlip,stride=nb.particleStride as number,cell=scene.voxelDomain?.finestCellSize_m??0.01;
 const samples=async()=>{const st=await rd(nb.state,0,16),count=st[0]!,out:number[][]=[];if(!count)return out;const f=new Float32Array((await rd(nb.activeParticles,0,stride*count)).buffer);
  for(let i=0;i<count;i++){const o=stride*i;if(f[o+3]!>0)out.push([0,1,2].map(a=>f[o+a]!+32*(st[12+a]!|0)).concat(Math.hypot(f[o+4]!,f[o+5]!,f[o+6]!)*dt/cell));}
  const fastest=new Map<string,number>(),key=(q:number[])=>q.slice(0,3).map(v=>Math.floor(v/16)).join(",");
  for(const q of out)fastest.set(key(q),Math.max(fastest.get(key(q))??0,q[3]!));
  return out.map(q=>[q[0]!,q[1]!,q[2]!,fastest.get(key(q))!]);};
 // A page's gap to the nearest sample's travel: the largest axis gap
 // between the sample and the page's 32 cells, less the sample's travel.
 // A loop, not a spread: the band can hold more samples than a call takes arguments.
 const gap=(page:number[],at:number[][])=>{let g=Infinity;for(const q of at)g=Math.min(g,Math.max(...[0,1,2].map(a=>Math.max(0,32*page[a]!-q[a]!,q[a]!-32*page[a]!-32)))-q[3]!);return g;};
 const seen=new Map<number,number[][]>([[step,await samples()]]);
 const promoted=new Map<number,{step:number;why:number;error:number;level:number;scan:number}>();
 for(;step<until;){
  await advance();seen.set(step,await samples());const now=new Set(await pages());
  for(const s of demoted)if(!now.has(s)&&!promoted.has(s)){
   const at=L.records+BULK16_PAGE_WORDS*s,why=(await read(at+BULK16_PAGE.promote,1))[0]!;
   const cause=await read(at+BULK16_PAGE.promoteError,1),error=new Float32Array(cause.buffer)[0]!,level=new Float32Array((await read(at+BULK16_PAGE.promoteLevel,1)).buffer)[0]!;
   promoted.set(s,{step,why,error,level,scan:cause[0]!});
  }
 }
 return [...promoted].map(([s,p])=>{const page=pageOf.get(s)!;return {page:page.join(","),y:page[1]!,step:p.step,why:p.why,
  gap:Math.round(Math.min(...[p.step-2,p.step-1,p.step].map(k=>gap(page,seen.get(k)!)))),...(p.why===2?{error:p.error,level:p.level}:{}),...(p.why===1?{scan:p.scan}:{})};});
}
function assertPromoteCauses(record:Promote[]){
 assert.deepEqual(record.filter(p=>![0,1,2].includes(p.why)),[],"every promote carries its cause word");
 assert.deepEqual(record.filter(p=>p.why===0&&p.gap>24),[],"a closure promote lies within a live sample's travel (no cascade)");
 const known=Object.values(BULK16_SCAN).reduce((a,b)=>a|b,0);
 assert.deepEqual(record.filter(p=>p.why===1&&!(p.scan!>0&&(p.scan!&~known)===0)),[],"a scan promote records the contact it measured");
 assert.deepEqual(record.filter(p=>p.why===2&&!(p.level!>0&&p.error!>p.level!)),[],"an error-test promote's measured error exceeds its level");
}

test("BULK16 promote: a falling drop asks back the column's top deep page before it lands, each page for a physical cause",{skip:!process.env.WEBGPU_NODE_MODULE},async()=>{
 await withUniformDevice("BULK16 disturbed promote",async device=>{
  // A one-page column at 1 cm, water to 1.30 m, and a 0.2 m cube of water
  // 3.1 m above it: the three deep pages demote (frame 35) before the drop
  // nears (near frame 46) and lands (near 49). Its fall stays under the
  // residency's reach (a full-section slab from 3 m does not, with or
  // without BULK16), so the scene is well posed. The falling samples'
  // closure reaches the top deep page y2 before they land, and that commit
  // takes it back (its promote word 0: neither scan asked). Measured: y1
  // by its error test or the closure in some runs, y0 by its error test in
  // two runs of twelve, the one with its cause recorded just over its
  // level (0.004738 against 0.004709 m/s, 1.006x): a borderline trigger,
  // the impact's pressure.
  const base=getSceneDefinition(RESERVOIR_SEEP_ID),drop=0.2;
  const {scene}=sceneDocumentAtLattice({...base,buildAt:(l:{cellSize_m:number})=>createReservoirSeepScene(l.cellSize_m,{widthScale:0.125,depthScale:1,heightScale:4})},{cellSize_m:0.01});
  const v=scene.fluid.initialLiquidVolumes![0]! as any,W=scene.container.width_m,D=scene.container.depth_m;scene.solidVoxels=[];
  scene.fluid.initialLiquidVolumes=[{...v,min_m:{x:-W/2,y:0,z:-D/2},max_m:{x:W/2,y:1.30,z:D/2}},
   {shape:"box",min_m:{x:-drop/2,y:4.4,z:-drop/2},max_m:{x:drop/2,y:4.4+drop,z:drop/2}}];
  scene.fluid.pagePool={...scene.fluid.pagePool,blocks:28};scene.fluid.dynamicViscosity_Pa_s=0;scene.fluid.surfaceTension_N_m=0;
  const solver=await createUniformSolver(device,scene,{...base.methodProfile!.overrides,bulkTier:"16h",volumeControlSeconds:0});
  try{
   const record=await followPromotes(device,solver,scene,3,110);
   console.log("disturbed promote",JSON.stringify(record));
   assert.ok(record.some(p=>p.y===2&&p.why===0),"the drop's closure asks the top deep page back");
   assertPromoteCauses(record);
  }finally{solver.destroy();}
 });
});

test("BULK16 promote: a drop at one side of a wide deep region asks back only the pages within its reach",{skip:!process.env.WEBGPU_NODE_MODULE},async()=>{
 await withUniformDevice("BULK16 local promote",async device=>{
  // Four pages across at 1 cm, water to 1.30 m (twelve deep pages demote;
  // 64 blocks, as the splash climbs: 48 ran out at frame 64 in one run),
  // and a 0.2 m cube 3.1 m above the middle of page x=0. Measured: the
  // splash's fast samples reach page x=2 (2,2,0 and 2,1,0 by closure), the
  // flow under the cavity later asks back the page below the impact (scan,
  // from its velocity detail) and the spreading flow pages x=2 and 3 under
  // the surface (error test, from frame 73).
  const base=getSceneDefinition(RESERVOIR_SEEP_ID),drop=0.2;
  const {scene}=sceneDocumentAtLattice({...base,buildAt:(l:{cellSize_m:number})=>createReservoirSeepScene(l.cellSize_m,{widthScale:0.5,depthScale:1,heightScale:4})},{cellSize_m:0.01});
  const v=scene.fluid.initialLiquidVolumes![0]! as any,W=scene.container.width_m,D=scene.container.depth_m;scene.solidVoxels=[];
  const x=-W/2+0.16;
  scene.fluid.initialLiquidVolumes=[{...v,min_m:{x:-W/2,y:0,z:-D/2},max_m:{x:W/2,y:1.30,z:D/2}},
   {shape:"box",min_m:{x:x-drop/2,y:4.4,z:-drop/2},max_m:{x:x+drop/2,y:4.4+drop,z:drop/2}}];
  scene.fluid.pagePool={...scene.fluid.pagePool,blocks:64};scene.fluid.dynamicViscosity_Pa_s=0;scene.fluid.surfaceTension_N_m=0;
  const solver=await createUniformSolver(device,scene,{...base.methodProfile!.overrides,bulkTier:"16h",volumeControlSeconds:0});
  try{
   const record=await followPromotes(device,solver,scene,12,110);
   console.log("local promote",JSON.stringify(record));
   assert.ok(record.some(p=>p.page.startsWith("0,")),"the impact column asks a page back");
   assertPromoteCauses(record);
  }finally{solver.destroy();}
 });
});
