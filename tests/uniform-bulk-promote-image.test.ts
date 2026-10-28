import assert from "node:assert/strict";
import test from "node:test";
import { getSceneDefinition } from "../lib/core/scenes";
import { sceneDocumentAtLattice } from "../lib/core/scene-definition";
import { createReservoirSeepScene, RESERVOIR_SEEP_ID } from "../lib/core/reservoir-seep-scene";
import { RESIDENCY_STAGES } from "../lib/methods/uniform/uniform-local-residency";
import { BULK16_PAGE, BULK16_PAGE_WORDS } from "../lib/methods/uniform/uniform-bulk-pages-reference";
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
 * Two compare to a bound: faces to 1e-4 m/s (a demote restricts them by
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
   assert.ok(pairs.length>=3,`the column promotes its whole-16h pages (${pairs.length})`);
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
   const pressure=(a:number,b:number)=>Math.abs(f32(a)-f32(b))<=1e-5*Math.max(1,Math.abs(f32(a)));
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
   for(step=1;step<=60&&!demoted.length;step++){assert.ok(solver.advanceTo(step*scene.numerics.fixedDt_s,[]));await solver.awaitFrameCompletion();demoted=await pages();}
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
