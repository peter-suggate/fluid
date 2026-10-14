/** Dawn proof of page initialise (lib/methods/uniform/uniform-page-initialise.ts)
 * on a real page pool and a real detail storage of a 96x64x64 lattice (12
 * pages; 14 blocks in a 2x2x4 grid): nine fields of every
 * class (a per-tile field, an atlas field, a vertex field with an adopted
 * base), two low-face buffers and two
 * topology-shaped buffers that are also the pool's mirrors.
 * Before every commit each texel and word is poisoned (a NaN pattern in the
 * textures, 0xdeadbeef in the buffers); after it everything is read back
 * whole: the texels, rows and tile words of the blocks the commit claimed
 * hold the AIR image and every other one still holds poison, the sentinel
 * blocks included. Commits: two host claims while the fields are base
 * blocks alone; then, with an h generation beside the bases, two commits
 * that claim nothing (the second retires pages), one that claims ten pages
 * (more than the eight claim slots), and encodeAll.
 * The h generation is admitted by handing the storage a stand-in for the
 * simulation ownership (its capacity alone); the topologies are plain
 * buffers of the topology word layout, not ownerships.
 * node --import tsx tools/probe-uniform-page-initialise.ts
 */
import assert from "node:assert/strict";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { managedGPUDevice } from "../lib/core/gpu-compilation-manager";
import { createProcessRetainedDawnGPU } from "../lib/harness/node-dawn-provider";
import { UniformDetailStorage } from "../lib/methods/uniform/uniform-detail-fields";
import { uniformLatticeAddress } from "../lib/methods/uniform/uniform-lattice-address";
import { UniformPagePool,uniformPagePoolBoxSeed,UNIFORM_PAGE_POOL_STATUS as STATUS } from "../lib/methods/uniform/uniform-page-pool";
import { UniformPageInitialise,UniformPageTileWords,UNIFORM_PAGE_INIT_KEYS as KEYS,UNIFORM_PAGE_INIT_TILE_WORD as TILE_WORD } from "../lib/methods/uniform/uniform-page-initialise";

const started=performance.now(),errors:string[]=[];
const dawn=await import(pathToFileURL(resolve("node_modules/webgpu/index.js")).href);Object.assign(globalThis,dawn.globals);
const adapter=await createProcessRetainedDawnGPU(dawn,["backend=metal"]).requestAdapter();assert.ok(adapter);
const device=managedGPUDevice(await adapter.requestDevice({requiredLimits:{maxStorageBuffersPerShaderStage:10,maxStorageTexturesPerShaderStage:8}}),{requireWorkerRealm:false});
device.addEventListener("uncapturederror",e=>{e.preventDefault();errors.push(e.error.message);console.error(e.error.message);});

const NAN=0x7fc0beef,DEAD=0xdeadbeef,EMPTY=6.5,NEAR=1.6;
const F=new Float32Array(1),U=new Uint32Array(F.buffer),bitsOf=(v:number)=>{F[0]=v;return U[0]!;};
const dims:[number,number,number]=[96,64,64];
const storage=new UniformDetailStorage(device,uniformLatticeAddress(dims)),a=storage.layout.address,G=a.blockGrid,bits=a.blockBits;
const field=(label:string,kind:"cell"|"face"|"vertex"|"atlas",perTile=false)=>storage.createField(label,kind,kind==="face"||kind==="atlas"?"rgba32float":"r32float",perTile,kind==="vertex"?EMPTY:0);
const velocityA=field("velocity A","face"),velocityB=field("velocity B","face"),taps=field("unit taps","atlas"),volumeA=field("volume A","cell");
const volumeB=field("volume B","cell"),gamma=field("gamma A","cell"),phi=field("vertex phi","vertex"),phiScratch=field("vertex phi scratch","vertex"),centre=field("pressure centre phi","cell",true);
const handles=[velocityA,velocityB,taps,volumeA,volumeB,gamma,phi,phiScratch,centre];
// The frame's published coarse phi: a caller's texture as the phi field's base.
const coarse=device.createTexture({label:"coarse vertex phi",size:[...a.wideVertexTexture],dimension:"3d",format:"r32float",usage:GPUTextureUsage.STORAGE_BINDING|GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_SRC|GPUTextureUsage.COPY_DST});
storage.adoptBase(phi,coarse);

const pool=new UniformPagePool(device,{table:8,blocks:a.blocks,idleSteps:2,box:{origin:[0,0,0],pages:a.pages}}),l=pool.layout,NB=l.blocks;
pool.seed(uniformPagePoolBoxSeed(l,{resident:(x,y,z)=>x+y+z===0}));
const n=a.tileKeys,usage=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST;
const topologies=[0,1].map(i=>device.createBuffer({label:`topology ${i}`,size:4*(5*n+l.tail.words+3),usage}));
await pool.initialize(topologies.map(buffer=>({buffer,offset:5*n})));
const ROW_OFFSET=5,lowFaces=[device.createBuffer({label:"negative A",size:4*a.lowFaceKeys,usage}),device.createBuffer({label:"walls",size:4*(a.lowFaceKeys+ROW_OFFSET+2),usage})];
const tileWords=new UniformPageTileWords(device,pool,topologies.map(buffer=>({buffer,tiles:n})));await tileWords.initialize();
const options={near:NEAR,lowFaces:[{buffer:lowFaces[0]!},{buffer:lowFaces[1]!,offset:ROW_OFFSET,value:1}],owners:[tileWords]};

assert.throws(()=>new UniformPageInitialise(device,pool,storage,{...options,near:EMPTY+1}),/nearer than the residency bound/);
assert.throws(()=>new UniformPageInitialise(device,pool,storage,{...options,chunks:3}),/chunks must be/);
assert.throws(()=>new UniformPageInitialise(device,pool,storage,{...options,lowFaces:[{buffer:lowFaces[0]!,offset:1}]}),/does not hold/);
{const other=new UniformPagePool(device,{table:8,blocks:NB+1,idleSteps:2});assert.throws(()=>new UniformPageInitialise(device,other,storage,options),/the pool 15/);other.destroy();}
const init=new UniformPageInitialise(device,pool,storage,options);
assert.throws(()=>init.encode(device.createCommandEncoder()),/not initialized/);
await init.initialize();
assert.equal(init.fields.length,9);assert.equal(init.passes,3);
console.log(`lattice ${dims.join("x")}: ${NB} blocks in ${G.join("x")}, ${init.fields.length} fields in ${init.passes} pipelines of ${init.groups} workgroups, limits ${device.limits.maxStorageTexturesPerShaderStage} storage textures / ${device.limits.maxStorageBuffersPerShaderStage} storage buffers`);

/** A field's textures now: its h store (when it has one) and its base. */
const texturesOf=(h:GPUTexture)=>{const base=storage.baseOf(h)!,p=storage.physical(h);return p===base?[base]:[p,base];};
const comps=(t:GPUTexture)=>t.format==="rgba32float"?4:1;
const reads=new Map<GPUTexture|GPUBuffer,GPUBuffer>();
const rowBytes=(t:GPUTexture)=>256*Math.ceil(t.width*4*comps(t)/256);
const readOf=(r:GPUTexture|GPUBuffer)=>{let b=reads.get(r);if(!b)reads.set(r,b=device.createBuffer({size:"format" in r?rowBytes(r)*r.height*r.depthOrArrayLayers:r.size,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST}));return b;};
const buffers=()=>[pool.state,...lowFaces,...topologies];
function poison():void{
 for(const h of handles)for(const t of texturesOf(h))device.queue.writeTexture({texture:t},new Uint32Array(t.width*t.height*t.depthOrArrayLayers*comps(t)).fill(NAN),{bytesPerRow:t.width*4*comps(t),rowsPerImage:t.height},[t.width,t.height,t.depthOrArrayLayers]);
 for(const b of lowFaces)device.queue.writeBuffer(b,0,new Uint32Array(b.size/4).fill(DEAD));
 for(const b of topologies)device.queue.writeBuffer(b,0,new Uint32Array(5*n).fill(DEAD));
}
async function readAll(encoder:GPUCommandEncoder):Promise<Map<GPUTexture|GPUBuffer,Uint32Array>>{
 for(const h of handles)for(const t of texturesOf(h))encoder.copyTextureToBuffer({texture:t},{buffer:readOf(t),bytesPerRow:rowBytes(t),rowsPerImage:t.height},[t.width,t.height,t.depthOrArrayLayers]);
 for(const b of buffers())encoder.copyBufferToBuffer(b,0,readOf(b),0,b.size);
 device.queue.submit([encoder.finish()]);
 const out=new Map<GPUTexture|GPUBuffer,Uint32Array>();
 await Promise.all([...reads].map(async([r,b])=>{await b.mapAsync(GPUMapMode.READ);out.set(r,new Uint32Array(b.getMappedRange()).slice());b.unmap();}));
 return out;
}
const blockOfTexel=(x:number,y:number,z:number,e:readonly number[])=>Math.floor(x/e[0]!)|(Math.floor(y/e[1]!)<<bits)|(Math.floor(z/e[2]!)<<(2*bits));
/** Every texel and word against the image of `claimed`; `owners`: the tile words were launched. */
function audit(label:string,data:Map<GPUTexture|GPUBuffer,Uint32Array>,claimed:ReadonlySet<number>,owners:boolean):number{
 let bad=0,texels=0,air=0;const miss=(m:string)=>{if(bad++<12)console.log(`  MISMATCH ${label}: ${m}`);};
 for(const h of handles)for(const t of texturesOf(h)){
  const words=data.get(t)!,c=comps(t),row=rowBytes(t)/4,e=[t.width/G[0],t.height/G[1],t.depthOrArrayLayers/G[2]];
  const value=bitsOf(storage.kindOf(h)==="vertex"?EMPTY:0);
  for(let z=0;z<t.depthOrArrayLayers;z++)for(let y=0;y<t.height;y++)for(let x=0;x<t.width;x++){
   const block=blockOfTexel(x,y,z,e),hit=claimed.has(block),at=row*(y+t.height*z)+c*x;texels++;if(hit)air++;
   for(let k=0;k<c;k++){const want=hit?(c===4||k===0?value:0):NAN;if(words[at+k]!==want){miss(`${t.label} ${t.width}x${t.height}x${t.depthOrArrayLayers} texel ${x},${y},${z}.${k} block ${block}: ${words[at+k]!.toString(16)}, want ${want.toString(16)}`);break;}}
  }
 }
 lowFaces.forEach((b,i)=>{const words=data.get(b)!,offset=i?ROW_OFFSET:0,value=i?bitsOf(1):0;
  for(let w=0;w<words.length;w++){const k=w-offset,want=k>=0&&k<a.lowFaceKeys&&claimed.has(Math.floor(k/KEYS.lowFace))?value:DEAD;if(words[w]!==want)miss(`${b.label} word ${w}: ${words[w]!.toString(16)}, want ${want.toString(16)}`);}});
 const state=data.get(pool.state)!;
 for(const b of topologies){const words=data.get(b)!;
  for(let t=0;t<n;t++){
   const hit=owners&&claimed.has(Math.floor(t/KEYS.tile)),want=hit?[TILE_WORD,DEAD,0,0,0]:[DEAD,DEAD,DEAD,DEAD,DEAD],got=[words[t],words[n+t],words[2*n+2*t],words[2*n+2*t+1],words[4*n+t]];
   if(got.some((g,i)=>g!==want[i]))miss(`${b.label} key ${t}: ${got.map(g=>g!.toString(16))}, want ${want.map(g=>g.toString(16))}`);
  }
  // The pool's tail behind the tile-keyed words is the commit's, untouched here.
  for(let i=0;i<l.tail.words;i++)if(words[5*n+i]!==state[l.state.table+i]){miss(`${b.label} tail word ${i}`);break;}
 }
 console.log(`${label}: ${claimed.size} blocks [${[...claimed].join(",")}], ${texels} texels read, ${air} in claimed blocks, ${bad} mismatches`);
 if(bad)errors.push(`${label}: ${bad} mismatches`);
 return bad;
}
const status=(data:Map<GPUTexture|GPUBuffer,Uint32Array>)=>{const s=data.get(pool.state)!;return {generation:s[STATUS.generation]!,resident:s[STATUS.resident]!,claimed:s[STATUS.claimed]!,free:s[STATUS.free]!,pending:s[STATUS.pending]!,fatal:s[STATUS.fatal]!,
 list:[...s.subarray(l.state.claimed,l.state.claimed+s[STATUS.claimed]!)]};};
/** Poison, commit the queued requests, initialise, read everything back and audit. */
async function commit(label:string,pages:readonly (readonly [number,number,number])[]){
 poison();for(const [x,y,z] of pages)pool.request(x,y,z);
 const encoder=device.createCommandEncoder();pool.encodeCommit(encoder);init.encode(encoder);
 const data=await readAll(encoder),s=status(data);
 assert.equal(s.fatal,0,`${label}: pool fatal ${s.fatal}`);assert.equal(s.claimed,pages.length,`${label}: ${s.claimed} claims for ${pages.length} requests`);
 for(const b of s.list)assert.ok(b>=1&&b<=NB-2,`${label}: new block ${b}`);
 console.log(`${label}: generation ${s.generation}, resident ${s.resident}, retired this commit ${s.pending}, free ${s.free}`);
 audit(label,data,new Set(s.list),true);return s;
}

// The fields are their base blocks alone: every h slot binds a stand-in.
assert.ok(storage.baseOnly);
await commit("base only, 2 claims",[[1,0,0],[2,1,1]]);
// An h generation beside the bases: the storage rebuilds the tracked groups.
(storage as unknown as {simulation:unknown}).simulation={capacity:{fineTiles:1}};storage.reserve({} as never);
assert.ok(!storage.baseOnly);
for(const h of handles)assert.equal(texturesOf(h).length,h===centre?1:2,`${h.label} textures`);
const idle=[await commit("h resident, no claim",[]),await commit("h resident, no claim, retiring",[])];
assert.ok(idle.some(s=>s.pending>0),"no page retired in the idle commits");
const every:[number,number,number][]=[];for(let z=0;z<a.pages[2];z++)for(let y=0;y<a.pages[1];y++)for(let x=0;x<a.pages[0];x++)every.push([x,y,z]);
const many=await commit("h resident, 10 claims",every.slice(0,10));
// Wall clock of the launches alone (submit to work done), the new list as the last commit left it.
const time=async(label:string,rounds:number,blocks:number)=>{
 await device.queue.onSubmittedWorkDone();const t0=performance.now(),encoder=device.createCommandEncoder();
 for(let i=0;i<rounds;i++)init.encode(encoder);device.queue.submit([encoder.finish()]);await device.queue.onSubmittedWorkDone();
 const ms=(performance.now()-t0)/rounds;console.log(`${label}: ${ms.toFixed(3)} ms an encode${blocks?`, ${(ms/blocks).toFixed(3)} ms a block`:""} (wall clock over ${rounds} encodes)`);
};
await time(`timing, ${many.claimed} blocks in the new list`,10,many.claimed);
await commit("h resident, no claim after ten",[]);
await time("timing, empty new list",200,0);
// encodeAll: every block of every field and low-face buffer, no owner.
{poison();const encoder=device.createCommandEncoder();init.encodeAll(encoder);audit("encodeAll",await readAll(encoder),new Set(Array.from({length:NB},(_,i)=>i)),false);}
// A field created after the initialise was built is fatal at the next encode.
storage.createField("late field","cell","r32float");
assert.throws(()=>init.encode(device.createCommandEncoder()),/holds 10 fields, 9 when/);
await device.queue.onSubmittedWorkDone();
console.log(`${errors.length?"FAILED":"passed"} in ${((performance.now()-started)/1000).toFixed(1)} s`);
assert.deepEqual(errors,[]);
init.destroy();pool.destroy();storage.destroy();coarse.destroy();for(const b of [...lowFaces,...topologies,...reads.values()])b.destroy();device.destroy();
