import assert from "node:assert/strict";
import test from "node:test";
import {UFB_EDGE,UFB_SHELL,UNIFORM_FINE_BRICK_FLOOR,UniformFineBrickStore,uniformFineBrickAtlasGrid,uniformFineBrickAtlasTexel,uniformFineBrickCapacity,uniformFineBrickParticleCapacity,uniformFineBrickShell} from "../lib/methods/uniform/uniform-fine-brick-store";
import {UNIFORM_ARENA_CLASS,UNIFORM_ARENA_RELEASE_ENTRY,UNIFORM_ARENA_TABLE_ENTRIES} from "../lib/methods/uniform/uniform-brick-arena";
import {withUniformDevice} from "./helpers/uniform-geometric";

test("fine brick capacity: the scene's bricks, else the t=0 census x 1.5 over a floor of 256, within the lattice",()=>{
 assert.equal(uniformFineBrickCapacity(100,0),UNIFORM_FINE_BRICK_FLOOR);
 assert.equal(uniformFineBrickCapacity(100,1000),1500);
 assert.equal(uniformFineBrickCapacity(100,1001),1502);
 assert.equal(uniformFineBrickCapacity(3,1000),192);
 assert.equal(uniformFineBrickCapacity(100,1000,{bricks:777}),777);
 assert.equal(uniformFineBrickCapacity(2,0,{bricks:777}),128);
 assert.throws(()=>uniformFineBrickCapacity(100,0,{bricks:0}));
 assert.equal(uniformFineBrickParticleCapacity(1000),4*512*1000);
 assert.equal(uniformFineBrickParticleCapacity(1000,{particlesPerFineCell:2}),2*512*1000);
});

test("fine brick atlas: a compact grid of disjoint 10³ slot boxes inside the 3D texture limit",()=>{
 for(const capacity of [1,7,256,1000,4097,12800]){
  const grid=uniformFineBrickAtlasGrid(capacity);
  assert.ok(grid[0]*grid[1]*grid[2]>=capacity&&grid.every(n=>UFB_EDGE*n<=2048),`${capacity}: ${grid}`);
  assert.ok(grid[0]*grid[1]*grid[2]<capacity+grid[0]*grid[1],`${capacity}: one partial layer at most`);
 }
 assert.throws(()=>uniformFineBrickAtlasGrid(205**3));
 const grid=uniformFineBrickAtlasGrid(50),seen=new Set<string>();
 for(let slot=0;slot<50;slot++)for(const l of [[-1,-1,-1],[8,8,8],[3,-1,8]]){const key=uniformFineBrickAtlasTexel(grid,slot,l).join();assert.ok(!seen.has(key));seen.add(key);}
 const store=new UniformFineBrickStore({blocks:10,capacity:300,channels:[{name:"u",layout:"face",sampled:true},{name:"phi",layout:"vertex",sampled:true,banks:2},{name:"p",layout:"cell"}],lists:[]});
 assert.equal(store.atlases.length,5);
 assert.equal(store.slotBytes,4*520+5*4*1000);
});

test("fine brick apron: 488 texels, each once, exactly the shell of the 10³ box",()=>{
 const seen=new Set<string>();
 for(let k=0;k<UFB_SHELL;k++){
  const l=uniformFineBrickShell(k);
  assert.ok(l.every(v=>v>=-1&&v<=8)&&l.some(v=>v===-1||v===8),`${k}: ${l}`);
  seen.add(l.join());
 }
 assert.equal(seen.size,488);
});

// A 4x2x2 page box in the 64³ window; page (1,0,1) AIR, the rest stored, block = page index + 1.
const PX=4,PY=2,PZ=2,BLOCKS=PX*PY*PZ+2,WINDOW=64**3;
const pageIndex=(x:number,y:number,z:number)=>x+PX*(y+PY*z);
const airPage=pageIndex(1,0,1);
function pageTable():Uint32Array<ArrayBuffer>{
 const t=new Uint32Array(WINDOW+BLOCKS).fill(0);
 t.fill(0xffffffff,WINDOW);
 for(let z=0;z<PZ;z++)for(let y=0;y<PY;y++)for(let x=0;x<PX;x++){
  const i=pageIndex(x,y,z);if(i===airPage)continue;
  t[x|(y<<6)|(z<<12)]=i+1;t[WINDOW+i+1]=x|(y<<6)|(z<<12);
 }
 return t;
}
const blockOf=(page:readonly number[])=>{
 const [x,y,z]=page;if(x!<0||y!<0||z!<0||x!>=PX||y!>=PY||z!>=PZ)return 0;const i=pageIndex(x!,y!,z!);return i===airPage?0:i+1;
};
const brickId=(c:readonly number[])=>{const b=blockOf(c.map(v=>Math.floor(v/32)));if(!b)return -1;const l=c.map(v=>(v>>3)&3);return 64*b+(l[0]!|(l[1]!<<2)|(l[2]!<<4));};
const origin=(id:number)=>{const b=id>>6,i=b-1,l=id&63;return [32*(i%PX)+8*(l&3),32*(Math.floor(i/PX)%PY)+8*((l>>2)&3),32*Math.floor(i/(PX*PY))+8*(l>>4)];};
/** The analytic legacy field: exact in f32. */
const field=(p:readonly number[])=>p[0]!+256*p[1]!+65536*p[2]!;
/** Tile mask of a brick: a brick with an odd id has its four low-z tiles live only. */
const TILE_MASK="select(0xffu,0x0fu,(id&1u)!=0u)";
const tileLive=(id:number,s:readonly number[])=>(id&1)===0||s[2]!<4;

const SHIM=`
const UM_BLOCKS:u32=${BLOCKS}u;
const UM_FREE:u32=0xffffffffu;
@group(0) @binding(0) var<storage,read_write> ufbTable:array<atomic<u32>>;
@group(0) @binding(1) var<storage,read> pages:array<u32>;
@group(0) @binding(2) var<storage,read_write> ufbPool:array<u32>;
@group(0) @binding(3) var atlas:texture_storage_3d<r32float,read_write>;
@group(0) @binding(4) var<storage,read_write> outWords:array<f32>;
@group(0) @binding(5) var<storage,read> marks:array<u32>;
fn umPageWord(p:vec3i)->u32{let q=vec3u(p&vec3i(63));return pages[q.x|(q.y<<6u)|(q.z<<12u)];}
fn umBlockWord(b:u32)->u32{return pages[${WINDOW}u+b];}
fn umBlockStored(b:u32)->bool{return b-1u<UM_BLOCKS-2u;}
fn umBlockPage(b:u32)->vec3u{let r=umBlockWord(b);return vec3u(r&63u,(r>>6u)&63u,(r>>12u)&63u);}
fn umCorner(k:u32,side:u32)->vec3u{return vec3u(k%side,(k/side)%side,k/(side*side));}
@compute @workgroup_size(64) fn mark(@builtin(global_invocation_id) gid:vec3u){if(gid.x<marks[0]){ubaMark(marks[1u+gid.x],${UNIFORM_ARENA_CLASS.fine}u);}}
fn legacy(p:vec3i)->f32{return f32(p.x)+256.0*f32(p.y)+65536.0*f32(p.z);}
fn seam(p:vec3i,word:u32)->f32{if(word==UB_NONE_ID){return -1.0;}if(word>=UBA_SEAM){return -2.0-f32(word&0xffffu);}return -1000.0;}
// Scatter: a dense box of owned indices (128x64x64), channel j at j*BOX.
const BOX:u32=${128*64*64}u;
fn outIndex(p:vec3i)->u32{return u32(p.x)+128u*(u32(p.y)+64u*u32(p.z));}
fn putAtlas(p:vec3i,v:f32){outWords[outIndex(p)]=v;}
fn putPool(p:vec3i,v:f32){outWords[BOX+outIndex(p)]=v;}
`;

async function run(device:GPUDevice,steps:readonly (readonly number[])[],capacity:number,after?:(store:UniformFineBrickStore,atlas:GPUTexture,words:Uint32Array)=>Promise<void>){
 const store=new UniformFineBrickStore({blocks:BLOCKS,capacity,channels:[{name:"u",layout:"face",sampled:true},{name:"q",layout:"cell"}],lists:[{name:"fine",any:UNIFORM_ARENA_CLASS.fine}]});
 const a=store.arena,layout=a.tableLayout;
 const code=SHIM+store.moduleWGSL(0,TILE_MASK)
  +store.copyWGSL("gather",{list:"enter",direction:"gather",channels:[{name:"u",component:0,fn:"legacy",binding:"atlas"},{name:"q",fn:"legacy"}],workgroup:64})
  +store.apronWGSL("apron",{list:"live",channels:[{name:"u",component:0,binding:"atlas",seam:"seam"}],workgroup:64})
  +store.copyWGSL("scatter",{list:"fine",direction:"scatter",channels:[{name:"u",component:0,fn:"putAtlas",binding:"atlas"},{name:"q",fn:"putPool"}],workgroup:64});
 const module=device.createShaderModule({code});
 const errors=(await module.getCompilationInfo()).messages.filter(m=>m.type==="error");
 assert.deepEqual(errors.map(m=>`${m.lineNum}: ${m.message}`),[]);
 const usage=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC;
 const table=device.createBuffer({size:4*a.tableWords,usage}),pages=device.createBuffer({size:4*(WINDOW+BLOCKS),usage});
 const pool=device.createBuffer({size:Math.max(16,4*a.poolWords),usage}),out=device.createBuffer({size:4*2*128*64*64,usage});
 const marks=device.createBuffer({size:4*(1+64*BLOCKS),usage});
 const atlas=store.createAtlases(device).get("u/0/0")!;
 device.queue.writeBuffer(pages,0,pageTable());
 for(const w of a.initialWrites(0))device.queue.writeBuffer(table,w.offset,w.data as Uint32Array<ArrayBuffer>);
 const bgl=device.createBindGroupLayout({entries:[
  ...[0,1,2,4,5].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,buffer:{type:(binding===1||binding===5?"read-only-storage":"storage") as GPUBufferBindingType}})),
  {binding:3,visibility:GPUShaderStage.COMPUTE,storageTexture:{access:"read-write" as const,format:"r32float" as const,viewDimension:"3d" as const}}]});
 const group=device.createBindGroup({layout:bgl,entries:[{binding:0,resource:{buffer:table}},{binding:1,resource:{buffer:pages}},{binding:2,resource:{buffer:pool}},{binding:3,resource:atlas.createView()},{binding:4,resource:{buffer:out}},{binding:5,resource:{buffer:marks}}]});
 const pipelineLayout=device.createPipelineLayout({bindGroupLayouts:[bgl]});
 const pipelines=new Map<string,GPUComputePipeline>();
 for(const entry of ["mark",...UNIFORM_ARENA_TABLE_ENTRIES,UNIFORM_ARENA_RELEASE_ENTRY,"gather","apron","scatter"])pipelines.set(entry,await device.createComputePipelineAsync({layout:pipelineLayout,compute:{module,entryPoint:entry}}));
 const read=async(buffer:GPUBuffer)=>{
  const staging=device.createBuffer({size:buffer.size,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
  const e=device.createCommandEncoder();e.copyBufferToBuffer(buffer,0,staging,0,buffer.size);device.queue.submit([e.finish()]);
  await staging.mapAsync(GPUMapMode.READ);const data=staging.getMappedRange().slice(0);staging.unmap();staging.destroy();return data;
 };
 const readAtlas=async()=>{
  const [w,h,d]=store.atlasExtent,row=Math.ceil(4*w/256)*256;
  const staging=device.createBuffer({size:row*h*d,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
  const e=device.createCommandEncoder();e.copyTextureToBuffer({texture:atlas},{buffer:staging,bytesPerRow:row,rowsPerImage:h},[w,h,d]);device.queue.submit([e.finish()]);
  await staging.mapAsync(GPUMapMode.READ);const data=new Float32Array(staging.getMappedRange().slice(0));staging.unmap();staging.destroy();
  return (t:readonly number[])=>data[(row/4)*(t[1]!+h*t[2]!)+t[0]!]!;
 };
 const results=[];
 for(const step of steps){
  device.queue.writeBuffer(marks,0,new Uint32Array([step.length,...step]));
  const e=device.createCommandEncoder();const clear=a.stepClear(0);e.clearBuffer(table,clear.offset,clear.size);e.clearBuffer(out);
  const pass=e.beginComputePass();pass.setBindGroup(0,group);
  for(const entry of ["mark",...UNIFORM_ARENA_TABLE_ENTRIES,"gather","apron","scatter",UNIFORM_ARENA_RELEASE_ENTRY]){pass.setPipeline(pipelines.get(entry)!);pass.dispatchWorkgroups(entry==="mark"?Math.ceil(64*BLOCKS/64):8);}
  pass.end();device.queue.submit([e.finish()]);
  results.push({words:new Uint32Array(await read(table)),out:new Float32Array(await read(out)),atlas:await readAtlas()});
 }
 if(after)await after(store,atlas,results[results.length-1]!.words);
 for(const b of [table,pages,pool,out,marks])b.destroy();atlas.destroy();
 return {store,layout,results};
}

test("fine brick store: persistent slots, churn lists, records, fail-fast, apron and gather/scatter round trips",{skip:!process.env.WEBGPU_NODE_MODULE},async()=>{
 // Bricks along a row through pages (0..3,0,0) and some of their neighbours, in two steps that overlap, then an overflow.
 const at=(x:number,y:number,z:number)=>brickId([x,y,z]);
 const first=[at(0,0,0),at(8,0,0),at(24,0,0),at(32,0,0),at(40,8,0),at(8,8,8),at(64,0,32)];
 const second=[at(8,0,0),at(24,0,0),at(32,0,0),at(48,0,0),at(56,0,0),at(8,8,8),at(32,32,0)];
 const many=Array.from({length:40},(_,i)=>64*(1+(i%16))+(i>>4)).filter(id=>(id>>6)-1!==airPage);
 assert.ok(first.every(id=>id>=0)&&second.every(id=>id>=0));
 await withUniformDevice("fine brick store",async device=>{
  const capacity=24,{store,layout,results}=await run(device,[first,second,many],capacity);
  const a=store.arena,COUNTS=18,fatal=0,top=1;
  const slotOf=(words:Uint32Array,id:number)=>words[layout.table+id]!;
  // Step 1: every marked brick entered with a slot of its own.
  {const w=results[0]!.words;
   assert.equal(w[fatal],0);assert.equal(w[COUNTS]!,first.length);assert.equal(w[COUNTS+1]!,0);assert.equal(w[COUNTS+2]!,first.length);
   const slots=first.map(id=>slotOf(w,id));assert.equal(new Set(slots).size,first.length);assert.ok(slots.every(s=>s<capacity));
   for(const [i,id] of first.entries())assert.equal(w[layout.brick+slots[i]!],id);
   assert.equal(w[top],capacity-first.length);}
  // Step 2: kept bricks keep their slots; left bricks are listed and their slots return to the stack.
  {const w0=results[0]!.words,w=results[1]!.words;
   for(const id of first.filter(id=>second.includes(id)))assert.equal(slotOf(w,id),slotOf(w0,id),`brick ${id} keeps its slot`);
   const left=first.filter(id=>!second.includes(id)),entered=second.filter(id=>!first.includes(id));
   assert.equal(w[COUNTS]!,entered.length);assert.equal(w[COUNTS+1]!,left.length);
   const leave=new Set(Array.from(w.subarray(layout.leave,layout.leave+left.length)));
   assert.deepEqual([...leave].sort(),left.map(id=>slotOf(w0,id)).sort());
   for(const id of left)assert.equal(slotOf(w,id),0xffffffff,`released brick ${id}`);
   assert.equal(w[top],capacity-second.length);
   const slots=second.map(id=>slotOf(w,id));assert.equal(new Set(slots).size,second.length);
   // Records: a live neighbour's slot, a seam to a stored page's block, or none outside stored pages.
   for(const id of second){
    const slot=slotOf(w,id),o=origin(id);
    for(let n=0;n<27;n++){
     const c=[o[0]!+8*((n%3)-1),o[1]!+8*((Math.floor(n/3)%3)-1),o[2]!+8*(Math.floor(n/9)-1)],other=brickId(c),word=w[layout.record+32*slot+n]!;
     if(other<0)assert.equal(word,0xffffffff,`record ${id}/${n}: outside`);
     else if(second.includes(other))assert.equal(word,slotOf(w,other),`record ${id}/${n}: live`);
     else assert.equal(word,(0x80000000|(other>>6))>>>0,`record ${id}/${n}: seam`);
    }
    assert.equal(w[layout.record+32*slot+28],(id&1)?0x0f:0xff);assert.equal(w[layout.record+32*slot+29],id);
   }
   // Gather -> apron -> scatter: cores hold the legacy field, aprons the neighbours' cores or the seam, the scatter returns both channels.
   const {atlas,out}=results[1]!,grid=store.atlasGrid;
   for(const id of second){
    const slot=slotOf(w,id),o=origin(id);
    for(let k=0;k<512;k++){
     const l=[k&7,(k>>3)&7,k>>6],p=[o[0]!+l[0]!,o[1]!+l[1]!,o[2]!+l[2]!];
     assert.equal(atlas(uniformFineBrickAtlasTexel(grid,slot,l)),field(p),`core ${id} ${l}`);
     const box=p[0]!+128*(p[1]!+64*p[2]!);
     assert.equal(out[box],field(p),`atlas scatter ${p}`);assert.equal(out[128*64*64+box],field(p),`pool scatter ${p}`);
    }
    for(let k=0;k<UFB_SHELL;k++){
     const l=uniformFineBrickShell(k),p=[o[0]!+l[0]!,o[1]!+l[1]!,o[2]!+l[2]!],other=brickId(p),s=p.map(v=>v&7);
     const expected=other<0?-1:!second.includes(other)?-2-(other>>6):tileLive(other,s)?field(p):-1000;
     assert.equal(atlas(uniformFineBrickAtlasTexel(grid,slot,l)),expected,`apron ${id} ${l}`);
    }
   }}
  // Step 3: more bricks than free slots is the fatal word, never growth.
  {const w=results[2]!.words;assert.notEqual(w[fatal],0);assert.ok((w[fatal]!&0x80000000)!==0);assert.ok(w[COUNTS+2]!<=capacity);}
  assert.equal(a.fineCapacity,capacity);
 });
});

test("fine brick atlas: hardware trilinear taps across a brick face read the neighbour through the apron",{skip:!process.env.WEBGPU_NODE_MODULE},async()=>{
 // Two live bricks side by side along x: taps of the left brick from owned
 // index x = 0 to 8 cross into the right brick's first plane through the
 // apron. y and z sit on texel centres, so only the x weight (8-bit) rounds.
 const ids=[brickId([0,0,0]),brickId([8,0,0])],N=65;
 await withUniformDevice("fine brick atlas taps",async device=>{
  await run(device,[ids],16,async(store,atlas,words)=>{
   const slot=words[store.arena.tableLayout.table+ids[0]!]!;
   const module=device.createShaderModule({code:`@group(0) @binding(0) var tex:texture_3d<f32>;
@group(0) @binding(1) var samp:sampler;
@group(0) @binding(2) var<storage,read_write> o:array<f32>;
${store.atlasWGSL()}
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) g:vec3u){if(g.x>=${N}u){return;}o[g.x]=textureSampleLevel(tex,samp,ufbAtlasPoint(${slot}u,vec3f(f32(g.x)/8.0,2.0,3.0)),0.0).x;}`});
   const pipeline=await device.createComputePipelineAsync({layout:"auto",compute:{module,entryPoint:"main"}});
   const out=device.createBuffer({size:4*N,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC}),staging=device.createBuffer({size:4*N,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
   const sampler=device.createSampler({magFilter:"linear",minFilter:"linear",addressModeU:"clamp-to-edge",addressModeV:"clamp-to-edge",addressModeW:"clamp-to-edge"});
   const group=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:atlas.createView()},{binding:1,resource:sampler},{binding:2,resource:{buffer:out}}]});
   const e=device.createCommandEncoder(),pass=e.beginComputePass();pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroups(2);pass.end();
   e.copyBufferToBuffer(out,0,staging,0,4*N);device.queue.submit([e.finish()]);
   await staging.mapAsync(GPUMapMode.READ);const values=new Float32Array(staging.getMappedRange().slice(0));staging.unmap();
   for(let i=0;i<N;i++){const x=i/8;assert.ok(Math.abs(values[i]!-field([x,2,3]))<0.03,`tap at x ${x}: ${values[i]} against ${field([x,2,3])}`);}
   out.destroy();staging.destroy();
  });
 });
});
