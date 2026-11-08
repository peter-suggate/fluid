import assert from "node:assert/strict";
import test from "node:test";
import {UNIFORM_ARENA_CLASS,UNIFORM_ARENA_RELEASE_ENTRY,UNIFORM_ARENA_TABLE_ENTRIES,UniformBrickArena} from "../lib/methods/uniform/uniform-brick-arena";
import {UBA_STAGE_WORDS,ubaStageWGSL,type UbaStageChannel} from "../lib/methods/uniform/uniform-brick-stage";
import {withUniformDevice} from "./helpers/uniform-geometric";

// A 4x2x2 page box in the 64³ page window; page (1,0,1) AIR, the rest stored, block = page index + 1.
const PX=4,PY=2,PZ=2,BLOCKS=PX*PY*PZ+2,WINDOW=64**3;
const pageIndex=(x:number,y:number,z:number)=>x+PX*(y+PY*z);
const airPage=pageIndex(1,0,1);
const held=Array.from({length:PX*PY*PZ},(_,i)=>i).filter(i=>i!==airPage).map(i=>i+1);
function pageTable():Uint32Array<ArrayBuffer>{
 const t=new Uint32Array(WINDOW+BLOCKS).fill(0);t.fill(0xffffffff,WINDOW);
 for(let z=0;z<PZ;z++)for(let y=0;y<PY;y++)for(let x=0;x<PX;x++){const i=pageIndex(x,y,z);if(i===airPage)continue;t[x|(y<<6)|(z<<12)]=i+1;t[WINDOW+i+1]=x|(y<<6)|(z<<12);}
 return t;
}
const blockOfPage=(page:readonly number[])=>{const [x,y,z]=page as [number,number,number];if(x<0||y<0||z<0||x>=PX||y>=PY||z>=PZ)return 0;const i=pageIndex(x,y,z);return i===airPage?0:i+1;};
const brickId=(c:readonly number[])=>{const b=blockOfPage(c.map(v=>Math.floor(v/32)));if(!b)return -1;const l=c.map(v=>(v>>3)&3);return 64*b+(l[0]!|(l[1]!<<2)|(l[2]!<<4));};
const pageOrigin=(b:number)=>{const i=b-1;return [i%PX,Math.floor(i/PX)%PY,Math.floor(i/(PX*PY))];};
const brickOrigin=(id:number)=>{const p=pageOrigin(id>>6),l=id&63;return [32*p[0]!+8*(l&3),32*p[1]!+8*((l>>2)&3),32*p[2]!+8*(l>>4)];};
/** An odd brick id has its four low-z tiles live only. */
const TILE_MASK="select(0xffu,0x0fu,(id&1u)!=0u)";
const tileLive=(id:number,l:readonly number[])=>(id&1)===0||l[2]!<4;

// The fields, f32-exact on both sides: an h field, a 4h field with a known flag, a weight.
const fineValue=(p:readonly number[])=>Math.fround(p[0]!*0.5+p[1]!*0.25-p[2]!*0.125+((p[0]!*3+p[1]!+p[2]!*5)&7)*0.0625);
const coarseValue=(c:readonly number[])=>Math.fround(3+c[0]!-2*c[1]!+0.5*c[2]!+((c[0]!*7+c[1]!*3+c[2]!)&3)*0.25);
const coarseKnown=(c:readonly number[])=>((c[0]!+2*c[1]!+3*c[2]!)%5+5)%5!==0;
const weightValue=(p:readonly number[])=>((p[0]!+p[1]!+p[2]!)&3)*0.5;
const OUTSIDE=-7;
const H0=2,H2=1,E0=8+2*H0,E2=8+2*H2;

const SHIM=/* wgsl */`
const UM_BLOCKS:u32=${BLOCKS}u;
const UM_FREE:u32=0xffffffffu;
@group(0) @binding(0) var<storage,read_write> table:array<atomic<u32>>;
@group(0) @binding(1) var<storage,read> pages:array<u32>;
@group(0) @binding(2) var<storage,read_write> pool:array<u32>;
@group(0) @binding(3) var<storage,read_write> outWords:array<f32>;
@group(0) @binding(4) var<storage,read> marks:array<u32>;
fn umPageWord(p:vec3i)->u32{let q=vec3u(p&vec3i(63));return pages[q.x|(q.y<<6u)|(q.z<<12u)];}
fn umBlockWord(b:u32)->u32{return pages[${WINDOW}u+b];}
fn umBlockStored(b:u32)->bool{return b-1u<UM_BLOCKS-2u;}
fn umBlockPage(b:u32)->vec3u{let r=umBlockWord(b);return vec3u(r&63u,(r>>6u)&63u,(r>>12u)&63u);}
fn umCorner(k:u32,side:u32)->vec3u{return vec3u(k%side,(k/side)%side,k/(side*side));}
const HELD=array<u32,${held.length}>(${held.map(b=>`${b}u`).join(",")});
fn umHeldBlockCount()->u32{return ${held.length}u;}
fn umHeldBlock(r:u32)->u32{return HELD[r];}
fn fineValue(p:vec3i)->f32{return f32(p.x)*0.5+f32(p.y)*0.25-f32(p.z)*0.125+f32((p.x*3+p.y+p.z*5)&7)*0.0625;}
fn coarseValue(c:vec3i)->f32{return 3.0+f32(c.x)-2.0*f32(c.y)+0.5*f32(c.z)+f32((c.x*7+c.y*3+c.z)&3)*0.25;}
fn coarse(c:vec3i)->vec2f{let m=((c.x+2*c.y+3*c.z)%5+5)%5;return vec2f(coarseValue(c),select(0.0,1.0,m!=0));}
fn weight(p:vec3i)->f32{return f32((p.x+p.y+p.z)&3)*0.5;}
fn outside(p:vec3i)->f32{return ${OUTSIDE}.0;}
// The fn core: the neighbour's brick id (resolved once) and its local index.
fn resolveBrick(w:u32,o:vec3i)->u32{if(w>=UBA_SEAM){return 0u;}return ubaBrick(w);}
fn brickCore(w:u32,base:u32,l:vec3u)->f32{return fineValue(vec3i(ubaOrigin(base))+vec3i(l))+1000.0;}
@compute @workgroup_size(64) fn mark(@builtin(global_invocation_id) gid:vec3u){if(gid.x<marks[0]){ubaMark(marks[1u+gid.x],${UNIFORM_ARENA_CLASS.fine}u);}}
// Pool cores: q at level 0 holds the h field, at level 2 the 4h field.
@compute @workgroup_size(64) fn fill(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) lane:u32){
 let n=ubaCount(UBA_LIVE_COUNT);
 for(var e=group.x;e<n;e+=8u){let slot=atomicLoad(&table[UBA_LIVE+e]);let o=vec3i(ubaOrigin(ubaBrick(slot)));
  for(var k=lane;k<512u;k+=64u){ubaStore(UBA_Q_L0_0_0,slot,k,fineValue(o+vec3i(vec3u(k&7u,(k>>3u)&7u,k>>6u))));}}
 let m=ubaLevelCount(2u);
 for(var e=group.x;e<m;e+=8u){let b=ubaLevelSlot(2u,e);let o=vec3i(umBlockPage(b)<<vec3u(3u));
  for(var k=lane;k<512u;k+=64u){ubaStore(UBA_Q_L2_0_0,b,k,coarseValue(o+vec3i(vec3u(k&7u,(k>>3u)&7u,k>>6u))));}}
}
`;
const level0:UbaStageChannel[]=[
 {name:"cell",layout:"cell",halo:H0,core:{pool:"q"},seam:"prolong",coarse:"coarse",outside:"outside"},
 {name:"face",layout:"face",axis:1,halo:H0,core:"brickCore",seam:"prolong",coarse:"coarse",outside:"outside"},
 {name:"vertex",layout:"vertex",halo:H0,core:{pool:"q"},seam:"prolong",coarse:"coarse"},
];
const level2:UbaStageChannel[]=[
 {name:"cell",layout:"cell",halo:H2,core:{pool:"q"},seam:"restrict",fine:"fineValue",weight:"weight",outside:"outside"},
 {name:"face",layout:"face",axis:2,halo:H2,core:{pool:"q"},seam:"restrict",fine:"fineValue",weight:"weight",outside:"outside"},
 {name:"vertex",layout:"vertex",halo:H2,core:{pool:"q"},seam:"restrict",fine:"fineValue"},
];
const E0_3=E0**3,E2_3=E2**3,PER0=3*E0_3,PER2=3*E2_3;
const MAX_LIVE=32,OUT2=MAX_LIVE*PER0;
/** Copies every staged window out: level 0 live slots from word 0, level 2 pages from OUT2. */
function stageEntries(arena:UniformBrickArena):string{
 const copy=(K:string,e3:number,per:number,at:string)=>["cell","face","vertex"].map((ch,j)=>`outWords[${at}+${per}u*e+${j*e3}u+k]=ubaW_${K}_${ch}[k];`).join("");
 return ubaStageWGSL(arena,"h",{level:0,workgroup:64,channels:level0,resolve:"resolveBrick"})+ubaStageWGSL(arena,"c",{level:2,workgroup:64,channels:level2})+/* wgsl */`
var<workgroup> count0:u32;
var<workgroup> count2:u32;
@compute @workgroup_size(64) fn stage0(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) lane:u32){
 if(lane==0u){count0=ubaCount(UBA_LIVE_COUNT);}
 let n=workgroupUniformLoad(&count0);
 for(var e=group.x;e<n;e+=8u){
  let slot=atomicLoad(&table[UBA_LIVE+e]);ubaStage_h(slot,lane);
  for(var k=lane;k<${E0_3}u;k+=64u){${copy("h",E0_3,PER0,"0u")}}
 }
}
@compute @workgroup_size(64) fn stage2(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) lane:u32){
 if(lane==0u){count2=ubaLevelCount(2u);}
 let n=workgroupUniformLoad(&count2);
 for(var e=group.x;e<n;e+=8u){
  let b=ubaLevelSlot(2u,e);ubaStage_c(b,lane);
  for(var k=lane;k<${E2_3}u;k+=64u){${copy("c",E2_3,PER2,`${OUT2}u`)}}
 }
}
`;
}


// The CPU twin of each staged value.
function prolong(layout:"cell"|"face"|"vertex",axis:number,p:readonly number[]):number{
 const known=(c:readonly number[])=>coarseKnown(c),val=(c:readonly number[])=>coarseValue(c);
 const f32=Math.fround;
 if(layout==="cell"){
  const C=p.map(v=>v>>2);if(!known(C))return val(C);
  let v=val(C);const x=p.map(q=>f32(f32(f32((q&3)+0.5)*0.25)-0.5));
  for(let a=0;a<3;a++){
   const lo=C.slice(),hi=C.slice();lo[a]!--;hi[a]!++;let s=0;
   if(known(lo)&&known(hi))s=f32(0.5*f32(val(hi)-val(lo)));else if(known(hi))s=f32(val(hi)-val(C));else if(known(lo))s=f32(val(C)-val(lo));
   v=f32(v+f32(s*x[a]!));
  }
  return v;
 }
 if(layout==="face"){
  const C=p.map(v=>v>>2),q=(p[axis]!&3)+1;if(q===4)return val(C);
  const lo=C.slice();lo[axis]!--;const w=q*0.25;
  if(known(lo)&&known(C))return val(lo)*(1-w)+val(C)*w;if(known(lo))return val(lo);return val(C);
 }
 const q=p.map(v=>v+1),b=q.map(v=>(v>>2)-1),f=q.map(v=>(v&3)*0.25);let sum=0,total=0;
 for(let k=0;k<8;k++){
  const bit=[k&1,(k>>1)&1,k>>2],w=bit.reduce((acc,bb,a)=>acc*(bb?f[a]!:1-f[a]!),1);
  if(w<=0)continue;const c=b.map((v,a)=>v+bit[a]!);if(known(c)){sum+=w*val(c);total+=w;}
 }
 return total>0?sum/total:0;
}
function restrict(layout:"cell"|"face"|"vertex",axis:number,c:readonly number[]):number{
 if(layout==="vertex")return fineValue(c.map(v=>4*v+3));
 const parts:number[][]=[];
 if(layout==="cell"){for(let i=0;i<64;i++)parts.push([4*c[0]!+(i&3),4*c[1]!+((i>>2)&3),4*c[2]!+(i>>4)]);}
 else{const u=(axis+1)%3,v=(axis+2)%3;for(let i=0;i<16;i++){const q=c.map(x=>4*x);q[axis]!+=3;q[u]!+=i&3;q[v]!+=i>>2;parts.push(q);}}
 let sum=0,total=0,plain=0;for(const q of parts){const w=weightValue(q),value=fineValue(q);sum+=w*value;total+=w;plain+=value;}
 return total>0?sum/total:plain/parts.length;
}

/** Marks `live`, builds the table, runs `entries` (8 groups each) and the release in one pass; reads the table and outWords back. */
async function runArena(device:GPUDevice,arena:UniformBrickArena,code:string,live:readonly number[],entries:readonly string[],outWords:number){
 const module=device.createShaderModule({code});
 const errors=(await module.getCompilationInfo()).messages.filter(m=>m.type==="error");
 assert.deepEqual(errors.map(m=>`${m.lineNum}: ${m.message}`),[]);
 const usage=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC;
 const table=device.createBuffer({size:4*arena.tableWords,usage}),pageBuffer=device.createBuffer({size:4*(WINDOW+BLOCKS),usage});
 const pool=device.createBuffer({size:Math.max(16,4*(arena.poolWords[0]??0)),usage}),out=device.createBuffer({size:4*outWords,usage}),marks=device.createBuffer({size:4*(1+live.length),usage});
 device.queue.writeBuffer(pageBuffer,0,pageTable());
 for(const w of arena.initialWrites(0))device.queue.writeBuffer(table,w.offset,w.data as Uint32Array<ArrayBuffer>);
 device.queue.writeBuffer(marks,0,new Uint32Array([live.length,...live]));
 const layout=device.createBindGroupLayout({entries:[0,1,2,3,4].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,buffer:{type:(binding===1||binding===4?"read-only-storage":"storage") as GPUBufferBindingType}}))});
 const group=device.createBindGroup({layout,entries:[table,pageBuffer,pool,out,marks].map((buffer,binding)=>({binding,resource:{buffer}}))});
 const pipelineLayout=device.createPipelineLayout({bindGroupLayouts:[layout]});
 const all=["mark",...UNIFORM_ARENA_TABLE_ENTRIES,...entries,UNIFORM_ARENA_RELEASE_ENTRY];
 const pipelines=await Promise.all(all.map(entryPoint=>device.createComputePipelineAsync({layout:pipelineLayout,compute:{module,entryPoint}})));
 const e=device.createCommandEncoder();const clear=arena.stepClear(0);e.clearBuffer(table,clear.offset,clear.size);
 const pass=e.beginComputePass();pass.setBindGroup(0,group);
 for(const [i,p] of pipelines.entries()){pass.setPipeline(p);pass.dispatchWorkgroups(all[i]==="mark"?1:8);}
 pass.end();device.queue.submit([e.finish()]);
 const read=async(buffer:GPUBuffer)=>{const s=device.createBuffer({size:buffer.size,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});const c=device.createCommandEncoder();c.copyBufferToBuffer(buffer,0,s,0,buffer.size);device.queue.submit([c.finish()]);await s.mapAsync(GPUMapMode.READ);const d=s.getMappedRange().slice(0);s.unmap();s.destroy();return d;};
 const result={words:new Uint32Array(await read(table)),values:new Float32Array(await read(out))};
 for(const buffer of [table,pageBuffer,pool,out,marks])buffer.destroy();
 return result;
}

test("ubaStage: the generator checks its budget, halos and rules",()=>{
 const arena=new UniformBrickArena({blocks:BLOCKS,channels:[{name:"q",levels:[0,2],layout:"cell"}],lists:[]});
 assert.throws(()=>ubaStageWGSL(arena,"k",{level:0,workgroup:64,channels:[{name:"a",layout:"cell",halo:9,core:{pool:"q"},seam:"prolong",coarse:"c"}]}),/halo/);
 assert.throws(()=>ubaStageWGSL(arena,"k",{level:2,workgroup:64,channels:[{name:"a",layout:"cell",halo:1,core:{pool:"q"},seam:"prolong",coarse:"c"}]}),/prolong/);
 assert.throws(()=>ubaStageWGSL(arena,"k",{level:0,workgroup:64,channels:[{name:"a",layout:"face",halo:1,core:{pool:"q"},seam:"prolong",coarse:"c"}]}),/axis/);
 assert.throws(()=>ubaStageWGSL(arena,"k",{level:0,workgroup:64,channels:[0,1].map(i=>({name:`a${i}`,layout:"cell" as const,halo:4,core:{pool:"q"},seam:"prolong" as const,coarse:"c"}))}),/budget/);
 assert.ok(2*(8+2*2)**3+128<=UBA_STAGE_WORDS);
 assert.throws(()=>ubaStageWGSL(arena,"k",{level:0,workgroup:64,channels:[{name:"a",layout:"cell",halo:1,core:{pool:"q"},seam:"prolong",coarse:"c"},{name:"b",layout:"cell",halo:2,core:{pool:"q"},seam:"prolong",coarse:"c",window:"a"}]}),/window/);
});

test("ubaStage: windows hold cores through the record, prolonged and restricted seams, and the outside kind",{skip:!process.env.WEBGPU_NODE_MODULE},async()=>{
 const at=(x:number,y:number,z:number)=>brickId([x,y,z]);
 // Bricks across page faces, around the AIR page, with odd (half-live) ones beside even ones.
 const live=[at(24,0,0),at(32,0,0),at(40,0,0),at(32,8,0),at(33,0,8),at(16,24,24),at(24,24,24),at(56,0,24),at(64,0,32),at(40,32,32),at(8,8,8),at(0,0,0)].filter((id,i,all)=>id>=0&&all.indexOf(id)===i);
 assert.ok(live.length<=MAX_LIVE);
 await withUniformDevice("brick stage",async device=>{
  const arena=new UniformBrickArena({blocks:BLOCKS,channels:[{name:"q",levels:[0,2],layout:"cell"}],lists:[{name:"fine",any:UNIFORM_ARENA_CLASS.fine}]});
  const {words,values}=await runArena(device,arena,SHIM+arena.tableWGSL("table","0u",TILE_MASK)+arena.poolWGSL(["pool"])+stageEntries(arena),live,["fill","stage0","stage2"],OUT2+held.length*PER2);
  const l=arena.tableLayout;
  assert.equal(words[0],0,"no fatal");
  const liveCount=words[18+2]!;assert.equal(liveCount,live.length);
  const isLive=(id:number)=>id>=0&&words[l.table+id]!==0xffffffff;
  const near=(got:number,want:number,what:string)=>assert.ok(Math.abs(got-want)<=1e-5*Math.max(1,Math.abs(want)),`${what}: ${got} against ${want}`);
  let seams=0,outsides=0,cores=0;
  // Level 0: every live slot's three windows.
  for(let e=0;e<liveCount;e++){
   const slot=words[l.live+e]!,id=words[l.brick+slot]!,o=brickOrigin(id);
   for(let k=0;k<E0_3;k++){
    const c=[k%E0-H0,Math.floor(k/E0)%E0-H0,Math.floor(k/(E0*E0))-H0],p=c.map((v,a)=>v+o[a]!),other=brickId(p),local=p.map(v=>v&7);
    const where=other<0?"outside":isLive(other)&&tileLive(other,local)?"core":"seam";
    if(where==="outside")outsides++;else if(where==="core")cores++;else seams++;
    const base=PER0*e;
    near(values[base+k]!,where==="core"?fineValue(p):where==="outside"?OUTSIDE:prolong("cell",0,p),`cell ${id} ${c}`);
    near(values[base+E0_3+k]!,where==="core"?fineValue(p)+1000:where==="outside"?OUTSIDE:prolong("face",1,p),`face ${id} ${c}`);
    near(values[base+2*E0_3+k]!,where==="core"?fineValue(p):where==="outside"?0:prolong("vertex",0,p),`vertex ${id} ${c}`);
   }
  }
  assert.ok(seams>0&&outsides>0&&cores>0,`${seams} seams, ${outsides} outside, ${cores} cores`);
  // Level 2: every held page's windows; a cell whose tile is live at h restricts.
  const pageCount=words[18+4]!;assert.equal(pageCount,held.length);
  let restricted=0;
  for(let e=0;e<pageCount;e++){
   const b=words[l.pages+e]!,o=pageOrigin(b).map(v=>8*v);
   for(let k=0;k<E2_3;k++){
    const c=[k%E2-H2,Math.floor(k/E2)%E2-H2,Math.floor(k/(E2*E2))-H2],P=c.map((v,a)=>v+o[a]!);
    const stored=blockOfPage(P.map(v=>Math.floor(v/8)))!==0,h=P.map(v=>4*v),id=brickId(h),fine=stored&&isLive(id)&&tileLive(id,h.map(v=>v&7));
    if(fine)restricted++;
    const base=OUT2+PER2*e;
    near(values[base+k]!,!stored?OUTSIDE:fine?restrict("cell",0,P):coarseValue(P),`4h cell ${b} ${c}`);
    near(values[base+E2_3+k]!,!stored?OUTSIDE:fine?restrict("face",2,P):coarseValue(P),`4h face ${b} ${c}`);
    near(values[base+2*E2_3+k]!,!stored?0:fine?restrict("vertex",0,P):coarseValue(P),`4h vertex ${b} ${c}`);
   }
  }
  assert.ok(restricted>0,"some 4h cells restrict from h");
 });
});
