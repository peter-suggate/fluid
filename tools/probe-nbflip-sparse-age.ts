/** Isolate the production sparse-particle lifetime query on fixed cell populations.
 * This uses production paged addressing and nbSparseAge, with no simulation/velocity work.
 * node --import tsx tools/probe-nbflip-sparse-age.ts
 */
import assert from 'node:assert/strict';
import {resolve} from 'node:path';
import {pathToFileURL} from 'node:url';
import {narrowBandSparseWGSL} from '../lib/methods/uniform/uniform-narrow-band-sparse.wgsl';
import {uniformLatticeAddress,uniformLatticeAddressWGSL} from '../lib/methods/uniform/uniform-lattice-address';
import {createProcessRetainedDawnGPU,paceDawnEventPolling} from '../lib/harness/node-dawn-provider';

const original=narrowBandSparseWGSL.replace(/ \/\/ The cell across the closest face[\s\S]*?(?= \/\/ Near cells first:)/,'').replace('if(all(offset==near)){continue;}','');
assert.notEqual(original,narrowBandSparseWGSL,'nearest-face shortcut is present');
paceDawnEventPolling();const dawn=await import(pathToFileURL(resolve('node_modules/webgpu/index.js')).href);Object.assign(globalThis,dawn.globals);
const gpu=createProcessRetainedDawnGPU(dawn,['backend=metal']),adapter=await gpu.requestAdapter();assert.ok(adapter);
const device=await adapter.requestDevice({requiredFeatures:['timestamp-query']});
const errors:string[]=[];device.addEventListener('uncapturederror',e=>{e.preventDefault();errors.push(e.error.message);});
const address=uniformLatticeAddress([64,32,32],true,{blocks:6,permute:3});
const blocks=address.blocks,cells=address.cellKeys;
const buffer=(size:number,usage:number)=>device.createBuffer({size:Math.max(4,size),usage});
const table=buffer(address.seed.table.byteLength,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST);device.queue.writeBuffer(table,0,address.seed.table);
const code=(body:string)=>`
@group(0) @binding(0) var<storage,read> table:array<u32>;
@group(0) @binding(1) var<storage,read_write> bins:array<atomic<u32>>;
@group(0) @binding(2) var<storage,read> links:array<u32>;
@group(0) @binding(3) var<storage,read> points:array<vec4f>;
@group(0) @binding(4) var<storage,read_write> output:array<f32>;
struct Params{hDt:vec4f,activity:vec4f};const params=Params(vec4f(1,1,1,1.0/60.0),vec4f(0,2,0,0));
fn umPageWord(p:vec3i)->u32{let q=vec3u(p)&vec3u(63);return table[q.x|(q.y<<6u)|(q.z<<12u)];}
fn umBlockWord(b:u32)->u32{return 0u;}
${uniformLatticeAddressWGSL({constant:true,paged:{blocks:`${blocks}u`}})}
fn nbCellLocal(c:vec3i)->u32{let l=vec3u(c)&vec3u(3u);return l.x|(l.y<<2u)|(l.z<<4u);}
fn nbRun(a:vec3i,b:vec3i)->vec2u{let tile=umTileKey(a>>vec3u(2u));if(!umKeyStored(tile)){return vec2u(0);}let last=64u*tile+nbCellLocal(b);return vec2u(links[64u*tile+nbCellLocal(a)],links[last]+atomicLoad(&bins[last]));}
fn nbCellRun(c:vec3i)->vec2u{return nbRun(c,c);}
fn nbPosition(i:u32)->vec3f{return points[i].xyz;}
${body}
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) gid:vec3u){let i=gid.x;if(i<arrayLength(&points)){output[i]=nbSparseAge(i,points[i].xyz,points[i].w);}}
`;
const layout=device.createBindGroupLayout({entries:[0,1,2,3,4].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,buffer:{type:binding===1||binding===4?'storage':'read-only-storage'}}))});
const pipelineLayout=device.createPipelineLayout({bindGroupLayouts:[layout]});
const pipelines=await Promise.all([original,narrowBandSparseWGSL].map(async body=>{const module=device.createShaderModule({code:code(body)});const info=await module.getCompilationInfo();assert.deepEqual(info.messages.filter(m=>m.type==='error'),[]);return device.createComputePipelineAsync({layout:pipelineLayout,compute:{module,entryPoint:'main'}});}));
const cellKey=(x:number,y:number,z:number)=>{const b=address.seed.table[(x>>5)|((y>>5)<<6)|((z>>5)<<12)]!;return 32768*b+64*((x>>2&7)|((y>>2&7)<<3)|((z>>2&7)<<6))+(x&3)+4*(y&3)+16*(z&3);};
const fixtures=['dense','compressed','sheet','seam','sparse'] as const;
const repeat=11,launches=32;
for(const fixture of fixtures){
 const binsWords=new Uint32Array(cells),starts=new Uint32Array(cells);const entries:{key:number,x:number,y:number,z:number,n:number}[]=[];
 for(let z=8;z<24;z++)for(let y=8;y<24;y++)for(let x=8;x<56;x++){
  const h=((Math.imul(x+81,73856093)^Math.imul(y+31,19349663)^Math.imul(z+79,83492791))>>>0);let n=8;
  if(fixture==='compressed')n=12;
  if(fixture==='sheet'&&y!==15)n=0;
  if(fixture==='seam'&&(x<30||x>33))n=0;
  if(fixture==='sparse')n=h%7===0?1+h%8:0;
  if(n)entries.push({key:cellKey(x,y,z),x,y,z,n});
 }
 entries.sort((a,b)=>a.key-b.key);let count=0;for(const c of entries){binsWords[c.key]=c.n;count+=c.n;}
 let first=0;for(let k=0;k<cells;k++){starts[k]=first;first+=binsWords[k]!;}
 const positions=new Float32Array(4*count);let i=0;
 for(const {x,y,z,n} of entries)for(let j=0;j<n;j++,i++){const f=fixture==='sparse'?[0.001,0.999,0.5,0.17,0.83,0.25,0.75,0.35]:[0.25,0.75];positions.set([x+f[j%f.length]!,y+f[(j>>1)%f.length]!,z+f[(j>>2)%f.length]!,i%3===0?1.99:0.4],4*i);}
 const makeInput=(v:Uint32Array<ArrayBuffer>|Float32Array<ArrayBuffer>)=>{const b=buffer(v.byteLength,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST);device.queue.writeBuffer(b,0,v);return b;};
 const bins=makeInput(binsWords),links=makeInput(starts),points=makeInput(positions),output=buffer(4*count,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC);
 const group=device.createBindGroup({layout,entries:[table,bins,links,points,output].map((buffer,binding)=>({binding,resource:{buffer}}))});
 const results=[0,1].map(()=>buffer(4*count,GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ));
 const queries=device.createQuerySet({type:'timestamp',count:4*repeat}),resolved=buffer(32*repeat,GPUBufferUsage.QUERY_RESOLVE|GPUBufferUsage.COPY_SRC),times=buffer(32*repeat,GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ);
 let e=device.createCommandEncoder();
 for(let arm=0;arm<2;arm++){const p=e.beginComputePass();p.setPipeline(pipelines[arm]!);p.setBindGroup(0,group);p.dispatchWorkgroups(Math.ceil(count/64));p.end();e.copyBufferToBuffer(output,0,results[arm]!,0,4*count);}
 for(let r=0;r<repeat;r++)for(let k=0;k<2;k++){const arm=(r+k)%2,index=4*r+2*k;const p=e.beginComputePass({timestampWrites:{querySet:queries,beginningOfPassWriteIndex:index,endOfPassWriteIndex:index+1}});p.setPipeline(pipelines[arm]!);p.setBindGroup(0,group);for(let n=0;n<launches;n++)p.dispatchWorkgroups(Math.ceil(count/64));p.end();}
 e.resolveQuerySet(queries,0,4*repeat,resolved,0);e.copyBufferToBuffer(resolved,0,times,0,32*repeat);device.queue.submit([e.finish()]);await Promise.all([...results.map(b=>b.mapAsync(GPUMapMode.READ)),times.mapAsync(GPUMapMode.READ)]);
 const a=new Uint32Array(results[0]!.getMappedRange()),b=new Uint32Array(results[1]!.getMappedRange()),t=new BigUint64Array(times.getMappedRange());let mismatches=0;for(let i=0;i<count;i++)if(a[i]!==b[i])mismatches++;
 const ms:number[][]=[[],[]];for(let r=0;r<repeat;r++)for(let k=0;k<2;k++){const index=4*r+2*k;ms[(r+k)%2]!.push(Number(t[index+1]!-t[index]!)/1e6/launches);}
 const med=ms.map(v=>v.sort((a,b)=>a-b)[Math.floor(v.length/2)]);console.log(JSON.stringify({fixture,count,mismatches,baseline_ms:med[0],candidate_ms:med[1],timings:ms,errors}));assert.equal(mismatches,0);assert.deepEqual(errors,[]);
 for(const b of [...results,bins,links,points,output,resolved,times])b.destroy();queries.destroy();
}
table.destroy();device.destroy();
