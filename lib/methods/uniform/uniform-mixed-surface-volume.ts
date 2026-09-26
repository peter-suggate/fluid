import type {UniformMixedOwnership} from "./uniform-mixed-ownership";
import {uniformMixedTopologyWGSL} from "./uniform-mixed-topology.wgsl";
import {uniformMixedVertexSamplingWGSL} from "./uniform-mixed-vertex-sampling.wgsl";
import {uniformSurfaceFillWGSL} from "./uniform-surface-volume.wgsl";

/** Native four-cell band and two 17-sample global normal-shift refinements,
 * reduced with physical owner volumes. Every field is borrowed. */
export class UniformMixedSurfaceVolume {
 readonly allocatedBytes=0;
 readonly scratchBytes:number;
 private readonly cells:number;
 private readonly vertices:number;
 private readonly groups:number;
 private readonly chunks:number;
 private readonly resources:GPUBindGroupLayout;
 private readonly pipelines=new Map<string,GPUComputePipeline>();
 constructor(private readonly device:GPUDevice,readonly ownership:UniformMixedOwnership){
  this.cells=ownership.layout.lattice.dimensions.reduce((n,d)=>n*d,1);
  this.vertices=ownership.layout.lattice.dimensions.reduce((n,d)=>n*(d+1),1);
  this.groups=Math.ceil(this.cells/64);this.chunks=Math.ceil(this.groups/64);
  this.scratchBytes=4*(this.cells*2+this.vertices+20*(this.groups+this.chunks)+8);
  this.resources=device.createBindGroupLayout({entries:[
   ...[0,1].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"unfilterable-float" as const,viewDimension:"3d" as const}})),
   {binding:2,visibility:GPUShaderStage.COMPUTE,storageTexture:{access:"write-only",format:"r32float",viewDimension:"3d"}},
   {binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},
  ]});
 }
 bind(phi:GPUTexture,volume:GPUTexture,output:GPUTexture,scratch:GPUBufferBinding):GPUBindGroup{
  const d=this.ownership.layout.lattice.dimensions;
  for(const [i,t] of [phi,volume,output].entries())if(t.format!=="r32float"||[t.width,t.height,t.depthOrArrayLayers].some((n,a)=>n!==d[a]!+(i===1?0:1)))throw new Error("Mixed surface constraint requires native vertex and cell fields");
  if(phi===output||(scratch.size??scratch.buffer.size-(scratch.offset??0))<this.scratchBytes)throw new Error("Mixed surface constraint needs disjoint output and sufficient scratch");
  return this.device.createBindGroup({layout:this.resources,entries:[...[phi,volume,output].map((t,binding)=>({binding,resource:t.createView()})),{binding:3,resource:{...scratch,size:this.scratchBytes}}]});
 }
 async initialize():Promise<void>{
  const N=this.cells,V=this.vertices,P=2*N+V,R=P+20*this.groups,S=R+20*this.chunks;
  const sourceScale=uniformMixedVertexSamplingWGSL.replace(/\bum(Vertex\w*|SampleVertex|LoadVertex)\b/g,name=>name.replace("um","umScale"));
  const module=this.device.createShaderModule({code:uniformMixedTopologyWGSL(this.ownership.layout,0)+/* wgsl */`
@group(1) @binding(0) var phi:texture_3d<f32>;
@group(1) @binding(1) var volume:texture_3d<f32>;
@group(1) @binding(2) var output:texture_storage_3d<r32float,write>;
@group(1) @binding(3) var<storage,read_write> scratch:array<f32>;
override parity:u32=0u;
const UM_H=vec3f(${this.ownership.layout.lattice.cellSize_m.join(",")});
fn umLoadVertex(p:vec3u)->f32{return textureLoad(phi,vec3i(p),0).x;}
${uniformMixedVertexSamplingWGSL}
fn umVertexIndex(p:vec3u)->u32{let d=UM_D+vec3u(1);return p.x+d.x*(p.y+d.y*p.z);}
fn umScaleLoadVertex(p:vec3u)->f32{return scratch[${2*N}u+umVertexIndex(p)];}
${sourceScale}
${uniformSurfaceFillWGSL}
fn umLiveCells()->u32{return umCounts.x*64u+umCounts.y*8u+umCounts.z;}
fn umShiftLimit()->f32{return min(UM_H.x,min(UM_H.y,UM_H.z))*f32(select(select(4u,2u,umCounts.y>0u),1u,umCounts.x>0u));}
@compute @workgroup_size(1) fn begin(){scratch[${S}u]=0.0;scratch[${S+1}u]=umShiftLimit();}
@compute @workgroup_size(64) fn seed(@builtin(global_invocation_id) gid:vec3u){
 let o=umAllOwner(gid);if(o.width==0u){return;}var low=1e30;var high=-1e30;
 for(var k=0u;k<umCounts.w;k++){let v=umVertexValue(umOrigin(o)+umCorner(k,2u)*o.width);low=min(low,v);high=max(high,v);}
 scratch[o.index]=select(0.0,5.0,low<=0.0&&high>=0.0);
}
@compute @workgroup_size(64) fn dilate(@builtin(global_invocation_id) gid:vec3u){
 let o=umAllOwner(gid);if(o.width==0u){return;}let input=parity*${N}u;let out=(parity^1u)*${N}u;var band=scratch[input+o.index];
 for(var axis=0u;axis<3u;axis++){for(var side=0u;side<2u;side++){
  let sign=select(-1,1,side==1u);let first=umFace(o,axis,sign,0u);
  for(var part=0u;part<first.count;part++){let other=umFace(o,axis,sign,part).neighbor;if(other.width!=0u){band=max(band,scratch[input+other.index]-1.0);}}
 }}scratch[out+o.index]=max(0.0,band);
}
@compute @workgroup_size(64) fn metric(@builtin(global_invocation_id) gid:vec3u){
 let o=umAllOwner(gid);if(o.width==0u){return;}
 for(var k=0u;k<umCounts.w;k++){
  let p=umOrigin(o)+umCorner(k,2u)*o.width;if(umVertexAuthority(p).index!=o.index){continue;}
  var band=0.0;
  for(var j=0u;j<umCounts.w;j++){let c=umOwnerAt(vec3i(p)+vec3i(umCorner(j,2u))-vec3i(1));if(c.width!=0u){band=max(band,scratch[c.index]);}}
  var gradient=vec3f(0);
  if(band>0.0){for(var axis=0u;axis<3u;axis++){
   if(p[axis]==0u||p[axis]==UM_D[axis]){continue;}
   var lo=vec3f(p);var hi=lo;lo[axis]=max(0.0,lo[axis]-f32(o.width));hi[axis]=min(f32(UM_D[axis]),hi[axis]+f32(o.width));
   gradient[axis]=(umSampleVertex(hi)-umSampleVertex(lo))/((hi[axis]-lo[axis])*UM_H[axis]);
  }}scratch[${2*N}u+umVertexIndex(p)]=band*0.2*max(0.1,length(gradient));
 }
}
var<workgroup> sums:array<vec4f,320>;
fn sumGroup(l:u32){workgroupBarrier();for(var stride=32u;stride>0u;stride/=2u){if(l<stride){for(var k=0u;k<5u;k++){sums[5u*l+k]+=sums[5u*(l+stride)+k];}}workgroupBarrier();}}
fn storeSum(at:u32,k:u32,value:vec4f){for(var c=0u;c<4u;c++){scratch[at+4u*k+c]=value[c];}}
fn loadSum(at:u32,k:u32)->vec4f{return vec4f(scratch[at+4u*k],scratch[at+4u*k+1u],scratch[at+4u*k+2u],scratch[at+4u*k+3u]);}
@compute @workgroup_size(64) fn measure(@builtin(global_invocation_id) gid:vec3u,@builtin(local_invocation_index) l:u32,@builtin(workgroup_id) group:vec3u){
 let o=umAllOwner(gid);var result:array<vec4f,5>;
 if(o.width!=0u){
  let origin=umOrigin(o);let mass=f32(o.width*o.width*o.width);result[4].y=textureLoad(volume,vec3i(origin),0).x*mass;
  var raw:array<f32,8>;var scale:array<f32,8>;var low=1e30;var high=-1e30;
  let centre=scratch[${S}u];let radius=scratch[${S+1}u];
  for(var k=0u;k<umCounts.w;k++){let vertex=origin+umCorner(k,2u)*o.width;raw[k]=umVertexValue(vertex);scale[k]=umScaleVertexValue(vertex);
   low=min(low,raw[k]-(centre+radius)*scale[k]);high=max(high,raw[k]-(centre-radius)*scale[k]);}
  for(var sample=0u;sample<17u;sample++){
   var fraction=0.0;if(high<0.0){fraction=1.0;}else if(low<0.0){
    let shift=centre+(f32(sample)/8.0-1.0)*radius;var values:array<f32,8>;var negative=0u;
    for(var k=0u;k<umCounts.w;k++){values[k]=raw[k]-shift*scale[k];negative+=select(0u,1u,values[k]<0.0);}
    if(negative==8u){fraction=1.0;}else if(negative!=0u){fraction=fill(values);}
   }result[sample/4u][sample%4u]=fraction*mass;
  }
 }
 for(var k=0u;k<5u;k++){sums[l*5u+k]=result[k];}sumGroup(l);
 if(l==0u){let at=${P}u+20u*(group.x+umDispatchX*group.y);for(var k=0u;k<5u;k++){storeSum(at,k,sums[k]);}}
}
@compute @workgroup_size(64) fn reduce(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) l:u32){
 let i=group.x*64u+l;let count=(umLiveCells()+63u)/64u;
 for(var k=0u;k<5u;k++){sums[l*5u+k]=vec4f(0);if(i<count){sums[l*5u+k]=loadSum(${P}u+20u*i,k);}}sumGroup(l);
 if(l==0u){for(var k=0u;k<5u;k++){storeSum(${R}u+20u*group.x,k,sums[k]);}}
}
@compute @workgroup_size(64) fn solve(@builtin(local_invocation_index) l:u32){
 let count=(umLiveCells()+4095u)/4096u;
 for(var k=0u;k<5u;k++){var sum=vec4f(0);for(var i=l;i<count;i+=64u){sum+=loadSum(${R}u+20u*i,k);}sums[l*5u+k]=sum;}sumGroup(l);
 if(l==0u){
  let desired=sums[4].y;let centre=scratch[${S}u];let radius=scratch[${S+1}u];var shift=centre;
  if(sums[4].x-sums[0].x>1e-6&&abs(sums[2].x-desired)>max(1e-5,1e-7*abs(desired))){
   shift=centre+select(-radius,radius,desired>sums[4].x);
   for(var k=0u;k<16u;k++){let a=sums[k/4u][k%4u];let b=sums[(k+1u)/4u][(k+1u)%4u];
    if(desired>=a&&desired<=b&&b>a){shift=centre+(f32(k)+clamp((desired-a)/(b-a),0.0,1.0)-8.0)*radius/8.0;break;}}
  }scratch[${S}u]=clamp(shift,-umShiftLimit(),umShiftLimit());scratch[${S+1}u]=radius/8.0;
 }
}
@compute @workgroup_size(64) fn apply(@builtin(global_invocation_id) gid:vec3u){
 let o=umAllOwner(gid);if(o.width==0u){return;}
 for(var k=0u;k<umCounts.w;k++){let p=umOrigin(o)+umCorner(k,2u)*o.width;
  if(umVertexAuthority(p).index==o.index){textureStore(output,vec3i(p),vec4f(umLoadVertex(p)-scratch[${S}u]*umScaleLoadVertex(p)));}}
}
`});
  const errors=(await module.getCompilationInfo()).messages.filter(m=>m.type==="error");if(errors.length)throw new Error(errors.map(m=>`${m.lineNum}: ${m.message}`).join("\n"));
  const layout=this.device.createPipelineLayout({bindGroupLayouts:[this.ownership.bindLayout,this.resources]});
  for(const entryPoint of ["begin","seed","metric","measure","reduce","solve","apply","dilate"]){
   for(const parity of entryPoint==="dilate"?[0,1]:[0])this.pipelines.set(entryPoint+parity,await this.device.createComputePipelineAsync({layout,compute:{module,entryPoint,constants:{parity,umDispatchX:this.ownership.dispatchX}}}));
  }
 }
 encode(encoder:GPUCommandEncoder,group:GPUBindGroup):void{
  if(this.pipelines.size!==9)throw new Error("Mixed surface constraint is not initialized");
  const pass=encoder.beginComputePass({label:"Uniform mixed global surface volume"});pass.setBindGroup(0,this.ownership.bindGroup);pass.setBindGroup(1,group);
  const run=(entry:string,parity=0)=>{const pipeline=this.pipelines.get(entry+parity)!;pass.setPipeline(pipeline);
   if(entry==="begin"||entry==="solve")pass.dispatchWorkgroups(1);
   else if(entry==="reduce")pass.dispatchWorkgroups(Math.ceil(this.ownership.layout.cellCount/4096));
   else this.ownership.dispatchAll(pass,pipeline);};
  run("begin");run("seed");for(let i=0;i<4;i++)run("dilate",i%2);run("metric");
  for(let i=0;i<2;i++){run("measure");run("reduce");run("solve");}run("apply");pass.end();
 }
}
