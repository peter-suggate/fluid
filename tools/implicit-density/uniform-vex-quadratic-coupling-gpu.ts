import { gpuCompilationManagerFor } from "../../lib/gpu/compilation-manager";

/** GPU outputs of one immutable UVM1 attempt. No decoded map or ROI enters. */
export interface UniformVexQuadraticInput {
  mapAndReceipt: GPUBuffer;
  nativeCenterCoverage: GPUBuffer;
}
export const UNIFORM_VEX_QUADRATIC_FAULT = Object.freeze({ map: 128, stale: 256, coverage: 512, empty: 1024, incomplete: 2048 });
export const UNIFORM_VEX_QUADRATIC_WGSL = /* wgsl */ `
@group(0) @binding(0) var<storage,read> source:array<f32>;
@group(0) @binding(1) var<storage,read_write> p:array<f32>;
@group(0) @binding(2) var<storage,read_write> work:array<u32>;
@group(0) @binding(3) var<storage,read_write> receipt:array<atomic<u32>>;
@group(0) @binding(4) var<storage,read> map:array<u32>;
@group(0) @binding(5) var<storage,read> coverage:array<u32>;
@group(0) @binding(6) var<storage,read> destination:array<f32>;
const INVALID:u32=0xffffffffu;
fn u(i:u32)->u32{return bitcast<u32>(p[i]);}
fn mf(i:u32)->f32{return bitcast<f32>(map[i]);}
fn fail(code:u32,id:u32){atomicOr(&receipt[0],code);atomicMin(&receipt[1],id);}
fn dims()->vec3u{return vec3u(u(4u),u(5u),u(6u));}
fn coord(id:u32)->vec3u{let n=dims();return vec3u(id%n.x,(id/n.x)%n.y,id/(n.x*n.y));}
fn cell(q:vec3i)->u32{let n=vec3i(dims());if(any(q<vec3i(0))||any(q>=n)){return INVALID;}
 return u32(q.x)+u(4u)*(u32(q.y)+u(5u)*u32(q.z));}
fn validSource(id:u32)->bool{return id!=INVALID&&bitcast<u32>(source[16u*id+10u])==u(20u);}
// Equal significands with a bounded exponent difference certify a REAL
// power-of-two ratio. A rounded h*offset==t test would not prove this.
fn certifiedOffset(t:f32,h:f32)->vec2f{
 if(t==0.0){return vec2f(0.0,1.0);}
 let a=bitcast<u32>(abs(t));let b=bitcast<u32>(h);
 let ea=(a>>23u)&255u;let eb=(b>>23u)&255u;let difference=i32(ea)-i32(eb);
 if(ea>0u&&ea<255u&&eb>0u&&eb<255u&&(a&0x7fffffu)==(b&0x7fffffu)&&difference>=-1&&difference<=4){
  return vec2f(sign(t)*bitcast<f32>(u32(127+difference)<<23u),1.0);
 }return vec2f(t/h,0.0);
}
@compute @workgroup_size(1) fn admitMap(){
 let count=u(7u);p[21]=bitcast<f32>(0u);p[40]=bitcast<f32>(0u);
 if(map[0]!=0x55564d31u||map[1]!=1u||map[2]!=1u||map[3]!=0u||map[8]==0u||map[8]>map[9]
  ||map[9]>map[11]||map[10]>=map[11]||map[12]!=count||map[14]!=3u||map[15]!=map[4]+1u
  ||map[6]>1u||map[7]>1u||map[33]<1u||map[33]>16u||!(mf(31u)>0.0&&mf(31u)<3.4e38)
  ||mf(32u)!=p[3]||any(vec3u(map[34],map[35],map[36])!=dims())){fail(128u,0u);return;}
 // Bounded research sequencing: initial field1 belongs to native source1;
 // each admitted map advances one native frame and one field generation.
 if(map[4]!=u(20u)||map[5]==0u){fail(256u,0u);return;}
 if(coverage[count]!=0x55564331u||coverage[count+5u]!=map[8]||coverage[count+6u]!=count||coverage[count+7u]!=map[32]){fail(512u,0u);return;}
 for(var i=0u;i<4u;i++){if(coverage[count+1u+i]!=map[4u+i]){fail(512u,0u);return;}}
 for(var axis=0u;axis<3u;axis++){
  for(var column=0u;column<3u;column++){
   let value=mf(16u+4u*axis+column);
   if(value!=select(0.0,1.0,axis==column)){fail(128u,0u);return;}
   p[8u+4u*axis+column]=value;p[28u+4u*axis+column]=value;
  }
  let translation=mf(19u+4u*axis);
  if(!(abs(translation)<3.4e38)||!(abs(translation/p[3])<=16.0)){fail(128u,0u);return;}
  p[11u+4u*axis]=translation;p[31u+4u*axis]=-translation;
  let offset=certifiedOffset(-translation,p[3]);p[44u+axis]=offset.x;
  if(offset.y==1.0){p[40]=bitcast<f32>(u(40u)|(1u<<axis));}
 }
 atomicStore(&receipt[9],map[4]);atomicStore(&receipt[10],map[5]);
}
@compute @workgroup_size(64) fn chooseCandidates(@builtin(global_invocation_id)gid:vec3u){
 let id=gid.x;if(id>=u(7u)||atomicLoad(&receipt[0])!=0u){return;}
 let q=vec3f(coord(id));let delta=vec3f(p[11],p[15],p[19])/p[3];
 var departure=delta;var error=8e-6*(vec3f(1.0)+abs(q)+abs(delta));
 for(var axis=0u;axis<3u;axis++){if((u(40u)&(1u<<axis))!=0u){departure[axis]=-p[44u+axis];error[axis]=0.0;}}
 let donorA=vec3i(floor(q+departure-error));let donorB=vec3i(ceil(q+vec3f(1.0)+departure+error))-vec3i(1);
 let span=donorB-donorA+vec3i(1);
 if(any(span<=vec3i(0))||any(span>vec3i(i32(u(23u))))||u32(span.x*span.y*span.z)>u(23u)){fail(4u,id);return;}
 for(var z=donorA.z;z<=donorB.z;z++){for(var y=donorA.y;y<=donorB.y;y++){for(var x=donorA.x;x<=donorB.x;x++){
  if(!validSource(cell(vec3i(x,y,z)))){atomicAdd(&receipt[13],1u);return;}
 }}}
 // Entire translated cube sweep, then the support of every native-center
 // trilinear basis function meeting it. Zero-weight endpoints remain included.
 let lo=q+min(vec3f(0.0),departure)-error;let hi=q+vec3f(1.0)+max(vec3f(0.0),departure)+error;
 let a=vec3i(floor(lo-vec3f(0.5)));let b=vec3i(ceil(hi-vec3f(0.5)));let extent=b-a+vec3i(1);
 if(any(extent<=vec3i(0))||any(extent>vec3i(32))||extent.x*extent.y*extent.z>4096){fail(4u,id);return;}
 for(var z=a.z;z<=b.z;z++){for(var y=a.y;y<=b.y;y++){for(var x=a.x;x<=b.x;x++){
  let at=cell(vec3i(x,y,z));
  if(at==INVALID){atomicAdd(&receipt[12],1u);return;}
  let owner=coverage[at];if(owner==0u){atomicAdd(&receipt[12],1u);return;}
  if(owner>map[11]){fail(512u,id);return;}
 }}}
 let rank=atomicAdd(&receipt[8],1u);work[rank]=id;
 atomicAdd(&receipt[11],u32(extent.x*extent.y*extent.z));
}
@compute @workgroup_size(1) fn finishCandidates(){
 if(atomicLoad(&receipt[0])!=0u){return;}let count=atomicLoad(&receipt[8]);
 if(count==0u||count>u(7u)){fail(1024u,0u);return;}p[21]=bitcast<f32>(count);
}
@compute @workgroup_size(64) fn checkAmounts(@builtin(global_invocation_id)gid:vec3u){
 if(gid.x>=u(21u)||atomicLoad(&receipt[0])!=0u){return;}let id=work[gid.x];let at=16u*id;
 let amount=destination[at+12u];let error=destination[at+13u];let volume=p[3]*p[3]*p[3];
 if(bitcast<u32>(destination[at+10u])!=u(20u)+1u||bitcast<u32>(destination[at+15u])!=u(20u)+1u
  ||!(amount>=-p[25]*volume&&amount<=(1.0+p[25])*volume)||!(error>=0.0&&error<=p[25])){fail(2048u,id);return;}
 atomicAdd(&receipt[14],1u);
}
@compute @workgroup_size(1) fn sealTransaction(){
 if(atomicLoad(&receipt[0])!=0u){return;}
 if(atomicLoad(&receipt[3])!=u(21u)||atomicLoad(&receipt[14])!=u(21u)||u(21u)==0u){fail(2048u,0u);return;}
 atomicStore(&receipt[15],u(20u)+1u);
}
`;

export interface UniformVexQuadraticPipelines { layout: GPUBindGroupLayout; pipelines: ReadonlyMap<string, GPUComputePipeline> }
const cache = new WeakMap<GPUDevice, Promise<UniformVexQuadraticPipelines>>();
export function uniformVexQuadraticPipelines(device: GPUDevice): Promise<UniformVexQuadraticPipelines> {
  let result = cache.get(device);
  if (!result) {
    result = (async () => {
      const compiler = gpuCompilationManagerFor(device);
      const module = compiler.createShaderModule({ label: "Research native VEX to current quadric admission", code: UNIFORM_VEX_QUADRATIC_WGSL });
      const layout = device.createBindGroupLayout({ entries: Array.from({ length: 7 }, (_, binding) => ({ binding,
        visibility: GPUShaderStage.COMPUTE, buffer: { type: [0, 4, 5, 6].includes(binding) ? "read-only-storage" as const : "storage" as const } })) });
      const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [layout] }), pipelines = new Map<string, GPUComputePipeline>();
      try {
        for (const entryPoint of ["admitMap", "chooseCandidates", "finishCandidates", "checkAmounts", "sealTransaction"]) {
          pipelines.set(entryPoint, await compiler.compileComputePipeline({ label: `Research VEX quadric ${entryPoint}`,
            layout: pipelineLayout, compute: { module, entryPoint } }, { priority: "critical" }));
        }
      } catch (error) {
        const info = await module.getCompilationInfo();
        throw new Error(`VEX quadric admission WGSL: ${info.messages.map(message => `${message.lineNum}:${message.linePos} ${message.message}`).join("\n")}`, { cause: error });
      }
      return { layout, pipelines };
    })();
    cache.set(device, result);
  }
  return result;
}
