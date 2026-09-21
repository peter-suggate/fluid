import { UniformPageGeneration, uniformPageFieldAccessWGSL } from "./uniform-page-generation";

/** Fixed-resolution closest-interface reconstruction. Offsets are relative to
 * each vertex, so signed pages millions of cells apart never become f32 world
 * positions. All passes consume the same accepted page generation.
 *
 * This is a numerical replacement under validation, not yet a production phi
 * update. It preserves signs; closest edge seeds approximate curved surfaces.
 */
export class UniformPageRedistance {
  readonly result: GPUBuffer;
  /** 0 = phase-only air, 1 = metric, 2 = phase-only liquid, 3 = unresolved near field. */
  readonly support: GPUBuffer;
  private readonly scratch: readonly [GPUBuffer,GPUBuffer];
  private readonly args: GPUBuffer;
  private readonly params: GPUBuffer;
  private readonly groups: readonly [GPUBindGroup,GPUBindGroup];
  private readonly steps: readonly number[];
  private constructor(private readonly device:GPUDevice, readonly pool:UniformPageGeneration,
    private readonly pipelines:readonly GPUComputePipeline[],layout:GPUBindGroupLayout,
    readonly spacing:readonly [number,number,number],readonly bandCells:number) {
    const count=pool.options.capacity*pool.options.edge**3;
    const make=(label:string,size:number)=>device.createBuffer({label,size,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});
    this.result=make("Uniform reconstructed metric phi",count*4);
    this.support=make("Uniform phi metric/phase support",count*4);
    this.scratch=[make("Uniform closest interface A",count*16),make("Uniform closest interface B",count*16)];
    this.args=device.createBuffer({label:"Uniform redistance indirect",size:12,usage:GPUBufferUsage.INDIRECT|GPUBufferUsage.COPY_DST});
    const steps:number[]=[];
    // Include two final unit sweeps to repair common jump-flood misses. This is
    // an approximate closest-seed transform; tests must bound its error.
    for(let step=2**Math.floor(Math.log2(bandCells));step>=1;step/=2)steps.push(step);
    steps.push(1,1);this.steps=steps;
    this.params=device.createBuffer({label:"Uniform redistance immutable passes",size:256*(steps.length+1),usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
    for(let i=0;i<=steps.length;i++)device.queue.writeBuffer(this.params,i*256,new Float32Array([...spacing,bandCells*Math.max(...spacing),steps[i]??1,0,0,0]));
    this.groups=[0,1].map(bank=>device.createBindGroup({layout,entries:[pool.accepted,pool.fields,this.scratch[bank]!,this.scratch[1-bank]!,this.result,this.support]
      .map((buffer,binding)=>({binding,resource:{buffer}})).concat([{binding:6,resource:{buffer:this.params,offset:0,size:32}}] as GPUBufferBindGroupEntry[])})) as unknown as readonly [GPUBindGroup,GPUBindGroup];
  }
  static async create(device:GPUDevice,pool:UniformPageGeneration,phiField:number,spacing:readonly [number,number,number],bandCells=16){
    if(!Number.isInteger(phiField)||phiField<0||phiField>=pool.options.initialCell.length
      ||!spacing.every(h=>Number.isFinite(h)&&h>0)||!Number.isInteger(bandCells)||bandCells<2||bandCells>32)
      throw new RangeError("Invalid page redistance geometry");
    const bytes=pool.options.capacity*pool.options.edge**3*16;
    if(bytes>device.limits.maxStorageBufferBindingSize)throw new RangeError("Closest-interface pool exceeds device binding budget");
    const layout=device.createBindGroupLayout({entries:[0,1,2,3,4,5,6].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,
      buffer:{type:binding===6?"uniform" as const:binding<3?"read-only-storage" as const:"storage" as const,...(binding===6?{hasDynamicOffset:true}: {})}}))});
    const module=device.createShaderModule({label:"Uniform page closest interface",code:shader(pool,phiField)});
    const pipelineLayout=device.createPipelineLayout({bindGroupLayouts:[layout]});
    const pipelines=await Promise.all(["seed","propagate","resolve"].map(entryPoint=>device.createComputePipelineAsync({layout:pipelineLayout,compute:{module,entryPoint}})));
    return new UniformPageRedistance(device,pool,pipelines,layout,spacing,bandCells);
  }
  encode(encoder:GPUCommandEncoder):void{
    encoder.copyBufferToBuffer(this.pool.accepted,32,this.args,0,12);
    const run=(pipeline:number,bank:number,offset:number)=>{const pass=encoder.beginComputePass({label:["Seed interface crossings","Propagate closest interface","Resolve metric phi"][pipeline]});
      pass.setPipeline(this.pipelines[pipeline]!);pass.setBindGroup(0,this.groups[bank]!,[offset]);pass.dispatchWorkgroupsIndirect(this.args,0);pass.end();};
    run(0,0,0);let bank=1;
    for(let i=0;i<this.steps.length;i++){run(1,bank,i*256);bank=1-bank;}
    run(2,bank,this.steps.length*256);
  }
  get allocatedBytes():number{return this.result.size+this.support.size+this.scratch[0].size+this.scratch[1].size+this.args.size+this.params.size;}
  destroy():void{for(const b of [this.result,this.support,...this.scratch,this.args,this.params])b.destroy();}
}
// Typed alias keeps the binding constructor explicit without exposing texture bindings.
type GPUBufferBindGroupEntry={binding:number;resource:GPUBufferBinding};

function shader(pool:UniformPageGeneration,phiField:number):string{
 const edge=pool.options.edge,stride=pool.options.initialCell.length;
 return /* wgsl */ `
struct Parameters{hBand:vec4f,step:vec4f}
@group(0) @binding(0) var<storage,read> accepted:array<u32>;
@group(0) @binding(1) var<storage,read> fields:array<u32>;
@group(0) @binding(2) var<storage,read> source:array<vec4f>;
@group(0) @binding(3) var<storage,read_write> destination:array<vec4f>;
@group(0) @binding(4) var<storage,read_write> result:array<f32>;
@group(0) @binding(5) var<storage,read_write> support:array<u32>;
@group(0) @binding(6) var<uniform> params:Parameters;
${uniformPageFieldAccessWGSL(pool.options)}
fn localPoint(index:u32)->vec3i{return vec3i(i32(index%${edge}u),i32((index/${edge}u)%${edge}u),i32(index/${edge*edge}u));}
fn coordinate(slot:u32)->vec3i{let at=16u+16u*slot;return bitcast<vec3i>(vec3u(accepted[at],accepted[at+1u],accepted[at+2u]));}
fn phi(index:u32)->f32{return bitcast<f32>(fields[index*${stride}u+${phiField}u]);}
fn address(q:vec3i,p:vec3i)->u32{let at=pageFieldAddress(q,p,${phiField}u);if(at==PAGE_FIELD_MISSING){return at;}return at/${stride}u;}
fn before(a:vec3f,b:vec3f)->bool{return a.x<b.x||(a.x==b.x&&(a.y<b.y||(a.y==b.y&&a.z<b.z)));}
fn closestSegment(a:vec3f,b:vec3f)->vec3f{
 let d=b-a;return a+d*clamp(-dot(a,d)/max(dot(d,d),1e-30),0.0,1.0);
}
fn closestTriangle(a:vec3f,b:vec3f,c:vec3f)->vec3f{
 let ab=b-a;let ac=c-a;let normal=cross(ab,ac);let norm=dot(normal,normal);
 if(norm>1e-24){
  let projection=normal*(dot(a,normal)/norm);let delta=projection-a;
  let aa=dot(ab,ab);let bb=dot(ac,ac);let crossTerm=dot(ab,ac);
  let u=(bb*dot(delta,ab)-crossTerm*dot(delta,ac))/norm;
  let v=(aa*dot(delta,ac)-crossTerm*dot(delta,ab))/norm;
  if(u>=0.0&&v>=0.0&&u+v<=1.0){return projection;}
 }
 var best=closestSegment(a,b);let bc=closestSegment(b,c);let ca=closestSegment(c,a);
 if(dot(bc,bc)<dot(best,best)){best=bc;}if(dot(ca,ca)<dot(best,best)){best=ca;}return best;
}
const TETS=array<vec4u,6>(vec4u(0,1,3,7),vec4u(0,3,2,7),vec4u(0,2,6,7),vec4u(0,6,4,7),vec4u(0,4,5,7),vec4u(0,5,1,7));
fn corner(i:u32)->vec3i{return vec3i(i32(i&1u),i32((i>>1u)&1u),i32((i>>2u)&1u));}
@compute @workgroup_size(64) fn seed(@builtin(global_invocation_id)id:vec3u){
 if(id.y>=accepted[1]||id.x>=${edge**3}u){return;}
 let slot=accepted[${pool.layout.activeBase}u+id.y];let index=slot*${edge**3}u+id.x;
 let p=localPoint(id.x);let q=coordinate(slot);let own=phi(index);
 var best=vec3f(0);var distance=3.402823e38;var valid=false;
 if(own==0.0){distance=0.0;valid=true;}
 // Affine metric planes are exact fixed points, including oblique planes.
 var gradient=vec3f(0);var plane=true;
 for(var a=0u;a<3u;a++){
  var lo=p;var hi=p;lo[a]--;hi[a]++;let l=address(q,lo);let r=address(q,hi);
  if(l==PAGE_FIELD_MISSING||r==PAGE_FIELD_MISSING){plane=false;continue;}
  gradient[a]=(phi(r)-phi(l))/(2.0*params.hBand[a]);
 }
 if(abs(length(gradient)-1.0)>0.0002){plane=false;}
 if(plane){for(var z=-1;z<=1;z++){for(var y=-1;y<=1;y++){for(var x=-1;x<=1;x++){
  let delta=vec3i(x,y,z);let n=address(q,p+delta);
  if(n==PAGE_FIELD_MISSING){plane=false;continue;}
  let expected=own+dot(gradient,vec3f(delta)*params.hBand.xyz);
  if(abs(phi(n)-expected)>0.00001*max(params.hBand.w,abs(own))){plane=false;}
 }}}}
 if(plane&&abs(own)<=params.hBand.w){best=-own*gradient/params.hBand.xyz;distance=abs(own);valid=true;}
 for(var axis=0u;axis<3u;axis++){for(var sign=-1;sign<=1;sign+=2){
  var n=p;n[axis]+=sign;let other=address(q,n);if(other==PAGE_FIELD_MISSING){continue;}
  let b=phi(other);if((own<0.0)==(b<0.0)&&own!=0.0&&b!=0.0){continue;}
  let denominator=abs(own)+abs(b);if(denominator<=1e-30){continue;}
  var offset=vec3f(0);offset[axis]=f32(sign)*abs(own)/denominator;
  let d=length(offset*params.hBand.xyz);
  if(d<distance||(d==distance&&before(offset,best))){best=offset;distance=d;valid=true;}
 }}

 if(!plane){
  for(var octant=0u;octant<8u;octant++){
   let base=-corner(octant);var values:array<f32,8>;var positions:array<vec3f,8>;
   var lo=3.402823e38;var hi=-3.402823e38;var complete=true;
   for(var k=0u;k<8u;k++){
    let delta=base+corner(k);let n=address(q,p+delta);if(n==PAGE_FIELD_MISSING){complete=false;continue;}
    values[k]=phi(n);positions[k]=vec3f(delta)*params.hBand.xyz;lo=min(lo,values[k]);hi=max(hi,values[k]);
   }
   if(!complete||lo>0.0||hi<0.0){continue;}
   for(var t=0u;t<6u;t++){
    let indices=TETS[t];var points:array<vec3f,4>;var count=0u;
    for(var a=0u;a<4u;a++){for(var b=a+1u;b<4u;b++){
     let ia=indices[a];let ib=indices[b];let va=values[ia];let vb=values[ib];
     if((va<0.0)==(vb<0.0)){continue;}
     let fraction=abs(va)/(abs(va)+abs(vb));points[count]=mix(positions[ia],positions[ib],fraction);count++;
    }}
    for(var a=0u;a<count;a++){for(var b=a+1u;b<count;b++){for(var c=b+1u;c<count;c++){
     let point=closestTriangle(points[a],points[b],points[c]);let d=length(point);let offset=point/params.hBand.xyz;
     if(d<distance||(d==distance&&before(offset,best))){best=offset;distance=d;valid=true;}
    }}}
   }
  }
 }
 destination[index]=vec4f(best,select(0.0,1.0,valid&&distance<=params.hBand.w));
}
@compute @workgroup_size(64) fn propagate(@builtin(global_invocation_id)id:vec3u){
 if(id.y>=accepted[1]||id.x>=${edge**3}u){return;}
 let slot=accepted[${pool.layout.activeBase}u+id.y];let index=slot*${edge**3}u+id.x;
 let p=localPoint(id.x);let q=coordinate(slot);var best=source[index];
 var distance=select(3.402823e38,length(best.xyz*params.hBand.xyz),best.w>0.5);
 let jump=i32(params.step.x);
 for(var z=-1;z<=1;z++){for(var y=-1;y<=1;y++){for(var x=-1;x<=1;x++){
  let delta=vec3i(x,y,z)*jump;let n=address(q,p+delta);if(n==PAGE_FIELD_MISSING){continue;}
  let point=source[n];if(point.w<0.5){continue;}
  let offset=point.xyz+vec3f(delta);let d=length(offset*params.hBand.xyz);
  if(d<distance||(d==distance&&before(offset,best.xyz))){best=vec4f(offset,1.0);distance=d;}
 }}}
 if(distance>params.hBand.w){best.w=0.0;}destination[index]=best;
}
@compute @workgroup_size(64) fn resolve(@builtin(global_invocation_id)id:vec3u){
 if(id.y>=accepted[1]||id.x>=${edge**3}u){return;}
 let slot=accepted[${pool.layout.activeBase}u+id.y];let index=slot*${edge**3}u+id.x;
 let own=phi(index);let best=source[index];let sign=select(1.0,-1.0,own<0.0);
 let metric=best.w>0.5;result[index]=sign*select(params.hBand.w,length(best.xyz*params.hBand.xyz),metric);
 support[index]=select(select(0u,2u,own<0.0),1u,metric);
 // A sub-grid source can carry V before any lattice edge straddles zero.
 // Preserve its near-field samples and explicitly mark them unresolved;
 // replacing them with ambient air would erase the source's geometric seed.
 if(!metric&&abs(own)<params.hBand.w){result[index]=own;support[index]=3u;}
}
`;
}
