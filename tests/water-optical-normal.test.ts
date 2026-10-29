import assert from "node:assert/strict";
import test from "node:test";
import {surfaceExtractionShader} from "../lib/core/webgpu-water-pipeline";

function normalFunction(){
 const start=surfaceExtractionShader.indexOf("fn uniformPhiNormal(");
 assert.ok(start>=0);
 return surfaceExtractionShader.slice(start,surfaceExtractionShader.indexOf("\n}",start)+2)
  .replaceAll("vec3i(textureDimensions(denseVertexPhi))","vec3i(65)")
  .replaceAll("textureLoad(denseVertexPhi,q,0).x","fixturePhi(vec3f(q))");
}
function oldGaussian(fn:string){
 return fn.replace("fn uniformPhiNormal(","fn oldGaussianNormal(")
  .replaceAll(/let distance=abs\(delta\);let shoulder=[\s\S]*?let derivativeWeight=vec3f\(d\.x\*w\.y\*w\.z,w\.x\*d\.y\*w\.z,w\.x\*w\.y\*d\.z\);/g,
   "let weight=exp(-0.5*dot(delta,delta)/(.85*.85));")
  .replaceAll(/if\(ox==-1&&oy==-1&&oz==-1\)\{phiReference=phi;\}[\s\S]*?phiDerivativeSum\+=relativePhi\*derivativeWeight;/g,
   "weightSum+=weight;phiSum+=weight*phi;derivativeWeightSum+=weight*delta;phiDerivativeSum+=weight*phi*delta;");
}
function unit(v:number[]){const n=Math.hypot(...v);return v.map(x=>x/n);}
function angle(a:number[],b:number[]){a=unit(a);b=unit(b);const cross=[a[1]!*b[2]!-a[2]!*b[1]!,a[2]!*b[0]!-a[0]!*b[2]!,a[0]!*b[1]!-a[1]!*b[0]!];return Math.atan2(Math.hypot(...cross),a.reduce((s,x,i)=>s+x*b[i]!,0))*180/Math.PI;}

test("actual water normal WGSL reproduces affine and quadratic gradients across phases and sampler routes",{skip:!process.env.WEBGPU_NODE_MODULE},async()=>{
 const {withUniformDevice}=await import("./helpers/uniform-geometric");
 const cases:{input:number[],expected:number[],kind:number,radius:number}[]=[];
 const add=(x:number[],kind:number,mode:number,radius=0)=>{
  const h=kind<2?[.25,.5,.75]:[1,1,1];
  const expected=kind===0?[.3,1,.1]:kind===1?x.map((v,i)=>v*h[i]!-[3.17,6.23,5.41][i]!):kind===2?x.map((v,i)=>v-[24.17,23.23,25.41][i]!):[0,1,0];
  cases.push({input:[...x,kind,...h,mode,kind===4?Number.NaN:radius,0,0,0],expected,kind,radius});
 };
 for(let mode=0;mode<6;mode++){
  for(const x of [-.49999,-.37,0,.17,.49999])for(const y of [-.49999,-.13,0,.39,.49999])for(const z of [-.49999,-.37,0,.17,.49999]){
   add([27+x,27+y,27+z],0,mode);add([27+x,27+y,27+z],1,mode);
  }
  for(const radius of [2,4,8,16])for(let k=0;k<128;k++){
   const z=1-2*(k+.5)/128,t=k*Math.PI*(3-Math.sqrt(5)),r=Math.sqrt(1-z*z);
   add([24.17+radius*r*Math.cos(t),23.23+radius*r*Math.sin(t),25.41+radius*z],2,mode,radius);
  }
  add([27.49,27.13,27.37],3,mode);add([27.49,27.13,27.37],4,mode);
  add([.25,27,27],5,mode);
 }
 await withUniformDevice("water optical normal invariants",async device=>{
  const fn=normalFunction();assert.match(fn,/derivativeWeight/);const baseline=oldGaussian(fn);assert.doesNotMatch(baseline,/derivativeWeight=/);
  const code=`
struct View{gridInfo:vec4f,container:vec4f}
struct Bounds{lo:vec3f,hi:vec3f}
struct Case{positionKind:vec4f,cellMode:vec4f,extra:vec4f}
@group(0) @binding(0) var<storage,read> inputs:array<Case>;
@group(0) @binding(1) var<storage,read_write> output:array<vec4f>;
var<private> u:View;var<private> kind:u32;var<private> radius:f32;var<private> testCell:vec3f;
var<private> normalCube:u32;var<private> normalCubeBase:vec3i;
var<private> uniformNormalGather:bool;var<private> uniformNormalLoops:bool;var<private> uniformCubeMemo:bool;var<private> mixed:bool;
fn fixturePhi(q:vec3f)->f32{
 if(kind==0u){return dot(q*testCell,vec3f(.3,1,.1))+3.7;}
 if(kind==1u){let p=q*testCell-vec3f(3.17,6.23,5.41);return dot(p,p)-4.0;}
 if(kind==2u){return length(q-vec3f(24.17,23.23,25.41))-radius;}
 if(kind==4u){return radius;}
 return 7.3;
}
fn umPresentationEnabled()->bool{return mixed;}
fn umPointBounds(p:vec3f,width:f32)->Bounds{return Bounds(vec3f(0),vec3f(64));}
fn umVertexValue(q:vec3u)->f32{return fixturePhi(vec3f(q));}
fn umLoadFineVertex(q:vec3u)->f32{return fixturePhi(vec3f(q));}
fn uniformNormalBase(q:vec3i)->f32{return fixturePhi(vec3f(q));}
fn uniformMemoVertex(q:vec3i)->f32{return fixturePhi(vec3f(q));}
fn uniformNormalWide(q:vec3i)->bool{return true;}
${fn}
${baseline}
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id:vec3u){
 let index=id.x;if(index>=arrayLength(&inputs)){return;}
 let c=inputs[index];kind=u32(c.positionKind.w);radius=c.extra.x;testCell=c.cellMode.xyz;
 let mode=u32(c.cellMode.w);mixed=mode!=0u;uniformNormalGather=mode>0u;
 uniformNormalLoops=mode<5u;uniformCubeMemo=mode==3u;normalCube=select(mode,3u,mode==5u);if(mode==4u){normalCube=0u;}
 normalCubeBase=vec3i(round(c.positionKind.xyz));u.gridInfo=vec4f(64,64,64,0);u.container=vec4f(testCell*64.0,0);
 let lattice=c.positionKind.xyz+vec3f(.5);let fallback=vec3f(0,1,0);
 output[2u*index]=vec4f(uniformPhiNormal(lattice,fallback),1);
 output[2u*index+1u]=vec4f(oldGaussianNormal(lattice,fallback),1);
}`;
  const module=device.createShaderModule({code});assert.deepEqual((await module.getCompilationInfo()).messages.filter(m=>m.type==="error").map(m=>`${m.lineNum}:${m.linePos} ${m.message}`),[]);
  const pipeline=await device.createComputePipelineAsync({layout:"auto",compute:{module,entryPoint:"main"}});
  const input=device.createBuffer({size:cases.length*48,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
  const output=device.createBuffer({size:cases.length*32,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
  const read=device.createBuffer({size:output.size,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});
  device.queue.writeBuffer(input,0,new Float32Array(cases.flatMap(c=>c.input)));
  try{
   const encoder=device.createCommandEncoder(),pass=encoder.beginComputePass();pass.setPipeline(pipeline);pass.setBindGroup(0,device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:input}},{binding:1,resource:{buffer:output}}]}));pass.dispatchWorkgroups(Math.ceil(cases.length/64));pass.end();encoder.copyBufferToBuffer(output,0,read,0,output.size);device.queue.submit([encoder.finish()]);await read.mapAsync(GPUMapMode.READ);
   const values=new Float32Array(read.getMappedRange()).slice();read.unmap();const maxima={plane:0,oldPlane:0,quadratic:0,sphere:{} as Record<string,number>,oldSphere:{} as Record<string,number>};
   cases.forEach((c,i)=>{
    const n=Array.from(values.slice(i*8,i*8+3)),old=Array.from(values.slice(i*8+4,i*8+7));assert.ok(n.every(Number.isFinite));
    if(c.kind===0){maxima.plane=Math.max(maxima.plane,angle(n,c.expected));maxima.oldPlane=Math.max(maxima.oldPlane,angle(old,c.expected));}
    else if(c.kind===1)maxima.quadratic=Math.max(maxima.quadratic,angle(n,c.expected));
    else if(c.kind===2){maxima.sphere[c.radius]=Math.max(maxima.sphere[c.radius]??0,angle(n,c.expected));maxima.oldSphere[c.radius]=Math.max(maxima.oldSphere[c.radius]??0,angle(old,c.expected));}
    else assert.deepEqual(n,[0,1,0],"constant, invalid or boundary samples preserve fallback");
   });
   assert.ok(maxima.oldPlane>1.5,`old Gaussian must expose affine defect: ${JSON.stringify(maxima)}`);
   assert.ok(maxima.plane<.002,JSON.stringify(maxima));assert.ok(maxima.quadratic<.002,JSON.stringify(maxima));
   for(const [radius,limit] of [[2,1.6],[4,.4],[8,.11],[16,.03]]){
    assert.ok(maxima.sphere[radius!]!<limit!,JSON.stringify(maxima));assert.ok(maxima.sphere[radius!]!*5<maxima.oldSphere[radius!]!,JSON.stringify(maxima));
   }
   console.log(JSON.stringify({cases:cases.length,maxima}));
  }finally{input.destroy();output.destroy();read.destroy();}
 });
});
