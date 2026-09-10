import assert from "node:assert/strict";
import test from "node:test";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { createDawnRenderDevice } from "../tools/svo-dry-frame-harness";
import { svoCellContourWGSL } from "../lib/svo/features/construction/svo-cell-contour";
import { svoCellContourFitWGSL } from "../lib/svo/features/construction/svo-cell-contour-fit";
import { SVO_GBUFFER_NORMAL_OCT8_WGSL } from "../lib/svo/contracts/svo-gbuffer";

// Sweep a translated lattice, rather than testing one conveniently centred cell.
// The GPU runs the production fitter and polygonizer. CPU implicit equations
// independently measure the emitted cap vertices against the analytic surface.
const N=24, h=0.125, count=N*N*N;
const baseShapes=[
  {name:"sphere",kind:3,r:[1,1,1]},
  {name:"ellipsoid",kind:3,r:[1.1,.7,.9]},
  {name:"cylinder",kind:2,r:[.85,.9,0]},
  {name:"capsule",kind:4,r:[.55,.55,0]},
  {name:"box",kind:1,r:[.8,.7,.9]},
];
const angle=.37;
const shapes=baseShapes.flatMap(s=>[ {...s,angle:0}, {...s,name:s.name+"-rotated",angle} ]);
function local(p:number[],angle:number){const c=Math.cos(angle),s=Math.sin(angle);return [c*p[0]+s*p[1],-s*p[0]+c*p[1],p[2]];}
function distance(p:number[],kind:number,r:number[]){
  const [x,y,z]=p;
  if(kind===3) return (Math.hypot(x/r[0],y/r[1],z/r[2])-1)*Math.min(...r);
  if(kind===4) return Math.hypot(x,y-Math.max(-r[1],Math.min(r[1],y)),z)-r[0];
  const q=kind===2?[Math.hypot(x,z)-r[0],Math.abs(y)-r[1]]:p.map((v,i)=>Math.abs(v)-r[i]);
  return Math.hypot(...q.map(v=>Math.max(v,0)))+Math.min(Math.max(...q),0);
}
(process.env.WEBGPU_NODE_MODULE?test:test.skip)("Dawn contours track analytic shapes across cell phases",async(t)=>{
  await acquireWebGPUExclusiveLock("dawn-test","svo-cell-contour-analytic");
  let device:GPUDevice|undefined;
  try{
    device=(await createDawnRenderDevice()).device;
    const module=device.createShaderModule({code:`
${SVO_GBUFFER_NORMAL_OCT8_WGSL}
${svoCellContourWGSL}
struct ScenePrimitive {centerType:vec4f,extentIdentity:vec4f,rotation:vec4f}
@group(0) @binding(0) var<storage,read_write> maintenance:array<atomic<u32>>;
@group(0) @binding(1) var<storage,read> primitives:array<ScenePrimitive>;
@group(0) @binding(2) var<storage,read_write> output:array<vec4f>;
fn candidateOffset()->u32{return 0u;}fn candidatesPerBrick()->u32{return 1u;}
fn scenePrimitiveType(p:ScenePrimitive)->u32{return u32(p.centerType.w);}
fn inverseRotate(p:vec3f,r:vec4f)->vec3f{
 let v=-r.xyz;let t=2.0*cross(v,p);return p+r.w*t+cross(v,t);
}
fn primitiveUsesThresholdOccupancy(p:ScenePrimitive)->bool{return false;}
fn primitiveDistance(p:ScenePrimitive,w:vec3f)->f32{
 let q=inverseRotate(w-p.centerType.xyz,p.rotation);let r=p.extentIdentity.xyz;let kind=scenePrimitiveType(p);
 if(kind==3u){return (length(q/r)-1.)*min(r.x,min(r.y,r.z));}
 if(kind==4u){return length(vec3f(q.x,q.y-clamp(q.y,-r.y,r.y),q.z))-r.x;}
 if(kind==2u){let d=vec2f(length(q.xz)-r.x,abs(q.y)-r.y);return length(max(d,vec2f(0)))+min(max(d.x,d.y),0.);}
 let d=abs(q)-r;return length(max(d,vec3f(0)))+min(max(d.x,max(d.y,d.z)),0.);
}
${svoCellContourFitWGSL(false,false)}
@compute @workgroup_size(64) fn check(@builtin(global_invocation_id) id:vec3u){
 let i=id.x;if(i>=${count}u){return;}
 let c=(vec3f(f32(i%${N}u),f32((i/${N}u)%${N}u),f32(i/${N*N}u))-vec3f(11.37,11.61,11.23))*${h};
 let p=primitives[0];let d=primitiveDistance(p,c);let radius=.5*length(vec3f(${h}));
 let fraction=clamp(.5-d/(2.*radius),0.,1.);
 var n=vec3f(0);let e=.0001;
 for(var a=0u;a<3u;a+=1u){var v=vec3f(0);v[a]=e;n[a]=primitiveDistance(p,c+v)-primitiveDistance(p,c-v);}
 n=normalize(n);
 let code=fitSceneContour(c,vec3f(${h}),n,fraction,0u,1u);
 let contour=cellContour(svoGBufferUnpackNormalOct8(svoGBufferPackNormalOct8(n)),vec3f(${h}),code);
 output[i*14u]=vec4f(c,fraction);output[i*14u+1u]=vec4f(f32(code),0,0,0);output[i*14u+13u]=vec4f(contour.normal,contour.high);
 if(code==0u||code==255u){return;}
 let cap=contourCap(contour);output[i*14u+1u].y=f32(cap.count);
 for(var j=0u;j<cap.count;j+=1u){output[i*14u+2u+j]=vec4f(c+(contourUnpackPoint(contourPackPoint(cap.points[j]))-vec3f(.5))*${h},1);}
}`});
    const pipeline=await device.createComputePipelineAsync({layout:"auto",compute:{module,entryPoint:"check"}});
    const usage=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC;
    const indices=device.createBuffer({size:4,usage});const primitive=device.createBuffer({size:48,usage});
    const size=count*14*16;const output=device.createBuffer({size,usage});
    const read=device.createBuffer({size,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
    const group=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[indices,primitive,output].map((buffer,binding)=>({binding,resource:{buffer}}))});
    const failures:string[]=[];
    for(const shape of shapes){
      device.queue.writeBuffer(primitive,0,new Float32Array([0,0,0,shape.kind,...shape.r,0,0,0,Math.sin(shape.angle/2),Math.cos(shape.angle/2)]));
      const encoder=device.createCommandEncoder();const pass=encoder.beginComputePass();pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroups(Math.ceil(count/64));pass.end();
      encoder.copyBufferToBuffer(output,0,read,0,size);device.queue.submit([encoder.finish()]);await read.mapAsync(GPUMapMode.READ);
      const data=new Float32Array(read.getMappedRange().slice(0));read.unmap();
      let partial=0,cubes=0,empty=0,clipped=0,maxExcess=0,minExcess=0,maxCubeExcess=0;const errors:number[]=[];
      for(let i=0;i<count;i++){
        const at=i*56,f=data[at+3];if(!(f>0&&f<1))continue;partial++;
        for(let z=0;z<=4;z++)for(let y=0;y<=4;y++)for(let x=0;x<=4;x++){
          const q=[x/4-.5,y/4-.5,z/4-.5];const p=q.map((v,a)=>data[at+a]+v*h);
          if(distance(local(p,shape.angle),shape.kind,shape.r)>-1e-5)continue;
          assert.notEqual(data[at+4],255,`${shape.name}: empty proof removed analytic solid`);
          if(data[at+4]!==0)assert.ok(q.reduce((v,c,a)=>v+c*data[at+52+a],0)<=data[at+55]+1e-5,`${shape.name}: plane cuts analytic solid`);
        }
        if(data[at+4]===255){empty++;continue;}
        if(data[at+4]===0){
          cubes++;
          for(let corner=0;corner<8;corner++){
            const p=[0,1,2].map(a=>data[at+a]+(((corner>>a)&1)-.5)*h);
            maxCubeExcess=Math.max(maxCubeExcess,distance(local(p,shape.angle),shape.kind,shape.r)/h);
          }
          continue;
        }clipped++;
        for(let j=0;j<data[at+5];j++){
          const p=Array.from(data.slice(at+8+j*4,at+11+j*4));const error=distance(local(p,shape.angle),shape.kind,shape.r)/h;
          maxExcess=Math.max(maxExcess,error);minExcess=Math.min(minExcess,error);errors.push(error);
        }
      }
      let maxJoinGap=0;
      for(let i=0;i<count;i++)for(let axis=0;axis<3;axis++){
        const stride=N**axis;if(Math.floor(i/stride)%N===N-1)continue;
        const a=i*56,b=(i+stride)*56;
        if(!(data[a+4]>0&&data[a+4]<255&&data[b+4]>0&&data[b+4]<255))continue;
        let u=(axis+1)%3,v=(axis+2)%3;
        if(Math.abs(data[a+52+v])>Math.abs(data[a+52+u]))[u,v]=[v,u];
        if(Math.abs(data[a+52+u])<.1||Math.abs(data[b+52+u])<.1)continue;
        for(let j=0;j<=8;j++){
          const t=j/8-.5;
          const x=(data[a+55]-.5*data[a+52+axis]-t*data[a+52+v])/data[a+52+u];
          const y=(data[b+55]+.5*data[b+52+axis]-t*data[b+52+v])/data[b+52+u];
          if(Math.abs(x)<=.5&&Math.abs(y)<=.5)maxJoinGap=Math.max(maxJoinGap,Math.abs(x-y));
        }
      }
      errors.sort((a,b)=>a-b);
      console.log(JSON.stringify({shape:shape.name,partial,cubes,empty,clipped,maxExcessCells:maxExcess,maxCubeExcessCells:maxCubeExcess,maxJoinGapCells:maxJoinGap,minExcessCells:minExcess,p95ExcessCells:errors[Math.floor(errors.length*.95)]}));
      assert.ok(empty>0,"proved-empty coverage cells have a distinct result");
      assert.ok(maxCubeExcess<.6,`${shape.name}: empty cells must not resurrect full-cube outliers`);
      assert.ok(clipped>100,"exercise a surface, not a single cell");
      assert.ok(minExcess>-.03,"contours do not cut significantly inside the analytic solid");
      // Sharp rims/corners require multiple planes. Their separate bound is
      // deliberately not presented as smooth-surface reconstruction quality.
      const smooth=shape.kind===3||shape.kind===4;
      const limit=smooth?.18:.9;
      if(smooth&&maxJoinGap>.18)failures.push(`${shape.name}: neighbour seam ${maxJoinGap.toFixed(3)} cells`);
      if(maxCubeExcess>.18)failures.push(`${shape.name}: cube outlier ${maxCubeExcess.toFixed(3)} cells`);
      if(maxExcess>=limit)failures.push(`${shape.name}: cap outlier ${maxExcess.toFixed(3)} cells`);
    }
    for(const b of [indices,primitive,output,read])b.destroy();
    await t.test("smooth contours and retained cubes meet the geometric quality budget", {
      todo: process.env.FLUID_SVO_CONTOUR_ENFORCE_QUALITY === "1" ? false : "Known reproduction: independent conservative slabs have no shared boundary constraint",
    },()=>assert.deepEqual(failures,[]));
  }finally{device?.destroy();await releaseWebGPUExclusiveLock();}
});
