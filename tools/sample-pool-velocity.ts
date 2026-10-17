/** Common-world-point QA through the production mixed MAC presentation sampler. */
import {uniformMixedPresentationWGSL,uniformMixedPresentationVelocityWGSL} from "../lib/methods/uniform/uniform-mixed-presentation.wgsl";
import {uniformDetailBaseOf} from "../lib/core/uniform-detail-abi";
export async function samplePoolVelocity(device:GPUDevice,solver:{velocityTexture:GPUTexture;negativeBoundaryVelocityBuffer:GPUBuffer},topology:GPUBufferBinding,phi:GPUTexture,coarsePhi:GPUTexture,origin:readonly number[],h:readonly number[],roi:{minimum:number[];maximum:number[]},impactXZ:readonly number[]){
 const positions:number[]=[],seen=new Set<string>();
 const add=(x:number,y:number,z:number)=>{const key=[x,y,z].map(v=>v.toFixed(7)).join(",");if(!seen.has(key)){seen.add(key);positions.push(x,y,z,1);}};
 // Globally fixed 4h cell centres in the authored liquid box, including a thin air layer.
 for(let z=roi.minimum[2]!+2*h[2]!;z<roi.maximum[2]!;z+=4*h[2]!)for(let y=roi.minimum[1]!+2*h[1]!;y<roi.maximum[1]!+4*h[1]!;y+=4*h[1]!)for(let x=roi.minimum[0]!+2*h[0]!;x<roi.maximum[0]!;x+=4*h[0]!)add(x,y,z);
 const coarseSamples=positions.length/4;
 // Fixed h centres in a 2m square around impact, spanning the upper liquid metre.
 for(let z=Math.max(roi.minimum[2]!,impactXZ[1]!-1)+.5*h[2]!;z<Math.min(roi.maximum[2]!,impactXZ[1]!+1);z+=h[2]!)for(let y=Math.max(roi.minimum[1]!,roi.maximum[1]!-1)+.5*h[1]!;y<roi.maximum[1]!+4*h[1]!;y+=h[1]!)for(let x=Math.max(roi.minimum[0]!,impactXZ[0]!-1)+.5*h[0]!;x<Math.min(roi.maximum[0]!,impactXZ[0]!+1);x+=h[0]!)add(x,y,z);
 const world=new Float32Array(positions),lattice=world.slice();for(let i=0;i<lattice.length;i+=4)for(let a=0;a<3;a++)lattice[i+a]=(world[i+a]!-origin[a]!)/h[a]!;
 const input=device.createBuffer({size:lattice.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});device.queue.writeBuffer(input,0,lattice);
 const output=device.createBuffer({size:world.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC}),read=device.createBuffer({size:world.byteLength,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
 const code=`@group(0) @binding(1) var phi:texture_3d<f32>;@group(0) @binding(2) var coarsePhi:texture_3d<f32>;
 @group(0) @binding(3) var velocity:texture_3d<f32>;@group(0) @binding(4) var velocityBase:texture_3d<f32>;
 @group(0) @binding(5) var<storage,read> negative:array<f32>;@group(0) @binding(6) var<storage,read> points:array<vec4f>;@group(0) @binding(7) var<storage,read_write> result:array<vec4f>;
 ${uniformMixedPresentationWGSL(0,"phi","coarsePhi")}
 fn umLoadMixedFace(anchor:vec3i,axis:u32)->f32{if(!umFaceLow(anchor,axis)){return udrLoadFace(velocity,velocityBase,anchor)[axis];}return negative[umLowFaceIndex(umLowFaceAbove(anchor,axis),axis)];}
 ${uniformMixedPresentationVelocityWGSL()}
 @compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id:vec3u){udrInit();if(id.x>=arrayLength(&points)){return;}result[id.x]=vec4f(umSampleVelocity(points[id.x].xyz-32.0*vec3f(umFramePage())),0);}`;
 const entries:GPUBindGroupLayoutEntry[]=Array.from({length:8},(_,binding)=>({binding,visibility:GPUShaderStage.COMPUTE,...([1,2,3,4].includes(binding)?{texture:{sampleType:"unfilterable-float" as const,viewDimension:"3d" as const}}:{buffer:{type:binding===7?"storage" as const:"read-only-storage" as const}})}));
 const layout=device.createBindGroupLayout({entries});
 try{
 const pipeline=await device.createComputePipelineAsync({layout:device.createPipelineLayout({bindGroupLayouts:[layout]}),compute:{module:device.createShaderModule({label:"Common-worldpoint authoritative velocity QA",code}),entryPoint:"main"}});
 const group=device.createBindGroup({layout,entries:[{binding:0,resource:topology},...[phi,coarsePhi,solver.velocityTexture,uniformDetailBaseOf(solver.velocityTexture)].map((texture,i)=>({binding:i+1,resource:texture.createView({dimension:"3d" as const})})),{binding:5,resource:{buffer:solver.negativeBoundaryVelocityBuffer}},{binding:6,resource:{buffer:input}},{binding:7,resource:{buffer:output}}]});
 const encoder=device.createCommandEncoder(),pass=encoder.beginComputePass();pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroups(Math.ceil(world.length/4/64));pass.end();encoder.copyBufferToBuffer(output,0,read,0,world.byteLength);device.queue.submit([encoder.finish()]);await read.mapAsync(GPUMapMode.READ);
 return {values:new Float32Array(read.getMappedRange()).slice(),positions:world,metadata:{components:4,coordinates:"world meters",sampling:"production mixed MAC umSampleVelocity, current window frame applied",coarseSamples,fineSamples:world.length/4-coarseSamples,roi,coarseSpacing_h:4,fineSpacing_h:1,impactXZ_m:Array.from(impactXZ)}};
 }finally{if(read.mapState==="mapped")read.unmap();input.destroy();output.destroy();read.destroy();}
}
