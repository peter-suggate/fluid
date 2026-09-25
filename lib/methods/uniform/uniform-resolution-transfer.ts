import { gpuCompilationManagerFor } from "../../core/gpu-compilation-manager";

export interface UniformResolutionFields {
  readonly volume: GPUTexture;
  readonly velocity: GPUTexture;
  readonly phi: GPUTexture;
  readonly negativeFaces: GPUBuffer;
}

/** Aligned h -> 4h restriction for open box interiors. V is in cell-volume
 * units, phi in metres, and velocity in metres/second. The 16 fine MAC faces
 * on a coarse face have equal open area in this first supported geometry. */
const source = /* wgsl */ `
@group(0) @binding(0) var fineV:texture_3d<f32>;
@group(0) @binding(1) var fineU:texture_3d<f32>;
@group(0) @binding(2) var finePhi:texture_3d<f32>;
@group(0) @binding(3) var<storage,read> fineBoundary:array<f32>;
@group(0) @binding(4) var coarseV:texture_storage_3d<r32float,write>;
@group(0) @binding(5) var coarseU:texture_storage_3d<rgba32float,write>;
@group(0) @binding(6) var coarsePhi:texture_storage_3d<r32float,write>;
@group(0) @binding(7) var<storage,read_write> coarseBoundary:array<f32>;
fn faceIndex(p:vec3u,axis:u32,d:vec3u)->u32{
 if(axis==0u){return p.y+d.y*p.z;}
 if(axis==1u){return d.y*d.z+p.x+d.x*p.z;}
 return d.y*d.z+d.x*d.z+p.x+d.x*p.y;
}
@compute @workgroup_size(4,4,4)
fn restrictCells(@builtin(global_invocation_id) id:vec3u){
 let d=textureDimensions(coarseV);if(any(id>=d)){return;}
 let base=vec3i(id*4u);var v=0.0;
 for(var z=0;z<4;z++){for(var y=0;y<4;y++){for(var x=0;x<4;x++){
  v+=textureLoad(fineV,base+vec3i(x,y,z),0).x;
 }}}
 textureStore(coarseV,vec3i(id),vec4f(v/64.0));
 var u=vec3f(0);
 for(var axis=0u;axis<3u;axis++){
  let a=(axis+1u)%3u;let b=(axis+2u)%3u;var sum=0.0;var boundary=0.0;
  for(var j=0;j<4;j++){for(var i=0;i<4;i++){
   var p=base;p[axis]+=3;p[a]+=i;p[b]+=j;
   sum+=textureLoad(fineU,p,0)[axis];
   if(id[axis]==0u){boundary+=fineBoundary[faceIndex(vec3u(p),axis,d*4u)];}
  }}
  u[axis]=sum/16.0;
  if(id[axis]==0u){coarseBoundary[faceIndex(id,axis,d)]=boundary/16.0;}
 }
 textureStore(coarseU,vec3i(id),vec4f(u,0));
}
@compute @workgroup_size(4,4,4)
fn restrictVertices(@builtin(global_invocation_id) id:vec3u){
 if(any(id>=textureDimensions(coarsePhi))){return;}
 // Coincident vertices retain signed distance in physical metres. A subsequent
 // volume-constrained reconstruction corrects the changed coarse quadrature.
 textureStore(coarsePhi,vec3i(id),textureLoad(finePhi,vec3i(id*4u),0));
}
`;

/** Conservative piecewise-constant V, trilinear physical phi, and MAC
 * interpolation only along each face normal. Every fine face on a coarse face
 * has its parent's velocity, so restriction recovers all coarse face fluxes.
 * The four interior normal intervals share the parent cell's divergence. */
const prolongSource = /* wgsl */ `
@group(0) @binding(0) var coarseV:texture_3d<f32>;
@group(0) @binding(1) var coarseU:texture_3d<f32>;
@group(0) @binding(2) var coarsePhi:texture_3d<f32>;
@group(0) @binding(3) var<storage,read> coarseBoundary:array<f32>;
@group(0) @binding(4) var fineV:texture_storage_3d<r32float,write>;
@group(0) @binding(5) var fineU:texture_storage_3d<rgba32float,write>;
@group(0) @binding(6) var finePhi:texture_storage_3d<r32float,write>;
@group(0) @binding(7) var<storage,read_write> fineBoundary:array<f32>;
fn faceIndex(p:vec3u,axis:u32,d:vec3u)->u32{
 if(axis==0u){return p.y+d.y*p.z;}
 if(axis==1u){return d.y*d.z+p.x+d.x*p.z;}
 return d.y*d.z+d.x*d.z+p.x+d.x*p.y;
}
@compute @workgroup_size(4,4,4)
fn prolongCells(@builtin(global_invocation_id) id:vec3u){
 let d=textureDimensions(fineV);if(any(id>=d)){return;}
 let parent=id/4u;let cd=textureDimensions(coarseV);
 textureStore(fineV,vec3i(id),textureLoad(coarseV,vec3i(parent),0));
 var u=vec3f(0);
 for(var axis=0u;axis<3u;axis++){
  let right=textureLoad(coarseU,vec3i(parent),0)[axis];
  var left=0.0;
  if(parent[axis]==0u){left=coarseBoundary[faceIndex(parent,axis,cd)];}
  else{var p=vec3i(parent);p[axis]-=1;left=textureLoad(coarseU,p,0)[axis];}
  u[axis]=mix(left,right,f32(id[axis]%4u+1u)/4.0);
  if(id[axis]==0u){fineBoundary[faceIndex(id,axis,d)]=left;}
 }
 textureStore(fineU,vec3i(id),vec4f(u,0));
}
@compute @workgroup_size(4,4,4)
fn prolongVertices(@builtin(global_invocation_id) id:vec3u){
 if(any(id>=textureDimensions(finePhi))){return;}
 let p=vec3i(id/4u);let f=vec3f(id%4u)/4.0;
 let hi=vec3i(textureDimensions(coarsePhi))-vec3i(1);var phi=0.0;
 for(var z=0;z<2;z++){for(var y=0;y<2;y++){for(var x=0;x<2;x++){
  let offset=vec3i(x,y,z);let w=select(vec3f(1)-f,f,offset != vec3i(0));
  phi+=textureLoad(coarsePhi,min(p+offset,hi),0).x*w.x*w.y*w.z;
 }}}
 textureStore(finePhi,vec3i(id),vec4f(phi));
}
`;

/** Pipelines and bindings are prepared before the first fine step. encode does
 * not allocate, compile, map, or submit. Source/destination are persistent and
 * disjoint; neither aliases the shared stage arena. */
export class UniformResolutionRestriction {
  private constructor(private readonly cells: GPUComputePipeline,
    private readonly vertices: GPUComputePipeline, private readonly group: GPUBindGroup,
    private readonly dims: readonly [number,number,number]) {}
  static async create(device:GPUDevice,fine:UniformResolutionFields,coarse:UniformResolutionFields,signal?:AbortSignal, prolong=false) {
    for(const fields of [fine,coarse]) {
      const d=[fields.volume.width,fields.volume.height,fields.volume.depthOrArrayLayers];
      for(const [field,pad,format] of [[fields.volume,0,"r32float"],[fields.velocity,0,"rgba32float"],[fields.phi,1,"r32float"]] as const)
        if(field.dimension!=="3d" || field.format!==format || ![field.width,field.height,field.depthOrArrayLayers].every((n,a)=>n===d[a]!+pad))
          throw new Error("Restriction requires native cell/vertex fields with matching extents and formats");
      if(fields.negativeFaces.size<4*(d[0]!*d[1]!+d[0]!*d[2]!+d[1]!*d[2]!))
        throw new Error("Restriction requires all three negative MAC face planes");
    }
    const coarseDims=[coarse.volume.width,coarse.volume.height,coarse.volume.depthOrArrayLayers] as const;
    const input=prolong?coarse:fine, output=prolong?fine:coarse;
    const dims=[output.volume.width,output.volume.height,output.volume.depthOrArrayLayers] as const;
    const fineDims=[fine.volume.width,fine.volume.height,fine.volume.depthOrArrayLayers];
    if(!coarseDims.every((d,a)=>fineDims[a]===4*d))throw new Error("Restriction requires aligned native h/4h textures");
    const layout=device.createBindGroupLayout({entries:[
      ...[0,1,2].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"unfilterable-float" as const,viewDimension:"3d" as const}})),
      {binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},
      ...(["r32float","rgba32float","r32float"] as const).map((format,i)=>({binding:4+i,visibility:GPUShaderStage.COMPUTE,storageTexture:{access:"write-only" as const,format,viewDimension:"3d" as const}})),
      {binding:7,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},
    ]});
    const compiler=gpuCompilationManagerFor(device);
    const module=compiler.createShaderModule({label:prolong?"Uniform 4h to h prolongation":"Uniform h to 4h restriction",code:prolong?prolongSource:source});
    const pipelineLayout=device.createPipelineLayout({bindGroupLayouts:[layout]});
    const cells=await compiler.compileComputePipeline({label:"Uniform restrict cells",layout:pipelineLayout,compute:{module,entryPoint:prolong?"prolongCells":"restrictCells"}},{priority:"visible",signal});
    const vertices=await compiler.compileComputePipeline({label:"Uniform restrict vertices",layout:pipelineLayout,compute:{module,entryPoint:prolong?"prolongVertices":"restrictVertices"}},{priority:"visible",signal});
    const group=device.createBindGroup({layout,entries:[
      ...[input.volume,input.velocity,input.phi].map((t,binding)=>({binding,resource:t.createView()})),
      {binding:3,resource:{buffer:input.negativeFaces}},
      ...[output.volume,output.velocity,output.phi].map((t,i)=>({binding:4+i,resource:t.createView()})),
      {binding:7,resource:{buffer:output.negativeFaces}},
    ]});
    return new UniformResolutionRestriction(cells,vertices,group,dims);
  }
  encode(encoder:GPUCommandEncoder):void {
    for(const [pipeline,pad] of [[this.cells,0],[this.vertices,1]] as const){
      const pass=encoder.beginComputePass({label:pad?"Resolution transfer vertices":"Resolution transfer volume and MAC flux"});
      pass.setPipeline(pipeline);pass.setBindGroup(0,this.group);
      pass.dispatchWorkgroups(Math.ceil((this.dims[0]+pad)/4),Math.ceil((this.dims[1]+pad)/4),Math.ceil((this.dims[2]+pad)/4));pass.end();
    }
  }
}

export class UniformResolutionProlongation {
  static create(device:GPUDevice,coarse:UniformResolutionFields,fine:UniformResolutionFields,signal?:AbortSignal) {
    return UniformResolutionRestriction.create(device,fine,coarse,signal,true);
  }
}
