/** Static embedded solids for fine mixed owners: the native packed voxel mask
 * (in the host's active scratch, at params.dropExtent.z) and the terrain
 * heightfield, read through the native parameter block. Every helper is a
 * transcription of its native namesake in webgpu-uniform-reference.wgsl.ts;
 * body terms are absent because rigid bodies are rejected by the host.
 * A frame compiles this library only for a scene with cut cells; otherwise
 * stages get inert stubs and keep their solid-free code, bit-identical.
 * Coarse (2h/4h) owners never reach these helpers: promotion keeps them one
 * full tile away from any cut cell. */
export interface UniformMixedSolidResources {params:GPUBuffer;scratch:GPUBuffer;terrain:GPUTexture}

export class UniformMixedSolid {
 readonly bindLayout:GPUBindGroupLayout;
 readonly bindGroup:GPUBindGroup;
 constructor(device:GPUDevice,resources:UniformMixedSolidResources){
  if(resources.params.size<272||resources.terrain.format!=="r32float")throw new Error("Mixed solids require the native parameter block and terrain heightfield");
  this.bindLayout=device.createBindGroupLayout({label:"Uniform mixed static solids",entries:[
   {binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}},
   {binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},
   {binding:2,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"unfilterable-float",viewDimension:"2d"}},
  ]});
  this.bindGroup=device.createBindGroup({layout:this.bindLayout,entries:[
   {binding:0,resource:{buffer:resources.params,size:272}},{binding:1,resource:{buffer:resources.scratch}},
   {binding:2,resource:resources.terrain.createView()},
  ]});
 }
}

/** Requires UM_D. `group` undefined emits inert stubs with the same ABI. */
export function uniformMixedSolidWGSL(group?:number):string{
 if(group===undefined)return /* wgsl */ `
fn umSolidEnabled()->bool{return false;}
fn umSolidValid(p:vec3i)->bool{return all(p>=vec3i(0))&&all(p<vec3i(UM_D));}
fn umSolidVoxel(p:vec3i)->bool{return false;}
fn umCellOpen(p:vec3i)->f32{return select(0.0,1.0,umSolidValid(p));}
fn umCellInsideSolid(p:vec3i)->bool{return false;}
fn umFaceOpen(id:vec3i,axis:u32)->f32{return 1.0;}
fn umPressureFaceV(id:vec3i,axis:u32)->f32{return 1.0;}
fn umSolidAtWorldCell(p:vec3f)->bool{return false;}
`;
 return /* wgsl */ `
struct UMSolidParams {
 dimsDt:vec4f,cellGravity:vec4f,container:vec4f,physical:vec4f,boundary:vec4f,
 inflowA:vec4f,inflowB:vec4f,inflowC:vec4f,tuning:vec4f,drop:vec4f,dropExtent:vec4f,
 twoLevel:vec4f,agreement:vec4f,lean:vec4f,splash:vec4f,splashB:vec4f,cleanup:vec4f,
}
@group(${group}) @binding(0) var<uniform> umSolidParams:UMSolidParams;
@group(${group}) @binding(1) var<storage,read> umSolidScratch:array<u32>;
@group(${group}) @binding(2) var umSolidTerrain:texture_2d<f32>;
fn umSolidEnabled()->bool{return true;}
fn umSolidValid(p:vec3i)->bool{return all(p>=vec3i(0))&&all(p<vec3i(UM_D));}
// staticSolidVoxelOccupied: lattice plus a one-cell halo (the box shell).
fn umSolidVoxel(p:vec3i)->bool{
 let shape=vec3i(UM_D)+vec3i(2);let q=p+vec3i(1);
 if(any(q<vec3i(0))||any(q>=shape)){return false;}
 let index=u32(q.x+shape.x*(q.y+shape.y*q.z));
 return (umSolidScratch[u32(round(umSolidParams.dropExtent.z))+4u+(index>>5u)]&(1u<<(index&31u)))!=0u;
}
fn umSolidHasTerrain()->bool{return umSolidParams.container.w>0.5;}
fn umSolidTerrainHeight(x:i32,z:i32)->f32{let d=vec3i(UM_D);return textureLoad(umSolidTerrain,vec2i(clamp(x,0,d.x-1),clamp(z,0,d.z-1)),0).x;}
fn umCellInsideTerrain(p:vec3i)->bool{if(!umSolidHasTerrain()){return false;}return f32(p.y)+0.5<umSolidTerrainHeight(p.x,p.z);}
fn umCellTerrainFraction(p:vec3i)->f32{if(!umSolidHasTerrain()){return 0.0;}return clamp(umSolidTerrainHeight(p.x,p.z)-f32(p.y),0.0,1.0);}
// cellOpenFraction without bodies.
fn umCellOpen(p:vec3i)->f32{
 if(!umSolidValid(p)||umSolidVoxel(p)){return 0.0;}
 return clamp((1.0-0.0)*(1.0-umCellTerrainFraction(p)),0.0,1.0);
}
// cellInsideSolid||cellInsideTerrain, the CM11a p_min=0 rows.
fn umCellInsideSolid(p:vec3i)->bool{return umSolidVoxel(p)||(umSolidValid(p)&&umCellInsideTerrain(p));}
fn umSolidWorldCell(id:vec3i)->vec3f{let h=umSolidParams.cellGravity.xyz;
 return vec3f(-0.5*umSolidParams.container.x+(f32(id.x)+0.5)*h.x,(f32(id.y)+0.5)*h.y,-0.5*umSolidParams.container.z+(f32(id.z)+0.5)*h.z);}
fn umSolidFaceWorld(id:vec3i,axis:u32)->vec3f{var world=umSolidWorldCell(id);world[axis]+=0.5*umSolidParams.cellGravity.xyz[axis];return world;}
fn umSolidTraceWorld(p:vec3f)->vec3f{let h=umSolidParams.cellGravity.xyz;
 return vec3f(-0.5*umSolidParams.container.x+p.x*h.x,p.y*h.y,-0.5*umSolidParams.container.z+p.z*h.z);}
fn umSolidVoxelAtWorld(world:vec3f)->bool{
 let h=umSolidParams.cellGravity.xyz;
 let p=vec3i(floor(vec3f((world.x+0.5*umSolidParams.container.x)/h.x,world.y/h.y,(world.z+0.5*umSolidParams.container.z)/h.z)));
 return umSolidVoxel(p);
}
fn umSolidTerrainAtWorld(world:vec3f)->bool{
 if(!umSolidHasTerrain()){return false;}
 let h=umSolidParams.cellGravity.xyz;let d=vec3i(UM_D);
 let x=clamp(i32(floor((world.x+0.5*umSolidParams.container.x)/h.x)),0,d.x-1);
 let z=clamp(i32(floor((world.z+0.5*umSolidParams.container.z)/h.z)),0,d.z-1);
 return world.y<umSolidTerrainHeight(x,z)*h.y;
}
fn umSolidAtWorld(world:vec3f)->bool{return umSolidVoxelAtWorld(world)||umSolidTerrainAtWorld(world);}
// insideAnyTraceSolid for a lattice-coordinate point: voxels only (terrain
// is not a trace solid natively).
fn umSolidAtWorldCell(p:vec3f)->bool{return umSolidVoxelAtWorld(umSolidTraceWorld(p));}
// faceOpenFraction: the Sec. 3.6 transverse face aperture.
fn umFaceOpen(id:vec3i,axis:u32)->f32{
 var neighbor=id;neighbor[axis]+=1;
 if(umSolidVoxel(id)||umSolidVoxel(neighbor)){return 0.0;}
 if(!umSolidValid(id)||!umSolidValid(neighbor)){return 1.0;}
 let world=umSolidFaceWorld(id,axis);let h=umSolidParams.cellGravity.xyz;
 let tangentA=(axis+1u)%3u;let tangentB=(axis+2u)%3u;var solid=0.0;
 for(var sampleIndex=0u;sampleIndex<4u;sampleIndex+=1u){
  var sampleWorld=world;
  sampleWorld[tangentA]+=select(-0.35,0.35,(sampleIndex&1u)!=0u)*h[tangentA];
  sampleWorld[tangentB]+=select(-0.35,0.35,(sampleIndex&2u)!=0u)*h[tangentB];
  solid+=select(0.0,1.0,umSolidAtWorld(sampleWorld));
 }
 return 1.0-solid*0.25;
}
// pressureFaceData(id,axis).w, geometric and static: the CM11a dual-cell V.
fn umPressureFaceV(id:vec3i,axis:u32)->f32{
 var neighbor=id;neighbor[axis]+=1;
 if(umSolidValid(id)!=umSolidValid(neighbor)){
  if(axis==2u&&umSolidParams.tuning.w>0.5){return 0.0;}
  let ambient=axis==1u&&max(id.y,neighbor.y)==i32(UM_D.y)&&umSolidParams.boundary.w>0.5;
  let open=umCellOpen(select(neighbor,id,umSolidValid(id)));
  return select(0.5*open,0.5*(1.0+open),ambient);
 }
 if(umSolidVoxel(id)||umSolidVoxel(neighbor)){return 0.5*(umCellOpen(id)+umCellOpen(neighbor));}
 if(!umSolidValid(id)||!umSolidValid(neighbor)){return 0.0;}
 let world=umSolidFaceWorld(id,axis);let h=umSolidParams.cellGravity.xyz;var solid=0.0;
 for(var sampleIndex=0u;sampleIndex<8u;sampleIndex+=1u){
  let sampleWorld=world+vec3f(select(-0.4,0.4,(sampleIndex&1u)!=0u)*h.x,select(-0.4,0.4,(sampleIndex&2u)!=0u)*h.y,select(-0.4,0.4,(sampleIndex&4u)!=0u)*h.z);
  solid+=select(0.0,1.0,umSolidAtWorld(sampleWorld));
 }
 return 1.0-solid/8.0;
}
`;
}
