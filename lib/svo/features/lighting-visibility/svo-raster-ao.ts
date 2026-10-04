import { SVO_SURFACE_MESH_STATE as M } from "../primary-visibility/svo-surface-mesh";

/** Preview backend. Fixed world-space coverage is deliberately independent of the camera. */
export const SVO_RASTER_SHADOW_SIZE = 2048;
export interface RasterAoPipelines {
  shadow: GPURenderPipeline;
  terrain: GPURenderPipeline;
  ao: GPUComputePipeline;
}

const paramsWGSL = /* wgsl */ `
struct RasterLightingParams { control:vec4u, farCenterExtent:vec4f, depthCell:vec4f, sun:vec4f }
@group(2) @binding(0) var<uniform> rasterLighting:RasterLightingParams;
`;

/** Shared by the caster and receiver: conventional [0,1] shadow depth, reversed camera depth unchanged. */
export const rasterAoConsumerWGSL = /* wgsl */ `
${paramsWGSL}
@group(2) @binding(1) var rasterSunDepth:texture_depth_2d_array;
@group(2) @binding(2) var rasterSunSampler:sampler_comparison;
@group(2) @binding(3) var rasterContact:texture_2d<f32>;
@group(2) @binding(5) var rasterShadowVisibility:texture_2d<f32>;
var<private> rasterResolvedVisibility:vec2f;
var<private> rasterPixel:vec2f;
fn rasterSunDirection()->vec3f{return normalize(rasterLighting.sun.xyz);}
fn rasterSunBasis()->mat3x3f{
  let sun=rasterSunDirection();let helper=select(vec3f(0,1,0),vec3f(1,0,0),abs(sun.y)>.95);
  let right=normalize(cross(helper,sun));return mat3x3f(right,cross(sun,right),sun);
}
fn rasterMapCenter(layer:u32)->vec3f{return select(uniforms.container.xyz*.5,rasterLighting.farCenterExtent.xyz,layer==1u);}
fn rasterMapExtent(layer:u32)->f32{return select(max(length(uniforms.container.xyz)*.7,.1),rasterLighting.farCenterExtent.w,layer==1u);}
fn rasterShadowPosition(world:vec3f,layer:u32)->vec3f{
  let basis=rasterSunBasis();let delta=world-rasterMapCenter(layer);let extent=rasterMapExtent(layer);
  return vec3f(dot(delta,basis[0])/extent,dot(delta,basis[1])/extent,.5-dot(delta,basis[2])/rasterLighting.depthCell.x);
}
fn rasterPcf(position:vec3f,normal:vec3f,layer:u32)->f32{
  let texel=2.0*rasterMapExtent(layer)/f32(${SVO_RASTER_SHADOW_SIZE});
  // World-space normal offset and a small depth bias, scaled with shadow texels.
  let biased=position+normal*(texel*1.5+rasterLighting.depthCell.y*(.18+dry.tuningRays0.x));
  let p=rasterShadowPosition(biased,layer);let uv=p.xy*vec2f(.5,-.5)+.5;
  if(any(uv<vec2f(0))||any(uv>vec2f(1))||p.z<=0.0||p.z>=1.0){return 1.0;}
  // Blocker search and a bounded PCSS filter retain contact hardening. The
  // softness control is the same angular diameter used by the cone reference.
  let offsets=array<vec2f,12>(vec2f(.13,.09),vec2f(-.32,.23),vec2f(.25,-.39),vec2f(-.43,-.32),vec2f(.60,.15),vec2f(-.09,.67),vec2f(-.69,.12),vec2f(.09,-.76),vec2f(.55,.63),vec2f(-.57,.62),vec2f(-.64,-.64),vec2f(.79,-.48));
  let angular=tan(dry.tuningRays1.y*.5);
  let search=max(2.0,length(uniforms.container.xyz)*angular/texel);
  let bias=texel*.3/rasterLighting.depthCell.x;var blocker=0.0;var count=0.0;
  for(var i=0u;i<12u;i++){
    let c=clamp(vec2i(uv*f32(${SVO_RASTER_SHADOW_SIZE})+offsets[i]*search),vec2i(0),vec2i(${SVO_RASTER_SHADOW_SIZE - 1}));
    let d=textureLoad(rasterSunDepth,c,i32(layer),0);
    if(d<p.z-bias){blocker+=d;count+=1.0;}
  }
  if(count==0.0){return 1.0;}
  let penumbra=clamp((p.z-blocker/count)*rasterLighting.depthCell.x*angular/texel,1.0,48.0);
  let disk=array<vec2f,24>(vec2f(0.144338,0.000000),vec2f(-0.184342,0.168873),vec2f(0.028217,-0.321513),vec2f(0.232351,0.303061),vec2f(-0.426393,-0.075423),vec2f(0.403917,-0.256939),vec2f(-0.135102,0.502574),vec2f(-0.257655,-0.496099),vec2f(0.559008,0.204149),vec2f(-0.581555,0.240057),vec2f(0.280348,-0.599087),vec2f(0.207170,0.660490),vec2f(-0.624412,-0.361860),vec2f(0.732507,-0.161040),vec2f(-0.447038,0.635865),vec2f(-0.103276,-0.796974),vec2f(0.634013,0.534347),vec2f(-0.853183,0.035282),vec2f(0.622332,-0.619303),vec2f(-0.041636,0.900426),vec2f(-0.592151,-0.709594),vec2f(0.938032,0.126211),vec2f(-0.794793,0.552996),vec2f(0.217183,-0.965401));
  var visibility=0.0;
  for(var i=0u;i<24u;i++){
    visibility+=textureSampleCompareLevel(rasterSunDepth,rasterSunSampler,uv+disk[i]*penumbra/f32(${SVO_RASTER_SHADOW_SIZE}),i32(layer),p.z-bias);
  }return visibility/24.0;
}
fn rasterSunVisibility(position:vec3f,normal:vec3f)->f32{
  let near=rasterShadowPosition(position,0u);let edge=max(abs(near.x),abs(near.y));
  if(edge<.85){return rasterPcf(position,normal,0u);}
  let far=rasterPcf(position,normal,1u);if(edge>=.98){return far;}
  return mix(rasterPcf(position,normal,0u),far,smoothstep(.85,.98,edge));
}
fn rasterOctEncode(n:vec3f)->vec2f{
  let p=n/max(abs(n.x)+abs(n.y)+abs(n.z),1e-6);
  return select((1.0-abs(p.yx))*select(vec2f(-1),vec2f(1),p.xy>=vec2f(0)),p.xy,p.z>=0.0);
}
fn rasterOctDecode(p:vec2f)->vec3f{
  var n=vec3f(p,1.0-abs(p.x)-abs(p.y));let t=clamp(-n.z,0.0,1.0);
  n.x+=select(t,-t,n.x>=0.0);n.y+=select(t,-t,n.y>=0.0);return normalize(n);
}
fn rasterVisibilityAt(position:vec3f,normal:vec3f)->vec2f{
  let depth=length(position-uniforms.cameraPosition.xyz);let q=rasterPixel*.5-.75;let base=vec2i(floor(q));let f=fract(q);
  var total=vec2f(0.0);var weight=0.0;let dims=vec2i(textureDimensions(rasterContact));
  for(var y=0;y<2;y++){for(var x=0;x<2;x++){
    let c=clamp(base+vec2i(x,y),vec2i(0),dims-1);let v=textureLoad(rasterContact,c,0);
    let bilinear=select(1.0-f.x,f.x,x==1)*select(1.0-f.y,f.y,y==1);
    let tolerance=max(depth*.005,rasterLighting.depthCell.y*.5);
    let w=bilinear*exp(-abs(v.y-depth)/max(tolerance,1e-4))*pow(max(dot(normal,rasterOctDecode(v.zw)),0.0),8.0);
    total+=vec2f(v.x,textureLoad(rasterShadowVisibility,c,0).x)*w;weight+=w;
  }}
  if(weight>1e-4){return total/weight;}
  // A subpixel voxel face may be absent from all four reduced samples. Fully
  // lit is not a valid reconstruction there: it punches white pinholes into
  // both shadows and AO. Evaluate only those receivers at their actual pixel.
  var contact=1.0;var shadow=1.0;
  if((dry.materialPublication.w&1u)!=0u){contact=min(rasterHorizonVisibility(rasterPixel,position,normal),rasterCoarseVisibility(position,normal));}
  if((dry.materialPublication.w&2u)!=0u){shadow=rasterSunVisibility(position,normal);}
  return vec2f(contact,shadow);
}
fn rasterContactVisibility()->vec3f{
  if((dry.materialPublication.w&1u)==0u){return vec3f(1);}
  return vec3f(mix(1.0,rasterResolvedVisibility.x,dry.tuningRays0.w));
}
`;

/** Local estimator shared with the GPU reconstruction regression fixture. */
export const rasterHorizonWGSL = /* wgsl */ `
fn rasterWorldAt(pixel:vec2f,depth:f32)->vec3f{
  let camera=dryRasterPrimaryCamera();let ndc=pixel/uniforms.viewport.xy*vec2f(2,-2)+vec2f(-1,1);
  return camera[0]+normalize(camera[1]+camera[2]*ndc.x*uniforms.viewport.x/max(uniforms.viewport.y,1.0)*cameraTanHalfFov()+camera[3]*ndc.y*cameraTanHalfFov())*depth;
}
// Stable local contact AO, shared by the reduced pass and uncovered receivers.
// Integrate the maximum obscured elevation per azimuth. Squaring each horizon
// erased moderate contact angles; apply contrast to visibility after averaging.
fn rasterHorizonVisibility(pixel:vec2f,position:vec3f,normal:vec3f)->f32{
  let fullDims=vec2i(textureDimensions(drySplitGeometryRead));
  let radius=dryContactVisibilityRadius();
  let viewDepth=dot(position-uniforms.cameraPosition.xyz,normalize(uniforms.cameraTarget.xyz-uniforms.cameraPosition.xyz));
  let projected=clamp(radius*uniforms.viewport.y/(2.0*max(viewDepth,.001)*cameraTanHalfFov()),2.0,160.0);
  var occlusion=0.0;
  // Six azimuths, six quadratic distance samples. A small tangent bias prevents
  // the receiver plane and fitted-normal noise from darkening themselves.
  for(var direction=0u;direction<6u;direction++){
    let angle=(f32(direction)+.5)*6.28318530718/6.0;let axis=vec2f(cos(angle),sin(angle));var horizon=0.0;
    for(var step=1u;step<=6u;step++){
      let fraction=f32(step)/6.0;let tap=vec2i(round(pixel-.5+axis*max(1.0,projected*fraction*fraction)));
      if(any(tap<vec2i(0))||any(tap>=fullDims)){continue;}
      let sample=drySplitGeometryAt(tap);if(!(sample.w<DRY_MISS)){continue;}
      let delta=rasterWorldAt(vec2f(tap)+.5,sample.w)-position;let distance=length(delta);
      if(distance<1e-5||distance>=radius){continue;}
      let cosine=max(dot(normal,delta/distance)-.08,0.0);
      horizon=max(horizon,cosine*(1.0-distance*distance/(radius*radius)));
    }occlusion+=horizon;
  }
  return pow(clamp(1.0-occlusion/6.0,0.0,1.0),2.0);
}
`;

/** Eight bounded opacity lookups, with no cone march or temporal history. */
export const rasterCoarseAoWGSL = /* wgsl */ `
fn rasterCoarseVisibility(position:vec3f,normal:vec3f)->f32{
  let strength=rasterLighting.sun.w;
  if(strength<=0.0||!dryNodeMipReady()){return 1.0;}
  let cell=max(dry.mapping.cellSize.x,max(dry.mapping.cellSize.y,dry.mapping.cellSize.z));
  let radius=max(cell*4.0,dryContactVisibilityRadius()*2.0);
  let helper=select(vec3f(0,1,0),vec3f(1,0,0),abs(normal.y)>.9);
  let tangent=normalize(cross(helper,normal));let bitangent=cross(normal,tangent);
  let origin=position+normal*cell*.5;
  var visibility=0.0;
  for(var direction=0u;direction<4u;direction++){
    let angle=(f32(direction)+.5)*1.57079632679;
    let ray=normalize(normal+(tangent*cos(angle)+bitangent*sin(angle))*.85);
    var transmission=1.0;
    var maximumDensity=0.0;
    var cache=DryNodeMipPageCache(vec3u(0u),0xffffffffu,vec3u(0u),0u,0u,0xffffffffu,0u);
    for(var shell=0u;shell<2u;shell++){
      let distance=radius*exp2(f32(shell));
      // A trilinear opacity lookup includes whole mip voxels, extending up to
      // 1.5 voxel widths from the query point on each axis. Keep that support
      // in front of the receiver plane, including on diagonal walls.
      let clearance=dot(normal,ray)*distance+cell*.5;
      let diameter=min(distance*.8,clearance*.9/(1.5*dot(abs(normal),vec3f(1))));
      let lod=max(0.0,floor(log2(diameter/cell)));
      if(cell*exp2(f32(dryNodeMipOpacityLevelFloor()))>diameter){continue;}
      let sample=dryNodeMipAt(origin+ray*distance,lod,&cache);
      if(sample.valid==0u){return 1.0;}
      // Mean occupancy retains the gaps between leaves. Max occupancy would
      // turn a sparse canopy into an opaque shell.
      let density=clamp(sample.sample.solidMean,0.0,1.0);
      maximumDensity=max(maximumDensity,density);
      transmission*=exp(-density*distance/(cell*exp2(lod))*2.0);
    }
    // Two shells resolve sparse canopy density, but cannot locate the horizon
    // of a solid floor: treating it as volume creates broad bands on walls.
    // Hand dense blockers back to screen-space contact AO, with a smooth fade.
    let sparseWeight=1.0-smoothstep(.15,.5,maximumDensity);
    visibility+=mix(1.0,transmission,sparseWeight);
  }
  return mix(1.0,visibility*.25,strength);
}
`;

export function rasterAoProducerWGSL(backdrop: boolean): string {
  return /* wgsl */ `
@group(2) @binding(4) var rasterContactOutput:texture_storage_2d<rgba16float,write>;
@group(2) @binding(6) var rasterShadowOutput:texture_storage_2d<r32float,write>;
@vertex fn rasterShadowVertex(@builtin(vertex_index) vertex:u32,@builtin(instance_index) instance:u32)->@builtin(position) vec4f{
  // Deliberately enumerate the complete front arena, never camera-visible indices.
  let quad=meshDrawnQuad(instance);let face=meshQuadFace(quad.face);let axis=face/2u;let u=(axis+1u)%3u;let v=(axis+2u)%3u;
  let corners=array<vec2u,4>(vec2u(1,0),vec2u(1,1),vec2u(0,0),vec2u(0,1));
  let corner=select(corners[vertex],corners[vertex].yx,(face&1u)==0u);var lattice=meshQuadOrigin(quad);lattice[u]+=corner.x*quad.extent[u];lattice[v]+=corner.y*quad.extent[v];
  var world=dry.mapping.worldOrigin+vec3f(lattice)*dry.mapping.cellSize;
  if(meshTriangle(quad)){world=dry.mapping.worldOrigin+(vec3f(meshQuadOrigin(quad))+meshTrianglePoint(quad,min(vertex,2u))*vec3f(meshQuadExtent(quad)))*dry.mapping.cellSize;}
  let p=rasterShadowPosition(world,rasterLighting.control.y);
  // Level zero is the complete geometric surface; other levels duplicate it.
  if(meshQuadDead(quad)||meshQuadLevel(quad.face)!=0u){return vec4f(0,0,0,1);}
  return vec4f(p,1);
}
@vertex fn rasterShadowClearVertex(@builtin(vertex_index) vertex:u32)->@builtin(position) vec4f{
  let p=array<vec2f,3>(vec2f(-1,-1),vec2f(3,-1),vec2f(-1,3));return vec4f(p[vertex],1,1);
}
@fragment fn rasterShadowTerrain(@builtin(position) pixel:vec4f)->@builtin(frag_depth) f32{
  ${backdrop ? `if(backdropTerrainLevels()==0u){return 1.0;}
  let layer=rasterLighting.control.y;let uv=pixel.xy/f32(${SVO_RASTER_SHADOW_SIZE})*vec2f(2,-2)+vec2f(-1,1);
  let basis=rasterSunBasis();let origin=rasterMapCenter(layer)+(basis[0]*uv.x+basis[1]*uv.y)*rasterMapExtent(layer)+basis[2]*rasterLighting.depthCell.x*.5;
  let hit=backdropTerrainSmoothTrace(origin,-basis[2],0.0,rasterLighting.depthCell.x);
  if(hit.t>=0.0){return clamp(hit.t/rasterLighting.depthCell.x,0.0,1.0);}` : ""}
  return 1.0;
}
${rasterHorizonWGSL}
${rasterCoarseAoWGSL}
@compute @workgroup_size(8,8) fn rasterContactMain(@builtin(global_invocation_id) id:vec3u){
  let dims=textureDimensions(rasterContactOutput);if(any(id.xy>=dims)){return;}
  let fullDims=vec2i(textureDimensions(drySplitGeometryRead));
  let coord=min(vec2i(id.xy)*2+vec2i(1),fullDims-1);let g=drySplitGeometryAt(coord);let pixel=vec2f(coord)+.5;
  if(!(g.w<DRY_MISS)||dot(g.xyz,g.xyz)<.1){textureStore(rasterContactOutput,vec2i(id.xy),vec4f(1,65504,0,0));textureStore(rasterShadowOutput,vec2i(id.xy),vec4f(1));return;}
  let normal=normalize(g.xyz);let position=rasterWorldAt(pixel,g.w);
  var contact=1.0;
  if((dry.materialPublication.w&1u)!=0u){contact=min(rasterHorizonVisibility(pixel,position,normal),rasterCoarseVisibility(position,normal));}
  var shadow=1.0;
  if((dry.materialPublication.w&2u)!=0u){shadow=rasterSunVisibility(position,normal);}
  textureStore(rasterShadowOutput,vec2i(id.xy),vec4f(shadow));
  textureStore(rasterContactOutput,vec2i(id.xy),vec4f(contact,min(g.w,65504.0),rasterOctEncode(normal)));
}
`;
}

const cacheWGSL = /* wgsl */ `
struct Params { control:vec4u, farCenterExtent:vec4f, depthCell:vec4f, sun:vec4f }
@group(0) @binding(0) var<storage,read> mesh:array<u32>;
@group(0) @binding(1) var<uniform> params:Params;
@group(0) @binding(2) var<storage,read_write> cache:array<u32>;
@compute @workgroup_size(1) fn main(){
  let revisions=array<u32,8>(params.control.x,mesh[${M.topologyRevision}],mesh[${M.geometryRevision}],mesh[${M.frontCursor}],mesh[${M.front}],mesh[${M.builds}],mesh[${M.consumedMaintenanceRevision}],mesh[${M.usable}]);
  var dirty=mesh[${M.building}]!=0u;
  for(var i=0u;i<8u;i++){dirty=dirty||cache[8u+i]!=revisions[i];cache[8u+i]=revisions[i];}
  // Clearing is an indirect full-screen draw so unchanged maps can use load.
  cache[0]=3u;cache[1]=select(0u,1u,dirty);cache[2]=0u;cache[3]=0u;
  cache[4]=4u;cache[5]=select(0u,mesh[${M.frontCursor}],dirty);cache[6]=0u;cache[7]=0u;
}
`;

export class SvoRasterAo {
  readonly consumerLayout: GPUBindGroupLayout;
  readonly shadowLayout: GPUBindGroupLayout;
  readonly aoLayout: GPUBindGroupLayout;
  readonly params: GPUBuffer;
  readonly shadow: GPUTexture;
  readonly cache: GPUBuffer;
  readonly sampler: GPUSampler;
  private cachePipeline?: GPUComputePipeline;
  private cacheCompile?: Promise<GPUComputePipeline>;
  private cacheGroup?: GPUBindGroup;
  private meshState?: GPUBuffer;
  private shadowGroups: GPUBindGroup[];
  private shadowViews: GPUTextureView[];
  private ao?: GPUTexture;
  private shadowVisibility?: GPUTexture;
  private aoGroup?: GPUBindGroup;
  consumer?: GPUBindGroup;
  private key = "";
  private generation = 0;
  private coarseAoStrength = 0;

  constructor(private readonly device: GPUDevice) {
    this.params=device.createBuffer({label:"Raster lighting parameters",size:512,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
    this.cache=device.createBuffer({label:"Raster shadow cache generations and indirect draws",size:64,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.INDIRECT|GPUBufferUsage.COPY_SRC});
    const uniform:GPUBindGroupLayoutEntry={binding:0,visibility:GPUShaderStage.VERTEX|GPUShaderStage.FRAGMENT|GPUShaderStage.COMPUTE,buffer:{type:"uniform"}};
    this.consumerLayout=device.createBindGroupLayout({label:"Raster lighting consumer",entries:[uniform,
      {binding:1,visibility:GPUShaderStage.FRAGMENT,texture:{sampleType:"depth",viewDimension:"2d-array"}},
      {binding:2,visibility:GPUShaderStage.FRAGMENT,sampler:{type:"comparison"}},
      {binding:3,visibility:GPUShaderStage.FRAGMENT,texture:{sampleType:"unfilterable-float"}},
      {binding:5,visibility:GPUShaderStage.FRAGMENT,texture:{sampleType:"unfilterable-float"}}]});
    this.shadowLayout=device.createBindGroupLayout({entries:[uniform]});
    this.aoLayout=device.createBindGroupLayout({entries:[uniform,
      {binding:1,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"depth",viewDimension:"2d-array"}},
      {binding:2,visibility:GPUShaderStage.COMPUTE,sampler:{type:"comparison"}},
      {binding:4,visibility:GPUShaderStage.COMPUTE,storageTexture:{access:"write-only",format:"rgba16float"}},
      {binding:6,visibility:GPUShaderStage.COMPUTE,storageTexture:{access:"write-only",format:"r32float"}}]});
    this.shadow=device.createTexture({label:"Cached world-space sun shadows",size:[SVO_RASTER_SHADOW_SIZE,SVO_RASTER_SHADOW_SIZE,2],format:"depth32float",usage:GPUTextureUsage.RENDER_ATTACHMENT|GPUTextureUsage.TEXTURE_BINDING});
    this.shadowViews=[0,1].map(baseArrayLayer=>this.shadow.createView({dimension:"2d",baseArrayLayer,arrayLayerCount:1}));
    this.shadowGroups=[0,1].map(layer=>device.createBindGroup({layout:this.shadowLayout,entries:[{binding:0,resource:{buffer:this.params,offset:layer*256,size:64}}]}));
    this.sampler=device.createSampler({compare:"less-equal",magFilter:"linear",minFilter:"linear"});
  }
  async compile(module:GPUShaderModule,scene:GPUBindGroupLayout,mesh:GPUBindGroupLayout,split:GPUBindGroupLayout):Promise<RasterAoPipelines>{
    this.cacheCompile ??= this.device.createComputePipelineAsync({label:"Raster shadow invalidation",layout:"auto",compute:{module:this.device.createShaderModule({code:cacheWGSL}),entryPoint:"main"}});
    this.cachePipeline = await this.cacheCompile;
    const layout=this.device.createPipelineLayout({bindGroupLayouts:[scene,mesh,this.shadowLayout]});
    const [shadow,terrain,ao]=await Promise.all([
      this.device.createRenderPipelineAsync({label:"Raster sun shadow casters",layout,vertex:{module,entryPoint:"rasterShadowVertex"},primitive:{topology:"triangle-strip",cullMode:"none"},depthStencil:{format:"depth32float",depthWriteEnabled:true,depthCompare:"less"}}),
      this.device.createRenderPipelineAsync({label:"Raster sun terrain and clear",layout,vertex:{module,entryPoint:"rasterShadowClearVertex"},fragment:{module,entryPoint:"rasterShadowTerrain",targets:[]},primitive:{topology:"triangle-list"},depthStencil:{format:"depth32float",depthWriteEnabled:true,depthCompare:"always"}}),
      this.device.createComputePipelineAsync({label:"Horizon contact AO",layout:this.device.createPipelineLayout({bindGroupLayouts:[scene,split,this.aoLayout]}),compute:{module,entryPoint:"rasterContactMain"}}),
    ]);return {shadow,terrain,ao};
  }
  update(origin:readonly number[],extent:readonly number[],cellSize:readonly number[],publicationKey:string,direction:readonly number[]):void{
    const key=[...origin,...extent,...cellSize,...direction,publicationKey].join(",");if(key===this.key)return;
    this.key=key;this.generation++;
    const radius=Math.max(.1,Math.hypot(...extent)*.5);
    for(let layer=0;layer<2;layer++){
      const buffer=new ArrayBuffer(64),f=new Float32Array(buffer),u=new Uint32Array(buffer);
      u.set([this.generation,layer,0,0]);f.set(origin.map((v,i)=>v+extent[i]*.5),4);f[7]=radius*1.02;
      f[8]=radius*4;f[9]=Math.max(...cellSize);f.set(direction,12);f[15]=this.coarseAoStrength;
      this.device.queue.writeBuffer(this.params,layer*256,buffer);
    }
  }
  setCoarseAoStrength(strength:number):void{
    if(strength===this.coarseAoStrength)return;
    this.coarseAoStrength=strength;
    for(let layer=0;layer<2;layer++)this.device.queue.writeBuffer(this.params,layer*256+60,new Float32Array([strength]));
  }
  ensureSize(width:number,height:number):void{
    width=Math.ceil(width/2);height=Math.ceil(height/2);
    if(this.ao?.width===width&&this.ao.height===height)return;
    this.ao?.destroy();this.ao=this.device.createTexture({label:"Half-resolution sun visibility and horizon AO",size:[width,height],format:"rgba16float",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.STORAGE_BINDING});
    this.shadowVisibility?.destroy();this.shadowVisibility=this.device.createTexture({label:"Half-resolution sun visibility",size:[width,height],format:"r32float",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.STORAGE_BINDING});
    const uniform={buffer:this.params,offset:0,size:64};const view=this.ao.createView();const shadowView=this.shadowVisibility.createView();
    this.aoGroup=this.device.createBindGroup({layout:this.aoLayout,entries:[{binding:0,resource:uniform},{binding:1,resource:this.shadow.createView({dimension:"2d-array"})},{binding:2,resource:this.sampler},{binding:4,resource:view},{binding:6,resource:shadowView}]});
    this.consumer=this.device.createBindGroup({layout:this.consumerLayout,entries:[{binding:0,resource:uniform},{binding:1,resource:this.shadow.createView({dimension:"2d-array"})},{binding:2,resource:this.sampler},{binding:3,resource:view},{binding:5,resource:shadowView}]});
  }
  encode(encoder:GPUCommandEncoder,pipelines:RasterAoPipelines,scene:GPUBindGroup,mesh:GPUBindGroup,split:GPUBindGroup,state:GPUBuffer,aoEnabled:boolean,shadowsEnabled:boolean,shadowComplete?:()=>void,aoComplete?:()=>void):void{
    if(state!==this.meshState){this.meshState=state;this.cacheGroup=this.device.createBindGroup({layout:this.cachePipeline!.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:state}},{binding:1,resource:{buffer:this.params,offset:0,size:64}},{binding:2,resource:{buffer:this.cache}}]});}
    if(shadowsEnabled){
    const invalidation=encoder.beginComputePass({label:"Raster shadow cache check"});invalidation.setPipeline(this.cachePipeline!);invalidation.setBindGroup(0,this.cacheGroup!);invalidation.dispatchWorkgroups(1);invalidation.end();
    for(let layer=0;layer<2;layer++){
      const pass=encoder.beginRenderPass({label:`Cached sun shadow ${layer}`,colorAttachments:[],depthStencilAttachment:{view:this.shadowViews[layer],depthLoadOp:"load",depthStoreOp:"store"}});
      pass.setBindGroup(0,scene);pass.setBindGroup(1,mesh);pass.setBindGroup(2,this.shadowGroups[layer]);pass.setPipeline(pipelines.terrain);pass.drawIndirect(this.cache,0);pass.setPipeline(pipelines.shadow);pass.drawIndirect(this.cache,16);pass.end();
    }
    shadowComplete?.();
    }
    if(aoEnabled||shadowsEnabled){const ao=encoder.beginComputePass({label:"Half-resolution sun visibility and horizon AO"});ao.setPipeline(pipelines.ao);ao.setBindGroup(0,scene);ao.setBindGroup(1,split);ao.setBindGroup(2,this.aoGroup!);ao.dispatchWorkgroups(Math.ceil(this.ao!.width/8),Math.ceil(this.ao!.height/8));ao.end();aoComplete?.();}
  }
  destroy():void{this.params.destroy();this.cache.destroy();this.shadow.destroy();this.ao?.destroy();this.shadowVisibility?.destroy();}
}
