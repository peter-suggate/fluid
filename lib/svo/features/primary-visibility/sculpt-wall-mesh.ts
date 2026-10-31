import type { SculptBakePlan, SculptPage } from "../scene-publication/svo-sculpt-substrate";

/** Fixed per-page budget; an overflowing page publishes its old exact field
 * proxy instead of truncating geometry. No count readback or host meshing. */
export const SCULPT_WALL_VERTEX_CAPACITY = 8190;
export const sculptWallVertexLayout: GPUVertexBufferLayout = { arrayStride: 32, attributes: [
  { shaderLocation: 0, offset: 0, format: "float32x3" }, { shaderLocation: 1, offset: 12, format: "uint32" },
  { shaderLocation: 2, offset: 16, format: "float32x3" }, { shaderLocation: 3, offset: 28, format: "uint32" },
] };
export const sculptWallMeshWGSL = /* wgsl */ `
struct SculptWallVertex { world:vec3f,material:u32,normal:vec3f,owner:u32 }
@group(1) @binding(90) var<storage,read_write> sculptWallVertices:array<SculptWallVertex>;
@group(1) @binding(91) var<storage,read_write> sculptWallArgs:array<atomic<u32>>;
struct SculptWallSource { distance:f32,primitive:u32 }
fn sculptWallSource(world:vec3f)->SculptWallSource{
  var result=SculptWallSource(1e20,0xffffffffu);
  for(var i=0u;i<dry.metadata.x;i++){
    let record=dryPrimitive(i);if(svoPrimitiveKind(record)!=SVO_KIND_FIELD_PROGRAM){continue;}
    let reference=svoPrimitiveFieldProgramReference(record);
    if(reference<DRY_SCENE_FIELD_PROGRAM_WORD_OFFSET||drySceneWord(reference)!=0xfffffffeu){continue;}
    let base=DRY_SCENE_FIELD_PROGRAM_WORD_OFFSET+drySceneWord(reference+1u);
    let d=dryFieldProgramBlockVolumeDistance(base,svoPrimitiveLocalPoint(record,world));
    if(d<result.distance){result=SculptWallSource(d,i);}
  }return result;
}
fn sculptWallEmit(a:vec3f,b:vec3f,c:vec3f){
  if(length(cross(b-a,c-a))<1e-14){return;}
  let source=sculptWallSource((a+b+c)/3.0);if(source.primitive==0xffffffffu){return;}
  let record=dryPrimitive(source.primitive);let reference=svoPrimitiveFieldProgramReference(record);
  let surface=dryFieldProgramBlockSurface((reference-DRY_SCENE_FIELD_PROGRAM_WORD_OFFSET)/DRY_SCENE_FIELD_PROGRAM_BLOCK_WORDS,svoPrimitiveLocalPoint(record,(a+b+c)/3.0));
  let material=select(svoPrimitiveMaterialId(record),surface.material,surface.material!=0u)+8192u;
  let at=atomicAdd(&sculptWallArgs[4],3u);if(at+3u>arrayLength(&sculptWallVertices)){return;}
  let points=array<vec3f,3>(a,b,c);
  for(var i=0u;i<3u;i++){
    var normal=-svoSculptCutSample(points[i]).normal;
    if(dot(normal,normal)<0.5){normal=normalize(cross(b-a,c-a));}
    sculptWallVertices[at+i]=SculptWallVertex(points[i],material,normal,svoPrimitiveOwnerId(record));
  }
}
fn sculptWallClip(a:vec3f,b:vec3f,c:vec3f){
  let input=array<vec3f,3>(a,b,c);
  let distance=array<f32,3>(sculptWallSource(a).distance,sculptWallSource(b).distance,sculptWallSource(c).distance);
  var polygon:array<vec3f,4>;var count=0u;
  for(var i=0u;i<3u;i++){
    let j=(i+1u)%3u;let inside=distance[i]<=0.0;let next=distance[j]<=0.0;
    if(inside){polygon[count]=input[i];count++;}
    if(inside!=next){polygon[count]=mix(input[i],input[j],clamp(distance[i]/(distance[i]-distance[j]),0.0,1.0));count++;}
  }
  for(var i=1u;i+1u<count;i++){sculptWallEmit(polygon[0],polygon[i],polygon[i+1u]);}
}
fn sculptWallEdge(a:vec4f,b:vec4f)->vec3f{return mix(a.xyz,b.xyz,clamp(a.w/(a.w-b.w),0.0,1.0));}
@compute @workgroup_size(64) fn sculptWallBuild(@builtin(global_invocation_id) id:vec3u){
  if(id.x>=512u){return;}
  let slot=atomicLoad(&sculptWallArgs[5]);let bounds=importedCutBounds(slot);let h=sculptFieldCell();
  let cell=vec3u(id.x%8u,(id.x/8u)%8u,id.x/64u);
  var corners:array<vec4f,8>;var low=1e20;var high=-1e20;
  for(var i=0u;i<8u;i++){
    let p=cell+vec3u(i&1u,(i>>1u)&1u,i>>2u);let d=sculptFieldValue(slot,p);
    corners[i]=vec4f(bounds[0]+vec3f(p)*h,d);low=min(low,d);high=max(high,d);
  }
  if(low>=0.0||high<0.0){return;}
  let tetra=array<vec4u,6>(vec4u(0,1,3,7),vec4u(0,3,2,7),vec4u(0,2,6,7),vec4u(0,6,4,7),vec4u(0,4,5,7),vec4u(0,5,1,7));
  for(var t=0u;t<6u;t++){
    var inside:array<vec4f,4>;var outside:array<vec4f,4>;var ni=0u;var no=0u;
    for(var j=0u;j<4u;j++){let p=corners[tetra[t][j]];if(p.w<0.0){inside[ni]=p;ni++;}else{outside[no]=p;no++;}}
    if(ni==1u){sculptWallClip(sculptWallEdge(inside[0],outside[0]),sculptWallEdge(inside[0],outside[1]),sculptWallEdge(inside[0],outside[2]));}
    if(ni==3u){sculptWallClip(sculptWallEdge(outside[0],inside[0]),sculptWallEdge(outside[0],inside[1]),sculptWallEdge(outside[0],inside[2]));}
    if(ni==2u){let a=sculptWallEdge(inside[0],outside[0]);let b=sculptWallEdge(inside[0],outside[1]);let c=sculptWallEdge(inside[1],outside[1]);let d=sculptWallEdge(inside[1],outside[0]);sculptWallClip(a,b,c);sculptWallClip(a,c,d);}
  }
}
@compute @workgroup_size(1) fn sculptWallPublish(){
  let count=atomicLoad(&sculptWallArgs[4]);
  if(count<=arrayLength(&sculptWallVertices)){atomicStore(&sculptWallArgs[0],count);return;}
  // Rare pathological page: publish a complete proxy, never a partial mesh.
  let slot=atomicLoad(&sculptWallArgs[5]);
  for(var i=0u;i<36u;i++){sculptWallVertices[i]=SculptWallVertex(importedCutProxy(i,slot),0xffffffffu,vec3f(0),slot);}
  atomicStore(&sculptWallArgs[0],36u);
}
`;

export const sculptWallPrimaryWGSL = /* wgsl */ `
struct SculptWallRasterVertex {
  @builtin(position) position:vec4f,@location(0) world:vec3f,@location(1) normal:vec3f,
  @location(2) @interpolate(flat) material:u32,@location(3) @interpolate(flat) owner:u32,
}
@vertex fn sculptWallVertex(@location(0) world:vec3f,@location(1) material:u32,@location(2) normal:vec3f,@location(3) owner:u32)->SculptWallRasterVertex{
  let camera=dryRasterPrimaryCamera();let relative=world-camera[0];let aspect=uniforms.viewport.x/max(uniforms.viewport.y,1.0);
  return SculptWallRasterVertex(vec4f(dot(relative,camera[2])/(aspect*cameraTanHalfFov()),dot(relative,camera[3])/cameraTanHalfFov(),DRY_REVERSED_Z_NEAR_M,dot(relative,camera[1])),world,normal,material,owner);
}
@fragment fn sculptWallFragment(input:SculptWallRasterVertex)->ImportedCutOut{
  let camera=dryRasterPrimaryCamera();let rd=normalize(input.world-camera[0]);
  var hit=DryHit(distance(input.world,camera[0]),normalize(input.normal),input.material,input.owner,SVO_FEATURE_SMOOTH,DRY_GBUFFER_FIELD_ANALYTIC,DRY_GBUFFER_MOTION_STATIC,1u,0.0,vec3u(0));
  if(input.material==0xffffffffu){hit=importedCutHit(input.owner,camera[0],rd,DRY_REVERSED_Z_NEAR_M/dot(rd,camera[1]),DRY_MISS);if(hit.t>=DRY_MISS){discard;}}
  let out=dryRasterPrimarySurface(hit,camera[0],rd,camera[1],SVO_GBUFFER_PRODUCER_SCENE_PRIMITIVE);
  return ImportedCutOut(out.packedSurface,out.identityMedia,out.geometry,out.opaqueIdentity,dryHardwareDepth(hit.t,rd,camera[1]));
}
`;

export interface SculptWallPipelines { build: GPUComputePipeline; publish: GPUComputePipeline }
interface CachedPage { page: SculptPage; vertices: GPUBuffer; args: GPUBuffer; group: GPUBindGroup }
export class SculptWallMesh {
  readonly layout: GPUBindGroupLayout;
  readonly pages = new Map<number, CachedPage>();
  revision = 0;
  builtPages = 0;
  private source?: unknown;
  constructor(private readonly device: GPUDevice) {
    this.layout = device.createBindGroupLayout({ entries: [90, 91].map(binding => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" as const } })) });
  }
  async compile(module: GPUShaderModule, scene: GPUBindGroupLayout): Promise<SculptWallPipelines> {
    const layout = this.device.createPipelineLayout({ bindGroupLayouts: [scene, this.layout] });
    const [build, publish] = await Promise.all(["sculptWallBuild", "sculptWallPublish"].map(entryPoint =>
      this.device.createComputePipelineAsync({ label: entryPoint, layout, compute: { module, entryPoint } })));
    return { build: build!, publish: publish! };
  }
  encode(encoder: GPUCommandEncoder, plan: SculptBakePlan | undefined, source: unknown, scene: GPUBindGroup, pipelines: SculptWallPipelines): void {
    const live = new Set([...plan?.pages.values() ?? []].map(page => page.slot));
    for (const [slot, held] of this.pages) if (!live.has(slot)) { held.vertices.destroy(); held.args.destroy(); this.pages.delete(slot); this.revision++; }
    const dirty: CachedPage[] = [];
    for (const page of plan?.pages.values() ?? []) {
      let held = this.pages.get(page.slot);
      if (!held) {
        const vertices = this.device.createBuffer({ label: "Cached sculpt wall triangles", size: SCULPT_WALL_VERTEX_CAPACITY * 32, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_SRC });
        const args = this.device.createBuffer({ label: "Cached sculpt wall draw", size: 32, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
        const group = this.device.createBindGroup({ layout: this.layout, entries: [{ binding: 90, resource: { buffer: vertices } }, { binding: 91, resource: { buffer: args } }] });
        held = { page, vertices, args, group }; this.pages.set(page.slot, held); this.revision++; dirty.push(held);
      } else if (held.page !== page || this.source !== source) { held.page = page; dirty.push(held); }
    }
    this.source = source;
    if (!dirty.length) return;
    for (const held of dirty) this.device.queue.writeBuffer(held.args, 0, new Uint32Array([0, 1, 0, 0, 0, held.page.slot, 0, 0]));
    const pass = encoder.beginComputePass({ label: "Cache changed sculpt wall triangles" });
    pass.setBindGroup(0, scene);
    pass.setPipeline(pipelines.build);
    for (const held of dirty) { pass.setBindGroup(1, held.group); pass.dispatchWorkgroups(8); }
    pass.setPipeline(pipelines.publish);
    for (const held of dirty) { pass.setBindGroup(1, held.group); pass.dispatchWorkgroups(1); }
    pass.end(); this.builtPages += dirty.length;
  }
  draw(pass: GPURenderPassEncoder | GPURenderBundleEncoder): void {
    for (const held of this.pages.values()) { pass.setVertexBuffer(0, held.vertices); pass.drawIndirect(held.args, 0); }
  }
  destroy(): void { for (const held of this.pages.values()) { held.vertices.destroy(); held.args.destroy(); } this.pages.clear(); }
}
