/**
 * GPU-resident audit views for the unified octree/power discretization.
 *
 * Slice axes (debug.x = 1..3) sample one point per pixel.  The volume axis
 * (debug.x = 4) walks the authoritative owner texture from leaf exit to leaf
 * exit, so it visualizes the adaptive volume without a CPU-built proxy mesh or
 * a stack of diagnostic slices.
 */

import { OCTREE_GENERATED_POWER_CATALOG_MANIFEST } from "./generated/octree-power-catalog";
import type { OctreeTechniqueDebugSource } from "./octree-technique-debug";
import { CAMERA_TAN_HALF_FOV } from "./webgpu-camera";

const auditSharedWGSL = /* wgsl */ `
struct Uniforms {
  viewport:vec4f, cameraPosition:vec4f, cameraTarget:vec4f, container:vec4f,
  options:vec4f, gridInfo:vec4f, debug:vec4f, environment:vec4f,
}
struct VertexOutput { @builtin(position) position:vec4f, @location(0) uv:vec2f }
struct CameraRay { origin:vec3f, direction:vec3f }
struct LeafHeader {
  cell:u32, entryStart:u32, entryCount:u32, size:u32,
  diagonal:f32, rhs:f32, pad0:u32, pad1:u32, gradient:vec4f,
}
struct AuditSample { color:vec3f, opacity:f32 }

@vertex fn vertexMain(@builtin(vertex_index) index:u32)->VertexOutput {
  var positions=array<vec2f,3>(vec2f(-1.0,-1.0),vec2f(3.0,-1.0),vec2f(-1.0,3.0));
  var result:VertexOutput;
  result.position=vec4f(positions[index],0.0,1.0);
  result.uv=positions[index]*0.5+0.5;
  return result;
}
fn finite(value:f32)->bool { return value==value&&abs(value)<=3.402823e38; }
fn worldMinimum()->vec3f { return vec3f(-0.5*u.container.x,0.0,-0.5*u.container.z); }
fn cellCoord(cell:u32)->vec3u {
  let dims=vec3u(max(u.gridInfo.xyz,vec3f(1.0)));
  return vec3u(cell%dims.x,(cell/dims.x)%dims.y,cell/(dims.x*dims.y));
}
fn worldToFine(point:vec3f)->vec3f { return (point-worldMinimum())/u.container.xyz*u.gridInfo.xyz; }
fn fineToWorld(point:vec3f)->vec3f { return worldMinimum()+point/u.gridInfo.xyz*u.container.xyz; }
fn cameraRay(uv:vec2f)->CameraRay {
  let ndc=uv*2.0-1.0;
  let origin=u.cameraPosition.xyz;
  let forward=normalize(u.cameraTarget.xyz-origin);
  var right=cross(forward,vec3f(0.0,1.0,0.0));
  if(length(right)<1e-5){right=vec3f(1.0,0.0,0.0);}
  right=normalize(right);
  let up=normalize(cross(right,forward));
  let direction=normalize(forward+right*ndc.x*u.viewport.x/max(u.viewport.y,1.0)*${CAMERA_TAN_HALF_FOV}+up*ndc.y*${CAMERA_TAN_HALF_FOV});
  return CameraRay(origin,direction);
}
fn boxInterval(ray:CameraRay,minimum:vec3f,maximum:vec3f)->vec2f {
  let inverse=1.0/select(vec3f(1e-20),ray.direction,abs(ray.direction)>vec3f(1e-20));
  let a=(minimum-ray.origin)*inverse;
  let b=(maximum-ray.origin)*inverse;
  let lo=min(a,b);
  let hi=max(a,b);
  return vec2f(max(max(lo.x,lo.y),max(lo.z,0.0)),min(min(hi.x,hi.y),hi.z));
}
fn slicePoint(uv:vec2f)->vec4f {
  let axis=i32(round(u.debug.x));
  if(axis<=0||axis==4){return vec4f(0.0,0.0,0.0,-1.0);}
  let ray=cameraRay(uv);
  let dims=max(u.gridInfo.xyz,vec3f(1.0));
  let minimum=worldMinimum();
  var denominator=ray.direction.z;
  var rayOrigin=ray.origin.z;
  var coordinate=minimum.z+(floor(clamp(u.debug.y,0.0,0.999999)*dims.z)+0.5)*u.container.z/dims.z;
  if(axis==2){
    denominator=ray.direction.x;rayOrigin=ray.origin.x;
    coordinate=minimum.x+(floor(clamp(u.debug.y,0.0,0.999999)*dims.x)+0.5)*u.container.x/dims.x;
  }
  if(axis==3){
    denominator=ray.direction.y;rayOrigin=ray.origin.y;
    coordinate=(floor(clamp(u.debug.y,0.0,0.999999)*dims.y)+0.5)*u.container.y/dims.y;
  }
  if(abs(denominator)<=1e-6){return vec4f(0.0,0.0,0.0,-1.0);}
  let distance=(coordinate-rayOrigin)/denominator;
  return vec4f(ray.origin+ray.direction*distance,distance);
}
fn headerContains(header:LeafHeader,pointFine:vec3f)->bool {
  let dims=vec3u(max(u.gridInfo.xyz,vec3f(1.0)));
  if(header.size==0u||header.cell>=dims.x*dims.y*dims.z){return false;}
  let origin=cellCoord(header.cell);
  return all(origin%vec3u(header.size)==vec3u(0u))
    &&all(origin+vec3u(header.size)<=dims)
    &&all(pointFine>=vec3f(origin))&&all(pointFine<vec3f(origin+vec3u(header.size)));
}
fn displayColor(linear:vec3f)->vec3f {
  let mapped=linear/(linear+vec3f(1.0));
  return pow(max(mapped,vec3f(0.0)),vec3f(1.0/2.2));
}
fn heat(value:f32)->vec3f {
  let t=clamp(value,0.0,1.0);
  if(t<0.5){return mix(vec3f(0.04,0.20,0.70),vec3f(0.06,0.78,0.55),t*2.0);}
  return mix(vec3f(0.06,0.78,0.55),vec3f(1.0,0.08,0.025),(t-0.5)*2.0);
}
fn signedHeat(value:f32)->vec3f {
  let magnitude=clamp(abs(value),0.0,1.0);
  let neutral=vec3f(0.12,0.58,0.42);
  return mix(neutral,select(vec3f(0.08,0.34,1.0),vec3f(1.0,0.08,0.04),value>=0.0),magnitude);
}
fn invalidSample()->AuditSample { return AuditSample(displayColor(vec3f(1.0,0.0,0.22)),0.96); }
fn ownerAt(point:vec3f)->u32 {
  let cell=vec3i(clamp(floor(worldToFine(point)),vec3f(0.0),u.gridInfo.xyz-vec3f(1.0)));
  return textureLoad(ownerRows,cell,0).x;
}
`;

export const octreeTechniqueOperatorAuditShader = /* wgsl */ `
${auditSharedWGSL}
struct PowerFace {
  negativeRow:u32, positiveRow:u32, geometryCode:u32, flags:u32,
  normalVelocity:f32, area:f32, inverseDistance:f32, openFraction:f32,
}
struct RowWork { faceCount:u32, incidenceCount:u32, faceOffset:u32, incidenceOffset:u32 }
struct Incidence { face:u32, sign:i32 }
@group(0) @binding(0) var<uniform> u:Uniforms;
@group(0) @binding(1) var ownerRows:texture_3d<u32>;
@group(0) @binding(2) var<storage,read> headers:array<LeafHeader>;
@group(0) @binding(3) var<storage,read> faces:array<PowerFace>;
@group(0) @binding(4) var<storage,read> rows:array<RowWork>;
@group(0) @binding(5) var<storage,read> incidences:array<Incidence>;
@group(0) @binding(6) var<storage,read> control:array<u32>;
const INVALID:u32=0xffffffffu;
const FACE_VALID:u32=0x80000000u;

fn reciprocalError(row:u32,work:RowWork)->vec2f {
  var checked=0u;
  var failed=0u;
  let count=min(work.incidenceCount,48u);
  for(var local=0u;local<count;local+=1u){
    let offset=work.incidenceOffset+local;
    if(offset>=arrayLength(&incidences)){failed+=1u;continue;}
    let incidence=incidences[offset];
    if(incidence.face>=arrayLength(&faces)){failed+=1u;continue;}
    let face=faces[incidence.face];
    checked+=1u;
    let endpoint=face.negativeRow==row||face.positiveRow==row;
    let expected=select(-1,1,face.negativeRow==row);
    if((face.flags&FACE_VALID)==0u||!endpoint||incidence.sign!=expected){failed+=1u;continue;}
    let other=select(face.negativeRow,face.positiveRow,face.negativeRow==row);
    if(other==INVALID){continue;}
    if(other>=arrayLength(&rows)){failed+=1u;continue;}
    let reverse=rows[other];
    if(reverse.incidenceOffset>arrayLength(&incidences)||reverse.incidenceCount>arrayLength(&incidences)-reverse.incidenceOffset){failed+=1u;continue;}
    var found=false;
    for(var candidate=0u;candidate<min(reverse.incidenceCount,48u);candidate+=1u){
      let reverseItem=incidences[reverse.incidenceOffset+candidate];
      found=found||(reverseItem.face==incidence.face&&reverseItem.sign==-incidence.sign);
    }
    if(!found){failed+=1u;}
  }
  return vec2f(f32(failed),f32(checked));
}

fn auditRow(row:u32,volume:bool)->AuditSample {
  if(arrayLength(&control)<=8u||control[8]!=FACE_VALID||row>=arrayLength(&headers)||row>=arrayLength(&rows)){return invalidSample();}
  let header=headers[row];
  let mode=i32(round(u.debug.w));
  let sliceOpacity=0.80;
  if(mode==19){
    if(!finite(header.diagonal)||header.diagonal<=0.0){return invalidSample();}
    let scaled=1.0-exp(-header.diagonal/max(f32(header.size),1.0));
    return AuditSample(displayColor(heat(scaled)),select(sliceOpacity,0.075+0.18*scaled,volume));
  }
  if(mode==20){
    if(!finite(header.rhs)||!finite(header.diagonal)){return invalidSample();}
    let ratio=header.rhs/max(abs(header.diagonal),1e-7);
    let scaled=ratio/(1.0+abs(ratio));
    return AuditSample(displayColor(signedHeat(scaled)),select(sliceOpacity,0.055+0.24*abs(scaled),volume));
  }
  let work=rows[row];
  if(work.incidenceOffset>arrayLength(&incidences)||work.incidenceCount>arrayLength(&incidences)-work.incidenceOffset){return invalidSample();}
  if(mode==21){
    let audit=reciprocalError(row,work);
    let error=select(1.0,audit.x/max(audit.y,1.0),audit.y>0.0);
    let color=mix(vec3f(0.04,0.72,0.42),vec3f(1.0,0.015,0.08),clamp(error*4.0,0.0,1.0));
    return AuditSample(displayColor(color),select(sliceOpacity,0.065+0.35*min(1.0,error*8.0),volume));
  }
  if(mode==22){
    var area=0.0;
    var openArea=0.0;
    for(var local=0u;local<min(work.incidenceCount,48u);local+=1u){
      let item=incidences[work.incidenceOffset+local];
      if(item.face>=arrayLength(&faces)){return invalidSample();}
      let face=faces[item.face];
      if((face.flags&FACE_VALID)==0u||!finite(face.area)||!finite(face.openFraction)||face.area<0.0||face.openFraction<0.0||face.openFraction>1.0){return invalidSample();}
      area+=face.area;
      openArea+=face.area*face.openFraction;
    }
    if(area<=0.0){return invalidSample();}
    let openness=clamp(openArea/area,0.0,1.0);
    let blocked=1.0-openness;
    let color=mix(vec3f(0.04,0.72,0.86),vec3f(1.0,0.06,0.20),blocked);
    return AuditSample(displayColor(color),select(sliceOpacity,0.055+0.28*blocked,volume));
  }
  return AuditSample(vec3f(0.0),0.0);
}

@fragment fn fragmentMain(input:VertexOutput)->@location(0) vec4f {
  let minimum=worldMinimum();
  let maximum=minimum+u.container.xyz;
  let volume=i32(round(u.debug.x))==4;
  if(!volume){
    let hit=slicePoint(input.uv);
    if(hit.w<=0.0||any(hit.xyz<minimum)||any(hit.xyz>maximum)){discard;}
    let row=ownerAt(hit.xyz);
    if(row==INVALID||row>=arrayLength(&headers)||!headerContains(headers[row],worldToFine(hit.xyz))){return vec4f(invalidSample().color,0.94);}
    let sample=auditRow(row,false);
    if(sample.opacity<=0.0){discard;}
    return vec4f(sample.color,sample.opacity);
  }
  let ray=cameraRay(input.uv);
  let interval=boxInterval(ray,minimum,maximum);
  if(interval.y<=interval.x){discard;}
  let minimumStep=max(1e-5,min(min(u.container.x/u.gridInfo.x,u.container.y/u.gridInfo.y),u.container.z/u.gridInfo.z)*0.08);
  var distance=interval.x+minimumStep;
  var accumulated=vec4f(0.0);
  let visitLimit=min(1024u,u32(ceil(u.gridInfo.x+u.gridInfo.y+u.gridInfo.z))+4u);
  for(var step=0u;step<visitLimit&&distance<interval.y&&accumulated.a<0.94;step+=1u){
    let point=ray.origin+ray.direction*distance;
    let row=ownerAt(point);
    var nextDistance=distance+minimumStep;
    if(row!=INVALID&&row<arrayLength(&headers)){
      let header=headers[row];
      let pointFine=worldToFine(point);
      if(headerContains(header,pointFine)){
        let origin=cellCoord(header.cell);
        let leafInterval=boxInterval(ray,fineToWorld(vec3f(origin)),fineToWorld(vec3f(origin+vec3u(header.size))));
        nextDistance=max(nextDistance,leafInterval.y+minimumStep);
        let sample=auditRow(row,true);
        let contribution=(1.0-accumulated.a)*sample.opacity*clamp(u.debug.y,0.05,1.0);
        accumulated=vec4f(accumulated.rgb+sample.color*contribution,accumulated.a+contribution);
      }
    }
    distance=nextDistance;
  }
  if(accumulated.a<=0.002){discard;}
  return vec4f(accumulated.rgb/accumulated.a,accumulated.a);
}`;

export const octreeTechniqueTetraValidityShader = /* wgsl */ `
${auditSharedWGSL}
struct Metric { topologyCode:u32, transformAndFlags:u32, volume:f32, reserved:u32 }
struct TetraHeader { first:u32, count:u32, flags:u32 }
struct TetraVertex { value:vec4f }
@group(0) @binding(0) var<uniform> u:Uniforms;
@group(0) @binding(1) var ownerRows:texture_3d<u32>;
@group(0) @binding(2) var<storage,read> headers:array<LeafHeader>;
@group(0) @binding(3) var<storage,read> metrics:array<Metric>;
@group(0) @binding(4) var<storage,read> tetraHeaders:array<TetraHeader>;
@group(0) @binding(5) var<storage,read> tetrahedra:array<u32>;
@group(0) @binding(6) var<storage,read> tetraVertices:array<TetraVertex>;
const INVALID:u32=0xffffffffu;
const TOPOLOGY_VALID:u32=0x80000000u;

fn auditRow(row:u32,volume:bool)->AuditSample {
  if(row>=arrayLength(&headers)||row>=arrayLength(&metrics)){return invalidSample();}
  let metric=metrics[row];
  if((metric.transformAndFlags&TOPOLOGY_VALID)==0u||metric.topologyCode>=arrayLength(&tetraHeaders)||!finite(metric.volume)||metric.volume<=0.0){return invalidSample();}
  let header=tetraHeaders[metric.topologyCode];
  if(header.count==0u||header.first>arrayLength(&tetrahedra)||header.count>arrayLength(&tetrahedra)-header.first){return invalidSample();}
  var reconstructed=0.0;
  var negative=0u;
  var invalid=0u;
  let count=min(header.count,${OCTREE_GENERATED_POWER_CATALOG_MANIFEST.maximumTetrahedra}u);
  for(var local=0u;local<count;local+=1u){
    let packed=tetrahedra[header.first+local];
    let selectors=vec3u(packed&255u,(packed>>8u)&255u,(packed>>16u)&255u);
    if(any(selectors>=vec3u(arrayLength(&tetraVertices)))||selectors.x==selectors.y||selectors.x==selectors.z||selectors.y==selectors.z){invalid+=1u;continue;}
    let a=tetraVertices[selectors.x].value.xyz;
    let b=tetraVertices[selectors.y].value.xyz;
    let c=tetraVertices[selectors.z].value.xyz;
    let determinant=dot(a,cross(b,c));
    if(!finite(determinant)||abs(determinant)<=1e-9){invalid+=1u;continue;}
    reconstructed+=abs(determinant)/6.0;
    negative+=select(0u,1u,determinant<0.0);
  }
  if(header.count>${OCTREE_GENERATED_POWER_CATALOG_MANIFEST.maximumTetrahedra}u){invalid+=header.count-${OCTREE_GENERATED_POWER_CATALOG_MANIFEST.maximumTetrahedra}u;}
  let mismatch=abs(reconstructed-metric.volume)/max(metric.volume,1e-8);
  if(!finite(reconstructed)||!finite(mismatch)){return invalidSample();}
  let failure=clamp(max(f32(invalid)/f32(max(header.count,1u)),mismatch*32.0),0.0,1.0);
  let handedness=f32(negative)/f32(max(count-invalid,1u));
  let validColor=mix(vec3f(0.02,0.78,0.48),vec3f(0.10,0.38,0.98),handedness);
  let color=mix(validColor,vec3f(1.0,0.015,0.10),failure);
  let opacity=select(0.82,0.06+0.34*failure,volume);
  return AuditSample(displayColor(color),opacity);
}

@fragment fn fragmentMain(input:VertexOutput)->@location(0) vec4f {
  let minimum=worldMinimum();
  let maximum=minimum+u.container.xyz;
  let volume=i32(round(u.debug.x))==4;
  if(!volume){
    let hit=slicePoint(input.uv);
    if(hit.w<=0.0||any(hit.xyz<minimum)||any(hit.xyz>maximum)){discard;}
    let row=ownerAt(hit.xyz);
    if(row==INVALID||row>=arrayLength(&headers)||!headerContains(headers[row],worldToFine(hit.xyz))){return vec4f(invalidSample().color,0.94);}
    let sample=auditRow(row,false);
    return vec4f(sample.color,sample.opacity);
  }
  let ray=cameraRay(input.uv);
  let interval=boxInterval(ray,minimum,maximum);
  if(interval.y<=interval.x){discard;}
  let minimumStep=max(1e-5,min(min(u.container.x/u.gridInfo.x,u.container.y/u.gridInfo.y),u.container.z/u.gridInfo.z)*0.08);
  var distance=interval.x+minimumStep;
  var accumulated=vec4f(0.0);
  let visitLimit=min(1024u,u32(ceil(u.gridInfo.x+u.gridInfo.y+u.gridInfo.z))+4u);
  for(var step=0u;step<visitLimit&&distance<interval.y&&accumulated.a<0.94;step+=1u){
    let point=ray.origin+ray.direction*distance;
    let row=ownerAt(point);
    var nextDistance=distance+minimumStep;
    if(row!=INVALID&&row<arrayLength(&headers)){
      let header=headers[row];
      if(headerContains(header,worldToFine(point))){
        let origin=cellCoord(header.cell);
        let leafInterval=boxInterval(ray,fineToWorld(vec3f(origin)),fineToWorld(vec3f(origin+vec3u(header.size))));
        nextDistance=max(nextDistance,leafInterval.y+minimumStep);
        let sample=auditRow(row,true);
        let contribution=(1.0-accumulated.a)*sample.opacity*clamp(u.debug.y,0.05,1.0);
        accumulated=vec4f(accumulated.rgb+sample.color*contribution,accumulated.a+contribution);
      }
    }
    distance=nextDistance;
  }
  if(accumulated.a<=0.002){discard;}
  return vec4f(accumulated.rgb/accumulated.a,accumulated.a);
}`;

export class OctreeTechniqueAuditOverlayPipeline {
  private operatorPipeline?: GPURenderPipeline;
  private tetraPipeline?: GPURenderPipeline;
  private source?: OctreeTechniqueDebugSource;
  private ownerRows?: GPUTexture;
  private operatorGroup?: GPUBindGroup;
  private tetraGroup?: GPUBindGroup;

  constructor(
    private readonly device: GPUDevice,
    private readonly targetFormat: GPUTextureFormat,
    private readonly uniformBuffer: GPUBuffer,
  ) {}

  private async createPipeline(label: string, code: string): Promise<GPURenderPipeline> {
    const shaderModule=this.device.createShaderModule({label,code});
    const compilation=await shaderModule.getCompilationInfo();
    const errors=compilation.messages.filter((message)=>message.type==="error");
    if(errors.length>0)throw new Error(errors.map((error)=>`${error.lineNum}:${error.linePos} ${error.message}`).join("\n"));
    return this.device.createRenderPipelineAsync({
      label,
      layout:"auto",
      vertex:{module:shaderModule,entryPoint:"vertexMain"},
      fragment:{
        module:shaderModule,
        entryPoint:"fragmentMain",
        targets:[{format:this.targetFormat,blend:{
          color:{srcFactor:"src-alpha",dstFactor:"one-minus-src-alpha"},
          alpha:{srcFactor:"one",dstFactor:"one-minus-src-alpha"},
        }}],
      },
      primitive:{topology:"triangle-list"},
    });
  }

  async initialize(): Promise<void> {
    [this.operatorPipeline,this.tetraPipeline]=await Promise.all([
      this.createPipeline("Octree power-operator audit overlay",octreeTechniqueOperatorAuditShader),
      this.createPipeline("Octree tetrahedron-validity audit overlay",octreeTechniqueTetraValidityShader),
    ]);
    this.rebuildGroups();
  }

  setSource(source: OctreeTechniqueDebugSource | undefined): void {
    if(this.source===source)return;
    this.source=source;
    this.rebuildGroups();
  }

  setOwnerRows(ownerRows: GPUTexture | undefined): void {
    if(this.ownerRows===ownerRows)return;
    this.ownerRows=ownerRows;
    this.rebuildGroups();
  }

  private rebuildGroups(): void {
    const source=this.source,ownerRows=this.ownerRows;
    if(!source||!ownerRows)return;
    if(this.operatorPipeline)this.operatorGroup=this.device.createBindGroup({
      layout:this.operatorPipeline.getBindGroupLayout(0),
      entries:[
        {binding:0,resource:{buffer:this.uniformBuffer}},
        {binding:1,resource:ownerRows.createView({dimension:"3d"})},
        {binding:2,resource:source.leafHeaders},
        {binding:3,resource:source.powerFaces},
        {binding:4,resource:source.incidenceRows},
        {binding:5,resource:source.incidence},
        {binding:6,resource:source.faceControl},
      ],
    });
    if(this.tetraPipeline)this.tetraGroup=this.device.createBindGroup({
      layout:this.tetraPipeline.getBindGroupLayout(0),
      entries:[
        {binding:0,resource:{buffer:this.uniformBuffer}},
        {binding:1,resource:ownerRows.createView({dimension:"3d"})},
        {binding:2,resource:source.leafHeaders},
        {binding:3,resource:source.topologyMetrics},
        {binding:4,resource:source.tetrahedronHeaders},
        {binding:5,resource:source.tetrahedra},
        {binding:6,resource:source.tetrahedronVertices},
      ],
    });
  }

  encode(encoder: GPUCommandEncoder, target: GPUTextureView, modeCode: number): boolean {
    const tetra=modeCode===23;
    const pipeline=tetra?this.tetraPipeline:this.operatorPipeline;
    const group=tetra?this.tetraGroup:this.operatorGroup;
    if(!pipeline||!group||modeCode<19||modeCode>23)return false;
    const pass=encoder.beginRenderPass({
      label:"Octree paper-technique audit overlay",
      colorAttachments:[{view:target,loadOp:"load",storeOp:"store"}],
    });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0,group);
    pass.draw(3);
    pass.end();
    return true;
  }
}
