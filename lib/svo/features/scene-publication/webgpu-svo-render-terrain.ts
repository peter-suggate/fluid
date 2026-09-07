import type { SceneDescription } from "../../../core/model";
import { solidWorldVoxelPatchBounds_m } from "../../../core/solid-world";
import { buildPondVesselPlanIndex, pondVesselPlanCurve, pondVesselFloorDishReach, POND_VESSEL_FLOOR_DISH, type WallProfile } from "../../../core/voxel-scenery/pond-vessel";
import type { SvoRenderTerrainField } from "./svo-render-solid-field";

const f=(v:number)=>{if(!Number.isFinite(v))throw new RangeError("Nonfinite terrain parameter");return Number.isInteger(v)?`${v}.0`:String(v);};
const v2=(v:readonly number[])=>`vec2f(${f(v[0])},${f(v[1])})`;
const wall=(p:WallProfile)=>`vec3f(${f(p.crestRadius_m)},${f(p.footRadius_m)},${f(p.batter_rad)})`;
function hashSigned(n:number){let h=Math.imul(n^0x9e3779b9,0x85ebca6b)>>>0;h=Math.imul(h^(h>>>13),0xc2b2ae35)>>>0;return 2*((h^(h>>>16))>>>0)/2**32-1;}

/** The authoring polyline and scalar spec are host metadata. Every height
 * sample, including the procedural lattice's bilinear reconstruction, is GPU
 * work. The returned f32 field is also the CPU refinement oracle's input. */
export function svoPondTerrainProgram(scene: SceneDescription,cellSize:readonly [number,number,number]) {
  const procedural=scene.terrain?.procedural;
  if(scene.terrain?.grid||procedural?.kind!=="pond-vessel")return undefined;
  const {spec,container,spacing_m}=procedural;
  const curve=pondVesselPlanCurve(spec),index=buildPondVesselPlanIndex(curve);
  const data:number[]=[];
  const append=(values:ArrayLike<number>)=>{const base=data.length;for(let i=0;i<values.length;i++)data.push(values[i]>>>0);return base;};
  const points=append(new Uint32Array(new Float32Array(curve.flat()).buffer));
  const cells=append(index.cellStart),items=append(index.cellItems),bands=append(index.bandStart),bandItems=append(index.bandItems);
  const nx=Math.max(1,Math.round(scene.container.width_m/cellSize[0])),nz=Math.max(1,Math.round(scene.container.depth_m/cellSize[2]));
  const gridNx=Math.max(2,Math.ceil(container.width_m/spacing_m)+1),gridNz=Math.max(2,Math.ceil(container.depth_m/spacing_m)+1);
  const spline=(salt:number)=>`array<f32,${spec.lobes+4}>(${Array.from({length:spec.lobes+4},(_,i)=>f(hashSigned(spec.seed+salt+61*i))).join(",")})`;
  const terraces=(spec.terraces??[]).map((t,i)=>{
    const rotation=t.rotation_rad??0,inner=t.faceRun_m===undefined?t.flat??0.45:
      Math.max(0,1-Math.min(0.95,Math.max(1e-3,t.faceRun_m/Math.sqrt(t.radius_m[0]*t.radius_m[1]))));
    return `fn terrace${i}(p:vec2f)->f32{
      let d=p-${v2(t.center_m)};let q=vec2f(${f(Math.cos(rotation))}*d.x+${f(Math.sin(rotation))}*d.y,-${f(Math.sin(rotation))}*d.x+${f(Math.cos(rotation))}*d.y);
      ${t.wall?`let wander=1.0+${f(t.wobble??0)}*sin(${f(t.lobes??5)}*atan2(q.y,q.x)+0.7);
      let r=max(vec2f(1e-4),${v2(t.radius_m)}*wander);let radius=length(q/r);let gradient=length(q/(r*r));
      var distance=(radius-1.0)*sqrt(r.x*r.y);if(gradient>1e-9){distance=(radius-1.0)*radius/gradient;}
      return wallHeight(${wall(t.wall)},${f(t.height_m)},distance);`
      :`let distance=length(q/${v2(t.radius_m)});return ${f(t.height_m)}*ramp((1.0-distance)/(1.0-${f(inner)}));`}
    }`;
  }).join("\n");
  const shader=/* wgsl */ `
@group(0) @binding(0) var<storage,read> data:array<u32>;
@group(0) @binding(1) var<storage,read_write> heights:array<f32>;
@group(0) @binding(2) var<uniform> batch:vec4u;
fn point(i:u32)->vec2f{let b=${points}u+2u*i;return bitcast<vec2f>(vec2u(data[b],data[b+1u]));}
fn distanceSquared(p:vec2f,i:u32)->f32{let a=point(i);let d=point((i+1u)%${curve.length}u)-a;let t=clamp(dot(p-a,d)/max(dot(d,d),1e-30),0.0,1.0);let q=p-a-t*d;return dot(q,q);}
fn planDistance(p:vec2f)->f32{
  let cell=vec2i(floor((p-${v2([index.gridOriginX,index.gridOriginZ])})*${f(index.gridInversePitch)}));var nearest=3.402823e38;
  if(all(cell>=vec2i(0))&&cell.x<${index.gridNx}&&cell.y<${index.gridNz}){
    let c=u32(cell.x+${index.gridNx}*cell.y);for(var slot=data[${cells}u+c];slot<data[${cells}u+c+1u];slot++){nearest=min(nearest,distanceSquared(p,data[${items}u+slot]));}
  }else{for(var i=0u;i<${curve.length}u;i++){nearest=min(nearest,distanceSquared(p,i));}}
  var inside=false;let band=i32(floor((p.y-(${f(index.bandOriginZ)}))*${f(index.bandInversePitch)}));
  if(band>=0&&band<${index.bandCount}){for(var slot=data[${bands}u+u32(band)];slot<data[${bands}u+u32(band)+1u];slot++){
    let i=data[${bandItems}u+slot];let a=point(i);let b=point((i+1u)%${curve.length}u);
    if((a.y>p.y)!=(b.y>p.y)){if(p.x<a.x+(p.y-a.y)/(b.y-a.y)*(b.x-a.x)){inside=!inside;}}
  }}return select(1.0,-1.0,inside)*sqrt(nearest);
}
fn ramp(v:f32)->f32{let t=clamp(v,0.0,1.0);return t*t*(3.0-2.0*t);}
fn innerFace(v:f32)->f32{let t=clamp(v,0.0,1.0);if(t<=0.65){return t/${f(0.825)};}let q=(t-0.65)/0.35;return (0.65+0.35*(q-0.5*q*q))/${f(0.825)};}
fn spline(values:array<f32,${spec.lobes+4}>,turn:f32)->f32{
  let position=fract(turn)*${f(spec.lobes+4)};let i=u32(floor(position));let t=fract(position);let t2=t*t;let t3=t2*t;
  let a=values[(i+${spec.lobes+3}u)%${spec.lobes+4}u];let b=values[i];let c=values[(i+1u)%${spec.lobes+4}u];let d=values[(i+2u)%${spec.lobes+4}u];
  return 0.5*(2.0*b+(c-a)*t+(2.0*a-5.0*b+4.0*c-d)*t2+(3.0*b-3.0*c+d-a)*t3);
}
fn wallHeight(profile:vec3f,rise:f32,d:f32)->f32{
  let batter=max(0.02,profile.z);let c=cos(batter);let s=sin(batter);let t=tan(batter);
  let appetite=(profile.x+profile.y)*(1.0-s);let fit=min(1.0,rise/max(appetite,1e-30));
  let crest=profile.x*fit;let foot=profile.y*fit;let centreY=rise-crest;let tangentD=crest*c;let tangentY=centreY+crest*s;
  let footD=tangentD+(tangentY-foot*(1.0-s))*t;let centreD=footD+foot*c;
  if(d<=0.0){return rise;}if(d>=centreD){return 0.0;}if(d<=tangentD){return centreY+sqrt(max(0.0,crest*crest-d*d));}
  if(d>=footD){let dx=d-centreD;return foot-sqrt(max(0.0,foot*foot-dx*dx));}return tangentY-(d-tangentD)/t;
}
fn wallRun(profile:vec3f,rise:f32)->f32{
  let batter=max(0.02,profile.z);let c=cos(batter);let s=sin(batter);let t=tan(batter);
  let fit=min(1.0,rise/max((profile.x+profile.y)*(1.0-s),1e-30));let crest=profile.x*fit;let foot=profile.y*fit;
  let end=crest*c+(rise-crest+crest*s-foot*(1.0-s))*t+foot*c;
  // Match the CPU's 0.5 mm stepping without f32 cancellation in foot - sqrt
  // causing an early zero before the true endpoint.
  return ceil(end/0.0005)*0.0005;
}
fn hashSigned(n:u32)->f32{var h=(n^0x9e3779b9u)*0x85ebca6bu;h=(h^(h>>13u))*0xc2b2ae35u;return 2.0*f32(h^(h>>16u))/4294967296.0-1.0;}
fn noiseCorner(seed:u32,p:vec2i)->f32{return hashSigned(seed+73856093u*bitcast<u32>(p.x)+19349663u*bitcast<u32>(p.y));}
fn relief(p:vec2f)->f32{
  var total=0.0;var amplitude=1.0;var frequency=26.0;var weight=0.0;
  for(var octave=0u;octave<2u;octave++){
    let q=p*frequency;let c=vec2i(floor(q));let v=fract(q);let s=v*v*(3.0-2.0*v);let seed=${(spec.seed^0x51ed2701)>>>0}u+0x9e37u*octave;
    let lower=noiseCorner(seed,c)*(1.0-s.x)+noiseCorner(seed,c+vec2i(1,0))*s.x;
    let upper=noiseCorner(seed,c+vec2i(0,1))*(1.0-s.x)+noiseCorner(seed,c+vec2i(1,1))*s.x;
    total+=amplitude*(lower*(1.0-s.y)+upper*s.y);weight+=amplitude;amplitude*=0.45;frequency*=2.7;
  }return total/weight;
}
${terraces}
fn heightAt(p:vec2f)->f32{
  let distance=planDistance(p);let q=p-${v2(spec.center_m)};let turn=atan2(q.y,q.x)/${f(2*Math.PI)};
  let rimHeight=${f(spec.rimHeight_m)}*(1.0+${f(spec.sectionHeightVariation)}*spline(${spline(0)},turn));
  let rimWidth=${f(spec.rimHalfWidth_m)}*(1.0+${f(spec.sectionWidthVariation)}*spline(${spline(977)},turn));
  let inward=-distance-rimWidth;
  var faceRun=${f(spec.innerFace_m)};
  ${spec.beach&&spec.beach.width>0?`faceRun+=${f(spec.beach.innerFace_m-spec.innerFace_m)}*ramp(1.0-abs(fract(turn-(${f(spec.beach.turn)})+0.5)-0.5)/${f(spec.beach.width)});`:""}
  var fall=0.0;if(distance<0.0){fall=${f(spec.basinDepth_m)}*((1.0-${f(spec.floorDish??POND_VESSEL_FLOOR_DISH)})*innerFace(inward/faceRun)+${f(spec.floorDish??POND_VESSEL_FLOOR_DISH)}*ramp(inward/${f(pondVesselFloorDishReach(spec,curve))}));}
  let crest=${spec.crest==="flat"?"0.0":spec.crest==="wall"&&spec.crestWall?`wallHeight(${wall(spec.crestWall)},rimHeight,abs(distance)-max(0.0,rimWidth-wallRun(${wall(spec.crestWall)},rimHeight)))`:`rimHeight*pow(1.0-pow(clamp(abs(distance)/rimWidth,0.0,1.0),2.0),1.5)`};
  var lift=0.0;let outside=ramp((distance-rimWidth)/rimWidth);
  ${(spec.terraces??[]).map((_,i)=>`lift=max(lift,terrace${i}(p)*outside);`).join("\n")}
  return clamp(${f(spec.groundHeight_m)}-fall+crest+lift+${f(spec.relief_m)}*relief(p),0.0,${f(container.height_m)});
}
fn at(i:vec2u)->f32{return heightAt(${v2([-container.width_m/2,-container.depth_m/2])}+vec2f(i)*${f(spacing_m)});}
fn sample(p:vec2f)->f32{
  let q=clamp((p-${v2([-container.width_m/2,-container.depth_m/2])})/${f(spacing_m)},vec2f(0.0),vec2f(${gridNx-1}.0,${gridNz-1}.0));
  let i=min(vec2u(floor(q)),vec2u(${gridNx-2}u,${gridNz-2}u));let t=q-vec2f(i);
  var lower=at(i);if(t.x!=0.0){lower=lower*(1.0-t.x)+at(i+vec2u(1,0))*t.x;}
  if(t.y==0.0){return max(0.0,lower);}var upper=at(i+vec2u(0,1));if(t.x!=0.0){upper=upper*(1.0-t.x)+at(i+vec2u(1,1))*t.x;}
  return max(0.0,lower*(1.0-t.y)+upper*t.y);
}
@compute @workgroup_size(64)
fn bake(@builtin(global_invocation_id) id:vec3u){
  let i=batch.x+id.x;if(i>=${nx*nz}u){return;}
  let p=${v2([-scene.container.width_m/2,-scene.container.depth_m/2])}+(vec2f(f32(i%${nx}u),f32(i/${nx}u))+0.5)*${v2([cellSize[0],cellSize[2]])};
  heights[i]=clamp(sample(p),0.0,${f(scene.container.height_m)});
}
`;
  return {shader,data:new Uint32Array(data),nx,nz};
}

export async function buildSvoRenderTerrainGpu(device:GPUDevice,scene:SceneDescription,cellSize:readonly [number,number,number],materialId:number,signal?:AbortSignal):Promise<SvoRenderTerrainField|undefined>{
  const program=svoPondTerrainProgram(scene,cellSize);if(!program)return undefined;
  const check=()=>{if(signal?.aborted)throw new DOMException("GPU initialization superseded","AbortError");};check();
  const owned:GPUBuffer[]=[];
  const make=(label:string,size:number,usage:number)=>{if(size>device.limits.maxStorageBufferBindingSize)throw new RangeError(`${label} exceeds GPU storage limits`);const b=device.createBuffer({label,size,usage});owned.push(b);return b;};
  try{
    const module=device.createShaderModule({label:"GPU procedural terrain bake",code:program.shader});
    const info=await module.getCompilationInfo();const errors=info.messages.filter(m=>m.type==="error");if(errors.length)throw new Error(errors.map(m=>m.message).join("\n"));
    const pipeline=await device.createComputePipelineAsync({label:"GPU procedural terrain bake",layout:"auto",compute:{module,entryPoint:"bake"}});
    const size=program.nx*program.nz*4;
    const data=make("Terrain polyline index",program.data.byteLength,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST);device.queue.writeBuffer(data,0,program.data);
    const output=make("GPU terrain heights",size,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC);
    const batch=make("Terrain bake range",16,GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST);
    const receipt=make("Terrain height receipt",size,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST);
    const group=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[data,output,batch].map((buffer,binding)=>({binding,resource:{buffer}}))});
    for(let base=0;base<program.nx*program.nz;base+=4096){check();device.queue.writeBuffer(batch,0,new Uint32Array([base,0,0,0]));
      const encoder=device.createCommandEncoder();const pass=encoder.beginComputePass();pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroups(Math.ceil(Math.min(4096,program.nx*program.nz-base)/64));pass.end();
      device.queue.submit([encoder.finish()]);await device.queue.onSubmittedWorkDone();}
    const encoder=device.createCommandEncoder();encoder.copyBufferToBuffer(output,0,receipt,0,size);device.queue.submit([encoder.finish()]);await receipt.mapAsync(GPUMapMode.READ);check();
    const heights_m=new Float32Array(receipt.getMappedRange().slice(0));receipt.unmap();
    return {origin_m:[-scene.container.width_m/2,-scene.container.depth_m/2],cellSize_m:[cellSize[0],cellSize[2]],dimensions:[program.nx,program.nz],heights_m,materialId,
      patches:scene.solidVoxels.map(p=>{const b=solidWorldVoxelPatchBounds_m(scene,p);return {operation:p.operation,minimum_m:b.minimum,maximum_m:b.maximum,materialId:p.materialId??1};})};
  }finally{for(const b of owned)b.destroy();}
}
