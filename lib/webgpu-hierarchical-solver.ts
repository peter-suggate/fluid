import type { SceneDescription } from "./model";
import type { RigidBodyState } from "./rigid-body";
import { createGPUHierarchyLayout, GPU_BRICK_META_WORDS, GPU_BRICK_SIZE, GPU_CELL_FLOATS, GPU_CELLS_PER_BRICK, rebuildGPUHierarchyLayout, type GPUHierarchyLayout } from "./webgpu-hierarchy-layout";
import type { GPUEulerianInfo, GPURigidLoad, GPUQuality } from "./webgpu-eulerian";

const shader = /* wgsl */ `
struct Cell {
  negAlpha: vec4f,
  posPressure: vec4f,
}
struct Params {
  dimsDt: vec4f,
  pageBrick: vec4f,
  originH: vec4f,
  physical: vec4f,
  containerBodies: vec4f,
  topology: vec4f,
  boundary: vec4f,
}
struct RigidBody {
  positionShape: vec4f,
  dimensionsMass: vec4f,
  orientation: vec4f,
  linearVelocity: vec4f,
  angularVelocity: vec4f,
  inverseInertia: vec4f,
}
struct LocatedCell { index:u32, slot:u32, local:vec3u, scale:u32 }
struct FacePressure { diagonal:f32, weighted:f32 }
struct PCGCell { x:f32, r:f32, p:f32, ap:f32 }
struct BodyTerm { value:f32, diagonal:f32 }

@group(0) @binding(0) var<storage,read> cellsIn:array<Cell>;
@group(0) @binding(1) var<storage,read_write> cellsOut:array<Cell>;
@group(0) @binding(2) var<storage,read> brickMeta:array<vec4u>;
@group(0) @binding(3) var<storage,read> pageTable:array<u32>;
@group(0) @binding(4) var<storage,read> params:Params;
@group(0) @binding(5) var<storage,read> rigidBodies:array<RigidBody>;
@group(0) @binding(6) var<storage,read_write> rigidExchange:array<atomic<i32>>;
@group(0) @binding(7) var<storage,read_write> reductions:array<atomic<u32>>;
@group(0) @binding(8) var presentation: texture_storage_3d<r32float,write>;
@group(0) @binding(9) var<storage,read_write> pcg:array<PCGCell>;
var<workgroup> reductionScratch:array<f32,256>;

const INVALID:u32=0xffffffffu;
const B:u32=${GPU_BRICK_SIZE}u;
const CELLS_PER_BRICK:u32=${GPU_CELLS_PER_BRICK}u;
fn loadScalar(index:u32)->f32{return bitcast<f32>(atomicLoad(&reductions[4u+index]));}
fn storeScalar(index:u32,value:f32){atomicStore(&reductions[4u+index],bitcast<u32>(value));}

fn fineDims()->vec3u{return vec3u(params.dimsDt.xyz);}
fn pageDims()->vec3u{return vec3u(params.pageBrick.xyz);}
fn leafCount()->u32{return u32(params.topology.x);}
fn meta0(slot:u32)->vec4u{return brickMeta[slot*2u];}
fn cellIndex(slot:u32,local:vec3u)->u32{return slot*CELLS_PER_BRICK+local.x+B*(local.y+B*local.z);}
fn localFromLinear(value:u32)->vec3u{return vec3u(value%B,(value/B)%B,value/(B*B));}
fn infoFromIndex(index:u32)->LocatedCell{let slot=index/CELLS_PER_BRICK;return LocatedCell(index,slot,localFromLinear(index%CELLS_PER_BRICK),meta0(slot).w);}
fn centreFine(info:LocatedCell)->vec3f{let m=meta0(info.slot);return vec3f(m.xyz)+(vec3f(info.local)+0.5)*f32(info.scale);}
fn worldFromFine(p:vec3f)->vec3f{return params.originH.xyz+p*params.originH.w;}
fn physicalPoint(p:vec3f)->bool{let w=worldFromFine(p);return w.x>=-0.5*params.containerBodies.x&&w.x<=0.5*params.containerBodies.x&&w.y>=0.0&&w.y<=params.containerBodies.y&&w.z>=-0.5*params.containerBodies.z&&w.z<=0.5*params.containerBodies.z;}

fn locate(p:vec3f)->LocatedCell{
  if(any(p<vec3f(0.0))||any(p>=vec3f(fineDims()))){return LocatedCell(INVALID,INVALID,vec3u(0),1u);}
  let q=vec3u(floor(p));let pb=q/B;let pd=pageDims();if(any(pb>=pd)){return LocatedCell(INVALID,INVALID,vec3u(0),1u);}
  let slot=pageTable[pb.x+pd.x*(pb.y+pd.y*pb.z)];let m=meta0(slot);let local=vec3u(clamp(floor((p-vec3f(m.xyz))/f32(m.w)),vec3f(0.0),vec3f(f32(B-1u))));
  return LocatedCell(cellIndex(slot,local),slot,local,m.w);
}
fn alphaAt(p:vec3f)->f32{let q=locate(p);if(q.index==INVALID||!physicalPoint(p)||bodyIndexAt(worldFromFine(p))>=0){return 0.0;}return cellsIn[q.index].negAlpha.w;}
fn pressureAt(p:vec3f)->f32{let q=locate(p);if(q.index==INVALID){return 0.0;}return cellsIn[q.index].posPressure.w;}
fn centreVelocityAt(p:vec3f)->vec3f{let q=locate(p);if(q.index==INVALID){return vec3f(0.0);}let c=cellsIn[q.index];return 0.5*(c.negAlpha.xyz+c.posPressure.xyz);}
fn sampleVelocity(p:vec3f)->vec3f{
  // Trilinear reconstruction on the finest logical lattice. Coarse leaves are
  // intentionally repeated by locate(), which makes cross-level sampling
  // continuous once restriction/prolongation halos agree.
  let q=clamp(p-0.5,vec3f(0.0),vec3f(fineDims())-1.0);let base=floor(q);let f=fract(q);
  let a=mix(centreVelocityAt(base+vec3f(0.5)),centreVelocityAt(base+vec3f(1.5,0.5,0.5)),f.x);
  let b=mix(centreVelocityAt(base+vec3f(0.5,1.5,0.5)),centreVelocityAt(base+vec3f(1.5,1.5,0.5)),f.x);
  let c=mix(centreVelocityAt(base+vec3f(0.5,0.5,1.5)),centreVelocityAt(base+vec3f(1.5,0.5,1.5)),f.x);
  let d=mix(centreVelocityAt(base+vec3f(0.5,1.5,1.5)),centreVelocityAt(base+vec3f(1.5)),f.x);
  return mix(mix(a,b,f.y),mix(c,d,f.y),f.z);
}
fn axisVector(axis:u32)->vec3f{return select(select(vec3f(0,0,1),vec3f(0,1,0),axis==1u),vec3f(1,0,0),axis==0u);}
fn tangentA(axis:u32)->vec3f{return select(select(vec3f(1,0,0),vec3f(1,0,0),axis==1u),vec3f(0,1,0),axis==0u);}
fn tangentB(axis:u32)->vec3f{return select(select(vec3f(0,1,0),vec3f(0,0,1),axis==1u),vec3f(0,0,1),axis==0u);}
fn faceCentre(info:LocatedCell,axis:u32,sign:f32)->vec3f{return centreFine(info)+0.5*sign*f32(info.scale)*axisVector(axis);}
fn faceSubdivision(info:LocatedCell,axis:u32,sign:f32)->u32{
  let q=locate(faceCentre(info,axis,sign)+0.01*sign*axisVector(axis));if(q.index==INVALID){return 1u;}return clamp(info.scale/max(q.scale,1u),1u,2u);
}
fn faceSamplePoint(info:LocatedCell,axis:u32,sign:f32,sub:u32,a:u32,b:u32)->vec3f{
  let width=f32(info.scale)/f32(sub);let oa=(f32(a)+0.5)*width-0.5*f32(info.scale);let ob=(f32(b)+0.5)*width-0.5*f32(info.scale);
  return faceCentre(info,axis,sign)+oa*tangentA(axis)+ob*tangentB(axis);
}
fn canonicalFaceSpeed(point:vec3f,axis:u32)->f32{
  let n=axisVector(axis);let lo=locate(point-0.01*n);let hi=locate(point+0.01*n);
  if(lo.index==INVALID||hi.index==INVALID||!physicalPoint(point)){return 0.0;}
  let world=worldFromFine(point);let loBody=bodyIndexAt(world-0.01*params.originH.w*n);let hiBody=bodyIndexAt(world+0.01*params.originH.w*n);if(loBody>=0){return bodyVelocityAt(u32(loBody),world)[axis];}if(hiBody>=0){return bodyVelocityAt(u32(hiBody),world)[axis];}
  // Read the two copies of this staggered face, not the two cell-centred
  // averages. The latter can cancel an otherwise valid projected flux.
  return 0.5*(cellsIn[lo.index].posPressure[axis]+cellsIn[hi.index].negAlpha[axis]);
}
fn integratedVolumeFlux(info:LocatedCell,axis:u32,sign:f32,withAlpha:bool)->f32{
  let sub=faceSubdivision(info,axis,sign);let width=f32(info.scale)/f32(sub)*params.originH.w;let area=width*width;var total=0.0;
  for(var b:u32=0u;b<2u;b+=1u){for(var a:u32=0u;a<2u;a+=1u){if(a>=sub||b>=sub){continue;}let p=faceSamplePoint(info,axis,sign,sub,a,b);let speed=canonicalFaceSpeed(p,axis);if(withAlpha){let n=axisVector(axis);let donor=select(p+0.01*n,p-0.01*n,speed>=0.0);total+=speed*area*alphaAt(donor);}else{total+=speed*area;}}}
  return total;
}
fn outwardAlphaRate(info:LocatedCell)->f32{let xp=integratedVolumeFlux(info,0u,1.0,true);let xm=integratedVolumeFlux(info,0u,-1.0,true);let yp=integratedVolumeFlux(info,1u,1.0,true);let ym=integratedVolumeFlux(info,1u,-1.0,true);let zp=integratedVolumeFlux(info,2u,1.0,true);let zm=integratedVolumeFlux(info,2u,-1.0,true);return max(xp,0.0)+max(-xm,0.0)+max(yp,0.0)+max(-ym,0.0)+max(zp,0.0)+max(-zm,0.0);}
fn inwardAlphaRate(info:LocatedCell)->f32{let xp=integratedVolumeFlux(info,0u,1.0,true);let xm=integratedVolumeFlux(info,0u,-1.0,true);let yp=integratedVolumeFlux(info,1u,1.0,true);let ym=integratedVolumeFlux(info,1u,-1.0,true);let zp=integratedVolumeFlux(info,2u,1.0,true);let zm=integratedVolumeFlux(info,2u,-1.0,true);return max(-xp,0.0)+max(xm,0.0)+max(-yp,0.0)+max(ym,0.0)+max(-zp,0.0)+max(zm,0.0);}
fn donorScaleFor(info:LocatedCell)->f32{if(info.index==INVALID){return 0.0;}return pcg[info.index].x;}
fn receiverScaleFor(info:LocatedCell)->f32{if(info.index==INVALID){return 0.0;}return pcg[info.index].r;}
fn integratedLimitedAlphaFlux(info:LocatedCell,axis:u32,sign:f32)->f32{let sub=faceSubdivision(info,axis,sign);let width=f32(info.scale)/f32(sub)*params.originH.w;let area=width*width;var total=0.0;let n=axisVector(axis);for(var b:u32=0u;b<2u;b+=1u){for(var a:u32=0u;a<2u;a+=1u){if(a>=sub||b>=sub){continue;}let point=faceSamplePoint(info,axis,sign,sub,a,b);let speed=canonicalFaceSpeed(point,axis);let lo=locate(point-0.01*n);let hi=locate(point+0.01*n);if(lo.index==INVALID||hi.index==INVALID){continue;}var donor=hi;var receiver=lo;var donorPoint=point+0.01*n;if(speed>=0.0){donor=lo;receiver=hi;donorPoint=point-0.01*n;}total+=speed*area*alphaAt(donorPoint)*min(donorScaleFor(donor),receiverScaleFor(receiver));}}return total;}
fn divergenceVolume(info:LocatedCell)->f32{
  return integratedVolumeFlux(info,0u,1.0,false)-integratedVolumeFlux(info,0u,-1.0,false)+integratedVolumeFlux(info,1u,1.0,false)-integratedVolumeFlux(info,1u,-1.0,false)+integratedVolumeFlux(info,2u,1.0,false)-integratedVolumeFlux(info,2u,-1.0,false);
}
fn pressureFace(info:LocatedCell,axis:u32,sign:f32)->FacePressure{
  let sub=faceSubdivision(info,axis,sign);let width=f32(info.scale)/f32(sub)*params.originH.w;let area=width*width;let ownAlpha=alphaAt(centreFine(info));var diagonal=0.0;var weighted=0.0;
  for(var b:u32=0u;b<2u;b+=1u){for(var a:u32=0u;a<2u;a+=1u){if(a>=sub||b>=sub){continue;}let p=faceSamplePoint(info,axis,sign,sub,a,b);let n=axisVector(axis);let neighbor=locate(p+0.01*sign*n);if(neighbor.index==INVALID||!physicalPoint(p)){continue;}if(bodyIndexAt(worldFromFine(p+0.01*sign*n))>=0){continue;}let otherAlpha=alphaAt(p+0.01*sign*n);let distance=0.5*f32(info.scale+neighbor.scale)*params.originH.w;var coefficient=area/max(distance,1e-8);if(otherAlpha<0.01){let theta=clamp(ownAlpha/max(ownAlpha-otherAlpha,1e-6),0.05,1.0);coefficient/=theta;}diagonal+=coefficient;if(otherAlpha>=0.01){weighted+=coefficient*cellsIn[neighbor.index].posPressure.w;}}}
  return FacePressure(diagonal,weighted);
}
fn vectorValueAt(point:vec3f,component:u32)->f32{let q=locate(point);if(q.index==INVALID){return 0.0;}let value=pcg[q.index];if(component==0u){return value.x;}if(component==1u){return value.r;}if(component==2u){return value.p;}return value.ap;}
fn operatorFace(info:LocatedCell,axis:u32,sign:f32,component:u32)->FacePressure{
  let sub=faceSubdivision(info,axis,sign);let width=f32(info.scale)/f32(sub)*params.originH.w;let area=width*width;let ownAlpha=alphaAt(centreFine(info));var diagonal=0.0;var weighted=0.0;
  for(var b:u32=0u;b<2u;b+=1u){for(var a:u32=0u;a<2u;a+=1u){if(a>=sub||b>=sub){continue;}let point=faceSamplePoint(info,axis,sign,sub,a,b);let n=axisVector(axis);let neighbor=locate(point+0.01*sign*n);if(neighbor.index==INVALID||!physicalPoint(point)){continue;}if(bodyIndexAt(worldFromFine(point+0.01*sign*n))>=0){continue;}let otherAlpha=alphaAt(point+0.01*sign*n);let distance=0.5*f32(info.scale+neighbor.scale)*params.originH.w;var coefficient=area/max(distance,1e-8);if(otherAlpha<0.01){let theta=clamp(ownAlpha/max(ownAlpha-otherAlpha,1e-6),0.05,1.0);coefficient/=theta;}diagonal+=coefficient;if(otherAlpha>=0.01){weighted+=coefficient*vectorValueAt(point+0.01*sign*n,component);}}}
  return FacePressure(diagonal,weighted);
}
fn diagonalAt(info:LocatedCell)->f32{var diagonal=0.0;for(var axis:u32=0u;axis<3u;axis+=1u){diagonal+=operatorFace(info,axis,-1.0,0u).diagonal+operatorFace(info,axis,1.0,0u).diagonal;}return diagonal+bodyOperatorTerm(info).diagonal;}
fn applyOperator(info:LocatedCell,component:u32)->f32{var diagonal=0.0;var weighted=0.0;for(var axis:u32=0u;axis<3u;axis+=1u){let lo=operatorFace(info,axis,-1.0,component);let hi=operatorFace(info,axis,1.0,component);diagonal+=lo.diagonal+hi.diagonal;weighted+=lo.weighted+hi.weighted;}let own=select(select(select(pcg[info.index].ap,pcg[info.index].p,component==2u),pcg[info.index].r,component==1u),pcg[info.index].x,component==0u);return diagonal*own-weighted+bodyOperatorTerm(info).value;}
fn projectedFace(info:LocatedCell,axis:u32,sign:f32)->f32{
  let sub=faceSubdivision(info,axis,sign);let own=cellsIn[info.index];let ownP=own.posPressure.w;let ownAlpha=alphaAt(centreFine(info));var total=0.0;var count=0.0;
  for(var b:u32=0u;b<2u;b+=1u){for(var a:u32=0u;a<2u;a+=1u){if(a>=sub||b>=sub){continue;}let p=faceSamplePoint(info,axis,sign,sub,a,b);let n=axisVector(axis);let neighbor=locate(p+0.01*sign*n);if(neighbor.index==INVALID||!physicalPoint(p)){count+=1.0;continue;}let solid=bodyIndexAt(worldFromFine(p+0.01*sign*n));if(solid>=0){total+=bodyVelocityAt(u32(solid),worldFromFine(p))[axis];count+=1.0;continue;}let other=cellsIn[neighbor.index];let otherAlpha=alphaAt(p+0.01*sign*n);if(ownAlpha<0.01&&otherAlpha<0.01){count+=1.0;continue;}var distance=0.5*f32(info.scale+neighbor.scale)*params.originH.w;var otherP=other.posPressure.w;if(otherAlpha<0.01){let theta=clamp(ownAlpha/max(ownAlpha-otherAlpha,1e-6),0.05,1.0);distance*=theta;otherP=0.0;}let base=canonicalFaceSpeed(p,axis);total+=base-params.dimsDt.w/params.physical.x*(otherP-ownP)*sign/max(distance,1e-8);count+=1.0;}}
  return total/max(count,1.0);
}

fn quaternionRotate(q:vec4f,v:vec3f)->vec3f{let uv=cross(q.yzw,v);let uuv=cross(q.yzw,uv);return v+2.0*(q.x*uv+uuv);}
fn quaternionInverseRotate(q:vec4f,v:vec3f)->vec3f{return quaternionRotate(vec4f(q.x,-q.yzw),v);}
fn insideRigid(body:RigidBody,world:vec3f)->bool{let p=quaternionInverseRotate(body.orientation,world-body.positionShape.xyz);let d=body.dimensionsMass.xyz;let shape=i32(round(body.positionShape.w));if(shape==0){return length(p)<=d.x;}if(shape==1){return all(abs(p)<=0.5*d);}if(shape==2){let cy=clamp(p.y,-0.5*d.y,0.5*d.y);return length(vec3f(p.x,p.y-cy,p.z))<=d.x;}return p.x*p.x+p.z*p.z<=d.x*d.x&&abs(p.y)<=0.5*d.y;}
fn bodyIndexAt(world:vec3f)->i32{let count=u32(params.containerBodies.w);for(var index:u32=0u;index<12u;index+=1u){if(index>=count){break;}if(insideRigid(rigidBodies[index],world)){return i32(index);}}return -1;}
fn bodyVelocityAt(index:u32,world:vec3f)->vec3f{let body=rigidBodies[index];let arm=world-body.positionShape.xyz;return body.linearVelocity.xyz+cross(body.angularVelocity.xyz,arm);}
fn inverseInertiaWorld(body:RigidBody,value:vec3f)->vec3f{let local=quaternionInverseRotate(body.orientation,value);return quaternionRotate(body.orientation,local*body.inverseInertia.xyz);}
fn bodyOperatorTerm(info:LocatedCell)->BodyTerm{if(alphaAt(centreFine(info))<0.01){return BodyTerm(0,0);}let centre=centreFine(info);let h=f32(info.scale)*params.originH.w;let area=h*h;var value=0.0;var diagonal=0.0;for(var axis:u32=0u;axis<3u;axis+=1u){for(var side:i32=-1;side<=1;side+=2){let sign=f32(side);let normal=sign*axisVector(axis);let face=centre+0.5*sign*f32(info.scale)*axisVector(axis);let bodyIndex=bodyIndexAt(worldFromFine(face+0.25*sign*f32(info.scale)*axisVector(axis)));if(bodyIndex<0){continue;}let body=rigidBodies[u32(bodyIndex)];let j=normal*area;let arm=worldFromFine(face)-body.positionShape.xyz;let jt=cross(arm,j);let base=u32(bodyIndex)*8u;let force=vec3f(f32(atomicLoad(&rigidExchange[base])),f32(atomicLoad(&rigidExchange[base+1u])),f32(atomicLoad(&rigidExchange[base+2u])))/1000.0;let torque=vec3f(f32(atomicLoad(&rigidExchange[base+3u])),f32(atomicLoad(&rigidExchange[base+4u])),f32(atomicLoad(&rigidExchange[base+5u])))/1000.0;value+=params.physical.x*(dot(j,body.linearVelocity.w*force)+dot(jt,inverseInertiaWorld(body,torque)));diagonal+=params.physical.x*(body.linearVelocity.w*dot(j,j)+dot(jt,inverseInertiaWorld(body,jt)));}}return BodyTerm(value,diagonal);}
fn alphaGradient(point:vec3f,scale:f32)->vec3f{let h=scale*params.originH.w;return vec3f(alphaAt(point+vec3f(scale,0,0))-alphaAt(point-vec3f(scale,0,0)),alphaAt(point+vec3f(0,scale,0))-alphaAt(point-vec3f(0,scale,0)),alphaAt(point+vec3f(0,0,scale))-alphaAt(point-vec3f(0,0,scale)))/max(2.0*h,1e-8);}
fn interfaceNormalAt(point:vec3f,scale:f32)->vec3f{let gradient=alphaGradient(point,scale);return gradient/max(length(gradient),1e-6);}
fn curvatureAtPoint(point:vec3f,scale:f32)->f32{let h=scale*params.originH.w;let nx=interfaceNormalAt(point+vec3f(scale,0,0),scale).x-interfaceNormalAt(point-vec3f(scale,0,0),scale).x;let ny=interfaceNormalAt(point+vec3f(0,scale,0),scale).y-interfaceNormalAt(point-vec3f(0,scale,0),scale).y;let nz=interfaceNormalAt(point+vec3f(0,0,scale),scale).z-interfaceNormalAt(point-vec3f(0,0,scale),scale).z;return -(nx+ny+nz)/max(2.0*h,1e-8);}
fn velocityLaplacian(point:vec3f,scale:f32)->vec3f{let h=scale*params.originH.w;let centre=sampleVelocity(point);return (sampleVelocity(point+vec3f(scale,0,0))+sampleVelocity(point-vec3f(scale,0,0))+sampleVelocity(point+vec3f(0,scale,0))+sampleVelocity(point-vec3f(0,scale,0))+sampleVelocity(point+vec3f(0,0,scale))+sampleVelocity(point-vec3f(0,0,scale))-6.0*centre)/max(h*h,1e-8);}

@compute @workgroup_size(64)
fn computeFluxScales(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) localIndex:u32){let slot=group.x;if(slot>=leafCount()){return;}let index=slot*CELLS_PER_BRICK+localIndex;let info=infoFromIndex(index);let h=f32(info.scale)*params.originH.w;let cellVolume=h*h*h;let alpha=alphaAt(centreFine(info));pcg[index].x=min(1.0,alpha*cellVolume/max(params.dimsDt.w*outwardAlphaRate(info),1e-12));pcg[index].r=min(1.0,(1.0-alpha)*cellVolume/max(params.dimsDt.w*inwardAlphaRate(info),1e-12));}

@compute @workgroup_size(64)
fn advect(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) localIndex:u32){
  let slot=group.x;if(slot>=leafCount()){return;}let index=slot*CELLS_PER_BRICK+localIndex;let info=infoFromIndex(index);let centre=centreFine(info);let dt=params.dimsDt.w;let h=params.originH.w;
  let v0=sampleVelocity(centre);let midpoint=centre-0.5*dt*v0/h;var v=sampleVelocity(centre-dt*sampleVelocity(midpoint)/h);let fluidAlpha=alphaAt(centre);if(fluidAlpha>0.001){v.y+=params.physical.w*dt;let nu=params.physical.y/params.physical.x;v+=dt*nu*velocityLaplacian(centre,f32(info.scale));if(params.physical.z>0.0&&fluidAlpha<0.999){v+=dt*params.physical.z/params.physical.x*curvatureAtPoint(centre,f32(info.scale))*alphaGradient(centre,f32(info.scale));}}atomicMax(&reductions[13],bitcast<u32>(length(v)));
  let cellVolume=pow(f32(info.scale)*h,3.0);let outgoing=integratedLimitedAlphaFlux(info,0u,1.0)-integratedLimitedAlphaFlux(info,0u,-1.0)+integratedLimitedAlphaFlux(info,1u,1.0)-integratedLimitedAlphaFlux(info,1u,-1.0)+integratedLimitedAlphaFlux(info,2u,1.0)-integratedLimitedAlphaFlux(info,2u,-1.0);
  let alpha=clamp(cellsIn[index].negAlpha.w-dt*outgoing/max(cellVolume,1e-12),0.0,1.0);cellsOut[index].negAlpha=vec4f(v,alpha);cellsOut[index].posPressure=vec4f(v,0.0);
}

@compute @workgroup_size(64)
fn jacobi(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) localIndex:u32){
  let slot=group.x;if(slot>=leafCount()){return;}let index=slot*CELLS_PER_BRICK+localIndex;let info=infoFromIndex(index);let cell=cellsIn[index];cellsOut[index]=cell;if(alphaAt(centreFine(info))<0.01){cellsOut[index].posPressure.w=0.0;return;}
  var diagonal=0.0;var weighted=0.0;for(var axis:u32=0u;axis<3u;axis+=1u){let lo=pressureFace(info,axis,-1.0);let hi=pressureFace(info,axis,1.0);diagonal+=lo.diagonal+hi.diagonal;weighted+=lo.weighted+hi.weighted;}
  let rhs=params.physical.x/params.dimsDt.w*divergenceVolume(info);let next=(weighted-rhs)/max(diagonal,1e-9);cellsOut[index].posPressure.w=mix(cell.posPressure.w,next,0.8);
}

@compute @workgroup_size(64)
fn initializePCG(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) localIndex:u32){
  let slot=group.x;if(slot>=leafCount()){return;}let index=slot*CELLS_PER_BRICK+localIndex;let info=infoFromIndex(index);if(alphaAt(centreFine(info))<0.01){pcg[index]=PCGCell(0,0,0,0);return;}let divergence=divergenceVolume(info);let h=f32(info.scale)*params.originH.w;atomicMax(&reductions[12],bitcast<u32>(abs(divergence/max(h*h*h,1e-12))));let diagonal=max(diagonalAt(info),1e-9);let rhs=-params.physical.x/params.dimsDt.w*divergence;let z=rhs/diagonal;pcg[index]=PCGCell(0.0,rhs,z,0.0);
}

@compute @workgroup_size(64)
fn applyPCGOperator(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) localIndex:u32){let slot=group.x;if(slot>=leafCount()){return;}let index=slot*CELLS_PER_BRICK+localIndex;let info=infoFromIndex(index);if(alphaAt(centreFine(info))<0.01){pcg[index].ap=0.0;return;}pcg[index].ap=applyOperator(info,2u);}

fn accumulateBodyVector(info:LocatedCell,pressure:f32,impulseScale:f32,fixedScale:f32){if(alphaAt(centreFine(info))<0.01){return;}let centre=centreFine(info);let h=f32(info.scale)*params.originH.w;let area=h*h;for(var axis:u32=0u;axis<3u;axis+=1u){for(var side:i32=-1;side<=1;side+=2){let sign=f32(side);let normal=sign*axisVector(axis);let face=centre+0.5*sign*f32(info.scale)*axisVector(axis);let bodyIndex=bodyIndexAt(worldFromFine(face+0.25*sign*f32(info.scale)*axisVector(axis)));if(bodyIndex<0){continue;}let body=rigidBodies[u32(bodyIndex)];let force=pressure*area*normal*impulseScale;let torque=cross(worldFromFine(face)-body.positionShape.xyz,force);let base=u32(bodyIndex)*8u;atomicAdd(&rigidExchange[base],i32(round(force.x*fixedScale)));atomicAdd(&rigidExchange[base+1u],i32(round(force.y*fixedScale)));atomicAdd(&rigidExchange[base+2u],i32(round(force.z*fixedScale)));atomicAdd(&rigidExchange[base+3u],i32(round(torque.x*fixedScale)));atomicAdd(&rigidExchange[base+4u],i32(round(torque.y*fixedScale)));atomicAdd(&rigidExchange[base+5u],i32(round(torque.z*fixedScale)));}}}
@compute @workgroup_size(64) fn reduceBodyJp(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) localIndex:u32){let slot=group.x;if(slot>=leafCount()){return;}let index=slot*CELLS_PER_BRICK+localIndex;accumulateBodyVector(infoFromIndex(index),pcg[index].p,1.0,1000.0);}
@compute @workgroup_size(64) fn accumulatePressureLoads(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) localIndex:u32){let slot=group.x;if(slot>=leafCount()){return;}let index=slot*CELLS_PER_BRICK+localIndex;accumulateBodyVector(infoFromIndex(index),pcg[index].x,params.dimsDt.w,1e6);}

fn reduceValue(index:u32,mode:u32)->f32{let q=pcg[index];if(mode==0u){let info=infoFromIndex(index);return q.r*q.r/max(diagonalAt(info),1e-9);}if(mode==1u){return q.p*q.ap;}return q.r*q.r;}
fn reduceSolver(mode:u32,localIndex:u32){var sum=0.0;let count=leafCount()*CELLS_PER_BRICK;for(var index=localIndex;index<count;index+=256u){sum+=reduceValue(index,mode);}reductionScratch[localIndex]=sum;workgroupBarrier();var stride=128u;loop{if(localIndex<stride){reductionScratch[localIndex]+=reductionScratch[localIndex+stride];}workgroupBarrier();if(stride==1u){break;}stride/=2u;}if(localIndex==0u){if(mode==0u){storeScalar(1u,loadScalar(0u));storeScalar(0u,reductionScratch[0]);if(atomicLoad(&reductions[11u])==0u){storeScalar(6u,reductionScratch[0]);atomicStore(&reductions[11u],1u);}}else if(mode==1u){storeScalar(2u,reductionScratch[0]);}else{storeScalar(5u,reductionScratch[0]);}}}
@compute @workgroup_size(1) fn resetPCG(){atomicStore(&reductions[10u],0u);atomicStore(&reductions[11u],0u);atomicStore(&reductions[14u],0u);atomicStore(&reductions[16u],leafCount());atomicStore(&reductions[17u],1u);atomicStore(&reductions[18u],1u);}
@compute @workgroup_size(256) fn reduceRZ(@builtin(local_invocation_index) localIndex:u32){reduceSolver(0u,localIndex);}
@compute @workgroup_size(256) fn reducePAP(@builtin(local_invocation_index) localIndex:u32){reduceSolver(1u,localIndex);}
@compute @workgroup_size(256) fn reduceResidual(@builtin(local_invocation_index) localIndex:u32){reduceSolver(2u,localIndex);}
@compute @workgroup_size(1) fn computeAlpha(){storeScalar(3u,loadScalar(0u)/max(abs(loadScalar(2u)),1e-20));}
@compute @workgroup_size(1) fn computeBeta(){let wasActive=atomicLoad(&reductions[16u])>0u;if(wasActive){atomicAdd(&reductions[14u],1u);}let rz=loadScalar(0u);storeScalar(4u,rz/max(abs(loadScalar(1u)),1e-20));let relative=sqrt(abs(rz)/max(abs(loadScalar(6u)),1e-20));atomicStore(&reductions[16u],select(0u,leafCount(),wasActive&&relative>max(params.boundary.x,1e-6)));atomicStore(&reductions[17u],1u);atomicStore(&reductions[18u],1u);}

@compute @workgroup_size(64)
fn updateXR(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) localIndex:u32){let slot=group.x;if(slot>=leafCount()){return;}let index=slot*CELLS_PER_BRICK+localIndex;let alpha=loadScalar(3u);pcg[index].x+=alpha*pcg[index].p;pcg[index].r-=alpha*pcg[index].ap;}
@compute @workgroup_size(64)
fn updateP(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) localIndex:u32){let slot=group.x;if(slot>=leafCount()){return;}let index=slot*CELLS_PER_BRICK+localIndex;let info=infoFromIndex(index);let z=pcg[index].r/max(diagonalAt(info),1e-9);pcg[index].p=z+loadScalar(4u)*pcg[index].p;}
@compute @workgroup_size(64)
fn commitPressure(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) localIndex:u32){let slot=group.x;if(slot>=leafCount()){return;}let index=slot*CELLS_PER_BRICK+localIndex;let cell=cellsIn[index];cellsOut[index]=Cell(cell.negAlpha,vec4f(cell.posPressure.xyz,pcg[index].x));}

@compute @workgroup_size(64)
fn project(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) localIndex:u32){
  let slot=group.x;if(slot>=leafCount()){return;}let index=slot*CELLS_PER_BRICK+localIndex;let info=infoFromIndex(index);let cell=cellsIn[index];var neg=vec3f(0.0);var pos=vec3f(0.0);for(var axis:u32=0u;axis<3u;axis+=1u){neg[axis]=projectedFace(info,axis,-1.0);pos[axis]=projectedFace(info,axis,1.0);}cellsOut[index].negAlpha=vec4f(neg,cell.negAlpha.w);cellsOut[index].posPressure=vec4f(pos,cell.posPressure.w);
}

// Conservative immersed exchange remains as a bounded fallback for viscous
// no-slip response. Pressure/normal coupling is handled by the cut-face
// projection path as that path is enabled per body.
@compute @workgroup_size(64)
fn coupleRigid(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) localIndex:u32){
  let slot=group.x;if(slot>=leafCount()){return;}let index=slot*CELLS_PER_BRICK+localIndex;let info=infoFromIndex(index);var cell=cellsIn[index];let world=worldFromFine(centreFine(info));let alpha=cell.negAlpha.w;let h=f32(info.scale)*params.originH.w;let mass=params.physical.x*h*h*h*alpha;let bodyCount=u32(params.containerBodies.w);
  for(var bodyIndex:u32=0u;bodyIndex<12u;bodyIndex+=1u){if(bodyIndex>=bodyCount){break;}let body=rigidBodies[bodyIndex];if(!insideRigid(body,world)){continue;}let arm=world-body.positionShape.xyz;let solidV=body.linearVelocity.xyz+cross(body.angularVelocity.xyz,arm);let oldV=0.5*(cell.negAlpha.xyz+cell.posPressure.xyz);let blend=clamp(45.0*params.dimsDt.w,0.0,1.0);let impulse=mass*(solidV-oldV)*blend;let v=oldV+impulse/max(mass,1e-9);cell.negAlpha=vec4f(v,cell.negAlpha.w);cell.posPressure=vec4f(v,cell.posPressure.w);let reaction=-impulse;let torque=cross(arm,reaction);let base=bodyIndex*8u;atomicAdd(&rigidExchange[base],i32(round(reaction.x*1e6)));atomicAdd(&rigidExchange[base+1u],i32(round(reaction.y*1e6)));atomicAdd(&rigidExchange[base+2u],i32(round(reaction.z*1e6)));atomicAdd(&rigidExchange[base+3u],i32(round(torque.x*1e6)));atomicAdd(&rigidExchange[base+4u],i32(round(torque.y*1e6)));atomicAdd(&rigidExchange[base+5u],i32(round(torque.z*1e6)));atomicAdd(&rigidExchange[base+6u],i32(round(alpha*h*h*h*1e9)));break;}
  cellsOut[index]=cell;
}

@compute @workgroup_size(64)
fn reduceDiagnostics(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) localIndex:u32){let slot=group.x;if(slot>=leafCount()){return;}let index=slot*CELLS_PER_BRICK+localIndex;let info=infoFromIndex(index);let cell=cellsIn[index];let h=f32(info.scale)*params.originH.w;atomicAdd(&reductions[0],u32(clamp(cell.negAlpha.w*h*h*h*1e9,0.0,4294967295.0)));atomicMax(&reductions[1],bitcast<u32>(length(0.5*(cell.negAlpha.xyz+cell.posPressure.xyz))));atomicMax(&reductions[2],bitcast<u32>(abs(divergenceVolume(info)/max(h*h*h,1e-12))));}

@compute @workgroup_size(4,4,4)
fn reconstructPresentation(@builtin(global_invocation_id) gid:vec3u){let d=vec3u(params.boundary.yzw);if(any(gid>=d)){return;}let scale=vec3f(fineDims())/vec3f(d);let p=(vec3f(gid)+0.5)*scale;textureStore(presentation,vec3i(gid),vec4f(alphaAt(p),0,0,0));}
`;

export class WebGPUHierarchicalSolver {
  readonly info: GPUEulerianInfo;
  readonly volumeTexture: GPUTexture;
  private layout: GPUHierarchyLayout;
  private cellsA: GPUBuffer;
  private cellsB: GPUBuffer;
  private metadata: GPUBuffer;
  private pageTable: GPUBuffer;
  private params: GPUBuffer;
  private readonly rigidBuffer: GPUBuffer;
  private readonly rigidExchangeBuffer: GPUBuffer;
  private readonly reductions: GPUBuffer;
  private readonly indirectDispatchBuffer:GPUBuffer;
  private pcgBuffer: GPUBuffer;
  private readonly bindGroupLayout: GPUBindGroupLayout;
  private groups: [GPUBindGroup, GPUBindGroup];
  private readonly advectPipeline: GPUComputePipeline;
  private readonly jacobiPipeline: GPUComputePipeline;
  private readonly projectPipeline: GPUComputePipeline;
  private readonly rigidPipeline: GPUComputePipeline;
  private readonly reductionPipeline: GPUComputePipeline;
  private readonly presentationPipeline: GPUComputePipeline;
  private readonly initializePCGPipeline: GPUComputePipeline;
  private readonly resetPCGPipeline: GPUComputePipeline;
  private readonly fluxScalePipeline: GPUComputePipeline;
  private readonly applyPCGPipeline: GPUComputePipeline;
  private readonly reduceRZPipeline: GPUComputePipeline;
  private readonly reducePAPPipeline: GPUComputePipeline;
  private readonly reduceResidualPipeline: GPUComputePipeline;
  private readonly computeAlphaPipeline: GPUComputePipeline;
  private readonly computeBetaPipeline: GPUComputePipeline;
  private readonly updateXRPipeline: GPUComputePipeline;
  private readonly updatePPipeline: GPUComputePipeline;
  private readonly commitPressurePipeline: GPUComputePipeline;
  private readonly reduceBodyJpPipeline: GPUComputePipeline;
  private readonly pressureLoadsPipeline: GPUComputePipeline;
  private readonly querySet?: GPUQuerySet;
  private readonly queryResolve?: GPUBuffer;
  private current = 0;
  private lastTime = 0;
  private readbackPending = false;
  private rigidReadbackPending = false;
  private validationChecked = false;
  private initialVolume_m3 = 0;
  private regridPending = false;
  private stepsSinceRegrid = 0;
  private regridAttempt = 0;
  private destroyed = false;
  private submissionsInFlight = 0;
  private parameterRevision = 0;

  constructor(private readonly device: GPUDevice, readonly scene: SceneDescription, private readonly quality: GPUQuality, private readonly onRigidLoads?: (loads: GPURigidLoad[]) => void) {
    this.layout = createGPUHierarchyLayout(scene, quality);
    const byteLength = this.layout.initialCells.byteLength;
    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    this.cellsA = device.createBuffer({ label: "Hierarchy cells A", size: byteLength, usage: storage });
    this.cellsB = device.createBuffer({ label: "Hierarchy cells B", size: byteLength, usage: storage });
    this.metadata = device.createBuffer({ label: "Hierarchy leaf metadata", size: this.layout.leafMetadata.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.pageTable = device.createBuffer({ label: "Hierarchy finest-brick page table", size: this.layout.pageTable.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.params = this.makeParamsBuffer(scene.numerics.fixedDt_s,0);
    this.rigidBuffer = device.createBuffer({ label: "Hierarchy rigid bodies", size: 12 * 96, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.rigidExchangeBuffer = device.createBuffer({ label: "Hierarchy rigid exchange", size: 12 * 8 * 4, usage: storage });
    this.reductions = device.createBuffer({ label: "Hierarchy diagnostics and PCG scalars", size: 256, usage: storage });
    this.indirectDispatchBuffer=device.createBuffer({label:"Hierarchy pressure indirect dispatch",size:12,usage:GPUBufferUsage.INDIRECT|GPUBufferUsage.COPY_DST});
    this.pcgBuffer = device.createBuffer({ label: "Hierarchy PCG vectors", size: this.layout.activeCellCount * 16, usage: storage });
    if(device.features.has("timestamp-query")){this.querySet=device.createQuerySet({type:"timestamp",count:10});this.queryResolve=device.createBuffer({label:"Hierarchy timestamp resolve",size:80,usage:GPUBufferUsage.QUERY_RESOLVE|GPUBufferUsage.COPY_SRC});}
    const pd = this.layout.physicalFinestCellDims;
    // Compatibility placeholder for the CPU presentation pipeline. The WebGPU
    // renderer reads leaf buffers directly and therefore does not allocate a
    // dense finest-resolution presentation volume.
    this.volumeTexture = device.createTexture({ label: "Hierarchy presentation placeholder", size: [1, 1, 1], dimension: "3d", format: "r32float", usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING });
    device.queue.writeBuffer(this.cellsA, 0, this.layout.initialCells);device.queue.writeBuffer(this.cellsB, 0, this.layout.initialCells);device.queue.writeBuffer(this.metadata, 0, this.layout.leafMetadata);device.queue.writeBuffer(this.pageTable, 0, this.layout.pageTable);
    this.initialVolume_m3 = this.computeInitialVolume();

    const shaderModule = device.createShaderModule({ label: "Hierarchical Eulerian kernels", code: shader });
    void shaderModule.getCompilationInfo().then((result) => { if (result.messages.length > 0) throw new Error(result.messages.map((message) => `Hierarchical WGSL ${message.type} ${message.lineNum}:${message.linePos} ${message.message}`).join("\n")); });
    this.bindGroupLayout = device.createBindGroupLayout({ entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 8, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: "write-only", format: "r32float", viewDimension: "3d" } }
      ,{ binding: 9, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } }
    ] });
    const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [this.bindGroupLayout] });
    device.pushErrorScope("validation");
    const pipeline = (entryPoint: string) => device.createComputePipeline({ label:`Hierarchy ${entryPoint}`,layout: pipelineLayout, compute: { module:shaderModule, entryPoint } });
    this.advectPipeline = pipeline("advect");this.jacobiPipeline = pipeline("jacobi");this.projectPipeline = pipeline("project");this.rigidPipeline = pipeline("coupleRigid");this.reductionPipeline = pipeline("reduceDiagnostics");this.presentationPipeline = pipeline("reconstructPresentation");
    this.initializePCGPipeline=pipeline("initializePCG");this.resetPCGPipeline=pipeline("resetPCG");this.applyPCGPipeline=pipeline("applyPCGOperator");this.reduceRZPipeline=pipeline("reduceRZ");this.reducePAPPipeline=pipeline("reducePAP");this.reduceResidualPipeline=pipeline("reduceResidual");this.computeAlphaPipeline=pipeline("computeAlpha");this.computeBetaPipeline=pipeline("computeBeta");this.updateXRPipeline=pipeline("updateXR");this.updatePPipeline=pipeline("updateP");this.commitPressurePipeline=pipeline("commitPressure");
    this.fluxScalePipeline=pipeline("computeFluxScales");
    this.reduceBodyJpPipeline=pipeline("reduceBodyJp");this.pressureLoadsPipeline=pipeline("accumulatePressureLoads");
    void device.popErrorScope().then((error) => { if (error) throw new Error(`Hierarchical pipeline validation: ${error.message}`); });
    this.groups = [this.group(this.cellsA, this.cellsB), this.group(this.cellsB, this.cellsA)];
    const pressureIterations = Math.min(scene.numerics.pressureMaxIterations,quality === "balanced" ? 48 : quality === "high" ? 64 : 80);
    this.info = {
      nx: pd.x, ny: pd.y, nz: pd.z,
      cellCount: this.layout.activeCellCount,
      cellSize_m: this.layout.topology.finestCellLength_m,
      pressureIterations,
      allocatedBytes: byteLength * 2 + this.layout.leafMetadata.byteLength + this.layout.pageTable.byteLength
        + this.layout.activeCellCount * 16 + 256 + 12 + 12 * (96 + 32) + 256,
      quality,
      hierarchyLevels: this.layout.topology.settings.levels,
      activeBrickCount: this.layout.topology.leaves.length,
      equivalentUniformCells: this.layout.equivalentUniformCells,
      compressionRatio: this.layout.activeCellCount / this.layout.equivalentUniformCells,
      topologySaturated: this.layout.topology.saturated,
      topologyRevision: 0,
      regridCount: 0,
      encodedSteps: 0,
      simulatedTime_s:0,
      queuedSubmissions:0,
      substepsLast:0,
      initialVolumeCellSum: this.initialVolume_m3 / this.layout.topology.finestCellLength_m ** 3,
      volumeCellSum: this.initialVolume_m3 / this.layout.topology.finestCellLength_m ** 3,
      volumeDrift: 0,
      rawVolumeDrift: 0,
      maxSpeed_m_s: 0,
      front_m: -scene.container.width_m / 2
    };
  }

  private computeInitialVolume(): number {
    let total = 0;
    for (let slot = 0; slot < this.layout.topology.leaves.length; slot += 1) {
      const scale = this.layout.leafMetadata[slot * GPU_BRICK_META_WORDS + 3];
      const volume = (scale * this.layout.topology.finestCellLength_m) ** 3;
      for (let local = 0; local < GPU_CELLS_PER_BRICK; local += 1) total += this.layout.initialCells[(slot * GPU_CELLS_PER_BRICK + local) * GPU_CELL_FLOATS + 3] * volume;
    }
    return total;
  }

  get hierarchyRenderResources() {
    return { cells: this.current === 0 ? this.cellsA : this.cellsB, metadata: this.metadata, pageTable: this.pageTable, params: this.params, revision: this.parameterRevision*2+this.current };
  }

  private group(input: GPUBuffer, output: GPUBuffer): GPUBindGroup {
    return this.device.createBindGroup({ layout: this.bindGroupLayout, entries: [
      { binding: 0, resource: { buffer: input } }, { binding: 1, resource: { buffer: output } },
      { binding: 2, resource: { buffer: this.metadata } }, { binding: 3, resource: { buffer: this.pageTable } },
      { binding: 4, resource: { buffer: this.params } }, { binding: 5, resource: { buffer: this.rigidBuffer } },
      { binding: 6, resource: { buffer: this.rigidExchangeBuffer } }, { binding: 7, resource: { buffer: this.reductions } },
      { binding: 8, resource: this.volumeTexture.createView() }
      ,{ binding: 9, resource: { buffer: this.pcgBuffer } }
    ] });
  }

  private parameterData(dt:number,bodyCount:number):Float32Array {
    const t = this.layout.topology, p = this.layout.physicalFinestCellDims, c = this.scene.container;
    return new Float32Array([
      t.paddedFinestCellDims.x, t.paddedFinestCellDims.y, t.paddedFinestCellDims.z, dt,
      t.finestBrickDims.x, t.finestBrickDims.y, t.finestBrickDims.z, GPU_BRICK_SIZE,
      t.origin_m.x, t.origin_m.y, t.origin_m.z, t.finestCellLength_m,
      this.scene.fluid.density_kg_m3, this.scene.fluid.dynamicViscosity_Pa_s, this.scene.fluid.surfaceTension_N_m, this.scene.fluid.gravity_m_s2.y,
      c.width_m, c.height_m, c.depth_m, bodyCount,
      t.leaves.length, t.settings.levels, this.layout.equivalentUniformCells, t.saturated ? 1 : 0,
      this.scene.numerics.pressureRelativeTolerance, p.x, p.y, p.z
    ]);
  }

  private makeParamsBuffer(dt:number,bodyCount:number):GPUBuffer {
    const buffer=this.device.createBuffer({label:"Hierarchy parameters",size:256,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});this.device.queue.writeBuffer(buffer,0,this.parameterData(dt,bodyCount));return buffer;
  }

  private writeParams(dt:number,bodyCount:number):void {
    this.device.queue.writeBuffer(this.params,0,this.parameterData(dt,bodyCount));
  }

  private uploadBodies(bodies: RigidBodyState[]): RigidBodyState[] {
    const active = bodies.slice(0, 12), data = new Float32Array(12 * 24), shape = { sphere: 0, box: 1, capsule: 2, cylinder: 3 } as const;
    active.forEach((body, index) => {
      const o = index * 24, d = body.description.dimensions_m, q = body.orientation;
      data.set([
        body.position_m.x, body.position_m.y, body.position_m.z, shape[body.description.shape],
        d.x, d.y, d.z, body.mass_kg,
        q.w, q.x, q.y, q.z,
        body.linearVelocity_m_s.x, body.linearVelocity_m_s.y, body.linearVelocity_m_s.z, body.inverseMass_kg,
        body.angularVelocity_rad_s.x, body.angularVelocity_rad_s.y, body.angularVelocity_rad_s.z, 0,
        body.inverseInertiaBody_kg_m2.x, body.inverseInertiaBody_kg_m2.y, body.inverseInertiaBody_kg_m2.z, 0
      ], o);
    });
    this.device.queue.writeBuffer(this.rigidBuffer, 0, data);
    return active;
  }

  private dispatch(pass: GPUComputePassEncoder, pipeline: GPUComputePipeline, groupIndex = this.current): void {
    pass.setPipeline(pipeline);pass.setBindGroup(0, this.groups[groupIndex]);pass.dispatchWorkgroups(this.layout.topology.leaves.length);
  }

  private dispatchSingle(pass: GPUComputePassEncoder, pipeline: GPUComputePipeline): void {
    pass.setPipeline(pipeline);pass.setBindGroup(0,this.groups[this.current]);pass.dispatchWorkgroups(1);
  }

  private dispatchPressure(pass:GPUComputePassEncoder,pipeline:GPUComputePipeline):void {
    pass.setPipeline(pipeline);pass.setBindGroup(0,this.groups[this.current]);pass.dispatchWorkgroupsIndirect(this.indirectDispatchBuffer,0);
  }

  private reconstructPresentation(encoder?: GPUCommandEncoder): void {
    const owned = !encoder, command = encoder ?? this.device.createCommandEncoder({ label: "Hierarchy presentation reconstruction" });
    const pass = command.beginComputePass();pass.setPipeline(this.presentationPipeline);pass.setBindGroup(0, this.groups[this.current]);const d = this.layout.physicalFinestCellDims;pass.dispatchWorkgroups(Math.ceil(d.x / 4), Math.ceil(d.y / 4), Math.ceil(d.z / 4));pass.end();if(owned)this.device.queue.submit([command.finish()]);
  }

  private scheduleRegrid(bodies: RigidBodyState[]): void {
    if (this.regridPending || this.destroyed) return;
    this.regridPending = true;
    const source = this.current === 0 ? this.cellsA : this.cellsB, bytes = this.layout.activeCellCount * GPU_CELL_FLOATS * 4;
    const readback = this.device.createBuffer({ label: "Hierarchy topology readback", size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const encoder = this.device.createCommandEncoder({ label: "Hierarchy regrid readback" });encoder.copyBufferToBuffer(source,0,readback,0,bytes);this.device.queue.submit([encoder.finish()]);
    const bodySnapshot = bodies.map((body) => ({ ...body, position_m: { ...body.position_m }, description: { ...body.description, dimensions_m: { ...body.description.dimensions_m } } }));
    void readback.mapAsync(GPUMapMode.READ).then(async () => {
      const cpuRegridStarted=performance.now();this.info.regridReadbackBytes=bytes;
      const fields = new Float32Array(readback.getMappedRange().slice(0));readback.unmap();if (this.destroyed) return;
      let volume=0,maxSpeed=0,maxPressure=0,nanCount=0,front=-this.scene.container.width_m/2;
      for(let slot=0;slot<this.layout.topology.leaves.length;slot+=1){
        const meta=slot*GPU_BRICK_META_WORDS,originX=this.layout.leafMetadata[meta],scale=this.layout.leafMetadata[meta+3],cellVolume=(scale*this.layout.topology.finestCellLength_m)**3;
        for(let local=0;local<GPU_CELLS_PER_BRICK;local+=1){
          const offset=(slot*GPU_CELLS_PER_BRICK+local)*GPU_CELL_FLOATS,alpha=fields[offset+3],vx=.5*(fields[offset]+fields[offset+4]),vy=.5*(fields[offset+1]+fields[offset+5]),vz=.5*(fields[offset+2]+fields[offset+6]);
          for(let component=0;component<GPU_CELL_FLOATS;component+=1)if(!Number.isFinite(fields[offset+component]))nanCount+=1;
          volume+=alpha*cellVolume;maxSpeed=Math.max(maxSpeed,Math.hypot(vx,vy,vz),Math.hypot(fields[offset],fields[offset+1],fields[offset+2]),Math.hypot(fields[offset+4],fields[offset+5],fields[offset+6]));maxPressure=Math.max(maxPressure,Math.abs(fields[offset+7]));
          if(alpha>=0.5){const localX=local%GPU_BRICK_SIZE;front=Math.max(front,this.layout.topology.origin_m.x+(originX+(localX+.5)*scale)*this.layout.topology.finestCellLength_m);}
        }
      }
      this.info.volumeCellSum=volume/this.info.cellSize_m**3;this.info.volumeDrift=(volume-this.initialVolume_m3)/Math.max(this.initialVolume_m3,1e-12);this.info.rawVolumeDrift=this.info.volumeDrift;this.info.maxSpeed_m_s=maxSpeed;this.info.pressureMax_Pa=maxPressure;this.info.nanCount=nanCount;this.info.front_m=front;
      // A one-level hierarchy uses this exact readback path for diagnostics but
      // has no topology to rebuild. This keeps uniform mode on the same solver
      // and data representation without manufacturing no-op topology revisions.
      if(this.scene.hierarchy.levels<=1){this.info.cpuRegrid_ms=performance.now()-cpuRegridStarted;return;}
      this.regridAttempt+=1;this.info.regridCount=(this.info.regridCount??0)+1;
      const next = rebuildGPUHierarchyLayout(this.scene,this.quality,this.layout,fields,bodySnapshot,this.regridAttempt%this.scene.hierarchy.coarsenDelay===0);
      const same = next.leafMetadata.length === this.layout.leafMetadata.length && next.leafMetadata.every((value,index) => value === this.layout.leafMetadata[index]);
      if (same || this.destroyed){this.info.cpuRegrid_ms=performance.now()-cpuRegridStarted;return;}
      await this.device.queue.onSubmittedWorkDone();if(this.destroyed)return;
      const storage=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC,cellBytes=next.initialCells.byteLength;
      const cellsA=this.device.createBuffer({label:"Hierarchy cells A regridded",size:cellBytes,usage:storage}),cellsB=this.device.createBuffer({label:"Hierarchy cells B regridded",size:cellBytes,usage:storage});
      const metadata=this.device.createBuffer({label:"Hierarchy leaf metadata regridded",size:next.leafMetadata.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
      const pageTable=this.device.createBuffer({label:"Hierarchy page table regridded",size:next.pageTable.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
      const pcg=this.device.createBuffer({label:"Hierarchy PCG vectors regridded",size:next.activeCellCount*16,usage:storage});
      this.device.queue.writeBuffer(cellsA,0,next.initialCells);this.device.queue.writeBuffer(cellsB,0,next.initialCells);this.device.queue.writeBuffer(metadata,0,next.leafMetadata);this.device.queue.writeBuffer(pageTable,0,next.pageTable);
      const old=[this.cellsA,this.cellsB,this.metadata,this.pageTable,this.pcgBuffer];this.cellsA=cellsA;this.cellsB=cellsB;this.metadata=metadata;this.pageTable=pageTable;this.pcgBuffer=pcg;this.layout=next;this.current=0;this.groups=[this.group(this.cellsA,this.cellsB),this.group(this.cellsB,this.cellsA)];this.parameterRevision+=1;
      this.info.cellCount=next.activeCellCount;this.info.activeBrickCount=next.topology.leaves.length;this.info.equivalentUniformCells=next.equivalentUniformCells;this.info.compressionRatio=next.activeCellCount/next.equivalentUniformCells;this.info.topologySaturated=next.topology.saturated;this.info.allocatedBytes=cellBytes*2+next.leafMetadata.byteLength+next.pageTable.byteLength+next.activeCellCount*16+256+12+12*(96+32)+256;
      this.info.topologyRevision=(this.info.topologyRevision??0)+1;
      this.writeParams(this.scene.numerics.fixedDt_s,bodySnapshot.length);for(const buffer of old)buffer.destroy();this.info.cpuRegrid_ms=performance.now()-cpuRegridStarted;
    }).catch((error:unknown)=>console.error(`Hierarchy regrid failed: ${error instanceof Error?error.message:String(error)}`)).finally(()=>{readback.destroy();this.regridPending=false;});
  }

  advanceTo(time_s: number, bodies: RigidBodyState[] = []): boolean {
    if(this.submissionsInFlight>=2)return true;
    if (time_s < this.lastTime) return false;
    const elapsed = Math.min(this.scene.numerics.maxDt_s, time_s - this.lastTime);if (elapsed < 1e-6) return true;this.lastTime += elapsed;
    const activeBodies = this.uploadBodies(bodies), waveSpeed = Math.sqrt(Math.abs(this.scene.fluid.gravity_m_s2.y) * this.scene.container.height_m * Math.max(this.scene.container.fillFraction, 0.01));
    const speed = Math.max(waveSpeed, this.info.maxSpeed_m_s ?? 0, 0.1), h = this.info.cellSize_m, sigma = this.scene.fluid.surfaceTension_N_m, rho = this.scene.fluid.density_kg_m3;
    const stable = Math.min(0.4 * h / speed, sigma > 0 ? 0.35 * Math.sqrt(rho * h ** 3 / (Math.PI * sigma)) : Infinity), substeps = Math.min(16, Math.max(1, Math.ceil(elapsed / stable))), dt = elapsed / substeps;
    this.writeParams(dt,activeBodies.length);if (!this.validationChecked) this.device.pushErrorScope("validation");
    const encoder = this.device.createCommandEncoder({ label: "Hierarchical Eulerian step" });encoder.clearBuffer(this.rigidExchangeBuffer);encoder.clearBuffer(this.reductions,0,12);encoder.clearBuffer(this.reductions,40,20);
    for (let substep = 0; substep < substeps; substep += 1) {
      const first=substep===0,last=substep===substeps-1;
      let pass=encoder.beginComputePass(this.querySet&&first?{timestampWrites:{querySet:this.querySet,beginningOfPassWriteIndex:0}}:undefined);this.dispatch(pass,this.fluxScalePipeline);pass.end();
      pass = encoder.beginComputePass(this.querySet&&last?{timestampWrites:{querySet:this.querySet,endOfPassWriteIndex:1}}:undefined);this.dispatch(pass, this.advectPipeline);pass.end();this.current = 1 - this.current;
      pass=encoder.beginComputePass();this.dispatchSingle(pass,this.resetPCGPipeline);pass.end();
      encoder.copyBufferToBuffer(this.reductions,64,this.indirectDispatchBuffer,0,12);
      pass=encoder.beginComputePass(this.querySet&&first?{timestampWrites:{querySet:this.querySet,beginningOfPassWriteIndex:2}}:undefined);this.dispatch(pass,this.initializePCGPipeline);pass.end();
      pass=encoder.beginComputePass();this.dispatchSingle(pass,this.reduceRZPipeline);pass.end();
      for (let iteration = 0; iteration < this.info.pressureIterations; iteration += 1) {
        encoder.clearBuffer(this.rigidExchangeBuffer);
        pass=encoder.beginComputePass();this.dispatchPressure(pass,this.reduceBodyJpPipeline);pass.end();
        pass=encoder.beginComputePass();this.dispatchPressure(pass,this.applyPCGPipeline);pass.end();
        pass=encoder.beginComputePass();this.dispatchSingle(pass,this.reducePAPPipeline);pass.end();
        pass=encoder.beginComputePass();this.dispatchSingle(pass,this.computeAlphaPipeline);pass.end();
        pass=encoder.beginComputePass();this.dispatchPressure(pass,this.updateXRPipeline);pass.end();
        pass=encoder.beginComputePass();this.dispatchSingle(pass,this.reduceRZPipeline);pass.end();
        pass=encoder.beginComputePass();this.dispatchSingle(pass,this.computeBetaPipeline);pass.end();
        encoder.copyBufferToBuffer(this.reductions,64,this.indirectDispatchBuffer,0,12);
        pass=encoder.beginComputePass();this.dispatchPressure(pass,this.updatePPipeline);pass.end();
      }
      pass=encoder.beginComputePass();this.dispatchSingle(pass,this.reduceResidualPipeline);pass.end();
      encoder.clearBuffer(this.rigidExchangeBuffer);pass=encoder.beginComputePass();this.dispatch(pass,this.pressureLoadsPipeline);pass.end();
      pass=encoder.beginComputePass(this.querySet&&last?{timestampWrites:{querySet:this.querySet,endOfPassWriteIndex:3}}:undefined);this.dispatch(pass,this.commitPressurePipeline);pass.end();this.current=1-this.current;
      pass = encoder.beginComputePass(this.querySet&&(first||last)?{timestampWrites:{querySet:this.querySet,...(first?{beginningOfPassWriteIndex:4}:{}),...(last?{endOfPassWriteIndex:5}:{})}}:undefined);this.dispatch(pass, this.projectPipeline);pass.end();this.current = 1 - this.current;
      if (activeBodies.length > 0) {pass = encoder.beginComputePass(this.querySet&&(first||last)?{timestampWrites:{querySet:this.querySet,...(first?{beginningOfPassWriteIndex:6}:{}),...(last?{endOfPassWriteIndex:7}:{})}}:undefined);this.dispatch(pass, this.rigidPipeline);pass.end();this.current = 1 - this.current;}
    }
    if(this.querySet&&activeBodies.length===0){const emptyRigidPass=encoder.beginComputePass({timestampWrites:{querySet:this.querySet,beginningOfPassWriteIndex:6,endOfPassWriteIndex:7}});emptyRigidPass.end();}
    const diagnosticPass = encoder.beginComputePass(this.querySet?{timestampWrites:{querySet:this.querySet,beginningOfPassWriteIndex:8,endOfPassWriteIndex:9}}:undefined);this.dispatch(diagnosticPass, this.reductionPipeline);diagnosticPass.end();
    if(this.querySet&&this.queryResolve)encoder.resolveQuerySet(this.querySet,0,10,this.queryResolve,0);
    let rigidReadback: GPUBuffer | undefined;if(activeBodies.length > 0 && this.onRigidLoads && !this.rigidReadbackPending){this.rigidReadbackPending=true;rigidReadback=this.device.createBuffer({size:12*8*4,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});encoder.copyBufferToBuffer(this.rigidExchangeBuffer,0,rigidReadback,0,12*8*4);}
    this.device.queue.submit([encoder.finish()]);this.submissionsInFlight+=1;this.info.queuedSubmissions=this.submissionsInFlight;void this.device.queue.onSubmittedWorkDone().finally(()=>{this.submissionsInFlight=Math.max(0,this.submissionsInFlight-1);this.info.queuedSubmissions=this.submissionsInFlight;});this.info.encodedSteps = (this.info.encodedSteps ?? 0) + substeps;this.info.simulatedTime_s=this.lastTime;this.info.substepsLast=substeps;
    this.stepsSinceRegrid+=substeps;if(this.stepsSinceRegrid>=this.scene.hierarchy.regridInterval){this.stepsSinceRegrid=0;this.scheduleRegrid(activeBodies);}
    if(rigidReadback){const buffer=rigidReadback;void buffer.mapAsync(GPUMapMode.READ).then(()=>{const words=new Int32Array(buffer.getMappedRange());const loads=activeBodies.map((body,index)=>{const b=index*8;return{bodyId:body.description.id,impulse_N_s:{x:words[b]/1e6,y:words[b+1]/1e6,z:words[b+2]/1e6},angularImpulse_N_m_s:{x:words[b+3]/1e6,y:words[b+4]/1e6,z:words[b+5]/1e6},couplingInterval_s:elapsed,displacedVolume_m3:words[b+6]/1e9};});buffer.unmap();buffer.destroy();this.onRigidLoads?.(loads);}).catch(()=>buffer.destroy()).finally(()=>{this.rigidReadbackPending=false;});}
    if(!this.validationChecked){this.validationChecked=true;void this.device.popErrorScope().then((error)=>{if(error)throw new Error(`Hierarchical GPU validation: ${error.message}`);});}
    return true;
  }

  async readStats(): Promise<GPUEulerianInfo> {
    if (this.readbackPending || (this.info.encodedSteps ?? 0) === 0) return this.info;this.readbackPending = true;
    const queryBytes=this.queryResolve?80:0,buffer = this.device.createBuffer({ size: 64+queryBytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });const encoder = this.device.createCommandEncoder();encoder.copyBufferToBuffer(this.reductions, 0, buffer, 0, 64);if(this.queryResolve)encoder.copyBufferToBuffer(this.queryResolve,0,buffer,64,80);this.device.queue.submit([encoder.finish()]);
    try {
      await buffer.mapAsync(GPUMapMode.READ);
      const words=new Uint32Array(buffer.getMappedRange(0,64)),floats=new Float32Array(words.buffer,words.byteOffset,words.length);
      const volume=words[0]/1e9;if(volume>0){this.info.volumeCellSum=volume/this.info.cellSize_m**3;this.info.volumeDrift=(volume-this.initialVolume_m3)/Math.max(this.initialVolume_m3,1e-12);this.info.rawVolumeDrift=this.info.volumeDrift;}if(floats[1]>0)this.info.maxSpeed_m_s=floats[1];this.info.divergenceMax_s=floats[2];this.info.divergenceBefore_s=floats[12];this.info.pressureResidual=Math.sqrt(Math.max(0,floats[9]));this.info.pressureIterationsExecuted=words[14];if(floats[13]>0)this.info.maxSpeed_m_s=Math.max(this.info.maxSpeed_m_s??0,floats[13]);
      if(queryBytes){const times=new BigUint64Array(buffer.getMappedRange(64,80)),advection=Number(times[1]-times[0])/1e6,pressure=Number(times[3]-times[2])/1e6,projection=Number(times[5]-times[4])/1e6,rigid=Number(times[7]-times[6])/1e6,diagnostics=Number(times[9]-times[8])/1e6,total=advection+pressure+projection+rigid+diagnostics,span=Math.max(total,Number(times[9]-times[0])/1e6),overhead=Math.max(0,span-total);if(span>0){this.info.gpuTimings={advection_ms:advection,pressure_ms:pressure,projection_ms:projection,rigidCoupling_ms:rigid,diagnostics_ms:diagnostics,overhead_ms:overhead,total_ms:span};this.info.gpuStep_ms=span;}}
      buffer.unmap();
    } finally {buffer.destroy();this.readbackPending=false;}
    return this.info;
  }

  destroy(): void {
    this.destroyed=true;
    for (const buffer of [this.cellsA,this.cellsB,this.metadata,this.pageTable,this.params,this.rigidBuffer,this.rigidExchangeBuffer,this.reductions,this.indirectDispatchBuffer,this.pcgBuffer]) buffer.destroy();this.querySet?.destroy();this.queryResolve?.destroy();this.volumeTexture.destroy();
  }
}
