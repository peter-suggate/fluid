/** Continuous heightfield contact for NB-FLIP. Positions are cell coordinates;
 * velocities and normals are in metres. The solid library supplies the native
 * column texture, voxel mask and rigid-body queries. */
export const narrowBandTerrainContactWGSL = /* wgsl */ `
// Height and its two derivatives from the same bilinear patch. Column samples
// live at cell centres; interior planar inclines are reproduced exactly.
fn nbTerrainSurface(q:vec3f)->vec3f{return umTerrainSurface(q);}
fn nbTerrainNormal(s:vec3f)->vec3f{
 let h=params.hDt.xyz;return normalize(vec3f(-s.y/h.x,1.0/h.y,-s.z/h.z));
}
// One record lookup, terrain sample and body query each: Metal inlines every
// call, and this runs in advectParticles beside the trace.
fn nbParticleBlocked(q:vec3f)->bool{
 let c=vec3i(floor(q));let record=umBoundaryRecord(c);let terrain=umSolidHasTerrain();
 if(record==0u&&!terrain){return umSolidVoxel(c)||umBodyAt(umSolidTraceWorld(q))>=0;}
 var solid:bool;
 if(record!=0u){solid=umBoundarySampleAt(q,c,record).w<0.0;}else{solid=umSolidVoxel(c);}
 return solid||(terrain&&q.y<nbTerrainSurface(q).x)||umBodyAt(umSolidTraceWorld(q))>=0;
}
fn nbParticleCellOpen(c:vec3i)->bool{
 return umCellOpen(c)>=select(0.5,1e-5,umSolidHasTerrain()||umBoundaryRecord(c)!=0u);
}
fn nbVoxelBlocked(q:vec3f)->bool{
 return umSolidVoxel(vec3i(floor(q)))&&umBoundaryRecord(vec3i(floor(q)))==0u;
}
struct NBTerrainContact{position:vec3f,velocity:vec3f}
// Called on each trajectory substep (at most half a cell per axis). Find
// contact along the segment, then carry the remaining tangential displacement.
// The closest-point correction is in metres, including anisotropic lattices.
// Every surface sample is at one call site (Metal inlines each): stage 0 the
// end, 1 the start, 2 the bisection's twelve midpoints, 3 the hit, 4 the
// slide's up to four re-evaluations after it.
fn nbTerrainContact(startPoint:vec3f,endPoint:vec3f,velocity:vec3f)->NBTerrainContact{
 var q=endPoint;var v=velocity;let skin=0.001;
 var low=0.0;var high=1.0;var mid=0.0;var stage=0u;var i=0u;var at=endPoint;
 loop{
  let s=nbTerrainSurface(at);
  if(stage==0u){if(at.y-s.x>skin){return NBTerrainContact(q,v);}stage=1u;at=startPoint;continue;}
  if(stage==1u||stage==2u){
   if(stage==2u){if(at.y-s.x>skin){low=mid;}else{high=mid;}i++;}
   if((stage==1u&&at.y-s.x>skin)||(stage==2u&&i<12u)){stage=2u;mid=0.5*(low+high);at=mix(startPoint,endPoint,mid);continue;}
   stage=3u;at=mix(startPoint,endPoint,low);continue;
  }
  let n=nbTerrainNormal(s);
  if(stage==3u){
   var travel=(endPoint-at)*params.hDt.xyz;
   travel-=min(dot(travel,n),0.0)*n;
   q=at+travel/params.hDt.xyz;
   v-=min(dot(v,n),0.0)*n;
   // A segment can cross a patch edge; re-evaluate its normal after sliding.
   stage=4u;i=0u;at=q;continue;
  }
  // Vertical height difference is signed distance times 1/normal.y.
  let penetration=(skin+s.x-q.y)*params.hDt.y;
  if(penetration<=0.0){break;}
  q+=penetration*n.y*n/params.hDt.xyz;
  v-=min(dot(v,n),0.0)*n;
  i++;if(i>=4u){break;}at=q;
 }
 return NBTerrainContact(q,v);
}

// Earliest zero of a cubic in Bernstein form. Convex-hull rejection also
// catches a ridge crossed between two positive endpoints. Depth 14 bounds the
// contact parameter error by 2^-14. A depth-first walk of the halvings, left
// half first, that holds only the current node: its curve and its dyadic path
// code (range code/2^depth). A right sibling's curve is rebuilt from the root
// by the same halvings, so every node's coefficients are the ones a stack of
// halved curves would hold, and no private array raises the kernel's
// register count. The tree has fewer than 2^15 nodes: the walk ends.
fn nbBoundaryCubicHit(initial:vec4f)->f32{
 var b=initial;var code=0u;var depth=0u;
 loop{
  if(min(min(b.x,b.y),min(b.z,b.w))>0.0){
   loop{if(depth==0u||(code&1u)==0u){break;}code>>=1u;depth--;}
   if(depth==0u){return -1.0;}
   code|=1u;b=initial;
   for(var k=depth;k>0u;k--){
    let a=0.5*(b.xyz+b.yzw);let c=0.5*(a.xy+a.yz);let mid=0.5*(c.x+c.y);
    b=select(vec4f(b.x,a.x,c.x,mid),vec4f(mid,c.y,a.z,b.w),((code>>(k-1u))&1u)!=0u);
   }
   continue;
  }
  if(b.x<=0.0||depth==14u){return f32(code)/f32(1u<<depth);}
  let a=0.5*(b.xyz+b.yzw);let c=0.5*(a.xy+a.yz);let mid=0.5*(c.x+c.y);
  b=vec4f(b.x,a.x,c.x,mid);code<<=1u;depth++;
 }
}
fn nbBoundaryHit(start:vec3f,end:vec3f)->f32{
 let delta=end-start;var t=0.0;
 // The caller's half-cell strides cross at most three cell faces.
 for(var step=0u;step<5u&&t<1.0;step++){
  let c=vec3i(floor(mix(start,end,t)+sign(delta)*1e-6));var stop=1.0;
  for(var a=0u;a<3u;a++){
   if(abs(delta[a])<1e-12){continue;}
   let face=f32(c[a])+select(0.0,1.0,delta[a]>0.0);
   let crossing=(face-start[a])/delta[a];if(crossing>t+1e-7){stop=min(stop,crossing);}
  }
  let record=umBoundaryRecord(c);
  if(record!=0u){
   let p=mix(start,end,t);let d=(stop-t)*delta;
   let f0=umBoundarySampleAt(p,c,record).w;let f1=umBoundarySampleAt(p+d/3.0,c,record).w;
   let f2=umBoundarySampleAt(p+2.0*d/3.0,c,record).w;let f3=umBoundarySampleAt(p+d,c,record).w;
   let hit=nbBoundaryCubicHit(vec4f(f0,3.0*f1-1.5*f2-5.0*f0/6.0+f3/3.0,3.0*f2-1.5*f1-5.0*f3/6.0+f0/3.0,f3));
   if(hit>=0.0){return mix(t,stop,hit);}
  }
  t=stop;
 }
 return -1.0;
}
fn nbProceduralContact(start:vec3f,end:vec3f,velocity:vec3f)->NBTerrainContact{
 if(umBoundaryBase()==0u){return NBTerrainContact(end,velocity);}
 let hit=nbBoundaryHit(start,end);if(hit<0.0){return NBTerrainContact(end,velocity);}
 let at=mix(start,end,hit);let s=umBoundarySample(at);let gradient=s.xyz/params.hDt.xyz;
 if(dot(gradient,gradient)<1e-12){return NBTerrainContact(start,velocity);}
 let n=normalize(gradient);var travel=(end-at)*params.hDt.xyz;
 travel-=min(dot(travel,n),0.0)*n;var q=at+travel/params.hDt.xyz;
 var v=velocity-min(dot(velocity,n),0.0)*n;
 for(var i=0u;i<4u;i++){
  let sample=umBoundarySample(q);let g=sample.xyz/params.hDt.xyz;let lengthG=length(g);
  if(lengthG<1e-12){break;}let normal=g/lengthG;
  let depth=0.001*min(params.hDt.x,min(params.hDt.y,params.hDt.z))-sample.w/lengthG;
  if(depth<=0.0){break;}q+=depth*normal/params.hDt.xyz;v-=min(dot(v,normal),0.0)*normal;
 }
 // A field face on a container wall would push the sample out of storage;
 // it stops at the hit instead.
 if(!umPointPresent(q)){return NBTerrainContact(at,v);}
 return NBTerrainContact(q,v);
}
`;

/** Moving rigid contact. A body may have overtaken a sample since the last
 * fluid step. Retain it, move it outside the analytic surface, and enforce
 * non-penetration relative to the wall velocity (including rotation). */
export const narrowBandRigidContactWGSL = /* wgsl */ `
struct NBRigidContact{position:vec3f,velocity:vec3f}
fn nbRigidContact(point:vec3f,velocity:vec3f)->NBRigidContact{
 let h=params.hDt.xyz;let skin=0.01*min(h.x,min(h.y,h.z));
 var world=umSolidTraceWorld(point);var v=velocity;
 for(var i=0u;i<umBodyCount();i++){
  if(!umBodyNear(i,world,skin)){continue;}
  for(var iteration=0u;iteration<4u;iteration++){
   let distance=umBodyDistance(i,world);if(distance>skin){break;}
   var gradient=vec3f(0);
   for(var axis=0u;axis<3u;axis++){
    var delta=vec3f(0);delta[axis]=skin;
    gradient[axis]=umBodyDistance(i,world+delta)-umBodyDistance(i,world-delta);
   }
   // A sample exactly at a shape's medial axis still needs a deterministic exit.
   var normal=vec3f(0,1,0);let g=length(gradient);if(g>1e-10){normal=gradient/g;}
   world+=(skin-distance)*normal;
   let wall=umBodyVelocity(i,world);
   v-=min(dot(v-wall,normal),0.0)*normal;
  }
 }
 return NBRigidContact(point+(world-umSolidTraceWorld(point))/h,v);
}
`;
