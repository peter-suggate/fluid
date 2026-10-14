/** Continuous heightfield contact for NB-FLIP. Positions are cell coordinates;
 * velocities and normals are in metres. The solid library supplies the native
 * column texture, voxel mask and rigid-body queries. */
export const narrowBandTerrainContactWGSL = /* wgsl */ `
// Height and its two derivatives from the same bilinear patch. Column samples
// live at cell centres; interior planar inclines are reproduced exactly.
fn nbTerrainSurface(q:vec3f)->vec3f{
 let high=vec2f(umSolidDimensions().xz)-vec2f(1);
 let p=clamp(q.xz-vec2f(0.5),vec2f(0),high);
 let a=vec2i(floor(p));let f=fract(p);
 let h00=umSolidTerrainHeight(a.x,a.y);let h10=umSolidTerrainHeight(a.x+1,a.y);
 let h01=umSolidTerrainHeight(a.x,a.y+1);let h11=umSolidTerrainHeight(a.x+1,a.y+1);
 let height=mix(mix(h00,h10,f.x),mix(h01,h11,f.x),f.y);
 let slope=vec2f(mix(h10-h00,h11-h01,f.y),mix(h01-h00,h11-h10,f.x));
 let interior=(q.xz>=vec2f(0.5))&(q.xz<=high+vec2f(0.5));
 return vec3f(height,select(vec2f(0),slope,interior));
}
fn nbTerrainNormal(s:vec3f)->vec3f{
 let h=params.hDt.xyz;return normalize(vec3f(-s.y/h.x,1.0/h.y,-s.z/h.z));
}
fn nbParticleBlocked(q:vec3f)->bool{
 if(umBoundaryRecord(vec3i(floor(q)))!=0u){return umBoundarySample(q).w<0.0||(umSolidHasTerrain()&&q.y<nbTerrainSurface(q).x)||umBodyAt(umSolidTraceWorld(q))>=0;}
 if(!umSolidHasTerrain()){return umCellOpen(vec3i(floor(q)))<0.5;}
 return umSolidVoxel(vec3i(floor(q)))||q.y<nbTerrainSurface(q).x
  ||umBodyAt(umSolidTraceWorld(q))>=0;
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
fn nbTerrainContact(startPoint:vec3f,endPoint:vec3f,velocity:vec3f)->NBTerrainContact{
 var q=endPoint;var v=velocity;let skin=0.001;
 let end=nbTerrainSurface(q);
 if(q.y-end.x>skin){return NBTerrainContact(q,v);}
 var low=0.0;var high=1.0;
 if(startPoint.y-nbTerrainSurface(startPoint).x>skin){
  for(var i=0u;i<12u;i++){
   let mid=0.5*(low+high);let p=mix(startPoint,endPoint,mid);
   if(p.y-nbTerrainSurface(p).x>skin){low=mid;}else{high=mid;}
  }
 }
 let hit=mix(startPoint,endPoint,low);let n=nbTerrainNormal(nbTerrainSurface(hit));
 var travel=(endPoint-hit)*params.hDt.xyz;
 travel-=min(dot(travel,n),0.0)*n;
 q=hit+travel/params.hDt.xyz;
 v-=min(dot(v,n),0.0)*n;
 // A segment can cross a patch edge; re-evaluate its normal after sliding.
 for(var i=0u;i<4u;i++){
  let surface=nbTerrainSurface(q);let normal=nbTerrainNormal(surface);
  let penetration=(skin+surface.x-q.y)*params.hDt.y;
  if(penetration<=0.0){break;}
  // Vertical height difference is signed distance times 1/normal.y.
  q+=penetration*normal.y*normal/params.hDt.xyz;
  v-=min(dot(v,normal),0.0)*normal;
 }
 return NBTerrainContact(q,v);
}

// Earliest zero of a cubic in Bernstein form. Convex-hull rejection also
// catches a ridge crossed between two positive endpoints. Fixed DFS storage
// bounds shader work; depth 14 bounds the contact parameter error by 2^-14.
fn nbBoundaryCubicHit(initial:vec4f)->f32{
 var curves:array<vec4f,16>;var ranges:array<vec2f,16>;var depths:array<u32,16>;
 curves[0]=initial;ranges[0]=vec2f(0,1);depths[0]=0u;var count=1u;
 for(var visit=0u;visit<32768u&&count>0u;visit++){
  count--;let b=curves[count];let range=ranges[count];let depth=depths[count];
  if(min(min(b.x,b.y),min(b.z,b.w))>0.0){continue;}
  if(b.x<=0.0||depth==14u){return range.x;}
  let a=0.5*(b.xyz+b.yzw);let c=0.5*(a.xy+a.yz);let mid=0.5*(c.x+c.y);let t=0.5*(range.x+range.y);
  curves[count]=vec4f(mid,c.y,a.z,b.w);ranges[count]=vec2f(t,range.y);depths[count]=depth+1u;count++;
  curves[count]=vec4f(b.x,a.x,c.x,mid);ranges[count]=vec2f(range.x,t);depths[count]=depth+1u;count++;
 }
 return -1.0;
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
 return NBTerrainContact(q,v);
}
`;
