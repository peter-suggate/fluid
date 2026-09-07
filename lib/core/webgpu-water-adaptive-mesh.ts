/** An explicit triangle fits the existing two-vec4 payload: a.xyz/c.x,
 * b.xyz/c.y, with c.z in the otherwise unused packed cube x word. */
export const ADAPTIVE_SURFACE_TRIANGLE_CODE = 193;

/** Boundary-conforming agglomeration. Only a single, convex projected contour
 * with a monotone scalar axis may be simplified. All other groups
 * retain their unit children. Boundary segments use the unit contour's exact edge
 * interpolation and ambiguity convention, including its quantization. */
export const adaptiveSurfaceMeshWGSL = /* wgsl */ `
// Keep proof failures outside loops. Chrome/Dawn on Metal was observed to
// truncate segment collection and reject valid loops with conditional returns
// inside them. Accumulating the proof also keeps rejection free of partial emits.
const ADAPTIVE_SEGMENTS:u32=96u;
struct AdaptiveBoundary{a:array<vec3f,96>,b:array<vec3f,96>,count:u32,fault:bool}
fn adaptiveCrossing(a:vec3f,b:vec3f,va:f32,vb:f32)->vec3f{
  let sa=va-.5;let sb=vb-.5;let denominator=abs(sa)+abs(sb);
  if(denominator<=0.){return .5*(a+b);}
  let t=.5+round((.5*(abs(sa)-abs(sb))/denominator)*65536.)/65536.;
  return a+(b-a)*t;
}
fn adaptiveSegment(boundary:ptr<function,AdaptiveBoundary>,a:vec3f,b:vec3f){
  if(all(a==b)){return;}
  var duplicate=false;
  for(var i=0u;i<(*boundary).count;i+=1u){
    let previousA=(*boundary).a[i];let previousB=(*boundary).b[i];
    duplicate=duplicate||(all(a==previousA)&&all(b==previousB))
      ||(all(b==previousA)&&all(a==previousB));}
  if(duplicate){return;}
  let i=(*boundary).count;if(i>=ADAPTIVE_SEGMENTS){(*boundary).fault=true;return;}
  (*boundary).a[i]=a;(*boundary).b[i]=b;(*boundary).count=i+1u;
}
fn adaptiveFace(boundary:ptr<function,AdaptiveBoundary>,q:vec3i,axis:u32){
  let ta=(axis+1u)%3u;let tb=(axis+2u)%3u;
  var da=vec3i(0);da[ta]=1;var db=vec3i(0);db[tb]=1;
  let corners=array<vec3i,4>(q,q+da,q+da+db,q+db);
  var v:array<f32,4>;var point:array<vec3f,4>;var edges:array<u32,4>;var count=0u;
  for(var i=0u;i<4u;i+=1u){v[i]=occupancy(phi(corners[i]-vec3i(1)));}
  for(var i=0u;i<4u;i+=1u){let next=(i+1u)&3u;
    if((v[i]>=.5)!=(v[next]>=.5)){
      point[i]=adaptiveCrossing(vec3f(corners[i]),vec3f(corners[next]),v[i],v[next]);
      edges[count]=i;count+=1u;}}
  if(count==2u){adaptiveSegment(boundary,point[edges[0]],point[edges[1]]);}
  if(count==4u){
    let determinant=(v[0]-.5)*(v[2]-.5)-(v[1]-.5)*(v[3]-.5);
    var inside=true;if(determinant>0.){inside=v[0]>=.5;}else if(determinant<0.){inside=v[1]>=.5;}
    for(var i=0u;i<4u;i+=1u){if((v[i]>=.5)!=inside){
      adaptiveSegment(boundary,point[(i+3u)&3u],point[i]);}}
  }
}
fn emitAdaptiveTriangle(a:vec3f,b:vec3f,c:vec3f,axis:u32,positive:bool){
  atomicStore(&drawArgs.globalFineAuthorityLatch,1u);atomicMin(&drawArgs.vertexAllocator,0u);
  let slot=atomicAdd(&drawArgs.activeCubeCount,1u);
  if(slot>=arrayLength(&activeCubes)||slot*2u+1u>=arrayLength(&cubeValues)){return;}
  let descriptor=${ADAPTIVE_SURFACE_TRIANGLE_CODE}u|(axis<<8u)|select(1024u,0u,positive);
  activeCubes[slot]=vec2u(bitcast<u32>(c.z),descriptor<<16u);
  cubeValues[slot*2u]=vec4f(a,c.x);cubeValues[slot*2u+1u]=vec4f(b,c.y);
}
// Query the same finest-cell scalar that owns boundary crossings. The
// gradient is in lattice coordinates, so phi/|gradient| measures finest cells
// independently of the accepted physics-cell width and field scaling.
fn adaptiveFieldSample(lattice:vec3f)->vec4f{
  let point=lattice-vec3f(1.);let base=vec3i(floor(point));let t=point-vec3f(base);
  var result=vec4f(0.);
  for(var z=0;z<2;z+=1){for(var y=0;y<2;y+=1){for(var x=0;x<2;x+=1){
    let side=vec3i(x,y,z);let w=select(vec3f(1.)-t,t,side!=vec3i(0));
    let d=select(vec3f(-1.),vec3f(1.),side!=vec3i(0));let value=phi(base+side);
    result+=value*vec4f(w.x*w.y*w.z,d.x*w.y*w.z,w.x*d.y*w.z,w.x*w.y*d.z);
  }}}
  return result;
}
fn classifyAdaptiveGroup(base:vec3i,size:i32)->bool{
  let dims=vec3i(params.sampleDimensions);
  // Exact tank walls, floor films, missing pages, and native macro transitions
  // keep their existing owners. This decision applies to the whole group.
  if(any(base<=vec3i(1))||any(base+vec3i(size)>dims)){return false;}
  var positive=vec3<bool>(true);var negative=vec3<bool>(true);
  var wet=false;var dry=false;var validSamples=true;
  for(var z=0;z<=size;z+=1){for(var y=0;y<=size;y+=1){for(var x=0;x<=size;x+=1){
    if(!validSamples){continue;}
    let local=vec3i(x,y,z);let q=base+local-vec3i(1);let address=compactSampleAddress(q);
    if(address.x==INVALID||compactSampleSpanScale(address.x)!=1u||!fineValidAt(q)){validSamples=false;}
    let value=phi(q);wet=wet||value<=0.;dry=dry||value>0.;
    for(var axis=0u;axis<3u;axis+=1u){if(local[axis]<size){var next=q;next[axis]+=1;
      let delta=phi(next)-value;positive[axis]=positive[axis]&&delta>=0.;negative[axis]=negative[axis]&&delta<=0.;}}
  }}}
  if(!validSamples){return false;}
  if(!wet||!dry){return true;}
  var boundary:AdaptiveBoundary;
  for(var axis=0u;axis<3u;axis+=1u){for(var side=0;side<2;side+=1){
    for(var b=0;b<size;b+=1){for(var a=0;a<size;a+=1){var q=base;
      q[axis]+=side*size;q[(axis+1u)%3u]+=a;q[(axis+2u)%3u]+=b;
      adaptiveFace(&boundary,q,axis);
    }}}}
  let count=boundary.count;if(boundary.fault||count<3u){return false;}
  // Order the undirected loop; branching or multiple components fail closed.
  var points:array<vec3f,96>;var used:array<bool,96>;var ordered=true;
  points[0]=boundary.a[0];var current=boundary.b[0];used[0]=true;
  for(var i=1u;i<count;i+=1u){points[i]=current;var found=ADAPTIVE_SEGMENTS;var next=vec3f(0);
    if(!ordered){continue;}
    for(var j=1u;j<count;j+=1u){if(used[j]){continue;}
      if(all(boundary.a[j]==current)||all(boundary.b[j]==current)){
        if(found!=ADAPTIVE_SEGMENTS){ordered=false;}found=j;
        next=select(boundary.a[j],boundary.b[j],all(boundary.a[j]==current));}}
    if(found==ADAPTIVE_SEGMENTS){ordered=false;}else{used[found]=true;current=next;}
  }
  if(!ordered||any(current!=points[0])){return false;}
  var area=vec3f(0);var centre=vec3f(0);
  for(var i=0u;i<count;i+=1u){area+=cross(points[i]-vec3f(base),points[(i+1u)%count]-vec3f(base));centre+=points[i]-vec3f(base);}
  centre=vec3f(base)+centre/f32(count);
  var axis=0u;if(abs(area.y)>abs(area.x)){axis=1u;}if(abs(area.z)>abs(area[axis])){axis=2u;}
  if(abs(area[axis])<1e-6||(!positive[axis]&&!negative[axis])){return false;}
  // A convex projection makes this fan a non-overlapping graph. Reject a
  // folded/non-star-shaped patch instead of introducing intersecting triangles.
  var convex=true;
  for(var i=0u;i<count;i+=1u){let a=points[i];let b=points[(i+1u)%count];let c=points[(i+2u)%count];
    if(cross(b-a,c-b)[axis]*area[axis]< -1e-5){convex=false;}}
  if(!convex){return false;}
  // The average boundary vertex lies inside a curved surface. Keep its
  // convex projection but solve the monotone coordinate against the actual
  // scalar interpolant, instead of shrinking the surface to that average.
  var lower=centre;var upper=centre;lower[axis]=f32(base[axis]);upper[axis]=f32(base[axis]+size);
  let lowerPhi=adaptiveFieldSample(lower).x;let upperPhi=adaptiveFieldSample(upper).x;
  if((lowerPhi<=0.)==(upperPhi<=0.)){return false;}
  for(var iteration=0u;iteration<20u;iteration+=1u){
    let middle=.5*(lower+upper);let value=adaptiveFieldSample(middle).x;
    if((value<=0.)==(lowerPhi<=0.)){lower=middle;}else{upper=middle;}
  }
  centre=.5*(lower+upper);
  let snappedCentre=round(centre*65536.)/65536.;
  var oriented=true;
  var accurate=true;
  for(var i=0u;i<count;i+=1u){
    let a=round(points[i]*65536.)/65536.;let b=round(points[(i+1u)%count]*65536.)/65536.;
    // Non-strict monotonicity can contain a vertical zero plateau. Its
    // projected fan has zero area and cannot inherit a unique orientation.
    // Require a strictly oriented graph after the renderer's quantization.
    if(cross(b-a,snappedCentre-a)[axis]*area[axis]<=0.){oriented=false;}
    // Closure alone cannot justify long triangles across curved water. Bound
    // interior error in finest cells; failed groups follow the existing
    // dyadic subdivision path while planar groups keep their large fans.
    let midpoint=adaptiveFieldSample(.5*(a+snappedCentre));
    let interior=adaptiveFieldSample((a+b+snappedCentre)/3.);
    if(abs(midpoint.x)>.125*max(length(midpoint.yzw),1e-8)
      ||abs(interior.x)>.125*max(length(interior.yzw),1e-8)){accurate=false;}
  }
  if(!oriented||!accurate){return false;}
  for(var i=0u;i<count;i+=1u){emitAdaptiveTriangle(points[i],points[(i+1u)%count],centre,axis,positive[axis]);}
  return true;
}
`;
