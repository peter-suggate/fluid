/** Coarse nearest-source continuation for narrow-band FLIP. */
export const uniformVelocityExtrapolationShader = /* wgsl */ `
struct Params {
  dimsDt: vec4f,
  cellGravity: vec4f,
  container: vec4f,
  physical: vec4f,
  boundary: vec4f,
  inflowPositionRadius: vec4f,
  inflowVelocityLength: vec4f,
  inflowTiming: vec4f,
  tuning: vec4f,
  drop: vec4f,
  dropExtent: vec4f,



  twoLevel: vec4f,
}

struct FrontParams {
  sourceParity: u32,
  targetParity: u32,
  hierarchySourceUsesBaseDims: u32,
  hierarchyTargetUsesBaseDims: u32,
  activeLevel: u32,
  _padding0: u32,
  _padding1: u32,
  _padding2: u32,
}

@group(0) @binding(1) var densityIn: texture_3d<f32>;

@group(0) @binding(2) var<uniform> params: Params;

@group(0) @binding(3) var primaryIn: texture_3d<f32>;

@group(0) @binding(4) var secondaryIn: texture_3d<f32>;

@group(0) @binding(5) var primaryOut: texture_storage_3d<rgba32float, write>;

@group(0) @binding(7) var<uniform> frontParams: FrontParams;

@group(0) @binding(8) var faceOpenIn: texture_3d<f32>;

@group(0) @binding(12) var<storage, read_write> tileScratch: array<u32>;

@group(0) @binding(13) var sourceOrigins: texture_3d<u32>;

@group(0) @binding(14) var existingOrigins: texture_3d<u32>;

@group(0) @binding(15) var outputOrigins: texture_storage_3d<rgba32uint, write>;

override FRONT_RECEIPTS:bool=false;

override CACHE_FINE_HIERARCHY:bool=false;

var<workgroup> hierarchyValues:array<vec4f,64>;

var<workgroup> hierarchyLower:array<vec4u,64>;

var<workgroup> hierarchyUpper:array<vec4u,64>;

var<workgroup> hierarchyCacheBase:vec3i;

fn hierarchyCacheIndex(q:vec3i)->u32{let p=vec3u(q-hierarchyCacheBase);return p.x+4u*(p.y+4u*p.z);}

fn hierarchyInput(q:vec3i)->vec4f{
  if(CACHE_FINE_HIERARCHY){return hierarchyValues[hierarchyCacheIndex(q)];}
  return textureLoad(primaryIn,q,0);
}

fn hierarchyOrigin(q:vec3i,sd:vec3i,upper:bool)->vec4u{
  if(CACHE_FINE_HIERARCHY){let i=hierarchyCacheIndex(q);return select(hierarchyLower[i],hierarchyUpper[i],upper);}
  return textureLoad(sourceOrigins,q+select(vec3i(0),vec3i(0,0,sd.z),upper),0);
}

override FRONT_INITIAL_SWEEP:bool=false;

override COMPACT_SHELL: bool = false;

override REUSE_CONVERGED_DISTANCE: bool = false;

override SOURCE_AWARE_HIERARCHY: bool = false;

override SHELL_HIERARCHY: bool = false;

fn baseDims() -> vec3i { return vec3i(textureDimensions(densityIn)); }

fn tileDims() -> vec3i { return vec3i(textureDimensions(faceOpenIn)); }

fn coarseDims() -> vec3i { return (tileDims() + vec3i(3)) / 4; }

fn coarseIndex(t: vec3i) -> u32 { let c = coarseDims(); return u32(t.x + c.x * (t.y + c.y * t.z)); }

fn tileTableBase() -> u32 { let d = tileDims(); return u32(d.x * d.y * d.z); }

fn tiledExtension() -> bool { return params.twoLevel.y > 0.5; }

fn shellAt(p: vec3i) -> bool {
  if (!tiledExtension()) { return true; }
  let cell = clamp(p, vec3i(0), tileDims() - vec3i(1));
  return (tileScratch[tileTableBase() + 4u * coarseIndex(cell / 4) + 3u] & 2u) != 0u;
}

fn inBounds(p: vec3i, d: vec3i) -> bool {
  return all(p >= vec3i(0)) && all(p < d);
}

fn d4Sum8Vec2(value:array<vec2f,8>)->vec2f{
  let y0=(value[0]+value[5])+(value[1]+value[4]);
  let y1=(value[2]+value[7])+(value[3]+value[6]);
  return y0+y1;
}

fn bitFor(component: u32) -> u32 { return 1u << component; }

fn componentKnown(state: vec4f, component: u32) -> bool {
  return (u32(round(state.w)) & bitFor(component)) != 0u;
}

fn hierarchySourceDims() -> vec3i {
  return select(vec3i(textureDimensions(primaryIn)), baseDims(),
    frontParams.hierarchySourceUsesBaseDims != 0u);
}

fn hierarchyTargetDims() -> vec3i {
  return select(vec3i(textureDimensions(primaryOut)), baseDims(),
    frontParams.hierarchyTargetUsesBaseDims != 0u);
}

fn sourceKnownAt(q: vec3i) -> bool {
  return frontParams.hierarchySourceUsesBaseDims == 0u || shellAt(q);
}

struct NearestSample { value: f32, weight: f32, lower: u32, upper: u32 }

fn faceLocation(p: vec3i, dims: vec3i, component: u32) -> vec3f {
  var point = (vec3f(p) + vec3f(0.5)) * vec3f(baseDims()) / vec3f(dims);
  point[component] = f32(p[component]+1) * f32(baseDims()[component]) / f32(dims[component]);
  return point;
}

fn originalFace(origin: u32) -> vec3i {
  let d = vec3u(baseDims()); let index = origin - 1u;
  return vec3i(vec3u(index % d.x, (index / d.x) % d.y, index / (d.x*d.y)));
}

var<private> nearestSelection:u32;

fn nearestHierarchySample(p: vec3i, sd: vec3i, td: vec3i, component: u32, footprint: bool) -> NearestSample {
  nearestSelection=0u;
  let location = faceLocation(p, td, component);
  var sourcePosition = (vec3f(p)+vec3f(0.5))*vec3f(sd)/vec3f(td)-vec3f(0.5);
  sourcePosition[component] = f32(p[component]+1)*f32(sd[component])/f32(td[component])-1.0;
  let lower = select(vec3i(floor(sourcePosition)), 2*p, footprint);


  let spacing=params.cellGravity.xyz;let h=spacing/min(spacing.x,min(spacing.y,spacing.z));
  var best = 1e30;
  var distances: array<f32,8>;
  var lowers: array<u32,8>;var uppers: array<u32,8>;
  var values: array<f32,8>;
  for(var k=0u;k<8u;k++){
    let q = clamp(lower + vec3i(i32(k&1u),i32((k>>1u)&1u),i32(k>>2u)),vec3i(0),sd-vec3i(1));
    let state = hierarchyInput(q);
    distances[k] = 1e30;
    let offset=vec3i(i32(k&1u),i32((k>>1u)&1u),i32(k>>2u));
    let fraction=fract(sourcePosition);
    let weights=select(vec3f(1.0)-fraction,fraction,offset==vec3i(1));

    if ((!footprint && any(weights<=vec3f(0.0))) || !componentKnown(state,component) || !sourceKnownAt(q)) { continue; }
    var lower: u32;var upper:u32;
    if (frontParams.hierarchySourceUsesBaseDims != 0u) {
      let d = baseDims();lower = u32(q.x+d.x*(q.y+d.y*q.z))+1u;upper=lower;
    } else {
      lower=hierarchyOrigin(q,sd,false)[component];
      upper=hierarchyOrigin(q,sd,true)[component];
    }
    if (lower == 0u) { continue; }
    let lo=faceLocation(originalFace(lower),baseDims(),component);
    let hi=faceLocation(originalFace(upper),baseDims(),component);
    let delta=(location-clamp(location,lo,hi))*h;
    let distance=(delta.x*delta.x+delta.z*delta.z)+delta.y*delta.y;
    distances[k]=distance;lowers[k]=lower;uppers[k]=upper;values[k]=state[component];
    best=min(best,distance);
  }
  var contributions:array<vec2f,8>;
  var supportLower=baseDims();var supportUpper=vec3i(-1);
  let epsilon=1e-6*max(1.0,best);
  for(var k=0u;k<8u;k++){
    contributions[k]=vec2f(0.0);
    if(lowers[k]!=0u && abs(distances[k]-best)<=epsilon){
      contributions[k]=vec2f(values[k],1.0);nearestSelection|=1u<<k;
      supportLower=min(supportLower,originalFace(lowers[k]));supportUpper=max(supportUpper,originalFace(uppers[k]));
    }
  }
  let sum = d4Sum8Vec2(contributions);
  let d=baseDims();
  let lo=select(0u,u32(supportLower.x+d.x*(supportLower.y+d.y*supportLower.z))+1u,sum.y>0.0);
  let hi=select(0u,u32(supportUpper.x+d.x*(supportUpper.y+d.y*supportUpper.z))+1u,sum.y>0.0);
  return NearestSample(select(0.0,sum.x/sum.y,sum.y>0.0),sum.y,lo,hi);
}

var<workgroup> mixedHierarchyLanes:array<vec4u,192>;

fn mixedHierarchyCell(wid:vec3u,cell:u32)->vec3i{return vec3i(wid*4u+vec3u(cell%4u,(cell/4u)%4u,cell/16u));}

fn mixedHierarchyEntry(sample:NearestSample)->vec4u{
  return select(vec4u(0u),vec4u(bitcast<u32>(sample.value),sample.lower,sample.upper,1u),sample.weight>0.0);
}

@compute @workgroup_size(192)
fn mixedRestrictKnownVelocity(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lane:u32){
  let component=lane/64u;let cell=lane%64u;let p=mixedHierarchyCell(wid,cell);
  let sd=hierarchySourceDims();let td=hierarchyTargetDims();let inside=inBounds(p,td);
  var entry=vec4u(0u);
  if(inside){
    var result=nearestHierarchySample(p,sd,td,component,false);
    if(result.weight<=0.0&&component==1u){result=nearestHierarchySample(p,sd,td,component,true);}
    entry=mixedHierarchyEntry(result);
  }
  mixedHierarchyLanes[lane]=entry;
  workgroupBarrier();
  if(component!=0u||!inside){return;}
  let x=mixedHierarchyLanes[cell];let y=mixedHierarchyLanes[cell+64u];let z=mixedHierarchyLanes[cell+128u];
  textureStore(outputOrigins,p,vec4u(x.y,y.y,z.y,0u));
  textureStore(outputOrigins,p+vec3i(0,0,td.z),vec4u(x.z,y.z,z.z,0u));
  textureStore(primaryOut,p,vec4f(bitcast<f32>(x.x),bitcast<f32>(y.x),bitcast<f32>(z.x),f32(x.w|(y.w<<1u)|(z.w<<2u))));
}

@compute @workgroup_size(192)
fn mixedProlongUnknownVelocity(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lane:u32){
  let component=lane/64u;let cell=lane%64u;let p=mixedHierarchyCell(wid,cell);
  let sd=hierarchySourceDims();let td=hierarchyTargetDims();let inside=inBounds(p,td);
  var entry=vec4u(0u);var existing=vec4f(0.0);
  if(inside){
    existing=textureLoad(secondaryIn,p,0);
    entry=vec4u(bitcast<u32>(existing[component]),textureLoad(existingOrigins,p,0)[component],
      textureLoad(existingOrigins,p+vec3i(0,0,td.z),0)[component],0u);
    if(componentKnown(existing,component)){entry.w=1u;}
    else{
      let result=nearestHierarchySample(p,sd,td,component,false);
      if(result.weight>0.0){entry=mixedHierarchyEntry(result);entry.w|=nearestSelection<<1u;}
    }
  }
  mixedHierarchyLanes[lane]=entry;
  workgroupBarrier();
  if(component!=0u||!inside){return;}
  let x=mixedHierarchyLanes[cell];let y=mixedHierarchyLanes[cell+64u];let z=mixedHierarchyLanes[cell+128u];
  textureStore(outputOrigins,p,vec4u(x.y,y.y,z.y,(x.w>>1u)|((y.w>>1u)<<8u)|((z.w>>1u)<<16u)));
  textureStore(outputOrigins,p+vec3i(0,0,td.z),vec4u(x.z,y.z,z.z,0u));
  let mask=u32(round(existing.w))|(x.w&1u)|((y.w&1u)<<1u)|((z.w&1u)<<2u);
  textureStore(primaryOut,p,vec4f(bitcast<f32>(x.x),bitcast<f32>(y.x),bitcast<f32>(z.x),f32(mask)));
}

@compute @workgroup_size(192)
fn mixedReplayUnknownVelocity(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lane:u32){
 let component=lane/64u;let cell=lane%64u;let p=mixedHierarchyCell(wid,cell);
 let sd=hierarchySourceDims();let td=hierarchyTargetDims();let inside=inBounds(p,td);var value=0.0;var mask=0u;
 if(inside){
  let existing=textureLoad(secondaryIn,p,0);value=existing[component];mask=u32(round(existing.w));
  if(!componentKnown(existing,component)){
   let selected=(textureLoad(existingOrigins,p,0).w>>(8u*component))&255u;
   if(selected!=0u){
    var sourcePosition=(vec3f(p)+0.5)*vec3f(sd)/vec3f(td)-0.5;
    sourcePosition[component]=f32(p[component]+1)*f32(sd[component])/f32(td[component])-1.0;
    let lower=vec3i(floor(sourcePosition));var terms:array<vec2f,8>;
    for(var k=0u;k<8u;k++){
     terms[k]=vec2f(0);if((selected&(1u<<k))!=0u){let q=clamp(lower+vec3i(i32(k&1u),i32((k>>1u)&1u),i32(k>>2u)),vec3i(0),sd-1);terms[k]=vec2f(textureLoad(primaryIn,q,0)[component],1.0);}
    }
    let sum=d4Sum8Vec2(terms);value=sum.x/sum.y;mask|=1u<<component;
   }
  }
 }
 mixedHierarchyLanes[lane]=vec4u(bitcast<u32>(value),0u,0u,mask);workgroupBarrier();
 if(component!=0u||!inside){return;}
 let x=mixedHierarchyLanes[cell];let y=mixedHierarchyLanes[cell+64u];let z=mixedHierarchyLanes[cell+128u];
 textureStore(primaryOut,p,vec4f(bitcast<f32>(x.x),bitcast<f32>(y.x),bitcast<f32>(z.x),f32(x.w|y.w|z.w)));
}

const MIXED_TAIL_LEVELS:u32=16u;

var<workgroup> mixedTailValue:array<f32,768>;

var<workgroup> mixedTailLower:array<u32,768>;

var<workgroup> mixedTailUpper:array<u32,768>;

fn mixedTailCell(c:u32,d:vec3i)->vec3i{let e=vec3u(d);return vec3i(vec3u(c%e.x,(c/e.x)%e.y,c/(e.x*e.y)));}

fn mixedTailSlot(offset:u32,q:vec3i,d:vec3i,component:u32)->u32{return 3u*(offset+u32(q.x+d.x*(q.y+d.y*q.z)))+component;}

fn mixedTailSample(p:vec3i,sd:vec3i,td:vec3i,component:u32,footprint:bool,fromTexture:bool,tail:u32)->NearestSample {
  let location = faceLocation(p, td, component);
  var sourcePosition = (vec3f(p)+vec3f(0.5))*vec3f(sd)/vec3f(td)-vec3f(0.5);
  sourcePosition[component] = f32(p[component]+1)*f32(sd[component])/f32(td[component])-1.0;
  let lower = select(vec3i(floor(sourcePosition)), 2*p, footprint);
  let spacing=params.cellGravity.xyz;let h=spacing/min(spacing.x,min(spacing.y,spacing.z));
  var best = 1e30;
  var distances: array<f32,8>;
  var lowers: array<u32,8>;var uppers: array<u32,8>;
  var values: array<f32,8>;
  for(var k=0u;k<8u;k++){
    let q = clamp(lower + vec3i(i32(k&1u),i32((k>>1u)&1u),i32(k>>2u)),vec3i(0),sd-vec3i(1));
    var value=0.0;var first=0u;var last=0u;
    if(fromTexture){
      let state=textureLoad(primaryIn,q,0);
      if(componentKnown(state,component)){
        value=state[component];first=textureLoad(sourceOrigins,q,0)[component];last=textureLoad(sourceOrigins,q+vec3i(0,0,sd.z),0)[component];
      }
    }else{
      let slot=mixedTailSlot(tail,q,sd,component);value=mixedTailValue[slot];first=mixedTailLower[slot];last=mixedTailUpper[slot];
    }
    distances[k] = 1e30;
    let offset=vec3i(i32(k&1u),i32((k>>1u)&1u),i32(k>>2u));
    let fraction=fract(sourcePosition);
    let weights=select(vec3f(1.0)-fraction,fraction,offset==vec3i(1));
    if ((!footprint && any(weights<=vec3f(0.0))) || first == 0u) { continue; }
    let lo=faceLocation(originalFace(first),baseDims(),component);
    let hi=faceLocation(originalFace(last),baseDims(),component);
    let delta=(location-clamp(location,lo,hi))*h;
    let distance=(delta.x*delta.x+delta.z*delta.z)+delta.y*delta.y;
    distances[k]=distance;lowers[k]=first;uppers[k]=last;values[k]=value;
    best=min(best,distance);
  }
  var contributions:array<vec2f,8>;
  var supportLower=baseDims();var supportUpper=vec3i(-1);
  let epsilon=1e-6*max(1.0,best);
  for(var k=0u;k<8u;k++){
    contributions[k]=vec2f(0.0);
    if(lowers[k]!=0u && abs(distances[k]-best)<=epsilon){
      contributions[k]=vec2f(values[k],1.0);
      supportLower=min(supportLower,originalFace(lowers[k]));supportUpper=max(supportUpper,originalFace(uppers[k]));
    }
  }
  let sum = d4Sum8Vec2(contributions);
  let d=baseDims();
  let lo=select(0u,u32(supportLower.x+d.x*(supportLower.y+d.y*supportLower.z))+1u,sum.y>0.0);
  let hi=select(0u,u32(supportUpper.x+d.x*(supportUpper.y+d.y*supportUpper.z))+1u,sum.y>0.0);
  return NearestSample(select(0.0,sum.x/sum.y,sum.y>0.0),sum.y,lo,hi);
}

@compute @workgroup_size(256)
fn mixedHierarchyTail(@builtin(local_invocation_index) lane:u32){
  var dims:array<vec3i,MIXED_TAIL_LEVELS>;var offsets:array<u32,MIXED_TAIL_LEVELS>;
  var sd=vec3i(textureDimensions(primaryIn));var td=(sd+vec3i(1))/2;var count=0u;var offset=0u;

  loop{
    dims[count]=td;offsets[count]=offset;
    let cells=u32(td.x*td.y*td.z);
    for(var i=lane;i<3u*cells;i+=256u){
      let component=i%3u;let c=i/3u;let p=mixedTailCell(c,td);
      let finerTail=select(0u,offsets[max(count,1u)-1u],count>0u);
      var result=mixedTailSample(p,sd,td,component,false,count==0u,finerTail);
      if(result.weight<=0.0&&component==1u){result=mixedTailSample(p,sd,td,component,true,count==0u,finerTail);}
      let known=result.weight>0.0;let slot=3u*(offset+c)+component;
      mixedTailValue[slot]=select(0.0,result.value,known);mixedTailLower[slot]=select(0u,result.lower,known);mixedTailUpper[slot]=select(0u,result.upper,known);
    }
    workgroupBarrier();
    count++;offset+=cells;
    if(max(td.x,max(td.y,td.z))<=1||count>=MIXED_TAIL_LEVELS){break;}
    sd=td;td=(td+vec3i(1))/2;
  }

  for(var level=i32(count)-2;level>=0;level--){
    let finer=dims[level];let coarser=dims[level+1];let cells=u32(finer.x*finer.y*finer.z);
    for(var i=lane;i<3u*cells;i+=256u){
      let component=i%3u;let c=i/3u;let slot=3u*(offsets[level]+c)+component;
      if(mixedTailLower[slot]!=0u){continue;}
      let result=mixedTailSample(mixedTailCell(c,finer),coarser,finer,component,false,false,offsets[level+1]);
      if(result.weight>0.0){mixedTailValue[slot]=result.value;mixedTailLower[slot]=result.lower;mixedTailUpper[slot]=result.upper;}
    }
    workgroupBarrier();
  }
  let finest=dims[0];let cells=u32(finest.x*finest.y*finest.z);
  for(var c=lane;c<cells;c+=256u){
    let p=mixedTailCell(c,finest);let s=3u*c;
    let lower=vec3u(mixedTailLower[s],mixedTailLower[s+1u],mixedTailLower[s+2u]);
    let known=select(vec3u(0u),vec3u(1u,2u,4u),lower!=vec3u(0u));
    textureStore(outputOrigins,p,vec4u(lower,0u));
    textureStore(outputOrigins,p+vec3i(0,0,finest.z),vec4u(mixedTailUpper[s],mixedTailUpper[s+1u],mixedTailUpper[s+2u],0u));
    textureStore(primaryOut,p,vec4f(mixedTailValue[s],mixedTailValue[s+1u],mixedTailValue[s+2u],f32(known.x|known.y|known.z)));
  }
}
`;
