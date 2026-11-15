/**
 * setCensus (brick-abi.md §2.5, lane E part 1): NB-FLIP's tile-tier census, the old engine's
 * Uniform_dynamic_ownership_census module as resolved for high-resolution-dam-break
 * (fluid-scratch/lane-f/wgsl-dam/063-Uniform_dynamic_ownership_census.wgsl; line numbers below are that
 * file's), ported on the brick storage. It reads PHI, US, CMASK and TILE_OCC, and writes WANT (16 words a C
 * slot, h iff the tile is band) and TRIG (a word an F slot, bit j: tile j's importance triggers, word 1 bits
 * 16-21, nbActivityTriggered).
 *
 * Passes (brickCensusSchedule), each the old kernel of the same rule:
 * - censusBegin: the census header words (H 224..231) clear; unsupported state fails.
 * - censusFine: classify (l.2091), a workgroup an F brick, its h tiles one after another, a lane a cell.
 * - censusCoarse: classifyCoarse (l.2175), a workgroup a C page, its 4h tiles 64 at a time, a lane a tile,
 *   with the queued face, error and depth jobs.
 * - censusCubes: boundCubes (l.2625).
 * - censusImportance: importanceTile and publishImportance (l.2342, l.2405), then interfaceBits (l.2534).
 * - censusDecide: decideTile (l.2839), then the residency clamp, WANT and TRIG.
 *
 * Verbatim: every rule and its arithmetic, with the dam's policy baked (CENSUS_POLICY: the values the old
 * encode wrote for the dam, uniform-mixed-dynamic.ts l.2358). Loads go through the brick storage:
 * - φ: nbVertexPhi (the vertex authority, brick-owner.wgsl.ts) times h, so the old metre arithmetic runs as
 *   written (umLoadFine, umLoadCorner).
 * - Velocity (US): the old texel of cell q on axis a is the face between q and q + e_a, which is
 *   nbFacePlane(q + e_a, a) (udLoad_velocity and udLoad4_velocity alike).
 * - Kinds: a cell or tile outside the lattice is the old SOLID page (a wall: umWallBeyond, umFaceLow,
 *   umDepthSample, umApproach); a page inside it that is not resident is AIR; BULK16 pages do not exist at M2.
 *
 * Deviations, all stated:
 * - Deep COLD is taken as D = 0 (no replay, no CC_DECIDED or CC_STATIC carried between censuses): every
 *   tile is classified and decided every census. brick-census.test.ts checks that the decisions of tiles
 *   that are not deep are the same with D = 0 (they never read D).
 * - No solids at M2: every census open mask is all cells (umCensusOpenCells), so the cut branches
 *   (umPlaneBuriedCorners, buried taps, closed edges) drop, and umObstacleAt is false. A body or solid edit
 *   fails explicitly (NB_FAIL_UNSUPPORTED).
 * - Storage: a 4h face toward a page that is not resident reads the brick storage (its C face below, 0
 *   above), where the old engine read its own texel. A face of a non-resident brick reads 0.
 * - Dead under the dam's policy and not ported: forwardFine (fastTravel 0) and its travel words, holds,
 *   the budget, reasons, strain and rotation, sources, regions, and the volume-fraction branches (surface
 *   authority). umResolutionError's h-tile branch is dead under shapeDisplacement.
 * - Residency clamp (lane E): a tile may be h in WANT only when its F brick and every in-range
 *   27-neighbour of it are resident. A clamped tile appends its missing bricks as candidates (setMark's
 *   append) and refines one census late.
 * - Counters: the old census words that feed no decision are not kept; the kept ones are H.CENSUS_*.
 */
import { ARGS, BrickLayout, H, LIST, brickAbiWGSL } from "./brick-abi";
import { CFG, type BrickBinding, type BrickKernel, declareBindings } from "./brick-kernel";
import { brickOwnerWGSL } from "./brick-owner.wgsl";

/** The census header words (set buffer, brick-abi.md §5.1). */
export const CENSUS_H = {
  /** Band tiles the residency clamp held at 4h this census. */ CENSUS_CLAMPED: 224,
  /** Candidate bricks the clamp appended. */ CENSUS_CAND: 225,
  /** The largest tile travel, h per step, as f32 bits (the old census[18]). */ CENSUS_REACH: 226,
  /** Crossing tiles (census[0]). */ CENSUS_CROSSING: 227,
  /** Required tiles (census[12]). */ CENSUS_REQUIRED: 228,
  /** Band tiles (census[2]). */ CENSUS_BAND: 229,
  /** Band tiles that are 4h now (census[4]). */ CENSUS_REFINED: 230,
  /** Tiles whose importance triggers (TRIG bits). */ CENSUS_TRIGGERED: 231,
} as const;

/** The dam's policy, as the old encode wrote it (uniform-mixed-dynamic.ts l.2358; brick-abi.md §2.5). */
export const CENSUS_POLICY = {
  /** step.y, shapeTolerance (h). */ tolerance: 0.5,
  /** surface.x, fastTravel. */ fastTravel: 0,
  /** surface.y, boundaryTravel. */ boundaryTravel: 1,
  /** importance.x, thin (h). */ thin: 4,
  /** importance.w, approach horizon (steps). */ approach: 2,
  /** flow.w, UNIFORM_MIXED_DYNAMIC_SURFACE_DRIFT. */ drift: 0.5,
  /** shaping.x criteria bits: shape 0, thin 1, impact 4, approach 5. */ criteria: 0b110011,
} as const;

/** Scratch words a C slot (the lane buffer `census`): per tile its velocity keys, classify's importance words,
 * the final word 1, its crossing cells and gap; per page its bound cubes and interface bits. */
export const CENSUS = {
  KEYS: 0, CUBES: 3072, W0: 3520, W1: 4032, FIN: 4544, CROSS: 5056, GAP: 6080, REQ: 6592, STRIDE: 6656,
} as const;
/** The lane buffers setCensus binds: `census`, CENSUS.STRIDE words a C slot. */
export const brickCensusBuffers = (L: BrickLayout): Record<string, number> => ({ census: CENSUS.STRIDE * L.cap.c });

const f = (x: number) => (Number.isInteger(x) ? x.toFixed(1) : String(x));

/** The census header constants, unless brick-abi.ts declares them already. */
const headerWGSL = () => Object.entries(CENSUS_H).map(([name, v]) => {
  const declared = (H as Record<string, number>)[name];
  if (declared !== undefined && declared !== v) throw new Error(`setCensus: H.${name} is ${declared}, the census uses ${v}`);
  return declared === undefined ? `const NB_H_${name}:u32=${v}u;` : "";
}).join("\n");

/**
 * The census's rules over hooks, so the kernels and the parity tests run one text. The includer defines:
 * ncH(), ncDt() (metres, seconds), ncUp() (the old reach.w), ncLoadFine(p), ncLoadCorner(p) (φ in metres at
 * h vertex p), ncWidePresent(v) (umWideVertexPresent of wide vertex v), ncCellKind(c) (0 stored, 1 AIR,
 * 2 SOLID), ncCellOwnerH(c), ncFace(q,a) (the old velocity texel), ncObstacleAt(q), ncWallBeyond(p,k),
 * ncTileState(q) (0 stored, 1 AIR, 2 closed), and the scratch readers ncPageSlot(page), ncListCount(),
 * ncListSlot(i), ncSlotPage(s), ncKeys(s,i), ncW1(s,i), ncFin(s,i), ncCross(s,i), ncReq(s,w), ncCube(s,j,c).
 */
export const CENSUS_FORMULAS = /* wgsl */`
const NC_TOL:f32=${f(CENSUS_POLICY.tolerance)};
const NC_FAST:f32=${f(CENSUS_POLICY.fastTravel)};
const NC_BOUNDARY:f32=${f(CENSUS_POLICY.boundaryTravel)};
const NC_THIN:f32=${f(CENSUS_POLICY.thin)};
const NC_APPROACH:f32=${f(CENSUS_POLICY.approach)};
const NC_DRIFT:f32=${f(CENSUS_POLICY.drift)};
const NC_CRITERIA:u32=${CENSUS_POLICY.criteria}u;
// l.1185-1210 under the dam's policy: no reasons, no hold (scoreFloor 1), scoreAll (scoreShaped), thinWanted.
const NC_SHAPE:u32=0u;const NC_THIN_K:u32=1u;const NC_IMPACT:u32=4u;const NC_APPROACH_K:u32=5u;
const NC_SCORE_FLOOR:f32=1.0;
const NC_CAP:u32=16u;
const NC_NO_WALL:u32=0xffffu;
const NC_NO_METRIC:u32=0x7f7fffffu;
const NC_VOLUME_SHAPE:f32=4.0;
const NC_MOVING_SECONDS:f32=0.5;
const NC_SAMPLE_REACH:f32=2.5;
const NC_REQUIRED:u32=33554432u;const NC_CROSSES:u32=268435456u;const NC_WET:u32=536870912u;
fn ncCorner(k:u32,side:u32)->vec3u{return vec3u(k%side,(k/side)%side,k/(side*side));}
fn ncCriterionOn(k:u32)->bool{return ((NC_CRITERIA>>k)&1u)!=0u;}
fn ncTriggerBit(k:u32)->u32{return 1u<<(16u+k);}
fn ncScoreByte(score:f32)->u32{return u32(clamp(score*64.0,0.0,255.0));}
// orderKey, orderValue (l.1214).
fn ncOrderKey(x:f32)->u32{let b=bitcast<u32>(x);return select(b|0x80000000u,~b,(b&0x80000000u)!=0u);}
fn ncOrderValue(k:u32)->f32{return bitcast<f32>(select(~k,k&0x7fffffffu,(k&0x80000000u)!=0u));}
fn ncTileLocal(q:vec3i)->u32{let l=vec3u(q&vec3i(7));return l.x|(l.y<<3u)|(l.z<<6u);}
// TileClass (l.1221).
struct NcClass{low:array<u32,3>,high:array<u32,3>,gap:array<u32,6>,reach:array<u32,6>,flags:u32,error:u32,crossing:vec2u,deep:vec2u}
fn ncEmptyClass()->NcClass{return NcClass(array<u32,3>(0xffffffffu,0xffffffffu,0xffffffffu),array<u32,3>(0u,0u,0u),array<u32,6>(15u,15u,15u,15u,15u,15u),array<u32,6>(15u,15u,15u,15u,15u,15u),0u,0u,vec2u(0u),vec2u(0u));}
fn ncIdentityKeys()->array<u32,6>{return array<u32,6>(0xffffffffu,0xffffffffu,0xffffffffu,0u,0u,0u);}

// umSurfaceDisplacement's workgroup state (l.1231-1243, 1277).
var<workgroup> ncTileSigned:atomic<i32>;
var<workgroup> ncTileCrossings:atomic<u32>;
var<workgroup> ncTileCurve:atomic<i32>;
var<workgroup> ncTileCurves:atomic<u32>;
var<workgroup> ncTileCorner:atomic<u32>;
var<workgroup> ncTileRaw:array<f32,125>;
var<workgroup> ncTileSmooth:array<f32,125>;
var<workgroup> ncCornerMetric:array<atomic<u32>,8>;
fn ncSmoothAt(v:vec3u)->f32{return ncTileSmooth[v.x+5u*v.y+25u*v.z];}
fn ncRawAt(v:vec3i)->f32{return ncTileRaw[v.x+5*v.y+25*v.z];}
// umSmoothCurvature (l.1281).
fn ncSmoothCurvature(p:vec3u,centre:f32)->vec2f{
 var g:vec3f;var d:vec3f;var m:vec3f;let h=ncH();
 for(var a=0u;a<3u;a++){
  var lo=p;lo[a]-=1u;var hi=p;hi[a]+=1u;let below=ncSmoothAt(lo);let above=ncSmoothAt(hi);
  g[a]=(above-below)/(2.0*h);d[a]=(above-2.0*centre+below)/(h*h);
  let b=(a+1u)%3u;
  var pp=p;pp[a]+=1u;pp[b]+=1u;var pm=p;pm[a]+=1u;pm[b]-=1u;var mp=p;mp[a]-=1u;mp[b]+=1u;var mm=p;mm[a]-=1u;mm[b]-=1u;
  m[a]=(ncSmoothAt(pp)-ncSmoothAt(pm)-ncSmoothAt(mp)+ncSmoothAt(mm))/(4.0*h*h);
 }
 let gg=dot(g,g);if(gg<1.0e-4){return vec2f(0.0);}
 let ghg=dot(g*g,d)+2.0*(g.x*g.y*m.x+g.y*g.z*m.y+g.z*g.x*m.z);
 return vec2f((gg*(d.x+d.y+d.z)-ghg)/(gg*sqrt(gg)),1.0);
}
// umStageSmoothTile (l.1313), every cell open: no buried taps.
fn ncStageSmoothTile(origin:vec3i,lane:u32){
 for(var i=lane;i<125u;i+=64u){ncTileRaw[i]=ncLoadFine(origin+vec3i(ncCorner(i,5u)));}
 workgroupBarrier();
 for(var i=lane;i<125u;i+=64u){
  let v=ncCorner(i,5u);
  let r=select(vec3u(0u),vec3u(1u),(v>vec3u(0u))&(v<vec3u(4u)));
  var sum=0.0;var weight=0.0;
  for(var z=v.z-r.z;z<=v.z+r.z;z++){for(var y=v.y-r.y;y<=v.y+r.y;y++){for(var x=v.x-r.x;x<=v.x+r.x;x++){
   let at=x+5u*y+25u*z;
   let w=select(1.0,2.0,x==v.x)*select(1.0,2.0,y==v.y)*select(1.0,2.0,z==v.z);
   sum+=w*ncTileRaw[at];weight+=w;
  }}}
  ncTileSmooth[i]=sum/weight;
 }
 workgroupBarrier();
}
// umMetricCorners (l.1437), every edge open and no corner buried.
fn ncMetricCorners(lane:u32,c:ptr<function,array<f32,8>>){
 if(lane<8u){atomicStore(&ncCornerMetric[lane],NC_NO_METRIC);}
 workgroupBarrier();
 if(lane<24u){
  let k=lane/3u;let a=lane%3u;let corner=vec3i(4u*ncCorner(k,2u));
  let inward=select(vec3i(1),vec3i(-1),corner==vec3i(4));
  var step=vec3i(0);step[a]=inward[a];
  for(var j=0;j<4;j++){
   let v=corner+j*step;let n=v+step;
   let u=ncRawAt(v);let w=ncRawAt(n);
   if((u<0.0)==(w<0.0)){continue;}
   let f=u/(u-w);
   var g=vec3f(0.0);g[a]=w-u;
   for(var b=0u;b<3u;b++){
    if(b==a){continue;}
    var t=vec3i(0);t[b]=inward[b];let p=v+t;let q=n+t;
    g[b]=f32(inward[b])*mix(ncRawAt(p)-u,ncRawAt(q)-w,f);
   }
   let along=abs(g[a])/max(length(g),1e-20);
   atomicMin(&ncCornerMetric[k],bitcast<u32>((f32(j)+f)*along));
   break;
  }
 }
 workgroupBarrier();
 for(var k=0u;k<8u;k++){
  let m=atomicLoad(&ncCornerMetric[k]);
  if(m!=NC_NO_METRIC&&(*c)[k]!=0.0){(*c)[k]=sign((*c)[k])*bitcast<f32>(m)*ncH();}
 }
}
// umCornerDistance (l.1473).
fn ncCornerDistance(v:vec3u,a:u32,f:f32,low:vec4f,high:vec4f,dx:vec4f)->f32{
 var t=vec3f(v)*0.25;t[a]+=0.25*f;
 let x0=mix(low,high,t.x);
 let y0=mix(vec2f(x0.x,x0.z),vec2f(x0.y,x0.w),t.y);
 let gx=mix(vec2f(dx.x,dx.z),vec2f(dx.y,dx.w),t.y);
 let gradient=vec3f(mix(gx.x,gx.y,t.z),mix(x0.y-x0.x,x0.w-x0.z,t.z),y0.y-y0.x);
 return 4.0*mix(y0.x,y0.y,t.z)/max(length(gradient),4.0e-3*ncH());
}
// umSurfaceDisplacement (l.1481) of the h tile at tile coordinate t, every cell open. Every lane calls it.
fn ncSurfaceDisplacement(t:vec3i,lane:u32)->f32{
 let origin=4*t;let h=ncH();
 ncStageSmoothTile(origin,lane);
 var c:array<f32,8>;var worst=0.0;var along=0.0;var crossings=0u;var curve=0.0;var curves=0u;
 for(var k=0u;k<8u;k++){c[k]=ncLoadCorner(origin+4*vec3i(ncCorner(k,2u)));}
 let held=c;ncMetricCorners(lane,&c);
 let low=vec4f(c[0],c[2],c[4],c[6]);let high=vec4f(c[1],c[3],c[5],c[7]);let dx=high-low;
 for(var i=lane;i<125u;i+=64u){
  let v=ncCorner(i,5u);let value=ncTileSmooth[i];
  if(all(v>=vec3u(1u))&&all(v<=vec3u(3u))&&abs(value)<=1.5*h){
   let kappa=ncSmoothCurvature(v,value);
   if(kappa.y>0.0){curve+=kappa.x*(4.0/3.0)*h;curves++;}
  }
  if(all(v%4u==vec3u(0u))){
   let k=v/4u;let centre=held[k.x+2u*k.y+4u*k.z];
   if(abs(centre)<=2.0*h){
    var lap=0.0;
    for(var a=0u;a<3u;a++){
     var lo=t+vec3i(k);lo[a]-=1;var hi=t+vec3i(k);hi[a]+=1;
     if(!ncWidePresent(lo)||!ncWidePresent(hi)){continue;}
     let below=ncLoadCorner(4*lo);let above=ncLoadCorner(4*hi);
     if(abs(below-centre)>6.0*h||abs(above-centre)>6.0*h){continue;}
     lap+=above-2.0*centre+below;
    }
    atomicMax(&ncTileCorner,bitcast<u32>(abs(lap)/(12.0*h)));
   }
  }
  for(var a=0u;a<3u;a++){
   if(v[a]>=4u){continue;}
   var n=v;n[a]+=1u;
   let next=ncSmoothAt(n);
   if((value<0.0)!=(next<0.0)){
    let d=ncCornerDistance(v,a,value/(value-next),low,high,dx);worst=max(worst,abs(d));
   }
   let raw=ncTileRaw[v.x+5u*v.y+25u*v.z];let rawNext=ncTileRaw[n.x+5u*n.y+25u*n.z];
   if((raw<0.0)!=(rawNext<0.0)){along+=ncCornerDistance(v,a,raw/(raw-rawNext),low,high,dx);crossings++;}
  }
 }
 if(crossings!=0u){atomicAdd(&ncTileSigned,i32(clamp(along,-1.0e5,1.0e5)*1024.0));atomicAdd(&ncTileCrossings,crossings);}
 if(curves!=0u){atomicAdd(&ncTileCurve,i32(clamp(curve,-1.0e5,1.0e5)*1024.0));atomicAdd(&ncTileCurves,curves);}
 return worst;
}
// umVolumeError, umCurveError, umCornerCurveError (l.1549-1572).
fn ncVolumeError()->f32{let n=atomicLoad(&ncTileCrossings);if(n==0u){return 0.0;}return NC_VOLUME_SHAPE*abs(f32(atomicLoad(&ncTileSigned))/(1024.0*f32(n)));}
fn ncCurveError()->f32{let n=atomicLoad(&ncTileCurves);if(n==0u){return 0.0;}return NC_VOLUME_SHAPE*abs(f32(atomicLoad(&ncTileCurve))/(1024.0*f32(n)));}
fn ncCornerCurveError()->f32{return NC_VOLUME_SHAPE*bitcast<f32>(atomicLoad(&ncTileCorner));}
// umMovingTravel, umClassMoving (l.1580).
fn ncMovingTravel()->f32{return 4.0*ncDt()/NC_MOVING_SECONDS;}
fn ncClassMoving(c:NcClass)->bool{
 var speed=0.0;
 for(var a=0u;a<3u;a++){
  if(c.low[a]!=0xffffffffu){speed=max(speed,abs(ncOrderValue(c.low[a]))*ncDt()/ncH());}
  if(c.high[a]!=0u){speed=max(speed,abs(ncOrderValue(c.high[a]))*ncDt()/ncH());}
 }
 return speed>=ncMovingTravel();
}
// umResolutionError (l.1591), its 4h branch (lane < 8 a corner): the h branch is dead under shapeDisplacement.
var<private> ncCoarseCurve:f32;
fn ncResolutionError4(t:vec3i,lane:u32)->f32{
 var worst=0.0;let h=ncH();
 if(lane>=8u){return 0.0;}
 let c=4*t+vec3i(ncCorner(lane,2u))*4;let centre=ncLoadCorner(c);
 if(abs(centre)>=8.0*h){return 0.0;}
 var lap=0.0;
 for(var a=0u;a<3u;a++){
  var lo=c/4;lo[a]-=1;var hi=c/4;hi[a]+=1;
  if(!ncWidePresent(lo)||!ncWidePresent(hi)){continue;}
  let below=ncLoadCorner(4*lo);let above=ncLoadCorner(4*hi);
  if(abs(below-centre)>6.0*h||abs(above-centre)>6.0*h){continue;}
  worst=max(worst,abs(above-2.0*centre+below)/(16.0*h));
  lap+=above-2.0*centre+below;
 }
 if(abs(centre)<=2.0*h){ncCoarseCurve=max(ncCoarseCurve,abs(lap)/(12.0*h));}
 return worst;
}
// umDirectionTravel, umBoundary (l.1740, l.1746).
fn ncDirectionTravel(c:NcClass,k:u32)->f32{
 let a=k%3u;
 if(k<3u){return max(-ncOrderValue(c.low[a]),0.0)*ncDt()/ncH();}
 return max(ncOrderValue(c.high[a]),0.0)*ncDt()/ncH();
}
struct NcBoundary{score:f32,triggered:bool}
fn ncBoundary(p:vec3i,width:u32,c:NcClass)->NcBoundary{
 var result=NcBoundary(0.0,false);
 let threshold=NC_BOUNDARY*select(1.0,0.5,width==1u);
 if(threshold<=0.0){return result;}
 let least=NC_SCORE_FLOOR*threshold;
 let margin=4.0*select(1.0,2.0,width==1u);
 var travel:array<f32,6>;
 for(var k=0u;k<6u;k++){travel[k]=ncDirectionTravel(c,k);}
 for(var k=0u;k<6u;k++){
  let a=k%3u;let high=k>=3u;
  let along=max(max(travel[(a+1u)%3u],travel[3u+(a+1u)%3u]),max(travel[(a+2u)%3u],travel[3u+(a+2u)%3u]));
  if(travel[k]<max(least,along)){continue;}
  let reach=margin-f32(c.reach[k]);
  if(reach<0.0){continue;}
  let beyond=ncWallBeyond(p,k);
  var met=4.0*f32(beyond)<=reach;
  for(var n=1u;!met&&n<=min(beyond,NC_CAP)&&4.0*f32(n-1u)<=reach;n++){
   var q=p;q[a]+=select(-i32(n),i32(n),high);
   met=ncObstacleAt(q);
  }
  if(!met){continue;}
  result.score=max(result.score,travel[k]/threshold);
  if(travel[k]>=threshold){result.triggered=true;return result;}
 }
 let upWord=ncUp();
 if((upWord&3u)==0u){return result;}
 let upAxis=upWord>>2u;
 let up=select(travel[upAxis],travel[3u+upAxis],(upWord&3u)==1u);
 if(up<least){return result;}
 for(var k=0u;k<6u;k++){
  let a=k%3u;if(a==upAxis){continue;}
  let b=3u-a-upAxis;let along=max(travel[b],travel[3u+b]);
  if(ncWallBeyond(p,k)==0u&&up>=along){result.score=max(result.score,up/threshold);if(up>=threshold){result.triggered=true;}}
 }
 return result;
}
// umDepthSample (l.1792): lane i of 64 one vertex, every stored cell open.
fn ncDepthSample(t:vec3i,i:u32)->vec2u{
 let v=t+vec3i(ncCorner(i,4u))-vec3i(1);
 if(!ncWidePresent(v)){return vec2u(0u);}
 let value=ncLoadCorner(v*4);
 var low=vec3u(0u);var high=vec3u(0u);
 for(var k=0u;k<8u;k++){
  let bit=ncCorner(k,2u);let cell=4*v+vec3i(bit)-1;
  if(ncCellKind(cell)>=2u){continue;}
  high+=bit;low+=vec3u(1u)-bit;
 }
 let wall=any((low==vec3u(0u))&(high!=vec3u(0u)))||any((high==vec3u(0u))&(low!=vec3u(0u)));
 let thickness=value*select(2.0,1.0,wall);
 return vec2u(ncOrderKey(-thickness),ncOrderKey(thickness));
}
// umFaceLow (l.368).
fn ncFaceLow(anchor:vec3i,axis:u32)->bool{return (anchor[axis]&31)==31&&ncCellKind(anchor)>=2u;}
// umOwnerBounds (l.1822) of an h owner (lane a cell): single patches at its origin (+) and below it (-).
fn ncOwnerBounds1(t:vec3i,lane:u32,c:ptr<function,NcClass>){
 let origin=4*t+vec3i(ncCorner(lane,4u));
 for(var axis=0u;axis<3u;axis++){
  var low=3.0e38;var high=-3.0e38;
  for(var sign=-1;sign<=1;sign+=2){
   var anchor=origin;if(sign<0){anchor[axis]-=1;}
   if(ncFaceLow(anchor,axis)){continue;}
   let u=ncFace(anchor,axis);
   let finite=abs(u)<=3.0e38;
   low=min(low,select(-3.0e38,u,finite));high=max(high,select(3.0e38,u,finite));
  }
  (*c).low[axis]=min((*c).low[axis],ncOrderKey(low));(*c).high[axis]=max((*c).high[axis],ncOrderKey(high));
 }
}
// umClassifyOwner (l.1847) under surface authority, every cell open (it always classifies).
fn ncClassifyOwner(t:vec3i,width:u32,lane:u32,c:ptr<function,NcClass>,bounds:bool){
 let side=4u/width;let h=ncH();
 let origin=4*t+vec3i(ncCorner(lane,side)*width);
 var inside=0u;var lowPhi=3.0e38;var highPhi=-3.0e38;
 for(var k=0u;k<8u;k++){
  var value=0.0;
  if(width==4u){value=ncLoadCorner(origin+4*vec3i(ncCorner(k,2u)));}else{value=ncLoadFine(origin+i32(width)*vec3i(ncCorner(k,2u)));}
  lowPhi=min(lowPhi,value);highPhi=max(highPhi,value);
  if(value<0.0){inside++;}
  if(value<4.0*h*f32(width)){(*c).flags|=8u;}
 }
 if(bounds){ncOwnerBounds1(t,lane,c);}
 let local=ncCorner(lane,side)*width;
 let epsilon=1e-6*h;
 let crosses=lowPhi<=epsilon&&highPhi>=-epsilon&&lowPhi<highPhi;
 if(crosses){
  if(width==4u){(*c).crossing=vec2u(0xffffffffu);}
  else{(*c).crossing[lane/32u]|=1u<<(lane%32u);}
  for(var a=0u;a<3u;a++){(*c).gap[a]=min((*c).gap[a],local[a]);(*c).gap[3u+a]=min((*c).gap[3u+a],4u-width-local[a]);}
 }
 let interior=!crosses&&inside==8u;
 let air=!crosses&&inside==0u;
 if(!air){(*c).flags|=4u;}
 if(!interior&&!air){
  for(var a=0u;a<3u;a++){(*c).reach[a]=min((*c).reach[a],local[a]);(*c).reach[3u+a]=min((*c).reach[3u+a],4u-width-local[a]);}
  (*c).flags|=select(1u,3u,crosses);
 }
}
// umFinishTile (l.1897): the tile's words (w0, w1, gap) and its reach (f32 bits); its keys are c's.
struct NcFinish{w0:u32,w1:u32,gap:u32,reach:u32}
fn ncFinishTile(p:vec3i,width:u32,c:NcClass)->NcFinish{
 let crossing=(c.flags&2u)!=0u;let scale=ncDt()/ncH();
 var speed=0.0;
 for(var a=0u;a<3u;a++){speed=max(speed,max(abs(ncOrderValue(c.low[a])),abs(ncOrderValue(c.high[a])))*ncDt()/ncH());}
 var reach=0.0;
 for(var a=0u;a<3u;a++){
  if(c.low[a]!=0xffffffffu){reach=max(reach,abs(ncOrderValue(c.low[a])*scale));}
  if(c.high[a]!=0u){reach=max(reach,abs(ncOrderValue(c.high[a])*scale));}
 }
 let fast=NC_FAST>0.0&&speed>=NC_FAST*select(0.5,1.0,width==1u);
 var w0=0u;var w1=select(0u,NC_CROSSES,crossing)|select(0u,NC_WET,(c.flags&4u)!=0u);
 var gap=0xffffffffu;
 if(crossing){
  let tolerant=NC_TOL>0.0;let error=bitcast<f32>(c.error);
  let shaped=!fast&&ncCriterionOn(NC_SHAPE)&&!(tolerant&&error<=NC_TOL);
  if(!fast){w0|=select(255u,ncScoreByte(error/max(NC_TOL,1e-30)),tolerant);}
  if(shaped){w1|=ncTriggerBit(NC_SHAPE);}
  if(all(c.deep!=vec2u(0u))){
   let thickness=max(min(ncOrderValue(c.deep.x),ncOrderValue(c.deep.y)),0.0)/ncH();let thin=NC_THIN;
   w0|=ncScoreByte(thin/max(thickness,0.25*thin))<<8u;
   if(thickness<thin&&ncCriterionOn(NC_THIN_K)){w1|=ncTriggerBit(NC_THIN_K);}
  }
  let boundary=ncBoundary(p,width,c);
  w1|=ncScoreByte(boundary.score);
  if(boundary.triggered&&ncCriterionOn(NC_IMPACT)){w1|=ncTriggerBit(NC_IMPACT);}
  gap=min(u32(ceil(speed)),255u)<<24u;for(var k=0u;k<6u;k++){gap|=c.gap[k]<<(4u*k);}
 }
 return NcFinish(w0,w1,gap,bitcast<u32>(reach));
}
// umTileTravel, umTileToward, umTileSpeed (l.2250-2263) over a tile's keys.
fn ncKeyTravel(keys:array<u32,6>,k:u32)->f32{
 let a=k%3u;
 if(k<3u){return max(-ncOrderValue(keys[a]),0.0)*ncDt()/ncH();}
 return max(ncOrderValue(keys[3u+a]),0.0)*ncDt()/ncH();
}
fn ncKeyToward(keys:array<u32,6>,k:u32)->f32{
 let a=k%3u;
 if(k<3u){return ncOrderValue(keys[3u+a])*ncDt()/ncH();}
 return -ncOrderValue(keys[a])*ncDt()/ncH();
}
fn ncTileSpeed(keys:array<u32,6>)->f32{
 var speed=0.0;
 for(var a=0u;a<3u;a++){speed=max(speed,max(abs(ncOrderValue(keys[a])),abs(ncOrderValue(keys[3u+a])))*ncDt()/ncH());}
 return speed;
}
// umApproach (l.2300) of tile p with keys and gap word.
fn ncApproach(p:vec3i,keys:array<u32,6>,word:u32)->f32{
 let horizon=NC_APPROACH;if(horizon<=0.0){return 0.0;}
 var best=0.0;
 for(var k=0u;k<6u;k++){
  let a=k%3u;let high=k>=3u;let travel=ncKeyTravel(keys,k);
  if(travel<=0.0){continue;}
  var airSide=true;
  for(var c=0u;c<4u;c++){
   var v=4*p;v[a]+=select(0,4,high);v[(a+1u)%3u]+=4*i32(c&1u);v[(a+2u)%3u]+=4*i32(c>>1u);
   if(ncLoadCorner(v)<0.0){airSide=false;}
  }
  if(!airSide){continue;}
  let gap=f32((word>>(4u*k))&15u);
  let limit=min(NC_CAP,u32(ceil(min(horizon*travel,64.0)/4.0))+1u);
  var air=false;var gapCells=-1.0;var closing=travel;
  for(var n=1u;n<=limit+1u;n++){
   var q=p;q[a]+=select(-i32(n),i32(n),high);
   let state=ncTileState(q);
   if(state==2u){gapCells=4.0*f32(n-1u)+gap;break;}
   if(n>limit){break;}
   if(state==1u){air=true;continue;}
   if(ncObstacleAt(q)){gapCells=4.0*f32(n-1u)+gap;break;}
   let s=ncPageSlot(q>>vec3u(3u));let i=ncTileLocal(q);
   if((ncW1(s,i)&NC_WET)==0u){air=true;continue;}
   if(air){gapCells=4.0*f32(n-1u)+gap;closing+=ncKeyToward(ncKeys(s,i),k);}
   break;
  }
  if(gapCells>=1.0){best=max(best,horizon*closing/gapCells);}
 }
 return best;
}
// importanceTile (l.2342): the final words of a tile from classify's, no hold, budget or source.
fn ncScoreByteOf(w0:u32,w1:u32,k:u32)->u32{if(k<4u){return (w0>>(8u*k))&255u;}return (w1>>(8u*(k-4u)))&255u;}
fn ncImportance(p:vec3i,keys:array<u32,6>,gapWord:u32,own:vec2u)->vec2u{
 var w0=own.x;var w1=own.y;
 let crossing=(w1&NC_CROSSES)!=0u;let wet=(w1&NC_WET)!=0u;
 if(crossing||wet){
  if(crossing){
   let score=ncApproach(p,keys,gapWord);w1|=ncScoreByte(score)<<8u;
   if(score>=1.0&&ncCriterionOn(NC_APPROACH_K)){w1|=ncTriggerBit(NC_APPROACH_K);}
  }
  var best=0u;var winner=0u;
  for(var k=0u;k<6u;k++){let byte=ncScoreByteOf(w0,w1,k);if(byte>best){best=byte;winner=k+1u;}}
  w1|=winner<<22u;
  let required=((w1>>16u)&63u)!=0u;
  w1|=select(0u,NC_REQUIRED,required);
 }
 return vec2u(w0,w1);
}
// rangeKeys (l.2659): the window clamp and cube-level choice verbatim; pages by the directory, every resident
// page (C_ALL) past 64.
fn ncRangeKeys(low:vec3i,high:vec3i)->array<u32,6>{
 var keys=ncIdentityKeys();
 let a=max(low,vec3i(0));let b=min(high,vec3i(2047));
 if(any(b<a)){return keys;}
 var j=0u;
 loop{
  let n=(b>>vec3u(j))-(a>>vec3u(j))+vec3i(1);
  if(j>=3u||n.x*n.y*n.z<=64){break;}
  j++;
 }
 let pa=a>>vec3u(3u);let pb=b>>vec3u(3u);let n=pb-pa+vec3i(1);
 if(n.x*n.y*n.z>64){
  for(var e=0u;e<ncListCount();e++){
   let s=ncListSlot(e);let q=ncSlotPage(s);if(any(q<pa)||any(q>pb)){continue;}
   let top=ncCube(s,3u,vec3u(0u));for(var k=0u;k<3u;k++){keys[k]=min(keys[k],top[k]);keys[3u+k]=max(keys[3u+k],top[3u+k]);}
  }
  return keys;
 }
 for(var z=pa.z;z<=pb.z;z++){for(var y=pa.y;y<=pb.y;y++){for(var x=pa.x;x<=pb.x;x++){
  let q=vec3i(x,y,z);let s=ncPageSlot(q);if(s==0xffffffffu){continue;}
  let la=vec3u(max(a-8*q,vec3i(0)))>>vec3u(j);let lb=vec3u(min(b-8*q,vec3i(7)))>>vec3u(j);
  for(var cz=la.z;cz<=lb.z;cz++){for(var cy=la.y;cy<=lb.y;cy++){for(var cx=la.x;cx<=lb.x;cx++){
   var at:array<u32,6>;
   if(j==0u){at=ncKeys(s,cx|(cy<<3u)|(cz<<6u));}else{at=ncCube(s,j,vec3u(cx,cy,cz));}
   for(var k=0u;k<3u;k++){keys[k]=min(keys[k],at[k]);keys[3u+k]=max(keys[3u+k],at[3u+k]);}
  }}}
 }}}
 return keys;
}
// sampledFlow (l.2700).
struct NcFlow{low:vec3f,high:vec3f}
fn ncSampledFlow(lo:vec3f,hi:vec3f,shift:vec3f)->NcFlow{
 let keys=ncRangeKeys(vec3i(floor((lo-NC_SAMPLE_REACH)/4.0)),vec3i(floor((hi+NC_SAMPLE_REACH)/4.0)));
 var f=NcFlow(vec3f(0),vec3f(0));
 for(var a=0u;a<3u;a++){
  if(keys[a]!=0xffffffffu){f.low[a]=ncOrderValue(keys[a]);}
  if(keys[3u+a]!=0u){f.high[a]=ncOrderValue(keys[3u+a]);}
 }
 f.low=min(vec3f(0),f.low+min(shift,vec3f(0)));f.high=max(vec3f(0),f.high+max(shift,vec3f(0)));
 return f;
}
// interfaceIn (l.2561) on a page's required bits (interfaceBits' words; the slab word is their union).
fn ncInterfaceIn(s:u32,low:vec3i,high:vec3i)->bool{
 let a=vec3u(max(low,vec3i(0)));let b=vec3u(min(high,vec3i(7)));
 let row=((2u<<(b.x-a.x))-1u)<<a.x;var mask=vec2u(0u);
 for(var y=a.y;y<=b.y;y++){mask[y>>2u]|=row<<(8u*(y&3u));}
 for(var z=a.z;z<=b.z;z++){
  if(((ncReq(s,2u*z)&mask.x)|(ncReq(s,2u*z+1u)&mask.y))!=0u){return true;}
 }
 return false;
}
// interfaceAny (l.2572): its 27-page shortcut gives the same answer as the page walk and is not kept.
fn ncInterfaceAny(low:vec3i,high:vec3i)->bool{
 let a=max(low,vec3i(0));let b=min(high,vec3i(2047));
 if(any(b<a)){return false;}
 let pa=a>>vec3u(3u);let pb=b>>vec3u(3u);
 let n=pb-pa+vec3i(1);
 if(n.x*n.y*n.z>64){
  for(var e=0u;e<ncListCount();e++){
   let s=ncListSlot(e);let q=ncSlotPage(s);if(any(q<pa)||any(q>pb)){continue;}
   if(ncInterfaceIn(s,a-8*q,b-8*q)){return true;}
  }
  return false;
 }
 for(var z=pa.z;z<=pb.z;z++){for(var y=pa.y;y<=pb.y;y++){for(var x=pa.x;x<=pb.x;x++){
  let q=vec3i(x,y,z);let s=ncPageSlot(q);if(s==0xffffffffu){continue;}
  if(ncInterfaceIn(s,a-8*q,b-8*q)){return true;}
 }}}
 return false;
}
// crossingCellsMeet (l.2742).
fn ncCrossingCellsMeet(q:vec3i,lo:vec3f,hi:vec3f)->bool{
 let origin=4.0*vec3f(q);
 let a=max(vec3i(floor(lo-origin-vec3f(1e-3))),vec3i(0));
 let b=min(vec3i(floor(hi-origin+vec3f(1e-3))),vec3i(3));
 if(any(b<a)){return false;}
 let s=ncPageSlot(q>>vec3u(3u));if(s==0xffffffffu){return false;}
 let mask=ncCross(s,ncTileLocal(q));
 if(all(mask==vec2u(0u))){return false;}
 let row=((1u<<u32(b.x-a.x+1))-1u)<<u32(a.x);
 for(var z=a.z;z<=b.z;z++){for(var y=a.y;y<=b.y;y++){
  let lane=4u*u32(y)+16u*u32(z);
  if((mask[lane/32u]&(row<<(lane%32u)))!=0u){return true;}
 }}
 return false;
}
// departureMeetsSurface (l.2760). The shell walks the window's tiles (umWindowTile); those outside the lattice
// hold no page, meet nothing, and are skipped (ncTileSpan).
fn ncDepartureMeetsSurface(lo:vec3f,hi:vec3f)->bool{
 let first=vec3i(floor((lo-1e-3)/4.0));let last=vec3i(floor((hi+1e-3)/4.0));
 if(!ncInterfaceAny(first,last)){return false;}
 let inner=vec3i(ceil((lo-1e-3)/4.0));let innerLast=last-vec3i(1);
 if(ncInterfaceAny(inner,innerLast)){return true;}
 let a=max(clamp(first,vec3i(0),vec3i(2047)),vec3i(0));let b=min(clamp(last,vec3i(0),vec3i(2047)),ncTileSpan()-vec3i(1));
 for(var z=a.z;z<=b.z;z++){for(var y=a.y;y<=b.y;y++){
  if(!ncInterfaceAny(vec3i(a.x,y,z),vec3i(b.x,y,z))){continue;}
  let core=y>=inner.y&&y<=innerLast.y&&z>=inner.z&&z<=innerLast.z;
  for(var x=a.x;x<=b.x;x++){
   if(core&&x>=inner.x&&x<=innerLast.x){x=innerLast.x;continue;}
   if(ncCrossingCellsMeet(vec3i(x,y,z),lo,hi)){return true;}
  }
 }}
 return false;
}
// umKeySpeed (l.2808).
fn ncKeySpeed(keys:array<u32,6>)->f32{
 var speed=0.0;
 for(var a=0u;a<3u;a++){
  if(keys[a]!=0xffffffffu){speed=max(speed,abs(ncOrderValue(keys[a]))*ncDt()/ncH());}
  if(keys[3u+a]!=0u){speed=max(speed,abs(ncOrderValue(keys[3u+a]))*ncDt()/ncH());}
 }
 return speed;
}
// movingShapeNear (l.2823); its box walks the lattice's tiles of the window's (the rest hold no page).
fn ncMovingShapeNear(p:vec3i,keys:array<u32,6>)->bool{
 let reach=clamp(i32(ceil(ncKeySpeed(keys)/4.0)),1,3);
 if(!ncInterfaceAny(p-vec3i(reach),p+vec3i(reach))){return false;}
 let want=NC_CROSSES|NC_REQUIRED|ncTriggerBit(NC_SHAPE);
 let lo=max(p-vec3i(reach),vec3i(0));let hi=min(p+vec3i(reach),ncTileSpan()-vec3i(1));
 for(var z=lo.z;z<=hi.z;z++){for(var y=lo.y;y<=hi.y;y++){for(var x=lo.x;x<=hi.x;x++){
  let q=vec3i(x,y,z);let s=ncPageSlot(q>>vec3u(3u));if(s==0xffffffffu||all(q==p)){continue;}
  let i=ncTileLocal(q);
  if((ncFin(s,i)&want)==want&&ncKeySpeed(ncKeys(s,i))>=ncMovingTravel()){return true;}
 }}}
 return false;
}
// surfaceAllowed (l.2795), surfaceDistance 0.
fn ncSurfaceAllowed(p:vec3i,fin:u32,keys:array<u32,6>)->bool{
 if((fin&NC_CROSSES)!=0u){return true;}
 return ncMovingShapeNear(p,keys);
}
// departureTravel, departureReachTiles (l.2603): one step (no gravity term), reach 0 and hysteresis 0.
fn ncDepartureReachTiles(reach:f32)->i32{
 let travel=min(reach+0.0,1.0e6);
 let margin=NC_DRIFT+4.0*f32(0);
 return i32(ceil((travel+margin+1e-3)/4.0))+2;
}
// decideTile (l.2839) with D = 0 (no deep COLD tile), no regions, sources or forward reach: the band bit.
fn ncDecide(p:vec3i,joined:bool,fin:u32,keys:array<u32,6>,reach:f32)->bool{
 let scale=ncDt()/ncH();
 if(!joined&&!ncSurfaceAllowed(p,fin,keys)){return false;}
 let far=ncDepartureReachTiles(reach);
 if(!joined&&!ncInterfaceAny(p-vec3i(far),p+vec3i(far))){return false;}
 let crossing=(fin&NC_REQUIRED)!=0u;
 var fine=joined||crossing;
 if(!fine){
  let margin=NC_DRIFT+4.0*f32(0);
  let s=scale;
  var lo=4.0*vec3f(p);var hi=lo+vec3f(4.0);
  let shift=vec3f(0.0);
  let start=ncSampledFlow(lo,hi,shift);
  let mid=ncSampledFlow(lo-0.5*start.high*s,hi-0.5*start.low*s,shift);
  lo=lo-mid.high*s;hi=hi-mid.low*s;
  fine=ncDepartureMeetsSurface(lo-vec3f(margin),hi+vec3f(margin));
 }
 return fine;
}
`;

const setRW: BrickBinding = { name: "set", kind: "rw", type: "array<atomic<u32>>" };
const setRO: BrickBinding = { name: "set", kind: "ro", type: "array<u32>" };
const dirRO: BrickBinding = { name: "dir", kind: "ro", type: "array<u32>" };
const velRO: BrickBinding = { name: "vel", kind: "ro", type: "array<f32>" };
const phiRO: BrickBinding = { name: "phi", kind: "ro", type: "array<u32>" };
const censusRW: BrickBinding = { name: "census", kind: "rw", type: "array<u32>" };
const censusRO: BrickBinding = { name: "census", kind: "ro", type: "array<u32>" };

const kernel = (L: BrickLayout, label: string, bindings: readonly BrickBinding[], body: string): BrickKernel => {
  const set = bindings.find(b => b.name === "set"), dir = bindings.find(b => b.name === "dir");
  const atomic = (b: BrickBinding) => (b.type ?? "").includes("atomic");
  const code = [
    brickAbiWGSL(L, { ...(set ? { set: { name: "nbSet", atomic: atomic(set) } } : {}), ...(dir ? { dir: { name: "dir", atomic: atomic(dir) } } : {}) }),
    headerWGSL(),
    declareBindings(bindings),
    body,
  ].join("\n");
  return { label, entry: label, code, bindings };
};

/** The hooks on the brick storage (CENSUS_FORMULAS' contract), for a module that binds dir, phi, vel and census. */
const HOOKS = /* wgsl */`
const NC_S:u32=${CENSUS.STRIDE}u;
var<private> cH:f32;var<private> cDt:f32;var<private> cUp:u32;
// The ring's h, dt and up (uniformMixedUp: the axis gravity pulls along most, the first of equals, and -g's sign).
fn ncInit(){
  cH=nbRingF(cfg.x,NB_RING_H);cDt=nbRingF(cfg.x,NB_RING_DT);
  let g=vec3f(nbRingF(cfg.x,NB_RING_GRAVITY),nbRingF(cfg.x,NB_RING_GRAVITY+1u),nbRingF(cfg.x,NB_RING_GRAVITY+2u));
  let a=abs(g.x);let b=abs(g.y);let c=abs(g.z);
  let upAxis=select(select(2u,0u,a>=c),1u,b>=a&&b>=c);let up=sign(-g[upAxis]);
  cUp=select(select(2u,1u,up>0.0)|(upAxis<<2u),0u,up==0.0);
}
fn ncH()->f32{return cH;}
fn ncDt()->f32{return cDt;}
fn ncUp()->u32{return cUp;}
fn ncTileSpan()->vec3i{return NB_LATTICE/4;}
fn ncPageIn(q:vec3i)->bool{return all(q>=vec3i(0))&&all(32*q<NB_LATTICE);}
// A lattice page's C slot, NB_NONE where it is not resident (AIR) or outside the lattice (SOLID).
fn ncPageSlot(q:vec3i)->u32{if(!ncPageIn(q)){return NB_NONE;}return nbCSlotOfPage(q);}
fn ncListCount()->u32{return nbListCount(NB_LIST_C_ALL);}
fn ncListSlot(e:u32)->u32{return nbListSlot(NB_LIST_C_ALL,e);}
fn ncSlotPage(s:u32)->vec3i{return nbKeyCoord(nbCRec(s,NB_REC_KEY));}
fn ncKeys(s:u32,i:u32)->array<u32,6>{let o=NC_S*s+${CENSUS.KEYS}u+6u*i;return array<u32,6>(census[o],census[o+1u],census[o+2u],census[o+3u],census[o+4u],census[o+5u]);}
fn ncCube(s:u32,j:u32,c:vec3u)->array<u32,6>{
  let k=3u-j;let o=NC_S*s+${CENSUS.CUBES}u+6u*(select(select(72u,64u,j==2u),0u,j==1u)+(c.x|(c.y<<k)|(c.z<<(2u*k))));
  return array<u32,6>(census[o],census[o+1u],census[o+2u],census[o+3u],census[o+4u],census[o+5u]);
}
fn ncW0(s:u32,i:u32)->u32{return census[NC_S*s+${CENSUS.W0}u+i];}
fn ncW1(s:u32,i:u32)->u32{return census[NC_S*s+${CENSUS.W1}u+i];}
fn ncFin(s:u32,i:u32)->u32{return census[NC_S*s+${CENSUS.FIN}u+i];}
fn ncCross(s:u32,i:u32)->vec2u{let o=NC_S*s+${CENSUS.CROSS}u+2u*i;return vec2u(census[o],census[o+1u]);}
fn ncGap(s:u32,i:u32)->u32{return census[NC_S*s+${CENSUS.GAP}u+i];}
fn ncReq(s:u32,w:u32)->u32{return census[NC_S*s+${CENSUS.REQ}u+w];}
fn ncCellKind(c:vec3i)->u32{if(!nbInLattice(c)){return 2u;}return select(0u,1u,ncPageSlot(c>>vec3u(5u))==NB_NONE);}
fn ncTileState(q:vec3i)->u32{if(!nbTileInLattice(q)){return 2u;}return select(0u,1u,ncPageSlot(q>>vec3u(3u))==NB_NONE);}
fn ncCellOwnerH(c:vec3i)->bool{return nbCellOwnerH(c);}
fn ncObstacleAt(q:vec3i)->bool{return false;}
// umWallBeyond (l.1732): the page across direction k outside the lattice is the wall.
fn ncWallBeyond(p:vec3i,k:u32)->u32{
  let a=k%3u;let high=k>=3u;var page=p>>vec3u(3u);page[a]+=select(-1,1,high);
  if(ncPageIn(page)){return NC_NO_WALL;}
  let l=vec3u(p&vec3i(7));return select(l[a],7u-l[a],high);
}
// umWideVertexSlot (l.310): a lattice page whose closure holds wide vertex v is resident.
fn ncWidePresent(v:vec3i)->bool{
  let page=(v-vec3i(1))>>vec3u(3u);let plane=(v&vec3i(7))==vec3i(0);
  for(var k=0u;k<8u;k++){
    let step=vec3<bool>((k&1u)!=0u,(k&2u)!=0u,(k&4u)!=0u);
    if(any(step&!plane)){continue;}
    if(ncPageSlot(page+select(vec3i(0),vec3i(1),step))!=NB_NONE){return true;}
  }
  return false;
}
fn ncLoadFine(p:vec3i)->f32{return nbVertexPhi(p)*cH;}
fn ncLoadCorner(p:vec3i)->f32{return nbVertexPhi(p)*cH;}
// The old velocity texel of cell q on axis a: the face between q and q + e_a, stored as the lower face of q + e_a.
fn ncFace(q:vec3i,a:u32)->f32{var e=vec3i(0);e[a]=1;return nbFacePlane(q+e,a);}
fn xTileOf(cs:u32,i:u32)->vec3i{return 8*nbKeyCoord(nbCRec(cs,NB_REC_KEY))+vec3i(i32(i&7u),i32((i>>3u)&7u),i32(i>>6u));}
`;

const OWNER = brickOwnerWGSL({ phi: "phi", vel: "vel", faceF: "NB_VEL_F_US0", faceC: "NB_VEL_C_US0" });

/** The tile's words, stored by the one lane that finishes it. */
const STORE = /* wgsl */`
fn ncStore(cs:u32,i:u32,keys:array<u32,6>,w0:u32,w1:u32,cross:vec2u,gap:u32){
  let o=NC_S*cs;
  for(var k=0u;k<6u;k++){census[o+${CENSUS.KEYS}u+6u*i+k]=keys[k];}
  census[o+${CENSUS.W0}u+i]=w0;census[o+${CENSUS.W1}u+i]=w1;census[o+${CENSUS.FIN}u+i]=w1;
  census[o+${CENSUS.CROSS}u+2u*i]=cross.x;census[o+${CENSUS.CROSS}u+2u*i+1u]=cross.y;
  census[o+${CENSUS.GAP}u+i]=gap;
}
fn ncStoreIdentity(cs:u32,i:u32){ncStore(cs,i,ncIdentityKeys(),0u,0u,vec2u(0u),0xffffffffu);}
fn ncStoreClass(cs:u32,i:u32,c:NcClass,r:NcFinish){
  ncStore(cs,i,array<u32,6>(c.low[0],c.low[1],c.low[2],c.high[0],c.high[1],c.high[2]),r.w0,r.w1,c.crossing,r.gap);
}
var<workgroup> wgReach:atomic<u32>;
var<workgroup> wgCrossing:atomic<u32>;
fn ncFlush(){
  let reach=atomicLoad(&wgReach);if(reach!=0u){atomicMax(&nbSet[NB_H_CENSUS_REACH],reach);}
  let n=atomicLoad(&wgCrossing);if(n!=0u){atomicAdd(&nbSet[NB_H_CENSUS_CROSSING],n);}
}
`;

export function brickCensusKernels(L: BrickLayout) {
  if (L.lattice.some(n => n % 32 !== 0)) throw new Error(`setCensus: the lattice ${L.lattice.join("x")} is not whole pages (32 cells an axis)`);
  if (L.cap.b > 0) throw new Error("setCensus: BULK16 pages are not supported at M2");
  const k = (label: string, bindings: readonly BrickBinding[], body: string) => kernel(L, label, bindings, body);

  // censusBegin: one lane. The header words clear; a body or solid edit, or a B or K brick, fails.
  const censusBegin = k("censusBegin", [CFG, setRW], /* wgsl */`
@compute @workgroup_size(1) fn censusBegin(){
  for(var w=NB_H_CENSUS_CLAMPED;w<=NB_H_CENSUS_TRIGGERED;w++){atomicStore(&nbSet[w],0u);}
  let flags=nbRing(cfg.x,NB_RING_FLAGS)&(NB_RING_FLAG_SOLID_EDIT|NB_RING_FLAG_BODIES);
  if(flags!=0u||atomicLoad(&nbSet[NB_H_LIVE_B])!=0u||atomicLoad(&nbSet[NB_H_LIVE_K])!=0u){nbFail(NB_FAIL_UNSUPPORTED,0x63000000u|flags);}
}`);

  // censusFine: classify (l.2091), a workgroup an F brick of F_ALL, each of its h tiles in turn, a lane a cell.
  const censusFine = k("censusFine", [CFG, setRW, dirRO, velRO, phiRO, censusRW], OWNER + HOOKS + CENSUS_FORMULAS + STORE + /* wgsl */`
var<workgroup> wgFlag:u32;
var<workgroup> tileCrossing:array<atomic<u32>,2>;
var<workgroup> tileLow:array<atomic<u32>,3>;
var<workgroup> tileHigh:array<atomic<u32>,3>;
var<workgroup> tileGap:array<atomic<u32>,6>;
var<workgroup> tileReach:array<atomic<u32>,6>;
var<workgroup> tileError:atomic<u32>;
var<workgroup> tileDeep:array<atomic<u32>,2>;
var<workgroup> mixedTile:atomic<u32>;
@compute @workgroup_size(64) fn censusFine(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lane:u32){
  ncInit();
  let g=nbGroup(wid);
  if(lane==0u){wgFlag=select(0u,1u,g<nbListCount(NB_LIST_F_ALL));atomicStore(&wgReach,0u);atomicStore(&wgCrossing,0u);}
  if(workgroupUniformLoad(&wgFlag)==0u){return;}
  let fs=nbListSlot(NB_LIST_F_ALL,g);let brick=nbKeyCoord(nbFRec(fs,NB_REC_KEY));
  for(var j=0u;j<8u;j++){
    let t=2*brick+vec3i(ncCorner(j,2u));
    if(lane==0u){wgFlag=select(0u,1u,nbTileOwnerH(t));}
    if(workgroupUniformLoad(&wgFlag)==0u){continue;}
    if(lane==0u){
      atomicStore(&mixedTile,0u);for(var q=0u;q<2u;q++){atomicStore(&tileCrossing[q],0u);atomicStore(&tileDeep[q],0u);}
      for(var a=0u;a<3u;a++){atomicStore(&tileLow[a],0xffffffffu);atomicStore(&tileHigh[a],0u);atomicStore(&tileGap[a],15u);atomicStore(&tileGap[3u+a],15u);atomicStore(&tileReach[a],15u);atomicStore(&tileReach[3u+a],15u);}
      atomicStore(&tileError,0u);atomicStore(&ncTileSigned,0);atomicStore(&ncTileCrossings,0u);atomicStore(&ncTileCurve,0);atomicStore(&ncTileCurves,0u);atomicStore(&ncTileCorner,0u);
    }
    workgroupBarrier();
    {
      var c=ncEmptyClass();ncClassifyOwner(t,1u,lane,&c,true);
      for(var a=0u;a<3u;a++){atomicMin(&tileLow[a],c.low[a]);atomicMax(&tileHigh[a],c.high[a]);}
      for(var q=0u;q<6u;q++){atomicMin(&tileGap[q],c.gap[q]);atomicMin(&tileReach[q],c.reach[q]);}
      atomicOr(&mixedTile,c.flags);
      for(var q=0u;q<2u;q++){if(c.crossing[q]!=0u){atomicOr(&tileCrossing[q],c.crossing[q]);}}
    }
    workgroupBarrier();
    if(lane==0u){wgFlag=atomicLoad(&mixedTile);}
    let shaping=(workgroupUniformLoad(&wgFlag)&2u)!=0u&&NC_TOL>0.0;
    if(shaping){atomicMax(&tileError,bitcast<u32>(ncSurfaceDisplacement(t,lane)));}
    if((atomicLoad(&mixedTile)&2u)!=0u){let deep=ncDepthSample(t,lane);for(var q=0u;q<2u;q++){atomicMax(&tileDeep[q],deep[q]);}}
    workgroupBarrier();
    if(lane==0u){
      var c=ncEmptyClass();
      for(var a=0u;a<3u;a++){c.low[a]=atomicLoad(&tileLow[a]);c.high[a]=atomicLoad(&tileHigh[a]);}
      for(var q=0u;q<6u;q++){c.gap[q]=atomicLoad(&tileGap[q]);c.reach[q]=atomicLoad(&tileReach[q]);}
      c.flags=atomicLoad(&mixedTile);c.error=atomicLoad(&tileError);
      c.error=bitcast<u32>(max(bitcast<f32>(c.error),max(ncVolumeError(),select(0.0,max(ncCurveError(),ncCornerCurveError()),ncClassMoving(c)))));
      c.crossing=vec2u(atomicLoad(&tileCrossing[0]),atomicLoad(&tileCrossing[1]));
      c.deep=vec2u(atomicLoad(&tileDeep[0]),atomicLoad(&tileDeep[1]));
      let r=ncFinishTile(t,1u,c);
      ncStoreClass(nbCSlotOfPage(t>>vec3u(3u)),nbTileIndex(t),c,r);
      atomicMax(&wgReach,r.reach);if((c.flags&2u)!=0u){atomicAdd(&wgCrossing,1u);}
    }
  }
  if(lane==0u){ncFlush();}
}`);

  // censusCoarse: classifyCoarse (l.2175), a workgroup a page of C_ALL, its tiles 64 at a time, a lane a tile.
  // Out-of-lattice tiles store identity keys and zero words; an h tile without its F brick fails (TIER).
  const censusCoarse = k("censusCoarse", [CFG, setRW, dirRO, velRO, phiRO, censusRW], OWNER + HOOKS + CENSUS_FORMULAS + STORE + /* wgsl */`
var<workgroup> wgFlag:u32;
var<workgroup> cqErrors:atomic<u32>;
var<workgroup> cqError:array<u32,64>;
var<workgroup> cqDeeps:atomic<u32>;
var<workgroup> cqDeep:array<u32,64>;
var<workgroup> cqWorst:array<atomic<u32>,64>;
var<workgroup> cqCurve:array<atomic<u32>,64>;
var<workgroup> cqDeepest:array<array<atomic<u32>,2>,64>;
var<workgroup> cqBounded:array<u32,64>;
var<workgroup> cqLow:array<array<atomic<u32>,3>,64>;
var<workgroup> cqHigh:array<array<atomic<u32>,3>,64>;
var<workgroup> cqParts:atomic<u32>;
var<workgroup> cqPart:array<vec4i,384>;
fn cqBound(owner:u32,axis:u32,anchor:vec3i){
  if(ncFaceLow(anchor,axis)){return;}
  let u=ncFace(anchor,axis);
  let finite=abs(u)<=3.0e38;
  atomicMin(&cqLow[owner][axis],ncOrderKey(select(-3.0e38,u,finite)));atomicMax(&cqHigh[owner][axis],ncOrderKey(select(3.0e38,u,finite)));
}
@compute @workgroup_size(64) fn censusCoarse(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lane:u32){
  ncInit();
  let g=nbGroup(wid);
  if(lane==0u){wgFlag=select(0u,1u,g<nbListCount(NB_LIST_C_ALL));atomicStore(&wgReach,0u);atomicStore(&wgCrossing,0u);}
  if(workgroupUniformLoad(&wgFlag)==0u){return;}
  let cs=nbListSlot(NB_LIST_C_ALL,g);
  for(var r=0u;r<8u;r++){
    if(lane==0u){atomicStore(&cqErrors,0u);atomicStore(&cqDeeps,0u);atomicStore(&cqParts,0u);}
    for(var a=0u;a<3u;a++){atomicStore(&cqLow[lane][a],ncOrderKey(3.0e38));atomicStore(&cqHigh[lane][a],ncOrderKey(-3.0e38));}
    cqBounded[lane]=0u;
    atomicStore(&cqWorst[lane],0u);atomicStore(&cqCurve[lane],0u);atomicStore(&cqDeepest[lane][0],0u);atomicStore(&cqDeepest[lane][1],0u);
    workgroupBarrier();
    let i=lane+64u*r;let t=xTileOf(cs,i);
    var c=ncEmptyClass();var classified=false;
    if(!nbTileInLattice(t)){ncStoreIdentity(cs,i);}
    else if(nbTileH(cs,i)){
      let b=t>>vec3u(1u);
      if(nbFSlotOfBrick(b)>=NB_SEAM){nbFail(NB_FAIL_TIER,nbKey(b));ncStoreIdentity(cs,i);}
    }else{
      classified=true;cqBounded[lane]=1u;
      ncClassifyOwner(t,4u,0u,&c,false);
      if((c.flags&2u)!=0u&&NC_TOL>0.0){cqError[atomicAdd(&cqErrors,1u)]=lane;}
      if((c.flags&2u)!=0u){cqDeep[atomicAdd(&cqDeeps,1u)]=lane;}
    }
    workgroupBarrier();
    // Face jobs (umFaceFirst, l.691): one patch, or sixteen h patches beside an h tile.
    for(var j=lane;j<6u*64u;j+=64u){
      let owner=j/6u;if(cqBounded[owner]==0u){continue;}
      let axis=(j%6u)/2u;let sign=select(-1,1,(j&1u)!=0u);
      let origin=4*xTileOf(cs,owner+64u*r);var probe=origin;probe[axis]+=select(-1,4,sign>0);
      var nw=0u;if(ncCellKind(probe)==0u){nw=select(4u,1u,ncCellOwnerH(probe));}
      let width=min(4u,select(4u,nw,nw!=0u));let side=4u/width;var anchor=probe;anchor[axis]-=select(0,1,sign>0);
      if(side*side==1u){cqBound(owner,axis,anchor);}
      else{cqPart[atomicAdd(&cqParts,1u)]=vec4i(anchor,i32(owner|(axis<<8u)|((side*side)<<16u)));}
    }
    let errors=atomicLoad(&cqErrors);
    for(var j=lane;j<8u*errors;j+=64u){
      let owner=cqError[j/8u];ncCoarseCurve=0.0;
      let error=ncResolutionError4(xTileOf(cs,owner+64u*r),j%8u);
      atomicMax(&cqWorst[owner],bitcast<u32>(error));atomicMax(&cqCurve[owner],bitcast<u32>(ncCoarseCurve));
    }
    let deeps=atomicLoad(&cqDeeps);
    for(var j=lane;j<64u*deeps;j+=64u){
      let owner=cqDeep[j/64u];let d=ncDepthSample(xTileOf(cs,owner+64u*r),j%64u);
      atomicMax(&cqDeepest[owner][0],d.x);atomicMax(&cqDeepest[owner][1],d.y);
    }
    workgroupBarrier();
    let parts=atomicLoad(&cqParts);
    for(var j=lane;j<16u*parts;j+=64u){
      let item=cqPart[j/16u];let word=u32(item.w);let part=j%16u;
      if(part>=(word>>16u)){continue;}
      let axis=(word>>8u)&3u;var anchor=item.xyz;
      anchor[(axis+1u)%3u]+=i32(part%4u);anchor[(axis+2u)%3u]+=i32(part/4u);
      cqBound(word&255u,axis,anchor);
    }
    workgroupBarrier();
    if(classified){
      for(var a=0u;a<3u;a++){c.low[a]=min(c.low[a],atomicLoad(&cqLow[lane][a]));c.high[a]=max(c.high[a],atomicLoad(&cqHigh[lane][a]));}
      if((c.flags&2u)!=0u&&NC_TOL>0.0){
        var error=bitcast<f32>(atomicLoad(&cqWorst[lane]));ncCoarseCurve=bitcast<f32>(atomicLoad(&cqCurve[lane]));
        if(ncClassMoving(c)){error=max(error,0.5*NC_VOLUME_SHAPE*ncCoarseCurve);}
        c.error=bitcast<u32>(error);
      }
      if((c.flags&2u)!=0u){c.deep=vec2u(atomicLoad(&cqDeepest[lane][0]),atomicLoad(&cqDeepest[lane][1]));}
      let f=ncFinishTile(t,4u,c);
      ncStoreClass(cs,i,c,f);
      atomicMax(&wgReach,f.reach);if((c.flags&2u)!=0u){atomicAdd(&wgCrossing,1u);}
    }
    workgroupBarrier();
  }
  if(lane==0u){ncFlush();}
}`);

  // censusCubes: boundCubes (l.2625), a workgroup a page of C_ALL.
  const censusCubes = k("censusCubes", [CFG, setRO, dirRO, censusRW], /* wgsl */`
const NC_S:u32=${CENSUS.STRIDE}u;
var<workgroup> boundPairs:array<u32,384>;
var<workgroup> boundFours:array<u32,48>;
fn ncIdentityKeys()->array<u32,6>{return array<u32,6>(0xffffffffu,0xffffffffu,0xffffffffu,0u,0u,0u);}
var<workgroup> wgGo:u32;
@compute @workgroup_size(64) fn censusCubes(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lane:u32){
  let g=nbGroup(wid);if(lane==0u){wgGo=select(0u,1u,g<nbListCount(NB_LIST_C_ALL));}if(workgroupUniformLoad(&wgGo)==0u){return;}
  let cs=nbListSlot(NB_LIST_C_ALL,g);let o=NC_S*cs;
  var keys=ncIdentityKeys();
  let c=vec3u(lane&3u,(lane>>2u)&3u,lane>>4u);
  for(var i=0u;i<8u;i++){
    let q=2u*c+vec3u(i&1u,(i>>1u)&1u,i>>2u);let t=q.x|(q.y<<3u)|(q.z<<6u);
    for(var k=0u;k<3u;k++){keys[k]=min(keys[k],census[o+${CENSUS.KEYS}u+6u*t+k]);keys[3u+k]=max(keys[3u+k],census[o+${CENSUS.KEYS}u+6u*t+3u+k]);}
  }
  for(var k=0u;k<6u;k++){census[o+${CENSUS.CUBES}u+6u*lane+k]=keys[k];boundPairs[6u*lane+k]=keys[k];}
  workgroupBarrier();
  if(lane<8u){
    keys=ncIdentityKeys();
    let c=vec3u(lane&1u,(lane>>1u)&1u,lane>>2u);
    for(var i=0u;i<8u;i++){
      let q=2u*c+vec3u(i&1u,(i>>1u)&1u,i>>2u);let l=6u*(q.x|(q.y<<2u)|(q.z<<4u));
      for(var k=0u;k<3u;k++){keys[k]=min(keys[k],boundPairs[l+k]);keys[3u+k]=max(keys[3u+k],boundPairs[l+3u+k]);}
    }
    for(var k=0u;k<6u;k++){census[o+${CENSUS.CUBES}u+6u*(64u+lane)+k]=keys[k];boundFours[6u*lane+k]=keys[k];}
  }
  workgroupBarrier();
  if(lane==0u){
    keys=ncIdentityKeys();
    for(var i=0u;i<8u;i++){let l=6u*i;for(var k=0u;k<3u;k++){keys[k]=min(keys[k],boundFours[l+k]);keys[3u+k]=max(keys[3u+k],boundFours[l+3u+k]);}}
    for(var k=0u;k<6u;k++){census[o+${CENSUS.CUBES}u+432u+k]=keys[k];}
  }
}`);

  // censusImportance: importanceTile and publishImportance (l.2342, l.2405), 8 tiles a lane; the page's
  // required bits (interfaceBits, l.2534) by workgroup atomics.
  const censusImportance = k("censusImportance", [CFG, setRW, dirRO, velRO, phiRO, censusRW], OWNER + HOOKS + CENSUS_FORMULAS + /* wgsl */`
var<workgroup> wgReq:array<atomic<u32>,16>;
var<workgroup> wgRequired:atomic<u32>;
var<workgroup> wgGo:u32;
@compute @workgroup_size(64) fn censusImportance(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lane:u32){
  ncInit();
  let g=nbGroup(wid);if(lane==0u){wgGo=select(0u,1u,g<nbListCount(NB_LIST_C_ALL));}if(workgroupUniformLoad(&wgGo)==0u){return;}
  let cs=nbListSlot(NB_LIST_C_ALL,g);let o=NC_S*cs;
  if(lane<16u){atomicStore(&wgReq[lane],0u);}
  if(lane==0u){atomicStore(&wgRequired,0u);}
  workgroupBarrier();
  for(var j=0u;j<8u;j++){
    let i=lane+64u*j;let t=xTileOf(cs,i);
    if(!nbTileInLattice(t)){continue;}
    let keys=ncKeys(cs,i);let own=vec2u(ncW0(cs,i),ncW1(cs,i));
    let w1=ncImportance(t,keys,ncGap(cs,i),own).y;
    census[o+${CENSUS.FIN}u+i]=w1;
    // publishImportance: the crossing cells and gap of a tile seed decide only if it is required.
    let required=(w1&NC_REQUIRED)!=0u;let crossing=(w1&NC_CROSSES)!=0u;
    if(!required){
      if(crossing){census[o+${CENSUS.CROSS}u+2u*i]=0u;census[o+${CENSUS.CROSS}u+2u*i+1u]=0u;census[o+${CENSUS.GAP}u+i]=0xffffffffu;}
      continue;
    }
    atomicAdd(&wgRequired,1u);atomicOr(&wgReq[i>>5u],1u<<(i&31u));
    if(!crossing){
      census[o+${CENSUS.CROSS}u+2u*i]=0xffffffffu;census[o+${CENSUS.CROSS}u+2u*i+1u]=0xffffffffu;
      census[o+${CENSUS.GAP}u+i]=min(u32(ceil(ncTileSpeed(keys))),255u)<<24u;
    }
  }
  workgroupBarrier();
  if(lane<16u){census[o+${CENSUS.REQ}u+lane]=atomicLoad(&wgReq[lane]);}
  if(lane==0u){let n=atomicLoad(&wgRequired);if(n!=0u){atomicAdd(&nbSet[NB_H_CENSUS_REQUIRED],n);}}
}`);

  // censusDecide: decideTile (l.2839) a tile, 8 a lane; then the residency clamp (lane E), WANT and TRIG.
  // A band tile is h only when its F brick and each in-range 27-neighbour of it are resident; a clamped tile's
  // missing bricks are appended as candidates, once a page (a bitmask over the 6^3 bricks around it), and it
  // refines one census late.
  const censusDecide = k("censusDecide", [CFG, setRW, dirRO, velRO, phiRO, censusRO], OWNER + HOOKS + CENSUS_FORMULAS + /* wgsl */`
var<workgroup> wgWant:array<atomic<u32>,16>;
var<workgroup> wgCand:array<atomic<u32>,7>;
var<workgroup> wgCount:array<atomic<u32>,4>;
fn ncJoined(t:vec3i)->bool{
  let b=t>>vec3u(1u);let s=nbFSlotOfBrick(b);if(s>=NB_SEAM){return false;}
  let j=u32(t.x&1)|(u32(t.y&1)<<1u)|(u32(t.z&1)<<2u);
  return ((atomicLoad(&nbSet[NB_S_TILE_OCC+s])>>(8u+j))&1u)!=0u;
}
var<workgroup> wgGo:u32;
@compute @workgroup_size(64) fn censusDecide(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lane:u32){
  ncInit();
  let g=nbGroup(wid);if(lane==0u){wgGo=select(0u,1u,g<nbListCount(NB_LIST_C_ALL));}if(workgroupUniformLoad(&wgGo)==0u){return;}
  let cs=nbListSlot(NB_LIST_C_ALL,g);
  if(lane<16u){atomicStore(&wgWant[lane],0u);}
  if(lane<7u){atomicStore(&wgCand[lane],0u);}
  if(lane<4u){atomicStore(&wgCount[lane],0u);}
  workgroupBarrier();
  let reach=bitcast<f32>(atomicLoad(&nbSet[NB_H_CENSUS_REACH]));
  let base=4*nbKeyCoord(nbCRec(cs,NB_REC_KEY))-vec3i(1);
  for(var j=0u;j<8u;j++){
    let i=lane+64u*j;let t=xTileOf(cs,i);
    if(!nbTileInLattice(t)){continue;}
    let band=ncDecide(t,ncJoined(t),ncFin(cs,i),ncKeys(cs,i),reach);
    if(!band){continue;}
    atomicAdd(&wgCount[0],1u);if(!nbTileH(cs,i)){atomicAdd(&wgCount[1],1u);}
    let b=t>>vec3u(1u);var clamped=false;
    for(var n=0u;n<27u;n++){
      let nb=b+nbNeighbourOffset(n);if(!nbBrickInRange(nb)){continue;}
      let s=nbFSlotOfBrick(nb);if(s<NB_SEAM){continue;}
      clamped=true;
      if(s==NB_NONE){let l=nb-base;atomicOr(&wgCand[u32(l.x+6*l.y+36*l.z)>>5u],1u<<(u32(l.x+6*l.y+36*l.z)&31u));}
      else{nbFail(NB_FAIL_UNSUPPORTED,nbKey(nb));}
    }
    if(clamped){atomicAdd(&wgCount[2],1u);continue;}
    atomicOr(&wgWant[i>>5u],1u<<(i&31u));
  }
  // TRIG: an F child's bit j is its tile j's importance triggers (final word 1 bits 16..21).
  let fs=nbChild(cs,lane);
  if(fs<NB_SEAM){
    let b=vec3u(lane&3u,(lane>>2u)&3u,lane>>4u);var bits=0u;
    for(var j=0u;j<8u;j++){let q=2u*b+vec3u(j&1u,(j>>1u)&1u,j>>2u);bits|=select(0u,1u<<j,((ncFin(cs,q.x|(q.y<<3u)|(q.z<<6u))>>16u)&63u)!=0u);}
    atomicStore(&nbSet[NB_S_TRIG+fs],bits);
    if(bits!=0u){atomicAdd(&wgCount[3],countOneBits(bits));}
  }
  workgroupBarrier();
  if(lane<16u){atomicStore(&nbSet[NB_S_WANT+16u*cs+lane],atomicLoad(&wgWant[lane]));}
  for(var e=lane;e<216u;e+=64u){
    if(((atomicLoad(&wgCand[e>>5u])>>(e&31u))&1u)==0u){continue;}
    let nb=base+vec3i(i32(e%6u),i32((e/6u)%6u),i32(e/36u));let key=nbKey(nb);
    let a=atomicAdd(&nbSet[NB_H_N_CAND],1u);
    if(a<NB_FCAP){atomicStore(&nbSet[NB_S_CAND+a],key);}else{nbFail(NB_FAIL_LIST,key);}
    atomicAdd(&nbSet[NB_H_CENSUS_CAND],1u);
  }
  if(lane==0u){
    let c=vec4u(atomicLoad(&wgCount[0]),atomicLoad(&wgCount[1]),atomicLoad(&wgCount[2]),atomicLoad(&wgCount[3]));
    if(c.x!=0u){atomicAdd(&nbSet[NB_H_CENSUS_BAND],c.x);}
    if(c.y!=0u){atomicAdd(&nbSet[NB_H_CENSUS_REFINED],c.y);}
    if(c.z!=0u){atomicAdd(&nbSet[NB_H_CENSUS_CLAMPED],c.z);}
    if(c.w!=0u){atomicAdd(&nbSet[NB_H_CENSUS_TRIGGERED],c.w);}
  }
}`);

  return { censusBegin, censusFine, censusCoarse, censusCubes, censusImportance, censusDecide };
}

/** setCensus's placements, in order (lane E schedules them before setTransfer). */
export function brickCensusSchedule(L: BrickLayout, list: (k: number, g: "G1") => { indirect: number }, cAll: number = LIST.C_ALL, fAll: number = LIST.F_ALL) {
  const c = brickCensusKernels(L);
  return [
    { kernel: c.censusBegin, dispatch: { direct: [1, 1, 1] as const } },
    { kernel: c.censusFine, dispatch: list(fAll, "G1") },
    { kernel: c.censusCoarse, dispatch: list(cAll, "G1") },
    { kernel: c.censusCubes, dispatch: list(cAll, "G1") },
    { kernel: c.censusImportance, dispatch: list(cAll, "G1") },
    { kernel: c.censusDecide, dispatch: list(cAll, "G1") },
  ];
}
/** The indirect word of list k's triple g (as brick-engine.ts places it). */
export const censusListArgs = (k: number, g: "G1") => ({ indirect: ARGS.PER_LIST * k + ARGS[g] });
