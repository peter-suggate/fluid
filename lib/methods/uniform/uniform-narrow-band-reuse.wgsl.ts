/** Bounded exact crossing-mask cache. Payloads survive every borrowed bins
 * bank; metadata is independent of those scratch lifetimes. */
export function narrowBandReuseLayout(base:number,regions:number,capacity=regions){
 const live=base,control=live+regions,history=control+3;
 // Each history epoch: phi revisions, world xyz, clear validity, then125
 // closure topology words and128binary-openmask words per region.
 // The other bank stays immutable.
 const historyWords=capacity?258*regions:0;
 const allowed=history+2*historyWords,valid=allowed+(capacity?regions:0);
 const metricValid=valid+64*capacity, nearestValid=metricValid+64*capacity;
 const nearestState=nearestValid+regions, nearestFlags=nearestState+158*capacity;
 const masks=nearestFlags+regions;
 const nearest=masks+128*capacity, metric=nearest+6144*capacity, plane=metric+8000*capacity, regionPlane=plane+384*capacity;
 return {base,regions,capacity,live,control,history,historyWords,allowed,valid,masks,metricValid,nearestValid,nearestState,nearestFlags,nearest,metric,plane,regionPlane,metadataWords:masks-base,words:regionPlane+8*capacity-base};
}
export type NarrowBandReuseLayout=ReturnType<typeof narrowBandReuseLayout>;
export function narrowBandReuseWGSL(l:NarrowBandReuseLayout){return /* wgsl */`
override nbMetricPlanes:bool=true;
const NB_REUSE_REGIONS:u32=${l.regions}u;const NB_REUSE_CAPACITY:u32=${l.capacity}u;
const NB_REUSE_LIVE:u32=${l.live}u;const NB_REUSE_CONTROL:u32=${l.control}u;
const NB_REUSE_HISTORY:u32=${l.history}u;const NB_REUSE_HISTORY_WORDS:u32=${l.historyWords}u;
const NB_REUSE_ALLOWED:u32=${l.allowed}u;const NB_REUSE_VALID:u32=${l.valid}u;const NB_REUSE_MASK:u32=${l.masks}u;
const NB_REUSE_REGION_PLANE:u32=${l.regionPlane}u;
const NB_REUSE_PLANE:u32=${l.plane}u;
const NB_REUSE_METRIC_VALID:u32=${l.metricValid}u;const NB_REUSE_METRIC:u32=${l.metric}u;const NB_REUSE_METRIC_CAPACITY:u32=${l.capacity}u;
const NB_REUSE_NEAREST_VALID:u32=${l.nearestValid}u;const NB_REUSE_NEAREST_STATE:u32=${l.nearestState}u;const NB_REUSE_NEAREST_FLAGS:u32=${l.nearestFlags}u;const NB_REUSE_NEAREST:u32=${l.nearest}u;
fn nbReuseRegion(tile:u32)->u32{return 8u*(tile>>9u)+((tile>>2u)&1u)+2u*((tile>>5u)&1u)+4u*((tile>>8u)&1u);}
fn nbReuseLocal(tile:u32)->u32{return (tile&3u)+4u*((tile>>3u)&3u)+16u*((tile>>6u)&3u);}
fn nbReuseTile(region:u32,lane:u32)->u32{
 let r=region&7u;let local=vec3u(lane&3u,(lane>>2u)&3u,lane>>4u)+4u*vec3u(r&1u,(r>>1u)&1u,r>>2u);
 return 512u*(region>>3u)+local.x+8u*(local.y+8u*local.z);
}
fn nbReuseEnabled()->bool{return atomicLoad(&bins[NB_REUSE_CONTROL+1u])!=0u;}
fn nbReuseCacheEnabled()->bool{return nbReuseEnabled()&&atomicLoad(&bins[NB_REUSE_CONTROL+2u])!=0u;}
fn nbReuseCurrent()->u32{return NB_REUSE_HISTORY+NB_REUSE_HISTORY_WORDS*atomicLoad(&bins[NB_REUSE_CONTROL]);}
fn nbReusePrevious()->u32{return NB_REUSE_HISTORY+NB_REUSE_HISTORY_WORDS*(1u-atomicLoad(&bins[NB_REUSE_CONTROL]));}
var<workgroup> nbReuseClean:atomic<u32>;
var<workgroup> nbCrossingLive:u32;
// Called uniformly by one 64-lane group per 16h region, before any fine
// crossing samples. A prior metadata bank is never written by this pass.
fn nbPrepareReuse(region:u32,lane:u32,tile:u32)->bool{
 if(lane==0u){nbCrossingLive=select(0u,1u,umTileLive(tile));}
 if(workgroupUniformLoad(&nbCrossingLive)==0u){
  if(nbReuseCacheEnabled()&&region<NB_REUSE_CAPACITY){
   if(lane==0u){atomicStore(&bins[nbReuseCurrent()+4u*NB_REUSE_REGIONS+region],0u);atomicStore(&bins[NB_REUSE_ALLOWED+region],0u);}
   atomicStore(&bins[NB_REUSE_VALID+64u*region+lane],0u);
   atomicStore(&bins[NB_REUSE_METRIC_VALID+64u*region+lane],0u);
   atomicStore(&bins[NB_REUSE_PLANE+384u*region+6u*lane+5u],0u);
  }
  return false;
 }
 if(lane==0u){atomicStore(&nbReuseClean,select(0u,1u,nbReuseCacheEnabled()&&region<NB_REUSE_CAPACITY));}
 workgroupBarrier();
 if(nbReuseCacheEnabled()&&region<NB_REUSE_CAPACITY){
  let current=nbReuseCurrent();let previous=nbReusePrevious();
  let origin=16u*(umTileCoord(tile)/4u);let revision=atomicLoad(&bins[NB_REUSE_LIVE+region]);
  if(lane==0u){
   let clear=umTileLive(tile);
   atomicStore(&bins[current+region],revision);
   for(var a=0u;a<3u;a++){atomicStore(&bins[current+NB_REUSE_REGIONS*(1u+a)+region],origin[a]);}
   atomicStore(&bins[current+4u*NB_REUSE_REGIONS+region],select(0u,1u,clear));
   var valid=clear&&revision>0u&&revision<0xfffffffeu&&revision==atomicLoad(&bins[previous+region]);
   valid=valid&&atomicLoad(&bins[previous+4u*NB_REUSE_REGIONS+region])!=0u;
   for(var a=0u;a<3u;a++){valid=valid&&atomicLoad(&bins[previous+NB_REUSE_REGIONS*(1u+a)+region])==origin[a];}
   if(!valid){atomicStore(&nbReuseClean,0u);}
  }
  // Both operators consume only the exact binary >1e-5 open predicate;
  // fractional solid geometry is not a dependency of these numerical results.
  var open=vec2u(0xffffffffu);if(umSolidEnabled()){open=umSolidOpenCells(tile);}
  for(var word=0u;word<2u;word++){
   let at=130u*NB_REUSE_REGIONS+128u*region+2u*lane+word;
   if(open[word]!=atomicLoad(&bins[previous+at])){atomicStore(&nbReuseClean,0u);}
   atomicStore(&bins[current+at],open[word]);
  }
  // A crossing cell uses this region's17^3 phi closure. Its positive-side
  // sampler may choose the tile just above an upper face, hence precisely
  // the5^3 tile topology closure (64 interior +61 upper-face sites).
  // Activation only decides whether a mask is requested; it does not change
  // an already computed mask, so it is deliberately not a dependency here.
  for(var k=lane;k<125u;k+=64u){
   let q=vec3i(origin/4u+umCorner(k,5u));let key=umTileKey(q);
   var topology=0xfffffff0u|umTileKind(q);if(umKeyStored(key)){topology=umTopology[key];}
   let at=5u*NB_REUSE_REGIONS+125u*region+k;
   if(topology!=atomicLoad(&bins[previous+at])){atomicStore(&nbReuseClean,0u);}
   atomicStore(&bins[current+at],topology);
  }
 }
 workgroupBarrier();let clean=atomicLoad(&nbReuseClean)!=0u;
 if(nbReuseCacheEnabled()&&region<NB_REUSE_CAPACITY){
  if(lane==0u){atomicStore(&bins[NB_REUSE_ALLOWED+region],select(0u,1u,clean));}
  if(!clean){atomicStore(&bins[NB_REUSE_VALID+64u*region+lane],0u);atomicStore(&bins[NB_REUSE_METRIC_VALID+64u*region+lane],0u);atomicStore(&bins[NB_REUSE_PLANE+384u*region+6u*lane+5u],0u);}
 }
 workgroupBarrier();return clean;
}
fn nbCrossingReuse(tile:u32)->bool{
 let region=nbReuseRegion(tile);if(region>=NB_REUSE_CAPACITY){return false;}
 return nbReuseCacheEnabled()&&atomicLoad(&bins[NB_REUSE_ALLOWED+region])!=0u&&atomicLoad(&bins[NB_REUSE_VALID+64u*region+nbReuseLocal(tile)])!=0u;
}
fn nbCrossingCachedMask(tile:u32)->vec2u{
 let at=NB_REUSE_MASK+128u*nbReuseRegion(tile)+2u*nbReuseLocal(tile);
 return vec2u(atomicLoad(&bins[at]),atomicLoad(&bins[at+1u]));
}
fn nbCommitCrossing(tile:u32,mask:vec2u){
 let region=nbReuseRegion(tile);if(!nbReuseCacheEnabled()||region>=NB_REUSE_CAPACITY){return;}
 let local=nbReuseLocal(tile);let at=NB_REUSE_MASK+128u*region+2u*local;
 atomicStore(&bins[at],mask.x);atomicStore(&bins[at+1u],mask.y);atomicStore(&bins[NB_REUSE_VALID+64u*region+local],1u);
}
// A metric query and its geometry/trace support fit inside the adjacent
// 16h regions. Exact producer revisions include their positive phi closure.
fn nbMetricRegionReusable(tile:u32)->bool{
 let r=nbReuseRegion(tile);return nbReuseCacheEnabled()&&r<NB_REUSE_CAPACITY&&(atomicLoad(&bins[NB_REUSE_NEAREST_FLAGS+r])&2u)!=0u;
}
fn nbNearestReusable(tile:u32)->bool{
 let r=nbReuseRegion(tile);return nbReuseCacheEnabled()&&r<NB_REUSE_CAPACITY&&(atomicLoad(&bins[NB_REUSE_NEAREST_FLAGS+r])&1u)!=0u;
}
fn nbNearestCacheAt(tile:u32,axis:u32,row:u32)->u32{
 return NB_REUSE_NEAREST+6144u*nbReuseRegion(tile)+1024u*axis+16u*nbReuseLocal(tile)+row;
}
fn nbNearestFinalAt(tile:u32,row:u32)->u32{
 return NB_REUSE_NEAREST+6144u*nbReuseRegion(tile)+2048u+64u*nbReuseLocal(tile)+4u*row;
}
fn nbMarkPhiChanged(p:vec3u){
 if(!nbReuseEnabled()){return;}
 for(var k=0u;k<8u;k++){
  let side=umCorner(k,2u);if(any((side!=vec3u(0u))&((p&vec3u(15u))!=vec3u(0u)))){continue;}
  let cell=vec3i(p)-vec3i(side);let tile=umTileKey(cell>>vec3u(2u));if(!umKeyStored(tile)){continue;}
  let at=NB_REUSE_LIVE+nbReuseRegion(tile);let old=atomicLoad(&bins[at]);
  if(old>0u){atomicMax(&bins[at],min(old,0xfffffffdu)+1u);}
 }
}
fn nbStoreReconstructed(p:vec3i,value:f32,width:u32){
 if(nbReuseEnabled()){
  var old=0.0;if(width==4u){old=umLoadCorner(vec3u(p));}else{old=umLoadVertex(vec3u(p));}
  if(bitcast<u32>(old)!=bitcast<u32>(value)){nbMarkPhiChanged(vec3u(p));}
 }
 textureStore(outputPhi,p,vec4f(value));
}
fn nbStoreInPlace(p:vec3u,value:f32){
 if(nbReuseEnabled()&&bitcast<u32>(textureLoad(outputPhi,vec3i(p)).x)!=bitcast<u32>(value)){nbMarkPhiChanged(p);}
 textureStore(outputPhi,vec3i(p),vec4f(value));
}
`;}
