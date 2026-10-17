/** Share bounded integer X/XY minima between neighboring search tiles.
 * Bank A's previous nodal metric expires before crossing search. The two
 * packed banks occupy half a word per domain cell; reconstruction or the
 * following metric build fills the bank before particle membership reads it.
 * Equal squared distances retain lower offsets, x before y before z. */
export const narrowBandNearestControlEntries = ["depthReuseSnapshot", "depthReuseSelect"] as const;
export const narrowBandNearestEntries = ["depthNearestX", "depthNearestY", "depthNearestZ"] as const;
export const narrowBandNearestWGSL=/* wgsl */`
var<workgroup> nbReuseLive:u32;
// Compare the masks actually consumed by this search, rather than phi
// revisions: continuous surface motion inside the same crossing cells does
// not change these integer minima. One group owns each 16h control region.
@compute @workgroup_size(64) fn depthReuseSnapshot(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) lane:u32){
 for(var region=group.x;region<NB_REUSE_CAPACITY;region+=1024u){
  let tile=nbReuseTile(region,lane);
  // Pooled storage includes free 32h pages. Invalidate without touching
  // their masks, world mapping, or halo page tables.
  // 2: free and its words cleared by this search's prepare or before (nbPrepareReuse).
  if(lane==0u){nbReuseLive=select(select(0u,2u,atomicLoad(&bins[NB_REUSE_ALLOWED+region])==2u),1u,umTileLive(tile));}
  let live=workgroupUniformLoad(&nbReuseLive);
  if(live!=1u){
   if(lane==0u){atomicStore(&bins[NB_REUSE_NEAREST_VALID+region],0u);atomicStore(&bins[NB_REUSE_NEAREST_FLAGS+region],0u);atomicStore(&bins[NB_REUSE_REGION_PLANE+8u*region+7u],0u);}
   if(live==0u){atomicStore(&bins[NB_REUSE_METRIC_VALID+64u*region+lane],0u);}continue;
  }
  let origin=(umTileCoord(tile)/4u)*4u;
  let base=NB_REUSE_NEAREST_STATE+158u*region;
  if(lane==0u){atomicStore(&nbReuseClean,select(0u,1u,nbReuseCacheEnabled()&&umTileLive(tile)&&(atomicLoad(&bins[NB_REUSE_NEAREST_VALID+region])&5u)==5u));}
  workgroupBarrier();
  for(var word=0u;word<2u;word++){
   let mask=atomicLoad(&bins[NB_BAND_MASK+2u*tile+word]);let at=base+2u*lane+word;
   if(mask!=atomicLoad(&bins[at])){atomicStore(&nbReuseClean,0u);}atomicStore(&bins[at],mask);
  }
  if(lane<27u){
   let key=umTileKey(vec3i(origin)+4*(vec3i(umCorner(lane,3u))-1));let at=base+128u+lane;
   if(key!=atomicLoad(&bins[at])){atomicStore(&nbReuseClean,0u);}atomicStore(&bins[at],key);
  }
  if(lane<3u){let at=base+155u+lane;if(origin[lane]!=atomicLoad(&bins[at])){atomicStore(&nbReuseClean,0u);}atomicStore(&bins[at],origin[lane]);}
  workgroupBarrier();
  if(lane==0u){atomicStore(&bins[NB_REUSE_NEAREST_VALID+region],select(1u,3u,atomicLoad(&nbReuseClean)!=0u)|select(0u,4u,umTileLive(tile)));}
  workgroupBarrier();
  nbAggregateMetricPlane(region,lane);
 }
}
var<workgroup> nbMetricClean:atomic<u32>;
@compute @workgroup_size(64) fn depthReuseSelect(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) lane:u32){
 for(var region=group.x;region<NB_REUSE_CAPACITY;region+=1024u){
  let tile=nbReuseTile(region,lane);
  if(lane==0u){nbReuseLive=select(0u,1u,umTileLive(tile));}
  if(workgroupUniformLoad(&nbReuseLive)==0u){continue;}
  if(lane==0u){let enabled=select(0u,1u,nbReuseCacheEnabled());atomicStore(&nbReuseClean,enabled);atomicStore(&nbMetricClean,enabled);}
  workgroupBarrier();
  if(lane<27u){
   let origin=vec3i((umTileCoord(tile)/4u)*4u);
   let key=umTileKey(origin+4*(vec3i(umCorner(lane,3u))-1));
   // An unstored neighbour may later be claimed. A claim promotes the
   // regions beside it to HOT, so a COLD region's metric stands without it.
   if(!umKeyStored(key)){if(!nbRegionCold(region)){atomicStore(&nbMetricClean,0u);}}
   else{
    let other=nbReuseRegion(key);
    if(other>=NB_REUSE_CAPACITY){atomicStore(&nbReuseClean,0u);atomicStore(&nbMetricClean,0u);}
    else{
     if((atomicLoad(&bins[NB_REUSE_NEAREST_VALID+other])&2u)==0u){atomicStore(&nbReuseClean,0u);}
     if(atomicLoad(&bins[NB_REUSE_ALLOWED+other])!=1u){atomicStore(&nbMetricClean,0u);}
    }
   }
  }
  workgroupBarrier();
  let nearest=atomicLoad(&nbReuseClean)!=0u;let metric=nearest&&atomicLoad(&nbMetricClean)!=0u;
  if(lane==0u){atomicStore(&bins[NB_REUSE_NEAREST_FLAGS+region],select(0u,1u,nearest)|select(0u,2u,metric));}
  // Clear omitted tiles too: neighbor changes cannot age back into hits.
  if(!metric){atomicStore(&bins[NB_REUSE_METRIC_VALID+64u*region+lane],0u);}
  workgroupBarrier();
 }
}
fn nbGlobalNearestRow(p:vec3i,axis:u32)->u32{
 let tile=umTileKey(p>>vec3u(2u));if(tile>=UM_TILES){return 0xffffffffu;}
 // A nonempty X or XY result is at most five cells from a crossing in
 // each processed axis, hence within the existing two-tile reach.
 // Unlisted tiles therefore represent empty rows, without clearing them.
 if((atomicLoad(&bins[NB_BAND_REACH+tile])&1u)==0u){return 0xffffffffu;}
 let local=vec3u(p)&vec3u(3u);
 if(nbNearestReusable(tile)){return atomicLoad(&bins[nbNearestCacheAt(tile,axis,local.y+4u*local.z)]);}
 return atomicLoad(&bins[NB_DEPTH_A+axis*(UM_CELL_KEYS/4u)+16u*tile+local.y+4u*local.z]);
}
@compute @workgroup_size(64) fn depthNearestX(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) lane:u32){
 let job=4u*group.x+lane/16u;if(job>=atomicLoad(&bins[NB_BAND+1u])){return;}
 let tile=atomicLoad(&bins[NB_BAND_SEARCH+job]);let coord=umTileCoord(tile);
 let p=vec3i(4u*coord+vec3u(0u,lane%4u,(lane%16u)/4u));
 let row=lane%16u;
 if(nbNearestReusable(tile)){return;}
 let bits=nbBandRow(coord,p.y,p.z);var packed=0xffffffffu;
 if(bits!=0u){
  packed=0u;
  for(var x=0u;x<4u;x++){
   let centre=x+5u;let left=bits&((1u<<(centre+1u))-1u);let right=bits>>centre;
   let dl=select(255u,centre-firstLeadingBit(left),left!=0u);let dr=select(255u,firstTrailingBit(right),right!=0u);
   let delta=min(dl,dr);let code=select(255u,select(5u+dr,5u-dl,dl<=dr),delta<=5u);
   packed|=code<<(8u*x);
  }
 }
 atomicStore(&bins[NB_DEPTH_A+16u*tile+lane%16u],packed);
 if(nbReuseCacheEnabled()&&nbReuseRegion(tile)<NB_REUSE_CAPACITY){atomicStore(&bins[nbNearestCacheAt(tile,0u,lane%16u)],packed);}
}
@compute @workgroup_size(64) fn depthNearestY(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) lane:u32){
 let job=4u*group.x+lane/16u;if(job>=atomicLoad(&bins[NB_BAND+1u])){return;}
 let tile=atomicLoad(&bins[NB_BAND_SEARCH+job]);let coord=umTileCoord(tile);
 let p=vec3i(4u*coord+vec3u(0u,lane%4u,(lane%16u)/4u));
 if(nbNearestReusable(tile)){return;}
 var best=vec4u(255u);var distance=vec4i(1000);
 for(var o=-5;o<=5;o++){
  let row=nbGlobalNearestRow(p+vec3i(0,o,0),0u);let code=(vec4u(row)>>vec4u(0u,8u,16u,24u))&vec4u(255u);
  let dx=vec4i(code)-5;let d=dx*dx+o*o;let closer=(code!=vec4u(255u))&(d<distance);
  best=select(best,code|vec4u(u32(o+5)<<4u),closer);distance=select(distance,d,closer);
 }
 let packed=best.x|(best.y<<8u)|(best.z<<16u)|(best.w<<24u);
 atomicStore(&bins[NB_DEPTH_A+UM_CELL_KEYS/4u+16u*tile+lane%16u],packed);
 if(nbReuseCacheEnabled()&&nbReuseRegion(tile)<NB_REUSE_CAPACITY){atomicStore(&bins[nbNearestCacheAt(tile,1u,lane%16u)],packed);}
}
@compute @workgroup_size(64) fn depthNearestZ(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) lane:u32){
 let job=4u*group.x+lane/16u;if(job>=atomicLoad(&bins[NB_BAND+1u])){return;}
 let tile=atomicLoad(&bins[NB_BAND_SEARCH+job]);let coord=umTileCoord(tile);
 let p=vec3i(4u*coord+vec3u(0u,lane%4u,(lane%16u)/4u));var nearest=vec4u(NB_NO_SURFACE);var distance=vec4i(1000);
 if(nbNearestReusable(tile)){
  let destination=NB_DEPTH_B+umCellAt(vec3u(p));let cached=nbNearestFinalAt(tile,lane%16u);
  for(var k=0u;k<4u;k++){atomicStore(&bins[destination+k],atomicLoad(&bins[cached+k]));}return;
 }
 for(var o=-5;o<=5;o++){
  let row=nbGlobalNearestRow(p+vec3i(0,0,o),1u);let code=(vec4u(row)>>vec4u(0u,8u,16u,24u))&vec4u(255u);
  let dx=vec4i(code&vec4u(15u))-5;let dy=vec4i(code>>vec4u(4u))-5;let d=dx*dx+dy*dy+o*o;
  let closer=(code!=vec4u(255u))&(d<distance);
  // A missing or losing row cannot change any nearest key. In particular,
  // avoid four page-table lookups for its discarded coordinates.
  if(!any(closer)){continue;}
  let x=vec4u(vec4i(p.x)+vec4i(0,1,2,3)+dx);let y=vec4u(vec4i(p.y)+dy);let z=u32(p.z+o);
  let at=vec4u(umCellAt(vec3u(x.x,y.x,z)),umCellAt(vec3u(x.y,y.y,z)),umCellAt(vec3u(x.z,y.z,z)),umCellAt(vec3u(x.w,y.w,z)));
  nearest=select(nearest,at,closer);distance=select(distance,d,closer);
 }
 let at=NB_DEPTH_B+umCellAt(vec3u(p));
 if(nbReuseCacheEnabled()&&nbReuseRegion(tile)<NB_REUSE_CAPACITY){
  let cached=nbNearestFinalAt(tile,lane%16u);for(var k=0u;k<4u;k++){atomicStore(&bins[cached+k],nearest[k]);}
 }
 atomicStore(&bins[at],nearest.x);atomicStore(&bins[at+1u],nearest.y);atomicStore(&bins[at+2u],nearest.z);atomicStore(&bins[at+3u],nearest.w);
}
`;
