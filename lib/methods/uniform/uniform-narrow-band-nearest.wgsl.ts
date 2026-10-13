/** Share bounded integer X/XY minima between neighboring search tiles.
 * Bank A's previous nodal metric expires before crossing search. The two
 * packed banks occupy half a word per domain cell; reconstruction or the
 * following metric build fills the bank before particle membership reads it.
 * Equal squared distances retain lower offsets, x before y before z. */
export const narrowBandNearestEntries = ["depthNearestX", "depthNearestY", "depthNearestZ"] as const;
export const narrowBandNearestWGSL=/* wgsl */`
fn nbGlobalNearestRow(p:vec3i,axis:u32)->u32{
 if(!umCellPresent(p)){return 0xffffffffu;}
 // A nonempty X or XY result is at most five cells from a crossing in
 // each processed axis, hence within the existing two-tile reach.
 // Unlisted tiles therefore represent empty rows, without clearing them.
 let tile=nbBandTile(vec3u(p));if((atomicLoad(&bins[NB_BAND_REACH+tile])&1u)==0u){return 0xffffffffu;}
 let local=vec3u(p)&vec3u(3u);
 return atomicLoad(&bins[NB_DEPTH_A+axis*(UM_CELL_KEYS/4u)+16u*tile+local.y+4u*local.z]);
}
@compute @workgroup_size(64) fn depthNearestX(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) lane:u32){
 let job=4u*group.x+lane/16u;if(job>=atomicLoad(&bins[NB_BAND+1u])){return;}
 let tile=atomicLoad(&bins[NB_BAND_SEARCH+job]);let coord=umTileCoord(tile);
 let p=vec3i(4u*coord+vec3u(0u,lane%4u,(lane%16u)/4u));
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
}
@compute @workgroup_size(64) fn depthNearestY(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) lane:u32){
 let job=4u*group.x+lane/16u;if(job>=atomicLoad(&bins[NB_BAND+1u])){return;}
 let tile=atomicLoad(&bins[NB_BAND_SEARCH+job]);let coord=umTileCoord(tile);
 let p=vec3i(4u*coord+vec3u(0u,lane%4u,(lane%16u)/4u));
 var best=vec4u(255u);var distance=vec4i(1000);
 for(var o=-5;o<=5;o++){
  let row=nbGlobalNearestRow(p+vec3i(0,o,0),0u);let code=(vec4u(row)>>vec4u(0u,8u,16u,24u))&vec4u(255u);
  let dx=vec4i(code)-5;let d=dx*dx+o*o;let closer=(code!=vec4u(255u))&(d<distance);
  best=select(best,code|vec4u(u32(o+5)<<4u),closer);distance=select(distance,d,closer);
 }
 atomicStore(&bins[NB_DEPTH_A+UM_CELL_KEYS/4u+16u*tile+lane%16u],best.x|(best.y<<8u)|(best.z<<16u)|(best.w<<24u));
}
@compute @workgroup_size(64) fn depthNearestZ(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) lane:u32){
 let job=4u*group.x+lane/16u;if(job>=atomicLoad(&bins[NB_BAND+1u])){return;}
 let tile=atomicLoad(&bins[NB_BAND_SEARCH+job]);let coord=umTileCoord(tile);
 let p=vec3i(4u*coord+vec3u(0u,lane%4u,(lane%16u)/4u));var nearest=vec4u(NB_NO_SURFACE);var distance=vec4i(1000);
 for(var o=-5;o<=5;o++){
  let row=nbGlobalNearestRow(p+vec3i(0,0,o),1u);let code=(vec4u(row)>>vec4u(0u,8u,16u,24u))&vec4u(255u);
  let dx=vec4i(code&vec4u(15u))-5;let dy=vec4i(code>>vec4u(4u))-5;let d=dx*dx+dy*dy+o*o;
  let closer=(code!=vec4u(255u))&(d<distance);
  let x=vec4u(vec4i(p.x)+vec4i(0,1,2,3)+dx);let y=vec4u(vec4i(p.y)+dy);let z=u32(p.z+o);
  let at=vec4u(umCellAt(vec3u(x.x,y.x,z)),umCellAt(vec3u(x.y,y.y,z)),umCellAt(vec3u(x.z,y.z,z)),umCellAt(vec3u(x.w,y.w,z)));
  nearest=select(nearest,at,closer);distance=select(distance,d,closer);
 }
 let at=NB_DEPTH_B+umCellAt(vec3u(p));
 atomicStore(&bins[at],nearest.x);atomicStore(&bins[at+1u],nearest.y);atomicStore(&bins[at+2u],nearest.z);atomicStore(&bins[at+3u],nearest.w);
}
`;
