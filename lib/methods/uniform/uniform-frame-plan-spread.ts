/** A separable maximum needs its intermediate samples. With sparse,
 * directional allocation, an AIR hole can cut a path between two stored
 * tiles. Propagate an unbounded certificate through that hole so the
 * destination uses the general sampler. SOLID remains a zero boundary.
 * `radius` is overridden only by the independent device oracle. A fixed
 * grid strides the resident tile keys (ulResidentCount, ulResidentBlock:
 * uniform-live-lists, or the including module's own). */
export function uniformFramePlanSpreadWGSL(axis: number, radius = "umCertificateRadius(bitcast<f32>(atomicLoad(&umSupport[4u*UM_TILES])))"): string {
  return /* wgsl */ `
@compute @workgroup_size(64) fn spread${axis}(@builtin(global_invocation_id) gid:vec3u,@builtin(num_workgroups) groups:vec3u){
 let reach=i32(${radius});
 for(var visit=gid.x;visit<512u*ulResidentCount();visit+=64u*groups.x){spread${axis}Tile(512u*ulResidentBlock(visit>>9u)+(visit&511u),reach);}
}
fn spread${axis}Tile(tile:u32,reach:i32){
 if(!umTileLive(tile)){return;}
 let base=${axis === 1 ? "UM_TILES" : "0u"};let into=${axis === 1 ? "0u" : "UM_TILES"};
 let p=vec3i(umTileCoord(tile));var plus=vec3u(0);var minus=vec3u(0);
 let first=clamp(p-vec3i(reach),vec3i(0),vec3i(511));let last=clamp(p+vec3i(reach),vec3i(0),vec3i(511));
 for(var q=first.${"xyz"[axis]};q<=last.${"xyz"[axis]};q++){
  var r=p;r.${"xyz"[axis]}=q;let k=umTileKey(r);
  if(!umKeyStored(k)){
   if(umTileKind(r)==UM_AIR){plus=vec3u(UM_REACH_SATURATED);minus=plus;}
   continue;
  }
  let value=speeds[base+k];plus=max(plus,umReachPlus(value));minus=max(minus,umReachMinus(value));
 }
 speeds[into+tile]=umPackReach(plus,minus);
}
`;
}
