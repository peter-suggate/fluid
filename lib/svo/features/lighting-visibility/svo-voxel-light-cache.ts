/** Direct-mapped visibility cache. Tags retain the complete virtual voxel ID;
 * collisions are misses, never another surface's lighting. Each frame queues
 * only one writer per physical slot, so a vec4 texture store cannot tear. */
export const SVO_VOXEL_LIGHT_CACHE_MAX_ENTRIES = 4 * 1024 * 1024;
export const SVO_VOXEL_LIGHT_CACHE_BYTES_PER_ENTRY = 16;
export function svoVoxelLightCacheShape(pages:number):[number,number,number]{
  if(!Number.isSafeInteger(pages)||pages<0)throw new RangeError("Invalid light-cache page count");
  const entries=Math.min(SVO_VOXEL_LIGHT_CACHE_MAX_ENTRIES,2**Math.ceil(Math.log2(Math.max(1,pages*512))));
  const x=Math.min(256,entries),y=Math.min(256,entries/x);
  return [x,y,entries/x/y];
}
export function svoVoxelLightCacheSlot(voxel:number,mask:number):number{
  let h=voxel>>>0;h=Math.imul(h^(h>>>16),0x7feb352d);h=Math.imul(h^(h>>>15),0x846ca68b);
  return ((h^(h>>>16))&mask)>>>0;
}
export const svoVoxelLightHashWGSL=/* wgsl */`
fn dryVoxelLightSlot(voxel:u32)->u32{
  var h=voxel;h=(h^(h>>16u))*0x7feb352du;h=(h^(h>>15u))*0x846ca68bu;
  return (h^(h>>16u))&dryVoxelLight.atlasShape.w;
}
fn dryVoxelLightTexel(voxel:u32)->vec3u{
  let slot=dryVoxelLightSlot(voxel);let shape=max(dryVoxelLight.atlasShape.xyz,vec3u(1));
  return vec3u(slot%shape.x,(slot/shape.x)%shape.y,slot/(shape.x*shape.y));
}
`;
