/** Continue tangential velocity across a whole, stationary, grid-aligned wall.
 * The exterior-page cache already holds its edge tile this way. An embedded
 * wall needs the same ghost values: ordinary air extrapolation can mix the
 * liquid's normal variation into its tangential velocity at the wall.
 *
 * umCacheTileKind: 0 wholly closed, 1 wholly open, 2 cut/moving/unavailable.
 * Only the component's two tangential directions may move. A unique nearest
 * open tile defines the reflection; ambiguous thin walls retain the existing
 * extension rather than coupling the two fluids through the wall. Diagonal
 * candidates cover the intersection of two walls without an axis-order bias.
 */
export const uniformVelocityWallCacheWGSL=/* wgsl */`
fn umCacheWallSource(tile:vec3i,axis:u32)->vec3i{
 if(umCacheTileKind(tile)!=0u){return tile;}
 let u=(axis+1u)%3u;let v=(axis+2u)%3u;
 var nearest=3u;var count=0u;var source=tile;
 for(var k=0u;k<9u;k++){
  let a=i32(k%3u)-1;let b=i32(k/3u)-1;let distance=u32(abs(a)+abs(b));
  if(distance==0u||distance>nearest){continue;}
  var q=tile;q[u]+=a;q[v]+=b;
  if(umCacheTileKind(q)!=1u){continue;}
  if(distance<nearest){nearest=distance;count=0u;}
  source=q;count++;
 }
 return select(tile,source,count==1u);
}
`;
