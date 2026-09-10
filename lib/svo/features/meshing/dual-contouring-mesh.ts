/** Extraction hook; common scheduler owns count/emit, arena rollback and publish. */
export const svoDualContouringMeshWGSL = /* wgsl */ `
fn meshDcWord(voxel:u32,lane:u32)->u32{
 let base=atomicLoad(&meshState[47u])&0x7fffffffu;let at=base+voxel*4u+lane;
 if(base==0u||voxel==SVO_INVALID||at>=arrayLength(&meshMaintenance)){return 0u;}
 return meshMaintenance[at];
}
fn meshDcPoint(voxel:u32)->vec3f{return bitcast<vec3f>(vec3u(meshDcWord(voxel,0u),meshDcWord(voxel,1u),meshDcWord(voxel,2u)));}
fn meshDcActive(voxel:u32)->bool{let signs=meshDcWord(voxel,3u)&255u;return signs!=0u&&signs!=255u;}
fn meshDcPack(p:vec3f)->u32{let q=vec3u(round(clamp(p*256.,vec3f(0),vec3f(1023))));return q.x|(q.y<<10u)|(q.z<<20u);}
fn meshDcExtract(base:vec3u,scale:u32,depth:u32,local:u32,n:u32){
 if(local>=3u*n){return;}let axis=local/n;let layer=local%n;let u=(axis+1u)%3u;let v=(axis+2u)%3u;
 for(var y=0u;y<n;y+=1u){for(var x=0u;x<n;x+=1u){
  var c=vec3u(0);c[axis]=layer;c[u]=x;c[v]=y;let edge=vec3f(base+c*scale);
  let own=meshRegionAt(edge+vec3f(.5*f32(scale)));let signs=meshDcWord(own.voxel,3u)&255u;
  if(((signs&1u)!=0u)==((signs&(1u<<(1u<<axis)))!=0u)){continue;}
  var ids:array<u32,4>;var vertices:array<vec3f,4>;var valid=true;var sharp=0u;
  for(var i=0u;i<4u;i+=1u){
   var p=edge+vec3f(.5*f32(scale));
   if(i==1u||i==2u){p[u]-=f32(scale);}if(i==2u||i==3u){p[v]-=f32(scale);}
   let region=meshRegionAt(p);ids[i]=region.voxel;
   if(region.size!=f32(scale)||!meshDcActive(region.voxel)){valid=false;break;}
   vertices[i]=meshDcPoint(region.voxel);sharp|=(meshDcWord(region.voxel,3u)>>24u)&1u;
  }
  if(!valid){continue;}
  var origin=edge;origin[u]-=f32(scale);origin[v]-=f32(scale);
  if(any(origin<vec3f(0))){continue;}
  let positive=(signs&1u)!=0u;
  for(var t=0u;t<2u;t+=1u){
   let a=0u;let b=select(t+2u,t+1u,positive);let d=select(t+1u,t+2u,positive);
   let p=meshDcPack((vertices[a]-origin)/f32(scale));let q=meshDcPack((vertices[b]-origin)/f32(scale));let r=meshDcPack((vertices[d]-origin)/f32(scale));
   if(p==q||q==r||r==p){continue;}
   meshAppend(vec3u(origin),vec3u(p,q,r),0xc0000000u|(sharp<<29u)|meshPackFace(0u,0u,depth),own.identity);
  }
 }}
}
`;
