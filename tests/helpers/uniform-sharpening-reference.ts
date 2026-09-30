import assert from "node:assert/strict";

// Previous one-tile-per-workgroup implementation. Keep the numerical
// reference independent of the production lane packing.
const referenceFunctions = /* wgsl */ `
fn shSweepJobs()->vec2u{
 let regular=(atomicLoad(&work[SH_ACTIVE_COUNT])+191u)/192u;return vec2u(regular,regular+atomicLoad(&work[SH_COUNTS+2u]));
}
fn shSeamLane(tile:u32,lane:u32)->SHSeamLane{
 if(tile>=UM_TILES){return SHSeamLane(UMOwner(),false,0u,0u);}
 let width=umTileWidth(tile);let per=3u*width*width*width;let side=4u/width;let o=lane/per;let r=lane%per;
 if(o>=side*side*side){return SHSeamLane(UMOwner(),false,0u,0u);}
 let owner=shSeamOwner(tile,umCorner(o,side)*width);return SHSeamLane(owner,r==0u,r/(width*width),r%(width*width));
}
fn shProposeSeam(tile:u32,lane:u32){
 let cell=lane%64u;let axis=lane/64u;let local=umCorner(cell,4u);let o=shSeamOwner(tile,local);if(o.width==0u){return;}
 let face=umPositiveFaceAtAnchor(o,axis,vec3i(umTileCoord(tile)*4u+local));
 if(face.width!=0u&&face.neighbor.width!=0u&&shListed(face.neighbor)){scratch[umRawAt(face)]=umProposal(o,face.neighbor,face);}
}`;
export function sharpeningReference(code:string):string {
 for(const fn of referenceFunctions.trim().split(/\n(?=fn )/)){
  const name=/^fn (\w+)/.exec(fn)![1]!;
  const pattern=new RegExp(`fn ${name}\\([^]*?\\n}`);
  assert.ok(pattern.test(code),`missing reference seam function ${name}`);
  code=code.replace(pattern,fn);
 }
 const packed="tile=shSeamTile(2u*(job-jobs.x)+lane/96u);";
 assert.equal(code.split(packed).length-1,3,"restore all three sweep entries");
 return code.replaceAll(packed,"tile=shSeamTile(job-jobs.x);");
}
