import {uniformMixedSolidWGSL} from "./uniform-mixed-solid.wgsl";

/** The obstacle discretization is independent of liquid ownership. Reuse the
 * solver's voxel/procedural, terrain and body samplers; never reconstruct a
 * wall from a pressure owner's continued liquid phi. Bindings 18..22 belong
 * to the level-set consumer's solidBoundary resources. */
export const uniformSolidContourWGSL = uniformMixedSolidWGSL(0,undefined,false)
 .replace(/@group\(0\) @binding\((\d+)\)/g,(_,binding:string)=>`@group(0) @binding(${({0:18,1:19,2:20,4:21,5:22} as Record<string,number>)[binding]})`)
 + /* wgsl */ `
fn surfaceHasSolids()->bool{
 return umPresentationEnabled()&&arrayLength(&umSolidScratch)>64u&&umSolidParams.lean.x<0.5;
}
// Positive in the fluid domain. Voxel samples are at cell centres, with a
// half-cell signed distance: their zero is the authored face, independent
// of how deep the liquid is or how far its pressure continuation reaches.
fn surfaceSolidClearance(q:vec3f)->f32{
 if(!surfaceHasSolids()){return 1e6;}
 let cell=vec3i(floor(q));let h=umSolidParams.cellGravity.xyz;
 let r=umBoundaryRecord(cell);var distance=1e6;
 if(r!=0u){distance=umBoundarySampleAt(q,cell,r).w;}
 else {
  let closed=umSolidVoxel(cell);distance=select(1e6,-1e6,closed);
  // Only voxel faces constrain this cell. A constant positive half-cell
  // everywhere would incorrectly cap terrain/body distances and shift their
  // zero when combined with a deeper negative endpoint.
  for(var axis=0u;axis<3u;axis++){for(var side=-1;side<=1;side+=2){
   var neighbour=cell;neighbour[axis]+=side;
   if(umSolidVoxel(neighbour)==closed||umBoundaryRecord(neighbour)!=0u){continue;}
   let reach=(0.5-f32(side)*(q[axis]-f32(cell[axis])-0.5))*h[axis];
   if(closed){distance=max(distance,-reach);}else{distance=min(distance,reach);}
  }}
 }
 if(umSolidHasTerrain()){distance=min(distance,(q.y-umTerrainSurface(q).x)*h.y);}
 let world=umSolidParams.frame.xyz+q*h;
 for(var i=0u;i<umBodyCount();i++){distance=min(distance,umBodyDistance(i,world));}
 return distance;
}
fn surfaceClipValue(cell:vec3i,value:f32)->f32{
 if(!surfaceHasSolids()){return value;}
 return min(value,0.5+surfaceSolidClearance(vec3f(cell)+0.5)/(u.container.y/u.gridInfo.y));
}
// The 5^3 contour samples of window g lie in these eight tiles. The solver's
// cut map includes each tile's one-cell halo, even when the tile runs at 4h.
// A liquid-only sign certificate cannot cull a window crossing an obstacle.
fn surfaceWindowHasSolid(g:vec3i)->bool{
 if(!surfaceHasSolids()){return false;}
 for(var k=0u;k<umPresentationLoopBound();k++){
  let cell=4*(g-vec3i(1)+vec3i(umCorner(k,2u)));
  let tile=umTileKey(cell>>vec3u(2u));
  if(umKeyStored(tile)&&umSolidScratch[umSolidCutMap()+tile]!=0u){return true;}
 }
 let h=umSolidParams.cellGravity.xyz;
 return umAnyBodyNear(umSolidParams.frame.xyz+(vec3f(4*g)+1.5)*h,4.0*length(h));
}
// Intersect two linear half-spaces along an edge, rather than interpolate
// max(phiLiquid,-phiSolid). The latter moves a fixed wall whenever the
// liquid distance changes. a/b are positive outside each constraint.
fn surfaceEdgeInterval(a:f32,b:f32,range:vec2f)->vec2f{
 if((a<=0.0)==(b<=0.0)){return range;}
 let t=clamp(a/(a-b),0.0,1.0);
 return select(vec2f(max(range.x,t),range.y),vec2f(range.x,min(range.y,t)),a<=0.0);
}
`;
