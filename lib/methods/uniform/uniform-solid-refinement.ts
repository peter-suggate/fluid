/** Exact static geometry test for the spatial census, independent of heat.
 * Requires solidRecord, UM_TILES and the topology accessors. Unknown records
 * and body contact never certify. Whole capacities and uniform face patches
 * mean the 4h geometric restriction loses no sub-cell occupancy information.
 * Static CM11a walls use a dual-cell face weight of 1/2, at both resolutions;
 * it is exact only when all sixteen native patches agree. */
export function uniformSolidRefinementWGSL(recordCount:number):string{return /* wgsl */`
// Static geometry exactly representable on the 4h lattice has no sub-tile
// capacity or face-area defect. Do not refine a whole wall collar merely
// because a neighbouring coarse cell is solid. Fractions and moving-body
// contact still require the native geometry, independent of update heat.
fn umCoarseSolidFaceExact(t:u32,axis:u32)->bool{
 let face=solidRecord[t][axis+1u];if(face!=0.0&&face!=0.5&&face!=1.0){return false;}
 // Six four-bit patch codes per exact-float word; compare whole words
 // instead of decoding sixteen patches on every census. Axis boundaries
 // split words 2 and 5, so mask only the nibbles owned by this face.
 let expected=(u32(8.0*face)+1u)*0x111111u;
 let first=vec4u(solidRecord[${recordCount}u+UM_TILES+3u*t+1u]);
 let second=vec4u(solidRecord[${recordCount}u+UM_TILES+3u*t+2u]);
 switch(axis){
  case 0u:{return first.x==expected&&first.y==expected&&(first.z&0xffffu)==(expected&0xffffu);}
  case 1u:{return (first.z>>16u)==(expected&0xffu)&&first.w==expected&&second.x==expected&&(second.y&0xffu)==(expected&0xffu);}
  default:{return (second.y>>8u)==(expected&0xffffu)&&second.z==expected&&second.w==expected;}
 }
}
// A closed page has no tile record. The adjacent stored tile owns all
// sixteen low-face apertures, so absence of a neighbour is not absence of
// geometric evidence. Unknown/air pages still cannot certify a wall.
fn umCoarseSolidLowFaceExact(t:u32,axis:u32)->bool{
 let row=${recordCount}u+23u*UM_TILES+UM_TILES/64u+1u+61u*t+49u+4u*axis;
 let face=solidRecord[row].x;if(face!=0.0&&face!=0.5&&face!=1.0){return false;}
 for(var i=0u;i<4u;i++){if(any(solidRecord[row+i]!=vec4f(face))){return false;}}
 return true;
}
fn umSolidGeometryExact(t:u32)->bool{
 let capacity=solidRecord[t].x;if(capacity!=0.0&&capacity!=1.0){return false;}
 let bodyRow=${recordCount}u+20u*UM_TILES+2u*t+1u;
 if(solidRecord[bodyRow].z>0.5){return false;}
 let c=vec3i(umTileCoord(t));
 for(var a=0u;a<3u;a++){
  var d=vec3i(0);d[a]=1;let n=umTileKey(c-d);
  if(!umCoarseSolidFaceExact(t,a)){return false;}
  if(!umKeyStored(n)){
   if(umTileKind(c-d)!=UM_SOLID||!umCoarseSolidLowFaceExact(t,a)){return false;}
   continue;
  }
  if(!umCoarseSolidFaceExact(n,a)){return false;}
  if(solidRecord[${recordCount}u+20u*UM_TILES+2u*n+1u].z>0.5){return false;}
 }
 return true;
}
`;
}
