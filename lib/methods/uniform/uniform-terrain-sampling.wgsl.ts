/** Shared heightfield samples for particle contact and cut-cell geometry. */
export const uniformTerrainSamplingWGSL = /* wgsl */ `
// Terrain columns by lattice column: the box's in a closed scene, the whole
// window's in an open world (UniformOpenWorldTerrain), whose ground runs on
// past the box. A read past the texture takes its edge column.
fn umSolidTerrainHigh()->vec2f{return vec2f(textureDimensions(umSolidTerrain))-vec2f(1);}
fn umSolidTerrainHeight(x:i32,z:i32)->f32{
 let d=vec2i(textureDimensions(umSolidTerrain));let f=32*umFramePage();let world=vec2i(x,z)+f.xz;
 if(umAnalyticTerrain.ground.w>0.5&&(any(world<vec2i(0))||any(world>=d))){
  let at=umAnalyticTerrain.frame.xy+(vec2f(world)+0.5)*umAnalyticTerrain.frame.zw;
  return owHeight(at.x,at.y)-f32(f.y);
 }
 return textureLoad(umSolidTerrain,clamp(world,vec2i(0),d-1),0).x-f32(f.y);
}
// One continuous height and normal for contact and pressure geometry.
// Finite columns live at cell centres. Continue the first/last bilinear patch
// across the outer half-cell to the physical wall, rather than flattening it
// at the first/last centre. A flattened strip traps ballistic particles at
// the uphill wall: gravity has no tangential component there.
// Only the patch index is held to the last valid pair; its fraction may lie
// in [-0.5,1.5]. Singleton axes repeat their one height and have zero slope.
// Outside the finite box the edge height stays constant. Analytic open-world
// terrain keeps sampling its unbounded columns, including its fallback.
fn umTerrainSurface(q:vec3f)->vec3f{
 let offset=32.0*vec2f(umFramePage().xz);let low=-offset;let high=umSolidTerrainHigh()-offset;
 let analytic=umAnalyticTerrain.ground.w>0.5;
 let p=select(clamp(q.xz-vec2f(0.5),low-0.5,high+0.5),q.xz-vec2f(0.5),analytic);
 let a=select(clamp(vec2i(floor(p)),vec2i(low),vec2i(max(low,high-1.0))),vec2i(floor(p)),analytic);
 let f=p-vec2f(a);
 var heights:array<f32,4>;for(var k=0u;k<4u;k++){heights[k]=umSolidTerrainHeight(a.x+i32(k&1u),a.y+i32(k>>1u));}
 let h00=heights[0];let h10=heights[1];let h01=heights[2];let h11=heights[3];
 let height=mix(mix(h00,h10,f.x),mix(h01,h11,f.x),f.y);
 let slope=vec2f(mix(h10-h00,h11-h01,f.y),mix(h01-h00,h11-h10,f.x));
 let interior=((q.xz>=low)&(q.xz<=high+1.0))|vec2<bool>(analytic);
 return vec3f(height,select(vec2f(0),slope,interior));
}
`;
