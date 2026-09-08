/** Native-density reconstruction from 57b6ae39 (7 September 2026).
 * Kept separate from the opt-in retained/current-map surface experiments. */
export const NATIVE_PRESENTATION_COARSE_COLUMN_PHI_WGSL = /* wgsl */ `
fn presentationCoarseColumnPhi(coarse:vec3i,cellScale:u32,
 cacheFirst:vec3i,cacheDimensions:vec3u,cacheFits:bool,densityOffset:u32)->f32{
  let cellsY=i32(p.dimensions.y/cellScale);
  var rho:array<f32,5>;
  for(var at=0u;at<5u;at+=1u){
    let q=coarse+vec3i(0,i32(at)-2,0);
    let canonical=presentationCanonicalCoarseCoordinate(q,cellScale,cm12PresentationBrick);
    let continued=canonical.y!=q.y;
    if(continued&&q.y>=cellsY){rho[at]=0.0;}
    else if(continued&&q.y<0){
      // Continue a represented floor film below the floor; a dry floor below
      // a detached body remains air and must not gain an artificial bridge.
      let floorDensity=presentationStencilDensityAt(vec3i(q.x,0,q.z),cellScale,
        cacheFirst,cacheDimensions,cacheFits,densityOffset);
      rho[at]=select(0.0,1.0,floorDensity>1e-6);
    }else{
      rho[at]=presentationStencilDensityAt(q,cellScale,
        cacheFirst,cacheDimensions,cacheFits,densityOffset);
    }
  }
  // The upper endpoint can belong to a detached liquid body beyond an air
  // gap. It is not volume in this lower interface bracket. A running minimum
  // removes that unrelated tail continuously, without a validity threshold.
  rho[4]=min(rho[4],rho[3]);
  return f32(cellScale)*presentationResolvedColumnPhi(rho);
}
`;

export const NATIVE_PRESENTATION_INTERPOLATED_VOLUME_PHI_WGSL = /* wgsl */ `
fn presentationInterpolatedVolumePhi(q:vec3i,cellScale:u32,
 cacheFirst:vec3i,cacheDimensions:vec3u,cacheFits:bool,densityOffset:u32)->f32{
  let position=(vec3f(q)+vec3f(0.5))/f32(cellScale)-vec3f(0.5);
  let lower=vec3i(floor(position));let t=fract(position);var phi=0.0;
  for(var dz=0;dz<2;dz+=1){for(var dy=0;dy<2;dy+=1){for(var dx=0;dx<2;dx+=1){
    let offset=vec3i(dx,dy,dz);
    let value=presentationCoarseColumnPhi(lower+offset,cellScale,
      cacheFirst,cacheDimensions,cacheFits,densityOffset);
    let weight=select(1.0-t.x,t.x,dx==1)*select(1.0-t.y,t.y,dy==1)
      *select(1.0-t.z,t.z,dz==1);
    phi+=weight*value;
  }}}
  return phi;
}
`;

export const NATIVE_SURFACE_PROOF_VIRTUAL_COLUMN_PHI_WGSL = /* wgsl */ `
fn surfaceProofVirtualColumnPhi(coarse:vec3i,factor:u32)->f32{
  let baseY=cm12PresentationBrickOrigin.y/i32(factor);
  let cellsY=i32(p.dimensions.y/factor);var rho:array<f32,5>;
  for(var at=0u;at<5u;at+=1u){
    let q=coarse+vec3i(0,i32(at)-2,0);let worldY=baseY+q.y;
    let world=cm12PresentationBrickOrigin/i32(factor)+q;
    let canonical=presentationCanonicalCoarseCoordinate(world,factor,cm12PresentationBrick);
    let continued=canonical.y!=worldY;
    if(continued&&worldY>=cellsY){rho[at]=0.0;}
    else if(continued&&worldY<0){
      let floorDensity=surfaceProofDensityAt(vec3i(q.x,-baseY,q.z));
      rho[at]=select(0.0,1.0,floorDensity>1e-6);
    }else{rho[at]=surfaceProofDensityAt(q);}
  }
  rho[4]=min(rho[4],rho[3]);
  return f32(factor)*presentationResolvedColumnPhi(rho);
}
`;

export const NATIVE_SURFACE_PROOF_VIRTUAL_VOLUME_PHI_WGSL = /* wgsl */ `
fn surfaceProofVirtualVolumePhi(local:vec3i,factor:u32)->f32{
  let position=(vec3f(local)+vec3f(0.5))/f32(factor)-vec3f(0.5);
  let lower=vec3i(floor(position));let t=fract(position);var phi=0.0;
  for(var dz=0;dz<2;dz+=1){for(var dy=0;dy<2;dy+=1){for(var dx=0;dx<2;dx+=1){
    let weight=select(1.0-t.x,t.x,dx==1)*select(1.0-t.y,t.y,dy==1)
      *select(1.0-t.z,t.z,dz==1);
    phi+=weight*surfaceProofVirtualColumnPhi(lower+vec3i(dx,dy,dz),factor);
  }}}
  return phi;
}
`;
