/** Geometry-only thin-feature witness. 0 = clear, 1 = thin, 2 = unresolved.
 * The source is the accepted adaptive trilinear phi, never density or the
 * candidate restriction. Crossings on all twelve edges cover off-centre detail.
 */
export function createLevelSetThinFeaturesWGSL(): string { return /* wgsl */ `
fn lsvThinGradient(stencil:LsvCellStencil,position:vec3f)->vec3f{
  let t=clamp((position-stencil.lower)/stencil.widths,vec3f(0.0),vec3f(1.0));
  var gradient=vec3f(0.0);
  for(var corner=0u;corner<8u;corner+=1u){
    let high=vec3u(corner&1u,(corner>>1u)&1u,(corner>>2u)&1u);
    let w=select(vec3f(1.0)-t,t,high!=vec3u(0u));
    let sign=select(vec3f(-1.0),vec3f(1.0),high!=vec3u(0u));
    gradient+=stencil.phi[corner]*sign*vec3f(w.y*w.z,w.x*w.z,w.x*w.y)/stencil.widths;
  }
  return gradient;
}
fn lsvThinSample(stencil:ptr<function,LsvCellStencil>,position:vec3f)->LsvPhiSample{
  // Successive probes usually remain in one neighbour. Cache its eight
  // corners instead of repeating the adaptive owner lookup at every step.
  if(!lsvStencilContains(*stencil,position)){*stencil=lsvStencilAtPosition(position);}
  return lsvStencilSampleAt(*stencil,position);
}
// Symmetric differences avoid choosing an arbitrary one-sided normal at a
// lattice edge. Only short-chord candidates pay for these extra samples.
fn lsvThinCrossingNormal(position:vec3f)->vec4f{
  var gradient=vec3f(0.0);
  let centre=lsvSampleAt(position);if(!centre.valid){return vec4f(0.0);}
  for(var axis=0u;axis<3u;axis+=1u){
    var offset=vec3f(0.0);offset[axis]=0.01;
    let lo=lsvSampleAt(position-offset);let hi=lsvSampleAt(position+offset);
    if(!lo.valid&&!hi.valid){return vec4f(0.0);}
    // Domain boundaries permit a one-sided derivative, e.g. an extruded
    // sheet at the front/back wall of a 2D scene in the 3D solver.
    gradient[axis]=select(2.0*(centre.phi-lo.phi),
      select(2.0*(hi.phi-centre.phi),hi.phi-lo.phi,lo.valid),hi.valid);
  }
  let magnitude=length(gradient);
  if(magnitude<1e-7){return vec4f(0.0);}
  return vec4f(gradient/magnitude,1.0);
}
fn lsvThinFeature(stencil:LsvCellStencil,width:f32)->u32{
  if(width<=0.0){return 0u;}if(!stencil.resolved){return 2u;}
  var minimum=1e30;var maximum=-1e30;
  for(var corner=0u;corner<8u;corner+=1u){
    if(stencil.support[corner]==0u||stencil.support[corner]==0xffffffffu){return 2u;}
    minimum=min(minimum,stencil.phi[corner]);maximum=max(maximum,stencil.phi[corner]);}
  if(minimum>=0.0||maximum<0.0){return 0u;}
  var unresolved=false;let step=min(0.25,width/8.0);let epsilon=min(0.01,step*0.1);
  for(var axis=0u;axis<3u;axis+=1u){for(var corner=0u;corner<8u;corner+=1u){
    if((corner&(1u<<axis))!=0u){continue;}
    let other=corner|(1u<<axis);let a=stencil.phi[corner];let b=stencil.phi[other];
    if((a<0.0)==(b<0.0)){continue;}
    var point=stencil.lower+stencil.widths*vec3f(f32(corner&1u),f32((corner>>1u)&1u),f32((corner>>2u)&1u));
    point[axis]+=stencil.widths[axis]*clamp(a/(a-b),0.0,1.0);
    let gradient=lsvThinGradient(stencil,point);let magnitude=length(gradient);
    if(magnitude<1e-6){unresolved=true;continue;}
    let normal=gradient/magnitude;
    // Also trace the crossing edge: at a lattice crease the one-sided
    // gradient can point outside both adjacent phases of a tiny droplet.
    for(var side=0u;side<4u;side+=1u){
      // An axis-aligned normal already traced this exact line in both directions.
      if(side>=2u&&abs(normal[axis])>1.0-1e-7){continue;}
      var direction=select(-normal,normal,(side&1u)==1u);
      if(side>=2u){direction=vec3f(0.0);direction[axis]=select(-1.0,1.0,(side&1u)==1u);}

      var probeStencil=stencil;
      let initial=lsvThinSample(&probeStencil,point+epsilon*direction);
      if(!initial.valid||abs(initial.phi)<1e-7){continue;}
      var previousPhi=initial.phi;var previousTravel=epsilon;
      // A second crossing bounds a liquid filament/sheet or an air gap. A
      // phase-only certificate can exclude a crossing, but is never a normal.
      for(var probe=1u;probe<=32u;probe+=1u){
        let travel=min(width,f32(probe)*step);
        let sample=lsvThinSample(&probeStencil,point+travel*direction);
        // Missing support outside this cell is not a thin-feature witness.
        // The independent restriction proof still validates candidate samples.
        if(!sample.valid){break;}
        if((sample.phi<0.0)!=(initial.phi<0.0)){
          let hitTravel=mix(previousTravel,travel,clamp(previousPhi/(previousPhi-sample.phi),0.0,1.0));
          let firstNormal=lsvThinCrossingNormal(point);
          let secondNormal=lsvThinCrossingNormal(point+hitTravel*direction);
          // A short tangential chord of a broad curved surface is not a sheet.
          if(firstNormal.w>0.0&&secondNormal.w>0.0
            &&dot(firstNormal.xyz,secondNormal.xyz)<-0.25){return 1u;}
          break;
        }
        previousPhi=sample.phi;previousTravel=travel;
        if(travel>=width){break;}
      }
    }
  }}
  return select(0u,2u,unresolved);
}
fn lsvThinFeatureCell(cell:u32,width:f32)->u32{
  return lsvThinFeature(lsvCellStencil(cell),width);
}
`; }
