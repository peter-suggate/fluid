/** Closing travel through the entry face of a swept source/receiver AABB. */
export const sparseCM12CoarseFirstPredictionWGSL = /* wgsl */ `
fn coarseFirstApproachTravel(delta:vec3f,sweep:vec3f,extent:f32)->f32{
  var enter=0.0;var leave=1.0;var approach=0.0;
  for(var axis=0u;axis<3u;axis++){
    let distance=abs(delta[axis]);let motion=sweep[axis];
    if(distance>=extent){
      // A touching face still needs positive closing motion. Tangential
      // travel cannot carry a feature through it, nor can a receding source.
      let closing=sign(delta[axis])*motion;
      if(closing<=0.0){return 0.0;}
      let axisEnter=(distance-extent)/closing;
      if(axisEnter>enter||approach==0.0){approach=closing;}
      else if(axisEnter==enter){approach=min(approach,closing);}
      enter=max(enter,axisEnter);
      leave=min(leave,(distance+extent)/closing);
    }else if(abs(motion)>1e-8){
      leave=min(leave,(extent+sign(motion)*delta[axis])/abs(motion));
    }
  }
  // Merely touching at one instant has no swept overlap. For diagonal entry,
  // the last entering face (slowest of tied faces) supplies closing travel.
  return select(0.0,approach,enter<leave);
}
`;
