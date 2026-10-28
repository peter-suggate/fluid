/** Depth-tested scene reflections shared by both water intervals.
 * Uses the retained dry-scene radiance/depth, with environment fallback for
 * offscreen and occluded geometry. No scene-specific reflection artwork.
 */
export const waterReflectionWGSL = /* wgsl */ `
override screenSpaceReflections:u32=1u;
fn waterReflectionHit(origin:vec3f,direction:vec3f,cellSize:f32)->vec4f{
  let eye=u.cameraPosition.xyz;
  let forward=normalize(u.cameraTarget.xyz-eye);
  let reach=4.0*max(max(u.container.x,u.container.z),length(origin-eye));
  let bias=max(.002,.025*cellSize);
  let start=origin+direction*bias;
  let startZ=dot(start-eye,forward);
  if(startZ<=.001){return vec4f(0);}
  var endT=reach;
  let directionZ=dot(direction,forward);
  if(directionZ<0.0){endT=min(endT,(.001-dot(origin-eye,forward))/directionZ);}
  let end=origin+direction*endT;
  let endZ=max(.001,dot(end-eye,forward));
  let startUV=project(start);let delta=project(end)-startUV;
  if(any(startUV<vec2f(0))||any(startUV>vec2f(1))){return vec4f(0);}
  var screenEnd=1.0;
  for(var axis=0u;axis<2u;axis+=1u){
    if(delta[axis]>0.0){screenEnd=min(screenEnd,(1.0-startUV[axis])/delta[axis]);}
    else if(delta[axis]<0.0){screenEnd=min(screenEnd,-startUV[axis]/delta[axis]);}
  }
  // Walk actual pixel boundaries, including both cells crossed by a diagonal.
  // A fixed major-axis stride can skip isolated leaves even at one pixel/step.
  let dimensions=max(u.viewport.xy,vec2f(1));
  let pixelStart=startUV*dimensions;
  let pixelDelta=delta*screenEnd*dimensions;
  let stepX=select(-1,1,pixelDelta.x>=0.0);let stepY=select(-1,1,pixelDelta.y>=0.0);
  var pixelX=clamp(i32(floor(pixelStart.x)),0,i32(dimensions.x)-1);
  var pixelY=clamp(i32(floor(pixelStart.y)),0,i32(dimensions.y)-1);
  var nextX=1e20;var nextY=1e20;var strideX=1e20;var strideY=1e20;
  if(abs(pixelDelta.x)>1e-8){
    nextX=max(0.0,(f32(pixelX)+select(0.0,1.0,stepX>0)-pixelStart.x)/pixelDelta.x);
    strideX=1.0/abs(pixelDelta.x);
  }
  if(abs(pixelDelta.y)>1e-8){
    nextY=max(0.0,(f32(pixelY)+select(0.0,1.0,stepY>0)-pixelStart.y)/pixelDelta.y);
    strideY=1.0/abs(pixelDelta.y);
  }
  // Cover diagonals of device-valid 16K targets as well as supersampled views.
  // Ordinary images still pay only their actual clipped pixel extent.
  let maxSteps=u32(clamp(ceil(abs(pixelDelta.x))+ceil(abs(pixelDelta.y))+3.0,1.0,32768.0));
  let offset=origin-eye;let along=dot(offset,direction);
  var lo=bias;
  for(var i=0u;i<maxSteps;i+=1u){
    let segmentEnd=min(1.0,min(nextX,nextY));
    let fraction=screenEnd*segmentEnd;
    let hi=mix(bias/startZ,endT/endZ,fraction)/mix(1.0/startZ,1.0/endZ,fraction);
    let uv=(vec2f(f32(pixelX),f32(pixelY))+.5)/dimensions;
    let depth=waterReflectionDepth(uv);
    if(depth<60000.0&&hi>lo){
      let closest=clamp(-along,lo,hi);
      let minimumDepth=length(offset+direction*closest);
      let maximumDepth=max(length(offset+direction*lo),length(offset+direction*hi));
      // Half-float depth and pixel-centre sampling each have finite precision.
      // Bound the latter by half a diagonal pixel footprint at the receiver.
      // Keep this in radial-depth units: near a tangent ray, converting it
      // directly into the same tolerance on ray travel rejects valid hits.
      let pixelRadius=depth*cameraTanHalfFov()/dimensions.y*1.414214;
      let tolerance=max(.02*cellSize,.001*depth)+pixelRadius;
      if(depth>=minimumDepth-tolerance&&depth<=maximumDepth+tolerance){
        let discriminant=along*along-dot(offset,offset)+depth*depth;
        var travel=closest;
        if(discriminant>=0.0){
          let root=sqrt(discriminant);let near=-along-root;let far=-along+root;
          let nearError=abs(near-clamp(near,lo,hi));let farError=abs(far-clamp(far,lo,hi));
          travel=clamp(select(far,near,nearError<=farError),lo,hi);
        }
        let edge=min(min(uv.x,uv.y),min(1.0-uv.x,1.0-uv.y));
        return vec4f(uv,travel,smoothstep(0.0,.055,edge));
      }
    }
    if(segmentEnd>=1.0){break;}
    // Advance both dimensions at a corner, without sampling a zero-area cell.
    let crossX=nextX<=segmentEnd;
    let crossY=nextY<=segmentEnd;
    if(crossX){pixelX+=stepX;nextX+=strideX;}
    if(crossY){pixelY+=stepY;nextY+=strideY;}
    if(pixelX<0||pixelY<0||pixelX>=i32(dimensions.x)||pixelY>=i32(dimensions.y)){break;}
    lo=hi;
  }
  return vec4f(0);
}
fn waterReflectionSample(origin:vec3f,direction:vec3f,cellSize:f32)->vec4f{
  let environment=waterEnvironmentRadiance(direction);
  if(screenSpaceReflections==0u){return vec4f(environment,0); }
  let hit=waterReflectionHit(origin,direction,cellSize);
  if(hit.w<=0.0){return vec4f(environment,0);}
  return vec4f(mix(environment,waterReflectionColor(hit.xy),hit.w),hit.z);
}
`;
