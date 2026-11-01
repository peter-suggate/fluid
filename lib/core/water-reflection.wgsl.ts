/** Depth-tested scene reflections shared by both water intervals.
 * Uses the retained dry-scene radiance/depth, with environment fallback for
 * offscreen and occluded geometry. No scene-specific reflection artwork.
 */
export function createWaterReflectionWGSL(depthBounds=false): string { return /* wgsl */ `
override screenSpaceReflections:u32=1u;
${depthBounds ? `// Keep the exact repeated-addition DDA values, but classify all crossings
// of an empty 8x8 tile together instead of branching once per crossed pixel.
struct WaterReflectionCrossings { low:vec4f, high:vec4f, after:f32 }
fn waterReflectionCrossings(next:f32,stride:f32)->WaterReflectionCrossings{
  let a=next+stride;let b=a+stride;let c=b+stride;let d=c+stride;
  let e=d+stride;let f=e+stride;let g=f+stride;let h=g+stride;
  return WaterReflectionCrossings(vec4f(next,a,b,c),vec4f(d,e,f,g),h);
}
fn waterReflectionCrossingAt(batch:WaterReflectionCrossings,index:u32)->f32{
  return select(select(batch.low[index&3u],batch.high[index&3u],index>=4u),batch.after,index==8u);
}
fn waterReflectionCrossingCount(batch:WaterReflectionCrossings,end:f32)->u32{
  let low=select(vec4u(0),vec4u(1),batch.low<=vec4f(end));
  let high=select(vec4u(0),vec4u(1),batch.high<=vec4f(end));
  return low.x+low.y+low.z+low.w+high.x+high.y+high.z+high.w;
}
` : ""}

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
  ${depthBounds ? `let useBounds=all(vec2u(dimensions)==waterReflectionDepthDimensions());
  var priorTile=vec2i(-1);var tileMayIntersect=true;` : ""}
  for(var i=0u;i<maxSteps;i+=1u){
    let segmentEnd=min(1.0,min(nextX,nextY));
    let fraction=screenEnd*segmentEnd;
    let hi=mix(bias/startZ,endT/endZ,fraction)/mix(1.0/startZ,1.0/endZ,fraction);
    let uv=(vec2f(f32(pixelX),f32(pixelY))+.5)/dimensions;
    ${depthBounds ? `if(useBounds){
      let tile=vec2i(pixelX,pixelY)/8;
      if(any(tile!=priorTile)){
        priorTile=tile;
        let remainingX=select(pixelX%8+1,8-pixelX%8,stepX>0);
        let remainingY=select(pixelY%8+1,8-pixelY%8,stepY>0);
        let tileEnd=min(1.0,min(nextX+strideX*f32(remainingX-1),nextY+strideY*f32(remainingY-1)));
        let fractionEnd=screenEnd*tileEnd;
        let travelEnd=mix(bias/startZ,endT/endZ,fractionEnd)/mix(1.0/startZ,1.0/endZ,fractionEnd);
        let nearest=offset+direction*clamp(-along,lo,travelEnd);
        let entry=offset+direction*lo;let exit=offset+direction*travelEnd;
        let range=waterReflectionDepthBounds(tile);
        // Use the greatest possible pixel tolerance in this tile. Additional
        // roundoff slack only admits extra work, never rejects a valid hit.
        let tolerance=max(.02*cellSize,.001*range.y)+range.y*cameraTanHalfFov()/dimensions.y*1.414214+max(.0001,range.y*1e-5);
        let low=max(0.0,range.x-tolerance);let high=range.y+tolerance;
        tileMayIntersect=range.x<=range.y&&high*high>=dot(nearest,nearest)&&low*low<=max(dot(entry,entry),dot(exit,exit));
      }
      if(!tileMayIntersect){
        let crossingsX=waterReflectionCrossings(nextX,strideX);
        let crossingsY=waterReflectionCrossings(nextY,strideY);
        let remainingX=u32(select(pixelX%8+1,8-pixelX%8,stepX>0));
        let remainingY=u32(select(pixelY%8+1,8-pixelY%8,stepY>0));
        let skippedEnd=min(1.0,min(waterReflectionCrossingAt(crossingsX,remainingX-1u),waterReflectionCrossingAt(crossingsY,remainingY-1u)));
        if(skippedEnd>=1.0){return vec4f(0);}
        let countX=waterReflectionCrossingCount(crossingsX,skippedEnd);
        let countY=waterReflectionCrossingCount(crossingsY,skippedEnd);
        pixelX+=i32(countX)*stepX;pixelY+=i32(countY)*stepY;
        nextX=waterReflectionCrossingAt(crossingsX,countX);
        nextY=waterReflectionCrossingAt(crossingsY,countY);
        if(pixelX<0||pixelY<0||pixelX>=i32(dimensions.x)||pixelY>=i32(dimensions.y)){return vec4f(0);}
        let skippedFraction=screenEnd*skippedEnd;
        lo=mix(bias/startZ,endT/endZ,skippedFraction)/mix(1.0/startZ,1.0/endZ,skippedFraction);
        continue;
      }
    }` : ""}
    let depth=waterReflectionDepth(uv);
    if(depth<60000.0&&hi>lo){
      let closest=clamp(-along,lo,hi);
      let nearest=offset+direction*closest;
      let entry=offset+direction*lo;let exit=offset+direction*hi;
      let minimumDepthSquared=dot(nearest,nearest);
      let maximumDepthSquared=max(dot(entry,entry),dot(exit,exit));
      // Half-float depth and pixel-centre sampling each have finite precision.
      // Bound the latter by half a diagonal pixel footprint at the receiver.
      // Keep this in radial-depth units: near a tangent ray, converting it
      // directly into the same tolerance on ray travel rejects valid hits.
      let pixelRadius=depth*cameraTanHalfFov()/dimensions.y*1.414214;
      let tolerance=max(.02*cellSize,.001*depth)+pixelRadius;
      // Depth and its tolerated interval are nonnegative. Compare their
      // squares to avoid three square roots at every crossed pixel; only an
      // accepted hit needs the quadratic root below.
      let depthLow=max(0.0,depth-tolerance);let depthHigh=depth+tolerance;
      if(depthHigh*depthHigh>=minimumDepthSquared&&depthLow*depthLow<=maximumDepthSquared){
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
`; }

export const waterReflectionWGSL = createWaterReflectionWGSL();
