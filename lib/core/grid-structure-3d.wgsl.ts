/** Topology-only grid traversal. Uses the overlay's camera, ownership and
 * stage-layout helpers, never scalar/velocity fields or level-set sampling. */
export const gridStructure3DWGSL = /* wgsl */ `
@fragment fn fragmentGrid3D(input:VertexOutput)->@location(0) vec4f {
  let opacity=layers.selection.z;
  if(opacity<=0.0 || u.gridInfo.w<=0.5){discard;}
  udrInit();
  let eye=u.cameraPosition.xyz;
  let forward=normalize(u.cameraTarget.xyz-eye);
  var right=cross(forward,vec3f(0,1,0));
  if(length(right)<1e-5){right=vec3f(1,0,0);}
  right=normalize(right);
  let up=normalize(cross(right,forward));
  let ndc=input.uv*2.0-1.0;
  let ray=normalize(forward+right*ndc.x*u.viewport.x/max(u.viewport.y,1.0)*cameraTanHalfFov()+up*ndc.y*cameraTanHalfFov());
  let frame=overlayGridFrame();
  let cellSize=frame.size/vec3f(frame.dimensions);
  let ro=(eye-frame.boundsMin)/cellSize;
  let rd=ray/cellSize;
  let inv=1.0/select(select(vec3f(-1e-20),vec3f(1e-20),rd>=vec3f(0)),rd,abs(rd)>vec3f(1e-20));
  let near=min(-ro*inv,(vec3f(frame.dimensions)-ro)*inv);
  let far=max(-ro*inv,(vec3f(frame.dimensions)-ro)*inv);
  var t=max(max(near.x,max(near.y,near.z)),0.0);
  let end=min(far.x,min(far.y,far.z));
  if(end<=t){discard;}
  let mixed=sliceLsvP.global.x==2u && umPresentationEnabled();
  let adaptive=u.debug.z>0.5;
  let tall=u.gridInfo.w>1.5 && u.gridInfo.w<2.5;
  // If even the widest possible owner is already below the line fade at
  // the box entrance, every later owner is too small as well.
  let entryPixel=max(t*2.0*cameraTanHalfFov()/max(u.viewport.y,1.0),1e-6);
  let widest=select(1.0,4.0,mixed)*min(cellSize.x,min(cellSize.y,cellSize.z));
  if(!adaptive && !tall && widest/entryPixel<=1.8){discard;}
  let stage=select(0xffffffffu,umStageBase(),mixed);
  let epsilon=1e-4/max(max(abs(rd.x),abs(rd.y)),abs(rd.z));
  var result=vec4f(0);
  // Jump directly to the next represented cell, or across an entire absent
  // 32h page. Coarse owners are visited once rather than sampled at h steps.
  let budget=u32(frame.dimensions.x+frame.dimensions.y+frame.dimensions.z)+3u;
  for(var visit=0u;visit<budget;visit+=1u){
    if(t>=end || result.a>0.97){break;}
    let q=clamp(vec3i(floor(ro+rd*(t+epsilon))),vec3i(0),frame.dimensions-vec3i(1));
    let cell=frame.minimumFine+q;
    var lower=vec3f(q);var upper=lower+vec3f(1);
    var present=true;
    var width=1u;
    if(mixed){
      let owner=umOwnerAt(cell);
      present=owner.width!=0u;
      if(present){width=owner.width;lower=vec3f(vec3i(umOrigin(owner))-frame.minimumFine);upper=lower+vec3f(f32(width));}
      else{lower=vec3f((cell/32)*32-frame.minimumFine);upper=lower+vec3f(32);}
    }else if(adaptive || tall){
      let represented=representedCell(cell,frame.dimensions,frame.minimumFine,frame.boundsMin,frame.size,adaptive,tall);
      lower=(represented.lower-frame.boundsMin)/cellSize;upper=(represented.upper-frame.boundsMin)/cellSize;
    }
    let next=(select(lower,upper,rd>=vec3f(0))-ro)*inv;
    let exit=min(end,min(next.x,min(next.y,next.z)));
    let pixel=max(t*2.0*cameraTanHalfFov()/max(u.viewport.y,1.0),1e-6);
    let pixelCells=vec3f(pixel)/cellSize;
    let extent=(upper-lower)*cellSize;
    let pixelsPerCell=min(extent.x,min(extent.y,extent.z))/pixel;
    let fade=smoothstep(1.8,4.2,pixelsPerCell);
    if(present && fade>0.0){
      let entryPoint=ro+rd*t;
      let exitPoint=ro+rd*exit;
      let edge=min(volumeBoxEdgePixels(entryPoint,lower,upper,pixelCells),volumeBoxEdgePixels(exitPoint,lower,upper,pixelCells));
      let halfWidth=min(1.5,max(pixelsPerCell*0.2,0.32));
      var alpha=gridLinePaint(edge,halfWidth)*fade;
      var color=mix(vec3f(0.03,0.08,0.09),vec3f(2.2,2.6,2.5),smoothstep(0.45,0.95,alpha));
      if(mixed){
        let tile=cell/4;
        let previous=umStagePreviousWidth(stage,tile);
        let transported=umStageTileWidth(stage,UM_STAGE_TRANSPORT,tile);
        let changed=(previous!=0u && previous!=width)||(transported!=0u && transported!=width);
        if(changed){
          let stripe=1.0-smoothstep(1.0,2.0,abs(fract((input.position.x+input.position.y)/8.0)-0.5)*8.0);
          let hatch=0.35*stripe*fade;
          color=mix(color,select(vec3f(3.0,1.35,0.12),sceneColor(LP_coarsened),previous!=0u && width>previous),hatch/max(alpha+hatch,1e-6));
          alpha=max(alpha,hatch);
        }
        // Only near a tile face can an edge be a resolution seam. Do not
        // fetch six neighbouring owners for every sample through its interior.
        if(alpha>0.0){
          let local=entryPoint+vec3f(frame.minimumFine)-vec3f(tile*4);
          for(var axis=0u;axis<3u;axis+=1u){
            let side=select(-1,1,local[axis]>2.0);
            let distance=min(abs(local[axis]),abs(4.0-local[axis]))/pixelCells[axis];
            if(distance>2.5){continue;}
            var neighbor=tile;neighbor[axis]+=side;
            if(umTilePresent(neighbor) && umTileWidth(umTileAt(vec3u(neighbor)))!=width){color=vec3f(3.0,1.35,0.12);}
          }
        }
      }
      if(alpha>0.0){
        let bounds=RepresentedCell(frame.boundsMin+lower*cellSize,frame.boundsMin+upper*cellSize);
        if(!gridBodySample(bounds).occupied){
          let contribution=(1.0-result.a)*alpha*opacity*0.35;
          result+=vec4f(color*contribution,contribution);
        }
      }
    }
    t=max(exit,t+epsilon);
  }
  if(result.a<=0.001){discard;}
  return vec4f(displayColor(result.rgb/result.a),result.a);
}
`;
