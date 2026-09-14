import { sparseCM12CoarseFirstPredictionWGSL } from "./sparse-cm12-coarse-first-prediction.wgsl";

/** Linked into the resident shader against its accepted-cell and world-directory ABI. */
export const coarseFirstWGSL = /* wgsl */ `fn coarseFirstEnabled()->bool{return p.coarseFirst.x!=0.0;}
// Integrate one logical-brick intersection in native accepted cells. A logical
// brick has one owner, even when that owner is a macro leaf. Resolve it once;
// each constant donor contributes its exact overlap volume, not h^3 lookups.
fn coarseFirstTileMass(lower:vec3i,extent:vec3i)->f32{
  let owner=compactOwnerCellAt(lower);
  if(owner.x==INVALID||!brickActive(owner.y)){return 0.0;}
  let origin=cm12WorldLeafCoordinate(owner.y)*i32(BRICK_FINE_RESOLUTION);
  let scale=i32(BRICK_FINE_RESOLUTION*brickSpan(owner.y)/owner.z);
  var upper=origin+vec3i(i32(BRICK_FINE_RESOLUTION*brickSpan(owner.y)));
  if(!brickHasUnclippedWorldGeometry(owner.y)){upper=min(upper,vec3i(p.dimensions.xyz));}
  let valid=vec3u((upper-origin+vec3i(scale-1))/scale);
  let range=templateBrickCellRange(owner.y,owner.z);
  let relative=lower-origin;
  var mass=0.0;var z=0;
  loop{if(z>=extent.z){break;}
    let dz=min(extent.z-z,scale-(relative.z+z)%scale);var y=0;
    loop{if(y>=extent.y){break;}
      let dy=min(extent.y-y,scale-(relative.y+y)%scale);var x=0;
      loop{if(x>=extent.x){break;}
        let dx=min(extent.x-x,scale-(relative.x+x)%scale);
        let local=vec3u((relative+vec3i(x,y,z))/scale);
        let offset=local.x+valid.x*(local.y+valid.y*local.z);
        if(offset<range.y){
          let cell=range.x+offset;
          if(cellOpenFraction(cell)<0.999){return -1.0;}
          mass+=f32(dx*dy*dz)*state[destinationDensity()+cell];
        }
        x+=dx;
      }y+=dy;
    }z+=dz;
  }
  return mass;
}
fn coarseFirstDensity(position:vec3f,h:f32)->f32{
  // Activity has not staged the tracer/transport workgroup directory. Query
  // the accepted topology directly: an unstaged TEI window reads zero leaves
  // near its default origin and a stale slot elsewhere, erasing curvature.
  let lower=vec3i(clamp(position-vec3f(0.5*h),cm12WorldFineLower(),
    max(cm12WorldFineLower(),cm12WorldFineUpper()-vec3f(h))));
  let owner=compactOwnerCellAt(lower);
  var scale=1u;var contained=false;
  var direct=false;var donorStrides=vec3u(0u);
  if(owner.x!=INVALID&&brickActive(owner.y)){
    let origin=cm12WorldLeafCoordinate(owner.y)*i32(BRICK_FINE_RESOLUTION);
    let upper=origin+vec3i(i32(BRICK_FINE_RESOLUTION*brickSpan(owner.y)));
    contained=all(lower>=origin)&&all(lower+vec3i(i32(h))<=upper);
    if(contained){scale=BRICK_FINE_RESOLUTION*brickSpan(owner.y)/owner.z;}
    if(contained&&f32(scale)>=h){
      if(cellOpenFraction(owner.x)<0.999){return -1.0;}
      return state[destinationDensity()+owner.x];
    }
    // Resolve a contained leaf once. Its accepted cells are a dense interval;
    // each restriction sample advances from the already resolved first donor.
    // Clipped support that leaves that interval keeps the spatial fallback.
    var validUpper=upper;
    if(!brickHasUnclippedWorldGeometry(owner.y)){
      validUpper=min(validUpper,vec3i(p.dimensions.xyz));
    }
    direct=contained&&all(lower+vec3i(i32(h))<=validUpper);
    donorStrides=vec3u((validUpper-origin+vec3i(i32(scale)-1))/i32(scale));
  }else if(h<=f32(BRICK_FINE_RESOLUTION)){
    return 0.0;
  }
  if(!direct){
    // Partition at logical-brick boundaries so mixed resolutions cannot be
    // skipped by the first donor's stride. Missing tiles are authoritative air.
    // Unlike finest quadrature, each tile resolves its sparse owner just once.
    let edge=i32(BRICK_FINE_RESOLUTION);let limit=i32(h);
    var mass=0.0;var z=0;
    loop{if(z>=limit){break;}
      let dz=min(limit-z,edge-((lower.z+z)%edge+edge)%edge);var y=0;
      loop{if(y>=limit){break;}
        let dy=min(limit-y,edge-((lower.y+y)%edge+edge)%edge);var x=0;
        loop{if(x>=limit){break;}
          let dx=min(limit-x,edge-((lower.x+x)%edge+edge)%edge);
          let tileMass=coarseFirstTileMass(lower+vec3i(x,y,z),vec3i(dx,dy,dz));
          if(tileMass<0.0){return -1.0;}
          mass+=tileMass;x+=dx;
        }y+=dy;
      }z+=dz;
    }
    return mass/(h*h*h);
  }
  // Contained restriction retains its native-cell summation order.
  var sum=0.0;var count=0.0;
  for(var z=0.0;z<h;z+=f32(scale)){for(var y=0.0;y<h;y+=f32(scale)){
    for(var x=0.0;x<h;x+=f32(scale)){
      let local=vec3u(vec3f(x,y,z))/scale;
      let cell=owner.x+local.x+donorStrides.x*(local.y+donorStrides.y*local.z);
        // A solid interface is not liquid curvature. Its independent solid
        // geometry/coupling floor owns this stencil; do not treat it as air.
      if(cellOpenFraction(cell)<0.999){return -1.0;}
      sum+=state[destinationDensity()+cell];
      count+=1.0;
    }
  }}
  return sum/max(count,1.0);
}
fn coarseFirstNormal(position:vec3f,h:f32)->vec3f{
  _=h;
  if(!lsvPhiMetricAt(position)){return vec3f(0.0);}
  let gradient=lsvGradientAt(position);
  let magnitude=length(gradient);
  return select(vec3f(0.0),gradient/max(magnitude,1e-8),magnitude>1e-6);
}
// Receiver-side, immutable accepted-state query. No scatter race, authored
// volume identity, or scene name enters the causal neighbourhood. Swept boxes
// conservatively include the space between samples. A bounded radius makes
// the performance/maximum anticipation reach explicit in the UI.
${sparseCM12CoarseFirstPredictionWGSL}
fn coarseFirstIncomingFloor(brick:u32)->u32{
  let horizon=p.coarseFirst.w;if(horizon<=0.0){return 1u;}
  let coordinate=cm12WorldLeafCoordinate(brick);
  let b=f32(BRICK_FINE_RESOLUTION);
  let center=(vec3f(coordinate)+vec3f(0.5*f32(brickSpan(brick))))*b;
  let receiverRecord=activityRecord(brick);
  var receiverVelocity=vec3f(0.0);
  if((atomicLoad(&activity[receiverRecord+1u])&64u)!=0u){
    receiverVelocity=vec3f(activityF32(receiverRecord+5u),
      activityF32(receiverRecord+6u),activityF32(receiverRecord+7u));
  }
  let radius=i32(p.coarseFirstHistory.x);var required=1u;
  for(var z=-radius;z<=radius;z++){for(var y=-radius;y<=radius;y++){
    for(var x=-radius;x<=radius;x++){
      if(x==0&&y==0&&z==0){continue;}
      let source=cm12WorldOwnerAt(coordinate+vec3i(x,y,z));
      if(source==INVALID||source==brick||!brickActive(source)){continue;}
      let record=activityRecord(source);let reasons=atomicLoad(&activity[record+1u]);
      if((reasons&64u)==0u||(reasons&(1u|256u))==0u){continue;}
      let velocity=vec3f(activityF32(record+5u),activityF32(record+6u),activityF32(record+7u));
      let sweep=horizon*(velocity-receiverVelocity);
      if(length(sweep)<=1.0){continue;}
      let origin=(vec3f(cm12WorldLeafCoordinate(source))+vec3f(0.5*f32(brickSpan(source))))*b;
      let delta=center-origin;
      let extent=0.5*b*f32(brickSpan(brick)+brickSpan(source));
      let approach=coarseFirstApproachTravel(delta,sweep,extent);
      if(approach<=1.0){continue;}
      let gap=length(max(abs(delta)-vec3f(extent),vec3f(0.0)));
      let demand=f32(BRICK_FINE_RESOLUTION)*min(1.0,approach/max(b,gap+b));
      var rung=1u;loop{if(rung>=BRICK_FINE_RESOLUTION||f32(rung)>=demand){break;}rung*=2u;}
      required=max(required,rung);
      if(required==BRICK_FINE_RESOLUTION){return required;}
    }
  }}
  return required;
}

`;
