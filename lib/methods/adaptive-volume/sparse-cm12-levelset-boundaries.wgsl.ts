/** Scalar boundary conditions share the accepted face field and old phi. No
 * destination-phi reads or additional whole-domain dispatch are required. */
export const sparseCM12LevelSetBoundariesWGSL = /* wgsl */ `
// Sweep exterior air into the phi field by the actual final MAC displacement
// of a released one-sided world face. Project the vertex onto each physical
// domain plane instead of searching only its adjacent cells: a large-CFL gap
// can cross more than one wall cell, while the applicable tangential patch is
// still found in constant work through the accepted owner directory.
fn cm12ReleasedWallPhi(positionFine:vec3f)->vec2f{
  var carved=-3.402823466e38;var hasRelease=false;
  let epsilon=max(1e-4,8.0*1.1920928955078125e-7
    *max(1.0,max(abs(positionFine.x),max(abs(positionFine.y),abs(positionFine.z)))));
  let gravityWeight=length(p.acceleration.xyz);
  for(var face=0u;face<6u;face+=1u){
    let axis=face/2u;let upper=(face&1u)!=0u;
    let boundary=select(0.0,f32(p.dimensions[axis]),upper);
    let expectedInward=select(1.0,-1.0,upper);
    // Pressure uses the same gravity-normal predicate, so other planes cannot
    // contain an applicable separating row for this frame.
    let canRelease=gravityWeight>1e-6&&expectedInward*p.acceleration[axis]>0.5*gravityWeight;
    if(!canRelease&&p.frame.w<=0.5){continue;}
    // Four signs select every cell incident to a tangential edge or corner.
    let tangent0=(axis+1u)%3u;let tangent1=(axis+2u)%3u;
    for(var quadrant=0u;quadrant<4u;quadrant+=1u){
      var probe=positionFine;
      probe[tangent0]+=epsilon*select(-1.0,1.0,(quadrant&1u)!=0u);
      probe[tangent1]+=epsilon*select(-1.0,1.0,(quadrant&2u)!=0u);
      probe[axis]=boundary+epsilon*expectedInward;
      let owner=compactOwnerCellAt(vec3i(floor(probe)));
      if(owner.x==INVALID){continue;}let cell=owner.x;
      let incidences=cnxCellIncidenceRangeUnchecked(cell);
      for(var at=incidences.x;at<incidences.y;at+=1u){
        let rowOrdinal=cnxIncidenceRowOrdinalUnchecked(at);
        let row=cnxStableRowUnchecked(rowOrdinal);
        if(rowKind(row)!=3u||rowAxis(row)!=axis){continue;}
        let open=rowOpenFraction(row);
        let released=canRelease&&rowSeparatingFromClosedWorld(row);
        // An explicit open DOMAIN plane can supply ambient air. An internal
        // sparse allocation edge cannot. Authored liquid sources win.
        let ambient=p.frame.w>0.5&&open>1e-8&&sparseCM12InflowFaceCoverage(row)==0.0;
        if(!released&&!ambient){continue;}
        let range=cnxRowTermRangeByOrdinalUnchecked(rowOrdinal);
        if(range.y-range.x!=1u||cnxRowTermCellUnchecked(range.x)!=cell){continue;}
        let coefficient=cnxRowTermCoefficientUnchecked(range.x);
        let inward=select(-1.0,1.0,coefficient>=0.0);
        let center=rowCenter(row);
        if(inward!=expectedInward||abs(center[axis]-boundary)>epsilon){continue;}
        var speed=airBoundaryFaceVelocity(row)-rowSolidVelocity(row);
        if(ambient){speed=(airBoundaryFaceVelocity(row)-(1.0-open)*rowSolidVelocity(row))/open;}
        let away=inward*speed;
        if(away<=1e-6){continue;}
        let widths=cellWidths(cell);var footprint=true;
        for(var tangent=0u;tangent<3u;tangent+=1u){if(tangent!=axis){
          footprint=footprint
            &&abs(positionFine[tangent]-center[tangent])<=0.5*widths[tangent]+epsilon;}}
        if(!footprint){continue;}
        let interiorDistance=inward*(positionFine[axis]-center[axis]);
        carved=max(carved,p.frame.x*away-interiorDistance);hasRelease=true;
      }
    }
  }
  return vec2f(carved,select(0.0,1.0,hasRelease));
}
// Only closed physical domain planes continue liquid contact. Gather all
// incident planes before taking one diagonal interior trace; no axis order
// and no already-written destination phi can affect the answer.
fn cm12ClosedWallPhi(position:vec3f)->vec2f{
  if(p.frame.w<=0.5||p.frame.x<=0.0){return vec2f(0.0);}
  let epsilon=max(1e-4,8.0*1.1920928955078125e-7*max(1.0,max(abs(position.x),max(abs(position.y),abs(position.z)))));
  var closed=0u;var separating=0u;
  for(var face=0u;face<6u;face+=1u){let axis=face/2u;let upper=(face&1u)!=0u;
    let boundary=select(0.0,f32(p.dimensions[axis]),upper);
    if(abs(position[axis]-boundary)>epsilon){continue;}
    let inward=select(1.0,-1.0,upper);let tangent0=(axis+1u)%3u;let tangent1=(axis+2u)%3u;
    for(var quadrant=0u;quadrant<4u;quadrant+=1u){var probe=position;
      probe[axis]=boundary+epsilon*inward;
      probe[tangent0]+=epsilon*select(-1.0,1.0,(quadrant&1u)!=0u);
      probe[tangent1]+=epsilon*select(-1.0,1.0,(quadrant&2u)!=0u);
      let owner=compactOwnerCellAt(vec3i(floor(probe)));if(owner.x==INVALID||cellOpenVolume(owner.x)<=1e-8){continue;}
      let incidence=cnxCellIncidenceRangeUnchecked(owner.x);
      for(var at=incidence.x;at<incidence.y;at+=1u){let ordinal=cnxIncidenceRowOrdinalUnchecked(at);
        let row=cnxStableRowUnchecked(ordinal);let range=cnxRowTermRangeByOrdinalUnchecked(ordinal);
        if(rowKind(row)!=3u||rowAxis(row)!=axis||range.y-range.x!=1u||rowOpenFraction(row)>1e-8
          ||abs(rowCenter(row)[axis]-boundary)>epsilon){continue;}
        closed|=1u<<face;if(rowSeparatingFromClosedWorld(row)){separating|=1u<<face;}
      }
    }
  }
  closed&=~separating;if(closed==0u){return vec2f(0.0);}
  var interior=position;
  for(var axis=0u;axis<3u;axis+=1u){
    if((closed&(1u<<(2u*axis)))!=0u){interior[axis]+=1.0;}
    if((closed&(2u<<(2u*axis)))!=0u){interior[axis]-=1.0;}}
  if(acceptedPointInsideSolid(interior)){return vec2f(0.0);}
  let first=lsvAcceptedVelocitySample(interior);
  let midpoint=lsvClipCharacteristic(interior,interior-0.5*p.frame.x*first.xyz);
  let second=lsvAcceptedVelocitySample(midpoint);
  let departure=lsvClipCharacteristic(interior,interior-p.frame.x*second.xyz);
  let sample=lsvSampleAtSlot(lsvAcceptedSlot(),departure);
  return vec2f(sample.phi,select(0.0,1.0,first.w>0.0&&second.w>0.0&&sample.valid&&sample.phi<0.0));
}
`;
