/**
 * Binding-free resident adapters for transitional CM12 interface geometry.
 * Requires geometricInterfaceWGSL and the resident cell/row/state accessors.
 * All lengths here are finest-lattice units. Convert to metres at publication.
 * This observes CM12 density; it never writes a volume authority or repairs it.
 */
export const geometricInterfaceResidentWGSL = /* wgsl */ `
struct GeometricResidentInterface {
  plane: GeometricInterfacePlane,
  valid: u32,
}

fn geometricResidentFill(cell:u32,densityOffset:u32)->f32{
  return state[densityOffset+cell]/max(cellOpenFraction(cell),1e-8);
}

// Physical tangential overlap, independent of coarse-port row coefficients.
// Only opposite-side terms are neighbours: fine cells on the same side of a
// mixed row must not masquerade as normal-direction samples.
fn geometricResidentOverlap(cell:u32,other:u32,axis:u32)->f32{
  let a=cellCenter(cell);let aw=0.5*cellWidths(cell);
  let b=cellCenter(other);let bw=0.5*cellWidths(other);
  let overlap=max(vec3f(0.0),min(a+aw,b+bw)-max(a-aw,b-bw));
  return overlap[(axis+1u)%3u]*overlap[(axis+2u)%3u];
}

fn geometricResidentInterface(cell:u32,densityOffset:u32)->GeometricResidentInterface{
  let widths=cellWidths(cell);
  let fill=geometricResidentFill(cell,densityOffset);
  var result:GeometricResidentInterface;
  result.plane=GeometricInterfacePlane(vec3f(0.0,1.0,0.0),0.0);
  result.valid=0u;
  // Scalar aperture is not a cut-solid polyhedron. Preserve the established
  // solid treatment until clipped open geometry is available. CM12 excess
  // density likewise remains CM12 state, never an alleged bounded volume.
  if(cellOpenFraction(cell)<0.999999||fill<=0.0||fill>=1.0){return result;}
  let centre=cellCenter(cell);
  var m0=vec3f(0.0);var m1=vec3f(0.0);var m2=vec3f(0.0);
  var rhs=vec3f(0.0);
  for(var incidence=incidenceBegin(cell);incidence<incidenceEnd(cell);incidence+=1u){
    let row=incidenceRow(incidence);
    if(!rowAccepted(row)||rowOpenFraction(row)<=1e-8){continue;}
    let own=termCoefficient(incidenceTerm(incidence));
    let range=rowTermRange(row);
    for(var term=range.x;term<range.y;term+=1u){
      if(own*termCoefficient(term)>=0.0){continue;}
      let other=termCell(term);
      if(other==cell||!cellActive(other)||cellOpenFraction(other)<=1e-8){continue;}
      let delta=cellCenter(other)-centre;
      let area=geometricResidentOverlap(cell,other,rowAxis(row));
      let weight=area/max(dot(delta,delta),1e-12);
      // The least-squares fit uses the actual 3D displacement at mixed seams,
      // rather than pretending that skew fine/coarse centres are axis aligned.
      m0+=weight*delta.x*delta;m1+=weight*delta.y*delta;m2+=weight*delta.z*delta;
      rhs+=weight*delta*(geometricResidentFill(other,densityOffset)-fill);
    }
  }
  let determinant=dot(m0,cross(m1,m2));
  let scale=max(max(m0.x,m1.y),m2.z);
  var gradient=vec3f(0.0);
  if(scale>1e-12&&abs(determinant)>1e-7*scale*scale*scale){
    gradient=(rhs.x*cross(m1,m2)+rhs.y*cross(m2,m0)
      +rhs.z*cross(m0,m1))/determinant;
  }else{
    // A rank-deficient patch cannot establish a 3D interface direction.
    // Returning invalid is preferable to inventing a gravity-aligned plane.
    return result;
  }
  if(dot(gradient,gradient)*dot(widths,widths)<=1e-12){return result;}
  // Density rises into liquid; the geometry kernel expects liquid-to-air.
  result.plane=geometricInterfaceFromFill(fill,-gradient,widths);
  result.valid=1u;
  return result;
}

fn geometricResidentSignedDistance(cell:u32,positionFine:vec3f,
 densityOffset:u32,fallbackFine:f32)->f32{
  let geometry=geometricResidentInterface(cell,densityOffset);
  if(geometry.valid==0u){return fallbackFine;}
  return geometricInterfaceSignedDistance(geometry.plane,positionFine-cellCenter(cell));
}
`;
