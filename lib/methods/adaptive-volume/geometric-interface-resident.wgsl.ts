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

// Transport accepts extensive volume within this relative capacity roundoff.
// Reconstruction must accept the same state: clamp only the local observation,
// never the conserved density/volume authority.
fn geometricResidentCertifiedFill(cell:u32,densityOffset:u32)->vec2f{
  let fill=geometricResidentFill(cell,densityOffset);
  let tolerance=9.5367431640625e-7;
  if(!(fill >= -tolerance && fill <= 1.0+tolerance)){return vec2f(0.0);}
  return vec2f(clamp(fill,0.0,1.0),1.0);
}

fn geometricResidentSameProjection(a:u32,b:u32,extrusion:u32)->bool{
  let ac=cellCenter(a);let aw=cellWidths(a);let bc=cellCenter(b);let bw=cellWidths(b);
  for(var axis=0u;axis<3u;axis+=1u){
    if(axis!=extrusion&&(ac[axis]!=bc[axis]||aw[axis]!=bw[axis])){return false;}
  }
  return true;
}

// Certify a locally extruded adaptive stencil from the accepted graph itself.
// Across an in-plane mixed row, equal projected neighbours may be split into
// several isotropic children along the extrusion axis. They must agree in fill
// and collectively cover this cell's complete extrusion interval. A row along
// the extrusion axis instead requires the ordinary opposite neighbour to agree
// with the centre. This retains true 3-D gradients and only removes a tangent
// attributed by skew fine/coarse centres from invariant scalar data.
fn geometricResidentAdaptiveExtrusionCertified(cell:u32,densityOffset:u32,
 extrusion:u32)->bool{
  let centreFill=geometricResidentFill(cell,densityOffset);
  let centre=cellCenter(cell);let widths=cellWidths(cell);
  let faces=cnxCellFaceRangeUnchecked(cell);var rowFirst=faces.x;
  var previousRow=INVALID;var previousFace=INVALID;
  for(var adjacency=faces.x;adjacency<faces.y;adjacency+=1u){
    let entry=cnxCellFaceEntryUnchecked(adjacency);let face=entry>>1u;
    let row=cnxPhysicalFaceRowUnchecked(face);
    // A repeated incidence replays the row's ascending face sequence. Retain
    // that boundary so projection grouping remains incidence local.
    if(row!=previousRow||(previousFace!=INVALID&&face<=previousFace)){rowFirst=adjacency;}
    previousRow=row;previousFace=face;
    if(rowOpenFraction(row)<=1e-8){continue;}
    let faceCells=cnxPhysicalFaceCellsUnchecked(face);let isNegative=(entry&1u)!=0u;
    let other=select(faceCells.x,faceCells.y,isNegative);
    if(other==INVALID){continue;}
    if(other==cell||!cellActive(other)||cellOpenFraction(other)<0.999999){return false;}
    let observed=geometricResidentFill(other,densityOffset);
    if(rowAxis(row)==extrusion){
      if(abs(observed-centreFill)>9.5367431640625e-7){return false;}
      continue;
    }
    var leader=true;
    for(var prior=rowFirst;prior<adjacency;prior+=1u){
      let priorEntry=cnxCellFaceEntryUnchecked(prior);let priorFace=priorEntry>>1u;
      if(cnxPhysicalFaceRowUnchecked(priorFace)!=row){continue;}
      let priorCells=cnxPhysicalFaceCellsUnchecked(priorFace);
      let priorOther=select(priorCells.x,priorCells.y,(priorEntry&1u)!=0u);
      if(priorOther!=INVALID
        &&geometricResidentSameProjection(priorOther,other,extrusion)){leader=false;break;}
    }
    if(!leader){continue;}
    var coverage=0.0;
    let lower=centre[extrusion]-0.5*widths[extrusion];
    let upper=centre[extrusion]+0.5*widths[extrusion];
    var priorCandidateFace=INVALID;
    for(var candidate=adjacency;candidate<faces.y;candidate+=1u){
      let candidateEntry=cnxCellFaceEntryUnchecked(candidate);
      let candidateFace=candidateEntry>>1u;
      if(cnxPhysicalFaceRowUnchecked(candidateFace)!=row
        ||(priorCandidateFace!=INVALID&&candidateFace<=priorCandidateFace)){break;}
      priorCandidateFace=candidateFace;
      let candidateCells=cnxPhysicalFaceCellsUnchecked(candidateFace);
      let member=select(candidateCells.x,candidateCells.y,(candidateEntry&1u)!=0u);
      if(member==INVALID||!geometricResidentSameProjection(member,other,extrusion)){continue;}
      if(!cellActive(member)||cellOpenFraction(member)<0.999999
        ||abs(geometricResidentFill(member,densityOffset)-observed)>9.5367431640625e-7){return false;}
      let mc=cellCenter(member)[extrusion];let mw=cellWidths(member)[extrusion];
      coverage+=max(0.0,min(upper,mc+0.5*mw)-max(lower,mc-0.5*mw));
    }
    if(coverage<widths[extrusion]-9.5367431640625e-7){return false;}
  }
  return true;
}

fn geometricResidentProjectedGradient(cell:u32,densityOffset:u32,
 extrusion:u32)->vec3f{
  let centre=cellCenter(cell);let fill=geometricResidentFill(cell,densityOffset);
  let axisU=(extrusion+1u)%3u;let axisV=(extrusion+2u)%3u;
  var m00=0.0;var m01=0.0;var m11=0.0;var b0=0.0;var b1=0.0;
  let faces=cnxCellFaceRangeUnchecked(cell);var rowFirst=faces.x;
  var previousRow=INVALID;var previousFace=INVALID;
  for(var adjacency=faces.x;adjacency<faces.y;adjacency+=1u){
    let entry=cnxCellFaceEntryUnchecked(adjacency);let face=entry>>1u;
    let row=cnxPhysicalFaceRowUnchecked(face);
    if(row!=previousRow||(previousFace!=INVALID&&face<=previousFace)){rowFirst=adjacency;}
    previousRow=row;previousFace=face;
    if(rowOpenFraction(row)<=1e-8||rowAxis(row)==extrusion){continue;}
    let faceCells=cnxPhysicalFaceCellsUnchecked(face);
    let other=select(faceCells.x,faceCells.y,(entry&1u)!=0u);
    if(other==INVALID||other==cell||!cellActive(other)
      ||cellOpenFraction(other)<0.999999){continue;}
    var leader=true;
    for(var prior=rowFirst;prior<adjacency;prior+=1u){
      let priorEntry=cnxCellFaceEntryUnchecked(prior);let priorFace=priorEntry>>1u;
      if(cnxPhysicalFaceRowUnchecked(priorFace)!=row){continue;}
      let priorCells=cnxPhysicalFaceCellsUnchecked(priorFace);
      let priorOther=select(priorCells.x,priorCells.y,(priorEntry&1u)!=0u);
      if(priorOther!=INVALID
        &&geometricResidentSameProjection(priorOther,other,extrusion)){leader=false;break;}
    }
    if(!leader){continue;}
    var delta=cellCenter(other)-centre;delta[extrusion]=0.0;
    let remaining=3u-rowAxis(row)-extrusion;
    let a=centre[remaining]-0.5*cellWidths(cell)[remaining];
    let b=centre[remaining]+0.5*cellWidths(cell)[remaining];
    let oc=cellCenter(other)[remaining];let ow=cellWidths(other)[remaining];
    let overlap=max(0.0,min(b,oc+0.5*ow)-max(a,oc-0.5*ow));
    let weight=overlap/max(dot(delta,delta),1e-12);
    let difference=geometricResidentFill(other,densityOffset)-fill;
    let du=delta[axisU];let dv=delta[axisV];
    m00+=weight*du*du;m01+=weight*du*dv;m11+=weight*dv*dv;
    b0+=weight*du*difference;b1+=weight*dv*difference;
  }
  let determinant=m00*m11-m01*m01;let scale=max(m00,m11);var result=vec3f(0.0);
  if(scale>1e-12&&abs(determinant)>1e-7*scale*scale){
    result[axisU]=(m11*b0-m01*b1)/determinant;
    result[axisV]=(-m01*b0+m00*b1)/determinant;
  }
  return result;
}

fn geometricResidentProjectedInterfaceFromFill(fill:f32,gradient:vec3f,
 widths:vec3f,extrusion:u32)->GeometricInterfacePlane{
  let maximum=max(abs(gradient.x),max(abs(gradient.y),abs(gradient.z)));
  if(!(maximum>1e-20)){return GeometricInterfacePlane(vec3f(0.0),0.0);}
  let scaled=gradient/maximum;let axisU=(extrusion+1u)%3u;let axisV=(extrusion+2u)%3u;
  let magnitude=sqrt(scaled[axisU]*scaled[axisU]+scaled[axisV]*scaled[axisV]);
  var normal=vec3f(0.0);normal[axisU]=scaled[axisU]/magnitude;
  normal[axisV]=scaled[axisV]/magnitude;
  return GeometricInterfacePlane(normal,geometricPlaneBoxOffset(normal,widths,fill));
}

// An ELVIRA integration orientation must come from the LS fallback. Choosing
// + for an exactly zero component invents a direction that reflection cannot
// map. Skipping it also removes work from axis-aligned fits.
fn geometricResidentIntegrationSupported(normal:vec3f,axis:u32)->bool{
  return normal[axis]!=0.0;
}

// Uniform, open, extruded stencils admit a volume-consistent 2D fit.
// Candidate slopes are differences of column-integrated liquid heights, not
// differences of cell-average fill. Every candidate retains the central volume.
// Mixed/cut stencils keep LS; supported 3D stencils use the volume fit below.
fn geometricResidentFitScore(plane:GeometricInterfacePlane, samples:array<f32,9>,
 axisU:u32,axisV:u32)->vec2f{
  var full=0.0;var sides=vec4f(0.0);
  for(var j=0u;j<3u;j+=1u){for(var i=0u;i<3u;i+=1u){
    var displacement=vec3f(0.0);displacement[axisU]=f32(i)-1.0;
    displacement[axisV]=f32(j)-1.0;
    let predicted=geometricPlaneBoxFraction(plane.normal,
      plane.offset-dot(plane.normal,displacement),vec3f(1.0));
    let difference=predicted-samples[3u*j+i];let error=difference*difference;
    full+=error;
    if(i<=1u){sides.x+=error;}if(i>=1u){sides.y+=error;}
    if(j<=1u){sides.z+=error;}if(j>=1u){sides.w+=error;}
  }}
  return vec2f(full/9.0,min(min(sides.x,sides.y),min(sides.z,sides.w))/6.0);
}

struct GeometricResidentVolumeFit {
  plane:GeometricInterfacePlane,
  supported:u32,
}
fn geometricResidentFitUniformExtrusion(cell:u32,densityOffset:u32,
 fallback:GeometricInterfacePlane)->GeometricResidentVolumeFit{
  if(any(cellWidths(cell)!=vec3f(1.0))){return GeometricResidentVolumeFit(fallback,0u);}
  let normal=fallback.normal;let magnitude=abs(normal);
  var extrusion=0u;
  if(magnitude.y<magnitude[extrusion]){extrusion=1u;}
  if(magnitude.z<magnitude[extrusion]){extrusion=2u;}
  if(magnitude[extrusion]>1e-6){return GeometricResidentVolumeFit(fallback,0u);}
  let axisU=(extrusion+1u)%3u;let axisV=(extrusion+2u)%3u;
  let centre=cellCenter(cell);var samples:array<f32,9>;
  for(var j=0u;j<3u;j+=1u){for(var i=0u;i<3u;i+=1u){
    var position=centre;position[axisU]+=f32(i)-1.0;position[axisV]+=f32(j)-1.0;
    let other=ownerCellAt(vec3i(floor(position)));
    if(other==INVALID){return GeometricResidentVolumeFit(fallback,0u);}
    if(!cellActive(other)||any(cellWidths(other)!=vec3f(1.0))
      ||any(cellCenter(other)!=position)||cellOpenFraction(other)<0.999999){return GeometricResidentVolumeFit(fallback,0u);}
    let certified=geometricResidentCertifiedFill(other,densityOffset);
    if(certified.y==0.0){return GeometricResidentVolumeFit(fallback,0u);}
    let fill=certified.x;
    samples[3u*j+i]=fill;
    // Certify extrusion through the entire stencil. A physical domain end
    // has no exterior sample; the available inward layer must still match.
    for(var side=0u;side<2u;side+=1u){
      var adjacent=position;adjacent[extrusion]+=select(-1.0,1.0,side==1u);
      let next=ownerCellAt(vec3i(floor(adjacent)));
      if(next==INVALID){
        // Signed sparse-world owners beyond the authored box are real data.
        // A missing sample is a mirror end only at an explicitly solid vessel
        // wall, never because the authored dimensions alone were exceeded.
        let authored=all(centre>=vec3f(0.0))&&all(centre<vec3f(p.dimensions.xyz));
        let outside=adjacent[extrusion]<0.0||adjacent[extrusion]>=f32(p.dimensions[extrusion]);
        if(authored&&outside&&cm12SolidVoxelFractionQ8(vec3i(floor(adjacent)))==255u){continue;}
        return GeometricResidentVolumeFit(fallback,0u);
      }
      if(!cellActive(next)||any(cellWidths(next)!=vec3f(1.0))
        ||any(cellCenter(next)!=adjacent)||cellOpenFraction(next)<0.999999
        ||abs(geometricResidentFill(next,densityOffset)-fill)>9.5367431640625e-7){return GeometricResidentVolumeFit(fallback,0u);}
    }
  }}
  let fill=samples[4u];
  var best=fallback;var bestScore=geometricResidentFitScore(best,samples,axisU,axisV);
  if(bestScore.x<=9.094947017729282e-13){return GeometricResidentVolumeFit(best,1u);}
  var corner=best;var cornerScore=bestScore;
  // Six ELVIRA-style candidates: backward/central/forward height slopes
  // with either coordinate as the column integration direction.
  for(var direction=0u;direction<2u;direction+=1u){
    var heights=vec3f(0.0);
    for(var column=0u;column<3u;column+=1u){
      for(var at=0u;at<3u;at+=1u){
        let index=select(3u*column+at,3u*at+column,direction==1u);
        heights[column]+=samples[index];
      }
    }
    let integration=select(axisU,axisV,direction==1u);
    let transverse=select(axisV,axisU,direction==1u);
    if(!geometricResidentIntegrationSupported(normal,integration)){continue;}
    for(var difference=0u;difference<3u;difference+=1u){
      var slope=heights.y-heights.x;
      if(difference==1u){slope=0.5*(heights.z-heights.x);}
      if(difference==2u){slope=heights.z-heights.y;}
      var candidate=vec3f(0.0);
      candidate[integration]=select(-1.0,1.0,normal[integration]>=0.0);
      candidate[transverse]=-slope;
      let plane=geometricInterfaceFromFill(fill,candidate,vec3f(1.0));
      let score=geometricResidentFitScore(plane,samples,axisU,axisV);
      if(score.x<bestScore.x){best=plane;bestScore=score;}
      if(score.y<cornerScore.y||(score.y==cornerScore.y&&score.x<cornerScore.x)){
        corner=plane;cornerScore=score;
      }
    }
  }
  // A globally consistent planar fit always wins. At an interface corner,
  // a plane can explain one connected half-stencil exactly while the second
  // interface makes a global single-plane fit impossible. Accept that local
  // model only at geometric evaluation roundoff, never by velocity or snapping.
  let fitRoundoff=9.094947017729282e-13; // 64*f32_epsilon^2, mean squared fill.
  if(bestScore.x<=fitRoundoff){return GeometricResidentVolumeFit(best,1u);}
  if(cornerScore.y<=fitRoundoff){return GeometricResidentVolumeFit(corner,1u);}
  return GeometricResidentVolumeFit(best,1u);
}

// Cell-volume observations for a certified uniform open stencil. INVALID
// ownership, inactive cells and scalar cut capacities do not define a cube.
fn geometricResidentUniformSample(position:vec3f,densityOffset:u32)->vec2f{
  let cell=ownerCellAt(vec3i(floor(position)));
  if(cell==INVALID){return vec2f(0.0);}
  if(!cellActive(cell)||any(cellWidths(cell)!=vec3f(1.0))
    ||any(cellCenter(cell)!=position)||cellOpenFraction(cell)<0.999999){return vec2f(0.0);}
  return geometricResidentCertifiedFill(cell,densityOffset);
}

fn geometricResidentVolumeFitScore3D(plane:GeometricInterfacePlane,
 samples:array<f32,27>)->f32{
  var error=0.0;
  for(var z=0u;z<3u;z+=1u){for(var y=0u;y<3u;y+=1u){for(var x=0u;x<3u;x+=1u){
    let displacement=vec3f(f32(x)-1.0,f32(y)-1.0,f32(z)-1.0);
    let predicted=geometricPlaneBoxFraction(plane.normal,
      plane.offset-dot(plane.normal,displacement),vec3f(1.0));
    let difference=predicted-samples[9u*z+3u*y+x];error+=difference*difference;
  }}}
  return error/27.0;
}

fn geometricResidentSlopePlane(fill:f32,slopes:vec2f,axis:u32,
 orientation:f32)->GeometricInterfacePlane{
  var normal=vec3f(0.0);normal[axis]=orientation;
  normal[(axis+1u)%3u]=slopes.x;normal[(axis+2u)%3u]=slopes.y;
  return geometricInterfaceFromFill(fill,normal,vec3f(1.0));
}

// Volume-residual refinement makes the complete 3^3 stencil sufficient even
// when the longer height columns hit a wall. The two unknown slopes retain
// the central cell volume through the analytic/monotone plane offset inverse.
fn geometricResidentRefineUniformPlane3D(samples:array<f32,27>,
 initial:GeometricInterfacePlane,initialError:f32)->GeometricInterfacePlane{
  var best=initial;var error=initialError;
  let magnitude=abs(initial.normal);var axis=0u;
  if(magnitude.y>magnitude[axis]){axis=1u;}
  if(magnitude.z>magnitude[axis]){axis=2u;}
  let orientation=select(-1.0,1.0,initial.normal[axis]>=0.0);
  var slopes=vec2f(initial.normal[(axis+1u)%3u],initial.normal[(axis+2u)%3u])/magnitude[axis];
  for(var iteration=0u;iteration<8u;iteration+=1u){
    if(error<=9.094947017729282e-13){break;}
    let h=0.002;
    let uPlus=geometricResidentSlopePlane(samples[13u],slopes+vec2f(h,0.0),axis,orientation);
    let uMinus=geometricResidentSlopePlane(samples[13u],slopes-vec2f(h,0.0),axis,orientation);
    let vPlus=geometricResidentSlopePlane(samples[13u],slopes+vec2f(0.0,h),axis,orientation);
    let vMinus=geometricResidentSlopePlane(samples[13u],slopes-vec2f(0.0,h),axis,orientation);
    var hessian=vec3f(0.0);var rhs=vec2f(0.0);
    for(var z=0u;z<3u;z+=1u){for(var y=0u;y<3u;y+=1u){for(var x=0u;x<3u;x+=1u){
      let d=vec3f(f32(x)-1.0,f32(y)-1.0,f32(z)-1.0);
      let residual=geometricPlaneBoxFraction(best.normal,best.offset-dot(best.normal,d),vec3f(1.0))
        -samples[9u*z+3u*y+x];
      let derivative=vec2f(
        geometricPlaneBoxFraction(uPlus.normal,uPlus.offset-dot(uPlus.normal,d),vec3f(1.0))
          -geometricPlaneBoxFraction(uMinus.normal,uMinus.offset-dot(uMinus.normal,d),vec3f(1.0)),
        geometricPlaneBoxFraction(vPlus.normal,vPlus.offset-dot(vPlus.normal,d),vec3f(1.0))
          -geometricPlaneBoxFraction(vMinus.normal,vMinus.offset-dot(vMinus.normal,d),vec3f(1.0)))/(2.0*h);
      hessian+=vec3f(derivative.x*derivative.x,derivative.x*derivative.y,derivative.y*derivative.y);
      rhs+=derivative*residual;
    }}}
    let determinant=hessian.x*hessian.z-hessian.y*hessian.y;
    if(!(determinant>1e-12*max(1.0,hessian.x*hessian.z))){break;}
    var step=vec2f(hessian.z*rhs.x-hessian.y*rhs.y,
      hessian.x*rhs.y-hessian.y*rhs.x)/determinant;
    step*=min(1.0,0.5/max(length(step),1e-20));
    var improved=false;
    for(var trial=0u;trial<4u;trial+=1u){
      let proposed=slopes-step;
      let plane=geometricResidentSlopePlane(samples[13u],proposed,axis,orientation);
      let nextError=geometricResidentVolumeFitScore3D(plane,samples);
      if(nextError<error){best=plane;error=nextError;slopes=proposed;improved=true;break;}
      step*=0.5;
    }
    if(!improved){break;}
  }
  return best;
}

fn geometricResidentFitUniformPlane3D(cell:u32,densityOffset:u32,
 fallback:GeometricInterfacePlane)->GeometricInterfacePlane{
  if(any(cellWidths(cell)!=vec3f(1.0))){return fallback;}
  let centre=cellCenter(cell);var samples:array<f32,27>;
  for(var z=0u;z<3u;z+=1u){for(var y=0u;y<3u;y+=1u){for(var x=0u;x<3u;x+=1u){
    let observed=geometricResidentUniformSample(centre+
      vec3f(f32(x)-1.0,f32(y)-1.0,f32(z)-1.0),densityOffset);
    if(observed.y==0.0){return fallback;}
    samples[9u*z+3u*y+x]=observed.x;
  }}}
  var best=fallback;var bestError=geometricResidentVolumeFitScore3D(best,samples);
  if(bestError<=9.094947017729282e-13){return best;}
  // Seven cells enclose the complete plane transition along a dominant axis:
  // central intersection gives |offset/n_axis|<=1.5; a transverse neighbour
  // adds <=1 and its cross-section half widths add <=1, hence |x|<=3.5.
  // Integrated heights on the +/- transverse columns therefore give exact
  // planar slopes. Three-cell sums can truncate a diagonal plane in 3D.
  for(var integration=0u;integration<3u;integration+=1u){
    if(!geometricResidentIntegrationSupported(fallback.normal,integration)){continue;}
    let axisU=(integration+1u)%3u;let axisV=(integration+2u)%3u;
    var heights=vec4f(0.0);var supported=true;
    for(var column=0u;column<4u;column+=1u){
      let transverse=select(axisU,axisV,column>=2u);
      var position=centre;position[transverse]+=select(-1.0,1.0,(column&1u)!=0u);
      for(var along=0u;along<7u;along+=1u){
        var query=position;query[integration]+=f32(along)-3.0;
        let observed=geometricResidentUniformSample(query,densityOffset);
        if(observed.y==0.0){supported=false;break;}
        heights[column]+=observed.x;
      }
      if(!supported){break;}
    }
    if(!supported){continue;}
    var candidate=vec3f(0.0);
    candidate[integration]=select(-1.0,1.0,fallback.normal[integration]>=0.0);
    candidate[axisU]=-0.5*(heights.y-heights.x);
    candidate[axisV]=-0.5*(heights.w-heights.z);
    let plane=geometricInterfaceFromFill(samples[13u],candidate,vec3f(1.0));
    let error=geometricResidentVolumeFitScore3D(plane,samples);
    if(error<bestError){best=plane;bestError=error;}
    if(bestError<=9.094947017729282e-13){return best;}
  }
  return geometricResidentRefineUniformPlane3D(samples,best,bestError);
}

fn geometricResidentReconstructInterface(cell:u32,densityOffset:u32)->GeometricResidentInterface{
  let widths=cellWidths(cell);
  let fill=geometricResidentFill(cell,densityOffset);
  var result:GeometricResidentInterface;
  result.plane=GeometricInterfacePlane(vec3f(0.0,1.0,0.0),0.0);
  result.valid=0u;
  // Interface reconstruction is a production consumer of the sealed accepted
  // topology. A missing or stale transport view fails closed; no second graph
  // is rebuilt from row terms in a scalar stage.
  if(!cnxTransportViewValidForAcceptedTopology()){return result;}
  // Scalar aperture is not a cut-solid polyhedron. Preserve the established
  // solid treatment until clipped open geometry is available. CM12 excess
  // density likewise remains CM12 state, never an alleged bounded volume.
  if(cellOpenFraction(cell)<0.999999||fill<=0.0||fill>=1.0){return result;}
  let centre=cellCenter(cell);
  var m0=vec3f(0.0);var m1=vec3f(0.0);var m2=vec3f(0.0);
  var rhs=vec3f(0.0);
  // CNX's signed cell-face CSR is the accepted physical neighbour graph. Its
  // entry order is the former incidence -> row -> opposite-term order, so this
  // removes repeated term discovery and overlap geometry without changing the
  // f32 least-squares accumulation sequence.
  let faces=cnxCellFaceRangeUnchecked(cell);
  for(var adjacency=faces.x;adjacency<faces.y;adjacency+=1u){
    let entry=cnxCellFaceEntryUnchecked(adjacency);let face=entry>>1u;
    let isNegative=(entry&1u)!=0u;let cells=cnxPhysicalFaceCellsUnchecked(face);
    let other=select(cells.x,cells.y,isNegative);
    if(other==INVALID||other==cell||!cellActive(other)
      ||cellOpenFraction(other)<0.999999){continue;}
    let row=cnxPhysicalFaceRowUnchecked(face);
    if(rowOpenFraction(row)<=1e-8){continue;}
    let delta=cellCenter(other)-centre;
    let weight=cnxPhysicalFaceAreaUnchecked(face)/max(dot(delta,delta),1e-12);
    // The least-squares fit uses the actual 3D displacement at mixed seams,
    // rather than pretending that skew fine/coarse centres are axis aligned.
    m0+=weight*delta.x*delta;m1+=weight*delta.y*delta;m2+=weight*delta.z*delta;
    rhs+=weight*delta*(geometricResidentFill(other,densityOffset)-fill);
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
  let magnitude=abs(gradient);var extrusion=0u;var projectedExtrusion=INVALID;
  if(magnitude.y<magnitude[extrusion]){extrusion=1u;}
  if(magnitude.z<magnitude[extrusion]){extrusion=2u;}
  if(geometricResidentAdaptiveExtrusionCertified(cell,densityOffset,extrusion)){
    let projected=geometricResidentProjectedGradient(cell,densityOffset,extrusion);
    if(dot(projected,projected)>0.0){gradient=projected;projectedExtrusion=extrusion;}
  }
  if(dot(gradient,gradient)*dot(widths,widths)<=1e-12){return result;}
  // Density rises into liquid; the geometry kernel expects liquid-to-air.
  result.plane=geometricInterfaceFromFill(fill,-gradient,widths);
  if(projectedExtrusion!=INVALID){
    result.plane=geometricResidentProjectedInterfaceFromFill(fill,-gradient,widths,projectedExtrusion);
  }
  let extruded=geometricResidentFitUniformExtrusion(cell,densityOffset,result.plane);
  result.plane=extruded.plane;
  if(extruded.supported==0u){
    result.plane=geometricResidentFitUniformPlane3D(cell,densityOffset,result.plane);
  }
  result.valid=select(0u,1u,geometricInterfacePlaneValid(result.plane));
  return result;
}

// The production encoder refreshes this cache after accepted scalar/topology
// changes and before pressure/presentation consumers. Its density bank is the
// current destination bank; standalone shader construction retains direct reads.
fn geometricResidentInterface(cell:u32,densityOffset:u32)->GeometricResidentInterface{
  if(!GEOMETRIC_INTERFACE_CACHE_ENABLED){
    return geometricResidentReconstructInterface(cell,densityOffset);
  }
  let at=GEOMETRIC_INTERFACE_CACHE_BASE+4u*cell;
  let plane=GeometricInterfacePlane(vec3f(state[at],state[at+1u],state[at+2u]),state[at+3u]);
  return GeometricResidentInterface(plane,select(0u,1u,geometricInterfacePlaneValid(plane)));
}

fn geometricResidentStoreInterface(cell:u32,densityOffset:u32){
  let geometry=geometricResidentReconstructInterface(cell,densityOffset);
  let at=GEOMETRIC_INTERFACE_CACHE_BASE+4u*cell;
  let normal=select(vec3f(0.0),geometry.plane.normal,geometry.valid!=0u);
  state[at]=normal.x;state[at+1u]=normal.y;state[at+2u]=normal.z;
  state[at+3u]=select(0.0,geometry.plane.offset,geometry.valid!=0u);
}

@compute @workgroup_size(64)
fn refreshGeometricInterface(@builtin(global_invocation_id)gid:vec3u){
  if(!GEOMETRIC_INTERFACE_CACHE_ENABLED){return;}
  let cell=acceptedTemplateCellInvocation(gid.x);if(cell==INVALID){return;}
  geometricResidentStoreInterface(cell,destinationDensity());
}

@compute @workgroup_size(64)
fn refreshGeometricInterfacePublished(@builtin(global_invocation_id)gid:vec3u){
  if(!GEOMETRIC_INTERFACE_CACHE_ENABLED){return;}
  let cell=acceptedTemplateCellInvocation(gid.x);if(cell==INVALID){return;}
  let densityOffset=select(p.stateOffsets0.x,p.stateOffsets0.y,
    cm12FramePlanAcceptedParity()!=0u);
  geometricResidentStoreInterface(cell,densityOffset);
}

// Extend supported planes once per accepted cell into a distinct cache. The
// raw cache is immutable during this dispatch, so neighboring invocation order
// cannot change which interfaces provide support.
fn geometricResidentSupportedInterface(cell:u32,densityOffset:u32)->GeometricResidentInterface{
  var result=GeometricResidentInterface(GeometricInterfacePlane(vec3f(0.0),0.0),0u);
  if(!cnxTransportViewValidForAcceptedTopology()){return result;}
  let geometry=geometricResidentInterface(cell,densityOffset);
  if(geometry.valid!=0u){return geometry;}
  if(cellOpenFraction(cell)<0.999999){return result;}
  var normal=vec3f(0.0);var phi=0.0;var totalWeight=0.0;
  let centre=cellCenter(cell);
  let faces=cnxCellFaceRangeUnchecked(cell);
  for(var adjacency=faces.x;adjacency<faces.y;adjacency+=1u){
    let entry=cnxCellFaceEntryUnchecked(adjacency);let face=entry>>1u;
    let isNegative=(entry&1u)!=0u;let cells=cnxPhysicalFaceCellsUnchecked(face);
    let other=select(cells.x,cells.y,isNegative);
    if(other==INVALID||!cellActive(other)){continue;}
    let row=cnxPhysicalFaceRowUnchecked(face);if(rowOpenFraction(row)<=1e-8){continue;}
    let candidate=geometricResidentInterface(other,densityOffset);
    if(candidate.valid==0u){continue;}
    let delta=cellCenter(other)-centre;
    let weight=cnxPhysicalFaceAreaUnchecked(face)/max(dot(delta,delta),1e-12);
    normal+=weight*candidate.plane.normal;
    phi+=weight*geometricInterfaceSignedDistance(candidate.plane,centre-cellCenter(other));
    totalWeight+=weight;
  }
  let normalLength=length(normal);
  if(!(totalWeight>1e-8&&normalLength>1e-6*totalWeight)){return result;}
  result.plane=GeometricInterfacePlane(normal/normalLength,-phi/normalLength);
  result.valid=1u;
  return result;
}

@compute @workgroup_size(64)
fn extendGeometricInterface(@builtin(global_invocation_id)gid:vec3u){
  if(!GEOMETRIC_INTERFACE_SUPPORT_CACHE_ENABLED){return;}
  let cell=acceptedTemplateCellInvocation(gid.x);if(cell==INVALID){return;}
  // The raw cache already carries the selected pressure or publication bank.
  let geometry=geometricResidentSupportedInterface(cell,destinationDensity());
  let at=GEOMETRIC_INTERFACE_SUPPORT_CACHE_BASE+4u*cell;
  state[at]=geometry.plane.normal.x;state[at+1u]=geometry.plane.normal.y;
  state[at+2u]=geometry.plane.normal.z;state[at+3u]=geometry.plane.offset;
}

// Area centroid of the PLIC polygon cut through the accepted cell box. plicRDF
// weights use the interface centre, rather than an arbitrary point on the same
// plane: the signed distance is unchanged, but the orientation/distance weight
// is not. The fixed 12-edge construction is bounded for every convex box.
fn geometricResidentInterfaceCentroid(cell:u32,
 plane:GeometricInterfacePlane)->vec3f{
  let halfWidths=0.5*cellWidths(cell);var points:array<vec3f,12>;var count=0u;
  for(var axis=0u;axis<3u;axis+=1u){
    let u=(axis+1u)%3u;let v=(axis+2u)%3u;
    for(var su=0u;su<2u;su+=1u){for(var sv=0u;sv<2u;sv+=1u){
      var a=vec3f(0.0);var b=vec3f(0.0);
      a[axis]=-halfWidths[axis];b[axis]=halfWidths[axis];
      a[u]=select(-halfWidths[u],halfWidths[u],su!=0u);b[u]=a[u];
      a[v]=select(-halfWidths[v],halfWidths[v],sv!=0u);b[v]=a[v];
      let fa=geometricInterfaceSignedDistance(plane,a);
      let fb=geometricInterfaceSignedDistance(plane,b);
      if((fa<=0.0&&fb>=0.0)||(fa>=0.0&&fb<=0.0)){
        let denominator=fa-fb;
        if(abs(denominator)>1e-12){
          let point=mix(a,b,clamp(fa/denominator,0.0,1.0));var unique=true;
          for(var prior=0u;prior<count;prior+=1u){
            unique=unique&&dot(points[prior]-point,points[prior]-point)>1e-10;
          }
          if(unique&&count<12u){points[count]=point;count+=1u;}
        }
      }
    }}
  }
  if(count<3u){return cellCenter(cell)+plane.normal*plane.offset;}
  var arithmetic=vec3f(0.0);
  for(var i=0u;i<count;i+=1u){arithmetic+=points[i];}
  arithmetic/=f32(count);
  let absolute=abs(plane.normal);var reference=vec3f(1.0,0.0,0.0);
  if(absolute.y<=absolute.x&&absolute.y<=absolute.z){reference=vec3f(0.0,1.0,0.0);}
  else if(absolute.z<=absolute.x&&absolute.z<=absolute.y){reference=vec3f(0.0,0.0,1.0);}
  let basisU=normalize(cross(plane.normal,reference));let basisV=cross(plane.normal,basisU);
  var angles:array<f32,12>;
  for(var i=0u;i<count;i+=1u){
    let delta=points[i]-arithmetic;angles[i]=atan2(dot(delta,basisV),dot(delta,basisU));
  }
  for(var i=0u;i<count;i+=1u){
    var first=i;
    for(var j=i+1u;j<count;j+=1u){if(angles[j]<angles[first]){first=j;}}
    if(first!=i){let point=points[i];points[i]=points[first];points[first]=point;
      let angle=angles[i];angles[i]=angles[first];angles[first]=angle;}
  }
  var area2=0.0;var centroid2=vec2f(0.0);
  for(var i=0u;i<count;i+=1u){
    let a=points[i]-arithmetic;let b=points[(i+1u)%count]-arithmetic;
    let ax=dot(a,basisU);let ay=dot(a,basisV);
    let bx=dot(b,basisU);let by=dot(b,basisV);let cross2=ax*by-bx*ay;
    area2+=cross2;centroid2+=cross2*vec2f(ax+bx,ay+by);
  }
  if(abs(area2)<=1e-10){return cellCenter(cell)+plane.normal*plane.offset;}
  centroid2/=3.0*area2;
  return cellCenter(cell)+arithmetic+basisU*centroid2.x+basisV*centroid2.y;
}

fn geometricResidentRdfPlaneContributionAt(targetCentre:vec3f,source:u32,
 densityOffset:u32)->vec2f{
  let geometry=geometricResidentInterface(source,densityOffset);
  if(geometry.valid==0u){return vec2f(0.0);}
  let interfaceCentre=geometricResidentInterfaceCentroid(source,geometry.plane);
  let delta=targetCentre-interfaceCentre;
  let distance=geometricInterfaceSignedDistance(geometry.plane,
    targetCentre-cellCenter(source));
  let squared=dot(delta,delta);
  // Equation 14 applies the same A=2 orientation weight to every interface
  // point neighbour, including the destination cell itself. For the singular
  // coincident-centroid case both numerator and the guarded ratio are zero;
  // other point-neighbour planes (or the explicit fill fallback) then carry
  // the RDF instead of inventing a unit self weight.
  let weight=distance*distance/max(squared,1e-12);
  return vec2f(weight*distance,weight);
}

fn geometricResidentRdfPlaneContribution(destinationCell:u32,source:u32,
 densityOffset:u32)->vec2f{
  return geometricResidentRdfPlaneContributionAt(cellCenter(destinationCell),
    source,densityOffset);
}

// Accepted phase at a point in one open owner. Pure cells certify their whole
// box; mixed cells use their own accepted PLIC rather than an extrapolated RDF.
// Zero is reserved for cells whose interface geometry is unresolved.
fn geometricResidentRdfAcceptedPhase(cell:u32,positionFine:vec3f,
 densityOffset:u32)->i32{
  let fill=clamp(state[densityOffset+cell]
    /max(cellOpenFraction(cell),1e-6),0.0,1.0);
  if(fill>=1.0-1e-6){return -1;}
  if(fill<=1e-6){return 1;}
  let geometry=geometricResidentInterface(cell,densityOffset);
  if(geometry.valid==0u){return 0;}
  let distance=geometricInterfaceSignedDistance(geometry.plane,
    positionFine-cellCenter(cell));
  return select(1,-1,distance<=0.0);
}

fn geometricResidentRdfBoundToAccepted(value:f32,acceptedDistance:f32,
 phase:i32)->f32{
  if(phase<0&&value>=0.0){return min(0.0,acceptedDistance);}
  if(phase>0&&value<=0.0){return max(0.0,acceptedDistance);}
  return value;
}

// Repair only an RDF sign inversion. For a mixed owner the replacement is its
// accepted PLIC distance. A pure owner's interface lies outside its box, so
// distance to the nearest box face is a conservative strict interior margin.
fn geometricResidentRdfBoundAtOwner(cell:u32,positionFine:vec3f,
 densityOffset:u32,value:f32)->f32{
  let phase=geometricResidentRdfAcceptedPhase(cell,positionFine,densityOffset);
  if(phase==0){return value;}
  let geometry=geometricResidentInterface(cell,densityOffset);
  if(geometry.valid!=0u){
    let accepted=geometricInterfaceSignedDistance(geometry.plane,
      positionFine-cellCenter(cell));
    return geometricResidentRdfBoundToAccepted(value,accepted,phase);
  }
  let widths=cellWidths(cell);let lower=cellCenter(cell)-0.5*widths;
  let upper=lower+widths;let interior=min(positionFine-lower,upper-positionFine);
  let margin=max(0.0,min(interior.x,min(interior.y,interior.z)));
  return geometricResidentRdfBoundToAccepted(value,select(margin,-margin,phase<0),phase);
}

// One accepted owner incident to a corner of the destination cell. Sampling
// the integer fine voxel immediately inside each of the corner's eight
// octants enumerates the paper's point-neighbour stencil. It also includes
// every fine owner along a nonconforming 2:1 face or edge, whereas walking
// physical face rows alone degenerates to six neighbours on a uniform grid.
fn geometricResidentRdfPointNeighborAt(vertex:vec3i,octant:u32)->u32{
  let query=vertex+vec3i(select(-1,0,(octant&1u)!=0u),
    select(-1,0,(octant&2u)!=0u),select(-1,0,(octant&4u)!=0u));
  let owner=compactOwnerCellAt(query);
  if(owner.x==INVALID||!brickActive(owner.y)||!cellActive(owner.x)){return INVALID;}
  return owner.x;
}

// Point-neighbour record for the presentation fit.  An allocated but inactive
// WDR leaf retains the accepted directory rung and template cell geometry, so
// it can supply the surrounding air-centre sample assumed by plicRDF without
// becoming simulation authority. x is the template cell and y is 1 for an
// active accepted cell or 2 for an inactive air template. Static solid/cut
// voxels and coordinates outside the physical domain provide no sample.
fn geometricResidentRdfPointNeighborRecord(vertex:vec3i,octant:u32)->vec2u{
  let query=vertex+vec3i(select(-1,0,(octant&1u)!=0u),
    select(-1,0,(octant&2u)!=0u),select(-1,0,(octant&4u)!=0u));
  if(any(query<vec3i(0))||any(query>=vec3i(p.dimensions.xyz))
    ||cm12SolidVoxelFractionQ8(query)>0u){return vec2u(INVALID,0u);}
  let owner=compactOwnerCellAt(query);
  if(owner.x==INVALID){return vec2u(INVALID,0u);}
  if(brickActive(owner.y)&&cellActive(owner.x)){
    if(cellOpenFraction(owner.x)<0.999999){return vec2u(INVALID,0u);}
    return vec2u(owner.x,1u);
  }
  return vec2u(owner.x,2u);
}

fn geometricResidentRdfCellCorner(cell:u32,corner:u32)->vec3i{
  let widths=cellWidths(cell);
  let lower=vec3i(round(cellCenter(cell)-0.5*widths));
  return lower+vec3i(round(vec3f(
    select(0.0,widths.x,(corner&1u)!=0u),
    select(0.0,widths.y,(corner&2u)!=0u),
    select(0.0,widths.z,(corner&4u)!=0u))));
}

// Return the open octants connected to the destination around one shared
// vertex. Point contact alone is not fluid connectivity: this local graph
// prevents RDF support from crossing a solid edge/corner whose intervening
// face-adjacent octants are closed. Partial-capacity owners remain excluded
// until their actual clipped polyhedra are available.
fn geometricResidentRdfConnectedVertexOwners(cell:u32,vertex:vec3i)->array<u32,8>{
  var owners:array<u32,8>;var reachable:array<u32,8>;
  for(var octant=0u;octant<8u;octant+=1u){
    let owner=geometricResidentRdfPointNeighborAt(vertex,octant);
    owners[octant]=select(INVALID,owner,owner!=INVALID
      &&cellOpenFraction(owner)>=0.999999);
    reachable[octant]=select(0u,1u,owners[octant]==cell);
  }
  for(var closure=0u;closure<8u;closure+=1u){
    for(var octant=0u;octant<8u;octant+=1u){
      if(reachable[octant]==0u){continue;}
      for(var axis=0u;axis<3u;axis+=1u){
        let adjacent=octant^(1u<<axis);
        if(owners[adjacent]!=INVALID){reachable[adjacent]=1u;}
      }
    }
  }
  for(var octant=0u;octant<8u;octant+=1u){
    if(reachable[octant]==0u){owners[octant]=INVALID;}
  }
  return owners;
}

fn geometricResidentRdfConnectedCornerOwners(cell:u32,corner:u32)->array<u32,8>{
  return geometricResidentRdfConnectedVertexOwners(cell,
    geometricResidentRdfCellCorner(cell,corner));
}

// Scheufler/Roenby reconstructed distance at accepted cell centres. Sources
// are the unique accepted cells sharing any vertex with the destination, as
// required by the point-neighbour definition on both uniform and 2:1 grids.
fn geometricResidentStoreRdfValue(cell:u32,densityOffset:u32){
  let fill=clamp(state[densityOffset+cell]
    /max(cellOpenFraction(cell),1e-6),0.0,1.0);
  let widths=cellWidths(cell);
  let fallback=(CM12_LIQUID_ISOVALUE-fill)*4.0*min(widths.x,min(widths.y,widths.z));
  let at=GEOMETRIC_INTERFACE_RDF_CACHE_BASE+4u*cell;
  state[at]=0.0;state[at+1u]=0.0;state[at+2u]=0.0;
  // The stored capacity is only a scalar fraction; it does not locate the
  // open polyhedron needed for a volume-correct interface point. Keep the
  // explicit fill fallback used by legacy presentation instead of inferring
  // a zero crossing from an adjacent open-cell plane.
  if(cellOpenFraction(cell)<0.999999){state[at+3u]=fallback;return;}
  var neighbors:array<u32,64>;var neighborCount=0u;
  for(var corner=0u;corner<8u;corner+=1u){
    let cornerOwners=geometricResidentRdfConnectedCornerOwners(cell,corner);
    for(var octant=0u;octant<8u;octant+=1u){
    let other=cornerOwners[octant];
    if(other==INVALID){continue;}var unique=true;
    for(var prior=0u;prior<neighborCount;prior+=1u){unique=unique&&neighbors[prior]!=other;}
    if(unique){neighbors[neighborCount]=other;neighborCount+=1u;}
  }}
  var sum=vec2f(0.0);
  for(var neighbor=0u;neighbor<neighborCount;neighbor+=1u){
    let other=neighbors[neighbor];
    if(cellOpenFraction(other)>=0.999999){
      sum+=geometricResidentRdfPlaneContribution(cell,other,densityOffset);
    }
  }
  var value=select(fallback,sum.x/sum.y,sum.y>1e-8);
  let geometry=geometricResidentInterface(cell,densityOffset);
  if(geometry.valid!=0u){
    let accepted=geometricInterfaceSignedDistance(geometry.plane,vec3f(0.0));
    let phase=select(1,-1,accepted<=0.0);
    value=geometricResidentRdfBoundToAccepted(value,accepted,phase);
  }else{
    let centreMargin=0.5*min(widths.x,min(widths.y,widths.z));
    if(fill>=1.0-1e-6){value=geometricResidentRdfBoundToAccepted(value,-centreMargin,-1);}
    else if(fill<=1e-6){value=geometricResidentRdfBoundToAccepted(value,centreMargin,1);}
  }
  state[at+3u]=value;
}

@compute @workgroup_size(64)
fn publishGeometricInterfaceRdfValues(@builtin(global_invocation_id)gid:vec3u){
  if(!GEOMETRIC_INTERFACE_RDF_CACHE_ENABLED){return;}
  let cell=acceptedTemplateCellInvocation(gid.x);if(cell==INVALID){return;}
  geometricResidentStoreRdfValue(cell,destinationDensity());
}

@compute @workgroup_size(64)
fn publishGeometricInterfaceRdfValuesPublished(@builtin(global_invocation_id)gid:vec3u){
  if(!GEOMETRIC_INTERFACE_RDF_CACHE_ENABLED){return;}
  let cell=acceptedTemplateCellInvocation(gid.x);if(cell==INVALID){return;}
  let densityOffset=select(p.stateOffsets0.x,p.stateOffsets0.y,
    cm12FramePlanAcceptedParity()!=0u);
  geometricResidentStoreRdfValue(cell,densityOffset);
}

// Least-squares gradient of the immutable RDF-centre values. This is a second
// dispatch: every .w operand is complete before any invocation publishes its
// independent xyz fit, so no workgroup ordering can alter the shared field.
@compute @workgroup_size(64)
fn fitGeometricInterfaceRdfGradients(@builtin(global_invocation_id)gid:vec3u){
  if(!GEOMETRIC_INTERFACE_RDF_CACHE_ENABLED){return;}
  let cell=acceptedTemplateCellInvocation(gid.x);if(cell==INVALID){return;}
  if(cellOpenFraction(cell)<0.999999){return;}
  let centre=cellCenter(cell);let at=GEOMETRIC_INTERFACE_RDF_CACHE_BASE+4u*cell;
  let value=state[at+3u];var mxx=0.0;var mxy=0.0;var mxz=0.0;
  var myy=0.0;var myz=0.0;var mzz=0.0;var rhs=vec3f(0.0);
  var neighbors:array<u32,64>;var neighborCount=0u;
  for(var corner=0u;corner<8u;corner+=1u){
    let cornerOwners=geometricResidentRdfConnectedCornerOwners(cell,corner);
    for(var octant=0u;octant<8u;octant+=1u){
    let other=cornerOwners[octant];
    if(other==INVALID||other==cell){continue;}var unique=true;
    for(var prior=0u;prior<neighborCount;prior+=1u){unique=unique&&neighbors[prior]!=other;}
    if(unique){neighbors[neighborCount]=other;neighborCount+=1u;}
  }}
  for(var neighbor=0u;neighbor<neighborCount;neighbor+=1u){
    let other=neighbors[neighbor];let delta=cellCenter(other)-centre;
    let squared=max(dot(delta,delta),1e-12);let weight=1.0/squared;
    let difference=state[GEOMETRIC_INTERFACE_RDF_CACHE_BASE+4u*other+3u]-value;
    mxx+=weight*delta.x*delta.x;mxy+=weight*delta.x*delta.y;
    mxz+=weight*delta.x*delta.z;myy+=weight*delta.y*delta.y;
    myz+=weight*delta.y*delta.z;mzz+=weight*delta.z*delta.z;
    rhs+=weight*difference*delta;
  }
  let determinant=mxx*(myy*mzz-myz*myz)-mxy*(mxy*mzz-myz*mxz)
    +mxz*(mxy*myz-myy*mxz);var gradient=vec3f(0.0);
  if(abs(determinant)>1e-10){
    gradient=vec3f(
      rhs.x*(myy*mzz-myz*myz)-mxy*(rhs.y*mzz-myz*rhs.z)
        +mxz*(rhs.y*myz-myy*rhs.z),
      mxx*(rhs.y*mzz-myz*rhs.z)-rhs.x*(mxy*mzz-myz*mxz)
        +mxz*(mxy*rhs.z-rhs.y*mxz),
      mxx*(myy*rhs.z-rhs.y*myz)-mxy*(mxy*rhs.z-rhs.y*mxz)
        +rhs.x*(mxy*myz-myy*mxz))/determinant;
  }else{
    gradient=vec3f(select(0.0,rhs.x/mxx,mxx>1e-10),
      select(0.0,rhs.y/myy,myy>1e-10),select(0.0,rhs.z/mzz,mzz>1e-10));
  }
  state[at]=gradient.x;state[at+1u]=gradient.y;state[at+2u]=gradient.z;
}

// Algorithm 2, step 5 of Scheufler/Roenby: interpolate the immutable
// cell-centre RDF values to one geometric mesh vertex with a free affine
// least-squares fit.  One canonical value is therefore shared by every cell
// incident to that vertex.  The reference cell selects the locally connected
// open octants, so a point contact across a solid edge cannot couple fields.
fn geometricResidentRdfVertexValue(referenceCell:u32,vertex:vec3i,
 densityOffset:u32)->vec2f{
  if(cellOpenFraction(referenceCell)<0.999999){return vec2f(0.0);}
  var records:array<vec2u,8>;var reachable:array<u32,8>;
  for(var octant=0u;octant<8u;octant+=1u){
    records[octant]=geometricResidentRdfPointNeighborRecord(vertex,octant);
    reachable[octant]=select(0u,1u,records[octant].x==referenceCell
      &&records[octant].y==1u);
  }
  // Connectivity is local to the eight incident octants. Inactive air may
  // complete the paper stencil, while rejected solid/cut octants still block
  // a diagonal plane from crossing a thin wall.
  for(var closure=0u;closure<8u;closure+=1u){
    for(var octant=0u;octant<8u;octant+=1u){
      if(reachable[octant]==0u){continue;}
      for(var axis=0u;axis<3u;axis+=1u){
        let adjacent=octant^(1u<<axis);
        if(records[adjacent].x!=INVALID){reachable[adjacent]=1u;}
      }
    }
  }
  var acceptedNeighbors:array<u32,8>;var activeCount=0u;
  for(var octant=0u;octant<8u;octant+=1u){
    if(reachable[octant]==0u||records[octant].y!=1u){continue;}
    let other=records[octant].x;var unique=true;
    for(var prior=0u;prior<activeCount;prior+=1u){
      unique=unique&&acceptedNeighbors[prior]!=other;
    }
    if(unique){acceptedNeighbors[activeCount]=other;activeCount+=1u;}
  }
  if(activeCount==0u){return vec2f(0.0);}
  var sampleCenters:array<vec3f,8>;var sampleValues:array<f32,8>;var count=0u;
  for(var neighbor=0u;neighbor<activeCount;neighbor+=1u){
    let other=acceptedNeighbors[neighbor];sampleCenters[count]=cellCenter(other);
    sampleValues[count]=state[GEOMETRIC_INTERFACE_RDF_CACHE_BASE+4u*other+3u];count+=1u;
  }
  for(var octant=0u;octant<8u;octant+=1u){
    if(reachable[octant]==0u||records[octant].y!=2u){continue;}
    let ghost=records[octant].x;let centre=cellCenter(ghost);var unique=true;
    for(var prior=0u;prior<count;prior+=1u){
      unique=unique&&dot(sampleCenters[prior]-centre,sampleCenters[prior]-centre)>1e-10;
    }
    if(!unique){continue;}
    var contribution=vec2f(0.0);
    for(var neighbor=0u;neighbor<activeCount;neighbor+=1u){
      contribution+=geometricResidentRdfPlaneContributionAt(centre,
        acceptedNeighbors[neighbor],densityOffset);
    }
    let widths=cellWidths(ghost);let margin=0.5*min(widths.x,min(widths.y,widths.z));
    let fallback=4.0*CM12_LIQUID_ISOVALUE*min(widths.x,min(widths.y,widths.z));
    sampleCenters[count]=centre;
    sampleValues[count]=max(margin,
      select(fallback,contribution.x/contribution.y,contribution.y>1e-8));
    count+=1u;
  }
  var meanCenter=vec3f(0.0);var meanValue=0.0;
  for(var neighbor=0u;neighbor<count;neighbor+=1u){
    meanCenter+=sampleCenters[neighbor];meanValue+=sampleValues[neighbor];
  }
  let inverseCount=1.0/f32(count);meanCenter*=inverseCount;meanValue*=inverseCount;
  var mxx=0.0;var mxy=0.0;var mxz=0.0;var myy=0.0;var myz=0.0;var mzz=0.0;
  var rhs=vec3f(0.0);var deltas:array<vec3f,8>;var differences:array<f32,8>;
  var maximumDeltaSquared=0.0;var firstBasisIndex=0u;
  for(var neighbor=0u;neighbor<count;neighbor+=1u){
    let delta=sampleCenters[neighbor]-meanCenter;
    let difference=sampleValues[neighbor]-meanValue;
    deltas[neighbor]=delta;differences[neighbor]=difference;
    let deltaSquared=dot(delta,delta);
    if(deltaSquared>maximumDeltaSquared){
      maximumDeltaSquared=deltaSquared;firstBasisIndex=neighbor;
    }
    mxx+=delta.x*delta.x;mxy+=delta.x*delta.y;mxz+=delta.x*delta.z;
    myy+=delta.y*delta.y;myz+=delta.y*delta.z;mzz+=delta.z*delta.z;
    rhs+=difference*delta;
  }
  let determinant=mxx*(myy*mzz-myz*myz)-mxy*(mxy*mzz-myz*mxz)
    +mxz*(mxy*myz-myy*mxz);var gradient=vec3f(0.0);
  let matrixScale=max(mxx,max(myy,mzz));
  let determinantCutoff=64.0*1.1920928955078125e-7
    *matrixScale*matrixScale*matrixScale;
  if(abs(determinant)>determinantCutoff){
    gradient=vec3f(
      rhs.x*(myy*mzz-myz*myz)-mxy*(rhs.y*mzz-myz*rhs.z)
        +mxz*(rhs.y*myz-myy*rhs.z),
      mxx*(rhs.y*mzz-myz*rhs.z)-rhs.x*(mxy*mzz-myz*mxz)
        +mxz*(mxy*rhs.z-rhs.y*mxz),
      mxx*(myy*rhs.z-rhs.y*myz)-mxy*(mxy*rhs.z-rhs.y*mxz)
        +rhs.x*(mxy*myz-myy*mxz))/determinant;
  }else if(maximumDeltaSquared>1e-12){
    // A domain face or edge legitimately gives Eq. 11 a rank-one or rank-two
    // stencil.  Preserve every observable slope with a pivoted, reorthogonalized
    // row-space basis; setting the whole gradient to zero displaces an otherwise
    // exact planar interface by almost one coarse cell near the boundary.
    let rankCutoff=64.0*1.1920928955078125e-7*maximumDeltaSquared;
    let e0=normalize(deltas[firstBasisIndex]);var e1=vec3f(0.0);
    var e2=vec3f(0.0);var secondResidualSquared=0.0;var secondBasisIndex=0u;
    for(var sample=0u;sample<count;sample+=1u){
      let residual=deltas[sample]-dot(deltas[sample],e0)*e0;
      let squared=dot(residual,residual);
      if(squared>secondResidualSquared){
        secondResidualSquared=squared;secondBasisIndex=sample;
      }
    }
    var rank=1u;
    if(secondResidualSquared>rankCutoff){
      e1=deltas[secondBasisIndex]-dot(deltas[secondBasisIndex],e0)*e0;
      e1=normalize(e1-dot(e1,e0)*e0);rank=2u;
      var thirdResidualSquared=0.0;var thirdBasisIndex=0u;
      for(var sample=0u;sample<count;sample+=1u){
        let residual=deltas[sample]-dot(deltas[sample],e0)*e0
          -dot(deltas[sample],e1)*e1;
        let squared=dot(residual,residual);
        if(squared>thirdResidualSquared){
          thirdResidualSquared=squared;thirdBasisIndex=sample;
        }
      }
      if(thirdResidualSquared>rankCutoff){
        e2=deltas[thirdBasisIndex]-dot(deltas[thirdBasisIndex],e0)*e0
          -dot(deltas[thirdBasisIndex],e1)*e1;
        e2=normalize(e2-dot(e2,e0)*e0-dot(e2,e1)*e1);rank=3u;
      }
    }
    var b00=0.0;var b01=0.0;var b02=0.0;var b11=0.0;var b12=0.0;var b22=0.0;
    var c=vec3f(0.0);
    for(var sample=0u;sample<count;sample+=1u){
      let projected=vec3f(dot(deltas[sample],e0),dot(deltas[sample],e1),
        dot(deltas[sample],e2));
      b00+=projected.x*projected.x;b01+=projected.x*projected.y;
      b02+=projected.x*projected.z;b11+=projected.y*projected.y;
      b12+=projected.y*projected.z;b22+=projected.z*projected.z;
      c+=differences[sample]*projected;
    }
    var h=vec3f(c.x/max(b00,1e-12),0.0,0.0);
    if(rank>=2u){
      let determinant2=b00*b11-b01*b01;
      if(determinant2>1e-10*max(b00*b11,1e-12)){
        h=vec3f((c.x*b11-b01*c.y)/determinant2,
          (b00*c.y-b01*c.x)/determinant2,0.0);
      }
    }
    if(rank==3u){
      let determinant3=b00*(b11*b22-b12*b12)-b01*(b01*b22-b12*b02)
        +b02*(b01*b12-b11*b02);
      if(abs(determinant3)>1e-10*max(b00*b11*b22,1e-12)){
        h=vec3f(
          c.x*(b11*b22-b12*b12)-b01*(c.y*b22-b12*c.z)
            +b02*(c.y*b12-b11*c.z),
          b00*(c.y*b22-b12*c.z)-c.x*(b01*b22-b12*b02)
            +b02*(b01*c.z-c.y*b02),
          b00*(b11*c.z-c.y*b12)-b01*(b01*c.z-c.y*b02)
            +c.x*(b01*b12-b11*b02))/determinant3;
      }
    }
    gradient=h.x*e0+h.y*e1+h.z*e2;
  }
  var value=meanValue+dot(gradient,vec3f(vertex)-meanCenter);
  var hasLiquid=false;var hasAir=false;
  for(var neighbor=0u;neighbor<activeCount;neighbor+=1u){
    let phase=geometricResidentRdfAcceptedPhase(acceptedNeighbors[neighbor],
      vec3f(vertex),densityOffset);
    hasLiquid=hasLiquid||phase<0;hasAir=hasAir||phase>0;
  }
  hasAir=hasAir||count>activeCount;
  if(hasLiquid&&!hasAir){value=min(value,0.0);}
  else if(hasAir&&!hasLiquid){value=max(value,0.0);}
  return vec2f(value,1.0);
}

// FPP stores a cell-centred nodal lattice: sample slot s represents the
// physical point s+0.5 in finest-cell coordinates.  Reconstruct the paper's
// values at the accepted cell's eight true topology vertices, then evaluate
// their trilinear interpolant at that exact producer point.  This preserves
// the established +0.5 render adapter while avoiding independent owner traces.
fn geometricResidentRdfPublishedDistance(cell:u32,positionFine:vec3f,
 densityOffset:u32)->vec2f{
  if(!GEOMETRIC_INTERFACE_RDF_CACHE_ENABLED||cellOpenFraction(cell)<0.999999){
    return vec2f(0.0);
  }
  let widths=cellWidths(cell);let lower=cellCenter(cell)-0.5*widths;
  let fraction=clamp((positionFine-lower)/widths,vec3f(0.0),vec3f(1.0));
  var sum=vec2f(0.0);
  for(var corner=0u;corner<8u;corner+=1u){
    let vertex=geometricResidentRdfCellCorner(cell,corner);
    let value=geometricResidentRdfVertexValue(cell,vertex,densityOffset);
    let wx=select(1.0-fraction.x,fraction.x,(corner&1u)!=0u);
    let wy=select(1.0-fraction.y,fraction.y,(corner&2u)!=0u);
    let wz=select(1.0-fraction.z,fraction.z,(corner&4u)!=0u);
    sum+=wx*wy*wz*value;
  }
  if(sum.y<=1e-8){return vec2f(0.0);}
  var value=sum.x/sum.y;
  value=geometricResidentRdfBoundAtOwner(cell,positionFine,densityOffset,value);
  return vec2f(value,1.0);
}

fn geometricResidentRdfDistance(cell:u32,positionFine:vec3f)->vec2f{
  if(!GEOMETRIC_INTERFACE_RDF_CACHE_ENABLED){return vec2f(0.0);}
  let at=GEOMETRIC_INTERFACE_RDF_CACHE_BASE+4u*cell;
  let value=state[at+3u]+dot(vec3f(state[at],state[at+1u],state[at+2u]),
    positionFine-cellCenter(cell));
  return vec2f(value,1.0);
}

fn geometricResidentDistanceSupport(cell:u32,positionFine:vec3f,densityOffset:u32)->vec2f{
  var geometry:GeometricResidentInterface;
  if(GEOMETRIC_INTERFACE_SUPPORT_CACHE_ENABLED){
    let at=GEOMETRIC_INTERFACE_SUPPORT_CACHE_BASE+4u*cell;
    let plane=GeometricInterfacePlane(vec3f(state[at],state[at+1u],state[at+2u]),state[at+3u]);
    geometry=GeometricResidentInterface(plane,select(0u,1u,geometricInterfacePlaneValid(plane)));
  }else{geometry=geometricResidentSupportedInterface(cell,densityOffset);}
  if(geometry.valid==0u){return vec2f(0.0);}
  return vec2f(geometricInterfaceSignedDistance(geometry.plane,
    positionFine-cellCenter(cell)),1.0);
}

fn geometricResidentSignedDistance(cell:u32,positionFine:vec3f,
 densityOffset:u32,fallbackFine:f32)->f32{
  let support=geometricResidentDistanceSupport(cell,positionFine,densityOffset);
  if(support.y==0.0){return fallbackFine;}
  return support.x;
}
`;
