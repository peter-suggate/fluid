/**
 * Bounded convex clipping for a tetrahedron and up to eight linear half-spaces.
 * Independently implemented using polygon clipping and centroid tetrahedra.
 * Solid SDF >= 0 is open; fluid dot(normal, position) - offset <= 0 is liquid.
 * Results are volumes of the supplied piecewise-linear geometry, not an exact
 * integration of a curved underlying SDF. Caller coordinates share one length
 * unit and should be local to the cell to avoid world-origin cancellation.
 *
 * A tetrahedron clipped by eight planes has at most 12 faces, 20 vertices and 30
 * edges. Each resulting face has at most 11 vertices. Storage allows
 * 16 faces, 16 vertices per face and 32 cap candidates.
 * Geometric equality uses 1e-6 times the tetrahedron's longest edge; only cap
 * deduplication uses this tolerance. Half-space membership uses exact f32 sign.
 * Finite coordinates/SDF values and positive box widths are caller preconditions.
 */
export const geometricCutCellWGSL = /* wgsl */ `
struct GeometricCutVertex { position:vec3f, solid:f32, }
struct GeometricCutPolyhedron {
  vertices:array<GeometricCutVertex,256>,
  counts:array<u32,16>,
  faces:u32,
}
struct GeometricCutVolumes { capacity:f32, liquid:f32, }

fn geometricTetrahedronVolume(a:vec3f,b:vec3f,c:vec3f,d:vec3f)->f32{
  return abs(dot(b-a,cross(c-a,d-a)))/6.0;
}
fn geometricCutSigned(v:GeometricCutVertex,normal:vec3f,offset:f32,solid:bool)->f32{
  return select(offset-dot(normal,v.position),v.solid,solid);
}

fn geometricClipPolyhedron(input:GeometricCutPolyhedron,normal:vec3f,
 offset:f32,solid:bool,tolerance:f32)->GeometricCutPolyhedron{
  var output:GeometricCutPolyhedron;
  var cap:array<GeometricCutVertex,32>;var capCount=0u;
  let tolerance2=tolerance*tolerance;
  for(var face=0u;face<input.faces;face+=1u){
    let count=input.counts[face];if(count<3u){continue;}
    var polygon:array<GeometricCutVertex,16>;var written=0u;
    var previous=input.vertices[face*16u+count-1u];
    var previousSigned=geometricCutSigned(previous,normal,offset,solid);
    for(var index=0u;index<count;index+=1u){
      let current=input.vertices[face*16u+index];
      let currentSigned=geometricCutSigned(current,normal,offset,solid);
      if((previousSigned>=0.0)!=(currentSigned>=0.0)){
        let t=clamp(previousSigned/(previousSigned-currentSigned),0.0,1.0);
        let point=GeometricCutVertex(mix(previous.position,current.position,t),
          mix(previous.solid,current.solid,t));
        if(written<16u){polygon[written]=point;written+=1u;}
        var unique=true;
        for(var prior=0u;prior<capCount;prior+=1u){
          let delta=cap[prior].position-point.position;
          if(dot(delta,delta)<=tolerance2){unique=false;}
        }
        if(unique&&capCount<32u){cap[capCount]=point;capCount+=1u;}
      }
      if(currentSigned>=0.0&&written<16u){polygon[written]=current;written+=1u;}
      previous=current;previousSigned=currentSigned;
    }
    if(written>=3u&&output.faces<16u){
      let destination=output.faces*16u;
      for(var index=0u;index<written;index+=1u){output.vertices[destination+index]=polygon[index];}
      output.counts[output.faces]=written;output.faces+=1u;
    }
  }
  if(capCount>=3u&&capCount<=16u&&output.faces<16u){
    var center=vec3f(0.0);
    for(var index=0u;index<capCount;index+=1u){center+=cap[index].position;}
    center/=f32(capCount);
    let axis=cap[0].position-center;
    var perpendicular=vec3f(0.0);var best=0.0;
    for(var index=1u;index<capCount;index+=1u){
      let candidate=cross(axis,cap[index].position-center);
      let magnitude=dot(candidate,candidate);
      if(magnitude>best){best=magnitude;perpendicular=candidate;}
    }
    if(best>0.0){
      let u=normalize(axis);let v=normalize(cross(perpendicular,u));
      var angles:array<f32,32>;
      for(var index=0u;index<capCount;index+=1u){
        let delta=cap[index].position-center;
        angles[index]=atan2(dot(delta,v),dot(delta,u));
      }
      for(var index=1u;index<capCount;index+=1u){
        let point=cap[index];let angle=angles[index];var destination=index;
        loop{
          if(destination==0u){break;}
          if(angles[destination-1u]<=angle){break;}
          cap[destination]=cap[destination-1u];angles[destination]=angles[destination-1u];
          destination-=1u;
        }
        cap[destination]=point;angles[destination]=angle;
      }
      let base=output.faces*16u;
      for(var index=0u;index<capCount;index+=1u){output.vertices[base+index]=cap[index];}
      output.counts[output.faces]=capCount;output.faces+=1u;
    }
  }
  return output;
}

fn geometricCutPolyhedronVolume(poly:GeometricCutPolyhedron)->f32{
  var center=vec3f(0.0);var samples=0u;
  for(var face=0u;face<poly.faces;face+=1u){
    for(var index=0u;index<poly.counts[face];index+=1u){
      center+=poly.vertices[face*16u+index].position;samples+=1u;
    }
  }
  if(samples==0u){return 0.0;}center/=f32(samples);
  var volume=0.0;
  for(var face=0u;face<poly.faces;face+=1u){
    let a=poly.vertices[face*16u].position;
    for(var index=1u;index+1u<poly.counts[face];index+=1u){
      volume+=geometricTetrahedronVolume(center,a,
        poly.vertices[face*16u+index].position,poly.vertices[face*16u+index+1u].position);
    }
  }
  return volume;
}

// clipAxes chooses bounded coordinates. Bounds use the original tetrahedron's
// coordinate frame. Interpolate solid SDF only on original tetrahedron edges;
// never resample a new prism's corners, which would change its discretization.
fn geometricClippedTetrahedronBoundsVolumes(positions:array<vec3f,4>,
 solidSDF:vec4f,fluidNormal:vec3f,fluidOffset:f32,
 lower:vec3f,upper:vec3f,clipAxes:vec3<bool>)->GeometricCutVolumes{
  for(var axis=0u;axis<3u;axis+=1u){
    if(clipAxes[axis]&&upper[axis]<=lower[axis]){return GeometricCutVolumes(0.0,0.0);}
  }
  let whole=geometricTetrahedronVolume(positions[0],positions[1],positions[2],positions[3]);
  if(whole<=0.0||(all(solidSDF<=vec4f(0.0))&&any(solidSDF<vec4f(0.0)))){return GeometricCutVolumes(0.0,0.0);}
  var longest2=0.0;
  for(var a=0u;a<4u;a+=1u){for(var b=a+1u;b<4u;b+=1u){
    let edge=positions[a]-positions[b];longest2=max(longest2,dot(edge,edge));
  }}
  let tolerance=sqrt(longest2)*1e-6;
  var poly:GeometricCutPolyhedron;poly.faces=4u;
  let faces=array<vec3u,4>(vec3u(0,1,2),vec3u(0,3,1),vec3u(0,2,3),vec3u(1,3,2));
  for(var face=0u;face<4u;face+=1u){
    poly.counts[face]=3u;
    for(var corner=0u;corner<3u;corner+=1u){let index=faces[face][corner];
      poly.vertices[face*16u+corner]=GeometricCutVertex(positions[index],solidSDF[index]);
    }
  }
  var capacity=whole;
  if(any(solidSDF<vec4f(0.0))){
    poly=geometricClipPolyhedron(poly,vec3f(0.0),0.0,true,tolerance);
    capacity=min(whole,geometricCutPolyhedronVolume(poly));
  }
  for(var dimension=0u;dimension<3u;dimension+=1u){
    if(!clipAxes[dimension]){continue;}
    var axis=vec3f(0.0);axis[dimension]=1.0;
    poly=geometricClipPolyhedron(poly,axis,upper[dimension],false,tolerance);
    poly=geometricClipPolyhedron(poly,-axis,-lower[dimension],false,tolerance);
  }
  if(any(clipAxes)){capacity=min(capacity,geometricCutPolyhedronVolume(poly));}
  var allLiquid=true;var allAir=true;
  for(var face=0u;face<poly.faces;face+=1u){
    for(var index=0u;index<poly.counts[face];index+=1u){
      let value=dot(fluidNormal,poly.vertices[face*16u+index].position)-fluidOffset;
      allLiquid=allLiquid&&value<=0.0;allAir=allAir&&value>=0.0;
    }
  }
  if(allLiquid){return GeometricCutVolumes(capacity,capacity);}
  if(allAir){return GeometricCutVolumes(capacity,0.0);}
  let liquid=geometricCutPolyhedronVolume(geometricClipPolyhedron(poly,
    fluidNormal,fluidOffset,false,tolerance));
  // Bounds correct accumulated f32 geometry roundoff, never solver state.
  return GeometricCutVolumes(capacity,min(capacity,liquid));
}

fn geometricClippedTetrahedronSlabVolumes(positions:array<vec3f,4>,
 solidSDF:vec4f,fluidNormal:vec3f,fluidOffset:f32,
 slabAxis:u32,slabLower:f32,slabUpper:f32)->GeometricCutVolumes{
  return geometricClippedTetrahedronBoundsVolumes(positions,solidSDF,fluidNormal,
    fluidOffset,vec3f(slabLower),vec3f(slabUpper),vec3u(0u,1u,2u)==vec3u(slabAxis));
}

fn geometricClippedTetrahedronVolumes(positions:array<vec3f,4>,
 solidSDF:vec4f,fluidNormal:vec3f,fluidOffset:f32)->GeometricCutVolumes{
  return geometricClippedTetrahedronSlabVolumes(positions,solidSDF,
    fluidNormal,fluidOffset,3u,0.0,0.0);
}

// Corner index x+2*y+4*z. Every translated cell uses the same 0--7 body
// diagonal; induced triangulation agrees on shared cube faces.
fn geometricClippedBoxBoundsVolumes(widths:vec3f,solidSDF:array<f32,8>,
 fluidNormal:vec3f,fluidOffset:f32,lower:vec3f,
 upper:vec3f,clipAxes:vec3<bool>)->GeometricCutVolumes{
  let tetrahedra=array<vec4u,6>(vec4u(0,1,3,7),vec4u(0,3,2,7),
    vec4u(0,2,6,7),vec4u(0,6,4,7),vec4u(0,4,5,7),vec4u(0,5,1,7));
  var result=GeometricCutVolumes(0.0,0.0);
  for(var tetra=0u;tetra<6u;tetra+=1u){
    var positions:array<vec3f,4>;var values=vec4f(0.0);
    for(var corner=0u;corner<4u;corner+=1u){let index=tetrahedra[tetra][corner];
      positions[corner]=(vec3f(f32(index&1u),f32((index>>1u)&1u),f32((index>>2u)&1u))-0.5)*widths;
      values[corner]=solidSDF[index];
    }
    let volume=geometricClippedTetrahedronBoundsVolumes(positions,values,fluidNormal,fluidOffset,
      lower,upper,clipAxes);
    result.capacity+=volume.capacity;result.liquid+=volume.liquid;
  }
  return result;
}
// Axis-aligned prism clipping retains the original tetrahedral solid field,
// including tangential bounds of an adaptive coarse/fine physical subface.
fn geometricClippedBoxPrismVolumes(widths:vec3f,solidSDF:array<f32,8>,
 fluidNormal:vec3f,fluidOffset:f32,lower:vec3f,upper:vec3f)->GeometricCutVolumes{
  return geometricClippedBoxBoundsVolumes(widths,solidSDF,fluidNormal,fluidOffset,
    lower,upper,vec3<bool>(true));
}

fn geometricClippedBoxSlabVolumes(widths:vec3f,solidSDF:array<f32,8>,
 fluidNormal:vec3f,fluidOffset:f32,slabAxis:u32,
 slabLower:f32,slabUpper:f32)->GeometricCutVolumes{
  return geometricClippedBoxBoundsVolumes(widths,solidSDF,fluidNormal,fluidOffset,
    vec3f(slabLower),vec3f(slabUpper),vec3u(0u,1u,2u)==vec3u(slabAxis));
}

fn geometricClippedBoxVolumes(widths:vec3f,solidSDF:array<f32,8>,
 fluidNormal:vec3f,fluidOffset:f32)->GeometricCutVolumes{
  return geometricClippedBoxSlabVolumes(widths,solidSDF,fluidNormal,fluidOffset,3u,0.0,0.0);
}
`;
