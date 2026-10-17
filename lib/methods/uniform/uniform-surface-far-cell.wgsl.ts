/** Current-cell trilinear error bound for replacing a far-band full trace. */
export const uniformSurfaceFarCellWGSL=/* wgsl */`
fn umSurfaceFarCellBound(center:f32,ownerWidth:u32,h:vec3f,origin:vec3u,cellWidth:u32,corners:array<f32,8>,interval:UMTraceInterval)->vec2f{
 if((ownerWidth!=1u&&ownerWidth!=4u)||(cellWidth!=1u&&cellWidth!=4u)||!all(h>vec3f(0.0))){return vec2f(0.0);}
 let lower=vec3f(origin);let upper=lower+vec3f(f32(cellWidth));
 // Strict upper containment keeps every possible endpoint in the very
 // same topology tile/cell used by the current Euler sample.
 if(!all(interval.lo>=lower)||!all(interval.hi<upper)||!all(interval.lo<=interval.hi)||!all(interval.euler>=lower)||!all(interval.euler<upper)){return vec2f(0.0);}
 var minimum=3.402823e38;var maximum=-3.402823e38;var magnitude=0.0;
 var lipschitz=vec3f(0.0);
 for(var k=0u;k<8u;k++){
  let v=corners[k];if(!(abs(v)<=3.402823e38)){return vec2f(0.0);}
  minimum=min(minimum,v);maximum=max(maximum,v);magnitude=max(magnitude,abs(v));
  for(var axis=0u;axis<3u;axis++){
   let bit=1u<<axis;if((k&bit)==0u){
    // One next-float makes the rounded subtraction an outward bound on
    // the exact difference between the stored f32 corner values.
    let edge=umFloatUp(abs(corners[k|bit]-v))/f32(cellWidth);
    lipschitz[axis]=max(lipschitz[axis],edge);
   }
  }
 }
 // Coordinate subtraction is exact for points in this integer-aligned
 // cell (Sterbenz); division by widths1/4 is exact. A trilinear weighted
 // term has three multiplications; the D4 sum adds three levels. 64eps
 // bounds each of two samplers, and also the remaining error arithmetic.
 let arithmetic=128.0*1.1920929e-7*magnitude;
 let band=2.0*f32(ownerWidth)*min(h.x,min(h.y,h.z));
 let far=(minimum>=band+arithmetic)||(maximum<=-band-arithmetic);
 let delta0=max(abs(interval.lo-interval.euler),abs(interval.hi-interval.euler));
 let delta=vec3f(umFloatUp(delta0.x),umFloatUp(delta0.y),umFloatUp(delta0.z));
 let error=dot(lipschitz,delta)+arithmetic;
 let accepted=abs(center)>=band&&abs(center)<=3.402823e38&&far&&error<=0.02*min(h.x,min(h.y,h.z));
 return vec2f(error,select(0.0,1.0,accepted));
}
`;
