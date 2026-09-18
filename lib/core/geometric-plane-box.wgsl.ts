/** Binding-free box integration shared by dense and adaptive geometric volume. */
export const geometricPlaneBoxWGSL = /* wgsl */ `
// Fraction of a centred axis-aligned box satisfying dot(normal,x) <= offset.
// The normal need not be unit length. Strictly positive widths are required.
fn geometricPlaneBoxFraction(normal:vec3f,offset:f32,widths:vec3f)->f32{
  let projected=abs(normal)*widths;
  let dominant=max(projected.x,max(projected.y,projected.z));
  if(dominant<=1e-20){return select(0.0,1.0,offset>=0.0);}
  var spans=vec3f(0.0);var dimensions=0u;
  for(var axis=0u;axis<3u;axis+=1u){
    let value=projected[axis]/dominant;
    if(value>=1e-6){spans[dimensions]=value;dimensions+=1u;}
  }
  let total=spans.x+spans.y+spans.z;
  let shifted=offset/dominant+0.5*total;
  if(shifted<=0.0){return 0.0;}
  if(shifted>=total){return 1.0;}
  // Complement symmetry keeps evaluation in the nearer tail and makes
  // opposite normals/offsets complementary without two large CDF evaluations.
  let complement=shifted>0.5*total;
  let x=select(shifted,total-shifted,complement);
  var fraction=0.0;
  if(dimensions==1u){fraction=x/spans.x;}
  else if(dimensions==2u){
    // Sort the widths so the middle branch avoids subtracting nearly equal
    // squares when one projected dimension is much smaller than the other.
    let a=min(spans.x,spans.y);let b=max(spans.x,spans.y);
    if(x<a){fraction=0.5*(x/a)*(x/b);}
    else{fraction=(x-0.5*a)/b;}
  }else{
    // Integrate the exact two-dimensional CDF over the third uniform span.
    // A stable piecewise quadratic/cubic form avoids the ill-conditioned
    // alternating sum when two projected widths are tiny.
    let a=min(spans.x,min(spans.y,spans.z));
    let c=max(spans.x,max(spans.y,spans.z));
    let b=max(min(spans.x,spans.y),min(max(spans.x,spans.y),spans.z));
    fraction=(geometricUniformSumPrimitive(x,a,b)
      -geometricUniformSumPrimitive(x-c,a,b))/c;
  }
  // Roundoff guard on the geometric CDF only; no state amount is modified.
  fraction=clamp(fraction,0.0,1.0);
  return select(fraction,1.0-fraction,complement);
}

// Integral from -infinity to x of the CDF of U[0,a]+U[0,b], 0<a<=b.
fn geometricUniformSumPrimitive(x:f32,a:f32,b:f32)->f32{
  if(x<=0.0){return 0.0;}
  if(x<a){return x*(x/a)*(x/b)/6.0;}
  if(x<=b){return (0.5*x*(x-a)+a*a/6.0)/b;}
  if(x<a+b){
    let tail=a+b-x;
    return x-0.5*(a+b)+tail*(tail/a)*(tail/b)/6.0;
  }
  return x-0.5*(a+b);
}

`;
