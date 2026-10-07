/** Shared exact tetrahedral volume curve for the global surface constraint. */
export const levelSetFillWGSL = /* wgsl */ `
// Adjacent pair, an outside value (not negative) after a negative one.
fn tetraOrder(a:f32,b:f32)->vec2f{return select(vec2f(a,b),vec2f(b,a),!(a<0.0)&&b<0.0);}
fn tetra(v:vec4f)->f32{
 // Stable partition, negatives first, by a fixed odd-even transposition
 // network: s holds the negatives then the outside values in their order,
 // in registers (no dynamically indexed private arrays).
 var s=v;var p=tetraOrder(s.x,s.y);s.x=p.x;s.y=p.y;p=tetraOrder(s.z,s.w);s.z=p.x;s.w=p.y;
 p=tetraOrder(s.y,s.z);s.y=p.x;s.z=p.y;
 p=tetraOrder(s.x,s.y);s.x=p.x;s.y=p.y;p=tetraOrder(s.z,s.w);s.z=p.x;s.w=p.y;
 p=tetraOrder(s.y,s.z);s.y=p.x;s.z=p.y;
 let count=dot(select(vec4u(0u),vec4u(1u),v<vec4f(0.0)),vec4u(1u));
 if(count==0u){return 0.0;}if(count==4u){return 1.0;}
 if(count==1u){return (-s.x/(s.y-s.x))*(-s.x/(s.z-s.x))*(-s.x/(s.w-s.x));}
 if(count==3u){return 1.0-(s.w/(s.w-s.x))*(s.w/(s.w-s.y))*(s.w/(s.w-s.z));}
 let a=-s.x/(s.z-s.x);let b=-s.x/(s.w-s.x);let c=-s.y/(s.z-s.y);let d=-s.y/(s.w-s.y);
 return clamp(a*b+b*c*(1.0-a)+c*d*(1.0-b),0.0,1.0);
}
fn fill(v:array<f32,8>)->f32{
 return (tetra(vec4f(v[0],v[1],v[3],v[7]))+tetra(vec4f(v[0],v[1],v[5],v[7]))
 +tetra(vec4f(v[0],v[2],v[3],v[7]))+tetra(vec4f(v[0],v[2],v[6],v[7]))
 +tetra(vec4f(v[0],v[4],v[5],v[7]))+tetra(vec4f(v[0],v[4],v[6],v[7])))/6.0;
}
`;
