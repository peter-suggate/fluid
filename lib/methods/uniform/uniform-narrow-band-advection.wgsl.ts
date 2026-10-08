/** The same RK4 characteristic for the particle forward trace and level-set
 * backtrace. Callers subdivide travel at half an h cell; pressure retains its
 * macro timestep. The velocity field is frozen over that macro timestep. */
export function narrowBandTraceWGSL(sample:string):string{return /* wgsl */`
fn nbTraceStep(p:vec3f,dt:f32)->vec3f{return nbTraceFrom(p,dt,${sample}(p)/params.hDt.xyz);}
// k1: the field at p in cells a second, for a caller that has sampled it.
fn nbTraceFrom(p:vec3f,dt:f32,k1:vec3f)->vec3f{
 let h=params.hDt.xyz;
 let k2=${sample}(clamp(p+0.5*dt*k1,vec3f(0),vec3f(UM_D)))/h;
 let k3=${sample}(clamp(p+0.5*dt*k2,vec3f(0),vec3f(UM_D)))/h;
 let k4=${sample}(clamp(p+dt*k3,vec3f(0),vec3f(UM_D)))/h;
 return p+dt*(k1+2.0*k2+2.0*k3+k4)/6.0;
}
`;}
