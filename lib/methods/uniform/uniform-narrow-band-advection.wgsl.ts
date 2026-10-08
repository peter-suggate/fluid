/** The same third-order characteristic for the particle forward trace and
 * level-set backtrace; pressure retains its macro timestep. The velocity
 * field is frozen over that macro timestep. Callers measure travel in base
 * substeps of one h cell at the start speed (nbTraceSteps). A step spans two
 * of them where the field is straight along it, and takes three field
 * samples either way: the sample that decides is a stage of both. */
export const NARROW_BAND_TRACE_BEND=0.125;
/** Base substeps past which a trajectory is rejected: 128 cells of travel. */
export const NARROW_BAND_TRACE_LIMIT=128;
export function narrowBandTraceWGSL(sample:string):string{return /* wgsl */`
// Base substeps for a start speed in cells a second over the step dt.
fn nbTraceSteps(speed:vec3f,dt:f32)->u32{return max(1u,u32(ceil(dt*max(abs(speed.x),max(abs(speed.y),abs(speed.z))))));}
// One step from p, where k1 is the field in cells a second: the end, and in
// w the base substeps (signed duration \`base\`) it spanned, two where \`most\`
// allows. The field one base substep ahead is the midpoint stage of Kutta's
// rule over two and the end stage of Shu and Osher's over one. Two need the
// step to stay within two cells at the local speed and that sample to differ
// from k1 by less than an eighth of a cell of travel over the step.
fn nbTraceSpan(p:vec3f,base:f32,k1:vec3f,most:u32)->vec4f{
 let h=params.hDt.xyz;let low=vec3f(0);let high=vec3f(UM_D);
 let k2=${sample}(clamp(p+base*k1,low,high))/h;let turn=abs(k2-k1);
 let two=most>=2u&&2.0*abs(base)*max(abs(k1.x),max(abs(k1.y),abs(k1.z)))<=2.001
  &&2.0*abs(base)*max(turn.x,max(turn.y,turn.z))<=${NARROW_BAND_TRACE_BEND};
 let k3=${sample}(clamp(select(p+0.25*base*(k1+k2),p+2.0*base*(2.0*k2-k1),two),low,high))/h;
 return select(vec4f(p+base*(k1+k2+4.0*k3)/6.0,1.0),vec4f(p+base*(k1+4.0*k2+k3)/3.0,2.0),two);
}
`;}
