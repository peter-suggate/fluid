/** The same third-order characteristic for the particle forward trace and
 * level-set backtrace; pressure retains its macro timestep. The velocity
 * field is frozen over that macro timestep. Callers measure travel in base
 * substeps of one h cell at the start speed (nbTraceSteps). A step spans two
 * of them where the field is straight along it, and takes three field
 * samples either way: the sample that decides is a stage of both. A stage is
 * handed to the field where it falls: the field's sample bounds its own taps. */
export const NARROW_BAND_TRACE_BEND=0.125;
/** Base substeps past which a trajectory is rejected: 128 cells of travel. */
export const NARROW_BAND_TRACE_LIMIT=128;
/** traceSubsteps "gradient": the field's turn a substep may span, dt|grad u|. */
export const NARROW_BAND_TRACE_TURN=0.25;
/** Substeps of a trace: "gradient" (default) as many as keep each substep's
 * dt|grad u| within NARROW_BAND_TRACE_TURN, "travel" one a cell of travel. */
export type NarrowBandTraceSubsteps="gradient"|"travel";
export function narrowBandTraceWGSL(sample:string,substeps:NarrowBandTraceSubsteps="travel"):string{return /* wgsl */`
// Base substeps for a start speed in cells a second over the step dt.
fn nbTraceSteps(speed:vec3f,dt:f32)->u32{return max(1u,u32(ceil(dt*max(abs(speed.x),max(abs(speed.y),abs(speed.z))))));}
// traceSubsteps: with gradient substeps a trace first tries the whole step
// (the travel substeps still bound it and reject a runaway trajectory). Its
// second stage, a whole step ahead, measures the field's turn over that
// span against its speed over at least a cell, dt|grad u|; the trace then
// restarts in as many substeps as keep that within NB_TRACE_TURN each, never
// more than the travel substeps, and never fewer than keep each within 32
// cells of travel: a span then crosses at most one page face an axis, as the
// page-face stops of its callers assume. The probe is that stage's own sample.
const NB_TRACE_GRADIENT:bool=${substeps==="gradient"};
const NB_TRACE_TURN:f32=${NARROW_BAND_TRACE_TURN};
fn nbTraceGradientSteps(k1:vec3f,k2:vec3f,dt:f32,travel:u32)->u32{
 let turn=abs(k2-k1);let speed=max(max(abs(k1.x),max(abs(k1.y),abs(k1.z))),1.0/abs(dt));
 let n=ceil(max(turn.x,max(turn.y,turn.z))/(speed*NB_TRACE_TURN));
 return select(travel,clamp(u32(n),max(1u,(travel+31u)/32u),travel),n<=f32(travel));
}
// One step from p, where k1 is the field in cells a second: the end, and in
// w the base substeps (signed duration \`base\`) it spanned, two where \`most\`
// allows. The field one base substep ahead is the midpoint stage of Kutta's
// rule over two and the end stage of Shu and Osher's over one. Two need the
// step to stay within two cells at the local speed and that sample to differ
// from k1 by less than an eighth of a cell of travel over the step.
// Its parts, for a caller that takes every sample of a trace at one call
// site (Metal inlines each call of the field: two calls compile it twice):
// whether the step spans two, the third stage's point, and the end.
fn nbTraceTwo(base:f32,k1:vec3f,k2:vec3f,most:u32)->bool{
 let turn=abs(k2-k1);
 return most>=2u&&2.0*abs(base)*max(abs(k1.x),max(abs(k1.y),abs(k1.z)))<=2.001
  &&2.0*abs(base)*max(turn.x,max(turn.y,turn.z))<=${NARROW_BAND_TRACE_BEND};
}
fn nbTraceThird(p:vec3f,base:f32,k1:vec3f,k2:vec3f,two:bool)->vec3f{return select(p+0.25*base*(k1+k2),p+2.0*base*(2.0*k2-k1),two);}
fn nbTraceEnd(p:vec3f,base:f32,k1:vec3f,k2:vec3f,k3:vec3f,two:bool)->vec4f{
 return select(vec4f(p+base*(k1+k2+4.0*k3)/6.0,1.0),vec4f(p+base*(k1+4.0*k2+k3)/3.0,2.0),two);
}
// The second and third stages in one loop: one call of the field.
fn nbTraceSpan(p:vec3f,base:f32,k1:vec3f,most:u32)->vec4f{
 let h=params.hDt.xyz;
 var x=p+base*k1;var k2=vec3f(0.0);var two=false;
 for(var stage=0u;stage<2u;stage++){
  let k=${sample}(x)/h;
  if(stage==1u){return nbTraceEnd(p,base,k1,k2,k,two);}
  k2=k;two=nbTraceTwo(base,k1,k2,most);x=nbTraceThird(p,base,k1,k2,two);
 }
 return vec4f(p,1.0);
}
`;}
