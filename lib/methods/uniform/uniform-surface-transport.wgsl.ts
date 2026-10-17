/** Current-state affine transport. Error is local phi distance, in metres;
 * it is not a bound on accumulated error against another simulation. */
export const UNIFORM_SURFACE_TRANSPORT_ERROR_H=0.02;
/** Nested fine/coarse MAC taps plus one-step RK stage support, in base cells. */
export const UNIFORM_SURFACE_VELOCITY_REACH_H=7;
/** rho,velocity lo/hi,flags (bit0 valid, bit1 canonical-open), then native-edge Lipschitz xyz. */
export const UNIFORM_SURFACE_SUPPORT_WORDS=11;
export const uniformSurfaceTransportWGSL=/* wgsl */`
fn umSurfaceTransport(c:f32,g:vec3f,rho:f32,lo:vec3f,hi:vec3f,h:vec3f,dt:f32,eps:f32,p:vec3f,width:f32)->vec4f{
 // The certified midpoint minimizes the worst component error over the
 // full RK velocity enclosure and requires no point velocity sampling.
 let u=0.5*lo+0.5*hi;let displacement=dt*u/h;
 // Catmull-Rom's 1D absolute weight sum is <=1.25, hence <=1.953125
 // in 3D. Current c contributes one residual; smoothing contributes at most
 // 32eps*rho, including gates turning on/off on an affine stencil.
 let velocityError=dt*max(abs(lo-u),abs(hi-u))/h;
 let positionError=8.0*1.1920929e-7*(abs(p)+vec3f(2.0*width+1.0));
 let otherError=32.0*eps*rho+dot(abs(g),velocityError)+dot(abs(g)+vec3f(6.0*rho),positionError);
 let oldError=2.953125*rho+otherError;
 let maximum=dt*max(abs(lo),abs(hi))/h;
 let magnitude=abs(c)+dot(abs(g),vec3f(2.0*width+1.0))+2.0*rho;
 // Canonical p anchors either native trilinear or clamped cubic. Their
 // residual change is <=6rho*sum(travel), including each incident cell.
 // Keep a nonshrinking evaluation allowance when this bound is selected.
 let newError=6.0*rho*dot(maximum,vec3f(1.0))+otherError+256.0*1.1920929e-7*magnitude;
 let error=min(oldError,newError);
 let budget=${UNIFORM_SURFACE_TRANSPORT_ERROR_H}*min(h.x,min(h.y,h.z));
 let finite=abs(c)<=3.402823e38&&all(abs(g)<=vec3f(3.402823e38))&&rho>=0.0&&rho<=3.402823e38;
 let accepted=finite&&dt>=0.0&&all(h>vec3f(0.0))&&all(lo<=u)&&all(u<=hi)&&all(dt*max(abs(lo),abs(hi))/h<=vec3f(0.5))&&error<=budget;
 return vec4f(c-dot(g,displacement),error,max(abs(displacement.x),max(abs(displacement.y),abs(displacement.z))),select(0.0,1.0,accepted));
}
// One positive-weight RK step. Integer next-float operations materialize
// outward displacement bounds before the coordinate additions, preserving
// their monotonic rounding instead of charging a whole-coordinate ULP.
struct UMTraceInterval {lo:vec3f,hi:vec3f,euler:vec3f}
fn umFloatDown(x:f32)->f32{if(x==0.0){return bitcast<f32>(0x80000001u);}return bitcast<f32>(bitcast<u32>(x)+select(0xffffffffu,1u,x<0.0));}
fn umFloatUp(x:f32)->f32{if(x==0.0){return bitcast<f32>(1u);}return bitcast<f32>(bitcast<u32>(x)+select(1u,0xffffffffu,x<0.0));}
fn umTraceEndpointInterval(foot:vec3f,shift:vec3f,lo:vec3f,hi:vec3f,u:vec3f,h:vec3f,dt:f32)->UMTraceInterval{
 let maximum=dt*max(abs(lo),abs(hi))/h;
 // Division, the weighted six-term sum, multiplication and /6 require
 // fewer than 10 epsilon*M absolute error; 32 also covers fused forms.
 let pad=32.0*1.1920929e-7*maximum+vec3f(1.17549435e-38);
 let lower=-dt*(hi/h)-pad;let upper=-dt*(lo/h)+pad;
 let dl=vec3f(umFloatDown(lower.x),umFloatDown(lower.y),umFloatDown(lower.z));
 let dh=vec3f(umFloatUp(upper.x),umFloatUp(upper.y),umFloatUp(upper.z));
 var result=UMTraceInterval(foot+dl,foot+dh,foot-dt*(u/h));
 if(any(shift!=vec3f(0.0))){result.lo+=shift;result.hi+=shift;result.euler+=shift;}
 return result;
}
// Native-edge slopes bound the reconstructed piecewise-trilinear field
// across cells; the arithmetic allowance also covers rounded hanging-node
// representation changes. Canonical p anchors the current field value.
fn umSurfaceNativeFarBound(c:f32,lipschitz:vec3f,h:vec3f,dt:f32,p:vec3f,width:f32,interval:UMTraceInterval)->vec4f{
 let maximum=max(abs(interval.lo-p),abs(interval.hi-p));let displacement=abs(interval.euler-p);
 let delta=max(abs(interval.lo-interval.euler),abs(interval.hi-interval.euler));
 let exact=all(maximum==vec3f(0.0))&&all(displacement==vec3f(0.0));
 // Every native stencil corner is at most width+.5 cells from p. Width1
 // canonical vertices have only fine incident cells; width4 covers both.
 let magnitude=abs(c)+dot(lipschitz,vec3f(width+0.5));
 let arithmetic=select(128.0*1.1920929e-7*magnitude,0.0,exact);
 let error=dot(lipschitz,delta)+arithmetic;
 let outside=abs(c)>=2.0*width*min(h.x,min(h.y,h.z))+dot(lipschitz,maximum)+arithmetic;
 let finite=abs(c)<=3.402823e38&&all(lipschitz>=vec3f(0.0))&&all(lipschitz<=vec3f(3.402823e38));
 let accepted=finite&&all(h>vec3f(0.0))&&dt>=0.0&&all(maximum<=vec3f(0.5))&&all(displacement<=vec3f(0.5))&&outside&&error<=${UNIFORM_SURFACE_TRANSPORT_ERROR_H}*min(h.x,min(h.y,h.z));
 return vec4f(error,max(maximum.x,max(maximum.y,maximum.z)),select(0.0,1.0,outside),select(0.0,1.0,accepted));
}
fn umSurfaceEndpointBound(c:f32,g:vec3f,rho:f32,h:vec3f,dt:f32,p:vec3f,width:f32,smoothing:f32,interval:UMTraceInterval,linear:bool)->vec4f{
 let maximum=max(abs(interval.lo-p),abs(interval.hi-p));let displacement=abs(interval.euler-p);
 let traceError=max(abs(interval.lo-interval.euler),abs(interval.hi-interval.euler));
 let exact=all(maximum==vec3f(0.0))&&all(displacement==vec3f(0.0));
 let magnitude=abs(c)+dot(abs(g),vec3f(2.0*width+1.0))+2.0*rho+abs(smoothing);
 // Absolute Catmull polynomial errors total <=28 epsilon per axis;
 // tensor propagation costs <132 epsilon, nested products/sums <32 more.
 // 256 epsilon also covers linear sampling and final smoothing addition.
 // Covers coefficient formation, nested cubic/linear products and sums,
 // clamping, and the final common smoothing addition. At the canonical
 // vertex both interpolants return exactly c, then add the same smoothing.
 let arithmetic=select(256.0*1.1920929e-7*magnitude,0.0,exact);
 var error=dot(abs(g),traceError)+rho*(6.0*dot(maximum,vec3f(1.0))+2.0*dot(displacement,vec3f(1.0)))+arithmetic;
 let lipschitz=abs(g)+vec3f(2.0*rho);
 let outside=abs(c)>=2.0*width*min(h.x,min(h.y,h.z))+dot(lipschitz,maximum)+arithmetic;
 if(linear){error=dot(lipschitz,traceError)+arithmetic;}
 let accepted=abs(c)<=3.402823e38&&rho>=0.0&&dt>=0.0&&all(h>vec3f(0.0))&&all(maximum<=vec3f(0.5))&&all(displacement<=vec3f(0.5))&&(!linear||outside)&&error<=${UNIFORM_SURFACE_TRANSPORT_ERROR_H}*min(h.x,min(h.y,h.z));
 return vec4f(error,max(maximum.x,max(maximum.y,maximum.z)),select(0.0,1.0,outside),select(0.0,1.0,accepted));
}
`;
