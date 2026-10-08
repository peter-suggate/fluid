/** EXNB-inspired surface authority. Classification reads the advected level
 * set before particles can change it. position.w is 1+activity, or 3 for spray.
 * Particle positions and the spatial index remain unchanged in this pass. */
export const narrowBandActivityWGSL=/* wgsl */`
fn nbCoherentNeighbors(p:vec3f)->u32{
 let base=vec3i(floor(p));var count=0u;
 for(var z=-1;z<=1;z++){for(var y=-1;y<=1;y++){for(var x=-1;x<=1;x++){
  let c=base+vec3i(x,y,z);if(any(c<vec3i(0))||any(c>=vec3i(UM_D))){continue;}
  var link=atomicLoad(&bins[2u*cellIndex(c)]);
  while(link!=0u){let i=link-1u;let sample=nbSurfaceSample(i);link=links[i];
   if(sample.w<3.0&&distance(p,sample.xyz)<1.0){count++;if(count>=6u){return count;}}
  }
 }}}
 return count;
}
@compute @workgroup_size(64) fn prepareSurface(@builtin(global_invocation_id) gid:vec3u){
 for(var i=gid.x;i<min(atomicLoad(&state[1]),arrayLength(&particles));i+=65536u){
  let q=particles[i].position.xyz;let d=bandPhi(q);let supported=gridSupported(q);
  let wasSpray=particles[i].position.w>=3.0||particles[i].before.w==1.0;
  // A droplet must cross inside resolved liquid before re-entry. Sparse
  // escaping samples cannot manufacture their own support through union.
  var spray=wasSpray&&!(supported&&d < -0.25);
  if(!wasSpray&&(d>0.5||!supported)){spray=nbCoherentNeighbors(q)<6u;}
  if(spray){particles[i].position.w=3.0;particles[i].before.w=1.0;continue;}
  if(wasSpray){particles[i].before.w=2.0;}
  var heat=select(max(0.0,min(1.0,particles[i].position.w-1.0)-2.0*params.hDt.w),0.0,wasSpray);
  if(d > -1.5){
   let u=sampleVelocity(q);var meanU=vec3f(0);var meanPhi=0.0;
   for(var axis=0u;axis<3u;axis++){
    var e=vec3f(0);e[axis]=1.0;
    meanU+=sampleVelocity(clamp(q-e,vec3f(0.01),vec3f(UM_D)-0.01))+sampleVelocity(clamp(q+e,vec3f(0.01),vec3f(UM_D)-0.01));
    meanPhi+=bandPhi(clamp(q-e,vec3f(0),vec3f(UM_D)))+bandPhi(clamp(q+e,vec3f(0),vec3f(UM_D)));
   }
   let h=min(params.hDt.x,min(params.hDt.y,params.hDt.z));
   let speedScale=max(0.1,sqrt(abs(params.settings.z)*h));
   let motion=smoothstep(0.05,0.3,length(u-meanU/6.0)/speedScale);
   let shape=smoothstep(0.08,0.3,abs(d-meanPhi/6.0));
   heat=max(heat,max(motion,shape));
  }
  particles[i].position.w=1.0+heat;
 }
}
@compute @workgroup_size(64) fn publishSurfaceSamples(@builtin(global_invocation_id) gid:vec3u){
 for(var i=gid.x;i<min(atomicLoad(&state[1]),arrayLength(&particles));i+=65536u){
  links[arrayLength(&particles)+4u*i+3u]=bitcast<u32>(particles[i].position.w);
 }
}
// Calm, nearly planar patches can lose cell-scale wrinkles even at zero
// velocity. Symmetric taps preserve planes; sharp/thin geometry and boundary
// contacts are excluded. This is a bounded local filter, not a mass correction.
fn nbRelaxedBulk(q:vec3i,bulk:f32,activity:f32)->f32{
 if(abs(bulk)>1.5||activity>0.1||any(q<vec3i(1))||any(q>=vec3i(UM_D))){return bulk;}
 let p=vec3f(q);var sum=0.0;
 for(var axis=0u;axis<3u;axis++){
  var e=vec3f(0);e[axis]=1.0;sum+=bandPhi(p-e)+bandPhi(p+e);
 }
 let lap=sum/6.0-bulk;
 let quiet=1.0-smoothstep(0.04,0.12,abs(lap));
 return bulk+quiet*(1.0-exp(-8.0*params.hDt.w))*clamp(lap,-0.05,0.05);
}
`;
