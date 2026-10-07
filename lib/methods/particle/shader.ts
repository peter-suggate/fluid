/** Quadratic B-spline APIC on staggered faces: D = diag(h² / 4).
 * Histogram-sorted contiguous bins avoid pointer chasing, floating-point
 * atomics and fixed bucket capacities.
 * Particle positions and the MAC grid use tank-local metres.
 */
export const APIC_SHADER = /* wgsl */ `
struct Params { dims:vec4u, hdt:vec4f, gravityNu:vec4f, material:vec4f, solve:vec4f }
struct Particle { position:vec4f, velocity:vec4f, cx:vec4f, cy:vec4f, cz:vec4f }
struct Settings { count:u32, radius:f32, support:f32, initialVolume:f32, transferMode:u32, flipRatio:f32, padding:vec2f }
@group(0) @binding(0) var<uniform> p:Params;
@group(0) @binding(1) var<storage,read_write> particles:array<Particle>;
@group(0) @binding(2) var<storage,read_write> heads:array<atomic<u32>>;
@group(0) @binding(3) var<storage,read> offsets:array<u32>;
@group(0) @binding(4) var<storage,read> solids:array<u32>;
@group(0) @binding(5) var<storage,read> grid:array<vec4f>;
@group(0) @binding(6) var<storage,read_write> transferred:array<vec4f>;
@group(0) @binding(7) var<storage,read_write> phi:array<f32>;
@group(0) @binding(8) var<storage,read_write> scalars:array<f32>;
@group(0) @binding(9) var<storage,read_write> status:array<atomic<u32>>;
@group(0) @binding(10) var<uniform> settings:Settings;
@group(0) @binding(11) var<storage,read_write> faceVolumes:array<vec4f>;
@group(0) @binding(12) var<storage,read_write> cursors:array<atomic<u32>>;
@group(0) @binding(13) var<storage,read_write> ordered:array<Particle>;
struct LedgerPartial { totals:vec4f, bound:vec4f }
@group(0) @binding(14) var<storage,read_write> ledgerParts:array<LedgerPartial>;
@group(0) @binding(15) var<storage,read_write> oldGrid:array<vec4f>;
fn flat(id:vec3u)->u32{return id.x+id.y*65535u*64u;}
fn cells()->u32{return p.dims.x*p.dims.y*p.dims.z;}
fn coordinate(i:u32,d:vec3u)->vec3i{return vec3i(vec3u(i%d.x,(i/d.x)%d.y,i/(d.x*d.y)));}
fn inside(q:vec3i)->bool{return all(q>=vec3i(0))&&all(q<vec3i(p.dims.xyz));}
fn index(q:vec3i)->u32{return u32(q.x)+p.dims.x*(u32(q.y)+p.dims.y*u32(q.z));}
fn axis(a:u32)->vec3i{var e=vec3i(0);e[a]=1;return e;}
fn closed(q:vec3i)->bool{
 if(!inside(q)){return !(p.material.w>0.0&&q.y>=i32(p.dims.y)&&q.x>=0&&q.z>=0&&q.x<i32(p.dims.x)&&q.z<i32(p.dims.z));}
 return solids[index(q)]!=0u;
}
fn stepActive()->bool{return scalars[12]==0.0&&scalars[21]>0.0;}
fn weight(x:f32)->f32{let a=abs(x);if(a<0.5){return 0.75-a*a;}if(a<1.5){return 0.5*(1.5-a)*(1.5-a);}return 0.0;}
fn kernel(d:vec3f)->f32{return weight(d.x)*weight(d.y)*weight(d.z);}
@compute @workgroup_size(64) fn clearBins(@builtin(global_invocation_id) id:vec3u){let i=flat(id);if(i<cells()&&stepActive()){atomicStore(&heads[i],0u);atomicStore(&cursors[i],0u);}}
@compute @workgroup_size(64) fn binParticles(@builtin(global_invocation_id) id:vec3u){
 let i=flat(id);if(i>=settings.count||!stepActive()||particles[i].position.w<=0.0){return;}
 let x=particles[i].position.xyz;
 if(!all(abs(x)<vec3f(1e20))){atomicStore(&status[0],5u);return;}
 let q=vec3i(floor(x/p.hdt.xyz));if(!inside(q)){atomicStore(&status[0],6u);return;}
 atomicAdd(&heads[index(q)],1u);
}
@compute @workgroup_size(64) fn scatterParticles(@builtin(global_invocation_id) id:vec3u){
 let i=flat(id);if(i>=settings.count||!stepActive()||particles[i].position.w<=0.0){return;}
 let q=vec3i(floor(particles[i].position.xyz/p.hdt.xyz));if(!inside(q)){return;}
 let bin=index(q);let slot=offsets[bin]+atomicAdd(&cursors[bin],1u);ordered[slot]=particles[i];
}
@compute @workgroup_size(64) fn transfer(@builtin(global_invocation_id) id:vec3u){
 let i=flat(id);if(i>=cells()||!stepActive()){return;}let q=coordinate(i,p.dims.xyz);
 var momentum=vec3f(0);var mass=vec3f(0);
 let centre=(vec3f(q)+vec3f(0.5))*p.hdt.xyz;let normal=(vec3f(q)+vec3f(1))*p.hdt.xyz;
 // All three staggered supports lie within these 4³ integer bins.
 for(var z=-1;z<=2;z++){for(var y=-1;y<=2;y++){for(var x=-1;x<=2;x++){
  let bin=q+vec3i(x,y,z);if(!inside(bin)){continue;}let b=index(bin);let end=offsets[b]+atomicLoad(&heads[b]);
  for(var j=offsets[b];j<end;j++){let particle=ordered[j];
   // Each face uses one normal coordinate and two cell-centre coordinates.
   // Six separable weights reproduce the same three quadratic stencils,
   // retaining the original face positions and x*y*z multiplication order.
   let d=centre-particle.position.xyz;let f=normal-particle.position.xyz;
   let dc=d/p.hdt.xyz;let df=f/p.hdt.xyz;
   let wc=vec3f(weight(dc.x),weight(dc.y),weight(dc.z));
   let wf=vec3f(weight(df.x),weight(df.y),weight(df.z));
   let w=vec3f(wf.x*wc.y*wc.z,wc.x*wf.y*wc.z,wc.x*wc.y*wf.z)*particle.position.w;
   var v=particle.velocity.xyz;
   if(settings.transferMode==0u){v+=vec3f(dot(particle.cx.xyz,vec3f(f.x,d.y,d.z)),
    dot(particle.cy.xyz,vec3f(d.x,f.y,d.z)),dot(particle.cz.xyz,vec3f(d.x,d.y,f.z)));}
   momentum+=w*v;mass+=w;
  }
 }}}
 var v=momentum/max(mass,vec3f(1e-30));
 for(var a=0u;a<3u;a++){if(closed(q)||closed(q+axis(a))){v[a]=0.0;}}
 transferred[i]=vec4f(v,0);
 // Capture before forces/projection overwrite transferred. Each grid lane owns
 // its snapshot, so FLIP needs neither a separate copy pass nor float atomics.
 if(settings.transferMode==2u){oldGrid[i]=vec4f(v,0);}
 faceVolumes[i]=vec4f(mass,0);
}
fn face(q:vec3i,a:u32,previous:bool)->f32{
 var r=q;for(var b=0u;b<3u;b++){if(b!=a){r[b]=clamp(r[b],0,i32(p.dims[b])-1);}}
 if(!inside(r)||closed(r)||closed(r+axis(a))){return 0.0;}
 if(previous){return oldGrid[index(r)][a];}return grid[index(r)][a];
}
// Value and APIC affine row share exactly the same quadratic stencil.
fn sample(x:vec3f,a:u32,previous:bool)->vec4f{
 var offset=vec3f(0.5);offset[a]=1.0;let g=x/p.hdt.xyz-offset;let base=vec3i(floor(g-vec3f(0.5)));
 var value=0.0;var c=vec3f(0);
 for(var z=0;z<3;z++){for(var y=0;y<3;y++){for(var k=0;k<3;k++){
  let q=base+vec3i(k,y,z);let d=(vec3f(q)-g)*p.hdt.xyz;let w=kernel(d/p.hdt.xyz);let v=face(q,a,previous);
  value+=w*v;if(settings.transferMode==0u){c+=4.0*w*v*d/(p.hdt.xyz*p.hdt.xyz);}
 }}}
 return vec4f(c,value);
}
fn velocity(x:vec3f)->vec3f{return vec3f(sample(x,0u,false).w,sample(x,1u,false).w,sample(x,2u,false).w);}
@compute @workgroup_size(64) fn moveParticles(@builtin(global_invocation_id) id:vec3u){
 let i=flat(id);if(i>=settings.count||!stepActive()||scalars[20]==0.0){return;}
 var particle=particles[i];if(particle.position.w<=0.0){return;}
 let original=particle.position.xyz;let cx=sample(original,0u,false);let cy=sample(original,1u,false);let cz=sample(original,2u,false);
 var v=vec3f(cx.w,cy.w,cz.w);let dt=scalars[16];let travel=dt*velocity(original+0.5*dt*v);
 // Advection follows the projected grid in every mode. Only the carried
 // particle velocity blends with FLIP's increment from the pre-force grid.
 if(settings.transferMode==2u){
  let before=vec3f(sample(original,0u,true).w,sample(original,1u,true).w,sample(original,2u,true).w);
  v=mix(v,particle.velocity.xyz+(v-before),settings.flipRatio);
 }
 if(!all(abs(travel)<vec3f(1e20))||!all(abs(v)<vec3f(1e20))){atomicStore(&status[0],5u);return;}
 let steps=u32(ceil(4.0*max(abs(travel.x/p.hdt.x),max(abs(travel.y/p.hdt.y),abs(travel.z/p.hdt.z)))))+1u;
 if(steps>64u){atomicStore(&status[0],6u);return;}
 var position=original;var hit=vec3<bool>(false);let extent=vec3f(p.dims.xyz)*p.hdt.xyz;
 for(var step=0u;step<steps;step++){
  for(var a=0u;a<3u;a++){
   var trial=position;trial[a]+=travel[a]/f32(steps);let low=0.0001*p.hdt[a];let high=extent[a]-low;
   if(trial[a]<low){trial[a]=low;hit[a]=true;}
   if(trial[a]>high&&!(a==1u&&p.material.w>0.0)){trial[a]=high;hit[a]=true;}
   if(closed(vec3i(floor(trial/p.hdt.xyz)))){hit[a]=true;}else{position=trial;}
  }
 }
 if(position.y>=extent.y&&p.material.w>0.0){particle.position.w=0.0;atomicAdd(&status[1],1u);}
 particle.position=vec4f(position,particle.position.w);particle.cx=vec4f(cx.xyz,0);particle.cy=vec4f(cy.xyz,0);particle.cz=vec4f(cz.xyz,0);
 if(any(hit)){
  v=select(v,vec3f(0),hit);particle.cx=vec4f(0);particle.cy=vec4f(0);particle.cz=vec4f(0);
  if(p.solve.w>0.0){v=vec3f(0);}
 }
 particle.velocity=vec4f(v,0);particles[i]=particle;
}
fn surfaceWeight(d:vec3f)->f32{let t=max(0.0,1.0-dot(d,d)/(settings.support*settings.support));return t*t*t;}
@compute @workgroup_size(64) fn surface(@builtin(global_invocation_id) id:vec3u){
 let i=flat(id);let dims=p.dims.xyz+vec3u(1);if(i>=dims.x*dims.y*dims.z){return;}
 // Initialization explicitly enables this hook. Failed frames retain their prior publication.
 if(!stepActive()){return;}let q=coordinate(i,dims);let x=vec3f(q);var sum=vec3f(0);var total=0.0;
 // Wall eligibility depends only on the vertex, not the gathered particle.
 // Preserve the closed-wall mirror stencil and skip all mirror work in the
 // interior; reflected subsets remain in their original ascending order.
 var wallMask=0u;var mirrorOrigin=vec3f(0);
 for(var a=0u;a<3u;a++){
  if(x[a]<settings.support){wallMask|=1u<<a;}
  else if(x[a]>f32(p.dims[a])-settings.support&&!(a==1u&&p.material.w>0.0)){
   wallMask|=1u<<a;mirrorOrigin[a]=2.0*f32(p.dims[a]);
  }
 }
 for(var z=-2;z<=1;z++){for(var y=-2;y<=1;y++){for(var k=-2;k<=1;k++){
  let bin=q+vec3i(k,y,z);if(!inside(bin)){continue;}let b=index(bin);let end=offsets[b]+atomicLoad(&heads[b]);
  for(var j=offsets[b];j<end;j++){let particle=ordered[j];let centre=particle.position.xyz/p.hdt.xyz;
   let w=surfaceWeight(x-centre)*particle.position.w;sum+=w*centre;total+=w;
   if(wallMask!=0u){let mirror=mirrorOrigin-centre;
    for(var mask=1u;mask<8u;mask++){
     if((mask&wallMask)!=mask){continue;}
     let reflected=vec3<bool>((mask&1u)!=0u,(mask&2u)!=0u,(mask&4u)!=0u);
     let c=select(centre,mirror,reflected);let w=surfaceWeight(x-c)*particle.position.w;sum+=w*c;total+=w;
    }
   }
  }
 }}}
 let h=min(p.hdt.x,min(p.hdt.y,p.hdt.z));var value=3.0*h;
 if(total>1e-30){value=(length(x-sum/total)-settings.radius)*h;}
 phi[i]=value;
}
// Four particles per lane, thousands of independent groups, then one small
// reduction over group receipts. The affine maximum remains mandatory per step.
var<workgroup> values:array<vec4f,64>;
var<workgroup> bounds:array<f32,64>;
var<workgroup> ledgerRunning:u32;
fn ledgerActive(lane:u32)->bool{
 if(lane==0u){ledgerRunning=select(0u,1u,stepActive());}workgroupBarrier();
 return workgroupUniformLoad(&ledgerRunning)!=0u;
}
fn foldLedger(lane:u32,v:vec4f,bound:f32){
 values[lane]=v;bounds[lane]=bound;workgroupBarrier();
 for(var stride=32u;stride>0u;stride/=2u){if(lane<stride){let a=values[lane];let b=values[lane+stride];
  values[lane]=vec4f(a.x+b.x,max(a.y,b.y),a.z+b.z,a.w+b.w);bounds[lane]=max(bounds[lane],bounds[lane+stride]);}workgroupBarrier();}
}
@compute @workgroup_size(64) fn ledgerPartial(@builtin(local_invocation_index) lane:u32,@builtin(workgroup_id) group:vec3u){
 if(!ledgerActive(lane)){return;}let block=group.x+65535u*group.y;
 var v=vec4f(0);var affineBound=0.0;
 for(var k=0u;k<4u;k++){let i=block*256u+lane+k*64u;if(i<settings.count){let particle=particles[i];if(particle.position.w>0.0){
  let speed=length(particle.velocity.xyz);v.x+=particle.position.w;v.y=max(v.y,speed);v.z+=0.5*p.material.x*particle.position.w*speed*speed;v.w+=1.0;
  var spread=vec3f(0);
  if(settings.transferMode==0u){spread=1.5*vec3f(dot(abs(particle.cx.xyz),p.hdt.xyz),dot(abs(particle.cy.xyz),p.hdt.xyz),dot(abs(particle.cz.xyz),p.hdt.xyz));}
  affineBound=max(affineBound,length(abs(particle.velocity.xyz)+spread));
  if(!(speed<1e20)){atomicStore(&status[0],5u);}
 }}}
 foldLedger(lane,v,affineBound);
 if(lane==0u){ledgerParts[block]=LedgerPartial(values[0],vec4f(bounds[0],0,0,0));}
}
@compute @workgroup_size(64) fn ledger(@builtin(local_invocation_index) lane:u32){
 if(!ledgerActive(lane)){return;}var v=vec4f(0);var bound=0.0;
 for(var i=lane;i<(settings.count+255u)/256u;i+=64u){let part=ledgerParts[i];let b=part.totals;
  v=vec4f(v.x+b.x,max(v.y,b.y),v.z+b.z,v.w+b.w);bound=max(bound,part.bound.x);}
 foldLedger(lane,v,bound);
 if(lane==0u){let result=values[0];scalars[5]=bounds[0];scalars[27]=result.x;scalars[28]=f32(atomicLoad(&status[1]))*p.hdt.x*p.hdt.y*p.hdt.z/8.0;
  scalars[29]=result.y;scalars[30]=result.z;scalars[31]=result.w;scalars[6]=max(scalars[6],result.y);
  let error=atomicLoad(&status[0]);if(error>0u){scalars[12]=f32(error);}
 }
}

`;

export const APIC_BINDINGS = {
  clearBins: [0, 2, 8, 12], binParticles: [0, 1, 2, 8, 9, 10],
  scatterParticles: [0, 1, 3, 8, 10, 12, 13],
  transfer: [0, 2, 3, 4, 6, 8, 10, 11, 13, 15], moveParticles: [0, 1, 4, 5, 8, 9, 10, 15],
  surface: [0, 2, 3, 7, 8, 10, 13], ledgerPartial: [0, 1, 8, 9, 10, 14], ledger: [0, 8, 9, 10, 14],
} as const;
export type ApicEntry = keyof typeof APIC_BINDINGS;
