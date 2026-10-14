/** Fit a shell seed to its nodal depth. The caller already sampled lattice.
 * The gradient probes and proposed step share one depth-sampler call site:
 * duplicating that graph makes native shader compilation substantially larger.
 * Probe order and floating-point reduction order match the original axis/side
 * traversal, including one-sided slopes beside blocked positions. */
export const narrowBandSeedShellWGSL = /* wgsl */ `
fn seedShell(lattice:vec3f,distance:f32)->vec4f{
 var q=lattice;var d=distance;
 for(var i=0u;i<4u;i++){
  let rise=d+NB_SEED_DEPTH;if(rise<=1e-3){break;}
  var g=vec3f(0);
  // Three central/one-sided slopes, then a proposed step along the gradient.
  for(var axis=0u;axis<4u;axis++){
   let stepping=axis==3u;var e=vec3f(0);var next=q;
   if(stepping){
    let g2=dot(g,g);if(g2<1e-8){return vec4f(q,d);}
    next=nbPointNearIn(vec3i(floor(q)),q-min(rise,0.5)*g*inverseSqrt(g2),0.01);
   }else{e[axis]=0.5;}
   var open=array<bool,2>(false,false);var depth=array<f32,2>(d,d);
   for(var side=0u;side<select(2u,1u,stepping);side++){
    var x=next;if(!stepping){x=q+select(-e,e,side==1u);}
    open[side]=stepping||!nbParticleBlocked(x);
    if(open[side]){depth[side]=particleDepth(x);}
   }
   if(stepping){
    if(depth[0]>=d){return vec4f(q,d);}q=next;d=depth[0];
   }else{
    let span=select(2.0,1.0,open[0]&&open[1]);
    g[axis]=select(0.0,span*(depth[1]-depth[0]),open[0]||open[1]);
   }
  }
 }
 return vec4f(q,d);
}
`;
