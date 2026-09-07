/** Compiled queries into an immutable density generation. Physics cell IDs
 * occur only in the operation image, never in the retained field image. */
export const retainedDensityWGSL = /* wgsl */`
@group(0) @binding(0) var<storage,read> support:array<u32>;
@group(0) @binding(1) var<storage,read> coefficients:array<f32>;
@group(0) @binding(2) var<storage,read> operations:array<u32>;
@group(0) @binding(3) var<storage,read> moments:array<f32>;
@group(0) @binding(4) var<storage,read_write> results:array<vec4f>;

fn basis(t:f32)->vec3f{return vec3f((1.0-t)*(1.0-t),2.0*t*(1.0-t),t*t);}
fn derivative(t:f32)->vec3f{return vec3f(2.0*t-2.0,2.0-4.0*t,2.0*t);}
fn validImage()->bool{return operations[1u]==support[0u];}

// Every operation was compiled from physical coordinates on the host. A
// query carries its retained support ordinal and normalized coordinates, so
// steady-state evaluation has no world lookup or reconstruction solve.
@compute @workgroup_size(64)
fn evaluateDensity(@builtin(global_invocation_id)gid:vec3u){
  let id=gid.x+gid.y*operations[3u];if(id>=operations[0u]){return;}
  if(!validImage()){results[id]=vec4f(-1.0);return;}
  let at=4u+4u*id;let cell=operations[at];
  let u=bitcast<vec3f>(vec3u(operations[at+1u],operations[at+2u],operations[at+3u]));
  let b=4u+28u*cell;let inverseWidth=bitcast<f32>(support[b+27u]);
  let bx=basis(u.x);let by=basis(u.y);let bz=basis(u.z);
  let dx=derivative(u.x);let dy=derivative(u.y);let dz=derivative(u.z);
  var value=vec4f(0.0);
  for(var z=0u;z<3u;z++){for(var y=0u;y<3u;y++){for(var x=0u;x<3u;x++){
    let c=coefficients[support[b+x+3u*y+9u*z]];
    value+=c*vec4f(bx[x]*by[y]*bz[z],dx[x]*by[y]*bz[z]*inverseWidth,
      bx[x]*dy[y]*bz[z]*inverseWidth,bx[x]*by[y]*dz[z]*inverseWidth);
  }}}
  results[id]=value;
}

// CSR entries contain a retained support ordinal and nine physical Bernstein
// basis integrals. Their tensor product is the exact box-overlap moment. The
// output stores both the full-cell mean and the physical integrated amount.
@compute @workgroup_size(64)
fn integrateDensity(@builtin(global_invocation_id)gid:vec3u){
  let id=gid.x+gid.y*operations[3u];if(id>=operations[0u]){return;}
  if(!validImage()){results[id]=vec4f(-1.0);return;}
  let row=4u+4u*id;let first=operations[row];let end=operations[row+1u];
  let inverseVolume=bitcast<f32>(operations[row+2u]);
  let entries=operations[2u];var amount=0.0;
  for(var item=first;item<end;item++){
    let cell=operations[entries+item];let b=4u+28u*cell;let m=9u*item;
    for(var z=0u;z<3u;z++){for(var y=0u;y<3u;y++){for(var x=0u;x<3u;x++){
      amount+=coefficients[support[b+x+3u*y+9u*z]]
        *moments[m+x]*moments[m+3u+y]*moments[m+6u+z];
    }}}
  }
  results[id]=vec4f(amount*inverseVolume,amount,0.0,0.0);
}
`;
