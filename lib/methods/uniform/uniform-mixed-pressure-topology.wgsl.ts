/** CM11a pressure topology on mixed owners: (open, V+x, V+y, V+z), the native
 * mgVolumeIn record. Level 0 borrows the native departure texture (texel at
 * each owner's origin; free between forces and projection). Coarser levels
 * use an arena vec4 per owner, then one per halo slot whose x is that wall's
 * V. Only built for scenes with static solids. */
export type UniformMixedPressureTopology = {texture:GPUTexture}|{buffer:GPUBufferBinding};

/** Requires the owner topology ABI (optionally namespaced by `prefix`), and
 * for buffers the matching umBoundaryIndex. With `phiBase`, the buffer record
 * is not its own binding: it is the tail of the caller's `phi:array<f32>`
 * binding from f32 index `phiBase` (the pressure level stage is at the
 * 10-storage-buffer ceiling with its seam-record group). */
export function uniformMixedPressureTopologyWGSL(kind:"texture"|"buffer",group:number,binding:number,prefix="",phiBase?:number):string{
 const row=(i:string)=>phiBase===undefined?`umTopologyBuffer[${i}]`:`umTopologyRow(${i})`;
 const declaration=phiBase===undefined?`@group(${group}) @binding(${binding}) var<storage,read_write> umTopologyBuffer:array<vec4f>;`
  :`fn umTopologyRow(i:u32)->vec4f{let b=${phiBase}u+4u*i;return vec4f(phi[b],phi[b+1u],phi[b+2u],phi[b+3u]);}`;
 const source=kind==="texture"?/* wgsl */ `
@group(${group}) @binding(${binding}) var umTopologyTexture:texture_3d<f32>;
fn umTopo(o:UMOwner)->vec4f{return textureLoad(umTopologyTexture,vec3i(umOrigin(o)),0);}
fn umPressureWallV(o:UMOwner,axis:u32,sign:i32)->f32{
 let t=umTopo(o);if(sign>0){return t[axis+1u];}return 0.5*t.x;
}
fn umPressureRegularV(o:UMOwner,axis:u32,sign:i32)->f32{
 var at=vec3i(umOrigin(o));if(sign<0){at[axis]-=i32(o.width);}
 return textureLoad(umTopologyTexture,at,0)[axis+1u];
}
`:/* wgsl */ `
${declaration}
fn umTopo(o:UMOwner)->vec4f{return ${row("o.index")};}
fn umPressureWallV(o:UMOwner,axis:u32,sign:i32)->f32{return ${row("umBoundaryIndex(o,axis,sign)")}.x;}
fn umPressureRegularV(o:UMOwner,axis:u32,sign:i32)->f32{
 if(sign>0){return ${row("o.index")}[axis+1u];}
 var at=vec3i(umOrigin(o));at[axis]-=i32(o.width);
 return ${row("umOwnerAt(at).index")}[axis+1u];
}
`;
 return prefix?source.replace(/\b(?:um[A-Z]\w*|UM_\w+|UMOwner|UMFace)\b/g,name=>prefix+name):source;
}
