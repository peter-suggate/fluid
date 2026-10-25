import { MeshoptSimplifier } from "meshoptimizer";
import { meshVolumeSurface, reorderMeshVolumeTriangles } from "../lib/core/imported-assets/mesh-volume-surface";
import { sampleMeshVolume, type MeshVolume } from "../lib/core/imported-assets/mesh-volume";

/** Offline experimental LODs. Shared vertex addresses and source shading stay
 * unchanged. Runtime replay replaces both the index buffer and indirect ranges. */
export function nativeIndexVariant(volume:MeshVolume,strategy:"zero-error"|"appearance") {
  const source=meshVolumeSurface(volume),points=source.positions;
  const native=source.indices.slice(0,source.levels[0]!.count);
  const attributes=new Float32Array(strategy==="appearance"?points.length/3*6:0);
  const material=new Uint32Array(strategy==="appearance"?points.length/3:0);
  const locks=new Uint8Array(strategy==="appearance"?points.length/3:0);
  if(strategy==="appearance"){
    for(let i=0;i<points.length;i+=3){
      const position=volume.header.origin.map((v,a)=>v+volume.header.dimensions[a]!*volume.header.cellSize/2+points[i+a]!) as [number,number,number];
      const sample=sampleMeshVolume(volume,position);
      material[i/3]=sample.material;
      attributes.set([...sample.shadingNormal,...(volume.header.materials[sample.material-1]?.baseColorLinear??[0,0,0])],i*2);
    }
    for(let i=0;i<native.length;i+=3){
      const a=native[i]!,b=native[i+1]!,c=native[i+2]!;
      if(material[a]!==material[b]||material[a]!==material[c])locks[a]=locks[b]=locks[c]=1;
    }
  }
  const buffers:Uint32Array[]=[],levels:{originalFirst:number;first:number;count:number}[]=[];
  let first=0;
  for(let i=0;i<source.levels.length;i++){
    const level=source.levels[i]!;
    const indices=i===0?MeshoptSimplifier.simplify(native,points,3,0,0,["ErrorAbsolute","LockBorder"])[0]
      :strategy==="appearance"?MeshoptSimplifier.simplifyWithAttributes(native,points,3,attributes,6,[.02,.02,.02,.05,.05,.05],locks,0,level.error*2,["ErrorAbsolute","LockBorder"])[0]
      :source.indices.slice(level.first,level.first+level.count);
    reorderMeshVolumeTriangles(indices);
    buffers.push(indices);levels.push({originalFirst:level.first,first,count:indices.length});first+=indices.length;
  }
  const indices=new Uint32Array(first);for(let i=0;i<buffers.length;i++)indices.set(buffers[i]!,levels[i]!.first);
  return {indices,levels};
}

export const nativeIndexRangePatchWGSL=/* wgsl */`
@group(0) @binding(0) var<storage,read_write> args:array<u32>;
@group(0) @binding(1) var<storage,read> ranges:array<u32>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id:vec3u){
  let group=id.x;if(group>=arrayLength(&args)/5u){return;}
  let base=group*32u;let first=args[group*5u+2u];
  for(var i=0u;i<ranges[base];i++){
    let row=base+1u+i*3u;
    if(ranges[row]==first){args[group*5u]=ranges[row+2u];args[group*5u+2u]=ranges[row+1u];return;}
  }
}
`;
