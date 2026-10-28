import type { MeshVolumeHeader } from "./mesh-volume";

export interface SurfaceTextureMaterial {
  name: string; baseColorFactor: readonly number[]; roughnessFactor: number; metallicFactor: number;
  colorRoughnessLayer: number; normalLayer: number; normalStrength: number;
}
export interface SurfaceMappingHeader {
  version: 1; asset: string; sourceSha256: string; fvxPayloadSha256: string;
  origin: readonly number[]; dimensions: readonly number[]; cellSize: number; brickCount: number;
  brickCells: 8; nodeStrideBytes: 8; mappingStrideBytes: 64; mappingCount: number;
  mappingByteOffset: number; nodeByteOffset: number; textureByteOffset: number;
  textureWidth: number; textureHeight: number; textureLayerCount: number; mipLevelCount: number;
  textureFormat: "rgba8unorm-srgb"; textureLayout: "layer-major-mip-major";
  textureOrigin: "blender-uv-bottom-left"; materials: readonly SurfaceTextureMaterial[];
}
export interface SurfaceMapping {
  header: SurfaceMappingHeader; mappings: Uint32Array<ArrayBuffer>; nodes: Uint32Array<ArrayBuffer>;
  textureBytes: Uint8Array<ArrayBuffer>;
}
export async function sha256(bytes: Uint8Array): Promise<string> {
  const copy = new Uint8Array(bytes);
  return [...new Uint8Array(await crypto.subtle.digest("SHA-256", copy))].map(v => v.toString(16).padStart(2,"0")).join("");
}
export function surfaceLayerByteLength(width: number, height: number, levels: number): number {
  let size = 0;
  for (let mip=0; mip<levels; mip++) size += Math.max(1,width>>mip)*Math.max(1,height>>mip)*4;
  return size;
}
/** Attribute maps contain no source positions or triangle indices. Visibility
 * and geometry remain entirely in the companion's verified FVX scalar field. */
export function decodeSurfaceMapping(bytes: Uint8Array, volume: MeshVolumeHeader): SurfaceMapping {
  if (bytes.length<8 || new TextDecoder().decode(bytes.subarray(0,4))!=="FVM1") throw new Error("Invalid surface mapping magic");
  const view=new DataView(bytes.buffer,bytes.byteOffset,bytes.byteLength), length=view.getUint32(4,true);
  if (length>bytes.length-8) throw new Error("Truncated surface mapping header");
  const h=JSON.parse(new TextDecoder().decode(bytes.subarray(8,8+length))) as SurfaceMappingHeader;
  const integer=(v:number)=>Number.isSafeInteger(v)&&v>=0;
  const power=(v:number)=>integer(v)&&v>=1&&v<=4096&&(v&(v-1))===0;
  const hash=(v:unknown)=>typeof v==="string"&&/^[0-9a-f]{64}$/.test(v);
  if (!h || h.version!==1 || h.asset!==volume.asset || h.origin?.join()!==volume.origin.join()
    || h.dimensions?.join()!==volume.dimensions.join() || h.cellSize!==volume.cellSize || h.brickCount!==volume.brickCount
    || h.brickCells!==8 || h.nodeStrideBytes!==8 || h.mappingStrideBytes!==64 || !integer(h.mappingCount)||h.mappingCount<1
    || !hash(h.sourceSha256)||!hash(h.fvxPayloadSha256) || !power(h.textureWidth)||!power(h.textureHeight)
    || !integer(h.textureLayerCount)||h.textureLayerCount>256 || h.mipLevelCount!==1+Math.floor(Math.log2(Math.max(h.textureWidth,h.textureHeight)))
    || h.textureFormat!=="rgba8unorm-srgb" || h.textureLayout!=="layer-major-mip-major" || h.textureOrigin!=="blender-uv-bottom-left"
    || h.mappingByteOffset!==0 || h.nodeByteOffset!==h.mappingCount*64
    || h.textureByteOffset!==h.nodeByteOffset+h.brickCount*729*8 || !Array.isArray(h.materials)) throw new Error("Invalid surface mapping header");
  const layer=(v:number)=>v===0xffffffff||(integer(v)&&v<h.textureLayerCount);
  if(!h.materials.every(m=>typeof m.name==="string"&&m.baseColorFactor?.length===3&&m.baseColorFactor.every((v:number)=>Number.isFinite(v)&&v>=0&&v<=1)
    && [m.roughnessFactor,m.metallicFactor].every(v=>Number.isFinite(v)&&v>=0&&v<=1)&&Number.isFinite(Math.fround(m.normalStrength))&&m.normalStrength>=0
    && layer(m.colorRoughnessLayer)&&layer(m.normalLayer))) throw new Error("Invalid surface texture material");
  const payload=bytes.subarray(8+length), textureSize=surfaceLayerByteLength(h.textureWidth,h.textureHeight,h.mipLevelCount)*h.textureLayerCount;
  if(textureSize>256*1024*1024) throw new Error("Surface texture companion exceeds 256 MiB image budget");
  if(payload.length!==h.textureByteOffset+textureSize) throw new Error("Surface mapping payload length mismatch");
  const mappings=new Uint32Array(payload.slice(0,h.nodeByteOffset).buffer), f=new Float32Array(mappings.buffer);
  const nodes=new Uint32Array(payload.slice(h.nodeByteOffset,h.textureByteOffset).buffer);
  if(mappings.subarray(0,16).some(v=>v!==0)) throw new Error("Invalid surface mapping sentinel");
  for(let i=1;i<h.mappingCount;i++) {
    const a=i*16;
    if(!f.subarray(a,a+12).every(Number.isFinite)||Math.abs(f[a+11]!)!==1||mappings[a+12]!>=h.materials.length
      ||mappings.subarray(a+13,a+16).some(v=>v!==0)) throw new Error("Invalid surface mapping record");
  }
  for(let i=0;i<nodes.length;i+=2) if(nodes[i]!>=h.mappingCount) throw new Error("Invalid surface mapping node reference");
  return {header:h,mappings,nodes,textureBytes:payload.slice(h.textureByteOffset)};
}
export async function attachSurfaceMapping(volume: import("./mesh-volume").MeshVolume, fvxBytes: Uint8Array, companionBytes: Uint8Array): Promise<void> {
  const descriptor=volume.header.surfaceMapping;
  if(!descriptor) throw new Error("Missing surface mapping descriptor");
  let bytes=companionBytes;
  if(bytes[0]===0x1f&&bytes[1]===0x8b) bytes=new Uint8Array(await new Response(new Blob([new Uint8Array(bytes)]).stream().pipeThrough(new DecompressionStream("gzip"))).arrayBuffer());
  if(await sha256(bytes)!==descriptor.sha256) throw new Error("Surface mapping SHA256 mismatch");
  const mapping=decodeSurfaceMapping(bytes,volume.header);
  const headerLength=new DataView(fvxBytes.buffer,fvxBytes.byteOffset,fvxBytes.byteLength).getUint32(4,true);
  if(await sha256(fvxBytes.subarray(8+headerLength))!==mapping.header.fvxPayloadSha256) throw new Error("Surface mapping belongs to different voxel samples");
  volume.surfaceMapping=mapping;
}
