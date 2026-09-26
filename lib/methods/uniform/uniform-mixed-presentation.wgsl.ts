import {uniformMixedFacesWGSL} from "./uniform-mixed-faces.wgsl";
import {uniformMixedVelocitySamplingWGSL} from "./uniform-mixed-velocity-sampling.wgsl";
import { uniformMixedVertexSamplingWGSL } from "./uniform-mixed-vertex-sampling.wgsl";

/** Read-only consumer of the same packed ownership and canonical vertices as
 * simulation. Dimensions come from the consumer's existing texture/uniform.
 * A one-word dummy disables mixed sampling for other methods. */
export function uniformMixedPresentationWGSL(binding:number,phi:string,dimensions:string,existingTopology?:string):string{
 const topology=/* wgsl */`
@group(0) @binding(${binding}) var<storage,read> umTopology:array<u32>;
const umRegularFine=false;
fn umDimensions()->vec3u{return ${dimensions};}
fn umTileDimensions()->vec3u{return umDimensions()/4u;}
fn umTileCount()->u32{let d=umTileDimensions();return d.x*d.y*d.z;}
fn umPresentationEnabled()->bool{return arrayLength(&umTopology)>1u;}
fn umTileWidth(t:u32)->u32{let word=umTopology[t];return select(select(4u,2u,(word&0x40000000u)!=0u),1u,(word&0x80000000u)!=0u);}
fn umTileStencil(t:u32)->vec2u{return vec2u(umTopology[2u*umTileCount()+2u*t],umTopology[2u*umTileCount()+2u*t+1u]);}
fn umTileMaximumWidth(t:u32)->u32{return umTileStencil(t).x>>27u;}
fn umTileAt(p:vec3u)->u32{let d=umTileDimensions();return p.x+d.x*(p.y+d.y*p.z);}
fn umTileCoord(t:u32)->vec3u{let d=umTileDimensions();return vec3u(t%d.x,(t/d.x)%d.y,t/(d.x*d.y));}
fn umCorner(k:u32,side:u32)->vec3u{return vec3u(k%side,(k/side)%side,k/(side*side));}
struct UMOwner {tile:u32,lane:u32,width:u32,index:u32}
fn umOwnerAt(p:vec3i)->UMOwner{
 if(any(p<vec3i(0))||any(p>=vec3i(umDimensions()))){return UMOwner();}
 let q=vec3u(p);let tile=umTileAt(q/4u);let width=umTileWidth(tile);let local=(q%4u)/width;let side=4u/width;
 let lane=local.x+side*(local.y+side*local.z);
 return UMOwner(tile,lane,width,(umTopology[tile]&0x3fffffffu)+lane);
}
fn umOrigin(o:UMOwner)->vec3u{return umTileCoord(o.tile)*4u+umCorner(o.lane,4u/o.width)*o.width;}
fn umLoadVertex(p:vec3u)->f32{return textureLoad(${phi},vec3i(p),0).x;}
`;
 const source=topology+uniformMixedVertexSamplingWGSL.replace(/\bumCounts.w\b/g,"8u").replace(/\bUM_D\b/g,"umDimensions()").replace(/\bUM_T\b/g,"umTileDimensions()");
 return existingTopology?source.replace(/@group\(0\) @binding\(\d+\) var<storage,read> umTopology:array<u32>;/,"").replace(/\bumTopology\b/g,existingTopology):source;
}

/** Uses exactly the simulation interpolation on read-only presentation fields.
 * The consumer supplies umLoadMixedFace, including negative boundary planes. */
export function uniformMixedPresentationVelocityWGSL():string{
 const faces=uniformMixedFacesWGSL.slice(0,uniformMixedFacesWGSL.indexOf("fn umOwnerAt("))
  +uniformMixedFacesWGSL.slice(uniformMixedFacesWGSL.indexOf("fn umFace("));
 return (faces+uniformMixedVelocitySamplingWGSL).replace(/\bUM_D\b/g,"umDimensions()")
  .replace(/\bUM_T\b/g,"umTileDimensions()").replace(/\bumCounts.w\b/g,"8u");
}
