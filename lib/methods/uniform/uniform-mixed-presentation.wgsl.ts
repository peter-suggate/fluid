import {uniformMixedFacesWGSL} from "./uniform-mixed-faces.wgsl";
import {uniformCompiledTopologyWGSL} from "./uniform-compiled-topology";
import {uniformMixedVelocitySamplingWGSL} from "./uniform-mixed-velocity-sampling.wgsl";
import { uniformMixedVertexSamplingSource } from "./uniform-mixed-vertex-sampling.wgsl";
import { uniformDetailRuntimeWGSL } from "../../core/uniform-detail-abi";

/** Every eight-tap loop in a presentation copy of the samplers runs to an
 * opaque bound. With literal bounds Metal unrolls the nested reconstruction
 * at every call site, and a consumer that samples from many sites (the grid
 * overlay) never finishes compiling its pipeline. */
function uniformMixedPresentationLoops(source:string):string{
 return source.replace(/\bumCounts\.w\b/g,"umPresentationLoopBound()").replace(/;k<8u;/g,";k<umPresentationLoopBound();");
}

/** Read-only consumer of the same packed ownership and canonical vertices as
 * simulation. Vertex phi is two textures: `coarsePhi`, the 4h vertex base
 * ((t+1)^3, texel g = phi at vertex 4g, never folded: its extent is the
 * lattice under every placement), and `phi`, the detail field, loaded through
 * the detail table's accessors (udr*). A tile corner read on behalf of a 4h
 * owner loads from the base, and every vertex read on behalf of an h tile
 * (its corners included: an h tile stores all of its vertices) from the
 * detail field. Which one a load site reads is decided by the site, not per
 * vertex: the all-h path is the same eight direct loads it was before the
 * base existed, and with no h tile nothing is loaded from the detail binding
 * (it may be any 1^3 texture). The same code runs at every occupancy.
 * Every consumer entry point that samples calls udrInit() first.
 * A one-word dummy disables mixed sampling for other methods.
 * vertexCache: the sampler's cacheLookup (WGSL run in umVertexValue once a
 * vertex is known to need reconstruction, p in scope; it may return that
 * function's own value). Empty, the text is unchanged. */
export function uniformMixedPresentationWGSL(binding:number,phi:string,coarsePhi:string,existingTopology?:string,vertexCache=""):string{
 const topology=/* wgsl */`
@group(0) @binding(${binding}) var<storage,read> umTopology:array<u32>;
const umRegularFine=false;
fn umPresentationEnabled()->bool{return arrayLength(&umTopology)>1u;}
${uniformDetailRuntimeWGSL("umTopology","umPresentationEnabled()")}
fn umDimensions()->vec3u{return 4u*(textureDimensions(${coarsePhi})-vec3u(1u));}
fn umTileDimensions()->vec3u{return umDimensions()/4u;}
fn umTileCount()->u32{let d=umTileDimensions();return d.x*d.y*d.z;}
// The samplers' eight-tap loop bound, opaque to the compiler. A literal 8u
// lets Metal unroll every nested reconstruction at every call site; the
// grid overlay, which samples from many sites, then never finishes compiling.
// Ownership is always longer than eight words when bound, so this is 8.
fn umPresentationLoopBound()->u32{return min(arrayLength(&umTopology),8u);}
fn umTileWidth(t:u32)->u32{return select(4u,1u,(umTopology[t]&0x80000000u)!=0u);}
fn umTileStencil(t:u32)->vec2u{return vec2u(umTopology[2u*umTileCount()+2u*t],umTopology[2u*umTileCount()+2u*t+1u]);}
fn umTileMaximumWidth(t:u32)->u32{return umTileStencil(t).x>>27u;}
fn umTileMinimumWidth(t:u32)->u32{return umTileStencil(t).y>>27u;}
${uniformCompiledTopologyWGSL.replace(/\bUM_TILES\b/g,"umTileCount()").replace(/\bUM_T\b/g,"umTileDimensions()")}
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
// A tile corner, from the 4h base.
fn umLoadCoarseVertex(p:vec3u)->f32{return textureLoad(${coarsePhi},vec3i(p>>vec3u(2u)),0).x;}
// A vertex of an h tile, from the detail field.
fn umLoadFineVertex(p:vec3u)->f32{return udrStoredVertex(${phi},p);}
// A canonical vertex whose site does not know which it is.
fn umLoadVertex(p:vec3u)->f32{
 if(all((p&vec3u(3u))==vec3u(0u))){return umLoadCoarseVertex(p);}
 return umLoadFineVertex(p);
}
`;
 const sampling=uniformMixedVertexSamplingSource(vertexCache,false,{fine:"umLoadFineVertex",coarse:"umLoadCoarseVertex"});
 const source=topology+uniformMixedPresentationLoops(sampling).replace(/\bUM_D\b/g,"umDimensions()").replace(/\bUM_T\b/g,"umTileDimensions()");
 return existingTopology?source.replace(/@group\(0\) @binding\(\d+\) var<storage,read> umTopology:array<u32>;/,"").replace(/\bumTopology\b/g,existingTopology):source;
}

/** Uses exactly the simulation interpolation on read-only presentation fields.
 * The consumer supplies umLoadMixedFace, including negative boundary planes,
 * and includes uniformMixedPresentationWGSL (for umPresentationLoopBound). */
export function uniformMixedPresentationVelocityWGSL():string{
 const faces=uniformMixedFacesWGSL.slice(0,uniformMixedFacesWGSL.indexOf("fn umOwnerAt("))
  +uniformMixedFacesWGSL.slice(uniformMixedFacesWGSL.indexOf("fn umFaceFirst("));
 const sampling=(faces+uniformMixedVelocitySamplingWGSL).replace(/\bUM_D\b/g,"umDimensions()")
  .replace(/\bUM_T\b/g,"umTileDimensions()");
 return uniformMixedPresentationLoops(sampling);
}
