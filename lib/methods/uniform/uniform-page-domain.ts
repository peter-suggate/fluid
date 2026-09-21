import {planUniformPages, type UniformPageCoordinate, type UniformPageEdge} from "./uniform-page-layout";

/** First accepted page-domain ABI. Dense field backing is a migration adapter;
 * cell/vertex membership and ownership come from this generation. No window
 * state or host readback may change its dispatches during an advance. */
export const UNIFORM_PAGE_DOMAIN_BASE = 256;
export const UNIFORM_PAGE_DOMAIN_HEADER = 16;
export const UNIFORM_PAGE_DOMAIN_RECORD = 16;
export interface UniformPageDomain {
  edge: UniformPageEdge;
  capacity: number;
  count: number;
  words: Uint32Array;
  cellDispatchOffset: number;
  vertexDispatchOffset: number;
}

export function initialUniformPageDomain(dimensions: readonly [number,number,number],
  edge:UniformPageEdge=32, reverse=false):UniformPageDomain {
  if(!dimensions.every(n=>Number.isSafeInteger(n)&&n>0))throw new RangeError("Page-domain dimensions must be positive integers");
  const extent=dimensions.map(n=>Math.ceil(n/edge));
  const coordinates:UniformPageCoordinate[]=[];
  for(let z=0;z<extent[2]!;z++)for(let y=0;y<extent[1]!;y++)for(let x=0;x<extent[0]!;x++)coordinates.push([x,y,z]);
  if(reverse)coordinates.reverse();
  const count=coordinates.length,capacity=count;
  const layout=planUniformPages(edge,capacity,coordinates);
  const words=new Uint32Array(UNIFORM_PAGE_DOMAIN_HEADER+17*capacity);
  const cells=[count*edge/4,edge/4,edge/4];
  const v=Math.ceil((edge+1)/4);
  const vertices=[count*v,v,v];
  words.set([...cells,0,...vertices,0,count,edge,0,0,...dimensions]);
  for(const slot of layout.activeSlots){
    const at=UNIFORM_PAGE_DOMAIN_HEADER+slot*UNIFORM_PAGE_DOMAIN_RECORD;
    words.set([...layout.coordinates[slot]!,slot,...layout.neighbors.subarray(slot*6,slot*6+6),1],at);
  }
  words.set(layout.activeSlots,UNIFORM_PAGE_DOMAIN_HEADER+16*capacity);
  return {edge,capacity,count,words,
    cellDispatchOffset:UNIFORM_PAGE_DOMAIN_BASE*4,
    vertexDispatchOffset:(UNIFORM_PAGE_DOMAIN_BASE+4)*4};
}

export function uniformPageDomainWGSL(domain?:Pick<UniformPageDomain,"edge"|"capacity">):string {
  if(!domain)return "";
  return /* wgsl */ `
const PAGE_DOMAIN_BASE:u32=${UNIFORM_PAGE_DOMAIN_BASE}u;
const PAGE_DOMAIN_EDGE:u32=${domain.edge}u;
const PAGE_DOMAIN_CAPACITY:u32=${domain.capacity}u;
fn pageDomainOrigin(page:u32)->vec3i{
 let slot=activeRegion[PAGE_DOMAIN_BASE+16u+16u*PAGE_DOMAIN_CAPACITY+page];
 let at=PAGE_DOMAIN_BASE+16u+16u*slot;
 return vec3i(vec3u(activeRegion[at],activeRegion[at+1u],activeRegion[at+2u]))*i32(PAGE_DOMAIN_EDGE);
}
fn pageDomainCell(g:vec3u)->vec3i{
 let page=g.x/PAGE_DOMAIN_EDGE;
 if(page>=activeRegion[PAGE_DOMAIN_BASE+8u]){return vec3i(-1);}
 let id=pageDomainOrigin(page)+vec3i(vec3u(g.x%PAGE_DOMAIN_EDGE,g.y,g.z));
 if(any(id>=dims())){return vec3i(-1);}return id;
}
fn pageDomainVertex(g:vec3u)->vec3i{
 let width=((PAGE_DOMAIN_EDGE+4u)/4u)*4u;let page=g.x/width;
 if(page>=activeRegion[PAGE_DOMAIN_BASE+8u]){return vec3i(-1);}
 let local=vec3u(g.x%width,g.y,g.z);let id=pageDomainOrigin(page)+vec3i(local);
 // A shared vertex belongs to the page on its positive side. Only physical
 // upper-boundary vertices are owned by the final page on the negative side.
 if(any((local>=vec3u(PAGE_DOMAIN_EDGE)) & (id!=dims()))||any(id>dims())){return vec3i(-1);}
 return id;
}
`;
}
