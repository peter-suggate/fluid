import { uniformPageDomainWGSL, UNIFORM_PAGE_DOMAIN_BASE, type UniformPageDomain } from "./uniform-page-domain";
import { uniformPressurePageWorkgroups } from "./uniform-pressure-pages";

export { uniformPressurePageWorkgroups as uniformVelocityPageWorkgroups };

function replaceFunction(source: string, name: string, body: string): string {
  const start=source.indexOf(`fn ${name}(`);
  if(start<0)throw new Error(`Missing velocity page operator ${name}`);
  const open=source.indexOf("{",start);
  let end=open+1,depth=1;
  while(depth && end<source.length){if(source[end]==="{")depth++;if(source[end]==="}")depth--;end++;}
  if(depth)throw new Error(`Unclosed velocity page operator ${name}`);
  return source.slice(0,open+1)+body+source.slice(end-1);
}

/** The extension hierarchy uses page work at every level. The narrow-band
 * convergence counter only gates the GPU dispatch; no window bounds participate. */
export function uniformVelocityPagedShader(source:string,dims:readonly [number,number,number],domain?:UniformPageDomain):string{
 let code=replaceFunction(source,"activeBaseId",domain?"return pageDomainCell(gid);":"return velocityPageCell(gid,baseDims());");
 code=replaceFunction(code,"hierarchyActiveId","if(frontParams.hierarchyTargetUsesBaseDims!=0u){return activeBaseId(gid);}return velocityPageCell(gid,hierarchyTargetDims());");
 code=code.replace("let p=vec3i(gid);if(!inBounds(p,baseDims()))", "let p=activeBaseId(gid);if(!inBounds(p,baseDims()))")
  .replace("let t = vec3i(gid); let c = coarseDims();","let c=coarseDims(); let t=velocityPageCell(gid,c);");
 const groups=uniformPressurePageWorkgroups(dims);
 for(let axis=0;axis<3;axis++)code=code.replace(`activeRegion[${13+axis}]`,domain?`activeRegion[${UNIFORM_PAGE_DOMAIN_BASE+axis}]`:`${groups[axis]}u`);
 return code+(domain?`fn dims()->vec3i{return baseDims();}${uniformPageDomainWGSL(domain)}`:"")+/* wgsl */ `
fn velocityPageCell(g:vec3u,d:vec3i)->vec3i{
 let grid=(vec3u(d)+vec3u(15u))/16u;let page=g.x/16u;
 if(page>=grid.x*grid.y*grid.z){return vec3i(-1);}
 return vec3i(vec3u(page%grid.x,(page/grid.x)%grid.y,page/(grid.x*grid.y))*16u+vec3u(g.x%16u,g.y,g.z));
}
`;
}
