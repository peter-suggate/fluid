/** Physical texture pages for the existing CM11a operators. Logical coordinates
 * remain shared across page seams; only field addressing changes here. */
export const UNIFORM_PRESSURE_PAGE_EDGE = 16;
export function uniformPressurePageExtent(dims: readonly [number, number, number]): [number,number,number] {
 const edge=UNIFORM_PRESSURE_PAGE_EDGE;
 const count=dims.reduce((n,d)=>n*Math.ceil(d/edge),1);
 const x=Math.min(8,count),y=Math.min(8,Math.ceil(count/x)),z=Math.ceil(count/(x*y));
 return [x*edge,y*edge,z*edge];
}
export function uniformPressurePageWorkgroups(dims: readonly [number,number,number]): [number,number,number] {
 const pages=dims.reduce((n,d)=>n*Math.ceil(d/UNIFORM_PRESSURE_PAGE_EDGE),1);
 return [pages*4,4,4];
}
export const uniformPressurePageAddressWGSL=/* wgsl */ `
fn mgPageAddress(p:vec3i,d:vec3u,atlas:vec3u)->vec3i{
 if(any(p<vec3i(0))||any(p>=vec3i(d))){return vec3i(-1);}
 let q=vec3u(p);let grid=(d+vec3u(15u))/16u;let page=q/16u;
 let slot=page.x+grid.x*(page.y+grid.y*page.z);let tiles=atlas/16u;
 return vec3i(q%16u+16u*vec3u(slot%tiles.x,(slot/tiles.x)%tiles.y,slot/(tiles.x*tiles.y)));
}
`;
/** Replace only CM11a field accesses, leaving discretization and arithmetic
 * ordering intact. Function-call parsing handles nested coordinate expressions. */
export function uniformPressurePagedShader(source:string, logicalDispatch=false):string{
 const declarations=[...source.matchAll(/@group\(1\) @binding\((\d+)\) var (mg\w+): (texture[^;]+);/g)];
 const fields=new Map(declarations.map(m=>[m[2]!,{binding:Number(m[1]),type:m[3]!}]));
 let helpers=uniformPressurePageAddressWGSL;
 for(const [name,{binding,type}] of fields){
  const address=`mgPageAddress(p,mg.fieldDims[${binding}].xyz,textureDimensions(${name}))`;
  helpers+=type.startsWith('texture_storage')
   ? `\nfn ${name}Store(p:vec3i,value:vec4f){let at=${address};if(any(at<vec3i(0))){return;}textureStore(${name},at,value);}`
   : `\nfn ${name}Load(p:vec3i)->vec4f{let at=${address};if(any(at<vec3i(0))){return vec4f(0);}return textureLoad(${name},at,0);}`;
 }
 let result=source.replace('  control: vec4u,','  control: vec4u,\n  fieldDims: array<vec4u,16>,');
 const start=result.indexOf("fn mgActiveId(");
 if(start>=0){
  let end=result.indexOf("{",start)+1,depth=1;
  while(depth>0&&end<result.length){if(result[end]==="{")depth++;if(result[end]==="}")depth--;end++;}
  if(depth!==0)throw new Error("Unclosed pressure page dispatch function");
  result=result.slice(0,start)+(logicalDispatch ? "fn mgActiveId(gid:vec3u)->vec3i{return vec3i(gid);}" : `fn mgActiveId(gid:vec3u)->vec3i{
    let grid=(mg.coarseDims.xyz+vec3u(15u))/16u;
    let slot=gid.x/16u;
    if(slot>=grid.x*grid.y*grid.z){return vec3i(-1);}
    let page=vec3u(slot%grid.x,(slot/grid.x)%grid.y,slot/(grid.x*grid.y));
    return vec3i(page*16u+vec3u(gid.x%16u,gid.y,gid.z));
  }`)+result.slice(end);
 }
 return rewritePressureTextureCalls(result,fields)+helpers;
}
export function rewritePressureTextureCalls(result:string,fields:ReadonlyMap<string,unknown>):string{
 const pattern=/texture(Load|Store)\(\s*(\w+)\s*,/g;
 let cursor=0,output='',match:RegExpExecArray|null;
 while((match=pattern.exec(result))){
  const field=match[2]!;if(!fields.has(field))continue;
  let at=pattern.lastIndex,depth=0,split=-1;
  for(;at<result.length;at++){
   const c=result[at];if(c==='('||c==='[')depth++;
   else if(c===']')depth--;
   else if(c===')'){if(depth===0)break;depth--;}
   else if(c===','&&depth===0)split=at;
  }
  if(split<0||at===result.length)throw new Error(`Cannot parse ${field} access`);
  const coordinate=rewritePressureTextureCalls(result.slice(pattern.lastIndex,split).trim(),fields);
  const value=rewritePressureTextureCalls(result.slice(split+1,at).trim(),fields);
  output+=result.slice(cursor,match.index)+`${field}${match[1]}(${coordinate}${match[1]==='Store'?`,${value}`:''})`;
  cursor=at+1;pattern.lastIndex=cursor;
 }
 return output+result.slice(cursor);
}
