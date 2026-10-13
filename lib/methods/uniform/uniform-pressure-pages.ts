/** Rewrite the named fields' textureLoad/textureStore calls to their <field>Load/<field>Store
 * accessors. Function-call parsing handles nested coordinate expressions. */
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
  if((split<0&&match[1]==='Store')||at===result.length)throw new Error(`Cannot parse ${field} access`);
  const coordinate=rewritePressureTextureCalls(result.slice(pattern.lastIndex,split<0?at:split).trim(),fields);
  const value=match[1]==='Store'?rewritePressureTextureCalls(result.slice(split+1,at).trim(),fields):'';
  output+=result.slice(cursor,match.index)+`${field}${match[1]}(${coordinate}${match[1]==='Store'?`,${value}`:''})`;
  cursor=at+1;pattern.lastIndex=cursor;
 }
 return output+result.slice(cursor);
}
