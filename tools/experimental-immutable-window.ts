/** Tool-local specialization of immutable window words24..31. Runtime frame,
 * directory entries and packed block records remain authoritative. */
import assert from 'node:assert/strict';
export function experimentalImmutableWindow(code:string,words:ArrayLike<number>,count?:(name:string,n:number)=>void):string{
 assert.equal(words.length,8);
 return code.replace(/fn\s+(\w*(?:PageWord|WorldSolid|WallLow))\([^)]*\)[^{]*\{[^}]*\}/g,(body:string,name:string)=>{
  let replaced=0;
  const result=body.replace(/\b\w+\[[^\[\]\n;]+?\+(262168|262169|262170|262171|262172|262173|262174|262175)u\]/g,(_read:string,index:string)=>{replaced++;return `${words[Number(index)-262168]!}u`;});
  if(replaced)count?.(name,replaced);return result;
 });
}
