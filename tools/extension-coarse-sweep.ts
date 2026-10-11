/** Diagnostic-only elimination of redundant coarse sweep liveness work. */
import assert from "node:assert/strict";
import { fn } from "./probe-extension-direct-snapshot";

export function coarseSweepSource(source:string,direct=false):string{
 const body=fn(source,"pair_ueSweepCoarsePack");
 const before=`if(face.width==0u||!ueFaceLive(face.anchor,face.axis,face.width)){continue;}
  let old=pairRead(umSlot(face.anchor,face.axis,face.width));
  if(old.y==0.0||umClosedWall(face)){continue;}`;
 assert.equal(body.split(before).length-1,1,"coarse sweep guard changed");
 const after=`if(face.width==0u){continue;}
  let old=pairRead(umSlot(face.anchor,face.axis,face.width));
  if(old.y==0.0||umClosedWall(face)){continue;}
  ${direct?"":"if(!ueFaceLive(face.anchor,face.axis,face.width)){continue;}"}`;
 return source.replace(body,body.replace(before,after));
}
