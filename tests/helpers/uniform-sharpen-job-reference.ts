import assert from "node:assert/strict";

/** Restore the old unconditional seam staging in each regular-owner job.
 * QA only: this retains the pre-partition sweep as a numerical reference. */
export function uniformSharpenJobReference(code:string):string{
 if(!code.includes("@compute @workgroup_size(192) fn propose("))return code;
 for(const name of ["Propose","Limit","Commit"]){
  const partitioned=`if(job<jobs.x){sh${name}(shActiveOwner(job,lane));}\n  else{sh${name}Seam(tile,lane);}`;
  assert.ok(code.includes(partitioned),`Missing partitioned sharpen ${name}`);
  code=code.replace(partitioned,`sh${name}Seam(tile,lane);\n  if(job<jobs.x){sh${name}(shActiveOwner(job,lane));}`);
 }
 return code;
}
