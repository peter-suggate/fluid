/** Diagnostic-only publication fusion. Reuses the hierarchy input binding,
 * which publication never writes, so the pipeline adds no resource limits. */
import assert from "node:assert/strict";
import { fn } from "./probe-extension-direct-snapshot";

export function extensionFineTapsSource(extension:string,taps:string):string{
 const body=fn(extension,"uePublishFine");
 const stores=[...body.matchAll(/(?:udStore_output\s*\(|textureStore\s*\(\s*output\s*,)[^;]*;/g)];
 assert.equal(stores.length,2,`Both fine publisher exits must duplicate the complete RGBA texel: ${body}`);
 const helper=taps.includes("fn udStore_unitVelocity(")?fn(taps,"udStore_unitVelocity").replaceAll("udStore_unitVelocity","ftStore").replaceAll(/\bunitVelocity\b/g,"coarseOut"):
  "fn ftStore(q:vec3i,value:vec4f){textureStore(coarseOut,q,value);}";
 assert.ok(!helper.includes("udb_"),"Fine tap output must remain atlas-only");
 assert.match(extension,/@group\(1\) @binding\(11\) var coarseOut:texture_storage_3d<rgba32float,write>/);
 let fused=body;for(const store of new Set(stores.map(match=>match[0])))fused=fused.replaceAll(store,`${store}ftStore(vec3i(origin),value);`);
 return extension.replace(body,fused)+"\n"+helper;
}
export function coarseOnlyTapsSource(source:string):string{
 const body=fn(source,"umUnitJobs");
 assert.match(body,/return \(umCounts(?:\.work)?\.x\+3u\)\/4u;/);
 return source.replace(body,"fn umUnitJobs()->u32{return 0u;}");
}
