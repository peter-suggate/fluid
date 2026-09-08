/** Compare tightly packed 1080p HDR captures; the fourth half-word is occlusion depth. */
import assert from "node:assert/strict";
import fs from 'node:fs';
import {gunzipSync} from 'node:zlib';
const load=path=>path.endsWith('.gz')?gunzipSync(fs.readFileSync(path)):fs.readFileSync(path);
const half=x=>{const s=(x&32768)?-1:1,e=(x>>10)&31,f=x&1023;return e===0?s*2**-14*f/1024:e===31?(f?NaN:s*Infinity):s*2**(e-15)*(1+f/1024);};
assert.ok(process.argv.length>=4,'Usage: node tools/compare-svo-shading-images.mjs REFERENCE CANDIDATE...');
const bytes=load(process.argv[2]);assert.equal(bytes.length,1920*1080*8);const a=new Uint16Array(bytes.buffer,bytes.byteOffset,bytes.length/2);
for(const path of process.argv.slice(3)){
 const bts=load(path),b=new Uint16Array(bts.buffer,bts.byteOffset,bts.length/2);assert.equal(bts.length,bytes.length);let sum=0,max=0,pixels=0,depth=0,nonfinite=0,toneSum=0,over8=0;
 for(let i=0;i<a.length;i+=4){let changed=false,large=false;for(let c=0;c<3;c++){const av=half(a[i+c]),bv=half(b[i+c]),d=Math.abs(av-bv);sum+=d;max=Math.max(max,d);changed||=a[i+c]!==b[i+c];nonfinite+=!Number.isFinite(bv);const tone=x=>Math.round(Math.min(1,Math.max(0,x))**(1/2.2)*255);const td=Math.abs(tone(av)-tone(bv));toneSum+=td;large||=td>8;}pixels+=changed;over8+=large;depth+=a[i+3]!==b[i+3];}
 console.log(JSON.stringify({path,rgbMae:sum/(a.length/4*3),rgbMax:max,changedPixels:pixels,depthChanged:depth,nonfinite,toneMae:toneSum/(a.length/4*3),over8}));
}
