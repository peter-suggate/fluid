/** Compare same-view lighting arms, preserving full-resolution depth. */
import assert from 'node:assert/strict';
import {readFileSync,writeFileSync} from 'node:fs';
import {inflateSync} from 'node:zlib';
const directory=process.argv[2]??'artifacts/hero-render-halving-2026-10-04';
const results=JSON.parse(readFileSync(`${directory}/results.json`,'utf8')) as {view:string;arm:{name:string};gpuMedian_ms:number;wallMedian_ms:number}[];
const {width,height}=JSON.parse(readFileSync(`${directory}/provenance.json`,'utf8')) as {width:number;height:number};
const pixels=width*height;
function png(path:string){
 const data=readFileSync(path),parts:Buffer[]=[];
 for(let i=8;i<data.length;){const n=data.readUInt32BE(i),type=data.toString('ascii',i+4,i+8);if(type==='IDAT')parts.push(data.subarray(i+8,i+8+n));i+=n+12;}
 const raw=inflateSync(Buffer.concat(parts)),rgb=new Uint8Array(pixels*3);
 assert.equal(raw.length,height*(width*3+1));
 for(let y=0;y<height;y++){assert.equal(raw[y*(width*3+1)],0,'Capture writer uses PNG filter zero');rgb.set(raw.subarray(y*(width*3+1)+1,(y+1)*(width*3+1)),y*width*3);}
 return rgb;
}
function half(u:number){const sign=u&0x8000?-1:1,exp=(u>>10)&31,m=u&1023;return sign*(exp===0?m*2**-24:exp===31?m?NaN:Infinity:(1+m/1024)*2**(exp-15));}
const summary=[];
for(const view of new Set(results.map(r=>r.view))){
 const baseline=readFileSync(`${directory}/${view}-baseline.rgba16f`),display=png(`${directory}/${view}-baseline.png`);
 for(const name of new Set(results.filter(r=>r.view===view).map(r=>r.arm.name))){
  const image=readFileSync(`${directory}/${view}-${name}.rgba16f`),rgb=png(`${directory}/${view}-${name}.png`);
  assert.equal(image.length,pixels*8);assert.equal(image.length,baseline.length);
  let depthMismatch=0,linearAbsolute=0,linearSquared=0,displayAbsolute=0,changedPixels=0;
  for(let p=0;p<pixels;p++){
   if(image.readUInt16LE(p*8+6)!==baseline.readUInt16LE(p*8+6))depthMismatch++;
   let maxDisplay=0;
   for(let c=0;c<3;c++){
    const delta=Math.abs(half(image.readUInt16LE(p*8+c*2))-half(baseline.readUInt16LE(p*8+c*2)));
    assert.ok(Number.isFinite(delta));linearAbsolute+=delta;linearSquared+=delta*delta;
    const displayDelta=Math.abs(rgb[p*3+c]-display[p*3+c]);displayAbsolute+=displayDelta;maxDisplay=Math.max(maxDisplay,displayDelta);
   }
   if(maxDisplay>8)changedPixels++;
  }
  const arms=results.filter(r=>r.view===view&&r.arm.name===name);
  summary.push({view,arm:name,gpuMedians_ms:arms.map(r=>r.gpuMedian_ms),wallMedians_ms:arms.map(r=>r.wallMedian_ms),depthMismatch,linearRgbMAE:linearAbsolute/(pixels*3),linearRgbRMSE:Math.sqrt(linearSquared/(pixels*3)),displayRgbMAE_255:displayAbsolute/(pixels*3),pixelsOver8_255_percent:100*changedPixels/pixels});
 }
}
writeFileSync(`${directory}/summary.json`,JSON.stringify(summary,null,2)+'\n');
console.log(JSON.stringify(summary,null,2));
