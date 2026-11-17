/** Offline exact arithmetic validation of the GPU's GLOBAL affine prefix.
 * Candidate bounds round outward; every emitted VT/EE is checked over a gap-free
 * cover. It validates only unprojected cloth motion, not solids or full physics.
 */
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {readFileSync,writeFileSync} from 'node:fs';
import {join} from 'node:path';
import {affineVertexTriangleCCD,affineEdgeEdgeCCD,type AffineVertex,type Vec3} from './cloth-affine-ccd';
import {clothSweptCandidates} from './cloth-swept-candidates';
const args=process.argv.slice(2),opt=(k:string,d:string)=>{const i=args.indexOf('--'+k);return i<0?d:args[i+1]!;};
const input=opt('input','/tmp/cloth-directions-forward'),bench=JSON.parse(readFileSync(opt('benchmark','/tmp/cloth-affine-gpu-endpoint-warm.json'),'utf8'));
const results=[];
for(const tick of [231,340]){
  const rows=bench.results.filter((r:{tick:number;wg:number})=>r.tick===tick&&r.wg===64);assert.equal(rows.length,2);
  const alpha=Math.min(...rows.map((r:{alpha:number})=>r.alpha)),thickness=Math.fround(rows[0].thickness),slot=rows[0].slot;
  assert.ok(alpha>0&&alpha<=1);assert.ok(rows.every((r:{slot:number;thickness:number})=>r.slot===slot&&Math.fround(r.thickness)===thickness));
  const manifest=JSON.parse(readFileSync(join(input,`tick${tick}.json`),'utf8'));
  const read=(name:string)=>{const f=manifest.files.find((f:{name:string})=>f.name===name);assert.ok(f);const b=readFileSync(join(input,f.filename));assert.equal(createHash('sha256').update(b).digest('hex'),f.sha256);return Uint8Array.from(b).buffer;};
  const x=new Float32Array(read(`${slot}-x`)),d=new Float32Array(read(`${slot}-d`)),t=new Uint32Array(read('triangles')),e=new Uint32Array(read('edges'));
  assert.equal(new Float32Array(read(`${slot}-control`))[0],0);
  const vertices:AffineVertex[]=Array.from({length:manifest.vertices},(_,i)=>({x:Array.from(x.subarray(4*i,4*i+3)) as unknown as Vec3,d:Array.from(d.subarray(4*i,4*i+3)) as unknown as Vec3}));
  const triangles=Array.from({length:manifest.triangles},(_,i)=>Array.from(t.subarray(4*i,4*i+3))).flat(),edges=Array.from({length:manifest.edges},(_,i)=>Array.from(e.subarray(4*i,4*i+2))).flat();
  let pairs=0,intervals=0;const start=performance.now();
  for(const c of clothSweptCandidates(vertices,triangles,edges,thickness)){
    const p=c.ids.map(i=>vertices[i]!) as [AffineVertex,AffineVertex,AffineVertex,AffineVertex];
    const r=(c.kind==='vt'?affineVertexTriangleCCD:affineEdgeEdgeCCD)(...p,{thickness,maxAlpha:alpha,exactEndpoints:true,sameTimeEndpoints:true});
    assert.equal(r.reason,'separated',JSON.stringify({tick,c,r,alpha}));assert.equal(r.alpha,alpha);pairs++;intervals+=r.intervals;
  }
  const row={tick,slot,alpha,thickness,pairs,intervals,cpuMs:performance.now()-start};results.push(row);console.log('PASS exact full affine prefix',JSON.stringify(row));
}
writeFileSync(opt('out','/tmp/cloth-affine-exact-certificate.json'),JSON.stringify({contract:'Exact dyadic supporting-plane certificate of unprojected affine CLOTH ONLY; no solid, projected path, or energy fidelity claim',results},null,2),{flag:'wx'});
