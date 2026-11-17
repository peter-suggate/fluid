/** Offline CPU work census, NOT GPU timing or live collision certification.
 * --input /tmp/cloth-directions-forward --ticks 231,340 --slots 0 --out <json>
 * All pairs in each requested complete affine sweep are enumerated. No sampling.
 */
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {readFileSync,writeFileSync} from 'node:fs';
import {join} from 'node:path';
import {affineVertexTriangleCCD,affineEdgeEdgeCCD,type AffineVertex,type Vec3} from './cloth-affine-ccd';
import {clothSweptCandidates} from './cloth-swept-candidates';
const args=process.argv.slice(2),opt=(k:string,d:string)=>{const i=args.indexOf('--'+k);return i<0?d:args[i+1]!;};
const input=opt('input','/tmp/cloth-directions-forward'),ticks=opt('ticks','231,340').split(',').map(Number),slots=opt('slots','0').split(',').map(Number);
const thickness=Number(opt('thickness','.001'));assert.ok(Number.isFinite(thickness)&&thickness>=0);
const results=[];
for(const tick of ticks){
  const manifest=JSON.parse(readFileSync(join(input,`tick${tick}.json`),'utf8'));
  assert.equal(manifest.vertices,22500);
  const read=(name:string):ArrayBuffer=>{
    const f=manifest.files.find((f:{name:string})=>f.name===name);assert.ok(f,'Missing snapshot '+name);
    const bytes=readFileSync(join(input,f.filename));assert.equal(bytes.byteLength,f.bytes);assert.equal(createHash('sha256').update(bytes).digest('hex'),f.sha256);
    return Uint8Array.from(bytes).buffer;
  };
  const packedT=new Uint32Array(read('triangles')),packedE=new Uint32Array(read('edges'));
  const triangles=Array.from({length:manifest.triangles},(_,i)=>Array.from(packedT.subarray(4*i,4*i+3))).flat();
  const edges=Array.from({length:manifest.edges},(_,i)=>Array.from(packedE.subarray(4*i,4*i+2))).flat();
  for(const slot of slots){
    const control=new Float32Array(read(`${slot}-control`));
    if(control[0]!==0){console.log('Inactive Newton slot excluded',tick,slot);continue;}
    const x=new Float32Array(read(`${slot}-x`)),d=new Float32Array(read(`${slot}-d`));
    const vertices:AffineVertex[]=Array.from({length:manifest.vertices},(_,i)=>({x:Array.from(x.subarray(4*i,4*i+3)) as unknown as Vec3,d:Array.from(d.subarray(4*i,4*i+3)) as unknown as Vec3}));
    const start=performance.now();
    const counts={vt:0,ee:0,separated:0,unresolved:0,budget:0,nonfinite:0,zeroPrefix:0,intervals:0};
    let alpha=1,firstZero:unknown,oracleMs=0;
    for(const candidate of clothSweptCandidates(vertices,triangles,edges,thickness)){
      counts[candidate.kind]++;
      const [a,b,c,e]=candidate.ids.map(i=>vertices[i]!) as [AffineVertex,AffineVertex,AffineVertex,AffineVertex];
      const before=performance.now();
      const r=(candidate.kind==='vt'?affineVertexTriangleCCD:affineEdgeEdgeCCD)(a,b,c,e,{thickness});
      oracleMs+=performance.now()-before;
      counts[r.reason]++;counts.intervals+=r.intervals;alpha=Math.min(alpha,r.alpha);
      if(r.alpha===0){counts.zeroPrefix++;firstZero??=candidate;}
    }
    const row={tick,slot,thickness,alpha,counts,firstZero,elapsedCpuMs:performance.now()-start,oracleCpuMs:oracleMs};results.push(row);console.log(JSON.stringify(row));
  }
}
writeFileSync(opt('out','/tmp/cloth-affine-census.json'),JSON.stringify({contract:'CPU numerical diagnostic; affine cloth-only, no solid or projected path certification',results},null,2),{flag:'wx'});
