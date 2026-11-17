/** Offline opportunity census, not a predictor: all eight recorded Newton
 * directions are known here. Measures how many genuinely NEW pair keys each
 * direction introduces and how many vertex boxes require local invalidation. */
import {readFileSync,writeFileSync} from 'node:fs';
import {join} from 'node:path';
import {createHash} from 'node:crypto';
import assert from 'node:assert/strict';
import {affineAxisBounds,clothSweptCandidates} from './cloth-swept-candidates';
import type {AffineVertex,Vec3} from './cloth-affine-ccd';
const input=process.argv[2]??'/tmp/cloth-directions-forward',output=process.argv[3]??'/tmp/cloth-coherence.json',results=[];
for(const tick of [231,340]){
 const manifest=JSON.parse(readFileSync(join(input,`tick${tick}.json`),'utf8'));
 const read=(name:string)=>{const f=manifest.files.find((f:{name:string})=>f.name===name);assert.ok(f);const b=readFileSync(join(input,f.filename));assert.equal(createHash('sha256').update(b).digest('hex'),f.sha256);return Uint8Array.from(b).buffer;};
 const t=new Uint32Array(read('triangles')),e=new Uint32Array(read('edges')),triangles=Array.from({length:manifest.triangles},(_,i)=>Array.from(t.subarray(4*i,4*i+3))).flat(),edges=Array.from({length:manifest.edges},(_,i)=>Array.from(e.subarray(4*i,4*i+2))).flat();
 const union=new Set<string>(),caches=[0,.0001,.00025,.0005,.001].map(skin=>({skin,boxes:undefined as Float64Array|undefined,pairs:new Set<string>()}));let previous=new Set<string>();
 for(let slot=0;slot<manifest.iterations;slot++){
  if(new Float32Array(read(`${slot}-control`))[0]!==0)continue;
  const x=new Float32Array(read(`${slot}-x`)),d=new Float32Array(read(`${slot}-d`)),vertices:AffineVertex[]=Array.from({length:manifest.vertices},(_,i)=>({x:Array.from(x.subarray(4*i,4*i+3)) as unknown as Vec3,d:Array.from(d.subarray(4*i,4*i+3)) as unknown as Vec3}));
  const current=new Set<string>();let newPairs=0,retained=0;
  for(const c of clothSweptCandidates(vertices,triangles,edges,.000002)){const key=c.kind+':'+c.ids.join(',');current.add(key);if(!union.has(key))newPairs++;if(previous.has(key))retained++;union.add(key);}
  const invalidation=caches.map(cache=>{
   const old=cache.boxes,boxes=old??new Float64Array(6*vertices.length),changed=new Uint8Array(vertices.length);let escaped=0;
   for(let v=0;v<vertices.length;v++){
    const p=vertices[v]!,bounds=[0,1,2].map(k=>affineAxisBounds(p.x[k]!,p.d[k]!,.000002));
    const outside=!old||bounds.some(([lo,hi],k)=>lo<boxes[6*v+2*k]!||hi>boxes[6*v+2*k+1]!);
    if(outside){escaped++;changed[v]=1;for(let k=0;k<3;k++){boxes[6*v+2*k]=bounds[k]![0]-cache.skin;boxes[6*v+2*k+1]=bounds[k]![1]+cache.skin;}}
   }
   cache.boxes=boxes;let faces=0,edgeCount=0;for(let i=0;i<triangles.length;i+=3)if(changed[triangles[i]!]||changed[triangles[i+1]!]||changed[triangles[i+2]!])faces++;for(let i=0;i<edges.length;i+=2)if(changed[edges[i]!]||changed[edges[i+1]!])edgeCount++;
   let paddedPairs:number|undefined,newPaddedPairs:number|undefined,paddedUnion:number|undefined;
   if(process.argv.includes('--padded')&&cache.skin===.0005){
    const boxVertices:AffineVertex[]=Array.from({length:vertices.length},(_,v)=>({x:[boxes[6*v]!,boxes[6*v+2]!,boxes[6*v+4]!],d:[boxes[6*v+1]!-boxes[6*v]!,boxes[6*v+3]!-boxes[6*v+2]!,boxes[6*v+5]!-boxes[6*v+4]!]}));
    paddedPairs=0;newPaddedPairs=0;
    for(const c of clothSweptCandidates(boxVertices,triangles,edges,0)){const key=c.kind+':'+c.ids.join(',');paddedPairs++;if(!cache.pairs.has(key))newPaddedPairs++;cache.pairs.add(key);}
    for(const key of current)assert.ok(cache.pairs.has(key),'Padded union omitted an actual swept pair');paddedUnion=cache.pairs.size;
   }
   return {skin:cache.skin,escapedVertices:escaped,changedFaces:faces,changedEdges:edgeCount,paddedPairs,newPaddedPairs,paddedUnion};
  });
  const row={tick,slot,pairs:current.size,newPairs,retainedFromPrevious:retained,union:union.size,invalidation};results.push(row);console.log(JSON.stringify(row));previous=current;
 }
}
writeFileSync(output,JSON.stringify({contract:'Offline recorded-direction reuse opportunity only; future directions are not known to a live solver',results},null,2),{flag:'wx'});
