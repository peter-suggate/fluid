import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {ClosedSourceMesh,loadSourceMesh,sourceMeshDistance,sourceMeshBounds,SPOT_SOURCE_ASSET} from '../lib/cloth/source-mesh';
import type {WetClothSourceField} from '../lib/cloth/source-types';

const tetra='v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\nf 1 3 2\nf 1 2 4\nf 1 4 3\nf 2 3 4\n';
test('closed source mesh distances resolve faces, edges and interior without an AABB proxy',()=>{
 const m=new ClosedSourceMesh(tetra);
 assert.ok(Math.abs(m.signedDistance([.1,.1,.1])+.1)<1e-12);
 assert.ok(Math.abs(m.signedDistance([1,1,1])-2/Math.sqrt(3))<1e-12);
 assert.equal(m.signedDistance([0,0,0]),0);
 assert.equal(m.signedDistance([-.3,0,0]),.3);
 assert.throws(()=>new ClosedSourceMesh(tetra.replace('f 2 3 4\n','')),/closed/);
 assert.throws(()=>new ClosedSourceMesh(tetra.replace('f 2 3 4','f 4 3 2')),/oriented/);
});

test('canonical Spot loads by checksum and preserves source scale, mean pivot and sampled SDF',async()=>{
 const m=await loadSourceMesh(SPOT_SOURCE_ASSET.filename,async url=>new Response(await readFile(`public${url}`)));
 assert.equal(m.vertices.length,2930);assert.equal(m.triangles.length,5856);
 const f:WetClothSourceField={index:0,kind:'file',usage:'solid',filename:SPOT_SOURCE_ASSET.filename,root:true,sampled:false,inside:false,
  group:0,scale:2,sampleSpacing_m:.00144,center_m:[.1,.2,.3],axis:[0,1,0],angle_rad:0,sourceAttributes:{}};
 const p:[number,number,number]=[.1,.2,.3],before=sourceMeshDistance(f,p),mean=m.centre.map(v=>.02*v);
 const q:[number,number,number]=[.1+mean[0]!-mean[2]!,.2,.3+mean[2]!+mean[0]!];
 assert.ok(Math.abs(sourceMeshDistance({...f,angle_rad:Math.PI/2},q)-before)<1e-12);
 assert.ok(Math.abs(sourceMeshDistance({...f,inside:true},p)+before)<1e-12);
 const bounds=sourceMeshBounds(f);
 for(let a=0;a<3;a++){
  assert.ok(Math.abs(bounds.min[a]!-f.center_m![a]!-m.bounds.min[a]!*.02)<1e-12);
  assert.ok(Math.abs(bounds.max[a]!-f.center_m![a]!-m.bounds.max[a]!*.02)<1e-12);
 }
});
