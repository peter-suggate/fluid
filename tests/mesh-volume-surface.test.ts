import test from 'node:test';
import assert from 'node:assert/strict';
import { meshVolumeSurface } from '../lib/core/imported-assets/mesh-volume-surface';
import type { MeshVolume, MeshVolumeBrick } from '../lib/core/imported-assets/mesh-volume';

test('native surface extraction shares brick seams and retains subcell geometry', () => {
  const bricks = new Map<string, MeshVolumeBrick>();
  for (let bx=0;bx<2;bx++) {
    const distance = new Float32Array(729);
    for(let z=0;z<9;z++) for(let y=0;y<9;y++) for(let x=0;x<9;x++) distance[x+9*(y+9*z)]=(y-3.25)*0.00625;
    const b: MeshVolumeBrick={origin:[bx*8,0,0],distance,geometricNormal:new Uint32Array(729),shadingNormal:new Uint32Array(729),material:new Uint32Array(729).fill(1)};
    bricks.set(b.origin.join(),b);
  }
  const volume: MeshVolume={header:{version:1,units:'metres',upAxis:'Y',asset:'plane',origin:[0,0,0],dimensions:[16,8,8],cellSize:0.00625,bandWidth:1,brickCount:2,brickCells:8,nodeStrideBytes:16,materials:[]},bricks};
  const mesh=meshVolumeSurface(volume);
  assert.equal(mesh.positions.length/3,4, "coplanar cells merge across brick seams");
  assert.equal(mesh.indices.length,6);
  for(let i=1;i<mesh.positions.length;i+=3) assert.ok(Math.abs(mesh.positions[i]!+0.75*0.00625)<1e-8);
  assert.ok([...mesh.indices].every(i=>i<mesh.positions.length/3));
  assert.equal(meshVolumeSurface(volume),mesh,'immutable assets reuse the extracted buffers');
  for(let i=0;i<mesh.indices.length;i+=3) assert.equal(new Set(mesh.indices.slice(i,i+3)).size,3);
});

test('curved native meshes keep their full resolution stream and outward winding', async () => {
  const {prepareMeshVolumeSurfaces}=await import('../lib/core/imported-assets/mesh-volume-surface');
  await prepareMeshVolumeSurfaces();
  const distance=new Float32Array(729);
  for(let z=0;z<9;z++)for(let y=0;y<9;y++)for(let x=0;x<9;x++)distance[x+9*(y+9*z)]=(Math.hypot(x-4,y-4,z-4)-2.7)*.00625;
  const brick:MeshVolumeBrick={origin:[0,0,0],distance,geometricNormal:new Uint32Array(729),shadingNormal:new Uint32Array(729),material:new Uint32Array(729).fill(1)};
  const volume:MeshVolume={header:{version:1,units:'metres',upAxis:'Y',asset:'sphere',origin:[0,0,0],dimensions:[8,8,8],cellSize:.00625,bandWidth:1,brickCount:1,brickCells:8,nodeStrideBytes:16,materials:[]},bricks:new Map([['0,0,0',brick]])};
  const mesh=meshVolumeSurface(volume), native=mesh.levels[0]!;
  assert.ok(mesh.levels.length<=7,"LOD header and levels fit the GPU instance record");
  assert.equal(native.first,0);assert.equal(native.error,0);assert.ok(native.count>100);
  for(let i=0;i<native.count;i+=3){
    const p=[0,1,2].map(j=>Array.from(mesh.positions.slice(mesh.indices[i+j]!*3,mesh.indices[i+j]!*3+3)));
    const a=p[1]!.map((v,j)=>v-p[0]![j]!),b=p[2]!.map((v,j)=>v-p[0]![j]!);
    const n=[a[1]!*b[2]!-a[2]!*b[1]!,a[2]!*b[0]!-a[0]!*b[2]!,a[0]!*b[1]!-a[1]!*b[0]!];
    assert.ok(n.reduce((s,v,j)=>s+v*p[0]![j]!,0)>0,'all native faces point out of the solid');
  }
  mesh.levels.forEach((level,i)=>{assert.equal(level.count%3,0);assert.ok(level.first+level.count<=mesh.indices.length);if(i)assert.ok(level.count<mesh.levels[i-1]!.count);});
});
