import assert from "node:assert/strict";
import test from "node:test";
import { readFileSync } from "node:fs";
import { gunzipSync } from "node:zlib";
import { decodeMeshVolume, sampleMeshVolume } from "./mesh-volume";
import { registerMeshAsset, packMeshAsset, meshAssetHeader, meshAssetVolume, sampledVolumeExtent, sampledVolumeHorizontalSlab, sampledVolumeSupport } from "./asset-library";
import { packSvoFieldProgramArena, SVO_FIELD_PROGRAM_BLOCK_WORDS, type SvoFieldProgram } from "../../svo/features/scene-publication/svo-field-program";
import { intersectSvoPrimitive, type SvoFieldProgramPrimitive } from "../../svo/contracts/svo-primitive-abi";
import { sampledVolumeMaterialRecords } from "../../svo/features/materials/svo-sampled-materials";
const asset="teak-lounger";
registerMeshAsset(asset,decodeMeshVolume(gunzipSync(readFileSync(new URL("../../../public/assets/pool-party/teak-lounger.fvx.gz",import.meta.url)))));
const program:SvoFieldProgram={ops:[],result:0,sampledVolume:{asset,materialBase:100}};
test("instances share one sampled image and retain the complete PBR palette",()=>{
  const image=packMeshAsset(asset),arena=packSvoFieldProgramArena([program,program]);
  assert.equal(arena.length,2*SVO_FIELD_PROGRAM_BLOCK_WORDS+image.length);
  assert.equal(arena[1],arena[SVO_FIELD_PROGRAM_BLOCK_WORDS+1]);
  const materials=sampledVolumeMaterialRecords([program,program],1),header=meshAssetHeader(asset);
  assert.equal(materials.length,header.materials.length);
  header.materials.forEach((m,i)=>{assert.equal(materials[i]!.roughness,m.roughness);assert.equal(materials[i]!.metallic,m.metallic);});
});
test("narrow-band chair rays reach the cushion after traversing the empty asset bounds",()=>{
  const header=meshAssetHeader(asset),e=sampledVolumeExtent(program.sampledVolume!);
  const center=header.origin.map((v,i)=>v+e[i]!);
  const descriptor:SvoFieldProgramPrimitive={kind:"field-program",primitiveId:1,ownerId:1,materialId:100,
    center_m:{x:center[0]!,y:center[1]!,z:center[2]!},envelopeRadii_m:{x:e[0],y:e[1],z:e[2]},fieldProgramReference:0,program};
  for(const x of [-.25,0,.25]){
    const hit=intersectSvoPrimitive(descriptor,{origin_m:{x,y:2,z:-.3},direction:{x:0,y:-1,z:0},tMin_m:0,tMax_m:4});
    assert.ok(hit,`chair cushion at ${x}`);assert.ok(hit.position_m.y>.4&&hit.position_m.y<.55);
  }
});

test("sampled canopy casts a shadow below while upward light rays above it stay clear", () => {
  const id = "garden-parasol";
  registerMeshAsset(id, decodeMeshVolume(gunzipSync(readFileSync(new URL("../../../public/assets/pool-party/garden-parasol.fvx.gz", import.meta.url)))));
  const p: SvoFieldProgram = { ops: [], result: 0, sampledVolume: { asset: id, materialBase: 2000 } };
  const h = meshAssetHeader(id), e = sampledVolumeExtent(p.sampledVolume!);
  const c = h.origin.map((v, i) => v + e[i]!);
  const canopy: SvoFieldProgramPrimitive = { kind: "field-program", primitiveId: 2, ownerId: 2, materialId: 2000,
    center_m: { x: c[0]!, y: c[1]!, z: c[2]! }, envelopeRadii_m: { x: e[0], y: e[1], z: e[2] },
    fieldProgramReference: 0, program: p };
  for (const x of [-1, -.5, .5, 1]) {
    const below = intersectSvoPrimitive(canopy, { origin_m: { x, y: 2, z: .2 }, direction: { x: 0, y: 1, z: 0 }, tMin_m: 0, tMax_m: 3 });
    assert.ok(below && below.position_m.y > 2.3 && below.position_m.y < 2.9);
    const above = intersectSvoPrimitive(canopy, { origin_m: { x, y: 3, z: .2 }, direction: { x: 0, y: 1, z: 0 }, tMin_m: 0, tMax_m: 3 });
    assert.equal(above, null);
  }
});


test("stone normal and roughness maps survive baking independently of the flat distance field", () => {
  const id = "honed-travertine-paver";
  const volume = decodeMeshVolume(gunzipSync(readFileSync(new URL("../../../public/assets/pool-party/honed-travertine-paver.fvx.gz", import.meta.url))));
  registerMeshAsset(id, volume);
  const roughness = new Set<number>();
  let perturbed = 0;
  for (let x = -.43; x < .44; x += .019) for (let z = -.43; z < .44; z += .023) {
    const s = sampleMeshVolume(volume, [x, .031, z]);
    assert.ok(s.distance > 0 && s.distance < .002, "normal maps must not displace the stone plane");
    assert.ok(s.geometricNormal[1] > .999, "the geometric normal stays vertical");
    assert.ok(s.shadingNormal[1] > .7, "pores retain the outward hemisphere");
    if (Math.hypot(s.shadingNormal[0], s.shadingNormal[2]) > .025) perturbed++;
    roughness.add(volume.header.materials[s.material - 1]!.roughness);
  }
  assert.ok(perturbed > 100, "the baked stone has spatial normal detail");
  assert.ok(roughness.size > 1, "the roughness map survives the PBR palette");
  const program: SvoFieldProgram = { ops: [], result: 0, sampledVolume: { asset: id, materialBase: 400 } };
  const instances = Array.from({ length: 160 }, () => program);
  const arena = packSvoFieldProgramArena(instances);
  assert.equal(arena.length, instances.length * SVO_FIELD_PROGRAM_BLOCK_WORDS + packMeshAsset(id).length,
    "160 terrace slabs must carry only one stone image");
});

test("flat slab certificates follow source centring and reject oblique rotations", () => {
  const field = { asset: "honed-travertine-paver" }, h = meshAssetHeader(field.asset);
  assert.ok(h.horizontalSlab);
  const c = h.origin.map((v,i)=>v+h.dimensions[i]!*h.cellSize/2);
  for (const a of [0,Math.PI/2,Math.PI,3*Math.PI/2]) {
    const center = { x: 2+Math.cos(a)*c[0]!+Math.sin(a)*c[2]!, y: 2+c[1]!, z: 3-Math.sin(a)*c[0]!+Math.cos(a)*c[2]! };
    const slab = sampledVolumeHorizontalSlab(field,center,{w:Math.cos(a/2),x:0,y:Math.sin(a/2),z:0})!;
    for (let i=0;i<3;i++) {
      assert.ok(Math.abs(slab.minimum[i]!-[1.55,2,2.55][i]!)<1e-7);
      assert.ok(Math.abs(slab.maximum[i]!-[2.45,2.03,3.45][i]!)<1e-7);
    }
  }
  assert.equal(sampledVolumeHorizontalSlab(field,{x:0,y:0,z:0},{w:Math.cos(.1),x:0,y:Math.sin(.1),z:0}),undefined);
});

test("sparse support rejects empty interiors without dropping any occupied brick under rotation", () => {
  const volume = meshAssetVolume(asset), h=volume.header;
  const c=h.origin.map((v,i)=>v+h.dimensions[i]!*h.cellSize/2);
  for(const a of [0,Math.PI/2,.73]) {
    const transform=(p:readonly number[])=>[2+Math.cos(a)*p[0]!+Math.sin(a)*p[2]!,3+p[1]!,4-Math.sin(a)*p[0]!+Math.cos(a)*p[2]!];
    const center=transform(c);
    const support=sampledVolumeSupport({asset},{x:center[0]!,y:center[1]!,z:center[2]!},{w:Math.cos(a/2),x:0,y:Math.sin(a/2),z:0})!;
    const empty=transform([0,.18,0]);
    assert.equal(support(empty,empty),false,"air below the cushion stays coarse");
    for(const b of volume.bricks.values()) {
      const p=transform(b.origin.map((v,i)=>h.origin[i]!+(v+4)*h.cellSize));
      assert.equal(support(p,p),true,`occupied brick ${b.origin} at rotation ${a}`);
    }
    assert.equal(support([-30,-30,-30],[-29,-29,-29]),false);
  }
});
