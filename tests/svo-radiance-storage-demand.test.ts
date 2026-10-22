import assert from "node:assert/strict";
import test from "node:test";
import { WebGpuLiveSvoTetrahedralRadiance } from "../lib/svo/features/radiance/webgpu-svo-tetrahedral-radiance";
import { WebGpuLiveSvoDerivedBuilder } from "../lib/svo/features/radiance/webgpu-svo-live-derived-builder";
import { planSvoNodeMipAddresses } from "../lib/svo/features/radiance/svo-node-mip-address-plan";

test("inactive radiance reserves only inert bindings and never publishes a readable generation", () => {
  const globals=globalThis as unknown as Record<string,unknown>,saved=globals.GPUTextureUsage;
  globals.GPUTextureUsage={TEXTURE_BINDING:1,STORAGE_BINDING:2,COPY_DST:4};
  const plan=planSvoNodeMipAddresses({occupiedBasePages:[[0,0,0]],basePageDimensions:[2,2,2],levelCount:2,addressCapacity:1024,generation:1});
  const textures:{size:readonly number[];destroyed:boolean}[]=[];
  const device={features:new Set(),limits:{maxTextureDimension2D:2048},createTexture(descriptor:{size:readonly number[]}){
    const texture={size:descriptor.size,destroyed:false,createView(){return {};},destroy(){this.destroyed=true;}};
    textures.push(texture);return texture;
  }} as unknown as GPUDevice;
  const options={pageCapacity:plan.pageCapacity,atlasTexels:plan.radianceAtlasTexels as [number,number,number],
    atlasPageCapacity:plan.radiancePageCapacity,slotOffset:plan.radianceSlotOffset,radianceFloorLevel:plan.radianceFloorLevel};
  try {
    const inactive=new WebGpuLiveSvoTetrahedralRadiance(device,{...options,storageEnabled:false});
    assert.deepEqual(textures.slice(0,4).map(t=>t.size),[[1,1,1],[1,1,1],[1,1,1],[1,1,1]]);
    assert.equal(inactive.gpuTarget().storageEnabled,false);
    assert.equal(inactive.pageState.validity.capacity,plan.pageCapacity,"opacity planner slot/validity space is retained");
    inactive.prepareGpuUpdate(plan.plan,plan.radianceSlotOffset);inactive.acceptGpuUpdate(1);
    assert.equal(inactive.visibleGeneration(),undefined,"inert bindings are never advertised as real radiance");
    const full=new WebGpuLiveSvoTetrahedralRadiance(device,options);
    full.prepareGpuUpdate(plan.plan,plan.radianceSlotOffset);full.acceptGpuUpdate(1);
    assert.ok(full.visibleGeneration());assert.equal(full.storageEnabled,true);
    assert.ok(full.allocatedBytes>inactive.allocatedBytes);
    inactive.destroy();full.destroy();assert.ok(textures.every(t=>t.destroyed));
  } finally {globals.GPUTextureUsage=saved;}
});

test("radiance dispatch cannot accidentally target inactive storage", () => {
  const builder=Object.assign(Object.create(WebGpuLiveSvoDerivedBuilder.prototype),{options:{radiance:{storageEnabled:false}},destroyed:false});
  assert.throws(()=>builder.encodeRound({},true),/requires allocated radiance storage/);
});
