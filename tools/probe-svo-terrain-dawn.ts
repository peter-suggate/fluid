import assert from "node:assert/strict";
import { createDawnRenderDevice } from "./svo-dry-frame-harness";
import { buildSvoRenderTerrainGpu } from "../lib/svo/features/scene-publication/webgpu-svo-render-terrain";
import { buildSvoRenderTerrainFieldSteps } from "../lib/svo/features/scene-publication/svo-render-solid-field";
import { completeCooperativeBuild } from "../lib/core/cooperative-build";
import { getScenePreset } from "../lib/core/scenes";

const {device,validationErrors}=await createDawnRenderDevice();
try {
  const scene=getScenePreset("hero-garden-hose-x10").create();
  for(const crest of ["flat","bullnose","wall"] as const){
    const variant=structuredClone(scene);variant.terrain!.procedural={...variant.terrain!.procedural!,spec:{...variant.terrain!.procedural!.spec,crest}};
    const cell=[0.005,0.005,0.005] as const;
    const cpu=completeCooperativeBuild(buildSvoRenderTerrainFieldSteps(variant,cell,1));
    const gpu=await buildSvoRenderTerrainGpu(device,variant,cell,1);
    assert.ok(cpu&&gpu);assert.deepEqual(gpu.dimensions,cpu.dimensions);
    let maxError_m=0,sum=0;
    for(let i=0;i<cpu.heights_m.length;i++){
      assert.ok(Number.isFinite(gpu.heights_m[i]));const error=Math.abs(cpu.heights_m[i]-gpu.heights_m[i]);maxError_m=Math.max(maxError_m,error);sum+=error;
    }
    assert.ok(maxError_m<1e-5,`Terrain height error ${maxError_m} m exceeds 10 micrometres`);
    console.log(JSON.stringify({crest,samples:cpu.heights_m.length,maxError_m,meanError_m:sum/cpu.heights_m.length}));
  }
  assert.deepEqual(validationErrors,[]);
}finally{device.destroy();}
