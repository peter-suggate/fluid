import assert from "node:assert/strict";
import test from "node:test";
import { planSparseSceneDomain } from "../lib/core/sparse-scene-domain";
import { getScenePreset } from "../lib/core/scenes";

test("bounds-only planning preserves padded integer claims without host brick enumeration", () => {
  const scene = getScenePreset("water-box-dam-break").create();
  const proxies = [
    {min:{x:-4,y:-.1,z:-2},max:{x:2,y:.1,z:3}},
    {min:{x:0,y:0,z:0},max:{x:0,y:0,z:0}},
  ];
  for (const padding of [0,1]) {
    const options = {solverClaim:"none" as const,conservativePaddingCells:padding};
    const cpu = planSparseSceneDomain(scene,[24,16,16],8,proxies,options);
    const gpu = planSparseSceneDomain(scene,[24,16,16],8,proxies,{...options,enumerateProxyBricks:false});
    assert.deepEqual(gpu, {...cpu,proxyBrickCoordinates:[],environmentBrickCoordinates:[],coordinates:[]});
    gpu.proxyBrickRanges.forEach((range,i)=>{
      const expected=[];
      for(let z=range.min[2];z<range.maxExclusive[2];z++)
        for(let y=range.min[1];y<range.maxExclusive[1];y++)
          for(let x=range.min[0];x<range.maxExclusive[0];x++)expected.push(`${x},${y},${z}`);
      assert.deepEqual(new Set(expected),new Set(cpu.proxyBrickCoordinates[i].map(p=>`${p.x},${p.y},${p.z}`)));
    });
  }
});
