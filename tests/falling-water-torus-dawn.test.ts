import assert from "node:assert/strict";
import test from "node:test";
import {getSceneDefinition} from "../lib/core/scenes";
import {sceneDocument} from "../lib/core/scene-definition";
import {withUniformDevice,createUniformSolver,advanceUniform,readUniformFields} from "./helpers/uniform-geometric";
(process.env.WEBGPU_NODE_MODULE?test:test.skip)("Uniform torus begins hollow and falls onto the floor",{timeout:120_000},()=>withUniformDevice("Uniform falling torus",async device=>{
 const scene=sceneDocument(getSceneDefinition("falling-water-torus"));scene.numerics.fixedDt_s=scene.numerics.maxDt_s=1/60;
 const solver=await createUniformSolver(device,scene);
 try{
    const [nx, ny, nz] = [solver.info.nx, solver.info.ny, solver.info.nz];
    const summarize = (density: Float32Array) => {
      let mass = 0, momentY = 0, floorMass = 0;
      for (let i = 0; i < density.length; i++) {
        assert.ok(Number.isFinite(density[i]));
        const rho = Math.max(0, density[i]!); const y = Math.floor(i / nx) % ny;
        mass += rho; momentY += rho * (y + .5) * .05;
        if (y < 4) floorMass += rho;
      }
      return { mass, centreY: momentY / mass, floorMass };
    };
    const initial = await readUniformFields(device,solver);
    const before = summarize(initial.density);
    // Expanded owner averages measure liquid across the complete domain.
    const beforeWorldMass = before.mass;
    assert.ok(before.mass > 0);
    for (let y = 0; y < ny; y++) {
      assert.equal(initial.density[Math.floor(nx/2) + nx * (y + ny * Math.floor(nz/2))], 0,
        "the torus hole must begin empty");
    }
    for (let step = 1; step <= 48; step++) {
      await advanceUniform(solver,step/60);
      if (step % 12 === 0) console.log(`torus step ${step}`);
    }
    const after = summarize((await readUniformFields(device,solver)).density);
    const afterWorldMass = after.mass;
    console.log(JSON.stringify({ before, after, beforeWorldMass, afterWorldMass }));
    assert.ok(after.centreY < before.centreY - .5, "the torus must fall");
    assert.ok(after.floorMass > .1 * before.mass, "liquid must reach the floor");
    // Expanded owner averages count the whole Uniform domain.
    assert.ok(afterWorldMass > .9 * beforeWorldMass, "impact must retain the liquid");

 }finally{solver.destroy();}
}));
