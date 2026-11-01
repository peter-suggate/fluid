import assert from "node:assert/strict";
import test from "node:test";
import { coarseVolumeScene, type CoarseVolumeCase } from "./helpers/uniform-coarse-surface-volume";
import { advanceUniform, createUniformSolver, readUniformFields, withUniformDevice } from "./helpers/uniform-geometric";
import { createNbflipVolumeProbe } from "../tools/probe-nbflip-volume";

// O6: no compensation. Compare the same phi-volume quadrature at construction
// and after transport: readStats initially reports the pressure fill instead.
for (const [kind, steps, tolerance] of [["slab",30,0.03],["resolved-slab",30,0.03],["drop",12,0.03],["film",60,0.001]] as const) {
  test(`4h ${kind} preserves its represented surface volume`, {
    skip: !process.env.WEBGPU_NODE_MODULE,
    todo: "O6: coarse surface transport investigation; see docs/plans/coarse-volume-loss-handoff-2026-10-30.md",
  }, async () => {
    await withUniformDevice(`coarse ${kind} volume`, async device => {
      const solver = await createUniformSolver(device, coarseVolumeScene(kind as CoarseVolumeCase), { volumeControlSeconds: 0 });
      const probe = await createNbflipVolumeProbe(device, solver, { gauge: true });
      try {
        const initial = (await probe.read()).reduce((a,b)=>a+b,0);
        // Request the gauge on each advance, and check that it agrees with
        // the independent spatial measurement at the final frame.
        await solver.readStats();
        for (let step=1; step<=steps; step++) { await advanceUniform(solver,step/60); await solver.readStats(); }
        const final = (await probe.read()).reduce((a,b)=>a+b,0);
        assert.equal(solver.info.uniformMixedFineTiles ?? 0,0);
        assert.ok(Math.abs(final-solver.info.volumeCellSum!) < 1e-4*initial, "spatial and scalar gauges agree");
        assert.ok(Math.abs(final/initial-1)<=tolerance, `${initial} to ${final} cells (${100*(final/initial-1)}%)`);
      } finally { probe.destroy(); solver.destroy(); }
    });
  });
}

test("a one-coarse-cell-radius sphere loads the analytic corners at construction", { skip: !process.env.WEBGPU_NODE_MODULE }, async () => {
  await withUniformDevice("coarse sphere construction", async device => {
    const solver = await createUniformSolver(device,coarseVolumeScene("small-drop"),{volumeControlSeconds:0});
    const probe = await createNbflipVolumeProbe(device,solver,{gauge:true});
    try {
      const fields = await readUniformFields(device,solver);
      assert.deepEqual([solver.info.nx,solver.info.ny,solver.info.nz],[64,64,64]);
      let checked=0;
      for(let z=0;z<=64;z+=4)for(let y=0;y<=64;y+=4)for(let x=0;x<=64;x+=4){
        const exact=Math.hypot(x*0.05-1.6,y*0.05-2.2,z*0.05-1.6)-0.2;
        if(Math.abs(exact)>0.21)continue;
        assert.ok(Math.abs(fields.vertex(x,y,z)-exact)<1e-6,`corner ${x},${y},${z}: ${fields.vertex(x,y,z)} vs ${exact}`);checked++;
      }
      assert.ok(checked>=25);
      const measured=(await probe.read()).reduce((a,b)=>a+b,0),sphere=4*Math.PI*0.2**3/(3*0.05**3);
      // This deliberately does not require the lattice's piecewise-linear
      // zero set to enclose the continuous sphere's analytic volume.
      assert.ok(measured>0.45*sphere&&measured<0.6*sphere,`${measured} represented vs ${sphere} analytic cells`);
    } finally {probe.destroy();solver.destroy();}
  });
});
