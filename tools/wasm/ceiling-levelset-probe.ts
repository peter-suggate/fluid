/** Verify pool-impact boundary release with every published WASM variant. */
import assert from "node:assert/strict";
import { writeFile } from "node:fs/promises";
import { sceneDocument } from "../../lib/core/scene-definition";
import { getSceneDefinition } from "../../lib/core/scenes";
import { decodePhysicsPublication, PhysicsPlane } from "../../lib/physics-wasm/publication";
import { parsePhysicsReceipt } from "../../lib/physics-wasm/protocol";
import { loadFluidWasmForNode } from "./load-module.mjs";

const results: Record<string, Record<string, unknown[]>> = {};
for (const artifact of ["scalar", "simd", "threaded"] as const) {
  const wasm = await loadFluidWasmForNode(undefined, { artifact });
  const lanes: Record<string, unknown[]> = {};
  for (const top of ["closed", "open"] as const) {
    const scene = sceneDocument(getSceneDefinition("coarse-first-pool-impact-half"));
    scene.container.top = top;
    if (top === "open") scene.solidVoxels = scene.solidVoxels.filter(p => p.minimum[1] !== 48);
    const world = wasm.FluidWorld.from_scene(JSON.stringify(scene), JSON.stringify({
      runEpoch: 1, commandSequence: 0, pressureIterations: 256,
      pressureRelativeTolerance: 1e-6, transportExperiment: "level-set-volume",
      production: { dtS: 1/30, timeStep: "paper" },
    }));
    const frames: unknown[] = [];
    let mass = 0, contacted = false;
    try {
      for (let frame=0; frame<=61; frame++) {
        const receipt = parsePhysicsReceipt(frame ? world.advance(frame+1, 1/30) : world.receipt());
        assert.equal(receipt.fault, null);
        const publication = decodePhysicsPublication({ id: frame, revision: receipt,
          bytes: world.snapshot(2).slice(), release() {} });
        try {
          const phi = publication.plane(PhysicsPlane.RdfVertices) as Float32Array;
          const segments = publication.plane(PhysicsPlane.RdfSegments) as Float32Array;
          const ceiling = phi.slice(48*65);
          const wet = ceiling.filter(v => v<=0).length;
          contacted ||= wet>0;
          const volume = Number(receipt.liquidMeasure);
          if (!frame) mass = volume;
          assert.ok(Math.abs(volume-mass)<1e-5*mass);
          assert.ok(phi.every(Number.isFinite));
          let maximumY = -Infinity;
          for (let i=1; i<segments.length; i+=2) maximumY=Math.max(maximumY,segments[i]!);
          if (frame===61) {
            assert.ok(contacted, `${artifact}/${top}: no contact tested`);
            assert.equal(wet,0,`${artifact}/${top}: stuck top`);
          }
          frames.push({frame, wetTopVertices:wet, minimumTopPhi:Math.min(...ceiling),
            maximumSurfaceY:maximumY, volume});
        } finally { publication.release(); }
      }
    } finally { world.free(); }
    lanes[top]=frames;
  }
  results[artifact]=lanes;
  if (artifact!=="scalar") assert.deepEqual(lanes,results.scalar);
}
await writeFile(process.argv[2] ?? "/tmp/ceiling-wasm.json",JSON.stringify(results,null,2)+"\n");
process.exit(0);
