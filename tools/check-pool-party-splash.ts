/** Full authored pool scene through ball impact, using the UI's Box3D/fluid coupling.
 * WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js node --import tsx tools/check-pool-party-splash.ts
 */
import assert from "node:assert/strict";
import { createBackyardPoolPartyScene } from "../lib/core/backyard-pool-party-scene";
import { interactionScene } from "../lib/interaction/adapters/scene";
import { SceneInteractionHost } from "../lib/interaction/runtime/host";
import { createUniformSolver, withUniformDevice } from "../tests/helpers/uniform-geometric";
import { paceDawnEventPolling } from "../lib/harness/node-dawn-provider";
import { gpuCompilationManagerFor } from "../lib/core/gpu-compilation-manager";

paceDawnEventPolling();
await withUniformDevice("pool party splash", async device => {
  const scene = createBackyardPoolPartyScene();
  console.log("Preparing authored pool solver");
  const manager = gpuCompilationManagerFor(device);
  const progress = setInterval(() => {
    const snapshot = manager.snapshot();
    console.log(JSON.stringify({ phase: "compile", active: snapshot.active, queued: snapshot.queued, current: snapshot.progress?.label }));
  }, 30_000);
  const solver = await createUniformSolver(device, interactionScene(scene)).finally(() => clearInterval(progress));
  const host = new SceneInteractionHost(scene, { fluid: { step: request => solver.advanceExternalRigidStep(request) } });
  let submerged = false, minY = Infinity;
  try {
    await host.ready;
    console.log("Pool solver ready");
    for (let frame = 1; frame <= 960; frame++) {
      await host.advance(1 / 60);
      const ball = host.currentBodies().find(body => body.description.id === "pool-party/ball")!;
      const fraction = host.getView().snapshot!.bodies.find(body => body.bodyId === ball.description.id)!.submergedFraction ?? 0;
      minY = Math.min(minY, ball.position_m.y);
      submerged ||= fraction > .1;
      if (frame % 60 === 0) console.log(JSON.stringify({ frame, time_s: frame / 60, position: ball.position_m, submerged: fraction }));
    }
    assert.equal(host.getView().snapshot!.tick, 960);
    assert.equal(host.getView().readiness.state, "ready");
    assert.ok(submerged, "the ball must enter the water");
    assert.ok(minY < 1.8, "the ball must pass through the waterline");
    console.log(JSON.stringify({ result: "passed", minY, submerged, completedTime_s: solver.info.completedTime_s }));
  } finally { await host.dispose(); solver.destroy(); }
}, ["subgroups"]);
